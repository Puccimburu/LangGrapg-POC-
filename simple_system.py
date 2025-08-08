#!/usr/bin/env python3
"""
Simplified LangGraph RAG System with Follow-up Questions
Essential features only - no unnecessary classes or functions
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Environment and logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core imports
import pymongo
from bson import ObjectId
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, field_validator

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/genaiexeco-development")

if not GEMINI_API_KEY:
    raise ValueError("❌ Please set GEMINI_API_KEY in .env file")

# MongoDB connection
try:
    client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client["genaiexeco-development"]
    client.admin.command('ping')
    logger.info("✅ MongoDB connected")
except Exception as e:
    logger.error(f"❌ MongoDB failed: {e}")
    db = None

# LLM setup
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, google_api_key=GEMINI_API_KEY)

# Simple models
class FollowUp(BaseModel):
    question: str = Field(description="Follow-up question or 'NONE'")
    query: Optional[Dict] = Field(None, description="MongoDB query if question provided")
    
    @field_validator('query', mode='before')
    @classmethod
    def parse_query(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON in query")
        return value

# Global state
conversation_state = {
    "messages": [],
    "follow_up": None,
    "follow_up_query": None
}

def execute_query(query_data: Dict) -> Dict:
    """Execute MongoDB query"""
    if not db:
        return {"success": False, "error": "No database connection", "results": []}
    
    try:
        collection = db[query_data.get("collection", "")]
        pipeline = query_data.get("pipeline", [])
        results = list(collection.aggregate(pipeline))
        
        # Convert ObjectIds to strings
        def convert_objectid(obj):
            if isinstance(obj, ObjectId):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_objectid(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_objectid(item) for item in obj]
            return obj
        
        results = [convert_objectid(doc) for doc in results]
        logger.info(f"✅ Query executed: {len(results)} records")
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        return {"success": False, "error": str(e), "results": []}

def generate_query(question: str) -> Dict:
    """Generate MongoDB query from question"""
    prompt = f"""Generate a MongoDB query for this question: "{question}"

Available collections:
- users: userId, firstName, lastName, emailId, role (admin/user/legal), authSource
- files: fileId, fileName, status, additionalFields
- prompts: promptId, promptName, promptType
- batches: batchId, batchName, status

For user questions, use: {{"collection": "users", "pipeline": [{{"$project": {{"firstName": 1, "lastName": 1, "emailId": 1, "role": 1, "userId": 1}}}}]}}
For admin users, add: {{"$match": {{"role": "admin"}}}}

Return ONLY valid JSON:"""

    try:
        response = llm.invoke(prompt)
        query_text = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(query_text)
    except Exception as e:
        logger.error(f"❌ Query generation failed: {e}")
        return {"collection": "users", "pipeline": [{"$limit": 10}]}

def generate_follow_up(data: List[Dict]) -> Optional[FollowUp]:
    """Generate follow-up question based on data"""
    if not data:
        return None
    
    prompt = f"""Based on this user data, suggest a follow-up question and MongoDB query.

Data sample: {json.dumps(data[:2], indent=2, default=str)}

Examples:
{{"question": "Would you like to filter by role?", "query": {{"collection": "users", "pipeline": [{{"$match": {{"role": "<user_input>"}}}}]}}}}
{{"question": "NONE", "query": null}}

Return JSON:"""

    try:
        structured_llm = llm.with_structured_output(FollowUp)
        follow_up = structured_llm.invoke(prompt)
        return follow_up if follow_up.question.upper() != "NONE" else None
    except Exception as e:
        logger.warning(f"⚠️ Follow-up generation failed: {e}")
        return None

def fuzzy_match_role(user_input: str) -> str:
    """Simple fuzzy matching for roles"""
    user_input = user_input.lower().strip()
    
    mappings = {
        "admins": "admin", "administrators": "admin", "admin": "admin",
        "users": "user", "regular": "user", "user": "user", 
        "legal": "legal", "lawyer": "legal", "attorney": "legal"
    }
    
    for key, value in mappings.items():
        if key in user_input or user_input in key:
            return value
    
    return user_input

def handle_follow_up(user_response: str) -> Dict:
    """Handle follow-up response"""
    if not conversation_state["follow_up_query"]:
        return {"success": False, "message": "No follow-up context"}
    
    # Check if user wants to proceed
    if not any(word in user_response.lower() for word in ["yes", "sure", "ok", "show"]):
        conversation_state["follow_up"] = None
        conversation_state["follow_up_query"] = None
        return {"success": True, "message": "Okay, what else can I help with?"}
    
    # Extract value and execute query
    query = conversation_state["follow_up_query"].copy()
    query_str = json.dumps(query)
    
    if "<user_input>" in query_str:
        # Simple extraction - get the meaningful word from response
        words = user_response.lower().split()
        extracted = next((w for w in words if w not in ["yes", "sure", "ok", "show", "me", "the", "a", "an"]), "")
        
        # Apply fuzzy matching if it's a role query
        if "role" in query_str:
            extracted = fuzzy_match_role(extracted)
        
        query_str = query_str.replace("<user_input>", extracted)
        query = json.loads(query_str)
    
    result = execute_query(query)
    conversation_state["follow_up"] = None
    conversation_state["follow_up_query"] = None
    
    return result

def format_table(data: List[Dict]) -> str:
    """Format data as a simple table"""
    if not data:
        return "No results found."
    
    # Get headers from first item
    headers = list(data[0].keys())
    
    # Create table
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for row in data[:10]:  # Limit to 10 rows
        values = [str(row.get(h, "")) for h in headers]
        table += "| " + " | ".join(values) + " |\n"
    
    return table

def process_question(question: str) -> str:
    """Main processing function"""
    logger.info(f"🚀 Processing: '{question}'")
    
    # Check if this is a follow-up response
    if conversation_state["follow_up"]:
        result = handle_follow_up(question)
        if result["success"] and "results" in result:
            return format_table(result["results"])
        else:
            return result.get("message", "No results found.")
    
    # Generate and execute query
    query = generate_query(question)
    result = execute_query(query)
    
    if not result["success"]:
        return f"Error: {result['error']}"
    
    # Format results
    output = format_table(result["results"])
    
    # Generate follow-up
    follow_up = generate_follow_up(result["results"])
    if follow_up:
        conversation_state["follow_up"] = follow_up.question
        conversation_state["follow_up_query"] = follow_up.query
        output += f"\n\n**{follow_up.question}**"
    
    return output

def main():
    """Interactive mode"""
    print("🚀 SIMPLE LANGGRAPH RAG SYSTEM")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            question = input("💬 Ask a question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue
            
            print("🔄 Processing...")
            answer = process_question(question)
            print(f"\n✅ Answer:\n{answer}\n")
            
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Goodbye!")

if __name__ == "__main__":
    main()