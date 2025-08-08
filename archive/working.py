# ==============================================================================
# MINIMAL LANGGRAPH RAG SYSTEM - ESSENTIAL FUNCTIONALITY ONLY
# ==============================================================================
# Streamlined version with core functionality preserved

import os
import logging
import time
import json
from datetime import datetime
from typing import TypedDict, List, Dict, Any

# Environment setup
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/genaiexeco-development")

if not GEMINI_API_KEY:
    raise ValueError("❌ Please set GEMINI_API_KEY in .env file")

# Core imports
import pymongo
from bson import ObjectId
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

print("🚀 MINIMAL LANGGRAPH RAG SYSTEM")
print(f"✅ Gemini API Key: {'Present' if GEMINI_API_KEY else 'Missing'}")
print(f"✅ MongoDB URI: {MONGODB_URI}")

# ==============================================================================
# CORE COMPONENTS
# ==============================================================================

# MongoDB Executor (Simplified)
class MongoDBExecutor:
    def __init__(self):
        self.connected = False
        try:
            self.client = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
            self.db = self.client["genaiexeco-development"]
            self.client.admin.command('ping')
            self.connected = True
            logger.info("✅ MongoDB connected")
        except Exception as e:
            logger.error(f"❌ MongoDB failed: {e}")
    
    def execute_query(self, query_data: Dict) -> Dict[str, Any]:
        if not self.connected:
            return {"success": False, "error": "MongoDB not connected", "results": [], "count": 0}
        
        try:
            collection_name = query_data.get("collection")
            pipeline = query_data.get("pipeline", [])
            
            if not collection_name:
                return {"success": False, "error": "No collection specified", "results": [], "count": 0}
            
            collection = self.db[collection_name]
            cursor = collection.aggregate(pipeline)
            results = [self._convert_objectid(doc) for doc in cursor]
            
            logger.info(f"✅ Query executed: {len(results)} records from {collection_name}")
            
            return {
                "success": True,
                "collection": collection_name,
                "results": results,
                "count": len(results),
                "pipeline": pipeline
            }
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {"success": False, "error": str(e), "results": [], "count": 0}
    
    def _convert_objectid(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_objectid(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_objectid(item) for item in obj]
        return obj

# Essential Keywords for Classification
DATABASE_KEYWORDS = [
    "list users", "show users", "all users", "admin users", "users",
    "list prompts", "show prompts", "all prompts", "prompts", "prompt",
    "list files", "show files", "all files", "processed files", "files",
    "list batches", "show batches", "batch processing", "batches",
    "cost evaluation", "ai costs", "processing costs", "total cost", "cost",
    "extractions", "document extractions", "confidence", "high confidence",
    "obligations", "obligation extractions", "legal obligations",
    "compliance", "compliance tracking", "compliance status",
    "agent activity", "agent performance", "activity"
]

SYSTEM_CAPABILITY_KEYWORDS = [
    "what do you do", "what can you do", "who are you", "what are you",
    "how can you help", "hello", "hi", "help", "capabilities"
]

# AI Components Setup
def setup_ai_components():
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    
    # Enhanced schema documents with detailed information
    detailed_schema_docs = [
        Document(
            page_content="""COMPREHENSIVE MONGODB SCHEMA - Document Intelligence Platform

CORE COLLECTIONS WITH DETAILED SCHEMAS:

1. USERS COLLECTION (6 users)
- userId (String, UUID): Primary key, referenced by all other collections
- emailId (String): Unique email, login identifier (e.g., @execo.com domain)
- firstName, lastName (String): User names
- role (String): "admin" or "user" - critical for filtering admin users
- authSource (String): "google" or "self" authentication method
- googleId (String/Float): Google OAuth identifier
- createdAt, updatedAt (Date/String): Timestamp tracking

2. FILES COLLECTION (98 documents)
- fileId (String): Primary key, referenced by extractions and costs
- fileName (String): Original file name with extension
- status (String): "Created", "Processed", "Failed"
- container, blobName, url (String): Storage location details
- size (Number): File size in bytes
- additionalFields (Object): Contains AI processing results
  - classificationType: Document type (e.g., "Master Services Agreement")
  - confidenceScore: Classification confidence
  - complianceOutput: GDPR/DORA compliance results JSON
  - totalCostPerFileIdInUSD: Processing cost per file
  - totalInputTokens, totalOutputTokens: Token usage

3. PROMPTS COLLECTION (AI instruction library)
- promptId (String): Primary key, referenced by documentmappings
- promptName (String): Human-readable name (e.g., "Term And Duration")
- promptType (String): "Attribute" or "Clause"
- promptText (String): Complete AI instructions for extraction
- description (String): Brief explanation of prompt purpose
- isSeeded (Boolean): Flag for initial system prompts

4. BATCHES COLLECTION (Processing jobs)
- batchId (String): Primary key, referenced by costs
- batchName (String): Human names like "MSA 3", "NDA test"
- status (String): "Processed", "Processing", "Failed", "Queued"
- files (Array): Contains fileId, fileName, totalTimeTakenByFile
- totalTimeTakenByBatch (String): Processing time with units

5. DOCUMENTEXTRACTIONS COLLECTION (2,600+ extractions)
- value (String): Extracted content from documents
- name (String): Extraction field name
- confidenceScore (Number): AI confidence (0-100, >90 is high confidence)
- type (String): "Attribute", "Clause", "Entity"
- reasoning (String): AI explanation for extraction
- citation (String): Source location in document
- batchId, fileId, promptId (String): References to parent records

6. COSTEVALUTIONFORLLM COLLECTION (Cost tracking)
- batchId, fileId, promptId (String): References for cost attribution
- inputTokens, outputTokens, totalTokens (Number): Token consumption
- totalCostInUSD (Number): Monetary cost in USD for cost analysis

7. LLMPRICING COLLECTION (3 pricing tiers)
- modelVariant (String): Model names like "Google Gemini 1.5 flash"
- ratePerMillionInputTokens, ratePerMillionOutputTokens (Number): Pricing rates

8. OBLIGATIONEXTRACTIONS COLLECTION (488 obligations)
- obligationExtractionId (String): Primary key
- name, description (String): Legal obligation details
- metadata (Object): Contains frequency, criticality, financial impact

9. DOCUMENTMAPPINGS COLLECTION (11 mappings)
- sysId (String): Primary key
- documentId (String): Document reference
- promptIds (Array): Array of prompt IDs (up to 87 prompts per mapping)

10. DOCUMENTTYPES COLLECTION (Classification vocabulary)
- documentId (String): Primary key
- documentType (String): Categories like "Share Purchase Agreement"
- description (String): Detailed type description

QUERY PATTERNS FOR BUSINESS INTELLIGENCE:

User Management:
- All users: {"collection": "users", "pipeline": [{"$project": {"userId": 1, "emailId": 1, "role": 1}}, {"$limit": 50}]}
- Admin users only: {"collection": "users", "pipeline": [{"$match": {"role": {"$regex": "admin", "$options": "i"}}}, {"$project": {"userId": 1, "emailId": 1, "firstName": 1, "lastName": 1}}]}

Document Processing:
- High confidence extractions: {"collection": "documentextractions", "pipeline": [{"$match": {"confidenceScore": {"$gt": 90}}}, {"$project": {"name": 1, "value": 1, "confidenceScore": 1}}, {"$limit": 50}]}
- Processed files: {"collection": "files", "pipeline": [{"$match": {"status": "Processed"}}, {"$project": {"fileName": 1, "status": 1, "additionalFields.classificationType": 1}}, {"$limit": 50}]}

Cost Analysis:
- Total AI costs: {"collection": "costevalutionforllm", "pipeline": [{"$group": {"_id": null, "totalCost": {"$sum": "$totalCostInUSD"}, "totalTokens": {"$sum": "$totalTokens"}}}, {"$project": {"_id": 0, "totalCost": 1, "totalTokens": 1}}]}
- Cost by batch: {"collection": "costevalutionforllm", "pipeline": [{"$group": {"_id": "$batchId", "totalCost": {"$sum": "$totalCostInUSD"}}}, {"$sort": {"totalCost": -1}}, {"$limit": 10}]}

Workflow Analytics:
- Batch processing status: {"collection": "batches", "pipeline": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]}
- Prompt usage by type: {"collection": "prompts", "pipeline": [{"$group": {"_id": "$promptType", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]}

Legal & Compliance:
- Legal obligations: {"collection": "obligationextractions", "pipeline": [{"$project": {"name": 1, "description": 1}}, {"$limit": 50}]}
- Document types: {"collection": "documenttypes", "pipeline": [{"$project": {"documentType": 1, "description": 1}}, {"$limit": 50}]}

RELATIONSHIPS AND FOREIGN KEYS:
- users.userId → files.createdBy, batches.createdBy, documentmappings.createdBy
- files.fileId → documentextractions.fileId, costevalutionforllm.fileId
- prompts.promptId → documentmappings.promptIds, costevalutionforllm.promptId  
- batches.batchId → documentextractions.batchId, costevalutionforllm.batchId""",
            metadata={"source": "detailed_schema.txt"}
        )
    ]
    
    db = Chroma.from_documents(detailed_schema_docs, embedding_function)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", 
        temperature=0, 
        google_api_key=GEMINI_API_KEY
    )
    
    # Enhanced RAG Chain with better instructions
    template = """You are a MongoDB query expert for a document intelligence platform.

DETAILED DATABASE SCHEMA:
{context}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. Choose the most appropriate collection based on question keywords
2. For user queries: use "users" collection
3. For admin users: filter by role field: {{"role": {{"$regex": "admin", "$options": "i"}}}}
4. For counting queries (e.g., "how many admins"): use $match and $count stages in the pipeline, and always use the "users" collection for admin counts.
5. For high confidence: use confidenceScore > 90 in documentextractions
6. For cost analysis: use costevalutionforllm collection with totalCostInUSD field
7. For prompts: use prompts collection with promptName, promptType fields
8. Always include proper $project stage to limit returned fields
9. For counting queries: use $count instead of $limit
10. For "all" or "show me everything" queries: do NOT include $limit
11. Only use $limit when user asks for "first few", "top 10", or similar limited results

Generate ONLY the MongoDB query in JSON format:
{{
    "collection": "exact_collection_name",
    "pipeline": [
        {{"$match": {{"field": "criteria"}}}},
        {{"$project": {{"field1": 1, "field2": 1}}}}
    ]
}}"""

    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm
    
    return retriever, llm, rag_chain

# Pydantic Models
class QuestionClassification(BaseModel):
    score: str = Field(description="'on-topic' for database questions, 'external' for system questions, 'off-topic' for unrelated")

# LangGraph State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    question: HumanMessage

# ==============================================================================
# WORKFLOW NODES (Simplified)
# ==============================================================================

def question_classifier(state: AgentState, llm, mongodb_executor):
    """Classify question type"""
    logger.info("🔍 Classifying question")
    
    question = state["question"].content.lower()
    
    # Check for system capability questions first
    if any(phrase in question for phrase in SYSTEM_CAPABILITY_KEYWORDS):
        state["on_topic"] = "external"
        state["messages"] = [state["question"]]
        
        # Generate system response
        response = f"""I'm a document intelligence assistant that can analyze your MongoDB database with 19 collections.

I can help you with:
• User Management: List users, admin roles, authentication data
• Document Processing: 98 files, processing status, document metadata  
• AI Extractions: 2,600+ extractions with confidence scores
• Cost Analysis: AI processing costs, token usage tracking
• Compliance: 488 legal obligations, regulatory tracking
• Workflow: Batch processing, agent performance metrics

Ask me database questions like: 'List all admin users' or 'Show high-confidence extractions'"""
        
        state["messages"].append(AIMessage(content=response))
        return state
    
    # Check for database questions - be more inclusive
    if (any(phrase in question for phrase in DATABASE_KEYWORDS) or 
        any(word in question for word in ["list", "show", "all"]) and 
        any(word in question for word in ["users", "prompts", "files", "batches", "extractions", "obligations", "compliance", "cost", "activity"])):
        state["on_topic"] = "on-topic"
        return state
    
    # AI classifier fallback - make it more inclusive for database questions
    system_msg = SystemMessage(content="""Classify questions for a document intelligence platform with MongoDB collections.
    
    Classify as 'on-topic' for ANY questions about:
    - Database queries: users, prompts, files, batches, extractions, obligations, compliance, costs
    - Business analytics: metrics, reports, analysis
    - Document processing: AI extractions, confidence scores
    - System data: any mention of listing, showing, or querying data
    
    Classify as 'off-topic' ONLY for completely unrelated topics like:
    - Weather, cooking, sports, entertainment, personal advice
    
    When in doubt, classify as 'on-topic'.
    Answer only 'on-topic' or 'off-topic'.""")
    
    structured_llm = llm.with_structured_output(QuestionClassification)
    result = structured_llm.invoke([system_msg, HumanMessage(content=question)])
    
    state["on_topic"] = result.score if result.score in ["on-topic", "off-topic"] else "on-topic"  # Default to on-topic
    logger.info(f"🎯 AI Classification: {state['on_topic']}")
    return state

def retrieve_and_generate(state: AgentState, retriever, rag_chain, mongodb_executor):
    """Retrieve schema and generate MongoDB query"""
    logger.info("📖 Retrieving and generating")
    
    if "messages" not in state:
        state["messages"] = [state["question"]]
    
    question = state["question"].content
    
    # Retrieve relevant schema
    documents = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in documents])
    
    # Generate MongoDB query
    response = rag_chain.invoke({"context": context, "question": question})
    generation = response.content.strip()
    
    # Try to execute MongoDB query
    if mongodb_executor.connected and "```json" in generation:
        try:
            # Extract JSON
            json_start = generation.find("```json") + 7
            json_end = generation.find("```", json_start)
            if json_end == -1:
                json_end = len(generation)
            
            json_content = generation[json_start:json_end].strip()
            query_data = json.loads(json_content)
            
            # Execute query
            logger.info(f"🔍 Executing query on: {query_data.get('collection')}")
            result = mongodb_executor.execute_query(query_data)
            
            if result["success"]:
                enhanced_response = f"""🎯 **MongoDB Query Executed Successfully!**

**Collection:** `{result['collection']}`
**Records Found:** {result['count']}

**📊 RESULTS:**
```json
{json.dumps(result['results'][:5], indent=2)}
```

**🔧 Query Pipeline:**
```json
{json.dumps(result['pipeline'], indent=2)}
```"""
                
                state["messages"].append(AIMessage(content=enhanced_response))
                logger.info(f"✅ Success: {result['count']} records returned")
            else:
                error_response = f"""⚠️ Query execution failed: {result['error']}

**Generated Query:**
```json
{json.dumps(query_data, indent=2)}
```"""
                state["messages"].append(AIMessage(content=error_response))
        
        except Exception as e:
            state["messages"].append(AIMessage(content=generation))
            logger.info(f"⚠️ JSON parsing failed: {e}")
    else:
        if not mongodb_executor.connected:
            no_db_response = f"""📝 **MongoDB Query Generated:**

{generation}

**Note:** MongoDB not connected. Query ready for execution."""
            state["messages"].append(AIMessage(content=no_db_response))
        else:
            state["messages"].append(AIMessage(content=generation))
    
    return state

def off_topic_response(state: AgentState):
    """Handle off-topic questions"""
    logger.info("🚫 Off-topic response")
    if "messages" not in state:
        state["messages"] = [state["question"]]
    state["messages"].append(AIMessage(content="I'm sorry! I cannot answer this question!"))
    return state

# ==============================================================================
# WORKFLOW COMPILATION
# ==============================================================================

def create_workflow():
    """Create minimal LangGraph workflow"""
    logger.info("🔧 Building workflow...")
    
    # Initialize components
    retriever, llm, rag_chain = setup_ai_components()
    mongodb_executor = MongoDBExecutor()
    
    # Create workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes with component injection
    workflow.add_node("classify", lambda state: question_classifier(state, llm, mongodb_executor))
    workflow.add_node("retrieve_generate", lambda state: retrieve_and_generate(state, retriever, rag_chain, mongodb_executor))
    workflow.add_node("off_topic", off_topic_response)
    
    # Router function
    def route_question(state: AgentState) -> str:
        classification = state.get("on_topic", "").strip().lower()
        if classification == "external":
            return END  # External questions already handled in classifier
        elif classification == "on-topic":
            return "retrieve_generate"
        else:
            return "off_topic"
    
    # Add edges
    workflow.add_conditional_edges("classify", route_question, {
        "retrieve_generate": "retrieve_generate",
        "off_topic": "off_topic",
        END: END
    })
    workflow.add_edge("retrieve_generate", END)
    workflow.add_edge("off_topic", END)
    workflow.set_entry_point("classify")
    
    # Compile
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("✅ Workflow compiled successfully")
    return graph, mongodb_executor

# ==============================================================================
# EXECUTION FUNCTIONS
# ==============================================================================

def process_question(graph, question: str):
    """Process a single question"""
    start_time = time.time()
    logger.info(f"🚀 Processing: '{question}'")
    
    try:
        input_data = {"question": HumanMessage(content=question)}
        result = graph.invoke(input_data, config={"configurable": {"thread_id": int(time.time())}})
        
        messages = result.get("messages", [])
        final_message = messages[-1] if messages else None
        
        processing_time = time.time() - start_time
        return {
            "question": question,
            "answer": final_message.content if final_message else "No response generated",
            "processing_time": f"{processing_time:.2f}s",
            "status": "success" if final_message else "failed"
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Processing failed: {e}")
        return {
            "question": question,
            "answer": f"Error: {str(e)}",
            "processing_time": f"{processing_time:.2f}s",
            "status": "error"
        }

def interactive_mode(graph, mongodb_executor):
    """Simple interactive mode"""
    print(f"\n🎯 INTERACTIVE MODE - Type 'quit' to exit")
    print(f"🔗 MongoDB: {'✅ Connected' if mongodb_executor.connected else '❌ Disconnected'}")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n💬 Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🔄 Processing...")
            result = process_question(graph, question)
            
            print(f"\n✅ Answer:")
            print(f"{result['answer']}")
            print(f"\n📊 Time: {result['processing_time']}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def batch_test(graph):
    """Quick batch test"""
    test_questions = [
        "What do you do?",  # External
        "List all admin users",  # Database
        "Give me a list of prompts",  # Database
        "Show high-confidence extractions",  # Database
        "What's the weather?",  # Off-topic
    ]
    
    print("\n🧪 BATCH TEST")
    print("=" * 30)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}: {question}")
        result = process_question(graph, question)
        print(f"   ✅ {result['status'].upper()}: {result['answer'][:80]}...")
        print(f"   ⏱️ {result['processing_time']}")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main function"""
    try:
        # Create workflow
        print("🔧 Initializing...")
        graph, mongodb_executor = create_workflow()
        
        # Mode selection
        print("\n🎯 SELECT MODE:")
        print("1. Interactive Mode")
        print("2. Batch Test")
        print("3. Single Question")
        
        choice = input("Choice (1-3): ").strip()
        
        if choice == "1":
            interactive_mode(graph, mongodb_executor)
        elif choice == "2":
            batch_test(graph)
        elif choice == "3":
            question = input("Enter question: ").strip()
            if question:
                result = process_question(graph, question)
                print(f"\n✅ Answer: {result['answer']}")
                print(f"⏱️ Time: {result['processing_time']}")
        else:
            print("❌ Invalid choice")
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()