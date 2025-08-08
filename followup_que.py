# ==============================================================================
# ENHANCED LANGGRAPH RAG SYSTEM - FULLY-FEATURED & COMPLETE
# ==============================================================================
# Final version with integrated schema, retries, conversational context, 
# format detection, and dynamic LLM responses for external questions.

import os
import logging
import time
import json
import uuid
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Literal, Optional

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

print("🚀 ENHANCED LANGGRAPH RAG SYSTEM")
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
            if not collection_name: return {"success": False, "error": "No collection specified"}
            
            collection = self.db[collection_name]
            cursor = collection.aggregate(pipeline)
            results = [self._convert_objectid(doc) for doc in cursor]
            
            logger.info(f"✅ Query executed: {len(results)} records from {collection_name}")
            return {"success": True, "collection": collection_name, "results": results, "count": len(results), "pipeline": pipeline}
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {"success": False, "error": str(e), "results": [], "count": 0}

    def _convert_objectid(self, obj):
        if isinstance(obj, ObjectId): return str(obj)
        elif isinstance(obj, dict): return {key: self._convert_objectid(value) for key, value in obj.items()}
        elif isinstance(obj, list): return [self._convert_objectid(item) for item in obj]
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

# ==============================================================================
# AI COMPONENTS & PROMPTS
# ==============================================================================

def setup_ai_components():
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)
    
    # MODIFIED: Full schema definition is now included
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
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GEMINI_API_KEY)
    
    template = """You are a MongoDB query expert for a document intelligence platform.
Your goal is to generate a precise MongoDB query based on the user's question and the conversation history.

CONVERSATION HISTORY:
{chat_history}

DETAILED DATABASE SCHEMA:
{context}

USER QUESTION: {question}

IMPORTANT INSTRUCTIONS:
1. Choose the most appropriate collection based on question keywords.
2. For user queries: use "users" collection.
3. For admin users: filter by role field: {{"role": {{"$regex": "admin", "$options": "i"}}}}
4. For high confidence: use confidenceScore > 90 in documentextractions.
5. For cost analysis: use costevalutionforllm collection.
6. For prompts: use prompts collection.
7. Always include a $project stage to return only necessary fields.
8. For counting: use $count. For "all": do NOT use $limit.
9. Generate ONLY the MongoDB query in a single JSON object.

JSON Query:"""

    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = prompt | llm
    
    return retriever, llm, rag_chain

# Pydantic models for structured output
class QuestionClassification(BaseModel):
    """Classify the user's question."""
    classification: str = Field(description="'internal' for database questions, 'external' for general/system questions, 'off-topic' for unrelated.")

class OutputFormat(BaseModel):
    """Determine the user's desired output format."""
    format: Literal['json', 'table', 'natural_language'] = Field(description="The desired format for the final output.")
    reasoning: str = Field(description="A brief explanation for the chosen format.")

class FollowUpSuggestion(BaseModel):
    """A model to hold a suggested follow-up question and its corresponding query."""
    follow_up_question: str = Field(description="A clear, concise, yes/no question to ask the user as a follow-up. Should be 'NONE' if no good follow-up exists.")
    contextual_query: Optional[Dict] = Field(None, description="The full MongoDB JSON query to be executed if the user answers 'yes' to the follow_up_question.")

# LangGraph State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    question: HumanMessage
    on_topic: str
    output_format: str
    retry_count: int
    follow_up_question: str
    follow_up_context: Optional[Dict]
    route: str
# ==============================================================================
# WORKFLOW NODES
# ==============================================================================

def format_chat_history(messages: List[BaseMessage]) -> str:
    """Helper to format message history for the LLM prompt."""
    return "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in messages])

def _format_as_table(results: List[Dict]) -> str:
    """Helper to format JSON results into a markdown table."""
    if not results: return "No results to display."
    headers = results[0].keys()
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    rows = [separator_line]
    for item in results:
        row = "| " + " | ".join(str(item.get(h, '')) for h in headers) + " |"
        rows.append(row)
    return header_line + "\n" + "\n".join(rows)

def question_classifier(state: AgentState, llm):
    logger.info("🔍 Classifying question...")
    question = state["question"].content.lower()

    if any(phrase in question for phrase in SYSTEM_CAPABILITY_KEYWORDS):
        state["on_topic"] = "external"
        return state
    
    if any(phrase in question for phrase in DATABASE_KEYWORDS):
        state["on_topic"] = "internal"
        return state

    system_msg = SystemMessage(content="""Classify questions for a document intelligence platform with MongoDB.
- 'internal': Questions about database data (users, files, costs, extractions).
- 'external': General questions, or questions about what the system can do.
- 'off-topic': Unrelated questions (weather, sports).
When in doubt, classify as 'internal'.""")
    
    structured_llm = llm.with_structured_output(QuestionClassification)
    result = structured_llm.invoke([system_msg, HumanMessage(content=question)])
    
    state["on_topic"] = result.classification
    logger.info(f"🎯 AI Classification: {state['on_topic']}")
    return state

def internal_search_node(state: AgentState, retriever, rag_chain, llm, mongodb_executor):
    logger.info("📖 Internal Search: Generating and executing MongoDB query...")
    MAX_RETRIES = 3
    state["retry_count"] = 0
    
    question = state["question"].content
    chat_history = format_chat_history(state["messages"])
    
    logger.info("... 1/4: Detecting output format")
    format_detector = llm.with_structured_output(OutputFormat)
    format_prompt = f"Based on the user's question, what is the best output format? Question: '{question}'"
    try:
        format_result = format_detector.invoke(format_prompt)
        state["output_format"] = format_result.format
        logger.info(f"... Format detected: {state['output_format']}")
    except Exception as e:
        logger.warning(f"⚠️ Format detection failed, defaulting to JSON. Error: {e}")
        state["output_format"] = "json"

    logger.info("... 2/4: Retrieving context")
    documents = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in documents])

    for attempt in range(MAX_RETRIES):
        state["retry_count"] = attempt + 1
        logger.info(f"... 3/4: Generating query (Attempt {state['retry_count']}/{MAX_RETRIES})")
        
        try:
            response = rag_chain.invoke({"context": context, "question": question, "chat_history": chat_history})
            generation = response.content.strip().replace("```json", "").replace("```", "")
            query_data = json.loads(generation)

            logger.info(f"...... Executing query on: {query_data.get('collection')}")
            result = mongodb_executor.execute_query(query_data)
            
            if result["success"]:
                logger.info(f"✅ Query successful on attempt {state['retry_count']}")
                logger.info(f"... 4/4: Formatting output as {state['output_format']}")
                final_answer = ""
                if state["output_format"] == "table":
                    final_answer = _format_as_table(result['results'])
                elif state["output_format"] == "natural_language":
                    summary_prompt = f"Summarize the following JSON results in a friendly, natural language response. Results:\n{json.dumps(result['results'][:5], indent=2)}"
                    final_answer = llm.invoke(summary_prompt).content
                else: # Default to JSON
                    final_answer = f"```json\n{json.dumps(result['results'][:5], indent=2)}\n```"

                enhanced_response = f"""🎯 **Query Executed Successfully!**
**Collection:** `{result['collection']}`
**Records Found:** {result['count']} (showing up to 5)

**📊 RESULTS ({state['output_format'].replace('_', ' ').title()}):**
{final_answer}"""
                state["messages"].append(AIMessage(content=enhanced_response))
                return state
            
            else:
                logger.warning(f"⚠️ Query execution failed: {result['error']}")
                chat_history += f"\nATTEMPT {state['retry_count']} FAILED. Error: {result['error']}. Generated Query: {generation}. Please generate a corrected query."

        except Exception as e:
            logger.error(f"❌ Query generation/parsing failed: {e}")
            chat_history += f"\nATTEMPT {state['retry_count']} FAILED. Error: {e}. The previous attempt was invalid JSON. Please generate a valid JSON query."

        time.sleep(1)

    logger.error("❌ All query attempts failed.")
    fail_message = "I tried multiple times but could not generate a valid query for your request. Please try rephrasing your question."
    state["messages"].append(AIMessage(content=fail_message))
    return state

def external_search_node(state: AgentState, llm):
    logger.info("🌐 External Search: Generating dynamic response...")
    question = state["question"].content
    chat_history = format_chat_history(state["messages"])
    
    prompt = f"""You are a helpful AI assistant for a document intelligence platform.
    
Previous conversation:
{chat_history}

The user is asking a general question. Provide a helpful and concise answer.
User Question: {question}"""
    
    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    return state

def generate_follow_up_node(state: AgentState, llm):
    logger.info("💡 Generating potential follow-up question...")
    
    last_message = state["messages"][-1]
    if "Query Executed Successfully" not in last_message.content:
        state["follow_up_question"] = "NONE"
        return state

    suggestion_llm = llm.with_structured_output(FollowUpSuggestion)
    
    prompt = f"""Given the user's last question and the results they received, suggest a single, helpful follow-up action.
    The action must be a question that can be answered with a new MongoDB query.
    If the follow-up requires user input (like a specific role or name), use a placeholder like '<user_input_needed>' in the query's value field.
    If no logical follow-up exists, respond with "NONE" for the follow_up_question.

    USER'S QUESTION: "{state['question'].content}"
    
    RESULTS THEY RECEIVED:
    {last_message.content}

    Example:
    If the user asked for "list users" and saw roles 'admin' and 'user', a good follow-up is "Would you like to filter by a specific role?".
    The contextual_query would be: {{"collection": "users", "pipeline": [{{"$match": {{"role": "<user_input_needed>"}}}}]}}
    """
    
    try:
        suggestion = suggestion_llm.invoke(prompt)
        if suggestion.follow_up_question and suggestion.follow_up_question.upper() != "NONE":
            logger.info(f"✅ Follow-up generated: '{suggestion.follow_up_question}'")
            state["follow_up_question"] = suggestion.follow_up_question
            state["follow_up_context"] = suggestion.contextual_query
            # Append the follow-up question to the last message for the user to see
            last_message.content += f"\n\n**{suggestion.follow_up_question}**"
        else:
            state["follow_up_question"] = "NONE"
    except Exception as e:
        logger.error(f"❌ Follow-up generation failed: {e}")
        state["follow_up_question"] = "NONE"

    return state

# Node to handle the user's "yes" or "no" response
def handle_follow_up_node(state: AgentState, mongodb_executor, llm):
    logger.info("🤝 Handling user's answer to follow-up question...")
    user_response = state["question"].content
    
    is_affirmative = any(user_response.lower().startswith(word) for word in ["yes", "sure", "ok", "yep"])
    
    if is_affirmative:
        # THE FIX: Check for the context more robustly.
        query_template = state.get("follow_up_context")
        if query_template is None:
            state["messages"].append(AIMessage(content="I seem to have lost the context. Could you ask again?"))
            # Clear state just in case, then return
            state["follow_up_question"] = "NONE"
            state["follow_up_context"] = None
            return state

        query_str = json.dumps(query_template)

        if "<user_input_needed>" in query_str:
            logger.info("... Query requires user input. Extracting...")
            
            extraction_prompt = f"""From the user's response, extract the specific value they want to use for filtering.
            User's response: "{user_response}"
            Example: If the response is "yes, admin role", the value is "admin".
            Example: If the response is "sure, show me the legal ones", the value is "legal".
            Extracted value:"""
            
            extracted_value = llm.invoke(extraction_prompt).content.strip().lower().replace("'", "").replace('"', '')
            logger.info(f"... Extracted value: '{extracted_value}'")

            final_query_str = query_str.replace("<user_input_needed>", extracted_value)
            final_query = json.loads(final_query_str)
            
        else:
            logger.info("... Query is self-contained. Executing as is.")
            final_query = query_template

        logger.info(f"👍 User said yes. Executing final query: {final_query}")
        result = mongodb_executor.execute_query(final_query)
        if result.get("success"):
            formatted_results = _format_as_table(result.get('results', []))
            response_message = f"Here you go!\n\n{formatted_results}"
            state["messages"].append(AIMessage(content=response_message))
        else:
            state["messages"].append(AIMessage(content=f"I'm sorry, I ran into an error: {result.get('error')}"))
    else:
        logger.info("👎 User said no or provided a new question.")
        state["messages"].append(AIMessage(content="Okay, sounds good. What else can I help you with?"))

    # Clear the follow-up state to prevent getting stuck
    state["follow_up_question"] = "NONE"
    state["follow_up_context"] = None
    return state

def pre_router_node(state: AgentState) -> Dict[str, Any]:
    """
    Determines the initial path and preserves ALL state.
    """
    logger.info("🚦 Pre-routing based on pending follow-up...")
    
    # Create a copy of the current state to avoid modifying it directly
    updates = state.copy()
    
    if state.get("follow_up_question") and state.get("follow_up_question", "").upper() != "NONE":
        logger.info("... Storing route: handle_follow_up")
        updates["route"] = "handle_follow_up"
    else:
        logger.info("... Storing route: classify")
        updates["route"] = "classify"
        
    # Return the entire updated state dictionary. This guarantees no keys are lost.
    return updates

def off_topic_response(state: AgentState):
    logger.info("🚫 Off-topic response")
    state["messages"].append(AIMessage(content="I'm sorry, that question is outside the scope of my capabilities. I can only answer questions about the document intelligence platform's database or my own functions."))
    return state

# ==============================================================================
# WORKFLOW COMPILATION
# ==============================================================================

def create_workflow():
    logger.info("🔧 Building proactive workflow...")
    retriever, llm, rag_chain = setup_ai_components()
    mongodb_executor = MongoDBExecutor()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("pre_router", pre_router_node)
    workflow.add_node("classify", lambda state: question_classifier(state, llm))
    workflow.add_node("internal_search", lambda state: internal_search_node(state, retriever, rag_chain, llm, mongodb_executor))
    workflow.add_node("external_search", lambda state: external_search_node(state, llm))
    workflow.add_node("off_topic", off_topic_response)
    workflow.add_node("generate_follow_up", lambda state: generate_follow_up_node(state, llm))
    workflow.add_node("handle_follow_up", lambda state: handle_follow_up_node(state, mongodb_executor, llm))
    
    def route_from_classification(state: AgentState) -> str:
        classification = state.get("on_topic", "").strip().lower()
        if classification == "internal":
            return "internal_search"
        elif classification == "external":
            return "external_search"
        else:
            return "off_topic"
            
    workflow.set_entry_point("pre_router")

    # MODIFIED: The conditional edge now reads the 'route' key from the state.
    workflow.add_conditional_edges(
        "pre_router",
        lambda state: state["route"], # Read the routing decision from the state
        {
            "handle_follow_up": "handle_follow_up",
            "classify": "classify",
        }
    )

    workflow.add_conditional_edges("classify", route_from_classification)
    workflow.add_edge("internal_search", "generate_follow_up")
    
    workflow.add_edge("generate_follow_up", END)
    workflow.add_edge("handle_follow_up", END)
    workflow.add_edge("external_search", END)
    workflow.add_edge("off_topic", END)
    
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("✅ Proactive workflow compiled successfully")
    return graph, mongodb_executor

# ==============================================================================
# EXECUTION FUNCTIONS
# ==============================================================================

def process_question(graph, question: str, thread_id: str, last_state: Optional[Dict] = None):
    start_time = time.time()
    logger.info(f"🚀 Processing: '{question}' for thread_id: {thread_id}")
    
    answer = "No response generated"
    final_state_for_next_turn = None # Initialize a variable to hold the state
    
    try:
        # THE FIX: Construct the input from the previous state.
        if last_state:
    # We are in a conversation. Use the last state.
            inputs = last_state.copy()  # Make a copy to avoid mutation
    # Update with the new question.
            inputs['question'] = HumanMessage(content=question)
    # Append the new question to the message history
            inputs['messages'] = inputs.get('messages', []) + [HumanMessage(content=question)]
        else:
            # This is the first turn.
            inputs = {
                "messages": [HumanMessage(content=question)],
                "question": HumanMessage(content=question)
            }
        
        for event in graph.stream(inputs, config={"configurable": {"thread_id": thread_id}}):
            for key, value in event.items():
                final_state_for_next_turn = value # Capture the final state of the graph
        
        if final_state_for_next_turn:
            messages = final_state_for_next_turn.get("messages", [])
            final_message = messages[-1] if messages else None
            if final_message:
                answer = final_message.content

    except Exception as e:
        logger.error(f"❌ Processing failed for thread {thread_id}: {e}", exc_info=True)
        answer = f"An unexpected error occurred: {str(e)}"
        final_state_for_next_turn = None # Reset state on error
        
    finally:
        processing_time = time.time() - start_time
        logger.info(f"✅ Finished processing in {processing_time:.2f}s")
        # Return both the answer AND the final state for the next turn
        return {"answer": answer, "time": f"{processing_time:.2f}s"}, final_state_for_next_turn

def interactive_mode(graph, mongodb_executor):
    thread_id = str(uuid.uuid4())
    print(f"\n🎯 INTERACTIVE MODE - New conversation started (Thread ID: {thread_id}). Type 'quit' to exit.")
    print(f"🔗 MongoDB: {'✅ Connected' if mongodb_executor.connected else '❌ Disconnected'}")
    print("=" * 60)
    
    # NEW: We will keep track of the last known state here.
    last_state = None

    while True:
        try:
            question = input("\n💬 Ask a question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question: continue
            
            print("🔄 Processing...")
            # Pass the last known state to the processing function
            result, last_state = process_question(graph, question, thread_id, last_state)
            
            print(f"\n✅ Answer:")
            print(f"{result['answer']}")
            print(f"\n📊 Time: {result['time']}")
            
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"❌ Error in interactive loop: {e}")
            last_state = None # Reset state on error
            
    print("👋 Goodbye!")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        graph, mongodb_executor = create_workflow()
        interactive_mode(graph, mongodb_executor)
    except Exception as e:
        logger.error(f"❌ Fatal error during initialization: {e}", exc_info=True)