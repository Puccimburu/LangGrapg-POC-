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
from typing import TypedDict, List, Dict, Any, Literal, Optional, Union, Annotated

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
from pydantic import BaseModel, Field, model_validator

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
    follow_up_question: str = Field(description="A clear, concise, yes/no question to ask the user as a follow-up. Must be 'NONE' if no good follow-up exists.")
    contextual_query: Optional[Dict] = Field(None, description="The full MongoDB JSON query to be executed if the user answers 'yes'. Must be provided if follow_up_question is not 'NONE'.")

    @model_validator(mode='after')
    def check_query_if_question_exists(self) -> 'FollowUpSuggestion':
        if self.follow_up_question.upper() != 'NONE' and not self.contextual_query:
            raise ValueError("If a follow_up_question is provided, contextual_query must also be provided.")
        return self

class TableComponent(BaseModel):
    type: Literal["table"] = "table"
    headers: List[str]
    rows: List[List[Any]]

class ChartDataset(BaseModel):
    label: str
    data: List[float]

class ChartData(BaseModel):
    labels: List[str]
    datasets: List[ChartDataset]

class ChartComponent(BaseModel):
    type: Literal["chart"] = "chart"
    chartType: Literal["bar", "line", "pie"]
    data: ChartData
    
class HtmlComponent(BaseModel):
    type: Literal["html"] = "html"
    content: str
    
class TextComponent(BaseModel):
    type: Literal["text"] = "text"
    content: str

# Create a discriminated union of the components
# This tells Pydantic to use the 'type' field to determine which model to validate against
UiComponent = Union[TableComponent, ChartComponent, HtmlComponent, TextComponent]

# The final response model that the LLM will generate
class StructuredResponse(BaseModel):
    """A structured response model containing a list of UI components."""
    components: List[Annotated[UiComponent, Field(discriminator="type")]]

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
    raw_data: Any # To hold the raw python objects (list of users, text, etc.)
    structured_output: Optional[List[Dict[str, Any]]] # The final JSON output
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
                state["raw_data"] = result['results']
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
    state["raw_data"] = response.content
    return state

def generate_follow_up_node(state: AgentState, llm):
    logger.info("💡 Generating potential follow-up question...")
    
    # We only try to generate a follow-up if there is raw data to analyze
    if not state.get("raw_data"):
        logger.info("... No raw data to analyze, skipping follow-up.")
        state["follow_up_question"] = "NONE"
        state["follow_up_context"] = None
        return state

    suggestion_llm = llm.with_structured_output(FollowUpSuggestion)
    
    # We'll give the LLM two attempts to get it right.
    for attempt in range(2):
        logger.info(f"... Follow-up generation attempt {attempt + 1}/2")
        
        # Construct the prompt for this attempt
        prompt = f"""You are a data analysis assistant. Your task is to suggest a helpful follow-up action based on a user's query and the data they received.

        **Chain of Thought Instructions:**
        1.  **Analyze the Data:** Look at the `raw_data` provided. What are the common fields? Are there obvious ways to filter or aggregate this data that would be useful? For example, filtering by 'role', 'status', or grouping by a category.
        2.  **Formulate a Question:** Based on your analysis, create a clear, simple `follow_up_question` that a user can answer with "yes" and some extra detail. The question should be a natural next step.
        3.  **Construct the Query:** Create the corresponding `contextual_query` in MongoDB JSON format. This query MUST be able to answer the question you formulated. If the user needs to provide a value (like a specific role), use the placeholder "<user_input_needed>" in the query's value field.
        4.  **Final Check:** If you were able to create both a valid question and a valid query, provide them. If not, you MUST set `follow_up_question` to "NONE".

        **User's Original Question:**
        "{state['question'].content}"
        
        **Data the User Received:**
        {json.dumps(state.get("raw_data", "No data provided."), indent=2, default=json_converter)}

        **Example Output:**
        {{
          "follow_up_question": "Would you like to filter by a specific role?",
          "contextual_query": {{
            "collection": "users",
            "pipeline": [
              {{
                "$match": {{
                  "role": "<user_input_needed>"
                }}
              }}
            ]
          }}
        }}

        Now, generate your suggestion.
        """
        
        try:
            suggestion = suggestion_llm.invoke(prompt)
            # The Pydantic validator will raise an error if the output is invalid,
            # which will be caught by the except block. If we get here, it's valid.
            if suggestion.follow_up_question and suggestion.follow_up_question.upper() != "NONE":
                logger.info(f"✅ Follow-up generated successfully on attempt {attempt + 1}.")
                state["follow_up_question"] = suggestion.follow_up_question
                state["follow_up_context"] = suggestion.contextual_query
                return state # Success! Exit the node.
            else:
                # The LLM decided there was no good follow-up. This is also a success.
                logger.info("... LLM decided no follow-up was needed.")
                state["follow_up_question"] = "NONE"
                state["follow_up_context"] = None
                return state # Success! Exit the node.

        except Exception as e:
            logger.warning(f"⚠️ Follow-up generation attempt {attempt + 1} failed: {e}")
            # If this is the last attempt, we give up.
            if attempt == 1:
                logger.error("❌ All follow-up generation attempts failed.")
                state["follow_up_question"] = "NONE"
                state["follow_up_context"] = None
                return state
            # Otherwise, the loop will continue for the next attempt.
    
    # This part should ideally not be reached, but as a fallback:
    state["follow_up_question"] = "NONE"
    state["follow_up_context"] = None
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
            state["raw_data"] = result.get('results', [])
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

def format_output_node(state: AgentState, llm):
    logger.info("🎨 Formatting final output into structured JSON...")
    
    raw_data = state.get("raw_data")
    if raw_data is None:
        logger.warning("... No raw data found to format. Creating a simple text response.")
        # Use the last message as a fallback
        last_message_content = state["messages"][-1].content
        final_components = [{"type": "text", "content": last_message_content}]
        state["structured_output"] = final_components
        return state

    # Prepare the formatter LLM
    formatter_llm = llm.with_structured_output(StructuredResponse)
    
    # Prepare the prompt
    prompt = f"""You are a data visualization expert. Your job is to convert raw data into a structured JSON format for a UI, based on the user's question.
    Choose the best component(s) (table, chart, text, html) to display the information.

    ---
    EXAMPLE OUTPUT FORMAT:
    {{
        "components": [
            {{
                "type": "table",
                "headers": ["Product", "Sales"],
                "rows": [
                    ["Widget A", 150],
                    ["Widget B", 200]
                ]
            }},
            {{
                "type": "text",
                "content": "Widget B is the top seller this quarter."
            }}
        ]
    }}
    ---

    USER'S QUESTION:
    "{state['question'].content}"

    RAW DATA (in Python/JSON format):
    {json.dumps(raw_data, indent=2, default=json_converter)}

    Follow-up question to include (if any): "{state.get('follow_up_question', 'NONE')}"

    INSTRUCTIONS:
    1. Analyze the user's question and the raw data.
    2. If the data is tabular (a list of objects), a "table" component is best.
    3. If the data is aggregated or statistical, a "chart" is a good choice.
    4. If the data is just text, use a "text" component.
    5. If there is a follow-up question to ask the user, add it as a final "text" component.
    6. Generate ONLY the final JSON output containing a list of these components, following the example format precisely.
    """
    
    try:
            # We need a separate LLM instance that does NOT have structured output for the raw call.
            raw_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GEMINI_API_KEY)
            
            # Get the raw string response from the LLM
            raw_response_content = raw_llm.invoke(prompt).content
            
            logger.info(f"... Raw LLM Response for formatting:\n{raw_response_content}")

            # Attempt to find and parse a JSON object from the raw response
            try:
                # Find the start and end of the JSON blob
                json_start = raw_response_content.find('{')
                json_end = raw_response_content.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = raw_response_content[json_start:json_end]
                    
                    # Parse the string into a Python dictionary
                    parsed_json = json.loads(json_str)
                    
                    # Now, validate this dictionary against our Pydantic model
                    validated_response = StructuredResponse.model_validate(parsed_json)
                    
                    # Convert the validated Pydantic models back to a list of dicts
                    final_components = [component.model_dump() for component in validated_response.components]
                    state["structured_output"] = final_components
                    logger.info("... ✅ Successfully parsed and validated structured JSON output.")
                else:
                    # This happens if the LLM response does not contain any JSON object.
                    raise ValueError("Could not find a valid JSON object in the LLM response.")

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️ Initial parsing failed: {e}. Falling back to a text component.")
                # Fallback: if parsing fails, just wrap the whole raw response in a text component.
                state["structured_output"] = [{"type": "text", "content": raw_response_content}]

    except Exception as e:
            logger.error(f"❌ Critical failure in format_output_node: {e}", exc_info=True)
            state["structured_output"] = [{"type": "text", "content": f"Error formatting output: {str(e)}"}]
    
    return state

def json_converter(o):
    if isinstance(o, (datetime, ObjectId)):
        return str(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

# ==============================================================================
# WORKFLOW COMPILATION
# ==============================================================================

def create_workflow():
    logger.info("🔧 Building structured output workflow...")
    retriever, llm, rag_chain = setup_ai_components()
    mongodb_executor = MongoDBExecutor()
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("pre_router", pre_router_node)
    workflow.add_node("classify", lambda state: question_classifier(state, llm))
    workflow.add_node("internal_search", lambda state: internal_search_node(state, retriever, rag_chain, llm, mongodb_executor))
    workflow.add_node("external_search", lambda state: external_search_node(state, llm))
    workflow.add_node("off_topic", off_topic_response)
    workflow.add_node("generate_follow_up", lambda state: generate_follow_up_node(state, llm))
    workflow.add_node("handle_follow_up", lambda state: handle_follow_up_node(state, mongodb_executor, llm))
    workflow.add_node("format_output", lambda state: format_output_node(state, llm))
    
    # Routers
    def route_from_pre_router(state: AgentState) -> str:
        return state.get("route", "classify")

    def route_from_classification(state: AgentState) -> str:
        return state.get("on_topic", "off_topic")
            
    # Define the new graph structure
    workflow.set_entry_point("pre_router")
    workflow.add_conditional_edges("pre_router", route_from_pre_router, {"handle_follow_up": "handle_follow_up", "classify": "classify"})
    workflow.add_conditional_edges("classify", route_from_classification, {"internal": "internal_search", "external": "external_search", "off_topic": "off_topic"})
    
    # CORRECTED WIRING
    workflow.add_edge("internal_search", "generate_follow_up")
    workflow.add_edge("generate_follow_up", "format_output") # The formatter runs AFTER the follow-up is generated
    workflow.add_edge("handle_follow_up", "format_output")   # and after a follow-up is handled
    workflow.add_edge("external_search", "format_output")    # and after an external search
    workflow.add_edge("off_topic", "format_output")      # and after an off-topic response

    # The formatter is the true final step before END
    workflow.add_edge("format_output", END)
    
    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("✅ Structured output workflow compiled successfully")
    return graph, mongodb_executor

# ==============================================================================
# EXECUTION FUNCTIONS
# ==============================================================================

def process_question(graph, question: str, thread_id: str):
    start_time = time.time()
    logger.info(f"🚀 Processing: '{question}' for thread_id: {thread_id}")
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # We only need to provide the new message. The checkpointer handles the history.
    inputs = {"messages": [HumanMessage(content=question)], "question": HumanMessage(content=question)}
    
    final_json_output = [{"type": "text", "content": "No response generated"}]
    
    try:
        # THE FIX: Use .invoke() to get the final result directly.
        # It runs the entire graph and returns the output of the last node(s).
        final_graph_output = graph.invoke(inputs, config=config)
        
        logger.info(f"Final Invoke Output: {final_graph_output}")

        # The output of our graph is the dictionary from the 'format_output' node.
        # We can access the 'structured_output' key directly.
        if final_graph_output and "structured_output" in final_graph_output:
            final_json_output = final_graph_output["structured_output"]
            logger.info("... Successfully extracted 'structured_output' from invoke result.")
        else:
            logger.warning("... 'structured_output' not found in final output. Something is wrong in the graph.")

    except Exception as e:
        logger.error(f"❌ Processing failed for thread {thread_id}: {e}", exc_info=True)
        final_json_output = [{"type": "text", "content": f"An unexpected error occurred: {str(e)}"}]
        
    finally:
        processing_time = time.time() - start_time
        logger.info(f"✅ Finished processing in {processing_time:.2f}s")
        # We no longer need to return the state, as the checkpointer handles it.
        return {"answer": final_json_output, "time": f"{processing_time:.2f}s"}
def interactive_mode(graph, mongodb_executor):
    thread_id = str(uuid.uuid4())
    print(f"\n🎯 INTERACTIVE MODE - New conversation started (Thread ID: {thread_id}). Type 'quit' to exit.")
    print(f"🔗 MongoDB: {'✅ Connected' if mongodb_executor.connected else '❌ Disconnected'}")
    print("=" * 60)

    while True:
        try:
            question = input("\n💬 Ask a question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question: continue
            
            print("🔄 Processing...")
            # We no longer pass or receive state here. The checkpointer handles it all.
            result = process_question(graph, question, thread_id)
            
            print(f"\n✅ Answer:")
            print(json.dumps(result['answer'], indent=4))
            print(f"\n📊 Time: {result['time']}")
            
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"❌ Error in interactive loop: {e}")
            
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