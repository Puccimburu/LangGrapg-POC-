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
    raise ValueError("[ERROR] Please set GEMINI_API_KEY in .env filee")

# Core imports
import pymongo
import msgpack
from bson import ObjectId
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Any

print("ENHANCED LANGGRAPH RAG SYSTEM")
print(f"[OK] Gemini API Key: {'Present' if GEMINI_API_KEY else 'Missing'}")
print(f"[OK] MongoDB URI: {MONGODB_URI}")

# ==============================================================================
# CORE COMPONENTS
# ==============================================================================

# Custom MongoDB Saver with Timestamps  
class TimestampedMongoDBSaver(MongoDBSaver):
    def __init__(self, *args, **kwargs):
        # Call parent constructor
        super().__init__(*args, **kwargs)
        # Store reference to database and collection names for our custom methods
        self.db_name = kwargs.get('db_name', 'checkpointing_db') 
        self.checkpoint_collection_name = kwargs.get('checkpoint_collection_name', 'checkpoints')
    
    def put(self, config, checkpoint, metadata, new_versions):
        """Override put method to add timestamp fields"""
        # Call parent's put method first
        result = super().put(config, checkpoint, metadata, new_versions)
        
        # After saving, add timestamp directly to the document
        try:
            # Access the collections using stored names
            db = self.client[self.db_name]
            checkpoint_collection = db[self.checkpoint_collection_name]
            
            # Add created_at field to the document
            thread_id = config.get("configurable", {}).get("thread_id")
            
            # Extract checkpoint ID (checkpoint is a dict with 'id' key)
            checkpoint_id = None
            if isinstance(checkpoint, dict) and 'id' in checkpoint:
                checkpoint_id = checkpoint['id']
            elif hasattr(checkpoint, 'id'):
                checkpoint_id = checkpoint.id
            else:
                # Skip timestamp addition if we can't extract ID
                logger.debug(f"Could not extract checkpoint ID from {type(checkpoint)}")
                return result
            
            if thread_id and checkpoint_id:
                # Update the document with timestamp
                update_result = checkpoint_collection.update_one(
                    {
                        "thread_id": thread_id,
                        "checkpoint_id": checkpoint_id
                    },
                    {
                        "$set": {
                            "created_at": datetime.now(),
                            "updated_at": datetime.now()
                        }
                    }
                )
                if update_result.modified_count > 0:
                    logger.debug(f"Added timestamp to checkpoint {checkpoint_id}")
                    
        except Exception as e:
            logger.warning(f"Failed to add timestamp to checkpoint: {e}")
        
        return result

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

The costevalutionforllm collection stores the cost and usage details for individual LLM (Large Language Model) API calls. Each document within this collection represents a single call and contains metrics like token usage and cost.

Schema Details
1. _id (ObjectId)
	• Description: A unique identifier automatically generated by MongoDB for each document.
	• Type: ObjectId
	• Required: Yes
2. batchId (String)
	• Description: A reference to a batch of uploaded files. This field links a specific LLM call to its larger processing context. It corresponds to the _id in a separate batches collection.
	• Type: String
	• Required: Yes
	• Index: Recommended for efficient lookups and grouping.
3. fileId (String)
	• Description: A reference to a specific file within a batch. This field links the LLM call to the original file it was processing. It corresponds to the _id in a separate files collection.
	• Type: String
	• Required: Yes
	• Index: Recommended for efficient lookups and grouping.
4. promptId (String)
	• Description: A reference to the specific prompt used for the LLM call. This field helps in analyzing the cost and performance of different prompts. It corresponds to the _id in a separate prompts collection.
	• Type: String
	• Required: Yes
	• Index: Recommended for efficient lookups and grouping.
5. inputTokens (Number)
	• Description: The number of tokens in the input prompt sent to the LLM.
	• Type: Number (Integer)
	• Required: Yes
6. outputTokens (Number)
	• Description: The number of tokens in the response generated by the LLM.
	• Type: Number (Integer)
	• Required: Yes
7. totalTokens (Number)
	• Description: The sum of inputTokens and outputTokens, representing the total tokens consumed by a single LLM call.
	• Type: Number (Integer)
	• Required: Yes
8. totalCostInUSD (Number)
	• Description: The monetary cost of the LLM call, calculated in US dollars. This is a crucial metric for cost analysis.
	• Type: Number (Double or Decimal)
	• Required: Yes


LLM Pricing Schema -- llmpricing collection

The llmpricing collection stores the pricing and rate information for various Large Language Models (LLMs). Each document within this collection represents a specific model variant and its associated costs for token processing. This collection is used as a reference to calculate the total cost of LLM API calls.

Schema Details
	1. _id (ObjectId)
		○ Description: A unique identifier automatically generated by MongoDB for each document.
		○ Type: ObjectId
		○ Required: Yes
	2. modelVariant (String)
		○ Description: The specific name or version of the Large Language Model, such as 'Google Gemini 1.5 flash' or 'OpenAI GPT-4o mini'. This field acts as the primary key for looking up a model's pricing.
		○ Type: String
		○ Required: Yes
		○ Index: Recommended. A unique index should be created on this field to ensure fast, unique lookups and prevent duplicate entries for the same model.
	3. ratePerMillionInputTokens (Number)
		○ Description: The cost in US dollars (USD) for processing one million input tokens for the given modelVariant.
		○ Type: Number (Double or Decimal128 for high precision financial calculations)
		○ Required: Yes
	4. ratePerMillionOutputTokens (Number)
		○ Description: The cost in US dollars (USD) for generating one million output tokens from the given modelVariant.
		○ Type: Number (Double or Decimal128 for high precision financial calculations)
		○ Required: Yes


Users Schema

The "users" collection is a core component of the application, managing user identities, authentication credentials, and profile information. Each document represents a single user, storing details necessary for account access, personalization, and tracking user activity across the platform. This collection serves as the master source for user data and is referenced by numerous other collections to attribute actions and manage permissions.

Schema Details

	1. googleId (String, Float)
	Description: The unique identifier provided by Google for users who authenticate via Google OAuth. This field will be null or NaN for users who sign up directly.
	Type: String or Float (due to NaN values from data import)
	Required: No

	2. authSource (String)
Description: The authentication method used by the user. Common values are "google" for Google OAuth or "self" for email/password-based registration.
Type: String
Required: Yes

	3. createdAt (Date, String)
Description: The date and time when the user account was created.
Type: Date or String (due to data format inconsistencies, e.g., "2025-07-16T12:03:42.196Z").
Required: Yes

	4. emailId (String)
Description: The user's unique email address, which serves as a primary login identifier.
Type: String
Required: Yes
Index: A unique index is enforced on this field to prevent duplicate accounts.
	
	5. firstName (String)
Description: The first name of the user.
Type: String
Required: Yes

	6. lastName (String)
Description: The last name of the user.
Type: String
Required: Yes

	7. role (String)
Description: The role assigned to the user within the application, which determines their permissions (e.g., "admin").
Type: String
Required: Yes
Index: Recommended for efficiently querying users by role.

	8. updatedAt (Date, String)
Description: The date and time when the user's record was last updated.
Type: Date or String (due to data format inconsistencies).
Required: Yes

	9. userId (String)
Description: A unique application-level identifier (UUID) for the user. This is the primary key used to reference the user in other collections.
Type: String
Required: Yes
Index: A unique index is enforced on this field to ensure data integrity and fast lookups.

Relationships from other collections

Referenced By: The users collection is a master data source. Its userId is used as a foreign key in many other collections to track creation and modification actions:
files: createdBy and updatedBy fields.
batches: createdBy and updatedBy fields.
compliances: createdBy and updatedBy fields.
conversations: userId, createdBy, and updatedBy fields.
asks: createdBy and updatedBy fields.
documenttypes: createdBy and updatedBy fields.
documentmappings: createdBy and updatedBy fields.


Document Types Schema
The documenttypes collection functions as a master definition list or a controlled vocabulary for classifying documents within the application. Each entry in this collection defines a specific type of document, such as "Share Purchase Agreement" or "Data Centre Agreement". This provides a standardized set of categories that can be used by the AI for classification and by users for filtering and organization.

Schema Details

	1. documentId (String)
Description: A unique application-level identifier (UUID) for the document type definition itself. This serves as the primary business key for this entry.
Type: String
Required: Yes
Index: Recommended with a unique constraint to prevent duplicate document type definitions.

	2. description (String)
Description: A detailed description of the document type, which may include key identifiers or characteristics of the document category.
Type: String
Required: Yes

	3. documentType (String)
Description: The official, human-readable name of the document category (e.g., "Share Purchase Agreement", "Classification"). This is the label used for classification throughout the application.
Type: String
Required: Yes
Index: Recommended for efficiently searching or grouping by type.

	4. createdBy (String)
Description: A reference to the user who created this document type definition. This field corresponds to the userId in the users collection.
Type: String
Required: Yes

	5. updatedBy (String)
Description: A reference to the user who last modified this document type definition. This field corresponds to the userId in the users collection.
Type: String
Required: Yes

	6. createdAt (String)
Description: The date and time when the document type was created, stored in ISO 8601 string format.
Type: String
Required: Yes
Format: ISO 8601 datetime string (e.g., "2025-07-14T11:50:58.875Z").

	7. updatedAt (String)
Description: The date and time when the document type was last modified, stored in ISO 8601 string format.
Type: String
Required: Yes
Format: ISO 8601 datetime string (e.g., "2025-07-02T17:05:11.029Z").

Relationships
Referenced By: This collection acts as a master data source. 
documentmappings collection have a reference to documentId. This collection stores all the prompts linked to this document type. One document type can be linked to multiple prompts.

While not directly referenced via a foreign key in other collections in the database schema, its documentType values are used by the application logic and are populated in the files. additionalFields.classificationType field after a document is processed.


Document Mappings Schema

The documentmappings collection stores the relationships between document types and the AI processing prompts used to extract information from them. Each document within this collection represents a mapping configuration that defines which prompts should be applied to specific documents during processing. This collection is essential for controlling and tracking the document analysis workflow.
Schema Details

sysId (String)
Description: A unique system identifier for the document mapping entry. This serves as the primary business key for referencing this specific mapping configuration across the system.
Type: String
Required: Yes
Index: Recommended with unique constraint for efficient lookups and preventing duplicate mappings.

documentId (String)
Description: A reference to the specific document that this mapping applies to. This identifier links the mapping to a document in the document management system and determines which document will be processed with the associated prompts.
Type: String
Required: Yes
Index: Recommended for efficient document-based queries and analysis.

promptIds (Array)
Description: An array of prompt identifiers that should be applied to the specified document. Each prompt ID corresponds to a promptId in the prompts collection. This field can contain multiple prompts (e.g., 87 prompts) allowing comprehensive document analysis with various extraction rules.
Type: Array
Required: Yes
References: prompts.promptId
Index: Recommended for prompt-based analysis and optimization.

createdBy (String)
Description: A reference to the user who created this document mapping configuration. This field corresponds to the userId in the users collection and enables tracking of who configured specific document processing workflows.
Type: String
Required: Yes
Index: Recommended for user-based analysis and audit trails.

updatedBy (String)
Description: A reference to the user who last modified this document mapping. This field corresponds to the userId in the users collection and tracks configuration change history.
Type: String
Required: Yes
Index: Recommended for audit trail and modification tracking.

createdAt (String)
Description: The date and time when the document mapping was created, stored in ISO 8601 string format. This enables temporal analysis of mapping configuration history.
Type: String
Required: Yes
Format: ISO 8601 datetime string (e.g., "2025-07-03T11:00:54.712Z")

updatedAt (String)
Description: The date and time when the document mapping was last modified, stored in ISO 8601 string format. This tracks the most recent changes to the mapping configuration.
Type: String
Required: Yes
Format: ISO 8601 datetime string (e.g., "2025-07-03T11:00:54.712Z")

Additional Index Information
Composite Index (documentId_promptId): A compound index on documentId and promptId fields is recommended with unique constraint to prevent duplicate prompt assignments to the same document, ensuring data integrity in the mapping configuration.

Batches Schema

The batches collection stores information about batch processing jobs for document workflows. Each document within this collection represents a single batch operation that processes multiple files together, containing timing information, status tracking, and user attribution. This collection is used for monitoring processing performance and analyzing workflow efficiency.
Schema Details

batchId (String)

Description: A unique identifier for the batch processing job, used to reference this batch across other collections. This field serves as the primary business key for batch operations.
Type: String
Required: Yes
Index: Recommended with unique constraint for efficient lookups and preventing duplicates.


batchName (String)

Description: A human-readable name for the batch, such as "MSA 3", "NDA test", or "Compliance Review Batch". This helps users identify and distinguish between different batch operations.
Type: String
Required: Yes


description (String)

Description: A detailed description of the batch processing job, explaining its purpose or containing additional context. This field may be empty for simple batch operations.
Type: String
Required: Yes


files (Array)

Description: An array of file objects contained within this batch. Each file object includes fileId, fileName, and processing time information (totalTimeTakenByFile). This represents all documents processed together in this batch.
Type: Array
Required: Yes
Structure: Array of objects with fileId, fileName, and timing data


status (String)

Description: The current processing status of the batch. Common values include "Processed", "Processing", "Failed", and "Queued". This field tracks the batch through its lifecycle.
Type: String
Required: Yes
Index: Recommended for filtering batches by processing status.


createdBy (String)

Description: A reference to the user who initiated this batch processing job. This field corresponds to the userId in the users collection and enables user-based analytics.
Type: String
Required: Yes
Index: Recommended for user-based analysis and activity tracking.

updatedBy (String)

Description: A reference to the user who last modified this batch record. This field corresponds to the userId in the users collection and tracks batch modification history.
Type: String
Required: Yes
Index: Recommended for audit trail and user activity analysis.


createdAt (Date)

Description: The date and time when the batch was created and initiated, stored in ISO 8601 format. This enables time-based analysis and processing trends.
Type: Date
Required: Yes
Index: Recommended for temporal analysis and date range queries.


updatedAt (Date)

Description: The date and time when the batch record was last modified, stored in ISO 8601 format. This tracks the most recent changes to the batch status or information.
Type: Date
Required: Yes
Index: Recommended for tracking recent activity and modifications.


totalTimeTakenByBatch (String)

Description: The total processing time for the entire batch, typically formatted as a string with units (e.g., "247.35 secs", "7934.26 secs"). This metric is crucial for performance analysis and optimization.
Type: String
Required: No
Format: Numeric value followed by time unit (usually "secs")

Relationship: 

fileId is primary key in the files collection. 

Files Schema

The files collection serves as the central metadata repository for every document uploaded to the system. Each document within this collection represents a single file, containing essential information such as its storage location, processing status, and the results of various AI analyses which are stored in a nested additionalFields object. This collection is fundamental for tracking a document throughout its entire lifecycle.

Schema Details

fileId (String)
Description: A unique business identifier for the file. This ID is used across the application to reference this specific file in other collections.
Type: String
Required: Yes
Index: Recommended with a unique constraint to ensure fast, unique lookups and data integrity.

fileName (String)
Description: The original name of the uploaded file, including its extension (e.g., "POC_TEST_LicenseAgreement.pdf").
Type: String
Required: Yes

container (String)
Description: The name of the storage container where the file is located, typically in a cloud storage service.
Type: String
Required: Yes

blobName (String)
Description: The full path or name of the file within the storage container, often including the fileId to ensure uniqueness.
Type: String
Required: Yes

url (String)
Description: A direct, often temporary or signed, URL to access the file in its storage location.
Type: String
Required: Yes

size (Number)
Description: The size of the file in bytes.
Type: Number (Integer)
Required: Yes

status (String)
Description: The current processing status of the file within the workflow (e.g., "Created", "Processed", "Failed").
Type: String
Required: Yes
Index: Recommended for efficiently querying files based on their processing state.

createdBy (String)
Description: A reference to the user who uploaded the file. This field corresponds to the userId in the users collection.
Type: String
Required: Yes
Index: Recommended for tracking user activity.

updatedBy (String)
Description: A reference to the user who last modified the file's record. This corresponds to the userId in the users collection.
Type: String
Required: Yes

createdAt (Date)
Description: The date and time when the file record was created, stored in ISO 8601 format.
Type: Date
Required: Yes
Index: Recommended for temporal analysis.

updatedAt (Date)
Description: The date and time when the file record was last updated, stored in ISO 8601 format.
Type: Date
Required: Yes

additionalFields (Object)
Description: A nested object containing key-value pairs of data generated from various AI processing pipelines, such as classification, compliance checks, and cost analysis.
Type: Object
Required: Yes
Sub-Fields:
complianceOutput (String): A JSON string detailing results from compliance checks (e.g., GDPR, DORA).
classificationType (String): The document type as determined by the classification agent (e.g., "Master Services Agreement").
confidenceScore (Number): The confidence score of the classification.
language (String): The detected language of the document.
output (String): A JSON string containing the primary extraction results.
totalCostPerFileIdInUSD (String): The total calculated cost for processing this file.
totalInputTokens / totalOutputTokens (String): The total tokens consumed during the file's processing.

Relationships
Referenced By: The files collection is a central hub and its fileId is referenced by:
documentextractions: To link each extraction back to its source file.
obligationmappings: To map extracted obligations to their source file.
costevalutionforllm: To attribute processing costs to a specific file.
batches: The files array within a batch document contains references to the files processed in that batch.

Prompts Schema

The prompts collection serves as a centralized library for all AI instructions used in the document analysis platform. Each document within this collection represents a specific, reusable prompt designed to perform a particular task, such as extracting an attribute or identifying a clause. This collection is critical for standardizing and managing the AI's behavior across different document processing workflows.

Schema Details

promptId (String)
Description: A unique business identifier for the prompt. This ID is used as a foreign key by other collections, like documentmappings, to reference a specific set of instructions.
Type: String
Required: Yes
Index: Recommended with a unique constraint to ensure fast, unique lookups and prevent duplicate prompts.

promptName (String)
Description: A human-readable name for the prompt that summarizes its purpose, such as "Disclosing Party Jurisdiction of Incorporation " or "Perpetual Confidentiality ".
Type: String
Required: Yes
Index: Recommended for easy searching and management of prompts.

description (String)
Description: A brief explanation of what the prompt is designed to achieve or what question it answers (e.g., "Under which jurisdiction's laws is the disclosing party incorporated or organised?").
Type: String
Required: Yes

promptType (String)
Description: The category of the prompt, which dictates the nature of the extraction task. Common values are "Attribute" or "Clause".
Type: String
Required: Yes
Index: Recommended for grouping and filtering prompts by their function.

promptText (String)
Description: The detailed and complete text of the instructions provided to the Large Language Model. This text includes the objective, output format rules, definitions, keywords, and specific scenarios to guide the AI in performing its task accurately.
Type: String
Required: Yes

createdBy (String)
Description: A reference to the user or system that created the prompt.
Type: String
Required: Yes

updatedBy (String)
Description: A reference to the user or system that last modified the prompt.
Type: String
Required: Yes

createdAt (String)
Description: The date and time when the prompt was created, stored in ISO 8601 string format.
Type: String
Required: Yes
Format: ISO 8601 datetime string (e.g., "2025-06-13T17:10:05.008Z").

updatedAt (String)
Description: The date and time when the prompt was last modified, stored in ISO 8601 string format.
Type: String
Required: Yes
Format: ISO 8601 datetime string (e.g., "2025-06-13T17:10:05.008Z").

promptVariables (Float)
Description: A field intended to hold dynamic variables for the prompt. Based on the sample data, this field is not currently in use (value is NaN).
Type: Float
Required: Yes

isSeeded (Boolean)
Description: A flag to indicate if the prompt is part of the initial "seeded" data set provided with the application.
Type: Boolean
Required: Yes

Relationships
Referenced By: The prompts collection is a master data source and its promptId is referenced by:
documentmappings: The promptIds array directly links a set of prompts to a specific document for processing.
costevalutionforllm: To link the cost of an AI call back to the specific prompt that was executed.

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

    # NEW: This validator will automatically convert a JSON string into a dictionary.
    @field_validator('contextual_query', mode='before')
    @classmethod
    def parse_query_from_string(cls, value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("contextual_query is a string but not valid JSON")
        return value

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
    chartType: Literal["bar", "line", "pie", "doughnut", "radar", "scatter"]
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
# ENHANCED OUTPUT FORMAT DETECTION
# ==============================================================================

def identify_output_format(state: AgentState):
    """Enhanced output format detection using detailed analysis"""
    logger.info("🎯 Identifying optimal output format...")
    question = state["question"].content
    
    system_message = SystemMessage(
        content="""You are an expert analytical AI. Your primary function is to analyze a user's query and determine the most effective and appropriate format(s) for the response. Your goal is to select from a predefined list of formats to ensure the user's request is answered clearly and efficiently.

        List of Allowed Output Formats:

        Table
        Bar Chart
        Plain Text
        Pie Chart
        Line Chart
        Doughnut Chart
        Radar Chart
        Scatter Chart
        HTML

        Core Instructions and Decision-Making Rules
        You must analyze the user's query and select one or more formats from the list above. Follow these rules precisely:

        Rule 1: The "Max One Chart" Rule
        You can select a maximum of ONE chart type for any given response.
        If multiple chart types could potentially work, choose the single most suitable one based on the nature of the data and the user's question. For example, for showing change over time, a Line Chart is superior to a Bar Chart.

        Rule 2: Combining Formats
        Plain Text is the default and can be combined with any other format. It should be used for explanations, summaries, definitions, or any non-structured textual response.
        A Table can be combined with Plain Text and/or one chart.
        HTML is a specialized format and should generally be selected only when the user explicitly asks for HTML code or a complex, formatted layout like a resume or web page structure. It can be combined with Plain Text for explanation.

        Guidelines for Selecting Each Format
        Use the following logic to decide which format(s) to choose. Analyze the query for keywords and intent.

        Non-Chart Formats
        Plain Text:
        When to use: For any question requiring a textual explanation, definition, list, code snippet, or conversational answer. This is the base format for most queries.
        Keywords: "what is," "explain," "how to," "list," "describe," "generate code for..."

        Table:
        When to use: For presenting structured data with clear rows and columns. Ideal for direct comparisons of features across multiple items.
        Keywords: "compare," "list features," "data for," "vs," "side-by-side."

        HTML:
        When to use: When the user explicitly requests HTML code or a web-based layout.
        Keywords: "HTML for," "create a webpage," "HTML boilerplate," "format a resume in HTML."

        Chart Formats (Remember: Max One!)
        Bar Chart:
        When to use: To compare distinct quantities across different categories. Excellent for ranking.
        Keywords: "compare," "rank," "which has more/less," "vs" (for quantities).
        Example: "Compare the box office revenue of the top 5 highest-grossing movies."

        Line Chart:
        When to use: To show a trend or changes in data over a continuous interval, most commonly time.
        Keywords: "trend," "over time," "growth," "change," "fluctuation," "history."
        Example: "Show the population growth of India from 1950 to today."

        Pie Chart / Doughnut Chart:
        When to use: To show the proportions or percentage breakdown of a whole. Use when parts sum to 100%. (These two are often interchangeable; select one).
        Keywords: "proportion," "percentage," "share," "breakdown," "composition."
        Example: "What is the market share of different operating systems?"

        Scatter Chart:
        When to use: To display the relationship or correlation between two different numerical variables.
        Keywords: "relationship between," "correlation," "does X affect Y," "distribution."
        Example: "Is there a relationship between hours of sleep and test scores?"

        Radar Chart:
        When to use: To compare multiple quantitative variables for one or more subjects. Ideal for showing strengths and weaknesses or performance profiles.
        Keywords: "compare features of," "performance analysis," "strengths and weaknesses."
        Example: "Compare two smartphones across battery life, camera quality, screen brightness, and price."

        Output Requirement
        Your final output must be a comma-separated list of the chosen format(s). Do not add any explanation.

        Examples
        User Query: "What is the capital of France?"
        Your Output: Plain Text

        User Query: "Compare the population, GDP, and area of the USA, China, and India."
        Your Output: Plain Text, Table, Bar Chart

        User Query: "Show me the trend of Google's stock price over the last 5 years."
        Your Output: Plain Text, Line Chart

        User Query: "Give me the HTML code for a basic contact form."
        Your Output: Plain Text, HTML

        User Query: "Show the percentage breakdown of Earth's atmosphere by gas."
        Your Output: Plain Text, Pie Chart"""
    )

    human_message = HumanMessage(content=f"\n\nOriginal question: {question}")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GEMINI_API_KEY)
    
    try:
        response = llm.invoke([system_message, human_message])
        output_format = response.content.strip()
        logger.info(f"🎯 Enhanced format detected: {output_format}")
        state["output_format"] = output_format
        return state
    except Exception as e:
        logger.warning(f"⚠️ Enhanced format detection failed, defaulting to Plain Text. Error: {e}")
        state["output_format"] = "Plain Text"
        return state

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
    
    # Normalize classification to match routing expectations
    classification = result.classification.replace("-", "_")  # Convert "off-topic" to "off_topic"
    state["on_topic"] = classification
    logger.info(f"🎯 AI Classification: {state['on_topic']}")
    return state

def internal_search_node(state: AgentState, retriever, rag_chain, llm, mongodb_executor):
    logger.info("📖 Internal Search: Generating and executing MongoDB query...")
    MAX_RETRIES = 3
    state["retry_count"] = 0
    
    question = state["question"].content
    chat_history = format_chat_history(state["messages"])
    
    logger.info("... 1/4: Detecting output format")
    state = identify_output_format(state)

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
                    summary_prompt = f"Summarize the following JSON results in a friendly, natural language response. Results:\n{json.dumps(result['results'][:5], indent=2, default=json_converter)}"
                    final_answer = llm.invoke(summary_prompt).content
                else: # Default to JSON
                    final_answer = f"```json\n{json.dumps(result['results'], indent=2, default=json_converter)}\n```"

                enhanced_response = f"""🎯 **Query Executed Successfully!**
**Collection:** `{result['collection']}`
**Records Found:** {result['count']}

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

    # Use a more powerful model for follow-up generation
    follow_up_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, google_api_key=GEMINI_API_KEY)
    suggestion_llm = follow_up_llm.with_structured_output(FollowUpSuggestion)
    
    # We'll give the LLM two attempts to get it right.
    for attempt in range(2):
        logger.info(f"... Follow-up generation attempt {attempt + 1}/2")
        
        # Construct the prompt for this attempt
        prompt = f"""You are a data analysis assistant. Generate a follow-up suggestion based on user data.

**CRITICAL REQUIREMENTS:**
- You MUST provide BOTH follow_up_question AND contextual_query together, or set follow_up_question to "NONE"
- The contextual_query must be a complete, valid MongoDB aggregation pipeline
- Use "<user_input_needed>" placeholder when user input is required

**User's Question:** "{state['question'].content}"

**Raw Data Analysis:**
{json.dumps(state.get("raw_data", [])[:3], indent=2, default=json_converter)}

**VALID EXAMPLES:**

Option 1 - Filter suggestion:
{{
  "follow_up_question": "Would you like to filter by a specific role?",
  "contextual_query": {{
    "collection": "users",
    "pipeline": [
      {{"$match": {{"role": "<user_input_needed>"}}}},
      {{"$project": {{"firstName": 1, "lastName": 1, "emailId": 1, "role": 1}}}}
    ]
  }}
}}

Option 2 - Count/Group suggestion:
{{
  "follow_up_question": "Would you like to see a count by role?",
  "contextual_query": {{
    "collection": "users",
    "pipeline": [
      {{"$group": {{"_id": "$role", "count": {{"$sum": 1}}}}}},
      {{"$sort": {{"count": -1}}}}
    ]
  }}
}}

Option 3 - No good follow-up:
{{
  "follow_up_question": "NONE",
  "contextual_query": null
}}

**ANALYSIS STEPS:**
1. Look at the raw data structure and fields available
2. For users data: suggest role filtering, authSource filtering, or role counting
3. For files data: suggest status filtering or type grouping  
4. For costs data: suggest batch grouping or time filtering
5. Create BOTH a question AND complete query, or set to "NONE"

Generate your response following exactly one of the example formats above:"""
        
        try:
            suggestion = suggestion_llm.invoke(prompt)
            # The Pydantic validator will raise an error if the output is invalid,
            # which will be caught by the except block. If we get here, it's valid.
            if suggestion.follow_up_question and suggestion.follow_up_question.upper() != "NONE":
                logger.info(f"✅ Follow-up generated successfully on attempt {attempt + 1}.")
                state["follow_up_question"] = suggestion.follow_up_question
                
                # Handle case where contextual_query might be a string instead of dict
                contextual_query = suggestion.contextual_query
                if isinstance(contextual_query, str):
                    try:
                        contextual_query = json.loads(contextual_query)
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Failed to parse contextual_query as JSON: {contextual_query}")
                        raise ValueError("Invalid contextual_query format")
                
                state["follow_up_context"] = contextual_query
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

        query_str = json.dumps(query_template, default=json_converter)

        if "<user_input_needed>" in query_str:
            logger.info("... Query requires user input. Extracting...")
            
            extraction_prompt = f"""From the user's response, extract the specific value they want to use for filtering.
            User's response: "{user_response}"
            Example: If the response is "yes, admin role", the value is "admin".
            Example: If the response is "sure, show me the legal ones", the value is "legal".
            Extracted value:"""
            
            extracted_value = llm.invoke(extraction_prompt).content.strip().lower().replace("'", "").replace('"', '')
            logger.info(f"... Extracted value: '{extracted_value}'")
            
            # Implement fuzzy matching for role values
            if "role" in query_str:
                available_roles = ["admin", "user", "legal"]  # Known roles from the database
                matched_role = fuzzy_match_role(extracted_value, available_roles)
                if matched_role:
                    logger.info(f"... Fuzzy matched '{extracted_value}' to '{matched_role}'")
                    extracted_value = matched_role
                else:
                    logger.info(f"... No fuzzy match found for '{extracted_value}', using as-is")

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

    # Note: Using raw LLM approach as it's more reliable than structured output
    
    # Prepare the prompt with better structure and clearer instructions
    prompt = f"""You are a data visualization expert. Your job is to convert raw data into a structured JSON format for a UI.

CRITICAL: You MUST return ONLY a valid JSON object starting with {{ and ending with }}.

EXAMPLE OUTPUT FORMAT:
{{
    "components": [
        {{
            "type": "table",
            "headers": ["promptId", "promptName", "description"],
            "rows": [
                ["448f0675", "Term And Duration", "Term And Duration"],
                ["04d5e6f7", "End User Definition", "How are end users defined"]
            ]
        }},
        {{
            "type": "text",
            "content": "Follow-up question text goes here"
        }}
    ]
}}

USER'S QUESTION: "{state['question'].content}"

RAW DATA: {json.dumps(raw_data if isinstance(raw_data, list) else raw_data, indent=2, default=json_converter)}

Follow-up question: "{state.get('follow_up_question', 'NONE')}"

INSTRUCTIONS:
1. For list data like prompts/users: Use "table" component with appropriate headers and rows
2. For aggregate data: Use "chart" component 
3. Always include follow-up as separate "text" component with "content" field if not "NONE"
4. CRITICAL: Text components MUST use "content" field, NOT "text" field
5. Return ONLY valid JSON - no explanations, no markdown, no extra text

JSON OUTPUT:"""
    
    try:
        # Use raw LLM with improved parsing (more reliable than structured output)
        raw_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0, google_api_key=GEMINI_API_KEY)
        raw_response_content = raw_llm.invoke(prompt).content
        
        logger.info(f"... Raw LLM Response for formatting:\n{raw_response_content}")
        logger.info(f"... Response type: {type(raw_response_content)}")

        # Multiple JSON extraction strategies
        json_str = None
        
        # Strategy 1: Look for complete JSON object
        json_start = raw_response_content.find('{')
        json_end = raw_response_content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = raw_response_content[json_start:json_end]
        
        # Strategy 2: Try to extract from code blocks
        if not json_str:
            import re
            json_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response_content, re.DOTALL)
            if json_blocks:
                json_str = json_blocks[0]
        
        # Strategy 3: Use the whole response if it looks like JSON
        if not json_str and raw_response_content.strip().startswith('{'):
            json_str = raw_response_content.strip()
        
        if json_str:
            try:
                parsed_json = json.loads(json_str)
                
                # Fix common field name issues
                if "components" in parsed_json:
                    for component in parsed_json["components"]:
                        if component.get("type") == "text" and "text" in component and "content" not in component:
                            component["content"] = component.pop("text")
                
                validated_response = StructuredResponse.model_validate(parsed_json)
                final_components = [component.model_dump() for component in validated_response.components]
                state["structured_output"] = final_components
                logger.info("... ✅ Successfully parsed and validated JSON from raw response.")
                return state
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"⚠️ JSON parsing failed: {e}")
        
        # Final fallback: Create simple table from raw data
        if isinstance(raw_data, list) and raw_data:
            if isinstance(raw_data[0], dict):
                headers = list(raw_data[0].keys())
                rows = [[str(item.get(h, '')) for h in headers] for item in raw_data]
                fallback_components = [{
                    "type": "table",
                    "headers": headers,
                    "rows": rows
                }]
                if state.get('follow_up_question', 'NONE') != 'NONE':
                    fallback_components.append({
                        "type": "text",
                        "content": f"Follow-up: {state['follow_up_question']}"
                    })
                state["structured_output"] = fallback_components
                logger.info("... ✅ Used fallback table generation.")
                return state

        # Last resort: plain text
        state["structured_output"] = [{"type": "text", "content": raw_response_content}]
        logger.warning("... ⚠️ Using plain text fallback.")

    except Exception as e:
        logger.error(f"❌ Critical failure in format_output_node: {e}", exc_info=True)
        state["structured_output"] = [{"type": "text", "content": f"Error formatting output: {str(e)}"}]
    
    return state

def json_converter(o):
    if isinstance(o, (datetime, ObjectId)):
        return str(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def fuzzy_match_role(user_input: str, available_roles: List[str]) -> Optional[str]:
    """
    Fuzzy match user input to available roles in the database.
    Handles common variations like plural forms, synonyms, etc.
    """
    user_input = user_input.lower().strip()
    
    # Direct match first
    if user_input in available_roles:
        return user_input
    
    # Handle common variations
    role_mappings = {
        "admins": "admin",
        "administrators": "admin", 
        "administrator": "admin",
        "admin user": "admin",
        "admin role": "admin",
        "users": "user",
        "regular user": "user",
        "user role": "user",
        "legal team": "legal",
        "legal role": "legal",
        "lawyer": "legal",
        "attorneys": "legal",
        "attorney": "legal"
    }
    
    # Check mappings
    if user_input in role_mappings:
        return role_mappings[user_input]
    
    # Check if user input contains any role name
    for role in available_roles:
        if role in user_input or user_input in role:
            return role
    
    # Try to find partial matches (substring matching)
    for role in available_roles:
        if user_input.startswith(role[:3]) or role.startswith(user_input[:3]):
            return role
    
    return None

# ==============================================================================
# CONVERSATION MANAGEMENT
# ==============================================================================

def list_previous_conversations(mongo_client):
    """List all previous conversation threads with rich context"""
    try:
        db = mongo_client["genaiexeco-development"]
        checkpoints = db["langgraph_checkpoints"]
        
        # Get unique thread_ids with their latest checkpoint info
        pipeline = [
            {"$sort": {"updated_at": -1}},
            {"$group": {
                "_id": "$thread_id",
                "latest_update": {"$first": "$updated_at"},
                "created_at": {"$first": "$created_at"},
                "checkpoint_count": {"$sum": 1},
                "latest_checkpoint": {"$first": "$$ROOT"}
            }},
            {"$sort": {"latest_update": -1}},
            {"$limit": 10}  # Show last 10 conversations
        ]
        
        conversations = list(checkpoints.aggregate(pipeline))
        
        # Enhance each conversation with context
        enhanced_conversations = []
        for i, conv in enumerate(conversations, 1):
            thread_id = conv['_id']
            
            # Extract context from latest checkpoint
            latest_checkpoint = conv.get('latest_checkpoint', {})
            checkpoint_binary = latest_checkpoint.get("checkpoint")
            
            conv_data = {
                'number': i,
                'thread_id': thread_id,
                'thread_id_short': thread_id[:8],
                'latest_update': conv.get('latest_update', 'Unknown'),
                'message_count': conv.get('checkpoint_count', 0),
                'title': 'Untitled Conversation',
                'first_message': None,
                'last_message': None,
                'last_response_type': None
            }
            
            # Decode checkpoint to get conversation context
            if checkpoint_binary:
                try:
                    # Handle both Binary objects and raw bytes
                    if hasattr(checkpoint_binary, 'binary'):
                        data_to_decode = checkpoint_binary.binary
                    else:
                        data_to_decode = checkpoint_binary
                        
                    checkpoint_data = msgpack.unpackb(data_to_decode, raw=False)
                    channel_values = checkpoint_data.get("channel_values", {})
                    messages = channel_values.get("messages", [])
                    structured_output = channel_values.get("structured_output", [])
                    
                    if messages:
                        # Extract human messages - handle both dict and ExtType formats
                        first_human_content = None
                        last_ai_content = None
                        
                        for msg in messages:
                            try:
                                if isinstance(msg, dict):
                                    # Simple dict format
                                    if msg.get("type") == "human" and not first_human_content:
                                        first_human_content = msg.get("content", "")
                                    elif msg.get("type") == "ai":
                                        last_ai_content = msg.get("content", "")
                                elif hasattr(msg, 'data') and hasattr(msg, 'code'):
                                    # msgpack ExtType - try to parse the data
                                    try:
                                        # The ExtType contains serialized LangChain message data
                                        # For now, we'll extract from the raw bytes if possible
                                        data_str = msg.data.decode('utf-8', errors='ignore')
                                        if 'human' in data_str.lower() and not first_human_content:
                                            # Look for content pattern in the serialized data
                                            import re
                                            # Try multiple patterns to extract clean content
                                            patterns = [
                                                r'content\x94([a-zA-Z][a-zA-Z0-9\s,.\?!\'"-]+?)[\x80\x94\xa0]',  # Pattern with msgpack delimiters
                                                r'content\x94([a-zA-Z][a-zA-Z0-9\s,.\?!\'"-]+)',  # Simpler pattern
                                                r'content.([a-zA-Z][a-zA-Z0-9\s,.\?!\'"-]{5,50})'  # Fallback pattern
                                            ]
                                            
                                            for pattern in patterns:
                                                try:
                                                    content_match = re.search(pattern, data_str)
                                                    if content_match:
                                                        extracted_content = content_match.group(1).strip()
                                                        # Additional cleaning - remove common msgpack artifacts
                                                        clean_content = re.sub(r'(additional.*|_kwargs.*|metadata.*)', '', extracted_content, flags=re.IGNORECASE).strip()
                                                        if clean_content and len(clean_content) > 3 and not clean_content.startswith('_'):
                                                            first_human_content = clean_content
                                                            break
                                                except:
                                                    continue
                                        elif 'ai' in data_str.lower():
                                            # Similar extraction for AI messages
                                            content_match = re.search(r'content.([^\\]+)', data_str)
                                            if content_match:
                                                extracted = content_match.group(1)
                                                # Take first reasonable portion
                                                if len(extracted) > 10:
                                                    last_ai_content = extracted[:200]
                                    except:
                                        pass  # Skip if can't parse
                                elif hasattr(msg, 'content') and hasattr(msg, 'type'):
                                    # Direct LangChain message object
                                    if msg.type == "human" and not first_human_content:
                                        first_human_content = msg.content
                                    elif msg.type == "ai":
                                        last_ai_content = msg.content
                            except Exception as e:
                                logger.debug(f"Error parsing message: {e}")
                                continue
                        
                        if first_human_content:
                            conv_data['first_message'] = first_human_content[:60] + "..." if len(first_human_content) > 60 else first_human_content
                            # Generate friendly title from first message
                            title_words = first_human_content.split()[:5]  # First 5 words
                            conv_data['title'] = " ".join(title_words).title()
                        
                        if last_ai_content:
                            conv_data['last_message'] = last_ai_content[:60] + "..." if len(last_ai_content) > 60 else last_ai_content
                    
                    # Check what type of output was generated
                    if structured_output:
                        output_types = [comp.get('type', 'unknown') for comp in structured_output if isinstance(comp, dict)]
                        unique_types = list(set(output_types))
                        if unique_types:
                            conv_data['last_response_type'] = ", ".join(unique_types)
                    
                except Exception as e:
                    logger.debug(f"Could not decode checkpoint context for {thread_id}: {e}")
            
            enhanced_conversations.append(conv_data)
        
        return enhanced_conversations
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return []

def get_conversation_summary(thread_id: str, mongo_client):
    """Get a summary of a conversation including first and last messages"""
    try:
        db = mongo_client["genaiexeco-development"]
        checkpoints = db["langgraph_checkpoints"]
        
        # Get the latest checkpoint for this thread
        latest_checkpoint = checkpoints.find_one(
            {"thread_id": thread_id},
            sort=[("updated_at", -1)]
        )
        
        if not latest_checkpoint:
            return None
            
        # Extract messages if available - decode msgpack data
        checkpoint_binary = latest_checkpoint.get("checkpoint")
        if checkpoint_binary:
            try:
                # Handle both Binary objects and raw bytes
                if hasattr(checkpoint_binary, 'binary'):
                    data_to_decode = checkpoint_binary.binary
                else:
                    data_to_decode = checkpoint_binary
                    
                checkpoint_data = msgpack.unpackb(data_to_decode, raw=False)
                channel_values = checkpoint_data.get("channel_values", {})
                messages = channel_values.get("messages", [])
            except Exception as e:
                logger.error(f"Error decoding checkpoint data: {e}")
                messages = []
                channel_values = {}
        else:
            messages = []
            channel_values = {}
        
        summary = {
            "thread_id": thread_id,
            "updated_at": latest_checkpoint.get("updated_at"),
            "created_at": latest_checkpoint.get("created_at"),
            "message_count": len(messages),
            "first_message": None,
            "last_message": None,
            "last_response": None
        }
        
        if messages:
            # Find first human message
            human_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("type") == "human"]
            if human_messages:
                summary["first_message"] = human_messages[0].get("content", "")[:100]
            
            # Find last AI message  
            ai_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("type") == "ai"]
            if ai_messages:
                last_ai_content = ai_messages[-1].get("content", "")
                summary["last_message"] = last_ai_content[:100]
        
        # Get structured output if available
        structured_output = channel_values.get("structured_output", [])
        if structured_output:
            summary["last_response"] = structured_output
            
        return summary
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        return None

def get_last_response_from_thread(thread_id: str, mongo_client):
    """Get the last structured response from a specific conversation thread"""
    try:
        db = mongo_client["genaiexeco-development"]
        checkpoints = db["langgraph_checkpoints"]
        
        # Find the latest checkpoint with structured_output
        pipeline = [
            {"$match": {"thread_id": thread_id}},
            {"$sort": {"updated_at": -1}},
            {"$limit": 1}
        ]
        
        latest_checkpoint = list(checkpoints.aggregate(pipeline))
        if not latest_checkpoint:
            return None
            
        # Decode msgpack checkpoint data
        checkpoint_binary = latest_checkpoint[0].get("checkpoint")
        if checkpoint_binary:
            try:
                # Handle both Binary objects and raw bytes
                if hasattr(checkpoint_binary, 'binary'):
                    data_to_decode = checkpoint_binary.binary
                else:
                    data_to_decode = checkpoint_binary
                    
                checkpoint_data = msgpack.unpackb(data_to_decode, raw=False)
                channel_values = checkpoint_data.get("channel_values", {})
                structured_output = channel_values.get("structured_output", [])
                return structured_output if structured_output else None
            except Exception as e:
                logger.error(f"Error decoding checkpoint for last response: {e}")
                return None
        else:
            return None
    except Exception as e:
        logger.error(f"Error getting last response: {e}")
        return None

# ==============================================================================
# WORKFLOW COMPILATION
# ==============================================================================

def create_workflow():
    logger.info("🔧 Building structured output workflow...")
    retriever, llm, rag_chain = setup_ai_components()
    mongodb_executor = MongoDBExecutor()

    mongo_client = pymongo.MongoClient(MONGODB_URI)
    checkpointer = TimestampedMongoDBSaver(
        client=mongo_client,
        db_name="genaiexeco-development",
        checkpoint_collection_name="langgraph_checkpoints"
    )
    logger.info("✅ Using MongoDB for persistent conversation checkpoints.")
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes  
    workflow.add_node("pre_router", pre_router_node)
    workflow.add_node("classify", lambda state: question_classifier(state, llm))
    workflow.add_node("identify_format", identify_output_format)
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
    workflow.add_edge("classify", "identify_format")
    workflow.add_conditional_edges("identify_format", route_from_classification, {"internal": "internal_search", "external": "external_search", "off_topic": "off_topic"})
    
    # CORRECTED WIRING
    workflow.add_edge("internal_search", "generate_follow_up")
    workflow.add_edge("generate_follow_up", "format_output") # The formatter runs AFTER the follow-up is generated
    workflow.add_edge("handle_follow_up", "format_output")   # and after a follow-up is handled
    workflow.add_edge("external_search", "format_output")    # and after an external search
    workflow.add_edge("off_topic", "format_output")      # and after an off-topic response

    # The formatter is the true final step before END
    workflow.add_edge("format_output", END)
    
    #checkpointer = MemorySaver()
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
    mongo_client = pymongo.MongoClient(MONGODB_URI)
    
    # Store conversation list for numeric selection
    cached_conversations = []
    
    print(f"\n[INTERACTIVE] New conversation started (Thread ID: {thread_id}). Type 'quit' to exit.")
    print(f"[MongoDB]: {'Connected' if mongodb_executor.connected else 'Disconnected'}")
    print("\n[COMMANDS]:")
    print("  /list - Show previous conversations")
    print("  /resume <number|thread_id> - Resume a previous conversation")
    print("  /last <number|thread_id> - Get last response from a conversation")
    print("  /help - Show this help message")
    print("=" * 60)

    while True:
        try:
            question = input("\n> Ask a question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question: continue
            
            # Handle special commands
            if question.startswith('/'):
                command_parts = question.split()
                command = command_parts[0].lower()
                
                if command == '/help':
                    print("\n[CONVERSATION COMMANDS]:")
                    print("  /list                    - Show recent conversations with context")
                    print("  /resume <number|id>      - Resume a conversation (e.g., '/resume 1' or '/resume 94cfd818...')")
                    print("  /last <number|id>        - Get the final response from a conversation")
                    print("  /help                    - Show this help message")
                    print("  quit/exit/q              - Exit the program")
                    print("\n[TIPS]:")
                    print("  * Use '/list' first to see numbered conversations")
                    print("  * Use numbers for easier access: '/resume 1' instead of long IDs")
                    print("  * All previous context is preserved when resuming conversations")
                    print("  * You can switch between conversations anytime during a session")
                    continue
                    
                elif command == '/list':
                    print("\n[RECENT CONVERSATIONS]:")
                    conversations = list_previous_conversations(mongo_client)
                    cached_conversations = conversations  # Cache for numeric selection
                    
                    if not conversations:
                        print("No previous conversations found.")
                    else:
                        for conv in conversations:
                            num = conv['number']
                            title = conv['title']
                            short_id = conv['thread_id_short']
                            msg_count = conv['message_count']
                            update_time = conv['latest_update']
                            first_msg = conv['first_message']
                            response_type = conv['last_response_type']
                            
                            print(f"\n  {num}. [{title}]")
                            print(f"     ID: {short_id}... | {msg_count} messages | {update_time}")
                            if first_msg:
                                print(f"     Question: \"{first_msg}\"")
                            if response_type:
                                print(f"     Generated: {response_type}")
                        
                        print(f"\n[TIP] Use '/resume <number>' (e.g., '/resume 1') or '/resume <full_id>' to continue a conversation")
                        print(f"[TIP] Use '/last <number>' to see the final response from any conversation")
                    continue
                    
                elif command == '/resume':
                    if len(command_parts) < 2:
                        print("[Error] Usage: /resume <number|thread_id>")
                        print("Example: /resume 1  or  /resume 94cfd818-b235-4dfa-a259-dc80ccbed1fe")
                        continue
                    
                    resume_input = command_parts[1]
                    resume_thread_id = None
                    
                    # Check if input is a number (referencing cached conversations)
                    if resume_input.isdigit():
                        conv_number = int(resume_input)
                        if cached_conversations and 1 <= conv_number <= len(cached_conversations):
                            resume_thread_id = cached_conversations[conv_number - 1]['thread_id']
                            conv_title = cached_conversations[conv_number - 1]['title']
                        else:
                            print(f"[Error] Invalid conversation number. Use '/list' to see available conversations (1-{len(cached_conversations) if cached_conversations else 0})")
                            continue
                    else:
                        # Direct thread ID
                        resume_thread_id = resume_input
                        conv_title = "Unknown"
                    
                    # Get detailed conversation summary
                    summary = get_conversation_summary(resume_thread_id, mongo_client)
                    if summary:
                        thread_id = resume_thread_id
                        print(f"\n[RESUMED] {conv_title}")
                        print(f"   Thread: {thread_id[:8]}...")
                        print(f"   Messages: {summary['message_count']}")
                        print(f"   Last active: {summary.get('updated_at', 'Unknown')}")
                        
                        if summary.get('first_message'):
                            print(f"   Started with: \"{summary['first_message']}\"")
                        if summary.get('last_message'):
                            print(f"   Last response: \"{summary['last_message'][:80]}...\"")
                        if summary.get('last_response'):
                            response_types = [comp.get('type', 'unknown') for comp in summary['last_response'] if isinstance(comp, dict)]
                            if response_types:
                                print(f"   Last output: {', '.join(set(response_types))}")
                        
                        print(f"\n[OK] You can now continue this conversation. All previous context is maintained.")
                    else:
                        print(f"[Error] Conversation {resume_thread_id} not found.")
                    continue
                    
                elif command == '/last':
                    if len(command_parts) < 2:
                        print("[Error] Usage: /last <number|thread_id>")
                        print("Example: /last 1  or  /last 94cfd818-b235-4dfa-a259-dc80ccbed1fe")
                        continue
                    
                    last_input = command_parts[1]
                    target_thread_id = None
                    conv_title = "Unknown"
                    
                    # Check if input is a number (referencing cached conversations)
                    if last_input.isdigit():
                        conv_number = int(last_input)
                        if cached_conversations and 1 <= conv_number <= len(cached_conversations):
                            target_thread_id = cached_conversations[conv_number - 1]['thread_id']
                            conv_title = cached_conversations[conv_number - 1]['title']
                        else:
                            print(f"[Error] Invalid conversation number. Use '/list' to see available conversations (1-{len(cached_conversations) if cached_conversations else 0})")
                            continue
                    else:
                        # Direct thread ID
                        target_thread_id = last_input
                    
                    last_response = get_last_response_from_thread(target_thread_id, mongo_client)
                    if last_response:
                        print(f"\n[LAST RESPONSE] {conv_title}")
                        print(f"   From: {target_thread_id[:8]}...")
                        
                        # Show response summary before full output
                        response_types = [comp.get('type', 'unknown') for comp in last_response if isinstance(comp, dict)]
                        component_count = len(last_response)
                        print(f"   Contains: {component_count} components ({', '.join(set(response_types))})")
                        
                        print(f"\n[Full Response]:")
                        print(json.dumps(last_response, indent=4, default=json_converter))
                    else:
                        print(f"[Error] No response found for conversation {target_thread_id}")
                    continue
                    
                else:
                    print(f"[Error] Unknown command: {command}. Type /help for available commands.")
                    continue
            
            print("[Processing...]")
            # We no longer pass or receive state here. The checkpointer handles it all.
            result = process_question(graph, question, thread_id)
            
            print(f"\n[Answer]:")
            print(json.dumps(result['answer'], indent=4, default=json_converter))
            print(f"\n[Time]: {result['time']}")
            
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"[Error] in interactive loop: {e}")
            
    print("Goodbye!")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        graph, mongodb_executor = create_workflow()
        interactive_mode(graph, mongodb_executor)
    except Exception as e:
        logger.error(f"❌ Fatal error during initialization: {e}", exc_info=True)