
#Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: source venv/Scripts/activate
Install dependencies
pip install -r requirements.txt


#Configuration

Copy environment template
cp .env.example .env

Edit .env file and add your Google API key:
GEMINI_API_KEY=your_google_api_key_here


#Run the standalone RAG
python working.py

Starting the System

  1. Open Terminal/Command Prompt
  cd "C:\Users\Execo\Desktop\langgraph test"
  python file.py
  2. System Startup
    - You'll see initialization messages
    - A new conversation thread ID is automatically created
    - MongoDB connection status is displayed

  Basic Usage

  Ask Questions

  > Ask a question: show me all users
  > Ask a question: get document count by type
  > Ask a question: find legal documents

  Exit the System

  > Ask a question: quit
  # or
  > Ask a question: exit
  # or
  > Ask a question: q

  Conversation Management Commands

  1. /help - Get Help

  > Ask a question: /help
  Shows all available commands with usage instructions.

  2. /list - View Previous Conversations

  > Ask a question: /list
  Output Example:
  [PREVIOUS CONVERSATIONS]:
    1. e89a8211... (13 messages) - 2025-08-12 16:23:21.519000
       Full ID: e89a8211-e479-484b-8626-fa8778277450
    2. ecab8b83... (20 messages) - 2025-08-12 15:54:41.555000
       Full ID: ecab8b83-a400-4bde-9056-072f8f7d25d1

  3. /resume <thread_id> - Continue Previous Conversation

  > Ask a question: /resume e89a8211-e479-484b-8626-fa8778277450
  What happens:
  - Switches to the specified conversation
  - Shows conversation summary
  - All new questions continue in that thread
  - Maintains full conversation context

  4. /last <thread_id> - Get Last Response

  > Ask a question: /last e89a8211-e479-484b-8626-fa8778277450
  Returns: Complete structured output (tables, charts, text) from that conversation's final response.

  Complete Workflow Examples

  Example 1: Starting Fresh

  # Start system
  python file.py

  # Ask questions in new conversation
  > Ask a question: show me user data
  [Gets table + pie chart of users]

  # Exit
  > Ask a question: quit

  Example 2: Resuming Previous Work

  # Start system
  python file.py

  # Check previous conversations
  > Ask a question: /list
  [See list of conversations]

  # Resume specific conversation
  > Ask a question: /resume e89a8211-e479-484b-8626-fa8778277450
  [RESUMED] Conversation e89a8211...
  Messages: 13
  You can now continue the conversation.

  # Continue asking questions in that context
  > Ask a question: filter by admin users only
  [System remembers previous context]

  # Exit
  > Ask a question: quit

  Example 3: Retrieving Past Results

  # Start system
  python file.py

  # Get previous result without resuming conversation
  > Ask a question: /last e89a8211-e479-484b-8626-fa8778277450
  [LAST RESPONSE from e89a8211...]:
  [
      {
          "type": "table",
          "headers": ["Role", "User Count"],
          "rows": [["admin", 4], ["user", 1], ["legal", 1]]
      }
  ]

  # Exit
  > Ask a question: quit

  Response Types You'll Get

  Structured JSON Output

  All responses include combinations of:

  Tables:
  {
      "type": "table",
      "headers": ["Name", "Email", "Role"],
      "rows": [["John", "john@example.com", "admin"]]
  }

  Charts:
  {
      "type": "chart",
      "chartType": "pie",
      "data": {
          "labels": ["admin", "user"],
          "datasets": [{"data": [4, 2]}]
      }
  }

  Text:
  {
      "type": "text",
      "content": "Found 6 total users."
  }

  Tips for Best Results

  Query Examples That Work Well:

  - "show me all users"
  - "get document count by type"
  - "find legal documents from last month"
  - "user data in table format"
  - "document trends over time"

  Pro Tips:

  1. Be specific - "show users by role" vs "show users"
  2. Use conversation history - Resume threads to build on previous queries
  3. Check previous work - Use /last to reference past results
  4. Organize sessions - Use /list to track your conversation history

  Troubleshooting

  Common Issues:

  - "No response found" - That conversation may not have generated output yet
  - "Conversation not found" - Check the thread ID is correct (use /list to verify)
  - MongoDB connection errors - Ensure MongoDB is running on localhost:27017

  Getting Help:

  > Ask a question: /help
  Always shows current available commands and syntax.




