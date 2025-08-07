
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your Google API key:
# GEMINI_API_KEY=your_google_api_key_here
```

# Run the standalone RAG
python working.py




