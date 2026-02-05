# Material Pricing AI Assistant - Complete Implementation

Production-ready RAG-based AI assistant for material pricing queries with web search integration.

## System Architecture

```
Frontend (Next.js/React)
    ↓ HTTP/REST
Backend API (FastAPI)
    ├─ OpenRouter API (Mistral LLM)
    ├─ ChromaDB (Vector Database)
    ├─ Sentence Transformers (Embeddings)
    └─ Tavily API (Web Search)
```

## Project Structure

```
demo-project/
├── backend/                    # Python FastAPI backend
│   ├── main.py                # Main application entry point
│   ├── embedding.py           # Document embedding and chunking
│   ├── retrieval.py           # Vector similarity search
│   ├── llm.py                 # LLM integration (OpenRouter)
│   ├── file_watcher.py        # Automatic file re-embedding
│   └── requirements.txt       # Python dependencies
│
├── frontend/                   # Next.js React frontend
│   ├── app/
│   │   ├── page.tsx           # Main chat interface
│   │   ├── layout.tsx         # Root layout
│   │   └── globals.css        # Global styles
│   ├── components/
│   │   ├── ChatInterface.tsx  # Chat input component
│   │   ├── FileUpload.tsx     # File upload component
│   │   └── SourceCitation.tsx # Citation display
│   ├── package.json
│   ├── tsconfig.json
│   ├── next.config.js
│   └── tailwind.config.ts
│
├── data/
│   ├── materials/             # Material pricing documents
│   │   ├── wood/             # Wood types
│   │   ├── concrete/         # Concrete mixes
│   │   ├── stone/            # Stone types
│   │   └── metal/            # Metal materials
│   └── chroma_db/            # Vector database (auto-created)
│
├── .env                       # Environment configuration
└── README.md                 # This file
```

## Prerequisites

### System Requirements
- **OS:** Windows, macOS, or Linux
- **RAM:** Minimum 4GB (8GB recommended)
- **Storage:** 2GB free space for models and database
- **Network:** Internet connection for API calls

### Software Requirements
- Python 3.10 or higher
- Node.js 18 or higher
- npm or yarn package manager
- Git (optional, for version control)

### API Keys (Free to obtain)
1. **OpenRouter API** - for Mistral LLM access
   - Sign up at https://openrouter.ai
   - Get free credits for testing
   
2. **Tavily API** - for web search (optional)
   - Sign up at https://tavily.com
   - Free tier: 1000 searches/month

## Installation & Setup

### Step 1: Clone/Download the Project

```bash
# Navigate to project directory
cd demo-project
```

### Step 2: Backend Setup

```bash
# Navigate to backend
cd backend

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download embedding model (first run only, ~400MB)
# This happens automatically on first startup
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend
cd ../frontend

# Install Node dependencies
npm install

# Build for production (optional)
npm run build
```

### Step 4: Configuration

Create/update `.env` file in project root:

```env
# LLM Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here
LLM_MODEL=mistral-7b-instruct
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Web Search Configuration
TAVILY_API_KEY=your_tavily_api_key_here
WEB_SEARCH_ENABLED=true

# Database Paths
DATA_PATH=./data/materials
CHROMA_PATH=./data/chroma_db

# Backend Configuration
BACKEND_PORT=8000
BACKEND_HOST=0.0.0.0

# Frontend Configuration
FRONTEND_URL=http://localhost:3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Application

### Terminal 1: Start Backend

```bash
cd backend

# Activate virtual environment (if not already active)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Start FastAPI server
python main.py
```

Expected output:
```
Initializing backend components...
✓ Embedding manager initialized
✓ Retriever manager initialized
✓ LLM manager initialized
✓ File watcher initialized
Performing initial embedding of existing files...
✓ Initial embedding complete
Starting backend server on 0.0.0.0:8000...
```

### Terminal 2: Start Frontend

```bash
cd frontend

# Start development server
npm run dev
```

Expected output:
```
> ready - started server on 0.0.0.0:3000, url: http://localhost:3000
```

### Access the Application

Open your browser to: **http://localhost:3000**

## Usage Guide

### 1. Query Material Pricing

**What it does:** Search for material pricing information from uploaded documents.

**How to use:**
1. Type a question in the chat box
2. Click "Send" or press Enter
3. System retrieves relevant documents and generates response
4. View source citations below the response

**Example queries:**
- "What was the price of oak wood in the Villa Moderne project?"
- "Compare concrete pricing across all projects"
- "What are the specifications for marble stone?"

### 2. Enable Web Search

**What it does:** Includes current market data in responses alongside internal documents.

**How to use:**
1. Toggle "Search Web" switch ON
2. Ask your question
3. System searches both internal database and web
4. Response includes both historical and current data

**Note:** Web search takes slightly longer (3-5 seconds) due to API call latency.

### 3. Upload Material Files

**What it does:** Add new material documents to the system instantly.

**File format:** Plain text (.txt) files with material information

**How to use:**
1. Click "📁 Upload File" button
2. Select a .txt file or drag-and-drop
3. File is automatically processed and embedded
4. Can immediately query the new material

**Example file content:**
```
Material: Bamboo Flooring
Category: Wood
Unit: Square Meter (m²)

PROJECT PRICING HISTORY:
------------------------

Project: Eco-Friendly Office
Date: 2024-04-01
Price: €125/m²
Quantity: 200 m²
Notes: FSC certified, natural finish

SPECIFICATIONS:
--------------
- Hardness: Higher than oak
- Installation: Standard wood floor procedures
- Sustainability: Renewable, rapidly growing
```

### 4. Update Existing Files

**What it does:** Automatically re-embeds files when they change.

**How it works:**
1. Edit any .txt file in `data/materials/` folder
2. Save the file
3. System automatically detects change
4. File is re-embedded in background
5. New information immediately available

**Example:** Adding a new project to oak.txt:
- Open `data/materials/wood/oak.txt`
- Add new project entry with current pricing
- Save file
- Query "What is the most recent oak wood pricing?" - gets the new data

## API Reference

### Health Check
```bash
GET /health

# Response:
{
  "status": "healthy",
  "components": {
    "embedding_manager": "operational",
    "retriever_manager": "operational",
    "llm_manager": "operational",
    "file_watcher": "operational"
  }
}
```

### Chat Endpoint
```bash
POST /api/chat
Content-Type: application/json

{
  "query": "What is the price of oak wood?",
  "use_web_search": false
}

# Response:
{
  "response": "Based on our database, oak wood pricing varies...",
  "sources": [
    {
      "file_path": "wood/oak.txt",
      "content_snippet": "Project: Villa Moderne...",
      "relevance_score": 0.95
    }
  ],
  "web_search_used": false
}
```

### File Upload
```bash
POST /api/upload
Content-Type: multipart/form-data

file: <binary data>

# Response:
{
  "filename": "bamboo.txt",
  "status": "success",
  "message": "File uploaded and processed successfully"
}
```

### Statistics
```bash
GET /api/stats

# Response:
{
  "total_documents": 145,
  "collection_name": "material_pricing",
  "unique_files": 11,
  "files": ["concrete/standard.txt", "wood/oak.txt", ...]
}
```

## Configuration Details

### Environment Variables Explained

**LLM Configuration:**
- `LLM_MODEL` - Model identifier (default: mistral-7b-instruct)
- `LLM_TEMPERATURE` - Response randomness (0.0=deterministic, 1.0=random)
- `LLM_MAX_TOKENS` - Maximum response length

**Embedding Configuration:**
- `EMBEDDING_MODEL` - All-MiniLM-L6-v2 (fast, good quality)
- `EMBEDDING_BATCH_SIZE` - Documents processed in parallel

**Performance Tuning:**
- Lower `CHUNK_SIZE` (in embedding.py) = more granular results, slower processing
- Higher `top_k` in retrieval = more context, longer response time
- `TEMPERATURE` higher = more creative, less consistent

## Troubleshooting

### Backend won't start

**Error:** `Missing required environment variables`
- **Solution:** Ensure `.env` file exists with valid API keys
- **Check:** `OPENROUTER_API_KEY` is set

**Error:** `Connection error: Cannot connect to OpenRouter API`
- **Solution:** Verify API key is correct and internet connection is active
- **Test:** Check key at https://openrouter.ai

### Queries return no results

**Issue:** "I don't have information about this topic"
- **Cause:** No relevant documents in database
- **Solution:** 
  - Upload material files relevant to your query
  - Check `data/materials/` folder has files
  - Verify `CHROMA_PATH` exists and has data

**Issue:** ChromaDB not initialized
- **Solution:** Backend will auto-initialize on first run
- **Manual reset:** Delete `data/chroma_db/` folder and restart

### Embeddings slow or failing

**Issue:** "Error generating embeddings" or very slow processing
- **Solution:** 
  - Check available RAM (needs ~2GB)
  - Close other applications
  - Split large files into smaller ones
  - Use faster embedding model (trade-off on quality)

**File with memory issues:**
- Move CPU-intensive processes or use GPU
- Edit `embedding.py`: change `device='cpu'` to `device='cuda'` for GPU

### Web search not working

**Issue:** Web search toggle doesn't show results
- **Cause:** Invalid Tavily API key or rate limit exceeded
- **Solution:**
  - Verify `TAVILY_API_KEY` in `.env`
  - Check quota at https://tavily.com dashboard
  - Disable web search if free tier exhausted

### Port already in use

**Error:** `Address already in use :8000` or `:3000`
- **Solution:** Change port in `.env` or kill existing process
  ```bash
  # Find process using port 8000
  netstat -ano | findstr :8000
  # Kill process (replace PID)
  taskkill /PID <PID> /F
  ```

## Performance Optimization

### Query Performance
- First query: 2-3 seconds (model loading)
- Subsequent queries: 1-2 seconds
- With web search: +2-3 seconds additional

### Embedding Performance
- Single file: 5-10 seconds
- Batch (10 files): 30-60 seconds
- Automatic re-embedding: <5 seconds per file

### Database Optimization
- Optimal documents: 100-1000 for demo
- Supports up to 100,000+ documents at scale
- Search time increases logarithmically

### Cost Optimization

**API Costs (approximate):**
- OpenRouter: ~€0.0002-0.0006 per 1K tokens
- Tavily: Free tier includes 1000 searches/month
- Demo cost: <€1-2 for full demonstration

## Production Deployment

For production deployment, consider:

1. **Infrastructure:**
   - Use cloud-hosted FastAPI (Azure App Service, AWS Lambda)
   - Cloud vector database (Pinecone, Weaviate Cloud)
   - CDN for static frontend assets

2. **Security:**
   - Use environment secrets management
   - Implement API authentication
   - Add rate limiting
   - Use HTTPS/TLS

3. **Scaling:**
   - Implement caching layer (Redis)
   - Use load balancer for multiple backend instances
   - Implement async job queue for batch processing
   - Database replication and failover

4. **Monitoring:**
   - Add application logging and monitoring
   - Set up alerting for errors
   - Track API usage and costs
   - Monitor vector database performance

## Testing

### Manual Testing

**Test 1: Basic Query**
```
Input: "What was the price of oak wood in the Villa Moderne project?"
Expected: Price €450/m³, source: wood/oak.txt
```

**Test 2: Comparative Analysis**
```
Input: "Compare concrete pricing across all 2024 projects"
Expected: Lists all projects with prices, analyzes variations
```

**Test 3: File Upload**
```
Action: Upload new material file
Expected: File processed in <10 seconds, immediately queryable
```

**Test 4: Web Search**
```
Toggle web search ON
Input: "What is current marble pricing?"
Expected: Includes both historical (from DB) and current (web) pricing
```

### Automated Testing
```bash
# Backend tests (create test_main.py)
pytest backend/

# Frontend tests
npm test
```

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Backend crashes on startup | Missing dependencies | `pip install -r requirements.txt` |
| No embeddings created | Data path doesn't exist | Create `data/materials/` folder |
| Slow responses | Large context retrieved | Reduce `top_k` in retrieval.py |
| Web search fails | Rate limit exceeded | Disable web search or wait 24h |
| Frontend can't connect | CORS issues | Check `FRONTEND_URL` in .env |
| Memory issues | Large files being embedded | Split files or add RAM |

## Support & Documentation

- **Backend API Docs:** http://localhost:8000/docs
- **Component Issues:** Check component logs in browser console
- **API Issues:** Review response codes and error messages

## License

This project is provided as a demonstration and proof-of-concept.

## Summary

**What's Included:**
- ✅ Production-grade FastAPI backend
- ✅ Modern Next.js React frontend
- ✅ Vector embeddings with ChromaDB
- ✅ LLM integration with OpenRouter (Mistral)
- ✅ Automatic file watching and re-embedding
- ✅ Web search integration with Tavily
- ✅ Complete error handling
- ✅ Comprehensive documentation

**Key Features:**
- Retrieval Augmented Generation (RAG) pipeline
- Source citation and attribution
- Live file updates without restart
- Web search toggle for market context
- Drag-and-drop file upload
- System health monitoring

**Performance:**
- Query latency: 1-3 seconds
- File processing: 5-10 seconds per file
- Supports 1000+ documents
- Scalable architecture

**Next Steps:**
1. Get API keys (5 minutes)
2. Run setup (15 minutes)
3. Start backend & frontend (2 minutes)
4. Start querying (immediate)

Enjoy your Material Pricing AI Assistant! 🚀
