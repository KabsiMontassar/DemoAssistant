# Minimal Demo Setup: Material Pricing AI Assistant
## Proof of Concept for Client Demonstration

---

## Demo Objectives

**Primary Goal:** Demonstrate core functionality to client in minimal time and cost

**What Client Will See:**
1. Chat interface to ask questions about material pricing
2. System retrieves answers from uploaded TXT files
3. Web search toggle for external price checking
4. Source citations showing which file/project the answer came from
5. File upload capability with automatic re-embedding



---

## Architecture Overview

### Simplified Stack

```
Frontend (React/Next.js)
    ↓
Backend API (FastAPI)
    ↓
├── OpenRouter API (Mistral)
├── ChromaDB (Local Vector DB)
└── Tavily API (Web Search)
```

### Component Choices

**Frontend:**
- Next.js with Shadcn UI components
- Simple chat interface
- File upload drag-and-drop
- Web search toggle button
- Source citation display

**Backend:**
- FastAPI (Python web framework)
- Runs locally on your development machine
- Handles file upload and processing
- Manages embeddings and retrieval

**Vector Database:**
- ChromaDB in embedded mode (no separate server)
- Stores document embeddings locally
- Persists to disk in project folder
- Zero configuration required

**LLM:**
- OpenRouter API with Mistral model
- Pay-per-use (no infrastructure needed)
- Multiple Mistral versions available
- Cost: ~€0.0002-0.0006 per 1K tokens

**Embeddings:**
- Sentence Transformers (local model)
- Model: all-MiniLM-L6-v2 (fast, good quality)
- Runs on CPU (no GPU needed for demo)
- Free (open source)

**Web Search:**
- Tavily API (AI-optimized search)
- Free tier: 1000 searches/month
- Returns clean, structured results
- Easy integration

---

## File Structure Setup

### Demo File Organization

```
demo-project/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── embedding.py            # Embedding generation
│   ├── retrieval.py            # Vector search logic
│   ├── llm.py                  # OpenRouter integration
│   ├── file_watcher.py         # Auto-update on file changes
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Main chat interface
│   │   └── api/
│   │       └── chat/route.ts   # API route handler
│   ├── components/
│   │   ├── ChatInterface.tsx   # Chat UI component
│   │   ├── FileUpload.tsx      # File upload component
│   │   └── SourceCitation.tsx  # Citation display
│   └── package.json
│
├── data/
│   ├── materials/              # Client's material files
│   │   ├── wood/
│   │   │   ├── oak.txt
│   │   │   └── pine.txt
│   │   ├── concrete/
│   │   │   └── standard.txt
│   │   └── stone/
│   │       └── marble.txt
│   └── chroma_db/              # Vector database storage (auto-created)
│
└── .env                        # API keys and configuration
```

### Example Material File Format

**File: data/materials/wood/oak.txt**

```
Material: Oak Wood - European White Oak
Category: Hardwood
Unit: Cubic Meter (m³)

PROJECT PRICING HISTORY:
------------------------

Project: Villa Moderne Residence
Date: 2024-01-15
Price: €450/m³
Quantity: 12 m³
Notes: Premium grade, kiln-dried

Project: Corporate Office Renovation
Date: 2024-02-20
Price: €475/m³
Quantity: 8 m³
Notes: Special cut requirements

Project: Restaurant Interior
Date: 2024-03-10
Price: €460/m³
Quantity: 15 m³
Notes: Standard grade

SPECIFICATIONS:
--------------
- Moisture Content: 8-10%
- Thickness: 25-50mm available
- Length: Up to 3m standard
- Supplier: Nordic Wood Supply Co.
- Lead Time: 2-3 weeks

NOTES:
------
Price variations based on:
- Order quantity (bulk discounts available)
- Seasonal availability
- Special cutting requirements
- Finishing specifications
```

**File: data/materials/concrete/standard.txt**

```
Material: Standard Concrete Mix
Category: Concrete
Unit: Cubic Meter (m³)

PROJECT PRICING HISTORY:
------------------------

Project: Foundation - Residential Complex A
Date: 2024-01-05
Price: €95/m³
Quantity: 250 m³
Notes: C30/37 strength class

Project: Parking Structure Extension
Date: 2024-02-12
Price: €98/m³
Quantity: 180 m³
Notes: C35/45 strength class, accelerated curing

Project: Retail Space Floor Slab
Date: 2024-03-18
Price: €92/m³
Quantity: 120 m³
Notes: C25/30 strength class

SPECIFICATIONS:
--------------
- Strength Classes Available: C20/25, C25/30, C30/37, C35/45
- Slump: 100-150mm
- Maximum Aggregate Size: 20mm
- Supplier: Regional Concrete Ltd.
- Delivery: Truck mixer, 6-8 m³ capacity

NOTES:
------
Price includes:
- Materials and mixing
- Delivery within 30km radius
- Standard additives

Additional costs for:
- Special additives (plasticizers, retarders)
- Extended delivery distance
- Weekend/holiday delivery
- Pump truck (if required)
```

---

## Setup Instructions

### Step 1: Environment Setup (30 minutes)

**Install Required Software:**

1. Python 3.10 or higher
2. Node.js 18 or higher
3. Git

**Create Project Structure:**

Create all folders as shown in file structure above.

**Get API Keys:**

1. OpenRouter API Key:
   - Sign up at openrouter.ai
   - Create API key
   - Free credits provided for testing

2. Tavily API Key:
   - Sign up at tavily.com
   - Free tier: 1000 searches/month
   - Get API key from dashboard

**Create .env file:**

```
OPENROUTER_API_KEY=your_openrouter_key_here
TAVILY_API_KEY=your_tavily_key_here
DATA_PATH=./data/materials
CHROMA_PATH=./data/chroma_db
```

### Step 2: Backend Setup (1-2 hours)

**Install Python Dependencies:**

Create requirements.txt with necessary packages:
- fastapi
- uvicorn
- chromadb
- sentence-transformers
- openai (for OpenRouter API compatibility)
- python-multipart (for file uploads)
- watchdog (for file monitoring)
- tavily-python (for web search)
- python-dotenv

**Initialize ChromaDB:**

Database automatically initializes on first run. No manual setup required.

**Download Embedding Model:**

First run automatically downloads sentence-transformers model. Takes 2-3 minutes. Model cached locally for subsequent runs.

### Step 3: Frontend Setup (1-2 hours)

**Initialize Next.js Project:**

Create Next.js application with TypeScript and Tailwind CSS.

**Install UI Components:**

Use Shadcn UI for pre-built components:
- Button, Input, Card
- Dialog for file upload
- Badge for source citations
- Switch for web search toggle

**Configure API Connection:**

Set backend API URL in environment variables.

### Step 4: Create Demo Data (30 minutes)

**Prepare Example Files:**

Create 10-15 example material files following the format shown above.

**File Organization:**

```
data/materials/
├── wood/
│   ├── oak.txt
│   ├── pine.txt
│   ├── walnut.txt
│   └── mahogany.txt
├── concrete/
│   ├── standard.txt
│   ├── high-strength.txt
│   └── lightweight.txt
├── stone/
│   ├── marble.txt
│   ├── granite.txt
│   └── limestone.txt
└── metal/
    ├── steel-rebar.txt
    └── aluminum.txt
```

**Ensure Variety:**

Include different:
- Price ranges
- Project types
- Date ranges
- Quantity scales
- Special notes

### Step 5: Initial Embedding (15 minutes)

**Run Embedding Process:**

Start backend server. It automatically:
1. Scans data/materials directory
2. Loads all TXT files
3. Chunks text into segments
4. Generates embeddings
5. Stores in ChromaDB

**For 15 files:** Takes approximately 2-3 minutes on standard laptop.

**Verify Success:**

Check that chroma_db folder is created and populated with data.

---

## User Workflows

### Workflow 1: Query Material Pricing

**User Action:**
1. Opens chat interface
2. Types: "What was the pricing for oak wood in the Villa Moderne project?"
3. Clicks send

**System Process:**
1. Backend receives query
2. Converts query to embedding
3. Searches ChromaDB for similar document chunks
4. Retrieves top 5 most relevant chunks
5. Sends query + context to Mistral via OpenRouter
6. Mistral generates response with citation
7. Response displayed with source file highlighted

**Expected Response:**

```
In the Villa Moderne Residence project, European White Oak was priced at €450 per 
cubic meter (m³). This was for a quantity of 12 m³ of premium grade, kiln-dried oak 
delivered on January 15, 2024.

Source: wood/oak.txt - Villa Moderne Residence project
```

### Workflow 2: Web Search for Current Prices

**User Action:**
1. Toggles "Search Web" switch ON
2. Types: "What is the current market price for concrete?"
3. Clicks send

**System Process:**
1. Backend detects web search flag enabled
2. Performs TWO parallel searches:
   - ChromaDB for internal historical pricing
   - Tavily API for current web data
3. Combines both results
4. Sends to Mistral with instruction to compare sources
5. Mistral synthesizes answer from both sources
6. Response shows both internal and external sources

**Expected Response:**

```
Based on our internal records, standard concrete (C30/37) was priced between €92-€98 
per m³ in early 2024 across various projects.

Current market research indicates that concrete prices in Europe are now ranging from 
€95-€110 per m³ depending on strength class and delivery location, with some regions 
experiencing price increases due to energy costs.

Sources: 
- Internal: concrete/standard.txt (multiple projects)
- Web: Construction Industry Report Q1 2024, European Concrete Association
```

### Workflow 3: Upload New File

**User Action:**
1. Clicks "Upload Files" button
2. Selects new file: "bamboo.txt"
3. File uploads

**System Process:**
1. Frontend sends file to backend API
2. Backend saves file to data/materials/wood/
3. File watcher detects new file
4. Triggers embedding process for new file only
5. Embeddings added to ChromaDB
6. User receives notification: "File processed successfully"
7. New file immediately available for queries

**Time:** 5-10 seconds for single file

### Workflow 4: Update Existing File

**User Action:**
1. Client edits oak.txt locally to add new project pricing
2. Saves file

**System Process:**
1. File watcher detects file modification
2. Computes file hash to confirm changes
3. Deletes old embeddings for that file from ChromaDB
4. Re-embeds updated file
5. Updates ChromaDB with new embeddings
6. Process completes silently in background

**Time:** 3-5 seconds

**User Verification:**
Query the new information to confirm it's retrievable.

---

## Demo Presentation Flow

### Introduction (2 minutes)

**Show:**
- Chat interface
- File upload area
- Web search toggle
- Example material files in file browser

**Explain:**
- System has been pre-loaded with 15 example material files
- Each file contains pricing history across different projects
- System can answer questions about any material or project

### Demo 1: Simple Query (3 minutes)

**Query 1:** "What was the price of oak wood in the Corporate Office Renovation?"

**Expected Behavior:**
- Response appears in 2-3 seconds
- Shows exact price: €475/m³
- Cites source file: wood/oak.txt
- Displays relevant project details

**Key Point:** System retrieves exact information from documents with proper citation.

### Demo 2: Comparative Query (3 minutes)

**Query 2:** "Compare the concrete pricing across all projects in 2024"

**Expected Behavior:**
- System retrieves multiple concrete pricing entries
- Lists all projects with dates and prices
- Shows price range: €92-€98/m³
- Cites concrete/standard.txt as source

**Key Point:** System can synthesize information across multiple entries in same file.

### Demo 3: Cross-Material Query (3 minutes)

**Query 3:** "Which material had the highest price variation across projects?"

**Expected Behavior:**
- System analyzes pricing data across all materials
- Identifies material with highest variance
- Explains reasoning with specific examples
- Cites multiple source files

**Key Point:** System can analyze and compare across different materials.

### Demo 4: Web Search Integration (4 minutes)

**Query 4 (Toggle OFF):** "What is the current price of marble?"

**Expected Behavior:**
- Returns only internal historical data
- Cites stone/marble.txt
- Shows historical project pricing

**Query 4 (Toggle ON):** "What is the current price of marble?"

**Expected Behavior:**
- Searches internal records AND web
- Compares historical vs current pricing
- Shows both sources clearly labeled
- Takes 3-5 seconds (web search adds latency)

**Key Point:** Web search provides current market context when needed.

### Demo 5: File Upload (5 minutes)

**Action:**
1. Show new file: bamboo.txt (prepare in advance)
2. Upload file via interface
3. Wait for processing confirmation (5-10 seconds)
4. Query: "Tell me about bamboo pricing"

**Expected Behavior:**
- File uploads successfully
- Processing notification appears
- Query returns information from newly uploaded file
- Cites bamboo.txt as source

**Key Point:** New materials can be added instantly without system restart.

### Demo 6: File Update (4 minutes)

**Action:**
1. Open oak.txt in text editor (show on screen)
2. Add new project entry with today's date and different price
3. Save file
4. Query: "What is the most recent oak wood pricing?"

**Expected Behavior:**
- System detects file change (background notification)
- Query returns the newly added entry
- Shows today's date
- No system restart required

**Key Point:** Updates to existing data are automatic and immediate.

### Demo 7: Edge Cases (3 minutes)

**Query 7:** "What is the price of diamond flooring?"

**Expected Behavior:**
- System searches but finds no relevant information
- Responds: "I don't have information about diamond flooring in the materials database"
- Does not hallucinate or make up information
- Suggests web search if toggle is off

**Key Point:** System acknowledges limitations and doesn't fabricate data.

### Conclusion (2 minutes)

**Summary:**
- System successfully retrieves pricing information from 1M+ files (demonstrate with 15 for now)
- Automatic updates when files change
- Web search for market context
- Accurate citations prevent hallucination
- Scalable to full production with proper infrastructure

**Next Steps:**
- Gather client feedback on interface and functionality
- Discuss full-scale deployment requirements
- Review production architecture plan
- Establish timeline for production system

---

## Technical Details for Client Discussion

### Current Demo Specifications

**Capacity:**
- Files: 15 example files (scalable to millions)
- Processing: 1 file per 2-3 seconds
- Query speed: 1-3 seconds per query
- Concurrent users: 1 (demo only)

**Costs (Demo Period):**
- OpenRouter API: ~€0.50 for 100 queries
- Tavily API: Free (1000 searches/month)
- Infrastructure: Local laptop (no server costs)
- Total: < €5 for entire demo period

## Demo Limitations (Be Transparent)

### Current Limitations

**Performance:**
- Query speed slower than production (1-3 sec vs <100ms)
- Single user only (no load balancing)
- Runs on laptop (not 24/7 available)

**Scalability:**
- Limited to ~1,000 files on laptop
- CPU-based embedding (slow for large batches)
- No redundancy or failover

**Features:**
- Basic chat interface (no advanced UI features)
- No user authentication
- No query history or analytics
- Limited error handling

**Infrastructure:**
- No monitoring or alerting
- No automated backups
- No disaster recovery
- Development-grade security only

### What Demo DOES Prove

**Core Functionality:**
- Retrieval from structured text files works
- LLM integration successful
- Web search integration functional
- Auto-update mechanism operational
- Citation system prevents hallucination

**User Experience:**
- Interface is intuitive
- Response quality is high
- Source attribution is clear
- Upload process is simple

**Technical Feasibility:**
- RAG pipeline functional end-to-end
- Mistral provides good responses
- Embedding quality is sufficient
- Architecture is sound

---


## Quick Start Commands

### Backend Setup

```bash
# Navigate to backend folder
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run backend server
python main.py

# Server runs on http://localhost:8000
```

### Frontend Setup

```bash
# Navigate to frontend folder
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Frontend runs on http://localhost:3000
```

### Access Demo

Open browser to: http://localhost:3000

---

## Troubleshooting

### Backend won't start

**Issue:** Missing API keys

**Solution:** Ensure .env file exists with valid OpenRouter and Tavily API keys

### Embeddings fail

**Issue:** Insufficient memory

**Solution:** Embedding model requires ~2GB RAM. Close other applications.

### Queries return no results

**Issue:** ChromaDB not initialized

**Solution:** Check that data/chroma_db folder exists and contains data. Restart backend to re-embed.

### File upload doesn't work

**Issue:** File path permissions

**Solution:** Ensure data/materials folder has write permissions

### Web search not working

**Issue:** Invalid Tavily API key or rate limit exceeded

**Solution:** Verify API key in .env file. Check Tavily dashboard for remaining quota.

---

## Summary

**Demo System:**
- Minimal setup (3-5 days)
- Low cost (< €100)
- Proves core functionality
- Identifies requirements

**Client Sees:**
- Working prototype
- Real material pricing queries
- Source citations
- Auto-update capability
- Web search integration

**Next Step:**
- Client approves concept
- Plan pilot or production phase
- Begin full implementation

This demo provides sufficient proof-of-concept while keeping investment minimal until client approval.
