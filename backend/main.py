"""
Material Pricing AI Assistant - Backend API
Main FastAPI application for handling chat requests and embeddings.
"""

import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import json

from embedding import EmbeddingManager
from retrieval import RetrieverManager
from llm import LLMManager
from file_watcher import FileWatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

# Validate required environment variables
REQUIRED_ENV_VARS = ['OPENROUTER_API_KEY', 'DATA_PATH', 'CHROMA_PATH']
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize managers (global state)
embedding_manager = None
retriever_manager = None
llm_manager = None
file_watcher = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle management.
    Initializes all managers and file watcher on startup.
    """
    global embedding_manager, retriever_manager, llm_manager, file_watcher
    
    try:
        logger.info("Initializing backend components...")
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        logger.info("✓ Embedding manager initialized")
        
        # Initialize retriever manager
        retriever_manager = RetrieverManager(embedding_manager)
        logger.info("✓ Retriever manager initialized")
        
        # Initialize LLM manager
        llm_manager = LLMManager()
        logger.info("✓ LLM manager initialized")
        
        # Initial embedding of existing files
        logger.info("Performing initial embedding of existing files...")
        embedding_manager.embed_directory()
        logger.info("✓ Initial embedding complete")
        
        # Initialize file watcher for automatic re-embedding on structure changes
        file_watcher = FileWatcher(
            watch_path=os.getenv('DATA_PATH', './data/materials'),
            embedding_manager=embedding_manager,
            retriever_manager=retriever_manager
        )
        file_watcher.start()
        logger.info("✓ File watcher started - monitoring structure for changes")
        
        yield
        
        # Shutdown
        logger.info("Shutting down...")
        if file_watcher:
            file_watcher.stop()
        logger.info("✓ Backend shutdown complete")
        
    except Exception as e:
        logger.error(f"Fatal error during startup: {str(e)}", exc_info=True)
        raise


# Create FastAPI app with lifespan management
app = FastAPI(
    title="Material Pricing AI Assistant",
    description="RAG-based AI assistant for material pricing information",
    version="0.1.0",
    lifespan=lifespan
)

# Configure CORS - Allow all origins for development
# This is needed for WebSocket connections from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str
    use_web_search: bool = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What was the price of oak wood in the Villa Moderne project?",
                "use_web_search": False
            }
        }


class SourceCitation(BaseModel):
    """Source citation model."""
    file_path: str
    content_snippet: str
    relevance_score: float


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    sources: list[SourceCitation]
    web_search_used: bool


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    components: dict


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify all components are running.
    
    Returns:
        HealthResponse: Status of all system components
    """
    components = {
        "embedding_manager": "operational" if embedding_manager else "error",
        "retriever_manager": "operational" if retriever_manager else "error",
        "llm_manager": "operational" if llm_manager else "error",
        "file_watcher": "operational" if file_watcher and file_watcher.is_alive() else "error",
    }
    
    all_operational = all(v == "operational" for v in components.values())
    
    return HealthResponse(
        status="healthy" if all_operational else "degraded",
        components=components
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for material pricing queries.
    
    Args:
        request: ChatRequest containing query and web search flag
        
    Returns:
        ChatResponse with AI response and source citations
        
    Raises:
        HTTPException: If components are not initialized or request is invalid
    """
    # Validate request
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if len(request.query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 characters)")
    
    # Verify managers are initialized
    if not all([embedding_manager, retriever_manager, llm_manager]):
        raise HTTPException(status_code=503, detail="Service not fully initialized")
    
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Step 1: Retrieve relevant documents from vector DB
        retrieved_docs = retriever_manager.retrieve(
            query=request.query,
            top_k=5
        )
        
        if not retrieved_docs:
            return ChatResponse(
                response="I don't have any information about this topic in the materials database. Please try a different query or enable web search for current market information.",
                sources=[],
                web_search_used=False
            )
        
        # Step 2: Format retrieved context
        context = "\n\n".join([
            f"Source: {doc['file_path']}\n{doc['content']}"
            for doc in retrieved_docs
        ])
        
        # Step 3: Generate response using LLM
        llm_response = llm_manager.generate_response(
            query=request.query,
            context=context,
            use_web_search=request.use_web_search
        )
        
        # Step 4: Format source citations
        sources = [
            SourceCitation(
                file_path=doc['file_path'],
                content_snippet=doc['content'][:200] + "...",
                relevance_score=doc['score']
            )
            for doc in retrieved_docs
        ]
        
        return ChatResponse(
            response=llm_response,
            sources=sources,
            web_search_used=request.use_web_search
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing chat request: {error_msg}", exc_info=True)
        
        # Return appropriate HTTP status codes based on error type
        if "Invalid API key" in error_msg:
            raise HTTPException(status_code=401, detail="Invalid OpenRouter API key. Please check your configuration.")
        elif "Rate limit" in error_msg:
            raise HTTPException(status_code=429, detail="API rate limit exceeded. Please try again later.")
        elif "401" in error_msg or "Unauthorized" in error_msg:
            raise HTTPException(status_code=401, detail="Authentication error. Check your API key.")
        else:
            raise HTTPException(status_code=500, detail=f"Error processing request: {error_msg}")




@app.get("/api/stats")
async def get_stats():
    """
    Get statistics about the system state.
    
    Returns:
        dict: Statistics about embedded documents and database
    """
    if not retriever_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        stats = retriever_manager.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving statistics")


@app.post("/api/reprocess")
async def reprocess_files():
    """
    Manually trigger reprocessing of all files in the materials directory.
    Useful for recovering from embedding failures or updating the database.
    
    Returns:
        dict: Status of reprocessing operation
    """
    if not embedding_manager or not retriever_manager:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        logger.info("Starting manual reprocessing of all files...")
        embedding_manager.embed_directory()
        stats = retriever_manager.get_stats()
        
        return {
            "status": "success",
            "message": "All files reprocessed successfully",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error reprocessing files: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error reprocessing files")


@app.get("/api/file-structure")
async def get_file_structure():
    """
    Get the file structure of the materials data directory.
    
    Returns:
        list: Hierarchical file structure
    """
    try:
        data_path = os.getenv('DATA_PATH', './data/materials')
        materials_path = os.path.join(data_path, 'materials')
        
        # Check if materials directory exists
        if not os.path.exists(materials_path):
            materials_path = data_path
        
        # Build the tree structure
        file_structure = build_file_tree(materials_path)
        return file_structure
        
    except Exception as e:
        logger.error(f"Error getting file structure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving file structure")


@app.get("/api/download")
async def download_file(path: str):
    """
    Download a file from the materials directory.
    
    Args:
        path: Relative path to the file within materials
        
    Returns:
        FileResponse: The file to download
    """
    try:
        data_path = os.getenv('DATA_PATH', './data/materials')
        
        # Construct the full file path
        if path.startswith('materials/'):
            base_path = data_path
        else:
            base_path = os.path.join(data_path, 'materials')
        
        full_path = os.path.join(base_path, path)
        
        # Resolve the path to prevent directory traversal attacks
        full_path = os.path.abspath(full_path)
        base_abs = os.path.abspath(base_path)
        
        # Verify the file is within the data directory
        if not full_path.startswith(base_abs):
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Get the file name for the download
        file_name = os.path.basename(full_path)
        
        return FileResponse(
            path=full_path,
            filename=file_name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error downloading file")


def build_file_tree(directory: str, relative_path: str = ""):
    """
    Recursively build a file tree structure.
    
    Args:
        directory: Full path to directory
        relative_path: Relative path for display
        
    Returns:
        dict with name, type, path, and children if folder
    """
    try:
        path = Path(directory)
        items = []
        
        # Sort items alphabetically, folders first
        sorted_items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        
        for item in sorted_items:
            rel_path = f"{relative_path}/{item.name}" if relative_path else item.name
            
            if item.is_dir() and not item.name.startswith('.'):
                # Recursively get children
                children = build_file_tree(str(item), rel_path)
                items.append({
                    "name": item.name,
                    "type": "folder",
                    "path": rel_path,
                    "children": children
                })
            elif item.is_file() and not item.name.startswith('.'):
                items.append({
                    "name": item.name,
                    "type": "file",
                    "path": rel_path
                })
        
        return items
    except Exception as e:
        logger.error(f"Error building file tree for {directory}: {str(e)}")
        return []


# ============================================================================
# Startup and Main
# ============================================================================


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Material Pricing AI Assistant",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    port = int(os.getenv('BACKEND_PORT', 8000))
    host = os.getenv('BACKEND_HOST', '0.0.0.0')
    
    logger.info(f"Starting backend server on {host}:{port}...")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
