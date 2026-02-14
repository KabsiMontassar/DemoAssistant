"""
Material Pricing AI Assistant - Backend API
Main FastAPI application for handling chat requests and embeddings.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import asyncio

from embedding import EmbeddingManager
from retrieval import RetrieverManager
from llm import LLMManager
from file_watcher import FileWatcher
from content_extraction import ContentExtractor
from chunking import ChunkManager
from scoring import ScoringEngine, RankingEngine
from metadata_tracker import FileMetadataTracker
from hybrid_retrieval import HybridRetriever
from reranking import ReRankingEngine
from prompt_verification import PromptVerifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

# Set defaults for optional environment variables
if not os.getenv('DATA_PATH'):
    os.environ['DATA_PATH'] = './data'
if not os.getenv('CHROMA_PATH'):
    os.environ['CHROMA_PATH'] = './data/chroma_db'

# Validate required environment variables
REQUIRED_ENV_VARS = ['OPENROUTER_API_KEY']
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize managers (global state)
embedding_manager = None
retriever_manager = None
llm_manager = None
file_watcher = None
content_extractor = None
chunk_manager = None
scoring_engine = None
ranking_engine = None
metadata_tracker = None
hybrid_retriever = None
reranking_engine = None
prompt_verifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle management.
    Initializes all managers and file watcher on startup.
    """
    global embedding_manager, retriever_manager, llm_manager, file_watcher
    global content_extractor, chunk_manager, scoring_engine, ranking_engine, metadata_tracker, hybrid_retriever, reranking_engine, prompt_verifier
    
    try:
        logger.info("Initializing backend components...")
        
        # Ensure data directories exist
        data_path = Path(os.getenv('DATA_PATH', './data/materials'))
        chroma_path = Path(os.getenv('CHROMA_PATH', './data/chroma_db'))
        data_path.mkdir(parents=True, exist_ok=True)
        chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Data directories verified: {data_path}, {chroma_path}")
        
        # Initialize prompt verifier with error boundary
        try:
            prompt_verifier = PromptVerifier()
            logger.info("\u2713 Prompt verifier initialized")
        except Exception as e:
            logger.error(f"Prompt verifier initialization failed: {e}. Continuing without prompt verification.")
            prompt_verifier = None
        
        # Initialize content extraction
        content_extractor = ContentExtractor()
        logger.info("✓ Content extractor initialized")
        
        # Initialize chunking
        chunk_manager = ChunkManager()
        logger.info("✓ Chunk manager initialized")
        
        # Initialize scoring and ranking
        scoring_engine = ScoringEngine()
        ranking_engine = RankingEngine(scoring_engine)
        logger.info("✓ Scoring and ranking engines initialized")
        
        # Initialize re-ranking engine
        reranking_engine = ReRankingEngine()
        logger.info("✓ Re-ranking engine initialized")
        
        # Initialize metadata tracker
        metadata_tracker = FileMetadataTracker()
        logger.info("✓ Metadata tracker initialized")
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        logger.info("✓ Embedding manager initialized")
        
        # Initialize retriever manager
        retriever_manager = RetrieverManager(embedding_manager)
        logger.info("✓ Retriever manager initialized")
        
        # Initialize hybrid retriever
        hybrid_retriever = HybridRetriever(vector_weight=0.6, keyword_weight=0.4)
        logger.info("✓ Hybrid retriever initialized")
        
        # Initialize LLM manager
        llm_manager = LLMManager()
        logger.info("✓ LLM manager initialized")
        
        # Initial embedding of existing files (non-blocking if no files exist)
        logger.info("Performing initial embedding of existing files...")
        try:
            result = embedding_manager.embed_directory()
            logger.info(f"✓ Initial embedding complete: {result.get('total_files', 0)} files processed")
        except Exception as embed_error:
            logger.warning(f"Initial embedding skipped or failed: {str(embed_error)}")
        
        # Initialize file watcher for automatic re-embedding on structure changes
        try:
            base_path = _get_base_path()
            if base_path.exists():
                file_watcher = FileWatcher(
                    watch_path=str(base_path),
                    embedding_manager=embedding_manager,
                    retriever_manager=retriever_manager,
                    metadata_tracker=metadata_tracker
                )
                file_watcher.start()
                logger.info(f"✓ File watcher started on {base_path} - monitoring structure for changes")
            else:
                logger.warning(f"File watcher not started - base path does not exist: {base_path}")
        except Exception as watcher_error:
            logger.warning(f"File watcher initialization failed: {str(watcher_error)}")
        
        logger.info("✓ Backend startup complete - ready to accept requests")
        
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


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint for material pricing queries.
    Uses full pipeline: verification -> retrieval -> scoring -> ranking -> answer generation.
    Returns a stream of status updates and the final result.
    """
    # Validate request
    if not request.query or len(request.query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if len(request.query) > 2000:
        raise HTTPException(status_code=400, detail="Query too long (max 2000 characters)")
    
    # Verify managers are initialized (allow operation without prompt_verifier)
    if not all([embedding_manager, retriever_manager, llm_manager, ranking_engine, reranking_engine]):
        raise HTTPException(status_code=503, detail="Service not fully initialized")

    async def event_generator():
        try:
            logger.info(f"Processing query: {request.query[:100]}...")
            
            # Step 0: Verify and fix prompt (skip if verifier not available)
            fixed_query = request.query
            detected_projects = []
            is_domain_relevant = True
            
            if prompt_verifier:
                yield json.dumps({"type": "status", "stage": "verification", "message": "Verifying and cleaning prompt..."}) + "\n"
                await asyncio.sleep(0.1) # Tiny yield to ensure frontend receives it
                
                verification_result = prompt_verifier.verify_and_fix(request.query)
                
                if not verification_result['is_valid']:
                    error_msg = verification_result['error']
                    yield json.dumps({"type": "error", "message": error_msg}) + "\n"
                    return

                fixed_query = verification_result['fixed_query']
                detected_projects = verification_result.get('detected_projects', [])
                is_domain_relevant = verification_result.get('is_domain_relevant', True)
                
                # Check if query is about the material/project domain
                if not is_domain_relevant:
                    response_data = {
                        "response": "I'm specialized in answering questions about material pricing and project information in this database. Feel free to ask me about:\n\n• Material prices (concrete, wood, metal, stone)\n• Project details and specifications\n• Supplier information\n• Material availability and lead times\n• Comparisons between different materials\n\nHow can I help you with your project or material needs?",
                        "sources": [],
                        "web_search_used": False
                    }
                    yield json.dumps({"type": "result", "data": response_data}) + "\n"
                    return
                
                if verification_result['changes_made']:
                    changes = ", ".join(verification_result['changes_made'])
                    yield json.dumps({
                        "type": "status", 
                        "stage": "verification_complete", 
                        "message": f"Fixed prompt: {fixed_query}",
                        "details": changes
                    }) + "\n"
                    logger.info(f"Fixed: '{fixed_query}'")

            # Step 1: Retrieval
            yield json.dumps({"type": "status", "stage": "retrieval", "message": "Searching knowledge base..."}) + "\n"
            
            # Extract material categories from prompt_verifier if available
            detected_categories = []
            if prompt_verifier:
                detected_categories = [c for c in prompt_verifier.material_categories if c in fixed_query.lower()]
            
            retrieved = retriever_manager.retrieve(
                query=fixed_query, 
                top_k=10,
                mentioned_projects=detected_projects,
                mentioned_categories=detected_categories
            )
            
            chunks = []
            vector_scores = []
            
            if retrieved:
                for r in retrieved:
                    chunks.append({
                        "text": r.get("content", ""),
                        "metadata": {
                            "file_path": r.get("file_path", "unknown"),
                            "project_name": r.get("project_name", "unknown"), # Ensure project_name is passed
                            "chunk_index": r.get("chunk_index", 0),
                            "semantic_type": "general"
                        }
                    })
                    vector_scores.append(r.get("score", 0))

            # Step 2: Ranking
            yield json.dumps({"type": "status", "stage": "ranking", "message": f"Analyzing {len(chunks)} documents..."}) + "\n"
            
            ranked = []
            if chunks:
                # Pass the first detected project for primary scoring boost
                primary_project = detected_projects[0] if detected_projects else None
                ranked = ranking_engine.rank_results(
                    chunks=chunks,
                    vector_scores=vector_scores,
                    query_project=primary_project,
                    top_k=10,
                    min_confidence="low"
                )

            # Step 3: Re-ranking
            yield json.dumps({"type": "status", "stage": "reranking", "message": "Refining results for relevance..."}) + "\n"
            
            reranked = []
            if ranked:
                # Update reranking to handle project filtering
                reranked = reranking_engine.rerank_with_relevance_scoring(
                    query=fixed_query,
                    chunks=ranked,
                    target_projects=detected_projects,
                    top_k=5
                )
                
                if reranked and reranking_engine.should_confidence_be_reduced(fixed_query, reranked[0]):
                    logger.warning("Re-ranking detected weak match, reducing confidence")
                    for chunk in reranked:
                        chunk["scores"]["confidence"] = "low"

            # Guard check
            if not reranked and not request.use_web_search:
                response_data = {
                    "response": "I couldn't find any specific information matching your query in the Atlas database. Please try broadening your search or enabling web search for current market data.",
                    "sources": [],
                    "web_search_used": False
                }
                yield json.dumps({"type": "result", "data": response_data}) + "\n"
                return

            # Step 4: Generation
            yield json.dumps({"type": "status", "stage": "generation", "message": "Synthesizing final answer..."}) + "\n"
            
            answer_result = llm_manager.generate_answer(
                query=fixed_query,
                retrieved_chunks=reranked,
                use_web_search=request.use_web_search
            )

            # Format chunks for response (similar to original code)
            unique_sources = {}
            for chunk in ranking_engine.format_batch(reranked):
                fpath = chunk["source"]["file_path"].replace('\\', '/')
                score = chunk["relevance"]["score"]
                
                if fpath not in unique_sources or score > unique_sources[fpath].relevance_score:
                    unique_sources[fpath] = SourceCitation(
                        file_path=fpath,
                        content_snippet=chunk["text"][:200] + "...",
                        relevance_score=score
                    )
            
            sources = list(unique_sources.values())
            
            if "web_sources" in answer_result:
                for ws in answer_result["web_sources"]:
                    sources.append(
                        SourceCitation(
                            file_path=ws["url"] if ws["url"] else f"Web: {ws['title']}",
                            content_snippet=ws["content"],
                            relevance_score=0.95
                        )
                    )

            # Filter negative responses
            negative_keywords = ["don't have that information", "couldn't find any specific information", "information not found"]
            if any(keyword in answer_result["answer"].lower() for keyword in negative_keywords):
                sources = []

            final_response = {
                "response": answer_result["answer"],
                "sources": [s.model_dump() for s in sources], # Convert Pydantic models to dict
                "web_search_used": answer_result["web_search_used"]
            }
            
            yield json.dumps({"type": "result", "data": final_response}) + "\n"
            
        except Exception as e:
            logger.error(f"Stream error: {str(e)}", exc_info=True)
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")




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
        # Clear collection first to remove old/inconsistent path formats
        embedding_manager.clear_collection()
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


def _get_base_path():
    """Helper to get the consistent base path for materials."""
    data_path = Path(os.getenv('DATA_PATH', './data'))
    materials_path = data_path / 'materials'
    
    # Check if materials directory exists within data_path
    if materials_path.exists() and materials_path.is_dir():
        return materials_path
    return data_path


@app.get("/api/file-structure")
async def get_file_structure():
    """
    Get the file structure of the materials data directory.
    
    Returns:
        list: Hierarchical file structure
    """
    try:
        base_path = _get_base_path()
        
        # Build the tree structure
        file_structure = build_file_tree(str(base_path))
        return file_structure
        
    except Exception as e:
        logger.error(f"Error getting file structure: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving file structure")


@app.get("/api/download")
async def download_file(path: str):
    """
    Download a file from the materials directory.
    Supports smart path resolution by checking both the materials subfolder and root data.
    """
    return await _serve_file(path, as_attachment=True)


@app.get("/api/view")
async def view_file(path: str):
    """
    View a file inline with proper content types for PDF, CSV, etc.
    """
    return await _serve_file(path, as_attachment=False)


async def _serve_file(path: str, as_attachment: bool = True):
    """Helper to serve files with correct headers and encoding."""
    try:
        data_path = _get_base_path().resolve()
        
        # Normalize slashes and remove leading slashes
        clean_path = path.replace('\\', '/').lstrip('/')
        
        # Candidate paths to try
        candidates = []
        
        # Case 1: AI gives path like "materials/projectAcme/..." 
        # but data_path IS already ".../materials"
        if clean_path.startswith('materials/'):
            sub_path = clean_path[len('materials/'):]
            candidates.append(data_path / sub_path)
            
        # Case 2: Standard relative path
        candidates.append(data_path / clean_path)
        
        # Case 3: Absolute or relative to data root
        data_root = data_path.parent if data_path.name == 'materials' else data_path
        candidates.append(data_root / clean_path)

        full_path = None
        for cand in candidates:
            try:
                resolved = cand.resolve()
                if resolved.exists() and resolved.is_file():
                    # Security check: ensure it's under the project's data directory
                    # We'll be a bit more permissive with the base check to allow data/ or backend/data/
                    if 'data' in str(resolved).lower():
                        full_path = resolved
                        break
            except Exception:
                continue
        
        if not full_path:
            logger.error(f"File not found. Path: {path}, Clean: {clean_path}, Candidates: {[str(c) for c in candidates]}")
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        
        # Determine media type based on extension
        ext = full_path.suffix.lower()
        media_types = {
            '.pdf': 'application/pdf',
            '.csv': 'text/csv; charset=utf-8',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.txt': 'text/plain; charset=utf-8',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
        }
        media_type = media_types.get(ext, 'application/octet-stream')
        
        content_disposition = 'attachment' if as_attachment else 'inline'
        
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type=media_type,
            headers={
                "Content-Disposition": f"{content_disposition}; filename=\"{full_path.name}\""
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving file {path}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")


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
        log_level="info",
        workers=1,  # 1 worker for 700MB RAM constraint
        timeout_keep_alive=5,  # Drop idle connections quickly
        timeout_graceful_shutdown=10  # Graceful shutdown timeout
    )
