"""
Uses sentence-transformers (all-MiniLM-L6-v2 model)
Converts text into 384-dimensional vectors
Stores vectors in ChromaDB (vector database)
Automatically chunks long documents
"""

import os
import logging
from pathlib import Path
from typing import Optional
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages document embedding and vector storage using ChromaDB and Sentence Transformers.
    Handles document processing, chunking, and embedding generation.
    """
    
    def __init__(self):
        """Initialize embedding manager with ChromaDB and Sentence Transformer model."""
        self.chroma_path = Path(os.getenv('CHROMA_PATH', './data/chroma_db'))
        self.data_path = Path(os.getenv('DATA_PATH', './data/materials'))
        self.chunk_size = 500  # Characters per chunk
        self.chunk_overlap = 100  # Overlap between chunks
        
        # Initialize ChromaDB client
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="material_pricing",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load embedding model (downloads on first run, ~400MB)
        logger.info("Loading Sentence Transformer model...")
        self.model = SentenceTransformer(
            os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
            device='cpu'  # Use CPU for demo (production could use GPU)
        )
        logger.info("✓ Embedding model loaded")
    
    def _chunk_text(self, text: str, file_path: str) -> list[dict]:
        """
        Split text into overlapping chunks for better embedding quality.
        
        Args:
            text: Full document text
            file_path: Source file path (for metadata)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        
        # Split into paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk_size, save current chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'file_path': file_path,
                    'length': len(current_chunk)
                })
                
                # Create overlap by including last part of previous chunk
                current_chunk = current_chunk[-(self.chunk_overlap):] + "\n\n" + paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'file_path': file_path,
                'length': len(current_chunk)
            })
        
        return chunks
    
    def embed_file(self, file_path: str) -> int:
        """
        Process a single file: read, chunk, embed, and store in vector DB.
        
        Args:
            file_path: Full path to the file to embed
            
        Returns:
            Number of chunks successfully embedded
            
        Raises:
            ValueError: If file doesn't exist or cannot be read
            Exception: If embedding or storage fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        if not file_path.suffix == '.txt':
            logger.warning(f"Skipping non-txt file: {file_path}")
            return 0
        
        try:
            logger.info(f"Embedding file: {file_path.name}")
            
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"File is empty: {file_path}")
                return 0
            
            # Get relative path for cleaner storage
            try:
                relative_path = file_path.relative_to(self.data_path)
            except ValueError:
                relative_path = file_path.name
            
            # Remove old embeddings for this file (handle re-embedding)
            try:
                self.collection.delete(
                    where={"file_path": str(relative_path)}
                )
                logger.info(f"Cleared old embeddings for: {relative_path}")
            except Exception as e:
                # File might not exist yet, which is fine
                logger.debug(f"No previous embeddings to clear: {e}")
            
            # Chunk the document
            chunks = self._chunk_text(content, str(relative_path))
            
            if not chunks:
                logger.warning(f"No chunks created from: {file_path}")
                return 0
            
            # Generate embeddings
            texts = [chunk['text'] for chunk in chunks]
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
            
            # Prepare documents for storage
            ids = [f"{relative_path}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "file_path": chunk['file_path'],
                    "chunk_index": str(i),
                    "chunk_length": str(chunk['length'])
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"✓ Successfully embedded {len(chunks)} chunks from {file_path.name}")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error embedding file {file_path}: {str(e)}", exc_info=True)
            raise
    
    def embed_directory(self, directory: Optional[str] = None) -> dict:
        """
        Recursively embed all .txt files in a directory.
        
        Args:
            directory: Directory path (defaults to DATA_PATH)
            
        Returns:
            Dictionary with embedding statistics
        """
        if directory is None:
            directory = self.data_path
        else:
            directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return {"status": "error", "message": f"Directory not found: {directory}"}
        
        logger.info(f"Starting directory embedding: {directory}")
        
        files = list(directory.rglob('*.txt'))
        if not files:
            logger.warning(f"No .txt files found in {directory}")
            return {
                "status": "success",
                "message": "No files to process",
                "total_files": 0,
                "total_chunks": 0
            }
        
        total_chunks = 0
        successful_files = 0
        failed_files = 0
        
        for file_path in sorted(files):
            try:
                chunks = self.embed_file(str(file_path))
                total_chunks += chunks
                successful_files += 1
            except Exception as e:
                logger.error(f"Failed to embed {file_path}: {str(e)}")
                failed_files += 1
        
        logger.info(
            f"✓ Directory embedding complete. "
            f"Files: {successful_files} successful, {failed_files} failed. "
            f"Total chunks: {total_chunks}"
        )
        
        return {
            "status": "success",
            "total_files": len(files),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_chunks": total_chunks
        }
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_collection(self):
        """
        Clear all embeddings from the collection (useful for reset/testing).
        WARNING: This is a destructive operation.
        """
        try:
            # Get all documents and delete them
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.warning(f"Cleared {len(results['ids'])} documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}", exc_info=True)
            raise
