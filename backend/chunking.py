"""
Chunking & Semantic Units
Splits extracted content into semantic chunks with full metadata.
"""

import logging
import hashlib
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        raise NotImplementedError


class SimpleChunking(ChunkingStrategy):
    """Simple fixed-size chunking strategy."""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        words = text.split()
        
        i = 0
        chunk_idx = 0
        while i < len(words):
            # Get chunk of words
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if chunk_text.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_idx,
                    "chunk_size": len(chunk_words),
                    "semantic_type": self._infer_type(chunk_text)
                })
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
            
            chunk_idx += 1
            i += self.chunk_size - self.overlap
        
        return chunks
    
    @staticmethod
    def _infer_type(text: str) -> str:
        """Infer semantic type from text content."""
        text_lower = text.lower()
        
        # Price detection
        if any(word in text_lower for word in ['price', 'cost', 'fee', 'rate', '$', '€', '¥']):
            return "price"
        
        # Specification detection
        if any(word in text_lower for word in ['spec', 'specification', 'dimension', 'size', 'weight', 'property']):
            return "specification"
        
        # Description detection
        if any(word in text_lower for word in ['description', 'about', 'detail', 'information']):
            return "description"
        
        return "general"


class SemanticChunking(ChunkingStrategy):
    """Semantic chunking based on content type."""
    
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split into semantic chunks based on file type and content."""
        file_type = metadata.get("file_type", "text")
        
        if file_type == "csv":
            return self._chunk_csv(text, metadata)
        elif file_type == "xlsx":
            return self._chunk_xlsx(text, metadata)
        elif file_type == "pdf":
            return self._chunk_pdf(text, metadata)
        else:
            return self._chunk_default(text, metadata)
    
    @staticmethod
    def _chunk_csv(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Each CSV row is a single chunk."""
        # CSV content is already per-row from content_extraction
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": 0,
            "semantic_type": "tabular",
            "chunk_size": 1
        })
        
        return [{
            "text": text,
            "metadata": chunk_metadata
        }]
    
    @staticmethod
    def _chunk_xlsx(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Each XLSX row is a single chunk."""
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": 0,
            "semantic_type": "tabular",
            "chunk_size": 1
        })
        
        return [{
            "text": text,
            "metadata": chunk_metadata
        }]
    
    @staticmethod
    def _chunk_pdf(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Each PDF paragraph is a single chunk."""
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            "chunk_index": 0,
            "semantic_type": SimpleChunking._infer_type(text),
            "chunk_size": len(text.split())
        })
        
        return [{
            "text": text,
            "metadata": chunk_metadata
        }]
    
    @staticmethod
    def _chunk_default(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Default chunking for unknown types."""
        chunker = SimpleChunking(chunk_size=512, overlap=50)
        return chunker.chunk(text, metadata)


class ChunkManager:
    """Manages chunking operations and metadata enrichment."""
    
    def __init__(self, strategy: ChunkingStrategy = None):
        self.strategy = strategy or SemanticChunking()
    
    def chunk_file(self, file_path: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process file content blocks into final chunks with all metadata.
        
        Args:
            file_path: Path to source file
            blocks: Content blocks from content_extraction
            
        Returns:
            List of chunks with full metadata
        """
        all_chunks = []
        
        # Extract path metadata
        path_obj = Path(file_path)
        path_parts = path_obj.parts
        
        # Infer project and material from path
        project = material = None
        if len(path_parts) >= 2:
            project = path_parts[0]
            material = path_parts[1] if len(path_parts) >= 2 else None
        
        # Get file metadata
        stat = path_obj.stat()
        file_hash = self._compute_hash(file_path)
        
        for block in blocks:
            # Chunk the content
            chunks = self.strategy.chunk(block["text"], block["metadata"])
            
            # Enrich with file-level metadata
            for chunk in chunks:
                chunk["metadata"].update({
                    "file_path": str(file_path),
                    "project_name": project,
                    "material": material,
                    "file_type": block["metadata"].get("file_type"),
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "version_hash": file_hash,
                    "file_size": stat.st_size
                })
                
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {file_path}")
        return all_chunks
    
    def chunk_batch(self, file_blocks: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Process multiple files into chunks.
        
        Args:
            file_blocks: Dict mapping file path to content blocks
            
        Returns:
            List of all chunks from all files
        """
        all_chunks = []
        for file_path, blocks in file_blocks.items():
            chunks = self.chunk_file(file_path, blocks)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    @staticmethod
    def _compute_hash(file_path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
        
        return sha256_hash.hexdigest()
