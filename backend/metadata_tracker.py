"""
Metadata Tracker
Tracks file metadata including versions, hashes, and timestamps for incremental indexing.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataStore:
    """Stores and retrieves file metadata for change detection."""
    
    def __init__(self, store_path: str = "data/metadata.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict[str, Dict[str, Any]] = self._load()
    
    def _load(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata store: {e}")
        return {}
    
    def _save(self):
        """Save metadata to disk."""
        try:
            with open(self.store_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata store: {e}")
    
    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a file."""
        return self.metadata.get(str(file_path))
    
    def set(self, file_path: str, metadata: Dict[str, Any]):
        """Store metadata for a file."""
        self.metadata[str(file_path)] = metadata
        self._save()
    
    def update(self, file_path: str, **kwargs):
        """Update metadata fields for a file."""
        file_path = str(file_path)
        if file_path not in self.metadata:
            self.metadata[file_path] = {}
        
        self.metadata[file_path].update(kwargs)
        self._save()
    
    def delete(self, file_path: str):
        """Delete metadata for a file."""
        file_path = str(file_path)
        if file_path in self.metadata:
            del self.metadata[file_path]
            self._save()
    
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all tracked files and their metadata."""
        return self.metadata.copy()


class FileMetadataTracker:
    """Tracks file changes and metadata for incremental indexing."""
    
    def __init__(self, metadata_store: MetadataStore = None):
        self.store = metadata_store or MetadataStore()
    
    def track_file(self, file_path: str, file_hash: str, content_blocks: int = 0):
        """
        Track a file with its metadata.
        
        Args:
            file_path: Path to file
            file_hash: SHA256 hash of file content
            content_blocks: Number of chunks created
        """
        path_obj = Path(file_path)
        stat = path_obj.stat()
        
        metadata = {
            "file_path": str(file_path),
            "file_hash": file_hash,
            "file_size": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "indexed_at": datetime.now().isoformat(),
            "content_blocks": content_blocks,
            "status": "indexed"
        }
        
        self.store.set(file_path, metadata)
        logger.info(f"Tracked file: {file_path}")
    
    def check_changes(self, file_path: str, current_hash: str) -> Dict[str, Any]:
        """
        Check if a file has changed since last indexing.
        
        Returns:
            {
                "changed": bool,
                "reason": str (if changed),
                "previous_hash": str,
                "previous_modified": str
            }
        """
        previous = self.store.get(file_path)
        
        if not previous:
            return {
                "changed": True,
                "reason": "not_indexed_before",
                "previous_hash": None,
                "previous_modified": None
            }
        
        # Check if file hash changed (content changed)
        if previous.get("file_hash") != current_hash:
            return {
                "changed": True,
                "reason": "content_changed",
                "previous_hash": previous.get("file_hash"),
                "previous_modified": previous.get("last_modified")
            }
        
        # Check if modification time changed
        path_obj = Path(file_path)
        stat = path_obj.stat()
        current_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        if previous.get("last_modified") != current_mtime:
            return {
                "changed": True,
                "reason": "timestamp_changed",
                "previous_hash": previous.get("file_hash"),
                "previous_modified": previous.get("last_modified")
            }
        
        return {
            "changed": False,
            "reason": "no_changes",
            "previous_hash": previous.get("file_hash"),
            "previous_modified": previous.get("last_modified")
        }
    
    def get_indexed_files(self) -> List[str]:
        """Get list of all indexed files."""
        return list(self.store.list_all().keys())
    
    def get_file_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get all metadata for a file."""
        return self.store.get(file_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get metadata tracking statistics."""
        all_metadata = self.store.list_all()
        
        return {
            "total_files_tracked": len(all_metadata),
            "total_size_bytes": sum(m.get("file_size", 0) for m in all_metadata.values()),
            "total_content_blocks": sum(m.get("content_blocks", 0) for m in all_metadata.values()),
            "indexed_files": [m.get("file_path") for m in all_metadata.values() if m.get("status") == "indexed"]
        }
    
    def cleanup_deleted_files(self, existing_files: List[str]) -> List[str]:
        """
        Remove metadata for files that no longer exist.
        
        Args:
            existing_files: List of files currently on disk
            
        Returns:
            List of files removed from metadata
        """
        existing_set = set(str(f) for f in existing_files)
        tracked = set(self.get_indexed_files())
        
        deleted_files = tracked - existing_set
        
        for file_path in deleted_files:
            self.store.delete(file_path)
            logger.info(f"Removed metadata for deleted file: {file_path}")
        
        return list(deleted_files)
