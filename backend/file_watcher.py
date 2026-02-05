"""
Monitors data/materials/ folder continuously
Automatically embeds new .txt files
Allows hot-adding materials without restart
"""

import logging
import threading
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent

logger = logging.getLogger(__name__)


class MaterialFileEventHandler(FileSystemEventHandler):
    """
    Handles file system events for material files.
    Triggers re-embedding when files are created or modified.
    """
    
    def __init__(self, embedding_manager, retriever_manager):
        """
        Initialize event handler with manager instances.
        
        Args:
            embedding_manager: EmbeddingManager instance
            retriever_manager: RetrieverManager instance
        """
        super().__init__()
        self.embedding_manager = embedding_manager
        self.retriever_manager = retriever_manager
        self.last_event_time = {}
        self.debounce_delay = 2  # Seconds to wait before processing
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory:
            return
        
        if not event.src_path.endswith('.txt'):
            return
        
        logger.info(f"New file detected: {event.src_path}")
        self._process_file_change(event.src_path, "created")
    
    def on_modified(self, event):
        """Handle file modification events with debouncing."""
        if event.is_directory:
            return
        
        if not event.src_path.endswith('.txt'):
            return
        
        # Debounce: avoid processing the same file too frequently
        # (some editors fire multiple events for single save)
        current_time = time.time()
        last_time = self.last_event_time.get(event.src_path, 0)
        
        if current_time - last_time < self.debounce_delay:
            logger.debug(f"Debouncing modified event for: {event.src_path}")
            return
        
        self.last_event_time[event.src_path] = current_time
        
        logger.info(f"File modified: {event.src_path}")
        self._process_file_change(event.src_path, "modified")
    
    def _process_file_change(self, file_path: str, event_type: str):
        """
        Process a file change by re-embedding it.
        
        Args:
            file_path: Path to the modified file
            event_type: Type of change (created/modified)
        """
        try:
            # Wait a moment for file to be fully written
            time.sleep(0.5)
            
            # Verify file still exists
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File no longer exists: {file_path}")
                return
            
            # Re-embed the file
            chunks = self.embedding_manager.embed_file(file_path)
            
            if chunks > 0:
                logger.info(
                    f"✓ File {event_type} processed successfully. "
                    f"Chunks created: {chunks}"
                )
            else:
                logger.warning(f"No chunks created for {event_type} file: {file_path}")
            
        except Exception as e:
            logger.error(
                f"Error processing {event_type} event for {file_path}: {str(e)}",
                exc_info=True
            )


class FileWatcher:
    """
    Watches material files directory for changes and triggers re-embedding.
    Runs as a background daemon thread.
    """
    
    def __init__(self, watch_path: str, embedding_manager, retriever_manager):
        """
        Initialize file watcher.
        
        Args:
            watch_path: Path to watch for changes
            embedding_manager: EmbeddingManager instance
            retriever_manager: RetrieverManager instance
        """
        self.watch_path = Path(watch_path)
        self.embedding_manager = embedding_manager
        self.retriever_manager = retriever_manager
        
        # Create observer
        self.observer = Observer()
        self.event_handler = MaterialFileEventHandler(
            embedding_manager,
            retriever_manager
        )
        
        # Track thread state
        self._running = False
        self._thread = None
    
    def start(self):
        """Start watching the directory."""
        if self._running:
            logger.warning("File watcher already running")
            return
        
        try:
            # Ensure watch path exists
            self.watch_path.mkdir(parents=True, exist_ok=True)
            
            # Schedule the event handler
            self.observer.schedule(
                self.event_handler,
                str(self.watch_path),
                recursive=True
            )
            
            # Start observer
            self.observer.start()
            self._running = True
            
            logger.info(f"✓ File watcher started for: {self.watch_path}")
            
        except Exception as e:
            logger.error(f"Error starting file watcher: {str(e)}", exc_info=True)
            raise
    
    def stop(self):
        """Stop watching the directory."""
        if not self._running:
            logger.debug("File watcher not running")
            return
        
        try:
            self.observer.stop()
            self.observer.join(timeout=5)
            self._running = False
            logger.info("✓ File watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping file watcher: {str(e)}", exc_info=True)
    
    def is_alive(self) -> bool:
        """
        Check if the file watcher is running.
        
        Returns:
            True if observer is running, False otherwise
        """
        return self._running and self.observer.is_alive()
    
    def __del__(self):
        """Cleanup on object deletion."""
        try:
            self.stop()
        except:
            pass
