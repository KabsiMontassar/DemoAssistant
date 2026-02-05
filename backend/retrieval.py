"""
Takes user query, converts to embedding
Searches ChromaDB using cosine similarity
Returns top-k most relevant documents with scores
Relevance scores show quality of match (0-100%)
"""

import logging
from typing import Optional
from embedding import EmbeddingManager

logger = logging.getLogger(__name__)


class RetrieverManager:
    """
    Manages document retrieval using vector similarity search.
    Finds the most relevant documents for a given query.
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize retriever with an embedding manager instance.
        
        Args:
            embedding_manager: EmbeddingManager instance with initialized ChromaDB
        """
        self.embedding_manager = embedding_manager
        self.collection = embedding_manager.collection
        self.model = embedding_manager.model
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        distance_threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.
        
        Args:
            query: Query string (natural language question)
            top_k: Number of top results to return (default: 5)
            distance_threshold: Optional threshold for relevance filtering
            
        Returns:
            List of retrieved documents with metadata and scores
            
        Raises:
            ValueError: If query is empty or invalid
            Exception: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        try:
            # Generate query embedding
            logger.debug(f"Encoding query: {query[:100]}...")
            query_embedding = self.model.encode(query).tolist()
            
            # Perform similarity search
            logger.debug(f"Searching for top {top_k} similar documents...")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                logger.info("No documents found matching query")
                return []
            
            # Process and format results
            retrieved_docs = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (cosine distance)
                # ChromaDB returns distances; lower distance = higher similarity
                similarity_score = 1 - distance
                
                # Optional: filter by distance threshold
                if distance_threshold is not None and distance > distance_threshold:
                    logger.debug(
                        f"Filtering out document {i}: "
                        f"distance {distance} > threshold {distance_threshold}"
                    )
                    continue
                
                retrieved_docs.append({
                    'content': doc,
                    'file_path': metadata.get('file_path', 'unknown'),
                    'chunk_index': metadata.get('chunk_index', '0'),
                    'score': round(similarity_score, 4),
                    'distance': round(distance, 4)
                })
            
            logger.info(
                f"Retrieved {len(retrieved_docs)} documents "
                f"for query: {query[:50]}..."
            )
            
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}", exc_info=True)
            raise
    
    def retrieve_by_file(
        self,
        file_path: str,
        top_k: int = 10
    ) -> list[dict]:
        """
        Retrieve all documents from a specific file.
        
        Args:
            file_path: Path of the file to retrieve from
            top_k: Maximum number of documents to return
            
        Returns:
            List of documents from the specified file
        """
        try:
            logger.debug(f"Retrieving documents from file: {file_path}")
            
            results = self.collection.get(
                where={"file_path": file_path},
                limit=top_k,
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get('documents'):
                logger.warning(f"No documents found for file: {file_path}")
                return []
            
            docs = []
            for doc, metadata in zip(results['documents'], results['metadatas']):
                docs.append({
                    'content': doc,
                    'file_path': metadata.get('file_path', 'unknown'),
                    'chunk_index': metadata.get('chunk_index', '0'),
                    'score': 1.0  # No similarity score for file-based retrieval
                })
            
            logger.info(f"Retrieved {len(docs)} documents from {file_path}")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving by file: {str(e)}", exc_info=True)
            raise
    
    def retrieve_by_metadata(
        self,
        metadata_filter: dict,
        top_k: int = 10
    ) -> list[dict]:
        """
        Retrieve documents matching specific metadata criteria.
        
        Args:
            metadata_filter: Dictionary of metadata filters (ChromaDB where clause)
            top_k: Maximum number of documents to return
            
        Returns:
            List of documents matching the metadata criteria
        """
        try:
            logger.debug(f"Retrieving documents with filter: {metadata_filter}")
            
            results = self.collection.get(
                where=metadata_filter,
                limit=top_k,
                include=["documents", "metadatas"]
            )
            
            if not results or not results.get('documents'):
                logger.warning("No documents found matching metadata filter")
                return []
            
            docs = []
            for doc, metadata in zip(results['documents'], results['metadatas']):
                docs.append({
                    'content': doc,
                    'file_path': metadata.get('file_path', 'unknown'),
                    'chunk_index': metadata.get('chunk_index', '0'),
                    'score': 1.0
                })
            
            logger.info(f"Retrieved {len(docs)} documents matching filter")
            return docs
            
        except Exception as e:
            logger.error(f"Error retrieving by metadata: {str(e)}", exc_info=True)
            raise
    
    def get_stats(self) -> dict:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary containing system statistics
        """
        try:
            collection_stats = self.embedding_manager.get_collection_stats()
            
            # Get unique files
            all_results = self.collection.get(include=["metadatas"])
            unique_files = set()
            if all_results and all_results.get('metadatas'):
                for metadata in all_results['metadatas']:
                    unique_files.add(metadata.get('file_path', 'unknown'))
            
            return {
                **collection_stats,
                "unique_files": len(unique_files),
                "files": sorted(list(unique_files))
            }
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {"error": str(e)}
