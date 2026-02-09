"""
Takes user query, converts to embedding
Searches ChromaDB using cosine similarity
Returns top-k most relevant documents with scores
Relevance scores show quality of match (0-100%)
"""

import logging
from typing import Optional
from enum import Enum
from embedding import EmbeddingManager

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Enumeration of query intent types for targeted retrieval strategies."""
    SPECIFICATION = "specification"      # Specific material/project query (highest precision needed)
    COMPARISON = "comparison"            # Comparing multiple items across projects
    CATEGORY = "category"                # Asking about a specific material category
    GENERAL = "general"                  # General information query (lowest precision priority)


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
    
    def _detect_query_intent(self, query_lower: str) -> QueryIntent:
        """
        Detect the intent of the user's query to optimize retrieval strategy.
        
        Args:
            query_lower: Lowercase query string
            
        Returns:
            QueryIntent enum indicating the type of query
        """
        # Count mentioned items
        mentioned_projects = len(self._extract_mentioned_items(query_lower, ['projectacme', 'projectfacebook']))
        mentioned_categories = len(self._extract_mentioned_items(query_lower, ['concrete', 'metal', 'stone', 'wood']))
        
        # Comparison keywords
        comparison_keywords = ['compare', 'vs', 'versus', 'difference', 'between', 'rather', 'instead']
        has_comparison_keyword = any(keyword in query_lower for keyword in comparison_keywords)
        
        # Specification keywords
        spec_keywords = ['what', 'price', 'cost', 'how much', 'specific', 'particular']
        has_spec_keyword = any(keyword in query_lower for keyword in spec_keywords)
        
        # Determination logic
        # If comparing multiple categories or has comparison keywords -> COMPARISON
        if (mentioned_categories >= 2) or has_comparison_keyword:
            logger.debug(f"Detected COMPARISON intent: {mentioned_categories} categories or comparison keyword")
            return QueryIntent.COMPARISON
        
        # If has project AND category specifics -> SPECIFICATION
        if mentioned_projects >= 1 and mentioned_categories >= 1 and has_spec_keyword:
            logger.debug(f"Detected SPECIFICATION intent: project + category + specification keyword")
            return QueryIntent.SPECIFICATION
        
        # If has category but no specific project -> CATEGORY
        if mentioned_categories >= 1 and mentioned_projects == 0:
            logger.debug(f"Detected CATEGORY intent: category without specific project")
            return QueryIntent.CATEGORY
        
        # Default to GENERAL
        logger.debug("Detected GENERAL intent: no specific markers")
        return QueryIntent.GENERAL
    
    def _extract_mentioned_items(self, query_lower: str, items: list[str]) -> list[str]:
        """
        Extract mentioned items (projects, categories) from the query.
        Handles typos and spaces in words (e.g., "wo od" matches "wood").
        
        Args:
            query_lower: Lowercase query string
            items: List of items to search for (lowercase)
            
        Returns:
            List of mentioned items found in the query
        """
        mentioned = []
        query_normalized = query_lower.replace(" ", "")
        
        for item in items:
            # Try exact match first (more specific)
            if item in query_lower:
                mentioned.append(item)
            # Then try with flexible spaces (catches "wo od", "m etal", etc)
            elif item.replace(" ", "") in query_normalized:
                if item not in mentioned:  # Avoid duplicates
                    mentioned.append(item)
        
        return mentioned
    
    def _extract_path_project_and_category(self, file_path: str) -> tuple[str, str]:
        """
        Extract project name and category from file path.
        Expected path format: projectXxx/category/filename
        
        Args:
            file_path: File path (e.g., projectAcme/wood/wood_spec.pdf)
            
        Returns:
            Tuple of (project_name, category) - both lowercase, empty string if not found
        """
        path_lower = file_path.lower()
        project = ""
        category = ""
        
        # Extract project name
        if "projectacme" in path_lower:
            project = "projectacme"
        elif "projectfacebook" in path_lower:
            project = "projectfacebook"
        
        # Extract category
        categories = ['concrete', 'metal', 'stone', 'wood']
        for cat in categories:
            if cat in path_lower:
                category = cat
                break
        
        return project, category
    
    def _apply_relevance_weights(
        self,
        base_score: float,
        file_path: str,
        mentioned_projects: list[str],
        mentioned_categories: list[str],
        query_intent: QueryIntent = QueryIntent.GENERAL
    ) -> float:
        """
        Apply weighted boost to the base similarity score based on metadata matches.
        Weights are adjusted based on detected query intent for targeted ranking.
        
        Intent-based weight strategies:
        - SPECIFICATION: High project/category weights (3.0/3.0) - need exact match
        - COMPARISON: High category weight (2.0), low project weight (1.2) - comparing items
        - CATEGORY: High category weight (3.0), very low project weight (1.0) - focused on material type
        - GENERAL: Moderate weights (2.0/1.5) - semantic relevance is primary
        
        Args:
            base_score: Original similarity score (0-1)
            file_path: File path containing project and category info
            mentioned_projects: List of project names mentioned in query
            mentioned_categories: List of material categories mentioned in query
            query_intent: Detected intent of the query
            
        Returns:
            Boosted relevance score
        """
        # Extract project and category from the file path
        file_project, file_category = self._extract_path_project_and_category(file_path)
        
        boosted_score = base_score
        
        # SET WEIGHTS BASED ON QUERY INTENT
        if query_intent == QueryIntent.SPECIFICATION:
            # User asked for specific project + material: maximize exact matches
            PROJECT_WEIGHT = 3.0
            CATEGORY_WEIGHT = 3.0
            logger.debug(f"Using SPECIFICATION weights: project={PROJECT_WEIGHT}, category={CATEGORY_WEIGHT}")
        elif query_intent == QueryIntent.COMPARISON:
            # User comparing materials: reduce project emphasis, emphasize categories
            PROJECT_WEIGHT = 1.2
            CATEGORY_WEIGHT = 2.0
            logger.debug(f"Using COMPARISON weights: project={PROJECT_WEIGHT}, category={CATEGORY_WEIGHT}")
        elif query_intent == QueryIntent.CATEGORY:
            # User focused on material type: maximize category, minimize project
            PROJECT_WEIGHT = 1.0
            CATEGORY_WEIGHT = 3.0
            logger.debug(f"Using CATEGORY weights: project={PROJECT_WEIGHT}, category={CATEGORY_WEIGHT}")
        else:
            # GENERAL: balanced approach
            PROJECT_WEIGHT = 2.0
            CATEGORY_WEIGHT = 1.5
            logger.debug(f"Using GENERAL weights: project={PROJECT_WEIGHT}, category={CATEGORY_WEIGHT}")
        
        # Apply project weight if project is mentioned in query AND matches file path
        if file_project and file_project in mentioned_projects:
            boosted_score *= PROJECT_WEIGHT
            logger.debug(f"Applied project weight ({PROJECT_WEIGHT}x) for {file_project}: new score {boosted_score}")
        
        # Apply category weight if category is mentioned in query AND matches file path
        if file_category and file_category in mentioned_categories:
            boosted_score *= CATEGORY_WEIGHT
            logger.debug(f"Applied category weight ({CATEGORY_WEIGHT}x) for {file_category}: new score {boosted_score}")
        
        # Return boosted score without capping - normalization happens after all scores are calculated
        # This ensures proper differentiation between results:
        # - Only the best matching result(s) will reach the highest normalized score
        # - Other results will be proportionally lower
        logger.debug(f"Final boosted score for {file_path}: {boosted_score} (before normalization)")
        
        return boosted_score
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        distance_threshold: Optional[float] = None
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a query.
        Applies weighted boosts for project name and material category matches.
        
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
            
            # Detect query intent for weighted strategy optimization
            query_lower = query.lower()
            query_intent = self._detect_query_intent(query_lower)
            
            # Perform similarity search - get more results to apply weighting
            logger.debug(f"Searching for top {top_k * 2} similar documents for re-ranking...")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 2, 20),  # Get more results for weighting
                include=["documents", "metadatas", "distances"]
            )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                logger.info("No documents found matching query")
                return []
            
            # Extract project name and categories from query for weighting
            mentioned_projects = self._extract_mentioned_items(query_lower, ['projectacme', 'projectfacebook'])
            mentioned_categories = self._extract_mentioned_items(query_lower, ['concrete', 'metal', 'stone', 'wood'])
            
            logger.debug(f"Mentioned projects: {mentioned_projects}, categories: {mentioned_categories}")
            logger.debug(f"Query intent: {query_intent.value}")
            
            # Process and format results with weighted scores
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
                
                # Apply weighted boost based on project and category matches
                # Weights are adjusted based on detected query intent
                boosted_score = self._apply_relevance_weights(
                    similarity_score,
                    metadata.get('file_path', ''),
                    mentioned_projects,
                    mentioned_categories,
                    query_intent
                )
                
                retrieved_docs.append({
                    'content': doc,
                    'file_path': metadata.get('file_path', 'unknown'),
                    'chunk_index': metadata.get('chunk_index', '0'),
                    'score': round(boosted_score, 4),
                    'distance': round(distance, 4),
                    'original_score': round(similarity_score, 4)
                })
            
            # Sort by boosted score
            retrieved_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # Normalize scores to 0-1 range based on max score
            # This ensures the best match(es) get the highest score, others are proportionally lower
            if retrieved_docs:
                max_score = max(doc['score'] for doc in retrieved_docs)
                if max_score > 0:
                    for doc in retrieved_docs:
                        normalized_score = doc['score'] / max_score
                        doc['score'] = round(normalized_score, 4)
                        logger.debug(f"Normalized score for {doc['file_path']}: {normalized_score}")
            
            # Return top_k results
            retrieved_docs = retrieved_docs[:top_k]
            
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
