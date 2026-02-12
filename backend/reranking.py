"""
Re-ranking Layer
Uses LLM-based re-ranking to improve result quality and reduce hallucination.
Implements cross-encoder style re-ranking before final answer generation.
"""

import logging
import re
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Project names in the system
PROJECT_NAMES = {
    'apexindustries', 'ecobuildersintel', 'nexgenstructures', 'omniconstructinc',
    'pinnacleconstructio', 'projectacme', 'projectfacebook', 'quantumbuildsystems',
    'skyriseenterprises', 'techvisioncorp', 'titanmaterialsgroup', 'venturebuildolutions'
}


class ReRankingEngine:
    """Re-ranks top results using semantic similarity to original query."""
    
    def __init__(self, llm_manager=None):
        """
        Initialize re-ranking engine.
        
        Args:
            llm_manager: LLMManager instance for LLM-based re-ranking
        """
        self.llm_manager = llm_manager
        self.project_names = PROJECT_NAMES
    
    def _extract_project_name(self, query: str) -> Optional[str]:
        """
        Extract project name from query if mentioned.
        Note: PromptVerifier now handles this better, but we keep this as a secondary check.
        """
        query_lower = query.lower()
        
        # Check if any project name appears in query as a whole word
        for project in self.project_names:
            pattern = r'\b' + re.escape(project.lower()) + r'\b'
            if re.search(pattern, query_lower):
                return project
        
        return None
    
    def rerank_with_query_similarity(self, 
                                    query: str,
                                    chunks: List[Dict[str, Any]],
                                    target_projects: List[str] = None) -> List[Dict[str, Any]]:
        """
        Re-rank chunks by computing semantic similarity score to original query.
        
        Args:
            query: Original user query (normalized)
            chunks: List of chunks with scores
            target_projects: List of project names explicitly mentioned in query
            
        Returns:
            Re-ranked chunks with new relevance scores
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        
        # Extract important terms (remove common words)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'of', 'for', 'and', 'or', 'in', 'on', 'at', 'to', 'from', 'what', 'how', 'when', 'where', 'why', 'which', 'who'}
        important_terms = [t for t in query_terms if t not in stopwords and len(t) > 2]
        
        # Use provided projects or extract if not provided
        projects = target_projects if target_projects is not None else [self._extract_project_name(query)]
        projects = [p for p in projects if p] # Remove None
        
        reranked = []
        
        for chunk in chunks:
            text = chunk.get("text", "").lower()
            file_path = chunk.get("metadata", {}).get("file_path", "").lower()
            metadata_project = chunk.get("metadata", {}).get("project_name", "").lower()
            
            # STRICT PROJECT FILTERING: If project mentioned in query, heavily penalize mismatches
            if projects:
                is_match = any(p.lower() in file_path or p.lower() == metadata_project for p in projects)
                if not is_match:
                    # Wrong project - give minimal score
                    logger.debug(f"Penalizing non-matching project: {file_path} (looking for {projects})")
                    chunk_copy = chunk.copy()
                    chunk_copy["rerank_score"] = 0.05  # Even lower score for wrong project
                    chunk_copy["scores"] = chunk.get("scores", {}).copy()
                    chunk_copy["scores"]["rerank_score"] = 0.05
                    reranked.append(chunk_copy)
                    continue
            
            # Count term matches
            matches = sum(1 for term in important_terms if term in text)
            term_score = min(1.0, matches / max(len(important_terms), 1)) * 0.3  # Weight: 30%
            
            # Check if chunk directly answers question terms
            query_type = self._infer_query_type(query_lower)
            content_score = self._score_content_relevance(text, query_type)  # Weight: 40%
            
            # Keep existing score
            existing_score = chunk.get("scores", {}).get("overall_score", 0.5)  # Weight: 30%
            
            # Combined re-ranking score
            rerank_score = (term_score * 0.3 + content_score * 0.4 + existing_score * 0.3)
            
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = rerank_score
            chunk_copy["scores"] = chunk.get("scores", {})
            chunk_copy["scores"]["rerank_score"] = rerank_score
            
            reranked.append(chunk_copy)
        
        # Sort by re-rank score (descending)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Filter out low-scoring mismatches if project was specified
        if projects:
            matching = [
                c for c in reranked 
                if any(p.lower() in c.get("metadata", {}).get("file_path", "").lower() or p.lower() == c.get("metadata", {}).get("project_name", "").lower() for p in projects)
            ]
            if matching:
                logger.info(f"Project filtering: keeping {len(matching)} matching results, filtering {len(reranked) - len(matching)} mismatches for {projects}")
                reranked = matching
        
        logger.debug(f"Re-ranked {len(reranked)} chunks for query: {query[:50]}")
        
        return reranked

        
        return reranked
    
    @staticmethod
    def _infer_query_type(query: str) -> str:
        """Infer the type of query to score content appropriately."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['price', 'cost', 'fee', 'rate', 'how much']):
            return "price"
        elif any(word in query_lower for word in ['spec', 'specification', 'dimension', 'size', 'what is']):
            return "specification"
        elif any(word in query_lower for word in ['how', 'why', 'when', 'where']):
            return "explanatory"
        else:
            return "general"
    
    @staticmethod
    def _score_content_relevance(text: str, query_type: str) -> float:
        """Score how relevant content is based on query type."""
        text_lower = text.lower()
        score = 0.3  # Base score
        
        # Price queries - boost if contains numbers/currency
        if query_type == "price":
            if any(char.isdigit() for char in text):
                score += 0.3  # Contains numbers
            if any(symbol in text for symbol in ['€', '$', '¥', 'EUR', 'USD', 'CHF']):
                score += 0.2  # Contains currency
            if any(word in text_lower for word in ['price', 'cost', 'fee', 'rate', 'per']):
                score += 0.2
        
        # Specification queries - boost if contains technical terms
        elif query_type == "specification":
            if any(word in text_lower for word in ['spec', 'specification', 'dimension', 'size', 'weight', 'material', 'property']):
                score += 0.3
            if any(char.isdigit() for char in text):
                score += 0.2
        
        # Explanatory queries - boost if has descriptive content
        elif query_type == "explanatory":
            # Longer text tends to be more explanatory
            if len(text.split()) > 10:
                score += 0.2
            if any(word in text_lower for word in ['because', 'due to', 'reason', 'therefore', 'thus', 'since']):
                score += 0.2
        
        return min(1.0, score)
    
    def rerank_with_relevance_scoring(self,
                                     query: str,
                                     chunks: List[Dict[str, Any]],
                                     target_projects: List[str] = None,
                                     top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Combined re-ranking using multiple signals.
        
        Args:
            query: Original query (already normalized/fixed by PromptVerifier)
            chunks: Chunks to re-rank
            target_projects: List of projects detected in query
            top_k: Return top K results after re-ranking
            
        Returns:
            Top K re-ranked chunks
        """
        # Apply query similarity re-ranking which handles project filtering
        reranked = self.rerank_with_query_similarity(query, chunks, target_projects=target_projects)
        
        # Return top K
        return reranked[:top_k]
    
    def should_confidence_be_reduced(self, 
                                    query: str,
                                    top_result: Dict[str, Any]) -> bool:
        """
        Determine if confidence should be reduced due to weak re-ranking score.
        
        Returns True if result quality is questionable.
        """
        rerank_score = top_result.get("rerank_score", 0)
        original_score = top_result.get("scores", {}).get("overall_score", 0)
        
        # Reduce confidence if re-ranking score much lower than original
        if original_score > 0.7 and rerank_score < 0.5:
            return True
        
        # Reduce confidence if absolute score is low
        if rerank_score < 0.4:
            return True
        
        return False
    
    def format_reranked_results(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format re-ranked chunks for output."""
        formatted = []
        
        for chunk in chunks:
            formatted.append({
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
                "scores": {
                    "original_score": chunk.get("scores", {}).get("overall_score", 0),
                    "rerank_score": chunk.get("rerank_score", 0),
                    "confidence": chunk.get("scores", {}).get("confidence", "low")
                }
            })
        
        return formatted
