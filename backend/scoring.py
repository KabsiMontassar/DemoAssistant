"""
Scoring & Ranking System
Scores and ranks retrieved results with confidence metrics.
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ScoringEngine:
    """Calculates relevance scores based on multiple components."""
    
    def __init__(self, 
                 vector_weight: float = 0.6,
                 metadata_weight: float = 0.3,
                 recency_weight: float = 0.1):
        """
        Initialize scoring weights.
        
        Args:
            vector_weight: Weight of vector similarity score
            metadata_weight: Weight of metadata match score
            recency_weight: Weight of recency score
        """
        self.vector_weight = vector_weight
        self.metadata_weight = metadata_weight
        self.recency_weight = recency_weight
        
        # Normalize weights
        total = vector_weight + metadata_weight + recency_weight
        self.vector_weight /= total
        self.metadata_weight /= total
        self.recency_weight /= total
    
    def score_result(self,
                    chunk: Dict[str, Any],
                    vector_score: float,
                    query_project: str = None,
                    query_material: str = None) -> Dict[str, Any]:
        """
        Calculate overall relevance score for a chunk.
        
        Args:
            chunk: Chunk with metadata
            vector_score: Semantic similarity score (0-1)
            query_project: Project filter (if any)
            query_material: Material filter (if any)
            
        Returns:
            Dict with overall score and component scores
        """
        metadata = chunk.get("metadata", {})
        
        # Vector similarity score (0-1, already normalized)
        v_score = min(1.0, max(0.0, vector_score))
        
        # Metadata match score
        m_score = self._compute_metadata_score(metadata, query_project, query_material)
        
        # Recency score
        r_score = self._compute_recency_score(metadata)
        
        # Weighted combination
        overall_score = (
            self.vector_weight * v_score +
            self.metadata_weight * m_score +
            self.recency_weight * r_score
        )
        
        return {
            "overall_score": overall_score,
            "vector_score": v_score,
            "metadata_score": m_score,
            "recency_score": r_score,
            "confidence": self._score_to_confidence(overall_score)
        }
    
    def score_batch(self,
                   chunks: List[Dict[str, Any]],
                   vector_scores: List[float],
                   query_project: str = None,
                   query_material: str = None) -> List[Dict[str, Any]]:
        """
        Score multiple chunks.
        
        Args:
            chunks: List of chunks
            vector_scores: List of vector similarity scores
            query_project: Project filter (if any)
            query_material: Material filter (if any)
            
        Returns:
            List of chunks with scores added to metadata
        """
        scored_chunks = []
        
        for chunk, vector_score in zip(chunks, vector_scores):
            scores = self.score_result(chunk, vector_score, query_project, query_material)
            
            chunk_copy = chunk.copy()
            chunk_copy["scores"] = scores
            scored_chunks.append(chunk_copy)
        
        return scored_chunks
    
    @staticmethod
    def _compute_metadata_score(metadata: Dict[str, Any],
                               query_project: str = None,
                               query_material: str = None) -> float:
        """
        Compute metadata match score based on project/material filters.
        
        Returns score between 0 and 1.
        """
        score = 0.5  # Base score for any result
        
        # Boost for project match
        if query_project and metadata.get("project_name", "").lower() == query_project.lower():
            score += 0.25
        
        # Boost for material match
        if query_material and metadata.get("material", "").lower() == query_material.lower():
            score += 0.25
        
        # Boost for semantic type
        semantic_type = metadata.get("semantic_type", "")
        if semantic_type in ["price", "specification"]:
            score += 0.05
        
        return min(1.0, score)
    
    @staticmethod
    def _compute_recency_score(metadata: Dict[str, Any]) -> float:
        """
        Compute recency score based on last_modified timestamp.
        
        Recent files get higher scores.
        """
        try:
            last_modified = metadata.get("last_modified")
            if not last_modified:
                return 0.5  # Base score if no timestamp
            
            # Parse ISO format timestamp
            mod_time = datetime.fromisoformat(last_modified)
            now = datetime.now(mod_time.tzinfo) if mod_time.tzinfo else datetime.now()
            
            # Days since modification
            days_old = (now - mod_time).days
            
            # Score decreases with age (0.5 at 30 days, 1.0 at 0 days)
            if days_old <= 0:
                return 1.0
            elif days_old >= 30:
                return 0.5
            else:
                return 1.0 - (days_old / 60)  # Linear decay
        except Exception as e:
            logger.debug(f"Error computing recency score: {e}")
            return 0.5
    
    @staticmethod
    def _score_to_confidence(score: float) -> str:
        """Convert numeric score to confidence level."""
        if score >= 0.8:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"


class RankingEngine:
    """Ranks and filters chunks for final results."""
    
    def __init__(self, scoring_engine: ScoringEngine = None):
        self.scoring_engine = scoring_engine or ScoringEngine()
    
    def rank_results(self,
                    chunks: List[Dict[str, Any]],
                    vector_scores: List[float],
                    query_project: str = None,
                    query_material: str = None,
                    top_k: int = 5,
                    min_confidence: str = "low") -> List[Dict[str, Any]]:
        """
        Score, rank, and filter chunks.
        
        Args:
            chunks: List of retrieved chunks
            vector_scores: List of vector similarity scores
            query_project: Project filter (if any)
            query_material: Material filter (if any)
            top_k: Return top K results
            min_confidence: Minimum confidence threshold ("very_low", "low", "medium", "high")
            
        Returns:
            Sorted list of top chunks with scores
        """
        # Score all chunks
        scored = self.scoring_engine.score_batch(
            chunks, vector_scores, query_project, query_material
        )
        
        # Filter by confidence threshold
        confidence_order = ["very_low", "low", "medium", "high"]
        min_idx = confidence_order.index(min_confidence)
        threshold_scores = {
            "very_low": 0.0,
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8
        }
        min_score = threshold_scores.get(min_confidence, 0.0)
        
        filtered = [c for c in scored if c["scores"]["overall_score"] >= min_score]
        
        # Sort by overall score (descending)
        sorted_results = sorted(
            filtered,
            key=lambda x: x["scores"]["overall_score"],
            reverse=True
        )
        
        # Return top K
        return sorted_results[:top_k]
    
    def format_result(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Format a chunk for final output."""
        metadata = chunk.get("metadata", {})
        scores = chunk.get("scores", {})
        
        return {
            "text": chunk.get("text", ""),
            "source": {
                "file_path": metadata.get("file_path"),
                "project": metadata.get("project_name"),
                "material": metadata.get("material"),
                "file_type": metadata.get("file_type"),
                "chunk_index": metadata.get("chunk_index"),
                "last_modified": metadata.get("last_modified")
            },
            "relevance": {
                "score": round(scores.get("overall_score", 0), 3),
                "confidence": scores.get("confidence", "unknown"),
                "vector_match": round(scores.get("vector_score", 0), 3),
                "metadata_match": round(scores.get("metadata_score", 0), 3)
            }
        }
    
    def format_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format multiple chunks for output."""
        return [self.format_result(chunk) for chunk in chunks]
