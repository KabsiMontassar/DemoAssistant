"""
Intent Classification Module
Classifies user queries as domain-relevant or chitchat using embedding similarity.
"""

import logging
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classifies query intent using embedding-based similarity.
    Routes chitchat/greetings away from retrieval pipeline.
    """
    
    def __init__(self, embedding_manager):
        """
        Initialize intent classifier with embedding manager.
        
        Args:
            embedding_manager: EmbeddingManager instance with loaded model
        """
        self.embedding_manager = embedding_manager
        self.model = embedding_manager.model
        
        # Pre-compute domain embedding (material pricing domain)
        self.domain_text = (
            "material pricing construction project concrete wood metal stone brick "
            "supplier cost fee rate price specification dimensions project management "
            "building materials procurement vendor quote estimate budget delivery lead time"
        )
        
        logger.info("Computing domain embedding for intent classification...")
        self.domain_embedding = self.model.encode(self.domain_text)
        logger.info("✓ Intent classifier initialized")
    
    def classify_query(self, query: str) -> Dict[str, any]:
        """
        Classify query intent using cosine similarity to domain embedding.
        
        Args:
            query: User query string
            
        Returns:
            Dict with intent, confidence score (0-100), and routing decision
        """
        try:
            # Encode query
            query_embedding = self.model.encode(query)
            
            # Compute cosine similarity
            similarity = self._cosine_similarity(query_embedding, self.domain_embedding)
            
            # Convert to percentage confidence (0-100)
            confidence = round(similarity * 100, 2)
            
            # Classification threshold: 30% confidence
            # Below 30% = chitchat/greeting
            # Above 30% = domain-relevant query
            CONFIDENCE_THRESHOLD = 30.0
            
            if confidence >= CONFIDENCE_THRESHOLD:
                intent = "domain"
                requires_retrieval = True
                routing = "retrieval_pipeline"
            else:
                intent = "chitchat"
                requires_retrieval = False
                routing = "chitchat_handler"
            
            result = {
                "intent": intent,
                "confidence": confidence,
                "requires_retrieval": requires_retrieval,
                "routing": routing,
                "threshold": CONFIDENCE_THRESHOLD
            }
            
            logger.info(
                f"Intent: {intent} | Confidence: {confidence}% | "
                f"Routing: {routing} | Query: {query[:50]}..."
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}", exc_info=True)
            # On error, default to domain query (safe fallback)
            return {
                "intent": "domain",
                "confidence": 50.0,
                "requires_retrieval": True,
                "routing": "retrieval_pipeline",
                "error": str(e)
            }
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def update_domain_embedding(self, custom_domain_text: str):
        """
        Update the domain embedding with custom text.
        Useful for fine-tuning to specific use cases.
        
        Args:
            custom_domain_text: Custom domain description text
        """
        logger.info("Updating domain embedding with custom text...")
        self.domain_text = custom_domain_text
        self.domain_embedding = self.model.encode(custom_domain_text)
        logger.info("✓ Domain embedding updated")
    
    def get_stats(self) -> Dict[str, any]:
        """Get classifier statistics."""
        return {
            "domain_text_length": len(self.domain_text),
            "embedding_dim": len(self.domain_embedding),
            "threshold": 30.0,
            "model": "multilingual-e5-base (reused)"
        }
