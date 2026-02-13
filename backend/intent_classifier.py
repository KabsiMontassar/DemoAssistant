"""
Intent Classification with Confidence Scoring
Determines whether a query requires domain retrieval or can be handled as chitchat.
Uses embedding similarity to domain keywords for confidence scoring.
"""

import logging
from typing import Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Classifies user query intent and provides confidence scoring.
    Routes queries to appropriate handlers based on domain relevance.
    """
    
    # Domain-specific keywords for material pricing queries
    DOMAIN_KEYWORDS = [
        "material pricing construction project concrete wood metal stone supplier",
        "price cost rate fee budget estimate quotation materials",
        "construction building infrastructure project procurement supply",
        "wood lumber timber oak pine maple concrete cement aggregate",
        "metal steel aluminum copper iron brass stone granite marble",
        "supplier vendor contractor manufacturer distributor",
        "specification spec dimension size weight property quality"
    ]
    
    # Chitchat patterns
    CHITCHAT_PATTERNS = [
        "hello hi hey greetings good morning afternoon evening",
        "how are you what's up how do you do",
        "thank you thanks appreciate gratitude",
        "help assist support guide instructions",
        "who are you what can you do capabilities features",
        "goodbye bye see you later farewell"
    ]
    
    def __init__(self, embedding_manager):
        """
        Initialize intent classifier with embedding manager.
        
        Args:
            embedding_manager: EmbeddingManager instance for encoding queries
        """
        self.embedding_manager = embedding_manager
        self.model = embedding_manager.model
        
        # Pre-compute domain embedding (average of domain keyword embeddings)
        logger.info("Computing domain embeddings for intent classification...")
        domain_embeddings = [self.model.encode(phrase) for phrase in self.DOMAIN_KEYWORDS]
        self.domain_embedding = np.mean(domain_embeddings, axis=0)
        
        # Pre-compute chitchat embedding
        chitchat_embeddings = [self.model.encode(phrase) for phrase in self.CHITCHAT_PATTERNS]
        self.chitchat_embedding = np.mean(chitchat_embeddings, axis=0)
        
        logger.info("✓ Intent classifier initialized")
    
    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent and return confidence score.
        
        Args:
            query: User query string
            
        Returns:
            Dict with:
                - intent: "domain" or "chitchat"
                - confidence: 0-100 (percentage)
                - domain_score: Raw domain similarity score
                - chitchat_score: Raw chitchat similarity score
                - requires_retrieval: Boolean flag
                - routing: Suggested handler ("rag_pipeline" or "chitchat_handler")
        """
        if not query or len(query.strip()) == 0:
            return {
                "intent": "invalid",
                "confidence": 0,
                "domain_score": 0,
                "chitchat_score": 0,
                "requires_retrieval": False,
                "routing": "error"
            }
        
        try:
            # Encode query
            query_embedding = self.model.encode(query)
            
            # Calculate cosine similarity to domain and chitchat embeddings
            domain_similarity = cosine_similarity(
                [query_embedding], 
                [self.domain_embedding]
            )[0][0]
            
            chitchat_similarity = cosine_similarity(
                [query_embedding], 
                [self.chitchat_embedding]
            )[0][0]
            
            # Convert to percentage (0-100)
            domain_confidence = float(domain_similarity * 100)
            chitchat_confidence = float(chitchat_similarity * 100)
            
            # Decision logic with graduated thresholds
            # High domain confidence (>= 30%) → Use RAG pipeline
            # High chitchat confidence AND low domain confidence → Direct LLM
            # Ambiguous → Default to domain (safer to retrieve than not)
            
            if domain_confidence >= 30:
                # Strong domain signal → RAG pipeline
                intent = "domain"
                confidence = domain_confidence
                requires_retrieval = True
                routing = "rag_pipeline"
            elif chitchat_confidence > domain_confidence and chitchat_confidence >= 40:
                # Clear chitchat signal → Direct response
                intent = "chitchat"
                confidence = chitchat_confidence
                requires_retrieval = False
                routing = "chitchat_handler"
            else:
                # Ambiguous or weak signals → Default to domain (better to over-retrieve)
                intent = "domain"
                confidence = max(domain_confidence, 25.0)  # Minimum 25% for ambiguous
                requires_retrieval = True
                routing = "rag_pipeline"
            
            logger.debug(
                f"Intent classification: {intent} (confidence: {confidence:.1f}%), "
                f"domain: {domain_confidence:.1f}%, chitchat: {chitchat_confidence:.1f}%"
            )
            
            return {
                "intent": intent,
                "confidence": round(confidence, 2),
                "domain_score": round(domain_confidence, 2),
                "chitchat_score": round(chitchat_confidence, 2),
                "requires_retrieval": requires_retrieval,
                "routing": routing
            }
            
        except Exception as e:
            logger.error(f"Error classifying intent: {str(e)}", exc_info=True)
            # On error, default to domain search (fail-safe)
            return {
                "intent": "domain",
                "confidence": 50,
                "domain_score": 50,
                "chitchat_score": 0,
                "requires_retrieval": True,
                "routing": "rag_pipeline",
                "error": str(e)
            }
    
    def should_use_retrieval(self, query: str) -> bool:
        """
        Quick check: Should this query use retrieval?
        
        Args:
            query: User query
            
        Returns:
            Boolean indicating if retrieval is needed
        """
        result = self.classify_query(query)
        return result.get("requires_retrieval", True)
    
    def get_confidence_level(self, confidence: float) -> str:
        """
        Convert confidence score to categorical level.
        
        Args:
            confidence: Confidence score (0-100)
            
        Returns:
            "high", "medium", "low", or "very_low"
        """
        if confidence >= 70:
            return "high"
        elif confidence >= 40:
            return "medium"
        elif confidence >= 20:
            return "low"
        else:
            return "very_low"
