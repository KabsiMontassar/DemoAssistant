"""
CrossEncoder Reranking Module
Uses transformer-based cross-encoder for accurate relevance scoring.
"""

import logging
from typing import List, Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    CrossEncoder reranker using ms-marco-MiniLM-L-6-v2.
    Provides accurate query-document relevance scores.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32):
        """
        Initialize CrossEncoder reranker.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self._initialized = False
        
        # Lazy loading - only load when first used
        logger.info(f"CrossEncoder reranker created (model: {model_name}, lazy loading)")
    
    def _ensure_initialized(self):
        """Initialize the model on first use (lazy loading)."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading CrossEncoder model: {self.model_name}...")
            start_time = time.time()
            
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=512)
            
            load_time = time.time() - start_time
            logger.info(f"✓ CrossEncoder loaded in {load_time:.2f}s")
            self._initialized = True
            
        except ImportError as e:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise ImportError(
                "sentence-transformers required for CrossEncoder. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error(f"Error loading CrossEncoder: {str(e)}", exc_info=True)
            raise
    
    def rerank(self, 
               query: str, 
               chunks: List[Dict[str, Any]], 
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank chunks using CrossEncoder scores.
        
        Args:
            query: User query
            chunks: List of chunks to rerank
            top_k: Return top K results (None = return all)
            
        Returns:
            Reranked chunks with cross_encoder_score added
        """
        if not chunks:
            return []
        
        # Ensure model is loaded
        self._ensure_initialized()
        
        try:
            start_time = time.time()
            
            # Prepare query-document pairs
            pairs = []
            for chunk in chunks:
                text = chunk.get("text", "")
                pairs.append([query, text])
            
            # Score all pairs
            logger.debug(f"Scoring {len(pairs)} query-document pairs...")
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            
            # Add scores to chunks
            reranked = []
            for i, chunk in enumerate(chunks):
                chunk_copy = chunk.copy()
                chunk_copy["cross_encoder_score"] = float(scores[i])
                reranked.append(chunk_copy)
            
            # Sort by cross-encoder score (descending)
            reranked.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
            
            # Apply top_k if specified
            if top_k is not None:
                reranked = reranked[:top_k]
            
            elapsed = time.time() - start_time
            logger.info(
                f"CrossEncoder reranked {len(chunks)} chunks in {elapsed:.3f}s "
                f"(~{elapsed/len(chunks)*1000:.1f}ms per chunk)"
            )
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error in CrossEncoder reranking: {str(e)}", exc_info=True)
            # Fallback: return original chunks without cross-encoder scores
            logger.warning("Falling back to original chunk order")
            return chunks
    
    def score_pairs(self, query: str, texts: List[str]) -> List[float]:
        """
        Score query-text pairs directly.
        
        Args:
            query: Query string
            texts: List of text strings to score
            
        Returns:
            List of scores (one per text)
        """
        if not texts:
            return []
        
        self._ensure_initialized()
        
        try:
            pairs = [[query, text] for text in texts]
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            return [float(score) for score in scores]
        except Exception as e:
            logger.error(f"Error scoring pairs: {str(e)}", exc_info=True)
            return [0.5] * len(texts)  # Fallback scores
    
    def combine_scores(self,
                      chunks: List[Dict[str, Any]],
                      cross_encoder_weight: float = 0.5,
                      original_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Combine cross-encoder scores with original scores.
        
        Args:
            chunks: Chunks with both cross_encoder_score and original scores
            cross_encoder_weight: Weight for cross-encoder score
            original_weight: Weight for original score
            
        Returns:
            Chunks with combined_score added
        """
        # Normalize weights
        total = cross_encoder_weight + original_weight
        cross_encoder_weight /= total
        original_weight /= total
        
        combined = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            
            # Get scores
            cross_score = chunk.get("cross_encoder_score", 0.5)
            original_score = chunk.get("scores", {}).get("overall_score", 0.5)
            
            # Normalize cross-encoder score to 0-1 range (typically -10 to 10)
            # Using sigmoid-like normalization
            cross_score_normalized = 1 / (1 + pow(2.718, -cross_score))
            
            # Combine scores
            combined_score = (
                cross_encoder_weight * cross_score_normalized +
                original_weight * original_score
            )
            
            chunk_copy["combined_score"] = combined_score
            chunk_copy["cross_encoder_score_normalized"] = cross_score_normalized
            combined.append(chunk_copy)
        
        # Sort by combined score
        combined.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "model": self.model_name,
            "batch_size": self.batch_size,
            "initialized": self._initialized,
            "status": "ready" if self._initialized else "lazy_load_pending"
        }
    
    def warmup(self):
        """Warmup the model by running a dummy inference."""
        if self._initialized:
            return
        
        logger.info("Warming up CrossEncoder model...")
        dummy_query = "What is the price of concrete?"
        dummy_text = "Concrete prices vary by project and supplier."
        
        try:
            self.score_pairs(dummy_query, [dummy_text])
            logger.info("✓ CrossEncoder warmup complete")
        except Exception as e:
            logger.warning(f"CrossEncoder warmup failed: {e}")
