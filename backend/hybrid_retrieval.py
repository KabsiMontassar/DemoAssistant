"""
Hybrid Retrieval Engine
Combines semantic vector search with BM25 keyword search for robust retrieval.
"""

import logging
from typing import List, Dict, Any, Tuple
import math

logger = logging.getLogger(__name__)


class BM25:
    """BM25 ranking algorithm for keyword search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with parameters.
        
        Args:
            k1: Controls term frequency saturation point
            b: Controls how much effect document length has on relevance
        """
        self.k1 = k1
        self.b = b
        self.avg_doc_len = 0
        self.doc_freqs = {}
        self.idf = {}
    
    def fit(self, documents: List[str]):
        """
        Fit BM25 model with document corpus.
        
        Args:
            documents: List of document texts
        """
        doc_lens = []
        
        # Calculate document frequencies and lengths
        for doc in documents:
            words = set(doc.lower().split())
            doc_lens.append(len(doc.split()))
            
            for word in words:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1
        
        # Calculate average document length
        self.avg_doc_len = sum(doc_lens) / len(doc_lens) if doc_lens else 0
        
        # Calculate IDF for all words
        num_docs = len(documents)
        for word, freq in self.doc_freqs.items():
            self.idf[word] = math.log(num_docs - freq + 0.5) - math.log(freq + 0.5)
    
    def score(self, query: str, document: str) -> float:
        """
        Score a document against a query using BM25.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            BM25 score (higher is more relevant)
        """
        query_words = query.lower().split()
        doc_words = document.lower().split()
        doc_len = len(doc_words)
        
        score = 0.0
        
        for word in query_words:
            if word not in self.idf:
                continue
            
            # Count occurrences of word in document
            word_count = doc_words.count(word)
            
            # BM25 formula
            idf = self.idf.get(word, 0)
            numerator = idf * word_count * (self.k1 + 1)
            denominator = word_count + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
            
            score += numerator / denominator
        
        return score
    
    def rank(self, query: str, documents: List[str]) -> List[Tuple[int, float]]:
        """
        Rank documents by BM25 score.
        
        Args:
            query: Query text
            documents: List of documents
            
        Returns:
            List of (doc_idx, score) tuples sorted by score descending
        """
        scores = []
        for idx, doc in enumerate(documents):
            score = self.score(query, doc)
            if score > 0:  # Only include non-zero scores
                scores.append((idx, score))
        
        # Sort by score descending
        return sorted(scores, key=lambda x: x[1], reverse=True)


class HybridRetriever:
    """Combines semantic and keyword search for hybrid retrieval."""
    
    def __init__(self, vector_weight: float = 0.6, keyword_weight: float = 0.4, use_rrf: bool = True):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_weight: Weight for vector similarity scores
            keyword_weight: Weight for BM25 keyword scores
            use_rrf: Use Reciprocal Rank Fusion instead of simple weighted combination
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.use_rrf = use_rrf
        self.bm25 = BM25()
        self._trained = False
        
        logger.info(f"HybridRetriever initialized (RRF: {use_rrf}, weights: {vector_weight}/{keyword_weight})")
    
    def train(self, chunks: List[Dict[str, Any]]):
        """
        Train BM25 on chunk corpus.
        
        Args:
            chunks: List of chunks with text
        """
        texts = [chunk.get("text", "") for chunk in chunks]
        self.bm25.fit(texts)
        self._trained = True
        logger.info(f"Trained BM25 on {len(texts)} documents")
    
    def hybrid_score(self,
                    vector_score: float,
                    keyword_score: float,
                    normalize_keyword: bool = True) -> float:
        """
        Combine vector and keyword scores using weighted combination.
        
        Args:
            vector_score: Vector similarity score (0-1)
            keyword_score: BM25 score (0-unbounded)
            normalize_keyword: Whether to normalize BM25 score to 0-1
            
        Returns:
            Combined score
        """
        # Normalize vector score to 0-1
        v_score = min(1.0, max(0.0, vector_score))
        
        # Normalize keyword score if needed
        if normalize_keyword:
            k_score = min(1.0, keyword_score / 10.0)  # Normalize BM25 (typical max ~10)
        else:
            k_score = keyword_score
        
        # Weighted combination
        total_weight = self.vector_weight + self.keyword_weight
        return (self.vector_weight * v_score + self.keyword_weight * k_score) / total_weight
    
    def reciprocal_rank_fusion(self,
                               chunks: List[Dict[str, Any]],
                               vector_scores: List[float],
                               bm25_scores: List[float],
                               k: int = 60) -> List[Dict[str, Any]]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        RRF is more robust than score-based fusion for combining different ranking systems.
        
        Formula: RRF_score = Σ 1 / (k + rank_i)
        
        Args:
            chunks: List of chunks
            vector_scores: Vector similarity scores
            bm25_scores: BM25 keyword scores
            k: RRF constant (default: 60, standard in literature)
            
        Returns:
            Chunks sorted by RRF score
        """
        # Create ranked lists
        # Vector ranking (by score descending)
        vector_ranked = sorted(enumerate(vector_scores), key=lambda x: x[1], reverse=True)
        vector_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(vector_ranked)}
        
        # BM25 ranking (by score descending)
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_ranked)}
        
        # Calculate RRF scores
        rrf_results = []
        for i, chunk in enumerate(chunks):
            vector_rank = vector_ranks.get(i, len(chunks) + 1)
            bm25_rank = bm25_ranks.get(i, len(chunks) + 1)
            
            # RRF formula
            rrf_score = (
                self.vector_weight * (1.0 / (k + vector_rank)) +
                self.keyword_weight * (1.0 / (k + bm25_rank))
            )
            
            chunk_copy = chunk.copy()
            chunk_copy["rrf_score"] = rrf_score
            chunk_copy["vector_rank"] = vector_rank
            chunk_copy["bm25_rank"] = bm25_rank
            chunk_copy["vector_score"] = vector_scores[i]
            chunk_copy["keyword_score"] = bm25_scores[i]
            
            rrf_results.append(chunk_copy)
        
        # Sort by RRF score (descending)
        rrf_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        logger.debug(f"RRF fusion: top score = {rrf_results[0]['rrf_score']:.4f}")
        
        return rrf_results
    
    def retrieve(self,
                query: str,
                chunks: List[Dict[str, Any]],
                vector_scores: List[float],
                top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: Search query
            chunks: List of chunks
            vector_scores: Vector similarity scores for each chunk
            top_k: Return top K results
            
        Returns:
            Top K chunks with combined scores
        """
        if not self._trained:
            self.train(chunks)
        
        # Get BM25 scores
        texts = [chunk.get("text", "") for chunk in chunks]
        bm25_scores = [0.0] * len(chunks)
        
        ranked = self.bm25.rank(query, texts)
        for idx, score in ranked:
            bm25_scores[idx] = score
        
        # Choose fusion method
        if self.use_rrf:
            # Use Reciprocal Rank Fusion
            combined_results = self.reciprocal_rank_fusion(
                chunks, vector_scores, bm25_scores, k=60
            )
            # Use RRF score as hybrid_score
            for result in combined_results:
                result["hybrid_score"] = result["rrf_score"]
        else:
            # Use simple weighted combination
            combined_results = []
            for i, chunk in enumerate(chunks):
                combined_score = self.hybrid_score(
                    vector_scores[i],
                    bm25_scores[i],
                    normalize_keyword=True
                )
                
                result = chunk.copy()
                result["hybrid_score"] = combined_score
                result["vector_score"] = vector_scores[i]
                result["keyword_score"] = bm25_scores[i]
                combined_results.append(result)
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        logger.debug(
            f"Hybrid {'RRF' if self.use_rrf else 'weighted'} search returned "
            f"{len(combined_results[:top_k])} results for query: {query[:50]}..."
        )
        
        return combined_results[:top_k]
