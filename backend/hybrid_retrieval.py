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
    
    def __init__(self, vector_weight: float = 0.6, keyword_weight: float = 0.4):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_weight: Weight for vector similarity scores
            keyword_weight: Weight for BM25 keyword scores
        """
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.bm25 = BM25()
        self._trained = False
    
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
        Combine vector and keyword scores.
        
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
                              vector_ranks: List[int],
                              keyword_ranks: List[int],
                              k: int = 60) -> List[float]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        RRF is more robust than score combination as it's rank-based.
        
        Formula: RRF_score = sum(1 / (k + rank_i)) for each ranking system
        
        Args:
            vector_ranks: Rank positions from vector search (0-indexed)
            keyword_ranks: Rank positions from keyword search (0-indexed)
            k: Constant for RRF (default 60, standard value)
            
        Returns:
            List of RRF scores
        """
        num_items = len(vector_ranks)
        rrf_scores = []
        
        for i in range(num_items):
            # RRF score combines ranks from both systems
            vector_contribution = 1.0 / (k + vector_ranks[i] + 1) if vector_ranks[i] >= 0 else 0
            keyword_contribution = 1.0 / (k + keyword_ranks[i] + 1) if keyword_ranks[i] >= 0 else 0
            
            rrf_score = vector_contribution + keyword_contribution
            rrf_scores.append(rrf_score)
        
        return rrf_scores
    
    def retrieve(self,
                query: str,
                chunks: List[Dict[str, Any]],
                vector_scores: List[float],
                top_k: int = 5,
                use_rrf: bool = True) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval combining semantic and keyword search.
        Uses Reciprocal Rank Fusion (RRF) for robust score combination.
        
        Args:
            query: Search query
            chunks: List of chunks
            vector_scores: Vector similarity scores for each chunk
            top_k: Return top K results
            use_rrf: Use RRF instead of weighted scores (recommended)
            
        Returns:
            Top K chunks with combined scores
        """
        if not self._trained:
            self.train(chunks)
        
        # Get BM25 keyword scores
        texts = [chunk.get("text", "") for chunk in chunks]
        bm25_results = self.bm25.rank(query, texts)
        
        # Create score and rank mappings
        bm25_scores = [0.0] * len(chunks)
        keyword_ranks = [-1] * len(chunks)  # -1 means not found
        
        for rank, (idx, score) in enumerate(bm25_results):
            bm25_scores[idx] = score
            keyword_ranks[idx] = rank
        
        # Create vector ranks (already sorted by score)
        vector_ranks = list(range(len(chunks)))
        
        # Combine using RRF (preferred) or weighted scores
        combined_results = []
        
        if use_rrf:
            # Use Reciprocal Rank Fusion
            rrf_scores = self.reciprocal_rank_fusion(vector_ranks, keyword_ranks)
            
            for i, chunk in enumerate(chunks):
                result = chunk.copy()
                result["hybrid_score"] = rrf_scores[i]
                result["vector_score"] = vector_scores[i]
                result["keyword_score"] = bm25_scores[i]
                result["fusion_method"] = "rrf"
                combined_results.append(result)
        else:
            # Use weighted score combination
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
                result["fusion_method"] = "weighted"
                combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        logger.debug(f"Hybrid search (RRF={use_rrf}) returned {len(combined_results[:top_k])} results for query: {query[:50]}")
        
        return combined_results[:top_k]
