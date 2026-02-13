"""
Configuration Management Module
Centralized configuration for all RAG pipeline components.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class IntentConfig:
    """Configuration for intent classification."""
    confidence_threshold: float = 0.30  # 30% minimum for domain queries
    domain_keywords: str = (
        "material pricing construction project concrete wood metal stone brick "
        "supplier cost fee rate price specification dimensions project management "
        "building materials procurement vendor quote estimate budget delivery lead time"
    )


@dataclass
class RetrievalConfig:
    """Configuration for retrieval components."""
    # Vector retrieval
    initial_top_k: int = 20
    final_top_k: int = 10
    
    # Boost weights (reduced from 2.0-3.0x to 1.05-1.25x)
    boost_specification_project: float = 1.25
    boost_specification_category: float = 1.20
    boost_comparison_project: float = 1.05
    boost_comparison_category: float = 1.15
    boost_category_project: float = 1.0
    boost_category_category: float = 1.20
    boost_general_project: float = 1.10
    boost_general_category: float = 1.08
    
    # Material categories
    material_categories: list = field(default_factory=lambda: [
        'wood', 'concrete', 'metal', 'stone', 'steel', 'aluminum',
        'copper', 'brass', 'iron', 'marble', 'granite', 'limestone'
    ])


@dataclass
class HybridConfig:
    """Configuration for hybrid retrieval."""
    vector_weight: float = 0.60  # 60% weight for semantic search
    keyword_weight: float = 0.40  # 40% weight for BM25 keyword search
    
    # BM25 parameters
    bm25_k1: float = 1.5  # Term frequency saturation
    bm25_b: float = 0.75  # Document length normalization


@dataclass
class RerankingConfig:
    """Configuration for reranking components."""
    # CrossEncoder settings
    use_cross_encoder: bool = True
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_batch_size: int = 32
    
    # Score combination weights
    cross_encoder_weight: float = 0.50  # 50% weight
    query_similarity_weight: float = 0.30  # 30% weight
    original_score_weight: float = 0.20  # 20% weight
    
    # Reranking parameters
    rerank_top_k: int = 10


@dataclass
class ScoreGateConfig:
    """Configuration for score-based filtering."""
    relevance_threshold: float = 0.20  # 20% minimum relevance
    confidence_threshold_low: float = 0.40  # Low confidence boundary
    confidence_threshold_medium: float = 0.60  # Medium confidence boundary
    confidence_threshold_high: float = 0.80  # High confidence boundary


@dataclass
class PIIConfig:
    """Configuration for PII redaction."""
    enabled: bool = True
    spacy_model: str = "en_core_web_sm"
    
    # Entity types to redact
    redact_entities: list = field(default_factory=lambda: [
        "PERSON", "EMAIL", "PHONE", "SSN", "CREDIT_CARD", 
        "IP_ADDRESS", "PASSWORD"
    ])
    
    # Regex patterns for additional PII
    enable_regex_patterns: bool = True
    redaction_text: str = "[REDACTED]"


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""
    primary_provider: str = "mistral"  # mistral or ollama
    fallback_provider: str = "ollama"
    
    # Mistral API settings
    mistral_model: str = "mistral-medium"
    mistral_temperature: float = 0.2
    mistral_max_tokens: int = 1000
    
    # Ollama settings
    ollama_model: str = "mistral:7b-instruct-q4_K_M"
    ollama_temperature: float = 0.2
    ollama_max_tokens: int = 1000
    
    # Generation settings
    use_streaming: bool = True
    timeout_seconds: int = 30


@dataclass
class MonitoringConfig:
    """Configuration for monitoring and logging."""
    log_level: str = "INFO"
    log_queries: bool = True
    log_scores: bool = True
    log_latencies: bool = True
    
    # Performance tracking
    track_component_latencies: bool = True
    slow_query_threshold_ms: float = 5000.0  # 5 seconds
    
    # Metrics export
    export_metrics: bool = False
    metrics_endpoint: str = "localhost:9090"


@dataclass
class RAGConfig:
    """Master configuration for the entire RAG pipeline."""
    intent: IntentConfig = field(default_factory=IntentConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    score_gate: ScoreGateConfig = field(default_factory=ScoreGateConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration values."""
        # Validate weights sum to reasonable values
        hybrid_total = self.hybrid.vector_weight + self.hybrid.keyword_weight
        if not 0.95 <= hybrid_total <= 1.05:
            logger.warning(
                f"Hybrid weights sum to {hybrid_total:.2f}, expected ~1.0. "
                f"Normalizing..."
            )
            total = hybrid_total
            self.hybrid.vector_weight /= total
            self.hybrid.keyword_weight /= total
        
        rerank_total = (
            self.reranking.cross_encoder_weight + 
            self.reranking.query_similarity_weight + 
            self.reranking.original_score_weight
        )
        if not 0.95 <= rerank_total <= 1.05:
            logger.warning(
                f"Reranking weights sum to {rerank_total:.2f}, expected ~1.0. "
                f"Normalizing..."
            )
            self.reranking.cross_encoder_weight /= rerank_total
            self.reranking.query_similarity_weight /= rerank_total
            self.reranking.original_score_weight /= rerank_total
        
        # Validate thresholds
        if not 0 <= self.intent.confidence_threshold <= 1:
            raise ValueError("Intent confidence threshold must be between 0 and 1")
        
        if not 0 <= self.score_gate.relevance_threshold <= 1:
            raise ValueError("Score gate relevance threshold must be between 0 and 1")
        
        logger.info("✓ Configuration validated successfully")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "intent": self.intent.__dict__,
            "retrieval": self.retrieval.__dict__,
            "hybrid": self.hybrid.__dict__,
            "reranking": self.reranking.__dict__,
            "score_gate": self.score_gate.__dict__,
            "pii": self.pii.__dict__,
            "llm": self.llm.__dict__,
            "monitoring": self.monitoring.__dict__
        }
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override from environment variables if present
        if threshold := os.getenv('INTENT_THRESHOLD'):
            config.intent.confidence_threshold = float(threshold)
        
        if threshold := os.getenv('SCORE_GATE_THRESHOLD'):
            config.score_gate.relevance_threshold = float(threshold)
        
        if weight := os.getenv('HYBRID_VECTOR_WEIGHT'):
            config.hybrid.vector_weight = float(weight)
            config.hybrid.keyword_weight = 1.0 - float(weight)
        
        if enabled := os.getenv('PII_REDACTION_ENABLED'):
            config.pii.enabled = enabled.lower() in ('true', '1', 'yes')
        
        if enabled := os.getenv('CROSS_ENCODER_ENABLED'):
            config.reranking.use_cross_encoder = enabled.lower() in ('true', '1', 'yes')
        
        if level := os.getenv('LOG_LEVEL'):
            config.monitoring.log_level = level.upper()
        
        logger.info("✓ Configuration loaded from environment")
        return config


# Global configuration instance
_config: RAGConfig = None


def get_config() -> RAGConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config


def set_config(config: RAGConfig):
    """Set the global configuration instance."""
    global _config
    _config = config


def reload_config():
    """Reload configuration from environment."""
    global _config
    _config = RAGConfig.from_env()
    logger.info("✓ Configuration reloaded")
