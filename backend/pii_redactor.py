"""
PII Redaction Module
Removes personally identifiable information from text using spaCy NER and regex.
"""

import logging
import re
from typing import List, Dict, Any, Set, Optional
import time

logger = logging.getLogger(__name__)


class PIIRedactor:
    """
    PII redaction using spaCy NER and regex patterns.
    Removes sensitive information before LLM processing.
    """
    
    def __init__(self, 
                 model_name: str = "en_core_web_sm",
                 redact_entities: Optional[List[str]] = None,
                 redaction_text: str = "[REDACTED]"):
        """
        Initialize PII redactor.
        
        Args:
            model_name: spaCy model name
            redact_entities: Entity types to redact
            redaction_text: Replacement text for redacted content
        """
        self.model_name = model_name
        self.redaction_text = redaction_text
        self.nlp = None
        self._initialized = False
        
        # Entity types to redact
        self.redact_entities = redact_entities or [
            "PERSON",  # Person names
            "EMAIL",   # Email addresses (custom)
            "PHONE",   # Phone numbers (custom)
            "SSN",     # Social security numbers (custom)
            "CREDIT_CARD",  # Credit card numbers (custom)
            "IP_ADDRESS",   # IP addresses (custom)
        ]
        
        # Regex patterns for common PII
        self.patterns = {
            "EMAIL": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "PHONE": re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'),
            "SSN": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "CREDIT_CARD": re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            "IP_ADDRESS": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        }
        
        logger.info(f"PII redactor created (model: {model_name}, lazy loading)")
    
    def _ensure_initialized(self):
        """Initialize spaCy model on first use (lazy loading)."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading spaCy model: {self.model_name}...")
            start_time = time.time()
            
            import spacy
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                logger.warning(
                    f"spaCy model '{self.model_name}' not found. "
                    f"Downloading..."
                )
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", self.model_name],
                    check=True
                )
                self.nlp = spacy.load(self.model_name)
            
            load_time = time.time() - start_time
            logger.info(f"✓ spaCy model loaded in {load_time:.2f}s")
            self._initialized = True
            
        except ImportError:
            logger.error(
                "spaCy not installed. Install with: pip install spacy"
            )
            raise ImportError(
                "spaCy required for PII redaction. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}", exc_info=True)
            raise
    
    def redact_text(self, text: str) -> Dict[str, Any]:
        """
        Redact PII from text.
        
        Args:
            text: Input text potentially containing PII
            
        Returns:
            Dict with redacted text and metadata
        """
        if not text or not text.strip():
            return {
                "text": text,
                "redacted": False,
                "entities_found": [],
                "patterns_found": []
            }
        
        # Ensure model is loaded
        self._ensure_initialized()
        
        try:
            start_time = time.time()
            original_text = text
            entities_found = []
            patterns_found = []
            
            # Step 1: Regex-based redaction (faster, catches specific patterns)
            for pattern_name, pattern in self.patterns.items():
                if pattern_name in self.redact_entities:
                    matches = pattern.findall(text)
                    if matches:
                        patterns_found.append(pattern_name)
                        text = pattern.sub(self.redaction_text, text)
                        logger.debug(f"Redacted {len(matches)} {pattern_name} patterns")
            
            # Step 2: NER-based redaction (catches names and complex entities)
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.redact_entities:
                    entities_found.append(ent.label_)
                    text = text.replace(ent.text, self.redaction_text)
                    logger.debug(f"Redacted {ent.label_}: {ent.text}")
            
            elapsed = time.time() - start_time
            redacted = text != original_text
            
            if redacted:
                logger.info(
                    f"Redacted PII in {elapsed:.3f}s: "
                    f"{len(set(entities_found))} entity types, "
                    f"{len(set(patterns_found))} pattern types"
                )
            
            return {
                "text": text,
                "redacted": redacted,
                "entities_found": list(set(entities_found)),
                "patterns_found": list(set(patterns_found)),
                "elapsed_ms": elapsed * 1000
            }
            
        except Exception as e:
            logger.error(f"Error redacting PII: {str(e)}", exc_info=True)
            # Fallback: return original text
            return {
                "text": text,
                "redacted": False,
                "error": str(e)
            }
    
    def redact_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Redact PII from multiple chunks.
        
        Args:
            chunks: List of chunks with text field
            
        Returns:
            Chunks with PII redacted
        """
        if not chunks:
            return []
        
        redacted_chunks = []
        total_redacted = 0
        
        for chunk in chunks:
            chunk_copy = chunk.copy()
            text = chunk.get("text", "")
            
            result = self.redact_text(text)
            chunk_copy["text"] = result["text"]
            
            # Add redaction metadata
            if "metadata" not in chunk_copy:
                chunk_copy["metadata"] = {}
            
            chunk_copy["metadata"]["pii_redacted"] = result["redacted"]
            if result["redacted"]:
                chunk_copy["metadata"]["pii_entities"] = result["entities_found"]
                chunk_copy["metadata"]["pii_patterns"] = result["patterns_found"]
                total_redacted += 1
            
            redacted_chunks.append(chunk_copy)
        
        if total_redacted > 0:
            logger.info(f"Redacted PII from {total_redacted}/{len(chunks)} chunks")
        
        return redacted_chunks
    
    def redact_query(self, query: str) -> str:
        """
        Redact PII from user query.
        
        Args:
            query: User query
            
        Returns:
            Query with PII redacted
        """
        result = self.redact_text(query)
        if result["redacted"]:
            logger.warning(
                f"PII detected in user query! Redacted: "
                f"{result.get('entities_found', [])} {result.get('patterns_found', [])}"
            )
        return result["text"]
    
    def add_custom_pattern(self, name: str, pattern: str):
        """
        Add custom regex pattern for PII detection.
        
        Args:
            name: Pattern name
            pattern: Regex pattern string
        """
        try:
            compiled = re.compile(pattern)
            self.patterns[name] = compiled
            if name not in self.redact_entities:
                self.redact_entities.append(name)
            logger.info(f"Added custom PII pattern: {name}")
        except re.error as e:
            logger.error(f"Invalid regex pattern for {name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get redactor statistics."""
        return {
            "model": self.model_name,
            "initialized": self._initialized,
            "redact_entities": self.redact_entities,
            "pattern_count": len(self.patterns),
            "redaction_text": self.redaction_text,
            "status": "ready" if self._initialized else "lazy_load_pending"
        }
    
    def warmup(self):
        """Warmup the model by processing dummy text."""
        if self._initialized:
            return
        
        logger.info("Warming up PII redactor...")
        dummy_text = "John Doe's email is john@example.com and phone is 555-1234."
        
        try:
            self.redact_text(dummy_text)
            logger.info("✓ PII redactor warmup complete")
        except Exception as e:
            logger.warning(f"PII redactor warmup failed: {e}")
    
    @staticmethod
    def is_available() -> bool:
        """Check if spaCy is available."""
        try:
            import spacy
            return True
        except ImportError:
            return False
