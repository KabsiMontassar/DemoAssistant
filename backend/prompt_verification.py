"""
Prompt Verification and Fixing Module
Cleans, corrects, and formats user queries before processing.
"""

import os
import re
import logging
from typing import Dict, List, Optional
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class PromptVerifier:
    """
    Verifies and fixes user prompts before they enter the RAG pipeline.
    Handles spelling correction, text cleaning, and entity normalization.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the PromptVerifier.
        
        Args:
            data_path: Path to the materials data directory for entity extraction
        """
        # Get data path and ensure it's absolute
        raw_path = data_path or os.getenv('DATA_PATH', './data')
        self.data_path = Path(raw_path).resolve()
        
        # Try materials subfolder first, then fall back to data_path
        materials_candidate = self.data_path / 'materials'
        if materials_candidate.exists() and materials_candidate.is_dir():
            self.materials_path = materials_candidate
        else:
            # Maybe data_path IS already the materials directory
            self.materials_path = self.data_path
        
        # Static spelling corrections - minimal set, let auto-fix handle the rest
        self.spelling_corrections = {}
        
        # Material categories
        self.material_categories = {
            'wood', 'concrete', 'metal', 'stone', 'steel', 'aluminum',
            'copper', 'brass', 'iron', 'marble', 'granite', 'limestone'
        }
        
        # Load project names from filesystem (with error handling)
        try:
            self.project_names = self._load_project_names()
            logger.info(f"Prompt verifier initialized with {len(self.project_names)} projects")
        except Exception as e:
            logger.error(f"Failed to load project names: {e}. Continuing with empty project list.")
            self.project_names = {}
    
    def _load_project_names(self) -> Dict[str, str]:
        """
        Load project names from the materials directory.
        Creates a mapping of lowercase variations to actual project names.
        
        Returns:
            Dict mapping lowercase/normalized names to actual project folder names
        """
        project_mapping = {}
        
        if not self.materials_path.exists():
            logger.warning(f"Materials path does not exist: {self.materials_path}. Running without project name validation.")
            return project_mapping
        
        if not self.materials_path.is_dir():
            logger.warning(f"Materials path is not a directory: {self.materials_path}. Running without project name validation.")
            return project_mapping
        
        try:
            for item in self.materials_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    actual_name = item.name
                    
                    # Create various mappings
                    # 1. Exact lowercase
                    project_mapping[actual_name.lower()] = actual_name
                    
                    # 2. With spaces (ApexIndustries -> apex industries)
                    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', actual_name).lower()
                    project_mapping[spaced] = actual_name
                    
                    # 3. Remove common suffixes like "Industries", "Inc", "Corp"
                    base = re.sub(r'\s*(industries|inc|corp|group|solutions|enterprises|systems|construction|builders|materials|intl|international)\s*$', '', spaced, flags=re.IGNORECASE)
                    if base and base != spaced:
                        project_mapping[base.strip()] = actual_name
            
            logger.info(f"Loaded {len(set(project_mapping.values()))} unique projects")
            
        except Exception as e:
            logger.error(f"Error loading project names: {e}")
        
        return project_mapping
    
    def _clean_whitespace(self, text: str) -> str:
        """
        Clean excessive whitespace from text.
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _fix_spelling(self, text: str) -> str:
        """
        Dynamically fix spelling mistakes using fuzzy matching against entities
        that exist in the system (project names, materials only).
        NO static word lists. Focuses on correcting domain entity typos.
        
        Args:
            text: Input text
            
        Returns:
            Text with spelling corrections applied
        """
        words = text.split()
        corrected_words = []
        
        for word in words:
            lower_word = word.lower()
            if len(lower_word) == 0:
                corrected_words.append(word)
                continue
            
            # Build candidate words from system entities only
            candidates = {}
            
            # Material categories (highest priority)
            for material in self.material_categories:
                candidates[material] = 1000
            
            # Project names - full and component parts
            for project in self.project_names.values():
                candidates[project.lower()] = 900
                
                # Extract individual words from camelCase
                spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', project).lower()
                for part in spaced.split():
                    if len(part) >= 2:  # Words of 2+ chars
                        candidates[part] = 850
                
                # Also add original with spaces if it has multiple words
                if ' ' in project.lower():
                    for word_part in project.lower().split():
                        if len(word_part) >= 2:
                            candidates[word_part] = 850
            
            best_match = None
            best_score = 0.0
            word_len = len(lower_word)
            
            # Only correct if word is NOT already a known entity
            if lower_word not in candidates:
                for candidate, priority in candidates.items():
                    candidate_len = len(candidate)
                    distance = _levenshtein_distance(lower_word, candidate)
                    max_len = max(word_len, candidate_len)
                    
                    # Similarity score (0-1, higher is better)
                    similarity = (1.0 - distance / max_len) if max_len > 0 else 0
                    
                    # More lenient thresholds to catch typos in entity names
                    # Shorter words allow more typos proportionally
                    if word_len <= 3:
                        min_similarity = 0.60  # Allow 1-2 char difference in very short words
                    elif word_len <= 6:
                        min_similarity = 0.65  # 1-2 char difference in medium words
                    else:
                        min_similarity = 0.70  # Standard threshold for longer words
                    
                    if similarity >= min_similarity:
                        weighted_score = similarity * (priority / 1000)
                        if weighted_score > best_score:
                            best_score = weighted_score
                            best_match = candidate
            
            # Apply correction if confidence is reasonable
            if best_match and best_score > 0.6:
                if word[0].isupper() and len(word) > 1:
                    best_match = best_match.capitalize()
                corrected_words.append(best_match)
                logger.debug(f"Auto-corrected '{word}' to '{best_match}' (score: {best_score:.3f})")
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _normalize_entities(self, text: str) -> tuple[str, List[str]]:
        """
        Normalize entity names (projects, materials) to match filesystem structure.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (normalized_text, list_of_detected_projects)
        """
        detected_projects = []
        
        # Sort project names by length (longest first) to handle overlapping matches
        # e.g., "apex industries" should match before "apex"
        sorted_variants = sorted(self.project_names.items(), key=lambda x: len(x[0]), reverse=True)
        
        # Normalize project names with word boundaries
        for variant, actual in sorted_variants:
            # Use word boundaries to match complete phrases only
            # This prevents "Apex Industries" from becoming "ApexIndustriesIndustries"
            pattern = re.compile(r'\b' + re.escape(variant) + r'\b', re.IGNORECASE)
            
            if pattern.search(text):
                if actual not in detected_projects:
                    detected_projects.append(actual)
                text = pattern.sub(actual, text)
        
        return text, detected_projects
    
    def _capitalize_properly(self, text: str) -> str:
        """
        Apply proper capitalization to the query.
        
        Args:
            text: Input text
            
        Returns:
            Text with proper capitalization
        """
        if not text:
            return text
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        # Capitalize after punctuation
        text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text
    
    def _validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Validate the query for basic requirements.
        
        Args:
            query: Input query
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not query or len(query.strip()) == 0:
            return False, "Query cannot be empty"
        
        if len(query) < 3:
            return False, "Query too short (minimum 3 characters)"
        
        if len(query) > 2000:
            return False, "Query too long (maximum 2000 characters)"
        
        # Check if query contains at least one alphabetic character
        if not re.search(r'[a-zA-Z]', query):
            return False, "Query must contain at least one letter"
        
        return True, None
    
    def _is_domain_relevant(self, query: str, detected_projects: list, detected_materials: list) -> tuple[bool, Optional[str]]:
        """
        Check if the query is relevant to the material pricing domain.
        
        Queries are domain-relevant if they:
        - Mention a project OR
        - Mention a material category OR  
        - Contain domain-specific keywords (price, cost, specification, etc.)
        
        Args:
            query: The query string (normalized)
            detected_projects: List of detected project names
            detected_materials: List of detected material categories
            
        Returns:
            Tuple of (is_relevant, rejection_message)
        """
        query_lower = query.lower()
        
        # Domain keywords that indicate material/pricing questions
        domain_keywords = {
            'price', 'cost', 'fee', 'rate', 'per', 'specification', 'spec',
            'dimension', 'size', 'weight', 'property', 'material', 'supplier',
            'quote', 'bid', 'estimate', 'lead time', 'availability', 'stock',
            'quality', 'grade', 'finish', 'strength', 'capacity', 'mpa',
            'compare', 'comparison', 'difference', 'versus', 'vs', 'vs.',
            'project', 'construction', 'building', 'order', 'purchase'
        }
        
        # Check if query has domain keywords
        has_domain_keywords = any(keyword in query_lower for keyword in domain_keywords)
        
        # Domain relevant if: has projects OR has materials OR has domain keywords
        is_relevant = bool(detected_projects) or bool(detected_materials) or has_domain_keywords
        
        if not is_relevant:
            return False, None  # Not relevant, will handle in main method
        
        return True, None
    
    def verify_and_fix(self, query: str) -> Dict[str, any]:
        """
        Main method to verify and fix a user query.
        
        Args:
            query: Raw user input query
            
        Returns:
            Dict containing:
                - fixed_query: The corrected and formatted query
                - original_query: The original input
                - changes_made: List of changes applied
                - is_valid: Whether the query passed validation
                - error: Error message if invalid
                - is_domain_relevant: Whether query relates to projects/materials
        """
        original_query = query
        changes_made = []
        
        # Step 1: Validate input
        is_valid, error = self._validate_query(query)
        if not is_valid:
            logger.warning(f"Query validation failed: {error}")
            return {
                'fixed_query': query,
                'original_query': original_query,
                'changes_made': [],
                'is_valid': False,
                'error': error,
                'is_domain_relevant': False,
                'detected_projects': []
            }
        
        # Step 2: Clean whitespace
        cleaned = self._clean_whitespace(query)
        if cleaned != query:
            changes_made.append('whitespace_normalized')
            logger.debug(f"Whitespace cleaned: '{query}' -> '{cleaned}'")
        query = cleaned
        
        # Step 3: Fix spelling
        spell_fixed = self._fix_spelling(query)
        if spell_fixed != query:
            changes_made.append('spelling_corrected')
            logger.debug(f"Spelling fixed: '{query}' -> '{spell_fixed}'")
        query = spell_fixed
        
        # Step 4: Normalize entities (project names, materials)
        entity_normalized, detected_projects = self._normalize_entities(query)
        if entity_normalized != query:
            changes_made.append('entities_normalized')
            logger.debug(f"Entities normalized: '{query}' -> '{entity_normalized}'")
        query = entity_normalized
        
        # Extract detected materials
        detected_materials = [c for c in self.material_categories if c in query.lower()]
        
        # Step 5: Check domain relevance - BEFORE capitalization
        is_domain_relevant, _ = self._is_domain_relevant(query, detected_projects, detected_materials)
        
        # Step 6: Proper capitalization
        capitalized = self._capitalize_properly(query)
        if capitalized != query:
            changes_made.append('capitalization_fixed')
            logger.debug(f"Capitalization fixed: '{query}' -> '{capitalized}'")
        query = capitalized
        
        # Log summary
        if changes_made:
            logger.info(f"Query verification complete. Changes: {', '.join(changes_made)}")
            logger.info(f"Original: '{original_query}' -> Fixed: '{query}'")
        else:
            logger.debug(f"Query passed verification without changes: '{query}'")
        
        return {
            'fixed_query': query,
            'original_query': original_query,
            'changes_made': changes_made,
            'is_valid': True,
            'error': None,
            'is_domain_relevant': is_domain_relevant,
            'detected_projects': detected_projects
        }
    
    def add_spelling_correction(self, incorrect: str, correct: str):
        """
        Add a custom spelling correction.
        
        Args:
            incorrect: The incorrect spelling
            correct: The correct spelling
        """
        self.spelling_corrections[incorrect.lower()] = correct.lower()
        logger.info(f"Added spelling correction: '{incorrect}' -> '{correct}'")
    
    def reload_projects(self):
        """Reload project names from the filesystem."""
        self.project_names = self._load_project_names()
        logger.info("Project names reloaded")
