"""
Query Preprocessing with Fuzzy Matching
Handles typos, entity extraction, and query enhancement.
"""

import logging
import re
from typing import Dict, List, Tuple, Any
from difflib import get_close_matches

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """
    Preprocesses queries to handle typos, extract entities, and enhance retrieval.
    """
    
    # Known entities in the system
    PROJECTS = [
        "projectacme", "projectfacebook", "apexindustries", "ecobuildersintel",
        "nexgenstructures", "omniconstructinc", "pinnacleconstructio",
        "quantumbuildsystems", "skyriseenterprises", "techvisioncorp",
        "titanmaterialsgroup", "venturebuildolutions"
    ]
    
    MATERIALS = ["concrete", "wood", "metal", "stone", "lumber", "timber", "steel", "granite", "marble"]
    
    # Common typos and corrections
    TYPO_CORRECTIONS = {
        "concret": "concrete",
        "conrete": "concrete",
        "concerte": "concrete",
        "wod": "wood",
        "woood": "wood",
        "metl": "metal",
        "matal": "metal",
        "ston": "stone",
        "stome": "stone",
        "pric": "price",
        "priice": "price",
        "projecr": "project",
        "projct": "project"
    }
    
    def __init__(self):
        """Initialize query preprocessor."""
        logger.info("✓ Query preprocessor initialized")
    
    def preprocess(self, query: str) -> Dict[str, Any]:
        """
        Preprocess query with typo correction, entity extraction, and enhancement.
        
        Args:
            query: Raw user query
            
        Returns:
            Dict with:
                - corrected_query: Query with typos fixed
                - original_query: Original input
                - entities: Extracted entities (projects, materials)
                - corrections_made: List of corrections applied
                - enhanced_query: Query enhanced with synonyms
        """
        if not query or len(query.strip()) == 0:
            return {
                "corrected_query": "",
                "original_query": query,
                "entities": {"projects": [], "materials": []},
                "corrections_made": [],
                "enhanced_query": ""
            }
        
        try:
            original = query
            corrected = query
            corrections_made = []
            
            # Step 1: Fix common typos
            corrected, typo_fixes = self._fix_typos(corrected)
            corrections_made.extend(typo_fixes)
            
            # Step 2: Fuzzy match project names and materials
            corrected, fuzzy_fixes = self._fuzzy_match_entities(corrected)
            corrections_made.extend(fuzzy_fixes)
            
            # Step 3: Extract entities
            entities = self._extract_entities(corrected)
            
            # Step 4: Enhance query with synonyms (optional)
            enhanced = self._enhance_query(corrected, entities)
            
            if corrections_made:
                logger.info(f"Query corrections: {corrections_made}")
            
            return {
                "corrected_query": corrected,
                "original_query": original,
                "entities": entities,
                "corrections_made": corrections_made,
                "enhanced_query": enhanced
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing query: {str(e)}", exc_info=True)
            # Return original query on error
            return {
                "corrected_query": query,
                "original_query": query,
                "entities": {"projects": [], "materials": []},
                "corrections_made": [],
                "enhanced_query": query,
                "error": str(e)
            }
    
    def _fix_typos(self, query: str) -> Tuple[str, List[str]]:
        """
        Fix common typos using predefined correction dictionary.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (corrected_query, list of corrections made)
        """
        corrections = []
        words = query.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower()
            # Strip punctuation for matching
            word_clean = re.sub(r'[^\w\s]', '', word_lower)
            
            if word_clean in self.TYPO_CORRECTIONS:
                corrected = self.TYPO_CORRECTIONS[word_clean]
                corrected_words.append(corrected)
                corrections.append(f"{word_clean} → {corrected}")
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words), corrections
    
    def _fuzzy_match_entities(self, query: str) -> Tuple[str, List[str]]:
        """
        Use fuzzy matching to correct project and material names.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (corrected_query, list of corrections made)
        """
        corrections = []
        query_lower = query.lower()
        corrected = query
        
        # Extract potential project/material mentions (words longer than 4 chars)
        words = [w for w in query_lower.split() if len(w) > 4]
        
        for word in words:
            # Try to match projects (fuzzy, cutoff=0.6 means 60% similarity)
            project_matches = get_close_matches(word, self.PROJECTS, n=1, cutoff=0.6)
            if project_matches and word != project_matches[0]:
                # Replace in query (case-insensitive)
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                corrected = pattern.sub(project_matches[0], corrected)
                corrections.append(f"{word} → {project_matches[0]} (project)")
            
            # Try to match materials
            material_matches = get_close_matches(word, self.MATERIALS, n=1, cutoff=0.7)
            if material_matches and word != material_matches[0]:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                corrected = pattern.sub(material_matches[0], corrected)
                corrections.append(f"{word} → {material_matches[0]} (material)")
        
        return corrected, corrections
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract project names and material types from query.
        
        Args:
            query: Query string (already corrected)
            
        Returns:
            Dict with lists of projects and materials found
        """
        query_lower = query.lower()
        query_normalized = query_lower.replace(" ", "")
        
        found_projects = []
        found_materials = []
        
        # Find projects
        for project in self.PROJECTS:
            if project in query_lower or project in query_normalized:
                if project not in found_projects:
                    found_projects.append(project)
        
        # Find materials
        for material in self.MATERIALS:
            if material in query_lower:
                if material not in found_materials:
                    found_materials.append(material)
        
        return {
            "projects": found_projects,
            "materials": found_materials
        }
    
    def _enhance_query(self, query: str, entities: Dict[str, List[str]]) -> str:
        """
        Enhance query with synonyms and context (optional - can improve retrieval).
        
        Args:
            query: Corrected query
            entities: Extracted entities
            
        Returns:
            Enhanced query string
        """
        # For now, just return the corrected query
        # Could add synonym expansion here if needed
        # e.g., "price" → "price cost rate fee"
        return query
    
    def extract_mentioned_items(self, query: str, items: List[str]) -> List[str]:
        """
        Extract mentioned items from query with flexible matching.
        Handles typos and spaces in words.
        
        Args:
            query: Query string
            items: List of items to search for (lowercase)
            
        Returns:
            List of mentioned items found in the query
        """
        mentioned = []
        query_lower = query.lower()
        query_normalized = query_lower.replace(" ", "")
        
        for item in items:
            # Try exact match first
            if item in query_lower:
                mentioned.append(item)
            # Then try with flexible spaces
            elif item.replace(" ", "") in query_normalized:
                if item not in mentioned:
                    mentioned.append(item)
            # Try fuzzy matching for typos
            else:
                matches = get_close_matches(item, query_lower.split(), n=1, cutoff=0.75)
                if matches and item not in mentioned:
                    mentioned.append(item)
        
        return mentioned
