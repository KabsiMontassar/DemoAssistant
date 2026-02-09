#!/usr/bin/env python3
"""
Test script to validate query intent detection logic.
Tests various query types and verifies correct intent classification.
"""

import sys
from enum import Enum

class QueryIntent(Enum):
    """Enumeration of query intent types for targeted retrieval strategies."""
    SPECIFICATION = "specification"      # Specific material/project query
    COMPARISON = "comparison"            # Comparing multiple items
    CATEGORY = "category"                # Asking about a specific material category
    GENERAL = "general"                  # General information query


def extract_mentioned_items(query_lower: str, items: list[str]) -> list[str]:
    """Extract mentioned items from query with fuzzy matching."""
    mentioned = []
    query_normalized = query_lower.replace(" ", "")
    
    for item in items:
        if item in query_lower:
            mentioned.append(item)
        elif item.replace(" ", "") in query_normalized:
            if item not in mentioned:
                mentioned.append(item)
    
    return mentioned


def detect_query_intent(query_lower: str) -> QueryIntent:
    """Detect the intent of the user's query."""
    # Count mentioned items
    mentioned_projects = len(extract_mentioned_items(query_lower, ['projectacme', 'projectfacebook']))
    mentioned_categories = len(extract_mentioned_items(query_lower, ['concrete', 'metal', 'stone', 'wood']))
    
    # Comparison keywords
    comparison_keywords = ['compare', 'vs', 'versus', 'difference', 'between', 'rather', 'instead']
    has_comparison_keyword = any(keyword in query_lower for keyword in comparison_keywords)
    
    # Specification keywords
    spec_keywords = ['what', 'price', 'cost', 'how much', 'specific', 'particular']
    has_spec_keyword = any(keyword in query_lower for keyword in spec_keywords)
    
    # Determination logic
    if (mentioned_categories >= 2) or has_comparison_keyword:
        return QueryIntent.COMPARISON
    
    if mentioned_projects >= 1 and mentioned_categories >= 1 and has_spec_keyword:
        return QueryIntent.SPECIFICATION
    
    if mentioned_categories >= 1 and mentioned_projects == 0:
        return QueryIntent.CATEGORY
    
    return QueryIntent.GENERAL


# Test cases with expected intents
test_cases = [
    # SPECIFICATION queries
    ("What is the wood price in Project Acme?", QueryIntent.SPECIFICATION, "Specific project + material + price question"),
    ("How much does concrete cost in projectFacebook?", QueryIntent.SPECIFICATION, "Specific project + material + cost question"),
    ("Tell me the metal specifications for projectAcme", QueryIntent.SPECIFICATION, "Specific project + material + specification"),
    
    # COMPARISON queries
    ("Compare concrete vs stone prices", QueryIntent.COMPARISON, "Two materials - explicit comparison"),
    ("What's the difference between wood and metal?", QueryIntent.COMPARISON, "Multiple materials with comparison keyword"),
    ("Metal or wood for projectAcme?", QueryIntent.COMPARISON, "Multiple materials - either/or query"),
    ("Compare prices across projects", QueryIntent.COMPARISON, "Comparison keyword present"),
    
    # CATEGORY queries (no specific project)
    ("Show me all wood prices", QueryIntent.CATEGORY, "Material type without specific project"),
    ("Tell me about stone materials", QueryIntent.CATEGORY, "Material category query"),
    ("What concrete options do we have?", QueryIntent.CATEGORY, "Category focused, no specific project"),
    
    # GENERAL queries
    ("Tell me about our pricing", QueryIntent.GENERAL, "General information - no specifics"),
    ("What do we need to know?", QueryIntent.GENERAL, "General question"),
    ("How's the project going?", QueryIntent.GENERAL, "General question unrelated to materials"),
]

def main():
    """Run test cases and report results."""
    passed = 0
    failed = 0
    
    print("=" * 80)
    print("QUERY INTENT DETECTION TEST SUITE")
    print("=" * 80)
    
    for query, expected_intent, description in test_cases:
        query_lower = query.lower()
        detected_intent = detect_query_intent(query_lower)
        
        # Get details for logging
        mentioned_projects = extract_mentioned_items(query_lower, ['projectacme', 'projectfacebook'])
        mentioned_categories = extract_mentioned_items(query_lower, ['concrete', 'metal', 'stone', 'wood'])
        
        passed_test = detected_intent == expected_intent
        status = "✓ PASS" if passed_test else "✗ FAIL"
        
        print(f"\n{status}")
        print(f"Query: \"{query}\"")
        print(f"Description: {description}")
        print(f"Expected: {expected_intent.value}")
        print(f"Detected: {detected_intent.value}")
        print(f"Projects mentioned: {mentioned_projects}")
        print(f"Categories mentioned: {mentioned_categories}")
        
        if passed_test:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
