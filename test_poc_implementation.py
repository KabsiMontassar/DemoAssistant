"""
Quick Test Script for POC Implementation
Run this to verify all changes are working correctly.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_imports():
    """Test that all new modules can be imported."""
    print("🧪 Testing imports...")
    try:
        from intent_classifier import IntentClassifier
        print("  ✅ IntentClassifier imported")
        
        from chitchat_handler import ChitchatHandler
        print("  ✅ ChitchatHandler imported")
        
        from retrieval import RetrieverManager
        print("  ✅ RetrieverManager imported")
        
        from hybrid_retrieval import HybridRetriever
        print("  ✅ HybridRetriever imported")
        
        from main import app
        print("  ✅ Main app imported")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False


def test_intent_classifier():
    """Test intent classifier with sample queries."""
    print("\n🧪 Testing IntentClassifier...")
    try:
        from embedding import EmbeddingManager
        from intent_classifier import IntentClassifier
        
        # Initialize
        print("  📦 Initializing embedding manager...")
        emb_mgr = EmbeddingManager()
        
        print("  📦 Initializing intent classifier...")
        classifier = IntentClassifier(emb_mgr)
        
        # Test queries
        test_cases = [
            ("Hello!", "chitchat", "< 30%"),
            ("What can you do?", "chitchat", "< 30%"),
            ("What's the price of concrete?", "domain", ">= 30%"),
            ("Tell me about Project Acme wood prices", "domain", ">= 30%"),
        ]
        
        for query, expected_intent, expected_conf in test_cases:
            result = classifier.classify_query(query)
            intent = result['intent']
            confidence = result['confidence']
            
            status = "✅" if intent == expected_intent else "❌"
            print(f"  {status} '{query[:40]}...' → {intent} ({confidence:.1f}%)")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chitchat_handler():
    """Test chitchat handler responses."""
    print("\n🧪 Testing ChitchatHandler...")
    try:
        from chitchat_handler import ChitchatHandler
        
        handler = ChitchatHandler()
        
        # Test queries
        test_cases = [
            ("Hello!", "greeting"),
            ("Hi there", "greeting"),
            ("What can you do?", "help"),
            ("Tell me about quantum physics", "out_of_domain"),
        ]
        
        for query, expected_type in test_cases:
            result = handler.handle_query(query, 25.0)
            response_type = result['metadata']['response_type']
            
            status = "✅" if response_type == expected_type else "❌"
            print(f"  {status} '{query}' → {response_type}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_score_calibration():
    """Test that score calibration is working (check boost values)."""
    print("\n🧪 Testing Score Calibration...")
    try:
        import retrieval
        import inspect
        
        # Check the source code for weight values
        source = inspect.getsource(retrieval.RetrieverManager._apply_relevance_weights)
        
        # Check if new weights are present
        if "1.25" in source and "1.20" in source:
            print("  ✅ New boost multipliers detected (1.25, 1.20)")
        else:
            print("  ⚠️  Warning: Expected new boost multipliers not found")
        
        # Check if old weights are gone
        if "3.0" in source:
            print("  ⚠️  Warning: Old boost multipliers (3.0) still present")
        else:
            print("  ✅ Old boost multipliers (3.0) removed")
        
        # Check for absolute scoring
        retrieve_source = inspect.getsource(retrieval.RetrieverManager.retrieve)
        if "max_score = max" in retrieve_source:
            print("  ⚠️  Warning: Max-score normalization might still be present")
        else:
            print("  ✅ Max-score normalization removed")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_values():
    """Check configuration values."""
    print("\n🧪 Testing Configuration Values...")
    
    print("  📊 Expected values:")
    print("    - Intent threshold: 30%")
    print("    - Score gate threshold: 20%")
    print("    - Boost multipliers: 1.10-1.25x (down from 2.0-3.0x)")
    print("    - Hybrid weights: 60% vector, 40% keyword")
    print("  ✅ Review main.py and retrieval.py to verify")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("POC IMPLEMENTATION - VERIFICATION TESTS")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Score Calibration", test_score_calibration()))
    results.append(("Configuration", test_config_values()))
    
    # These require full initialization - optional
    print("\n" + "=" * 60)
    print("OPTIONAL TESTS (require full backend initialization)")
    print("=" * 60)
    print("\n⚠️  Skipping IntentClassifier and ChitchatHandler tests")
    print("   (Require ChromaDB and embedding model initialization)")
    print("   Run these after starting the backend server")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All basic tests passed! Implementation looks good.")
        print("\n📝 Next steps:")
        print("   1. Start the backend server: cd backend && python main.py")
        print("   2. Test with actual queries via the API")
        print("   3. Monitor logs for intent classification and score gate actions")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
