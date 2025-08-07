#!/usr/bin/env python3
"""
Test script for JSON-only system
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from src.core.document_specific_matcher import DocumentSpecificMatcher

async def test_json_only_system():
    """Test the JSON-only document matching system"""
    print("🧪 Testing JSON-Only Document Matching System")
    print("=" * 50)
    
    # Initialize matcher
    print("1. Initializing DocumentSpecificMatcher...")
    matcher = DocumentSpecificMatcher("question.json")
    
    # Test document extraction
    print("\n2. Testing document name extraction...")
    test_url = "https://example.com/INDIAN_CONSTITUTION.pdf"
    doc_name, doc_key = matcher.extract_document_name_from_url(test_url)
    print(f"   URL: {test_url}")
    print(f"   Document name: {doc_name}")
    print(f"   Document key: {doc_key}")
    
    # Test document-specific matching
    print("\n3. Testing document-specific question matching...")
    test_question = "What is the official name of India according to Article 1 of the Constitution?"
    match = matcher.find_best_match_in_document(test_question, "INDIAN_CONSTITUTION")
    
    if match:
        print(f"   ✅ Found match!")
        print(f"   Q: {test_question}")
        print(f"   A: {match['answer'][:100]}...")
    else:
        print(f"   ❌ No match found")
    
    # Test default section matching
    print("\n4. Testing default section matching...")
    default_question = "What is the capital of Australia?"
    default_match = matcher.find_best_match_in_default(default_question)
    
    if default_match:
        print(f"   ✅ Found match in default!")
        print(f"   Q: {default_question}")
        print(f"   A: {default_match['answer'][:100]}...")
    else:
        print(f"   ❌ No match found in default")
    
    # Test full analysis
    print("\n5. Testing full document analysis...")
    test_questions = [
        "What is the official name of India according to Article 1 of the Constitution?",
        "What is the capital of Australia?",
        "This question should not match anything"
    ]
    
    result = await matcher.analyze_document(
        "https://example.com/INDIAN_CONSTITUTION.pdf",
        test_questions
    )
    
    print(f"   ✅ Analysis completed!")
    print(f"   JSON matches: {result.get('json_matches', 0)}")
    print(f"   Default matches: {result.get('default_matches', 0)}")
    print(f"   No answers: {result.get('no_answers', 0)}")
    print(f"   Total answers: {len(result.get('answers', []))}")
    
    # Print answers
    print("\n6. Generated answers:")
    for i, (question, answer) in enumerate(zip(test_questions, result.get('answers', [])), 1):
        print(f"   Q{i}: {question}")
        print(f"   A{i}: {answer[:100]}{'...' if len(answer) > 100 else ''}")
        print()
    
    # Test statistics
    print("7. System statistics:")
    stats = matcher.get_stats()
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Total questions: {stats['total_questions']}")
    print(f"   Document keys: {stats['document_keys'][:5]}...")  # Show first 5
    print(f"   Search scope: {stats['search_scope']}")
    print(f"   Features: {len(stats['features'])} features")
    
    print("\n✅ JSON-Only system test completed successfully!")
    print("🎉 The system now uses ONLY question.json - no RAG functionality!")

if __name__ == "__main__":
    asyncio.run(test_json_only_system())