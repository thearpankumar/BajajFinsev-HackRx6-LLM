#!/usr/bin/env python3
"""
Test script for Direct Gemini implementation
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.direct_gemini_processor import DirectGeminiProcessor
from src.core.config import settings

async def test_direct_gemini():
    """Test the Direct Gemini processor"""
    
    print("🧪 Testing Direct Gemini Processor")
    print(f"Gemini Model: {settings.GOOGLE_MODEL}")
    
    # Create processor
    try:
        processor = DirectGeminiProcessor()
        print("✅ Direct Gemini processor created")
    except Exception as e:
        print(f"❌ Failed to create processor: {e}")
        return
    
    # Test document URL - using a simple PDF for testing
    test_document_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    
    # Test questions
    test_questions = [
        "What is this document about?",
        "What are the main topics covered?",
        "Is this a test document?"
    ]
    
    print(f"\nTesting with document: {test_document_url}")
    print(f"Questions: {len(test_questions)}")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")
    
    try:
        print("\n🚀 Starting analysis...")
        result = await processor.analyze_document(test_document_url, test_questions)
        
        print(f"\n✅ Analysis completed!")
        print(f"Status: {result.get('status')}")
        print(f"Processing time: {result.get('processing_time', 0):.2f}s")
        print(f"Method: {result.get('method')}")
        print(f"File type: {result.get('file_type')}")
        
        print(f"\nAnswers ({len(result.get('answers', []))}):")
        for i, answer in enumerate(result.get('answers', []), 1):
            print(f"\n{i}. Q: {test_questions[i-1]}")
            print(f"   A: {answer}")
            
        # Get stats
        stats = processor.get_stats()
        print(f"\nProcessor Stats:")
        print(f"  Total requests: {stats['stats']['total_requests']}")
        print(f"  Successful: {stats['stats']['successful_requests']}")
        print(f"  Failed: {stats['stats']['failed_requests']}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await processor.cleanup()
        print("\n🧹 Cleanup completed")

if __name__ == "__main__":
    # Check environment variables
    if not settings.GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY not set in environment")
        sys.exit(1)
    
    print(f"✅ Google API Key configured: {settings.GOOGLE_API_KEY[:10]}...")
    
    # Run test
    asyncio.run(test_direct_gemini())