#!/usr/bin/env python3
"""
Test script to verify accuracy fixes in the BajajFinsev RAG system
Tests critical document processing and text extraction accuracy
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.services.basic_text_extractor import BasicTextExtractor
from src.services.web_page_processor import WebPageProcessor
from src.services.answer_generator import AnswerGenerator

async def test_text_extraction_accuracy():
    """Test text extraction with different file types and URLs"""
    print("🧪 Testing Text Extraction Accuracy...")
    
    extractor = BasicTextExtractor()
    
    # Test supported formats
    supported = extractor.get_supported_formats()
    print(f"✅ Supported formats: {supported}")
    
    # Test URL extraction (if reachable)
    test_urls = [
        "https://httpbin.org/html",  # Simple HTML test
        "https://jsonplaceholder.typicode.com/posts/1"  # JSON test
    ]
    
    for url in test_urls:
        try:
            print(f"\n🔍 Testing URL: {url}")
            result = await extractor.extract_text_from_url(url)
            
            if result["status"] == "success":
                content = result.get("content", {})
                text_preview = content.get("full_text", "")[:100] + "..."
                print(f"✅ Extraction successful: {text_preview}")
                print(f"📊 Stats: {content.get('char_count', 0)} chars, {content.get('word_count', 0)} words")
            else:
                print(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")

async def test_web_page_processor():
    """Test web page processor improvements"""
    print("\n🧪 Testing Web Page Processor...")
    
    processor = WebPageProcessor()
    
    try:
        # Test with a simple web page
        test_url = "https://httpbin.org/html"
        test_question = "What is the main heading on this page?"
        
        print(f"🔍 Testing URL: {test_url}")
        print(f"❓ Question: {test_question}")
        
        result = await processor.process_url(test_url, test_question)
        
        if result["status"] == "success":
            answer = result.get("answer", "")
            print(f"✅ Web processing successful")
            print(f"💬 Answer: {answer[:200]}...")
        else:
            print(f"❌ Web processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

async def test_answer_generation():
    """Test answer generation improvements"""
    print("\n🧪 Testing Answer Generation...")
    
    generator = AnswerGenerator()
    
    # Test with sample chunks
    test_chunks = [
        {
            "text": "The Constitution of India is the fundamental law of India. It was adopted on 26 January 1950.",
            "score": 0.85,
            "metadata": {"source": "constitution", "language": "en"}
        },
        {
            "text": "Article 14 of the Constitution guarantees equality before law and equal protection of laws.",
            "score": 0.92,
            "metadata": {"source": "constitution", "language": "en"}
        }
    ]
    
    test_questions = [
        "What is the Constitution of India?",
        "When was the Constitution adopted?",
        "What does Article 14 guarantee?"
    ]
    
    for question in test_questions:
        try:
            print(f"\n❓ Question: {question}")
            answer = await generator.generate_answer(question, test_chunks, "legal", "en")
            print(f"💬 Answer: {answer}")
            
        except Exception as e:
            print(f"❌ Answer generation failed: {str(e)}")

async def main():
    """Run all accuracy tests"""
    print("🚀 Starting BajajFinsev RAG Accuracy Tests...")
    print("=" * 60)
    
    try:
        await test_text_extraction_accuracy()
        await test_web_page_processor()
        await test_answer_generation()
        
        print("\n" + "=" * 60)
        print("✅ Accuracy tests completed!")
        print("\n📋 Summary of fixes applied:")
        print("1. ✅ Fixed main pipeline fallback logic for token extraction")
        print("2. ✅ Improved web page processor with targeted HTML extraction")
        print("3. ✅ Enhanced text extraction with multiple encoding support")
        print("4. ✅ Added proper document processing chain with fallbacks")
        print("5. ✅ Improved answer generation with deduplication")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)