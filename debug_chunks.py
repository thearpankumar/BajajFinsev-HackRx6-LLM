#!/usr/bin/env python3
"""
Debug script to check what chunks are being generated from documents.
"""

import sys
import os
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.ingestion_service import IngestionService

async def debug_document_chunks():
    """Debug what chunks are being generated from a document."""
    
    print("🔍 Debug: Document Chunk Analysis\n")
    
    # Test URL (replace with your actual document URL)
    test_url = "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
    
    # Initialize ingestion service
    ingestion_service = IngestionService()
    
    try:
        print(f"📄 Processing document: {test_url}")
        
        # Process with fast mode settings
        chunks = await ingestion_service.process_and_extract(
            url=test_url,
            chunk_size=4000,
            overlap=400
        )
        
        print(f"✅ Generated {len(chunks)} chunks")
        print()
        
        # Show first 5 chunks
        print("📋 First 5 Chunks:")
        for i, chunk in enumerate(chunks[:5]):
            text = chunk['text']
            metadata = chunk['metadata']
            
            print(f"\n--- Chunk {i+1} ---")
            print(f"Length: {len(text)} characters")
            print(f"Page: {metadata.get('page_number', 'Unknown')}")
            print(f"Preview: {text[:200]}...")
            
            # Check if it contains Newton-related content
            newton_terms = ['newton', 'force', 'motion', 'gravity', 'law', 'principia']
            found_terms = [term for term in newton_terms if term.lower() in text.lower()]
            if found_terms:
                print(f"🎯 Contains: {', '.join(found_terms)}")
            else:
                print("⚠️  No obvious Newton-related terms found")
        
        # Test keyword matching
        print(f"\n🔍 Testing Keyword Matching:")
        test_question = "How does Newton demonstrate that gravity is inversely proportional to the square of the distance?"
        
        # Simple keyword matching test
        question_words = set(test_question.lower().split())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'how', 'does', 'that'}
        meaningful_words = question_words - stop_words
        
        print(f"Question: {test_question}")
        print(f"Keywords: {meaningful_words}")
        
        # Score chunks
        scored_chunks = []
        for i, chunk in enumerate(chunks[:20]):  # Test first 20 chunks
            chunk_text = chunk['text'].lower()
            
            score = 0
            for word in meaningful_words:
                if word in chunk_text:
                    score += 1
            
            if score > 0:
                scored_chunks.append((score, i, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        print(f"\n📊 Top 3 Matching Chunks:")
        for score, idx, chunk in scored_chunks[:3]:
            print(f"\nChunk {idx} (Score: {score}):")
            print(f"Page: {chunk['metadata'].get('page_number', 'Unknown')}")
            print(f"Preview: {chunk['text'][:150]}...")
        
        if not scored_chunks:
            print("⚠️  No chunks matched the keywords!")
            print("This explains why answers might be generic.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the debug analysis."""
    asyncio.run(debug_document_chunks())

if __name__ == "__main__":
    main()
