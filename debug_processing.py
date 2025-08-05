#!/usr/bin/env python3
"""
Debug script to identify document processing issues
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.high_performance_rag_engine import HighPerformanceRAGEngine
from src.core.config import settings

async def debug_document_processing():
    """Debug the document processing pipeline"""
    
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    test_questions = [
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?",
        "How does the policy define a 'Hospital'?"
    ]
    
    print("🔍 DEBUG: Initializing RAG Engine...")
    rag_engine = HighPerformanceRAGEngine()
    
    try:
        await rag_engine.initialize()
        print("✅ RAG Engine initialized successfully")
        
        # Step 1: Test document download
        print(f"\n📥 Step 1: Testing document download...")
        print(f"Document URL: {document_url}")
        
        try:
            document_data, content_type = await rag_engine.document_processor._download_document(document_url)
            print(f"✅ Document downloaded: {len(document_data)} bytes, type: {content_type}")
        except Exception as e:
            print(f"❌ Document download failed: {str(e)}")
            return
        
        # Step 2: Test document processing
        print(f"\n🏗️ Step 2: Testing document processing...")
        try:
            chunks, doc_metadata = await rag_engine.document_processor.process_document(document_url)
            print(f"✅ Document processed: {len(chunks)} chunks created")
            print(f"📊 Metadata: {doc_metadata}")
            
            # Show first few chunks
            print(f"\n📄 First 3 chunks:")
            for i, chunk in enumerate(chunks[:3]):
                print(f"Chunk {i+1}: {chunk.text[:200]}...")
                print(f"  Metadata: {chunk.metadata}")
                
        except Exception as e:
            print(f"❌ Document processing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 3: Test retrieval initialization
        print(f"\n🔍 Step 3: Testing retrieval initialization...")
        try:
            await rag_engine.retrieval_engine.initialize(chunks, document_url)
            print("✅ Retrieval engine initialized")
        except Exception as e:
            print(f"❌ Retrieval initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 4: Test indexing
        print(f"\n📚 Step 4: Testing chunk indexing...")
        try:
            await rag_engine._index_chunks_optimized(chunks, document_url)
            print("✅ Chunks indexed successfully")
        except Exception as e:
            print(f"❌ Chunk indexing failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 5: Test search
        print(f"\n🔎 Step 5: Testing search functionality...")
        for i, question in enumerate(test_questions):
            print(f"\n--- Testing Question {i+1}: {question}")
            try:
                search_results = await rag_engine.retrieval_engine.search(
                    question, k=settings.TOP_K_RETRIEVAL, document_url=document_url
                )
                print(f"✅ Search completed: {len(search_results)} results found")
                
                if search_results:
                    for j, result in enumerate(search_results[:3]):
                        print(f"  Result {j+1}: Score={result.score:.3f}")
                        print(f"    Text: {result.chunk.text[:150]}...")
                else:
                    print("⚠️ No search results found!")
                    
            except Exception as e:
                print(f"❌ Search failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Step 6: Test answer generation
        print(f"\n💬 Step 6: Testing answer generation...")
        for i, question in enumerate(test_questions[:1]):  # Test just one question
            print(f"\n--- Testing Answer Generation for: {question}")
            try:
                answer = await rag_engine._answer_single_question_optimized(
                    question, chunks, document_url
                )
                print(f"✅ Answer generated: {answer}")
                
            except Exception as e:
                print(f"❌ Answer generation failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎯 DEBUG COMPLETE")
        
    except Exception as e:
        print(f"❌ Overall error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        await rag_engine.cleanup()

if __name__ == "__main__":
    asyncio.run(debug_document_processing())