#!/usr/bin/env python3
"""
Comprehensive Test for BajajFinsev RAG System Accuracy Issues
Tests the entire pipeline: Document Download -> PDF Processing -> Embedding -> Retrieval
"""

import asyncio
import sys
import os
import tempfile
import json
sys.path.insert(0, os.path.abspath('.'))

from src.services.document_downloader import DocumentDownloader
from src.services.pdf_processor import PDFProcessor
from src.services.basic_text_extractor import BasicTextExtractor
from src.services.embedding_service import EmbeddingService
from src.core.parallel_vector_store import ParallelVectorStore, VectorDocument
from src.core.parallel_document_processor import ParallelDocumentProcessor, ProcessingTask
from src.core.gpu_service import GPUService

async def test_document_download():
    """Test document downloading functionality"""
    print("🧪 Testing Document Download Pipeline...")
    
    # Test URLs with different file types
    test_urls = [
        ("https://httpbin.org/json", "JSON"),
        ("https://httpbin.org/html", "HTML"),
        ("https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf", "PDF")
    ]
    
    async with DocumentDownloader() as downloader:
        for url, description in test_urls:
            try:
                print(f"\n🔍 Testing {description} download: {url}")
                
                # First get file info
                info_result = await downloader.get_file_info(url)
                if info_result["status"] == "success":
                    print(f"✅ File info: {info_result['file_type']}, {info_result.get('file_size_mb', 'unknown')}MB")
                    
                    # Validate file
                    validation = info_result.get('validation', {})
                    if validation.get('overall_valid', False):
                        # Download the file
                        download_result = await downloader.download_document(url)
                        if download_result["status"] == "success":
                            print(f"✅ Download successful: {download_result['filename']}")
                            
                            # Test file exists and is readable
                            filepath = download_result['filepath']
                            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                                print(f"✅ File validation passed: {filepath}")
                            else:
                                print(f"❌ File validation failed: {filepath}")
                        else:
                            print(f"❌ Download failed: {download_result.get('error', 'Unknown error')}")
                    else:
                        print(f"⚠️ File validation failed: {validation.get('type_message', 'Unknown')}")
                else:
                    print(f"❌ File info failed: {info_result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Test failed for {description}: {str(e)}")

async def test_pdf_processing():
    """Test PDF processing with PyMuPDF"""
    print("\n🧪 Testing PDF Processing Pipeline...")
    
    processor = PDFProcessor()
    
    # Create a simple test PDF content
    test_content = "This is a test PDF document for accuracy testing."
    
    try:
        # Use a publicly available test PDF
        test_pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        
        async with DocumentDownloader() as downloader:
            print(f"🔄 Downloading test PDF from: {test_pdf_url}")
            download_result = await downloader.download_document(test_pdf_url)
            
            if download_result["status"] == "success":
                pdf_path = download_result["filepath"]
                print(f"✅ PDF downloaded: {pdf_path}")
                
                # Process the PDF
                print("🔄 Processing PDF with PDFProcessor...")
                result = await processor.process_pdf(
                    pdf_path, 
                    extract_tables=True, 
                    extract_images=True
                )
                
                if result["status"] == "success":
                    content = result.get("content", {})
                    full_text = content.get("full_text", "")
                    page_count = content.get("page_count", 0)
                    
                    print(f"✅ PDF processing successful:")
                    print(f"   - Pages: {page_count}")
                    print(f"   - Characters: {len(full_text)}")
                    print(f"   - Words: {len(full_text.split())}")
                    print(f"   - Text sample: {full_text[:100]}...")
                    
                    if len(full_text.strip()) > 50:
                        print("✅ PDF text extraction working properly")
                        return True
                    else:
                        print("❌ PDF text extraction returned minimal content")
                        return False
                else:
                    print(f"❌ PDF processing failed: {result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"❌ PDF download failed: {download_result.get('error', 'Unknown error')}")
                return False
                
    except Exception as e:
        print(f"❌ PDF processing test failed: {str(e)}")
        return False

async def test_embedding_pipeline():
    """Test embedding generation and vector storage"""
    print("\n🧪 Testing Embedding & Vector Storage Pipeline...")
    
    try:
        # Initialize GPU service
        gpu_service = GPUService()
        gpu_info = gpu_service.initialize()
        print(f"🎯 GPU Service: {gpu_info}")
        
        # Initialize embedding service
        embedding_service = EmbeddingService(gpu_service)
        embed_result = await embedding_service.initialize()
        
        if embed_result["status"] != "success":
            print(f"❌ Embedding service initialization failed: {embed_result.get('error', 'Unknown')}")
            return False
        
        print(f"✅ Embedding service initialized: {embed_result['model_name']}")
        
        # Test text embedding
        test_texts = [
            "This is a legal document about constitutional rights.",
            "The Indian Constitution guarantees fundamental rights.",
            "Banking regulations require compliance with RBI guidelines."
        ]
        
        print("🔄 Generating embeddings for test texts...")
        encode_result = await embedding_service.encode_texts(test_texts)
        
        if encode_result["status"] == "success":
            embeddings = encode_result["embeddings"]
            print(f"✅ Embeddings generated: {embeddings.shape} ({encode_result['dimension']}D)")
            
            # Test vector store
            print("🔄 Testing vector storage...")
            vector_store = ParallelVectorStore(embedding_service, gpu_service)
            store_result = await vector_store.initialize()
            
            if store_result["status"] == "success":
                print(f"✅ Vector store initialized: {store_result['index_type']}")
                
                # Create test documents
                documents = []
                for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
                    doc = VectorDocument(
                        doc_id=f"test_doc_{i}",
                        embedding=embedding,
                        metadata={"source": "test", "doc_type": "legal", "index": i},
                        text_content=text,
                        chunk_id=f"chunk_{i}"
                    )
                    documents.append(doc)
                
                # Add documents to vector store
                add_result = await vector_store.add_documents(documents)
                if add_result["status"] == "success":
                    print(f"✅ Documents added to vector store: {add_result['documents_added']}")
                    
                    # Test similarity search
                    search_result = await vector_store.search(
                        query_text="constitutional rights in India",
                        k=2
                    )
                    
                    if search_result["status"] == "success":
                        results = search_result["results"]
                        print(f"✅ Vector search successful: {len(results)} results")
                        
                        for i, result in enumerate(results):
                            print(f"   Result {i+1}: score={result.score:.3f}, text='{result.text_content[:50]}...'")
                        
                        return True
                    else:
                        print(f"❌ Vector search failed: {search_result.get('error', 'Unknown')}")
                        return False
                else:
                    print(f"❌ Document addition failed: {add_result.get('error', 'Unknown')}")
                    return False
            else:
                print(f"❌ Vector store initialization failed: {store_result.get('error', 'Unknown')}")
                return False
        else:
            print(f"❌ Embedding generation failed: {encode_result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Embedding pipeline test failed: {str(e)}")
        return False

async def test_parallel_document_processor():
    """Test the complete parallel document processing pipeline"""
    print("\n🧪 Testing Parallel Document Processor Pipeline...")
    
    try:
        # Initialize the processor
        processor = ParallelDocumentProcessor()
        init_result = await processor.initialize()
        
        if init_result["status"] != "success":
            print(f"❌ Processor initialization failed: {init_result.get('error', 'Unknown')}")
            return False
        
        print(f"✅ Parallel processor initialized with {init_result['worker_count']} workers")
        
        # Create test processing tasks
        test_urls = [
            "https://httpbin.org/json",
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
        ]
        
        tasks = []
        for i, url in enumerate(test_urls):
            task = ProcessingTask(
                task_id=f"test_task_{i}",
                document_url=url,
                target_language="en",
                processing_options={
                    "extract_tables": True,
                    "extract_images": True,
                    "chunk_size": 500
                }
            )
            tasks.append(task)
        
        # Process documents
        print("🔄 Processing documents in parallel...")
        results = await processor.process_documents_parallel(tasks)
        
        if results["status"] == "success":
            processed = results["results"]
            print(f"✅ Parallel processing completed: {len(processed)} documents processed")
            
            # Check individual results
            all_success = True
            for result in processed:
                if result["status"] == "success":
                    content = result.get("content", {})
                    text = content.get("full_text", "")
                    print(f"   ✅ {result['task_id']}: {len(text)} characters extracted")
                else:
                    print(f"   ❌ {result['task_id']}: {result.get('error', 'Unknown error')}")
                    all_success = False
            
            return all_success
        else:
            print(f"❌ Parallel processing failed: {results.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Parallel processor test failed: {str(e)}")
        return False

async def main():
    """Run comprehensive accuracy tests"""
    print("🚀 Starting BajajFinsev RAG System Comprehensive Accuracy Tests")
    print("=" * 80)
    
    test_results = {}
    
    try:
        # Test 1: Document Download
        print("\n" + "="*50)
        test_results["document_download"] = await test_document_download()
        
        # Test 2: PDF Processing
        print("\n" + "="*50)
        test_results["pdf_processing"] = await test_pdf_processing()
        
        # Test 3: Embedding Pipeline
        print("\n" + "="*50)
        test_results["embedding_pipeline"] = await test_embedding_pipeline()
        
        # Test 4: Parallel Document Processor
        print("\n" + "="*50)
        test_results["parallel_processor"] = await test_parallel_document_processor()
        
        # Results Summary
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE TEST RESULTS SUMMARY:")
        print("=" * 80)
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result is True)
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:25} : {status}")
        
        print(f"\n📈 OVERALL SCORE: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL ACCURACY TESTS PASSED! The RAG system is working properly.")
            return 0
        else:
            print("⚠️ Some tests failed. Check the output above for specific issues.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed with exception: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    print(f"\nTest suite completed with exit code: {exit_code}")
    sys.exit(exit_code)