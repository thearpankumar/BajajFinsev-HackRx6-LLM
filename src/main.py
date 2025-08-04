"""
BajajFinsev RAG System - Main FastAPI Application
High-accuracy document analysis for Insurance, Legal, HR, and Compliance domains
"""

import os
import time
import json
import warnings
from typing import Optional
from contextlib import asynccontextmanager

# Suppress multiprocessing warnings if needed
if os.getenv('SUPPRESS_MULTIPROCESSING_WARNINGS', 'false').lower() == 'true':
    warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Set multiprocessing start method to reduce semaphore leaks
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.config import settings
from src.core.rag_engine import RAGEngine
from src.core.enhanced_document_processor import EnhancedDocumentProcessor
from src.core.performance_monitor import PerformanceMonitor
from src.models.schemas import (
    AnalysisRequest,
    StreamResponse,
    HealthResponse,
    PerformanceMetrics,
)

# Global instances
rag_engine: Optional[RAGEngine] = None
doc_processor: Optional[EnhancedDocumentProcessor] = None
performance_monitor = PerformanceMonitor()
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global rag_engine, doc_processor

    # Startup
    print("🚀 Initializing BajajFinsev RAG System...")

    try:
        # Initialize components
        print("📄 Initializing enhanced document processor...")
        doc_processor = EnhancedDocumentProcessor()
        print("✅ Enhanced document processor initialized")

        print("🧠 Initializing RAG engine...")
        rag_engine = RAGEngine()

        # Initialize vector database and models
        print("🗄️ Initializing vector database...")
        await rag_engine.initialize()
        print("✅ Vector database initialized")

        print("✅ RAG System initialized successfully!")

    except Exception as e:
        print(f"❌ Failed to initialize RAG System: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

    yield

    # Cleanup
    print("🔄 Shutting down RAG System...")
    try:
        if rag_engine:
            # Use asyncio.wait_for to prevent hanging
            import asyncio
            await asyncio.wait_for(rag_engine.cleanup(), timeout=10.0)
        
        # Simple garbage collection
        import gc
        gc.collect()
        
        print("✅ Shutdown complete!")
    except asyncio.TimeoutError:
        print("⚠️ Cleanup timed out, forcing shutdown")
    except Exception as e:
        print(f"⚠️ Error during shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title="BajajFinsev Advanced Document Analysis API",
    description="High-performance, enterprise-grade API for deep analysis of business documents",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests before processing"""

    # Log request details
    print(f"\n{'=' * 80}")
    print("📥 INCOMING REQUEST")
    print(f"{'=' * 80}")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers: {dict(request.headers)}")

    # Read and log request body
    if request.method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
        if body:
            try:
                body_json = json.loads(body.decode())
                print("Request Body:")
                print(json.dumps(body_json, indent=2))
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"Request Body (raw): {body.decode()}")

        # Important: Create a new request with the body for downstream processing
        async def receive():
            return {"type": "http.request", "body": body}

        request._receive = receive

    print(f"{'=' * 80}\n")

    # Process the request
    response = await call_next(request)

    # Log response
    print(f"📤 RESPONSE STATUS: {response.status_code}")

    return response


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication"""
    print(f"🔐 Verifying API key: {credentials.credentials[:10]}...")

    if credentials.credentials != settings.API_KEY:
        print("❌ Invalid API key provided")
        raise HTTPException(status_code=401, detail="Invalid API key")

    print("✅ API key verified successfully")
    return credentials.credentials


@app.post("/api/v1/hackrx/run")
async def analyze_document(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    """
    Main endpoint for document analysis
    Returns only the answers array with concise responses
    """
    print("\n🔍 STARTING DOCUMENT ANALYSIS")
    print(f"Document URL: {request.documents}")
    print(f"Number of questions: {len(request.questions)}")
    print("Questions:")
    for i, q in enumerate(request.questions, 1):
        print(f"  {i}. {q}")

    start_time = time.time()

    try:
        # Track performance
        performance_monitor.start_request()

        print("\n⚡ Processing document and generating answers...")

        # Process document and get answers
        result = await rag_engine.analyze_document(
            document_url=str(request.documents), questions=request.questions
        )

        processing_time = time.time() - start_time

        print(f"\n✅ Analysis completed in {processing_time:.2f} seconds")
        print(f"Generated {len(result['answers'])} answers")

        # Log performance metrics
        background_tasks.add_task(
            performance_monitor.log_request,
            processing_time=processing_time,
            document_size=result.get("document_size", 0),
            num_questions=len(request.questions),
            success=True,
        )

        # Return only the answers array as requested
        response = {"answers": result["answers"]}

        print("\n📋 FINAL ANSWERS:")
        for i, answer in enumerate(response["answers"], 1):
            print(f"\n{i}. Q: {request.questions[i - 1]}")
            print(f"   A: {answer}")

        return response

    except Exception as e:
        processing_time = time.time() - start_time

        print(f"\n❌ Analysis failed after {processing_time:.2f} seconds")
        print(f"Error: {str(e)}")

        # Log error
        background_tasks.add_task(
            performance_monitor.log_request,
            processing_time=processing_time,
            num_questions=len(request.questions),
            success=False,
            error=str(e),
        )

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/hackrx/stream", response_model=StreamResponse)
async def stream_analysis(
    request: AnalysisRequest, api_key: str = Depends(verify_api_key)
):
    """
    Streaming endpoint for faster initial responses
    Returns quick answers while detailed analysis continues
    """
    try:
        print("\n🌊 STREAMING ANALYSIS STARTED")
        print(f"Document: {request.documents}")
        print(f"Questions: {len(request.questions)}")

        # Start streaming analysis
        stream_result = await rag_engine.stream_analyze(
            document_url=str(request.documents), questions=request.questions
        )

        return StreamResponse(
            initial_answers=stream_result["initial_answers"],
            status="processing",
            estimated_completion_time=stream_result.get("eta", 30),
        )

    except Exception as e:
        print(f"❌ Streaming analysis failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Streaming analysis failed: {str(e)}"
        )


@app.get("/api/v1/hackrx/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        print("\n🏥 HEALTH CHECK")

        # Check system components
        vector_db_status = (
            await rag_engine.check_vector_db_health() if rag_engine else False
        )
        model_status = await rag_engine.check_model_health() if rag_engine else False

        overall_status = (
            "healthy" if (vector_db_status and model_status) else "degraded"
        )

        print(f"Vector DB: {'✅' if vector_db_status else '❌'}")
        print(f"Models: {'✅' if model_status else '❌'}")
        print(f"Overall: {overall_status}")

        return HealthResponse(
            status=overall_status,
            components={
                "vector_database": "healthy" if vector_db_status else "unhealthy",
                "embedding_model": "healthy" if model_status else "unhealthy",
                "generation_model": "healthy" if model_status else "unhealthy",
            },
            timestamp=time.time(),
        )

    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy", components={}, timestamp=time.time(), error=str(e)
        )


@app.get("/api/v1/hackrx/performance", response_model=PerformanceMetrics)
async def get_performance_metrics(api_key: str = Depends(verify_api_key)):
    """Get detailed performance metrics"""
    print("\n📊 PERFORMANCE METRICS REQUESTED")
    return performance_monitor.get_metrics()


@app.post("/api/v1/hackrx/performance/reset")
async def reset_performance_metrics(api_key: str = Depends(verify_api_key)):
    """Reset performance counters"""
    print("\n🔄 RESETTING PERFORMANCE METRICS")
    performance_monitor.reset_metrics()
    return {"message": "Performance metrics reset successfully"}


@app.post("/api/v1/hackrx/performance/complex-questions")
async def set_complex_question_mode(
    fast_mode: bool = True, 
    api_key: str = Depends(verify_api_key)
):
    """Set complex question processing mode: fast or comprehensive"""
    print(f"\n⚡ SETTING COMPLEX QUESTION MODE: {'fast' if fast_mode else 'comprehensive'}")

    settings.FAST_COMPLEX_QUESTIONS = fast_mode
    
    if fast_mode:
        settings.ENABLE_RERANKING = False  # Disable reranking for speed
        return {
            "message": "Fast complex question mode enabled", 
            "mode": "fast",
            "reranking_disabled": True,
            "expected_speedup": "2-3x faster"
        }
    else:
        settings.ENABLE_RERANKING = True   # Enable reranking for accuracy
        return {
            "message": "Comprehensive complex question mode enabled", 
            "mode": "comprehensive",
            "reranking_enabled": True,
            "expected_quality": "higher accuracy"
        }


@app.post("/api/v1/hackrx/performance/mode")
async def set_performance_mode(mode: str, api_key: str = Depends(verify_api_key)):
    """Set performance mode: 'fast' or 'accurate'"""
    print(f"\n⚡ SETTING PERFORMANCE MODE: {mode}")

    if mode.lower() == "fast":
        settings.FAST_MODE = True
        settings.ENABLE_RERANKING = False
        settings.MAX_CHUNKS_FOR_GENERATION = 3
        return {"message": "Fast mode enabled", "mode": "fast"}
    elif mode.lower() == "accurate":
        settings.FAST_MODE = False
        settings.ENABLE_RERANKING = True
        settings.MAX_CHUNKS_FOR_GENERATION = 10
        return {"message": "Accurate mode enabled", "mode": "accurate"}
    else:
        raise HTTPException(status_code=400, detail="Mode must be 'fast' or 'accurate'")


@app.get("/api/v1/hackrx/performance/mode")
async def get_performance_mode(api_key: str = Depends(verify_api_key)):
    """Get current performance mode"""
    mode = "fast" if settings.FAST_MODE else "accurate"
    return {
        "mode": mode,
        "fast_mode": settings.FAST_MODE,
        "enable_reranking": settings.ENABLE_RERANKING,
        "max_chunks_for_generation": settings.MAX_CHUNKS_FOR_GENERATION,
        "parallel_processing": settings.PARALLEL_PROCESSING,
        "max_parallel_questions": settings.MAX_PARALLEL_QUESTIONS,
        "question_batch_size": settings.QUESTION_BATCH_SIZE,
    }


@app.get("/api/v1/hackrx/processing/stats")
async def get_processing_stats(api_key: str = Depends(verify_api_key)):
    """Get document processing statistics and capabilities"""
    try:
        print("\n📊 PROCESSING STATS REQUESTED")
        
        if not doc_processor:
            raise HTTPException(status_code=503, detail="Document processor not initialized")
        
        stats = await doc_processor.get_processing_stats()
        
        print(f"Enhanced processing available: {'✅' if stats.get('enhanced_available') else '❌'}")
        print(f"LlamaIndex features: {stats}")
        
        return {
            "processing_stats": stats,
            "current_processor": "EnhancedDocumentProcessor",
            "fallback_available": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"❌ Failed to get processing stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get processing stats: {str(e)}")


@app.get("/api/v1/hackrx/processing/test")
async def test_enhanced_processing(api_key: str = Depends(verify_api_key)):
    """Test enhanced processing capabilities"""
    try:
        print("\n🧪 TESTING ENHANCED PROCESSING")
        
        if not doc_processor:
            raise HTTPException(status_code=503, detail="Document processor not initialized")
        
        # Check if enhanced processing is available
        enhanced_available = doc_processor.is_enhanced_available()
        
        test_results = {
            "enhanced_available": enhanced_available,
            "llamaindex_installed": enhanced_available,
            "processor_type": "EnhancedDocumentProcessor",
            "fallback_processor": "DocumentProcessor",
            "test_status": "passed" if enhanced_available else "fallback_mode"
        }
        
        if enhanced_available:
            print("✅ Enhanced processing with LlamaIndex is available")
        else:
            print("⚠️ LlamaIndex not available, will use fallback processor")
        
        return test_results
        
    except Exception as e:
        print(f"❌ Enhanced processing test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


@app.get("/api/v1/hackrx/cache/stats")
async def get_cache_stats(api_key: str = Depends(verify_api_key)):
    """Get comprehensive document cache statistics"""
    try:
        print("\n📊 CACHE STATISTICS REQUESTED")
        
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Get persistent cache stats
        persistent_cache_stats = {}
        if rag_engine.document_cache_manager:
            persistent_cache_stats = await rag_engine.document_cache_manager.get_cache_stats()
        
        # Get memory cache stats
        memory_cache_stats = {
            "memory_cache_documents": len(rag_engine.document_cache),
            "memory_chunk_cache_documents": len(rag_engine.chunk_cache),
        }
        
        # Get vector database stats
        vector_db_stats = {}
        if rag_engine.vector_store:
            vector_db_stats = await rag_engine.vector_store.get_stats()
        
        return {
            "persistent_cache": persistent_cache_stats,
            "memory_cache": memory_cache_stats,
            "vector_database": vector_db_stats,
            "performance_stats": rag_engine.stats,
            "caching_enabled": {
                "persistent_cache": settings.ENABLE_PERSISTENT_DOCUMENT_CACHE,
                "check_vector_db": settings.CHECK_VECTOR_DB_BEFORE_DOWNLOAD,
                "skip_duplicates": settings.SKIP_DUPLICATE_DOCUMENTS
            }
        }
        
    except Exception as e:
        print(f"❌ Failed to get cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@app.post("/api/v1/hackrx/cache/clear")
async def clear_document_cache(api_key: str = Depends(verify_api_key)):
    """Clear all document caches"""
    try:
        print("\n🗑️ CLEARING DOCUMENT CACHE")
        
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Clear persistent cache
        persistent_cleared = False
        if rag_engine.document_cache_manager:
            persistent_cleared = await rag_engine.document_cache_manager.clear_cache()
        
        # Clear memory caches
        rag_engine.document_cache.clear()
        rag_engine.chunk_cache.clear()
        
        print("✅ All caches cleared successfully")
        
        return {
            "message": "All document caches cleared successfully",
            "persistent_cache_cleared": persistent_cleared,
            "memory_cache_cleared": True
        }
        
    except Exception as e:
        print(f"❌ Failed to clear cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.delete("/api/v1/hackrx/cache/document")
async def remove_document_from_cache(
    document_url: str, 
    api_key: str = Depends(verify_api_key)
):
    """Remove specific document from all caches"""
    try:
        print(f"\n🗑️ REMOVING DOCUMENT FROM CACHE: {document_url}")
        
        if not rag_engine:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
        
        # Remove from persistent cache
        persistent_removed = False
        if rag_engine.document_cache_manager:
            persistent_removed = await rag_engine.document_cache_manager.remove_document(document_url)
        
        # Remove from memory caches
        memory_removed = False
        if document_url in rag_engine.document_cache:
            del rag_engine.document_cache[document_url]
            memory_removed = True
        
        if document_url in rag_engine.chunk_cache:
            del rag_engine.chunk_cache[document_url]
        
        # Optionally remove from vector database
        try:
            await rag_engine.vector_store.delete_by_source(document_url)
            vector_db_removed = True
        except Exception as e:
            print(f"⚠️ Failed to remove from vector database: {str(e)}")
            vector_db_removed = False
        
        print(f"✅ Document removed from cache: {document_url}")
        
        return {
            "message": f"Document removed from cache: {document_url}",
            "persistent_cache_removed": persistent_removed,
            "memory_cache_removed": memory_removed,
            "vector_db_removed": vector_db_removed
        }
        
    except Exception as e:
        print(f"❌ Failed to remove document from cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to remove document: {str(e)}")


# Root endpoint for testing
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BajajFinsev RAG System is running!",
        "version": "1.0.0",
        "processor": "EnhancedDocumentProcessor with LlamaIndex",
        "authentication": "Bearer token required for all endpoints",
        "endpoints": {
            "analyze": "/api/v1/hackrx/run",
            "stream": "/api/v1/hackrx/stream",
            "health": "/api/v1/hackrx/health",
            "performance": "/api/v1/hackrx/performance",
            "performance_mode": "/api/v1/hackrx/performance/mode",
            "complex_questions": "/api/v1/hackrx/performance/complex-questions",
            "processing_stats": "/api/v1/hackrx/processing/stats",
            "processing_test": "/api/v1/hackrx/processing/test",
            "cache_stats": "/api/v1/hackrx/cache/stats",
            "cache_clear": "/api/v1/hackrx/cache/clear",
            "cache_remove": "/api/v1/hackrx/cache/document",
        },
        "note": "All endpoints require Authorization: Bearer <token> header"
    }


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
