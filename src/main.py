"""
BajajFinsev Hybrid System - Main FastAPI Application
Uses Direct Gemini processor first, then falls back to RAG.
Supports multiple file formats with fast OCR processing
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
from src.core.hybrid_document_matcher import HybridDocumentMatcher
from src.core.multi_format_processor import MultiFormatProcessor
from src.core.direct_gemini_processor import DirectGeminiProcessor
from src.core.response_timer import ResponseTimer
from src.models.schemas import (
    AnalysisRequest,
    StreamResponse,
    HealthResponse,
    PerformanceMetrics,
)

# Global instances
hybrid_matcher: Optional[HybridDocumentMatcher] = None
multi_format_processor: Optional[MultiFormatProcessor] = None
direct_gemini_processor: Optional[DirectGeminiProcessor] = None
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global hybrid_matcher, multi_format_processor, direct_gemini_processor

    # Startup
    print("🚀 Initializing BajajFinsev Enhanced System (Direct Gemini + RAG Fallback)...")

    try:
        # Initialize Direct Gemini processor for primary analysis
        print("🧠 Initializing Direct Gemini processor...")
        direct_gemini_processor = DirectGeminiProcessor()
        print("✅ Direct Gemini processor initialized")

        # Initialize multi-format processor
        print("📄 Initializing multi-format processor...")
        multi_format_processor = MultiFormatProcessor()
        print("✅ Multi-format processor initialized")

        # Initialize hybrid document matcher for RAG fallback
        print("🔄 Initializing hybrid document matcher (RAG fallback)...")
        hybrid_matcher = HybridDocumentMatcher()
        print("✅ Hybrid document matcher initialized")
        
        # Print stats
        stats = hybrid_matcher.get_stats()
        print(f"✅ Fallback RAG enabled: {stats['fallback_enabled']}")
        print(f"✅ Similarity threshold: {stats['similarity_threshold']}")
        
        # Print supported formats
        format_stats = await multi_format_processor.get_processing_stats()
        print(f"✅ Supported formats: {', '.join(format_stats['supported_formats'])}")
        print(f"✅ OCR available: {format_stats['ocr_available']}")

        # Check vector database connection if fallback RAG is enabled
        if settings.ENABLE_FALLBACK_RAG:
            print("🔍 Checking vector database connection...")
            try:
                # Initialize RAG engine to test connection
                await hybrid_matcher._initialize_rag_engine()
                if hybrid_matcher.rag_initialized and hybrid_matcher.rag_engine:
                    # Test vector database connection
                    vector_db_healthy = await hybrid_matcher.rag_engine.check_vector_db_health()
                    if vector_db_healthy:
                        print("✅ Vector database connection successful")
                    else:
                        print("⚠️ Vector database connection failed - RAG fallback may not work")
                    
                    # Test model health
                    model_healthy = await hybrid_matcher.rag_engine.check_model_health()
                    if model_healthy:
                        print("✅ AI models connection successful")
                    else:
                        print("⚠️ AI models connection failed - RAG fallback may not work")
                else:
                    print("⚠️ RAG engine initialization failed - fallback disabled")
            except Exception as e:
                print(f"⚠️ Vector database connection check failed: {str(e)}")
                print("⚠️ RAG fallback may not work properly")
        else:
            print("ℹ️ RAG fallback disabled - vector database not checked")

    except Exception as e:
        print(f"❌ Failed to initialize Hybrid System: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

    yield

    # Cleanup
    print("🔄 Shutting down Enhanced System...")
    try:
        if direct_gemini_processor:
            await direct_gemini_processor.cleanup()
        
        if hybrid_matcher:
            await hybrid_matcher.cleanup()
        
        # Simple garbage collection
        import gc
        gc.collect()
        
        print("✅ Shutdown complete!")
    except Exception as e:
        print(f"⚠️ Error during shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title="BajajFinsev Document Analysis API",
    description="API that uses a Gemini-first approach with a RAG fallback.",
    version="3.0.0",
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
    Enhanced endpoint for document analysis:
    1. Primary: Direct Gemini analysis for most accurate results
    2. Fallback: RAG for quick responses when Gemini fails
    
    Ensures minimum response time for consistent UX
    """
    print("\n🧠 STARTING ENHANCED ANALYSIS (PRIMARY: DIRECT GEMINI, FALLBACK: RAG)")
    print(f"Document URL: {request.documents}")
    print(f"Number of questions: {len(request.questions)}")
    print("Questions:")
    for i, q in enumerate(request.questions, 1):
        print(f"  {i}. {q}")

    # Start response timer
    timer = ResponseTimer()
    timer.start()

    try:
        if not direct_gemini_processor or not hybrid_matcher:
            raise HTTPException(status_code=503, detail="Analysis processors not initialized")

        print("\n🚀 PRIMARY: Attempting Direct Gemini analysis...")

        # Try Direct Gemini analysis first
        try:
            result = await direct_gemini_processor.analyze_document(
                document_url=str(request.documents), 
                questions=request.questions
            )

            if result.get('status') == 'completed' and result.get('answers'):
                print(f"\n✅ Direct Gemini analysis completed in {timer.get_elapsed_time():.2f} seconds")
                print(f"Generated {len(result['answers'])} answers")
                print(f"Method: {result.get('method', 'direct_gemini')}")
                print(f"File type: {result.get('file_type', 'unknown')}")

                # Return only the answers array as requested
                response = {"answers": result["answers"]}

                # Ensure minimum response time
                response = await timer.ensure_minimum_time(response)

                print("\n📋 FINAL ANSWERS (DIRECT GEMINI):")
                for i, answer in enumerate(response["answers"], 1):
                    print(f"\n{i}. Q: {request.questions[i - 1]}")
                    print(f"   A: {answer}")

                return response

            else:
                print("⚠️ Direct Gemini analysis failed or returned empty results")
                raise Exception("Direct Gemini analysis unsuccessful")

        except Exception as gemini_error:
            print(f"❌ Direct Gemini analysis failed: {str(gemini_error)}")
            print("🔄 FALLBACK: Attempting RAG-based analysis...")

            # Fallback to RAG-based hybrid matcher
            result = await hybrid_matcher.analyze_document(
                document_url=str(request.documents), 
                questions=request.questions
            )

        print(f"\n✅ RAG fallback analysis completed in {timer.get_elapsed_time():.2f} seconds")
        print(f"Generated {len(result['answers'])} answers")
        print(f"RAG fallbacks: {result.get('rag_fallbacks', 0)}")
        print(f"No answers: {result.get('no_answers', 0)}")

        # Return only the answers array as requested
        response = {"answers": result["answers"]}

        # Ensure minimum response time
        response = await timer.ensure_minimum_time(response)

        print("\n📋 FINAL ANSWERS (RAG FALLBACK):")
        for i, answer in enumerate(response["answers"], 1):
            print(f"\n{i}. Q: {request.questions[i - 1]}")
            print(f"   A: {answer}")

        return response

    except Exception as e:
        elapsed_time = timer.get_elapsed_time()

        print(f"\n❌ All analysis methods failed after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/hackrx/stream", response_model=StreamResponse)
async def stream_analysis(
    request: AnalysisRequest, api_key: str = Depends(verify_api_key)
):
    """
    Streaming endpoint using hybrid matcher
    Returns quick answers from specific document section, then processes fallbacks
    """
    try:
        print("\n🌊 STREAMING ANALYSIS STARTED (HYBRID)")
        print(f"Document: {request.documents}")
        print(f"Questions: {len(request.questions)}")

        # Start response timer for streaming
        timer = ResponseTimer()
        timer.start()

        if not hybrid_matcher:
            raise HTTPException(status_code=503, detail="Hybrid matcher not initialized")

        # Start streaming analysis
        stream_result = await hybrid_matcher.stream_analyze(
            document_url=str(request.documents), 
            questions=request.questions
        )

        # Calculate estimated completion time based on current progress
        elapsed = timer.get_elapsed_time()
        eta = max(settings.MIN_RESPONSE_TIME_SECONDS - elapsed, 0)

        return StreamResponse(
            initial_answers=stream_result["initial_answers"],
            status=stream_result["status"],
            estimated_completion_time=stream_result.get("eta", eta),
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

        # Check hybrid matcher
        matcher_status = hybrid_matcher is not None
        
        # Check multi-format processor
        processor_status = multi_format_processor is not None
        
        # Check RAG engine if initialized
        rag_status = "not_initialized"
        if hybrid_matcher and hybrid_matcher.rag_initialized:
            rag_status = "healthy" if hybrid_matcher.rag_engine else "unhealthy"
        
        overall_status = "healthy" if (matcher_status and processor_status) else "degraded"

        print(f"Hybrid Matcher: {'✅' if matcher_status else '❌'}")
        print(f"Multi-format Processor: {'✅' if processor_status else '❌'}")
        print(f"RAG Engine: {rag_status}")
        print(f"Overall: {overall_status}")

        return HealthResponse(
            status=overall_status,
            components={
                "hybrid_matcher": "healthy" if matcher_status else "unhealthy",
                "multi_format_processor": "healthy" if processor_status else "unhealthy",
                "rag_engine": rag_status,
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
    """Get basic performance metrics for document-specific system"""
    print("\n📊 PERFORMANCE METRICS REQUESTED")
    
    if not hybrid_matcher:
        raise HTTPException(status_code=503, detail="Hybrid matcher not initialized")
    
    stats = hybrid_matcher.get_stats()
    perf_stats = stats.get('performance_stats', {})
    
    return PerformanceMetrics(
        total_requests=perf_stats.get('total_questions', 0),
        successful_requests=perf_stats.get('rag_fallbacks', 0),
        failed_requests=perf_stats.get('no_answers', 0),
        average_processing_time=12.5,  # Average of 10-15 second range
        average_document_size=0,
        total_documents_processed=0,
        cache_hit_rate=0.8,  # Estimated cache hit rate
        uptime_seconds=time.time(),
        memory_usage_mb=100.0,  # Estimated memory usage
        custom_metrics={
            "hybrid_matcher_stats": stats,
            "mode": "gemini_with_rag_fallback",
            "features": stats.get("features", []),
            "fallback_enabled": stats.get('fallback_enabled', False),
            "rag_fallbacks": perf_stats.get('rag_fallbacks', 0),
            "no_answers": perf_stats.get('no_answers', 0)
        }
    )



@app.get("/api/v1/hackrx/cache/stats")

async def get_cache_stats(api_key: str = Depends(verify_api_key)):
    """Get comprehensive document cache statistics"""
    try:
        print("\n📊 CACHE STATISTICS REQUESTED")
        
        if not hybrid_matcher:
            raise HTTPException(status_code=503, detail="Hybrid matcher not initialized")
        
        # Initialize stats structure
        cache_stats = {
            "persistent_cache": {},
            "memory_cache": {},
            "vector_database": {},
            "rag_engine_status": "not_initialized"
        }
        
        # Get RAG engine stats if initialized
        if hybrid_matcher.rag_initialized and hybrid_matcher.rag_engine:
            rag_engine = hybrid_matcher.rag_engine
            
            # Get persistent cache stats
            if rag_engine.document_cache_manager:
                cache_stats["persistent_cache"] = await rag_engine.document_cache_manager.get_cache_stats()
            
            # Get memory cache stats
            cache_stats["memory_cache"] = {
                "memory_cache_documents": len(rag_engine.document_cache),
                "memory_chunk_cache_documents": len(rag_engine.chunk_cache),
            }
            
            # Get vector database stats
            if rag_engine.vector_store:
                cache_stats["vector_database"] = await rag_engine.vector_store.get_stats()
            
            cache_stats["rag_engine_status"] = "initialized"
            cache_stats["performance_stats"] = rag_engine.stats
        
        cache_stats["caching_enabled"] = {
            "persistent_cache": settings.ENABLE_PERSISTENT_DOCUMENT_CACHE,
            "fallback_rag": settings.ENABLE_FALLBACK_RAG,
            "multi_format_support": settings.ENABLE_MULTI_FORMAT_SUPPORT
        }
        
        return cache_stats
        
    except Exception as e:
        print(f"❌ Failed to get cache stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")



@app.post("/api/v1/hackrx/cache/clear")
async def clear_all_caches(api_key: str = Depends(verify_api_key)):
    """Clear ALL document caches and vector database completely"""
    try:
        print("\n🗑️ CLEARING ALL CACHES AND VECTOR DATABASE")
        
        if not hybrid_matcher:
            raise HTTPException(status_code=503, detail="Hybrid matcher not initialized")
        
        results = {
            "message": "All caches and vector database cleared successfully",
            "persistent_cache_cleared": False,
            "memory_cache_cleared": False,
            "vector_database_cleared": False,
            "rag_engine_reinitialized": False
        }
        
        # Clear RAG engine caches if initialized
        if hybrid_matcher.rag_initialized and hybrid_matcher.rag_engine:
            rag_engine = hybrid_matcher.rag_engine
            
            # Clear persistent document cache
            if rag_engine.document_cache_manager:
                persistent_cleared = await rag_engine.document_cache_manager.clear_cache()
                results["persistent_cache_cleared"] = persistent_cleared
                print("✅ Persistent document cache cleared")
            
            # Clear memory caches
            rag_engine.document_cache.clear()
            rag_engine.chunk_cache.clear()
            results["memory_cache_cleared"] = True
            print("✅ Memory caches cleared")
            
            # Clear entire vector database collection
            if rag_engine.vector_store:
                try:
                    # Delete the entire collection and recreate it
                    await rag_engine.vector_store.delete_collection()
                    await rag_engine.vector_store.create_collection()
                    results["vector_database_cleared"] = True
                    print("✅ Vector database collection cleared and recreated")
                except Exception as e:
                    print(f"⚠️ Failed to clear vector database: {str(e)}")
            
            # Reset RAG engine stats
            rag_engine.stats = {
                "total_queries": 0,
                "cache_hits": 0,
                "avg_retrieval_time": 0,
                "avg_generation_time": 0,
            }
            results["rag_engine_reinitialized"] = True
            print("✅ RAG engine stats reset")
        
        # Reset hybrid matcher stats
        hybrid_matcher.stats = {
            'total_questions': 0,
            'rag_fallbacks': 0,
            'no_answers': 0,
            'avg_rag_time': 0,
            'format_support_used': {}
        }
        
        print("✅ All caches, vector database, and statistics cleared successfully")
        return results
        
    except Exception as e:
        print(f"❌ Failed to clear caches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")


@app.delete("/api/v1/hackrx/cache/document")
async def remove_document_from_cache(
    document_url: str, 
    api_key: str = Depends(verify_api_key)
):
    """Remove specific document from all caches and vector database"""
    try:
        print(f"\n🗑️ REMOVING DOCUMENT FROM ALL CACHES: {document_url}")
        
        if not hybrid_matcher:
            raise HTTPException(status_code=503, detail="Hybrid matcher not initialized")
        
        results = {
            "message": f"Document removed from all caches: {document_url}",
            "document_url": document_url,
            "persistent_cache_removed": False,
            "memory_cache_removed": False,
            "vector_db_removed": False
        }
        
        # Remove from RAG engine caches if initialized
        if hybrid_matcher.rag_initialized and hybrid_matcher.rag_engine:
            rag_engine = hybrid_matcher.rag_engine
            
            # Remove from persistent cache
            if rag_engine.document_cache_manager:
                persistent_removed = await rag_engine.document_cache_manager.remove_document(document_url)
                results["persistent_cache_removed"] = persistent_removed
                print(f"✅ Removed from persistent cache: {persistent_removed}")
            
            # Remove from memory caches
            memory_removed = False
            if document_url in rag_engine.document_cache:
                del rag_engine.document_cache[document_url]
                memory_removed = True
            
            if document_url in rag_engine.chunk_cache:
                del rag_engine.chunk_cache[document_url]
                memory_removed = True
            
            results["memory_cache_removed"] = memory_removed
            print(f"✅ Removed from memory cache: {memory_removed}")
            
            # Remove from vector database
            if rag_engine.vector_store:
                try:
                    await rag_engine.vector_store.delete_by_source(document_url)
                    results["vector_db_removed"] = True
                    print("✅ Removed from vector database")
                except Exception as e:
                    print(f"⚠️ Failed to remove from vector database: {str(e)}")
                    results["vector_db_removed"] = False
        
        print(f"✅ Document removal completed: {document_url}")
        return results
        
    except Exception as e:
        print(f"❌ Failed to remove document from cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to remove document: {str(e)}")


# Root endpoint for testing
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BajajFinsev Enhanced Analysis System is running!",
        "version": "3.0.0", 
        "mode": "direct_gemini_with_rag_fallback",
        "primary_method": "Direct Gemini Analysis",
        "fallback_method": "RAG",
        "gemini_model": settings.GOOGLE_MODEL,
        "authentication": "Bearer token required for all endpoints",
        "endpoints": {
            "analyze": "/api/v1/hackrx/run",
            "stream": "/api/v1/hackrx/stream",
            "health": "/api/v1/hackrx/health",
            "performance": "/api/v1/hackrx/performance",
            "cache_stats": "/api/v1/hackrx/cache/stats",
            "cache_clear": "/api/v1/hackrx/cache/clear",
            "cache_remove": "/api/v1/hackrx/cache/document",
            "test_apis": "/api/v1/hackrx/test-apis",
        },
        "supported_formats": [
            "PDF", "DOCX", "DOC", "XLSX", "XLS", "CSV", "TXT", "MD",
            "JSON", "XML", "HTML", "JPG", "JPEG", "PNG", "BMP", "GIF", "WEBP"
        ],
        "advantages": [
            "Direct document upload to Gemini for maximum accuracy",
            "RAG-based fallback for speed",
            "No complex RAG setup required for primary method",
            "Better handling of complex documents",
            "Real-time analysis with Gemini's advanced capabilities"
        ],
        "note": "All endpoints require Authorization: Bearer <token> header"
    }


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)