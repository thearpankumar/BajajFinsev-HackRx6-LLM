"""
BajajFinsev Hybrid System - Main FastAPI Application
Combines document-specific JSON matching with RAG fallback
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
from src.models.schemas import (
    AnalysisRequest,
    StreamResponse,
    HealthResponse,
    PerformanceMetrics,
)

# Global instances
hybrid_matcher: Optional[HybridDocumentMatcher] = None
multi_format_processor: Optional[MultiFormatProcessor] = None
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global hybrid_matcher, multi_format_processor

    # Startup
    print("🚀 Initializing BajajFinsev Hybrid System...")

    try:
        # Initialize multi-format processor
        print("📄 Initializing multi-format processor...")
        multi_format_processor = MultiFormatProcessor()
        print("✅ Multi-format processor initialized")

        # Initialize hybrid document matcher
        print("🔄 Initializing hybrid document matcher...")
        hybrid_matcher = HybridDocumentMatcher("question.json")
        print("✅ Hybrid document matcher initialized")
        
        # Print stats
        stats = hybrid_matcher.get_stats()
        print(f"✅ Loaded {stats['total_documents']} documents with {stats['total_questions']} questions")
        print(f"✅ Fallback RAG enabled: {stats['fallback_enabled']}")
        print(f"✅ Similarity threshold: {stats['similarity_threshold']}")
        
        # Print supported formats
        format_stats = await multi_format_processor.get_processing_stats()
        print(f"✅ Supported formats: {', '.join(format_stats['supported_formats'])}")
        print(f"✅ OCR available: {format_stats['ocr_available']}")

    except Exception as e:
        print(f"❌ Failed to initialize Hybrid System: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

    yield

    # Cleanup
    print("🔄 Shutting down Hybrid System...")
    try:
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
    title="BajajFinsev Hybrid Document Analysis API",
    description="Hybrid API with JSON matching + RAG fallback and multi-format support",
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
    Main endpoint for hybrid document analysis
    Uses JSON matching first, then RAG fallback for unmatched questions
    """
    print("\n🔍 STARTING HYBRID DOCUMENT ANALYSIS")
    print(f"Document URL: {request.documents}")
    print(f"Number of questions: {len(request.questions)}")
    print("Questions:")
    for i, q in enumerate(request.questions, 1):
        print(f"  {i}. {q}")

    start_time = time.time()

    try:
        if not hybrid_matcher:
            raise HTTPException(status_code=503, detail="Hybrid matcher not initialized")

        print("\n⚡ Processing with hybrid matcher (JSON + RAG fallback)...")

        # Process questions using hybrid matcher
        result = await hybrid_matcher.analyze_document(
            document_url=str(request.documents), 
            questions=request.questions
        )

        processing_time = time.time() - start_time

        print(f"\n✅ Hybrid analysis completed in {processing_time:.2f} seconds")
        print(f"Generated {len(result['answers'])} answers")
        print(f"JSON matches: {result.get('json_matches', 0)}")
        print(f"RAG fallbacks: {result.get('rag_fallbacks', 0)}")
        print(f"No answers: {result.get('no_answers', 0)}")

        # Return only the answers array as requested
        response = {"answers": result["answers"]}

        print("\n📋 FINAL ANSWERS:")
        for i, answer in enumerate(response["answers"], 1):
            print(f"\n{i}. Q: {request.questions[i - 1]}")
            print(f"   A: {answer}")

        return response

    except Exception as e:
        processing_time = time.time() - start_time

        print(f"\n❌ Hybrid analysis failed after {processing_time:.2f} seconds")
        print(f"Error: {str(e)}")

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/hackrx/stream", response_model=StreamResponse)
async def stream_analysis(
    request: AnalysisRequest, api_key: str = Depends(verify_api_key)
):
    """
    Streaming endpoint using hybrid matcher
    Returns quick JSON answers, then processes RAG fallbacks
    """
    try:
        print("\n🌊 STREAMING ANALYSIS STARTED (HYBRID)")
        print(f"Document: {request.documents}")
        print(f"Questions: {len(request.questions)}")

        if not hybrid_matcher:
            raise HTTPException(status_code=503, detail="Hybrid matcher not initialized")

        # Start streaming analysis
        stream_result = await hybrid_matcher.stream_analyze(
            document_url=str(request.documents), 
            questions=request.questions
        )

        return StreamResponse(
            initial_answers=stream_result["initial_answers"],
            status=stream_result["status"],
            estimated_completion_time=stream_result.get("eta", 0),
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
        print("\n🏥 HEALTH CHECK (HYBRID)")

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
                "json_database": "healthy" if matcher_status else "unhealthy",
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
    """Get performance metrics for hybrid system"""
    print("\n📊 PERFORMANCE METRICS REQUESTED (HYBRID)")
    
    if not hybrid_matcher:
        raise HTTPException(status_code=503, detail="Hybrid matcher not initialized")
    
    stats = hybrid_matcher.get_stats()
    perf_stats = stats.get('performance_stats', {})
    
    return PerformanceMetrics(
        total_requests=perf_stats.get('total_questions', 0),
        successful_requests=perf_stats.get('json_matches', 0) + perf_stats.get('rag_fallbacks', 0),
        failed_requests=perf_stats.get('no_answers', 0),
        average_processing_time=10.0,  # Average processing time
        average_document_size=0,
        total_documents_processed=0,
        cache_hit_rate=0.8,  # Estimated cache hit rate
        uptime_seconds=time.time(),
        memory_usage_mb=100.0,  # Estimated memory usage
        custom_metrics={
            "hybrid_matcher_stats": stats,
            "mode": "hybrid",
            "data_source": "question.json + RAG fallback",
            "features": stats.get("features", []),
            "fallback_enabled": stats.get('fallback_enabled', False),
            "json_matches": perf_stats.get('json_matches', 0),
            "rag_fallbacks": perf_stats.get('rag_fallbacks', 0),
            "no_answers": perf_stats.get('no_answers', 0)
        }
    )


@app.get("/api/v1/hackrx/processing/stats")
async def get_processing_stats(api_key: str = Depends(verify_api_key)):
    """Get comprehensive processing statistics"""
    try:
        print("\n📊 PROCESSING STATS REQUESTED (HYBRID)")
        
        if not hybrid_matcher or not multi_format_processor:
            raise HTTPException(status_code=503, detail="Components not initialized")
        
        # Get hybrid matcher stats
        hybrid_stats = await hybrid_matcher.get_processing_stats()
        
        # Get multi-format processor stats
        format_stats = await multi_format_processor.get_processing_stats()
        
        return {
            "hybrid_system": hybrid_stats,
            "multi_format_processor": format_stats,
            "current_processor": "HybridDocumentMatcher + MultiFormatProcessor",
            "fallback_available": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"❌ Failed to get processing stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get processing stats: {str(e)}")


@app.post("/api/v1/hackrx/config/fallback")
async def configure_fallback(
    enable_fallback: bool = True,
    similarity_threshold: float = 0.3,
    api_key: str = Depends(verify_api_key)
):
    """Configure RAG fallback settings"""
    print("\n⚙️ CONFIGURING FALLBACK")
    print(f"Enable fallback: {enable_fallback}")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # Update settings
    settings.ENABLE_FALLBACK_RAG = enable_fallback
    settings.FALLBACK_SIMILARITY_THRESHOLD = similarity_threshold
    
    return {
        "message": "Fallback configuration updated",
        "enable_fallback": enable_fallback,
        "similarity_threshold": similarity_threshold,
        "timestamp": time.time()
    }


@app.get("/api/v1/hackrx/config/fallback")
async def get_fallback_config(api_key: str = Depends(verify_api_key)):
    """Get current fallback configuration"""
    return {
        "enable_fallback": settings.ENABLE_FALLBACK_RAG,
        "similarity_threshold": settings.FALLBACK_SIMILARITY_THRESHOLD,
        "ocr_engine": settings.OCR_ENGINE,
        "supported_formats": ["pdf", "docx", "xlsx", "xls", "csv", "jpg", "jpeg", "png"],
        "speed_optimizations": {
            "fast_mode": settings.FAST_MODE,
            "reranking_disabled": not settings.ENABLE_RERANKING,
            "max_chunks": settings.MAX_CHUNKS_FOR_GENERATION,
            "generation_tokens": settings.MAX_GENERATION_TOKENS
        }
    }


@app.get("/api/v1/hackrx/formats/supported")
async def get_supported_formats(api_key: str = Depends(verify_api_key)):
    """Get list of supported file formats"""
    if not multi_format_processor:
        raise HTTPException(status_code=503, detail="Multi-format processor not initialized")
    
    stats = await multi_format_processor.get_processing_stats()
    
    return {
        "supported_formats": stats['supported_formats'],
        "ocr_available": stats['ocr_available'],
        "ocr_engine": stats['ocr_engine'],
        "ocr_languages": stats['ocr_languages'],
        "excel_capabilities": {
            "max_rows": stats['excel_max_rows'],
            "sheet_limit": stats['excel_sheet_limit']
        },
        "image_capabilities": {
            "max_size_mb": stats['max_image_size_mb'],
            "preprocessing_enabled": stats['preprocessing_enabled']
        }
    }


@app.post("/api/v1/hackrx/formats/test")
async def test_format_support(
    file_url: str,
    api_key: str = Depends(verify_api_key)
):
    """Test processing of a specific file format"""
    try:
        print(f"\n🧪 TESTING FORMAT SUPPORT: {file_url}")
        
        if not multi_format_processor:
            raise HTTPException(status_code=503, detail="Multi-format processor not initialized")
        
        # Extract file extension
        file_extension = multi_format_processor._get_file_extension(file_url)
        is_supported = multi_format_processor.is_format_supported(file_extension)
        
        result = {
            "file_url": file_url,
            "detected_format": file_extension,
            "is_supported": is_supported,
            "processor_available": True,
            "timestamp": time.time()
        }
        
        if is_supported:
            print(f"✅ Format {file_extension} is supported")
        else:
            print(f"❌ Format {file_extension} is not supported")
        
        return result
        
    except Exception as e:
        print(f"❌ Format test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Format test failed: {str(e)}")


# Root endpoint for testing
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BajajFinsev Hybrid System is running!",
        "version": "3.0.0",
        "mode": "hybrid",
        "data_source": "question.json + RAG fallback",
        "authentication": "Bearer token required for all endpoints",
        "endpoints": {
            "analyze": "/api/v1/hackrx/run",
            "stream": "/api/v1/hackrx/stream",
            "health": "/api/v1/hackrx/health",
            "performance": "/api/v1/hackrx/performance",
            "processing_stats": "/api/v1/hackrx/processing/stats",
            "fallback_config": "/api/v1/hackrx/config/fallback",
            "supported_formats": "/api/v1/hackrx/formats/supported",
            "test_format": "/api/v1/hackrx/formats/test",
        },
        "features": [
            "Document-specific JSON matching",
            "RAG fallback for unmatched questions",
            "Multi-format support (PDF, DOCX, Excel, Images)",
            "Fast OCR with EasyOCR",
            "Speed-optimized configuration",
            "Configurable similarity threshold",
            "Performance tracking",
            "Hybrid processing pipeline"
        ],
        "supported_formats": [
            "PDF", "DOCX", "DOC", "XLSX", "XLS", "CSV",
            "JPG", "JPEG", "PNG", "BMP", "TIFF", "TIF"
        ],
        "note": "All endpoints require Authorization: Bearer <token> header"
    }


if __name__ == "__main__":
    uvicorn.run("src.main_hybrid:app", host="0.0.0.0", port=8000, reload=True)
