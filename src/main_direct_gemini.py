"""
BajajFinsev Direct Gemini System - Main FastAPI Application
Directly sends documents to Gemini for analysis without RAG implementation
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
from src.core.direct_gemini_processor import DirectGeminiProcessor
from src.core.response_timer import ResponseTimer
from src.models.schemas import (
    AnalysisRequest,
    StreamResponse,
    HealthResponse,
    PerformanceMetrics,
)

# Global instances
direct_gemini_processor: Optional[DirectGeminiProcessor] = None
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global direct_gemini_processor

    # Startup
    print("🚀 Initializing BajajFinsev Direct Gemini System...")

    try:
        # Initialize direct Gemini processor
        print("🧠 Initializing Direct Gemini processor...")
        direct_gemini_processor = DirectGeminiProcessor()
        print("✅ Direct Gemini processor initialized")
        
        # Print configuration
        print(f"✅ Using Gemini model: {settings.GOOGLE_MODEL}")
        print(f"✅ Response time range: {settings.MIN_RESPONSE_TIME_SECONDS}-{settings.MAX_RESPONSE_TIME_SECONDS}s")
        print("✅ Direct document analysis - no RAG, no vector database required")

    except Exception as e:
        print(f"❌ Failed to initialize Direct Gemini System: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

    yield

    # Cleanup
    print("🔄 Shutting down Direct Gemini System...")
    try:
        if direct_gemini_processor:
            await direct_gemini_processor.cleanup()
        
        # Simple garbage collection
        import gc
        gc.collect()
        
        print("✅ Shutdown complete!")
    except Exception as e:
        print(f"⚠️ Error during shutdown: {str(e)}")


# Create FastAPI app
app = FastAPI(
    title="BajajFinsev Direct Gemini Analysis API",
    description="Direct document analysis using Google Gemini - no RAG implementation required",
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
    Main endpoint for document analysis using Direct Gemini:
    1. Download document
    2. Send directly to Gemini for analysis
    3. Parse and return answers
    
    Ensures minimum response time for consistent UX
    """
    print("\n🧠 STARTING DIRECT GEMINI ANALYSIS")
    print(f"Document URL: {request.documents}")
    print(f"Number of questions: {len(request.questions)}")
    print("Questions:")
    for i, q in enumerate(request.questions, 1):
        print(f"  {i}. {q}")

    # Start response timer
    timer = ResponseTimer()
    timer.start()

    try:
        if not direct_gemini_processor:
            raise HTTPException(status_code=503, detail="Direct Gemini processor not initialized")

        print("\n🚀 Processing with Direct Gemini...")

        # Process questions using direct Gemini processor
        result = await direct_gemini_processor.analyze_document(
            document_url=str(request.documents), 
            questions=request.questions
        )

        print(f"\n✅ Direct Gemini analysis completed in {timer.get_elapsed_time():.2f} seconds")
        print(f"Generated {len(result['answers'])} answers")
        print(f"Method: {result.get('method', 'direct_gemini')}")
        print(f"File type: {result.get('file_type', 'unknown')}")

        # Return only the answers array as requested
        response = {"answers": result["answers"]}

        # Ensure minimum response time
        response = await timer.ensure_minimum_time(response)

        print("\n📋 FINAL ANSWERS:")
        for i, answer in enumerate(response["answers"], 1):
            print(f"\n{i}. Q: {request.questions[i - 1]}")
            print(f"   A: {answer}")

        return response

    except Exception as e:
        elapsed_time = timer.get_elapsed_time()

        print(f"\n❌ Direct Gemini analysis failed after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")

        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/v1/hackrx/stream", response_model=StreamResponse)
async def stream_analysis(
    request: AnalysisRequest, api_key: str = Depends(verify_api_key)
):
    """
    Streaming endpoint using Direct Gemini
    Returns processing status and estimated completion time
    """
    try:
        print("\n🌊 STREAMING ANALYSIS STARTED (DIRECT GEMINI)")
        print(f"Document: {request.documents}")
        print(f"Questions: {len(request.questions)}")

        # Start response timer for streaming
        timer = ResponseTimer()
        timer.start()

        if not direct_gemini_processor:
            raise HTTPException(status_code=503, detail="Direct Gemini processor not initialized")

        # For streaming, provide initial status
        initial_answers = [f"Analyzing with Gemini: {q}" for q in request.questions]

        # Calculate estimated completion time
        elapsed = timer.get_elapsed_time()
        eta = max(settings.MIN_RESPONSE_TIME_SECONDS - elapsed, 10)  # At least 10 seconds for Gemini

        return StreamResponse(
            initial_answers=initial_answers,
            status="processing",
            estimated_completion_time=eta,
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
        print("\n🏥 HEALTH CHECK (DIRECT GEMINI)")

        # Check direct Gemini processor
        processor_status = direct_gemini_processor is not None
        
        overall_status = "healthy" if processor_status else "degraded"

        print(f"Direct Gemini Processor: {'✅' if processor_status else '❌'}")
        print(f"Overall: {overall_status}")

        return HealthResponse(
            status=overall_status,
            components={
                "direct_gemini_processor": "healthy" if processor_status else "unhealthy",
                "gemini_model": settings.GOOGLE_MODEL,
                "api_integration": "direct",
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
    """Get performance metrics for Direct Gemini system"""
    print("\n📊 PERFORMANCE METRICS REQUESTED (DIRECT GEMINI)")
    
    if not direct_gemini_processor:
        raise HTTPException(status_code=503, detail="Direct Gemini processor not initialized")
    
    stats = direct_gemini_processor.get_stats()
    processor_stats = stats.get('stats', {})
    
    return PerformanceMetrics(
        total_requests=processor_stats.get('total_requests', 0),
        successful_requests=processor_stats.get('successful_requests', 0),
        failed_requests=processor_stats.get('failed_requests', 0),
        average_processing_time=processor_stats.get('avg_processing_time', 0),
        average_document_size=0,
        total_documents_processed=processor_stats.get('total_documents_processed', 0),
        cache_hit_rate=0.0,  # No caching in direct mode
        uptime_seconds=time.time(),
        memory_usage_mb=50.0,  # Estimated lower memory usage
        custom_metrics={
            "direct_gemini_stats": stats,
            "mode": "direct_gemini",
            "model": settings.GOOGLE_MODEL,
            "features": stats.get("features", []),
            "processing_method": "direct_document_upload"
        }
    )


# Root endpoint for testing
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BajajFinsev Direct Gemini System is running!",
        "version": "3.0.0", 
        "mode": "direct_gemini",
        "model": settings.GOOGLE_MODEL,
        "description": "Direct document analysis with Google Gemini - no RAG implementation",
        "authentication": "Bearer token required for all endpoints",
        "endpoints": {
            "analyze": "/api/v1/hackrx/run",
            "stream": "/api/v1/hackrx/stream",
            "health": "/api/v1/hackrx/health",
            "performance": "/api/v1/hackrx/performance",
        },
        "supported_formats": [
            "PDF", "DOCX", "DOC", "XLSX", "XLS", "CSV", "TXT", "MD",
            "JSON", "XML", "HTML", "JPG", "JPEG", "PNG", "BMP", "GIF", "WEBP"
        ],
        "advantages": [
            "Direct document upload to Gemini",
            "No vector database setup required",
            "No chunking or embedding computation",
            "Simplified architecture",
            "Real-time document analysis",
            "Better accuracy for complex documents"
        ],
        "note": "All endpoints require Authorization: Bearer <token> header"
    }


if __name__ == "__main__":
    uvicorn.run("src.main_direct_gemini:app", host="0.0.0.0", port=8000, reload=True)