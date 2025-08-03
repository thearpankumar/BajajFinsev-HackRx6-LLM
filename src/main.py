"""
BajajFinsev RAG System - Main FastAPI Application
High-accuracy document analysis for Insurance, Legal, HR, and Compliance domains
"""

import time
import json
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.core.config import settings
from src.core.rag_engine import RAGEngine
from src.core.document_processor import DocumentProcessor
from src.core.performance_monitor import PerformanceMonitor
from src.models.schemas import (
    AnalysisRequest,
    StreamResponse,
    HealthResponse,
    PerformanceMetrics,
)

# Global instances
rag_engine: Optional[RAGEngine] = None
doc_processor: Optional[DocumentProcessor] = None
performance_monitor = PerformanceMonitor()
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup application resources"""
    global rag_engine, doc_processor

    # Startup
    print("🚀 Initializing BajajFinsev RAG System...")

    # Initialize components
    doc_processor = DocumentProcessor()
    rag_engine = RAGEngine()

    # Initialize vector database and models
    await rag_engine.initialize()

    print("✅ RAG System initialized successfully!")

    yield

    # Cleanup
    print("🔄 Shutting down RAG System...")
    if rag_engine:
        await rag_engine.cleanup()
    print("✅ Shutdown complete!")


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
async def get_performance_metrics():
    """Get detailed performance metrics"""
    print("\n📊 PERFORMANCE METRICS REQUESTED")
    return performance_monitor.get_metrics()


@app.post("/api/v1/hackrx/performance/reset")
async def reset_performance_metrics():
    """Reset performance counters"""
    print("\n🔄 RESETTING PERFORMANCE METRICS")
    performance_monitor.reset_metrics()
    return {"message": "Performance metrics reset successfully"}


@app.post("/api/v1/hackrx/performance/mode")
async def set_performance_mode(mode: str):
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
async def get_performance_mode():
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


@app.post("/api/v1/hackrx/performance/parallel")
async def set_parallel_settings(
    enabled: bool = True, max_parallel: int = 40, batch_size: int = 10
):
    """Configure parallel processing settings"""
    print("\n⚡ CONFIGURING PARALLEL PROCESSING")
    print(f"   Enabled: {enabled}")
    print(f"   Max Parallel: {max_parallel}")
    print(f"   Batch Size: {batch_size}")

    # Validate settings
    if max_parallel < 1 or max_parallel > 100:
        raise HTTPException(
            status_code=400, detail="max_parallel must be between 1 and 100"
        )

    if batch_size < 1 or batch_size > max_parallel:
        raise HTTPException(
            status_code=400, detail="batch_size must be between 1 and max_parallel"
        )

    # Update settings
    settings.PARALLEL_PROCESSING = enabled
    settings.MAX_PARALLEL_QUESTIONS = max_parallel
    settings.QUESTION_BATCH_SIZE = batch_size

    return {
        "message": "Parallel processing settings updated",
        "parallel_processing": enabled,
        "max_parallel_questions": max_parallel,
        "question_batch_size": batch_size,
    }


# Root endpoint for testing
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "BajajFinsev RAG System is running!",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/api/v1/hackrx/run",
            "stream": "/api/v1/hackrx/stream",
            "health": "/api/v1/hackrx/health",
            "performance": "/api/v1/hackrx/performance",
        },
    }


if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
