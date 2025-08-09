"""
Speed-Optimized Main Application
Ultra-fast RAG with GPU acceleration and intelligent caching
"""

import asyncio
import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.core.speed_rag_pipeline import speed_rag_pipeline
from src.core.config import config

# Request/Response models
class SpeedAnalysisRequest(BaseModel):
    documents: str  # Single document URL for speed
    questions: List[str]

class SpeedAnalysisResponse(BaseModel):
    answers: List[str]
    processing_time: float
    cache_stats: dict

# FastAPI app
app = FastAPI(
    title="BajajFinsev Speed RAG API",
    description="Ultra-fast RAG with GPU acceleration",
    version="1.0.0"
)

@app.on_event("startup")
async def startup():
    """Initialize the speed pipeline"""
    print("🚀 Initializing Speed RAG System...")
    
    if not await speed_rag_pipeline.initialize():
        raise RuntimeError("Failed to initialize Speed RAG Pipeline")
    
    # Preload for maximum speed
    await speed_rag_pipeline.preload_for_speed()
    
    print("✅ Speed RAG System ready!")

@app.post("/api/v1/hackrx/speed", response_model=SpeedAnalysisResponse)
async def speed_analyze(request: SpeedAnalysisRequest):
    """
    Ultra-fast document analysis endpoint
    - GPU-accelerated embeddings
    - Multi-level caching
    - Optimized for RTX 3050
    """
    print(f"\n🚀 SPEED RAG ANALYSIS")
    print(f"Document: {request.documents}")
    print(f"Questions: {len(request.questions)}")
    
    start_time = time.time()
    
    try:
        # Progress callback for document processing
        async def progress_callback(message: str, progress: float):
            print(f"📊 {message} ({progress:.1f}%)")
        
        # Add document (with caching)
        success = await speed_rag_pipeline.add_document_url(
            request.documents, 
            progress_callback=progress_callback
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        # Process questions in parallel for speed
        print(f"\n🔍 Processing {len(request.questions)} questions...")
        
        # Use asyncio.gather for parallel processing
        question_tasks = [
            speed_rag_pipeline.query_fast(question, max_results=3) 
            for question in request.questions
        ]
        
        results = await asyncio.gather(*question_tasks)
        
        # Extract answers
        answers = [result.answer for result in results]
        
        total_time = time.time() - start_time
        cache_stats = speed_rag_pipeline.get_performance_stats()
        
        print(f"\n✅ Speed analysis completed in {total_time:.2f}s")
        print(f"📊 Cache hit rate: {cache_stats.get('cache_hit_rate', 0):.2%}")
        print(f"⚡ GPU utilization: {cache_stats.get('gpu_memory_used_mb', 0):.1f}MB")
        
        return SpeedAnalysisResponse(
            answers=answers,
            processing_time=total_time,
            cache_stats={
                'cache_hit_rate': cache_stats.get('cache_hit_rate', 0),
                'gpu_memory_used_mb': cache_stats.get('gpu_memory_used_mb', 0),
                'total_embeddings_computed': cache_stats.get('embeddings_computed', 0),
                'faiss_index_size': cache_stats.get('faiss_index_size', 0),
                'average_query_time': cache_stats.get('average_query_time', 0)
            }
        )
        
    except Exception as e:
        print(f"❌ Speed analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/v1/hackrx/speed/stats")
async def get_speed_stats():
    """Get detailed performance statistics"""
    if not speed_rag_pipeline.is_initialized:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return speed_rag_pipeline.get_performance_stats()

@app.post("/api/v1/hackrx/speed/cache/clear")
async def clear_speed_cache():
    """Clear all caches for fresh start"""
    speed_rag_pipeline.clear_all_caches()
    return {"message": "All caches cleared", "status": "success"}

@app.get("/")
async def root():
    """Root endpoint with system info"""
    stats = speed_rag_pipeline.get_performance_stats() if speed_rag_pipeline.is_initialized else {}
    
    return {
        "message": "BajajFinsev Speed RAG System",
        "version": "1.0.0",
        "optimization": "RTX 3050 GPU + Multi-level Caching",
        "features": [
            "🚀 GPU-accelerated embeddings (CUDA + FP16)",
            "💾 Multi-level caching (GPU memory → CPU memory)",
            "⚡ FAISS GPU vector search",
            "🔄 Batch processing optimization",
            "📊 Real-time performance monitoring"
        ],
        "performance": {
            "pipeline_initialized": speed_rag_pipeline.is_initialized,
            "total_documents": stats.get('total_documents', 0),
            "cache_hit_rate": f"{stats.get('cache_hit_rate', 0):.2%}",
            "gpu_memory_used_mb": stats.get('gpu_memory_used_mb', 0),
            "average_query_time": f"{stats.get('average_query_time', 0):.3f}s"
        },
        "endpoints": {
            "analyze": "/api/v1/hackrx/speed",
            "stats": "/api/v1/hackrx/speed/stats",
            "clear_cache": "/api/v1/hackrx/speed/cache/clear"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.speed_main:app", host="0.0.0.0", port=8001, reload=False)