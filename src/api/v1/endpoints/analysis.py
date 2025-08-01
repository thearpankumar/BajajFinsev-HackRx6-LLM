from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
import logging
import json
from typing import Optional

from src.schemas.analysis import AnalysisRequest, AnalysisResponse
from src.core.security import validate_bearer_token
from src.services import ingestion_service, rag_workflow_service, text_extraction_service
from src.services.streaming_service import streaming_service
from src.utils.performance_monitor import performance_monitor, monitor_performance
from src.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/hackrx/run",
    response_model=AnalysisResponse,
    summary="Run Business Document Analysis",
    description="Analyzes business documents (insurance, legal, HR, compliance) using AI-powered hierarchical processing, OpenAI GPT-4o-mini, and advanced embedding retrieval for accurate information extraction.",
    response_description="Business-focused analysis results with comprehensive answers"
)
async def run_analysis(
    request: AnalysisRequest,
    token: str = Depends(validate_bearer_token)
) -> AnalysisResponse:
    """
    Business document analysis endpoint specialized for enterprise domains:
    1. Downloads and processes business documents (insurance, legal, HR, compliance)
    2. Applies domain-aware hierarchical processing for large documents
    3. Uses OpenAI GPT-4o-mini with business-focused prompts
    4. Employs advanced embedding retrieval with business terminology
    5. Generates comprehensive, domain-specific answers
    """
    logger.info(f"🚀 Starting optimized analysis for document and {len(request.questions)} questions.")
    
    # Use hierarchical processing for large documents
    hierarchical_enabled = settings.ENABLE_HIERARCHICAL_PROCESSING
    
    try:
        # Step 1: Check document size first to determine processing strategy
        performance_monitor.start_operation("document_download")
        
        # Download document to check size
        document_url = str(request.documents)
        extension, file_name = ingestion_service._get_file_info(document_url)
        file_content = await ingestion_service.download_document(document_url)
        file_size_bytes = len(file_content)
        
        performance_monitor.end_operation(
            "document_download", 
            document_size=file_size_bytes,
            success=True
        )
        
        # Determine if this is a large document based on file size
        is_large_document = file_size_bytes >= settings.LARGE_DOCUMENT_THRESHOLD
        
        logger.info(f"📄 Downloaded document: {file_name} ({file_size_bytes/1024/1024:.2f}MB)")
        logger.info(f"🔍 Large document detected: {is_large_document}")
        logger.info(f"⚡ Processing strategy: {'hierarchical' if is_large_document and hierarchical_enabled else 'standard'}")

        # Step 2: Choose processing strategy based on document size
        if is_large_document and hierarchical_enabled:
            # Use hierarchical workflow for large documents (>20MB)
            performance_monitor.start_operation("hierarchical_workflow")
            
            # Extract full text for hierarchical processing
            if extension == ".pdf":
                document_text = await text_extraction_service.extract_text_from_pdf(file_content)
            elif extension == ".docx":
                from src.utils.document_parsers import parse_docx
                document_text = parse_docx(file_content)
            elif extension == ".eml":
                from src.utils.document_parsers import parse_eml
                document_text = parse_eml(file_content)
            else:
                document_text = await text_extraction_service.extract_text_from_pdf(file_content)
            
            answers, workflow_metrics = await rag_workflow_service.run_hierarchical_workflow(
                questions=request.questions,
                document_text=document_text
            )
            
            performance_monitor.end_operation(
                "hierarchical_workflow",
                document_size=len(document_text),
                chunks_processed=workflow_metrics.get('total_questions', 0),
                success=True,
                metadata=workflow_metrics
            )
            
        else:
            # Use ORIGINAL fast workflow for smaller documents (<20MB)
            performance_monitor.start_operation("standard_workflow")
            
            # Use original chunking approach - much faster!
            if extension == ".pdf":
                document_chunks = await ingestion_service._process_pdf(file_content, file_name)
            elif extension in [".docx", ".eml"]:
                document_chunks = await ingestion_service._process_text_document(file_content, extension, file_name)
            else:
                document_chunks = await ingestion_service._process_pdf(file_content, file_name)
            
            if not document_chunks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No text could be extracted from the document."
                )
            
            # Use ORIGINAL fast parallel workflow
            answers = await rag_workflow_service.run_parallel_workflow(
                questions=request.questions,
                document_chunks=document_chunks
            )
            
            workflow_metrics = {
                'hierarchical_used': False,
                'total_chunks': len(document_chunks),
                'processing_method': 'standard_parallel_fast'
            }
            
            performance_monitor.end_operation(
                "standard_workflow",
                document_size=sum(len(chunk) for chunk in document_chunks),
                chunks_processed=len(document_chunks),
                success=True
            )
        
        # Step 3: Compile response with performance data
        performance_summary = performance_monitor.get_performance_summary()
        
        logger.info(f"✅ Generated {len(answers)} answers successfully.")
        logger.info(f"📊 Cache hit rate: {performance_monitor.get_cache_hit_rate():.1f}%")
        
        # Clean response with only answers
        response_data = {
            "answers": answers
        }
        
        return AnalysisResponse(**response_data)
        
    except HTTPException:
        performance_monitor.record_error()
        raise
    except Exception as e:
        performance_monitor.record_error()
        logger.error(f"❌ Error during optimized analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )

@router.post(
    "/hackrx/stream",
    summary="Stream Document Analysis",
    description="Stream document analysis responses in phases for improved user experience with large documents.",
    tags=["streaming"]
)
async def stream_analysis(
    request: AnalysisRequest,
    token: str = Depends(validate_bearer_token)
):
    """
    Stream document analysis for large documents:
    1. Quick scan phase: Fast initial answers (30-60 seconds)
    2. Detailed analysis phase: Full hierarchical processing
    """
    if not settings.ENABLE_STREAMING_RESPONSES:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Streaming responses are not enabled. Set ENABLE_STREAMING_RESPONSES=true in configuration."
        )
    
    logger.info(f"🌊 Starting streaming analysis for {len(request.questions)} questions")
    
    try:
        # Extract document text
        document_text = await ingestion_service.extract_full_text(url=str(request.documents))
        
        if not document_text or len(document_text.strip()) < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No meaningful text could be extracted from the document."
            )
        
        # Stream responses
        async def response_generator():
            async for response in streaming_service.stream_document_analysis(document_text, request.questions):
                formatted_response = streaming_service.format_streaming_response(response)
                yield f"data: {json.dumps(formatted_response)}\n\n"
        
        return StreamingResponse(
            response_generator(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"❌ Error in streaming analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming analysis failed: {e}"
        )

@router.get(
    "/hackrx/performance",
    summary="Performance Metrics",
    description="Get detailed performance metrics and statistics",
    tags=["monitoring"]
)
async def get_performance_metrics(token: str = Depends(validate_bearer_token)):
    """Get comprehensive performance metrics for monitoring and optimization."""
    try:
        metrics = performance_monitor.get_performance_summary()
        recent_operations = performance_monitor.get_recent_metrics(20)
        
        return {
            "performance_summary": metrics,
            "recent_operations": recent_operations,
            "cache_statistics": {
                "hit_rate": performance_monitor.get_cache_hit_rate(),
                "total_hits": performance_monitor.cache_stats["hits"],
                "total_misses": performance_monitor.cache_stats["misses"]
            },
            "configuration": {
                "hierarchical_processing": settings.ENABLE_HIERARCHICAL_PROCESSING,
                "document_cache": settings.ENABLE_DOCUMENT_CACHE,
                "large_doc_threshold": settings.LARGE_DOCUMENT_THRESHOLD,
                "max_chunks_per_query": settings.MAX_CHUNKS_PER_QUERY,
                "embedding_batch_size": settings.EMBEDDING_BATCH_SIZE,
                "parallel_batches": settings.PARALLEL_BATCHES
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve performance metrics: {e}"
        )

@router.post(
    "/hackrx/performance/reset",
    summary="Reset Performance Metrics",
    description="Reset all performance counters and metrics",
    tags=["monitoring"]
)
async def reset_performance_metrics(token: str = Depends(validate_bearer_token)):
    """Reset all performance metrics and counters."""
    performance_monitor.reset_metrics()
    return {"message": "Performance metrics reset successfully"}

@router.get(
    "/hackrx/health",
    summary="Health Check",
    description="Check if the analysis service is running properly with optimization features",
    tags=["monitoring"]
)
async def health_check():
    """Enhanced health check with optimization features status."""
    try:
        # Test embedding service
        test_embedding = await rag_workflow_service.embedding_service.generate_embedding("test")
        embedding_healthy = len(test_embedding) > 0
        
        return {
            "status": "healthy",
            "service": "Business Document Analysis API",
            "version": "5.0.0",
            "specialization": "Insurance, Legal, HR, and Compliance domains",
            "features": {
                "hierarchical_processing": settings.ENABLE_HIERARCHICAL_PROCESSING,
                "document_caching": settings.ENABLE_DOCUMENT_CACHE,
                "parallel_processing": True,
                "streaming_responses": settings.ENABLE_STREAMING_RESPONSES,
                "performance_monitoring": True,
                "business_domain_awareness": True
            },
            "models": {
                "embedding_model": settings.EMBEDDING_MODEL,
                "main_llm_model": "gpt-4o-mini",
                "query_enhancement_model": "gemini-2.5-flash-lite"
            },
            "performance": {
                "cache_hit_rate": round(performance_monitor.get_cache_hit_rate(), 1),
                "total_operations": len(performance_monitor.metrics_history)
            },
            "embedding_service": "healthy" if embedding_healthy else "error",
            "message": "Business document analysis service is running with enterprise-grade AI capabilities"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "service": "Business Document Analysis API",
            "version": "5.0.0",
            "specialization": "Insurance, Legal, HR, and Compliance domains",
            "error": str(e),
            "message": "Service is running but some features may be impaired"
        }