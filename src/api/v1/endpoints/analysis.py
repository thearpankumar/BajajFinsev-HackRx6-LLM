from fastapi import APIRouter, Depends, HTTPException, status
import logging
import asyncio

from src.schemas.analysis import AnalysisRequest, AnalysisResponse
from src.core.security import validate_bearer_token
from src.services.ingestion_service import ingestion_service
from src.services.rag_workflow import rag_workflow_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/hackrx/run",
    response_model=AnalysisResponse,
    summary="Run Document Analysis",
    description="Analyze documents and answer questions using a unified RAG workflow.",
    response_description="Analysis results with answers to the provided questions"
)
async def run_analysis(
    request: AnalysisRequest,
    token: str = Depends(validate_bearer_token)
) -> AnalysisResponse:
    """
    This endpoint orchestrates the entire process:
    1.  **On-the-fly Ingestion**: Downloads, parses, and embeds the provided documents.
    2.  **RAG Workflow**: Answers the questions based on the ingested documents.
    """
    logger.info(f"🎯 Starting analysis for {len(request.documents)} documents and {len(request.questions)} questions.")
    
    try:
        # --- On-the-fly Ingestion ---
        ingestion_tasks = [
            ingestion_service.process_document(str(doc_url)) for doc_url in request.documents
        ]
        document_ids = await asyncio.gather(*ingestion_tasks)
        logger.info(f"✅ Successfully ingested {len(document_ids)} documents. IDs: {document_ids}")

        # --- RAG Workflow ---
        answers = await rag_workflow_service.run_workflow(request.questions, document_ids)
        logger.info(f"✅ Generated {len(answers)} answers.")
        
        return AnalysisResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred during analysis: {str(e)}"
        )

@router.get(
    "/hackrx/health",
    summary="Health Check",
    description="Check if the analysis service is running properly",
    tags=["monitoring"]
)
async def health_check():
    """Health check endpoint for monitoring service availability."""
    return {
        "status": "healthy",
        "service": "RAG Analysis API",
        "version": "2.0.0",
        "message": "Service is running properly"
    }
