from fastapi import APIRouter, Depends, HTTPException, status
import logging

from src.schemas.analysis import AnalysisRequest, AnalysisResponse
from src.core.security import validate_bearer_token
from src.services import ingestion_service, rag_workflow_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/hackrx/run",
    response_model=AnalysisResponse,
    summary="Run Document Analysis with Embedding-based RAG",
    description="Analyzes documents using text extraction, embeddings, and Gemini 2.5 Pro for high accuracy and speed.",
    response_description="Analysis results with answers to the provided questions"
)
async def run_analysis(
    request: AnalysisRequest,
    token: str = Depends(validate_bearer_token)
) -> AnalysisResponse:
    """
    This endpoint orchestrates the new embedding-based RAG workflow:
    1. Downloads the document from the provided URL
    2. Extracts text and splits into chunks
    3. Generates embeddings for document chunks
    4. For each question, finds relevant chunks using similarity search
    5. Generates answers using Gemini 2.5 Pro with relevant context
    """
    logger.info(f"🎯 Starting embedding-based analysis for document and {len(request.questions)} questions.")
    
    try:
        # Step 1 & 2: Process the document and extract text chunks
        document_chunks = await ingestion_service.process_and_extract(url=str(request.documents))
        
        if not document_chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the document. Please check the document format."
            )
        
        logger.info(f"📄 Extracted {len(document_chunks)} text chunks from document")

        # Step 3-5: Run the parallel embedding-based RAG workflow
        answers = await rag_workflow_service.run_parallel_workflow(
            questions=request.questions,
            document_chunks=document_chunks
        )
        
        logger.info(f"✅ Generated {len(answers)} answers successfully.")
        return AnalysisResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
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
        "service": "Embedding-based RAG Analysis API",
        "version": "4.0.0", # Version bump for embedding-based architecture
        "message": "Service is running properly with OpenAI embeddings and Gemini 2.5 Pro"
    }