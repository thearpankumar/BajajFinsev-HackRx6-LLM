from fastapi import APIRouter, Depends, HTTPException, status
import logging
import asyncio
import google.generativeai as genai
import mimetypes
from urllib.parse import urlparse

from src.schemas.analysis import AnalysisRequest, AnalysisResponse
from src.core.security import validate_bearer_token
from src.services import ingestion_service, rag_workflow_service

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/hackrx/run",
    response_model=AnalysisResponse,
    summary="Run Document Analysis with Gemini 2.5 Pro",
    description="Directly analyzes a document with Gemini 2.5 Pro to answer questions.",
    response_description="Analysis results with answers to the provided questions"
)
async def run_analysis(
    request: AnalysisRequest,
    token: str = Depends(validate_bearer_token)
) -> AnalysisResponse:
    """
    This endpoint orchestrates the new, simplified workflow:
    1.  Downloads the document from the provided URL.
    2.  Uploads the document directly to the Gemini API.
    3.  Runs a parallel RAG workflow for each question using Gemini 2.5 Pro.
    4.  Cleans up the uploaded file.
    """
    logger.info(f"🎯 Starting analysis for document and {len(request.questions)} questions.")
    
    gemini_file = None
    try:
        # Step 1: Download the document content
        document_url = str(request.documents)
        document_content = await ingestion_service.download_document(document_url)
        
        # Determine the MIME type from the URL path
        path = urlparse(document_url).path
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            # Fallback for URLs without a clear extension
            mime_type = "application/octet-stream"
        
        # Step 2: Upload the document to the Gemini API
        gemini_file = await ingestion_service.upload_to_gemini(
            file_content=document_content,
            display_name=document_url,
            mime_type=mime_type
        )

        # Step 3: Run the parallel RAG workflow
        answers = await rag_workflow_service.run_parallel_workflow(
            questions=request.questions,
            document_file=gemini_file
        )
        
        logger.info(f"✅ Generated {len(answers)} answers successfully.")
        return AnalysisResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"❌ Error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )
    finally:
        # Step 4: Clean up the uploaded file from Gemini
        if gemini_file:
            logger.info(f"Cleaning up Gemini file: {gemini_file.name}")
            await asyncio.to_thread(genai.delete_file, name=gemini_file.name)
            logger.info("File cleanup successful.")

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
        "version": "3.0.0", # Version bump for new architecture
        "message": "Service is running properly with Gemini 2.5 Pro"
    }