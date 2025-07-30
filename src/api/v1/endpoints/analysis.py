from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
import logging
import asyncio
import google.generativeai as genai
import mimetypes
from urllib.parse import urlparse
import subprocess
import sys

from src.schemas.analysis import AnalysisRequest, AnalysisResponse
from src.core.security import validate_bearer_token
from src.services import ingestion_service, rag_workflow_service

logger = logging.getLogger(__name__)

router = APIRouter()

def run_cleanup_process(file_name: str):
    """
    Launches the cleanup script in a new, completely detached process
    to ensure it does not block the main application.
    """
    logger.info(f"Launching fully detached fire-and-forget cleanup for: {file_name}")
    
    # Using Popen with these arguments ensures the subprocess is fully detached
    # from the parent process, preventing any blocking.
    subprocess.Popen(
        [sys.executable, "src/utils/cleanup_task.py", file_name],
        stdout=subprocess.DEVNULL,  # Do not connect to stdout
        stderr=subprocess.DEVNULL,  # Do not connect to stderr
        start_new_session=True      # Run in a new session, fully independent
    )

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
    4.  Launches a non-blocking background process for file cleanup.
    """
    logger.info(f"🎯 Starting analysis for document and {len(request.questions)} questions.")
    
    gemini_file = None
    try:
        # Step 1 & 2: Process the document based on its type and upload to Gemini
        gemini_file = await ingestion_service.process_and_upload(url=str(request.documents))

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
        # Step 4: Launch the fire-and-forget cleanup process
        if gemini_file:
            run_cleanup_process(gemini_file.name)

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