from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse
from typing import Optional, Union
import aiohttp
from io import BytesIO
import os
import logging
from src.utils.document_parsers import get_parser
from src.core.security import validate_bearer_token
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class URLModel(BaseModel):
    url: str

router = APIRouter()

async def process_document(file_content: bytes, filename: str) -> str:
    
    try:
        parser = get_parser(filename)
        text = parser(file_content)
        logger.info(f"Successfully processed document: {filename}")
        return text
    except Exception as e:
        logger.error(f"Error processing document {filename}: {str(e)}")
        raise

async def download_file(url: str) -> tuple[bytes, str]:
    
    logger.info(f"Downloading file from URL: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                logger.error(f"Failed to download file from URL: {url}")
                raise HTTPException(status_code=400, detail="Could not download file from URL")
            
            # Try to get filename from Content-Disposition header or URL
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition and "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"')
            else:
                filename = url.split("/")[-1]
            
            content = await response.read()
            logger.info(f"Successfully downloaded file: {filename}")
            return content, filename

@router.post("/upload/file", 
    summary="Upload Document File",
    description="Upload a document file (PDF, DOCX, or EML)",
    response_class=JSONResponse)
async def upload_document_file(
    file: UploadFile = File(..., description="Upload a document file (PDF, DOCX, or EML)"),
    token: str = Depends(validate_bearer_token)
) -> JSONResponse:
   
    try:
        logger.info(f"Processing uploaded file: {file.filename}")
        content = await file.read()
        
        # Process document
        extracted_text = await process_document(content, file.filename)
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Document processed successfully",
                "filename": file.filename,
                "status": "completed",
                "extracted_text": extracted_text
            }
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/upload/url", 
    summary="Process Document from URL",
    description="Process a document by providing its URL",
    response_class=JSONResponse)
async def process_document_url(
    url_data: URLModel,
    token: str = Depends(validate_bearer_token)
) -> JSONResponse:
 
    try:
        logger.info(f"Processing document from URL: {url_data.url}")
        content, filename = await download_file(url_data.url)
        
        # Process document
        extracted_text = await process_document(content, filename)
        
        return JSONResponse(
            status_code=202,
            content={
                "message": "Document processed successfully",
                "filename": filename,
                "status": "completed",
                "extracted_text": extracted_text
            }
        )
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 