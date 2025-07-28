import fitz  # PyMuPDF
from docx import Document
import email
import boto3
from typing import Callable, Dict
import io
import logging
import os
from urllib.parse import urlparse
from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf(file_content: bytes) -> str:
    """
    Parse PDF content with fallback to AWS Textract for scanned documents.
    
    Args:
        file_content (bytes): The PDF file content in bytes
    
    Returns:
        str: Extracted text from the PDF
    """
    logger.info("Starting PDF parsing")
    try:
        # First attempt: Direct text extraction using PyMuPDF
        pdf_text = ""
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        
        logger.info(f"Processing PDF with {pdf_document.page_count} pages")
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            pdf_text += page_text
            
        # If minimal text is extracted (indicating possibly scanned document)
        MIN_EXTRACTABLE_TEXT_LENGTH = 100  # Make threshold configurable
        if len(pdf_text.strip()) < MIN_EXTRACTABLE_TEXT_LENGTH:
            logger.info("Minimal text extracted, falling back to AWS Textract")
            # Fall back to AWS Textract
            if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
                logger.warning("AWS credentials not configured, skipping OCR fallback")
                return pdf_text.strip()
                
            textract = boto3.client('textract',
                                  aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                  aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                  region_name=settings.AWS_REGION)
            
            response = textract.detect_document_text(
                Document={'Bytes': file_content}
            )
            
            pdf_text = ""
            for item in response['Blocks']:
                if item['BlockType'] == 'LINE':
                    pdf_text += item['Text'] + "\n"
            
            logger.info("AWS Textract processing completed")
        else:
            logger.info("Successfully extracted text using PyMuPDF")
        
        return pdf_text.strip()
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def parse_docx(file_content: bytes) -> str:
    """
    Parse DOCX content using python-docx.
    
    Args:
        file_content (bytes): The DOCX file content in bytes
    
    Returns:
        str: Extracted text from the DOCX
    """
    logger.info("Starting DOCX parsing")
    try:
        doc = Document(io.BytesIO(file_content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        logger.info("Successfully parsed DOCX file")
        return text
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        raise

def parse_email(file_content: bytes) -> str:
    """
    Parse email content using Python's email standard library.
    
    Args:
        file_content (bytes): The email file content in bytes
    
    Returns:
        str: Extracted text from the email
    """
    logger.info("Starting email parsing")
    try:
        email_message = email.message_from_bytes(file_content)
        
        # Extract email body
        body = ""
        if email_message.is_multipart():
            logger.info("Processing multipart email")
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode() + "\n"
        else:
            logger.info("Processing single-part email")
            body = email_message.get_payload(decode=True).decode()
        
        logger.info("Successfully parsed email")
        return body.strip()
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise

def get_parser(filename: str) -> Callable[[bytes], str]:
    """
    Factory function to return the correct parser based on file extension,
    correctly handling URLs with query parameters.
    
    Args:
        filename (str): Name of the file or URL
    
    Returns:
        Callable[[bytes], str]: Appropriate parser function for the file type
    """
    # Parse the URL to isolate the path and handle query strings
    parsed_url = urlparse(filename)
    path = parsed_url.path
    
    # Extract the extension from the path component of the URL
    extension = path.lower().split('.')[-1]
    
    parsers: Dict[str, Callable[[bytes], str]] = {
        'pdf': parse_pdf,
        'docx': parse_docx,
        'eml': parse_email,
        'msg': parse_email
    }
    
    if extension not in parsers:
        logger.error(f"Unsupported file type: {extension} (from path: {path})")
        raise ValueError(f"Unsupported file type: {extension}")
    
    logger.info(f"Selected parser for file type: {extension}")
    return parsers[extension]