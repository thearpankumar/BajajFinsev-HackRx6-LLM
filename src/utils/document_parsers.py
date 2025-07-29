import fitz  # PyMuPDF
from docx import Document
import email
import boto3
from typing import Callable, Dict
import io
import logging
from urllib.parse import urlparse
from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf(file_content: bytes) -> str:
    """
    Parse PDF content using AWS Textract with enhanced pre-processing,
    fallback to PyMuPDF text extraction if Textract fails.
    
    Args:
        file_content (bytes): The PDF file content in bytes
    
    Returns:
        str: Extracted text from the PDF
    """
    logger.info("Starting PDF parsing with enhanced pre-processing and AWS Textract")
    
    # First, try PyMuPDF fallback in case Textract fails
    def pymupdf_fallback(pdf_bytes: bytes) -> str:
        """Fallback text extraction using PyMuPDF"""
        logger.info("Using PyMuPDF fallback for text extraction")
        try:
            pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]
                text += page.get_text() + "\n"
            pdf_doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"PyMuPDF fallback failed: {str(e)}")
            raise ValueError(f"Failed to extract text using PyMuPDF: {str(e)}")

    # --- Step 1: Enhanced PDF pre-processing and sanitization ---
    sanitized_pdf_bytes = b''
    try:
        # Open the original PDF
        source_pdf = fitz.open(stream=file_content, filetype="pdf")
        
        # Create a new, empty PDF document
        sanitized_pdf = fitz.open()
        
        # Process each page individually for better compatibility
        for page_num in range(source_pdf.page_count):
            source_page = source_pdf[page_num]
            
            # Create a new page in the sanitized PDF
            new_page = sanitized_pdf.new_page(width=source_page.rect.width, 
                                            height=source_page.rect.height)
            
            # Get the page as a pixmap and then insert as image
            # This flattens all content including forms, annotations, etc.
            mat = fitz.Matrix(1.0, 1.0)  # No scaling
            pix = source_page.get_pixmap(matrix=mat, alpha=False)
            
            # Insert the flattened page as an image
            img_bytes = pix.tobytes("png")
            new_page.insert_image(new_page.rect, stream=img_bytes)
            
            pix = None  # Free memory
        
        # Generate PDF bytes with optimization for Textract
        sanitized_pdf_bytes = sanitized_pdf.tobytes(garbage=4,  # Clean up
                                                   deflate=True)  # Compress
        
        source_pdf.close()
        sanitized_pdf.close()
        logger.info("Enhanced PDF pre-processing completed successfully")
        
    except Exception as e:
        logger.info(f"PDF pre-processing encountered issue: {str(e)}")
        # If preprocessing fails, try with original bytes
        logger.info("Using original PDF bytes for Textract")
        sanitized_pdf_bytes = file_content

    # --- Step 2: Check for AWS credentials ---
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        logger.warning("AWS credentials not configured. Using PyMuPDF fallback.")
        return pymupdf_fallback(file_content)

    # --- Step 3: Try AWS Textract with sanitized PDF ---
    try:
        textract = boto3.client('textract',
                              aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                              region_name=settings.AWS_REGION)
        
        # Verify PDF size is within Textract limits (10MB for synchronous)
        pdf_size_mb = len(sanitized_pdf_bytes) / (1024 * 1024)
        logger.info(f"Sanitized PDF size: {pdf_size_mb:.2f} MB")
        
        if pdf_size_mb > 10:
            logger.warning("PDF too large for synchronous Textract. Using PyMuPDF fallback.")
            return pymupdf_fallback(file_content)
        
        # Try Textract with sanitized PDF
        response = textract.detect_document_text(
            Document={'Bytes': sanitized_pdf_bytes}
        )
        
        pdf_text = ""
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                pdf_text += item['Text'] + "\n"
        
        logger.info("AWS Textract processing completed successfully")
        return pdf_text.strip()
        
    except Exception as textract_error:
        logger.info(f"Textract processing failed, using PyMuPDF fallback: {str(textract_error)}")
        
        # If Textract fails, use PyMuPDF as fallback
        try:
            return pymupdf_fallback(file_content)
        except Exception as fallback_error:
            # If both methods fail, try with original PDF using Textract one more time
            logger.info("Attempting Textract with original PDF as last resort")
            try:
                response = textract.detect_document_text(
                    Document={'Bytes': file_content}
                )
                
                pdf_text = ""
                for item in response['Blocks']:
                    if item['BlockType'] == 'LINE':
                        pdf_text += item['Text'] + "\n"
                
                logger.info("AWS Textract with original PDF succeeded")
                return pdf_text.strip()
                
            except Exception as final_error:
                error_msg = f"All PDF parsing methods failed. Textract error: {str(textract_error)}. PyMuPDF error: {str(fallback_error)}. Final attempt error: {str(final_error)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

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
        
        # Extract text from paragraphs
        paragraphs_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        # Also extract text from tables if present
        tables_text = ""
        for table in doc.tables:
            for row in table.rows:
                row_text = "\t".join([cell.text for cell in row.cells])
                tables_text += row_text + "\n"
        
        # Combine paragraph and table text
        full_text = paragraphs_text
        if tables_text.strip():
            full_text += "\n\nTables:\n" + tables_text
        
        logger.info("Successfully parsed DOCX file")
        return full_text.strip()
        
    except Exception as e:
        logger.error(f"Error processing DOCX: {str(e)}")
        raise ValueError(f"Failed to parse DOCX file: {str(e)}")

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
        
        # Extract headers
        headers = []
        for key in ['From', 'To', 'Subject', 'Date']:
            value = email_message.get(key)
            if value:
                headers.append(f"{key}: {value}")
        
        header_text = "\n".join(headers) + "\n\n"
        
        # Extract email body
        body = ""
        if email_message.is_multipart():
            logger.info("Processing multipart email")
            for part in email_message.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    try:
                        payload = part.get_payload(decode=True)
                        if payload:
                            body += payload.decode('utf-8', errors='ignore') + "\n"
                    except Exception as decode_error:
                        logger.warning(f"Failed to decode email part: {decode_error}")
                        continue
        else:
            logger.info("Processing single-part email")
            try:
                payload = email_message.get_payload(decode=True)
                if payload:
                    body = payload.decode('utf-8', errors='ignore')
            except Exception as decode_error:
                logger.warning(f"Failed to decode email: {decode_error}")
                body = ""
        
        full_email = header_text + body.strip()
        logger.info("Successfully parsed email")
        return full_email
        
    except Exception as e:
        logger.error(f"Error processing email: {str(e)}")
        raise ValueError(f"Failed to parse email file: {str(e)}")

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
    if '.' not in path:
        logger.error(f"No file extension found in path: {path}")
        raise ValueError(f"Cannot determine file type from: {filename}")
    
    extension = path.lower().split('.')[-1]
    
    parsers: Dict[str, Callable[[bytes], str]] = {
        'pdf': parse_pdf,
        'docx': parse_docx,
        'doc': parse_docx,  # Also handle .doc files (though may need separate handling)
        'eml': parse_email,
        'msg': parse_email
    }
    
    if extension not in parsers:
        supported_types = ', '.join(parsers.keys())
        logger.error(f"Unsupported file type: {extension} (from path: {path})")
        raise ValueError(f"Unsupported file type: {extension}. Supported types: {supported_types}")
    
    logger.info(f"Selected parser for file type: {extension}")
    return parsers[extension]
