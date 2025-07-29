import fitz  # PyMuPDF
from docx import Document
import email
import boto3
from typing import Callable, Dict
import io
import logging
from urllib.parse import urlparse
from src.core.config import settings
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_pdf(file_content: bytes) -> str:
    """
    Parse PDF content using AWS Textract with improved error handling and fallbacks.
    
    Args:
        file_content (bytes): The PDF file content in bytes
    
    Returns:
        str: Extracted text from the PDF
    """
    logger.info("Starting PDF parsing with AWS Textract")
    
    # PyMuPDF fallback function
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

    # Check for AWS credentials first
    if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
        logger.warning("AWS credentials not configured. Using PyMuPDF fallback.")
        return pymupdf_fallback(file_content)

    # Check PDF size (Textract limit: 10MB for synchronous calls)
    pdf_size_mb = len(file_content) / (1024 * 1024)
    logger.info(f"PDF size: {pdf_size_mb:.2f} MB")
    
    if pdf_size_mb > 10:
        logger.warning("PDF too large for synchronous Textract. Using PyMuPDF fallback.")
        return pymupdf_fallback(file_content)

    # Initialize Textract client
    try:
        textract = boto3.client('textract',
                              aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                              aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                              region_name=settings.AWS_REGION)
    except Exception as client_error:
        logger.error(f"Failed to initialize Textract client: {client_error}")
        return pymupdf_fallback(file_content)

    # Strategy 1: Try DetectDocumentText first (simpler, often works better for PDFs)
    try:
        logger.info("Attempting Textract with DetectDocumentText API")
        response = textract.detect_document_text(
            Document={'Bytes': file_content}
        )
        
        pdf_text = ""
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                pdf_text += item['Text'] + "\n"
        
        if pdf_text.strip():  # Only return if we got meaningful text
            logger.info("AWS Textract DetectDocumentText succeeded")
            return pdf_text.strip()
        else:
            logger.warning("DetectDocumentText returned empty text")
            
    except Exception as detect_error:
        logger.info(f"DetectDocumentText failed: {str(detect_error)}")

    # Strategy 2: Try AnalyzeDocument as original approach
    try:
        logger.info("Attempting Textract with AnalyzeDocument API")
        response = textract.analyze_document(
            Document={'Bytes': file_content},
            FeatureTypes=['TABLES', 'FORMS']
        )
        
        pdf_text = ""
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                pdf_text += item['Text'] + "\n"
        
        if pdf_text.strip():
            logger.info("AWS Textract AnalyzeDocument succeeded")
            return pdf_text.strip()
            
    except Exception as analyze_error:
        logger.info(f"AnalyzeDocument failed: {str(analyze_error)}")
        
        # Strategy 3: If UnsupportedDocumentException, try page-by-page processing
        if "UnsupportedDocumentException" in str(analyze_error):
            logger.info("Attempting page-by-page PDF processing")
            try:
                page_text = process_pdf_pages_individually(file_content, textract)
                if page_text.strip():
                    logger.info("Page-by-page processing succeeded")
                    return page_text.strip()
            except Exception as page_error:
                logger.info(f"Page-by-page processing failed: {page_error}")
            
            # Strategy 4: Try PDF repair and conversion
            logger.info("Attempting PDF repair for Textract")
            try:
                repaired_pdf_bytes = repair_pdf_for_textract(file_content)
                
                if (repaired_pdf_bytes and 
                    repaired_pdf_bytes != file_content and 
                    len(repaired_pdf_bytes) > 1000):
                    
                    # Try DetectDocumentText with repaired PDF first
                    try:
                        response = textract.detect_document_text(
                            Document={'Bytes': repaired_pdf_bytes}
                        )
                        
                        pdf_text = ""
                        for item in response['Blocks']:
                            if item['BlockType'] == 'LINE':
                                pdf_text += item['Text'] + "\n"
                        
                        if pdf_text.strip():
                            logger.info("Textract with repaired PDF (DetectDocumentText) succeeded")
                            return pdf_text.strip()
                            
                    except Exception:
                        # Try AnalyzeDocument with repaired PDF
                        response = textract.analyze_document(
                            Document={'Bytes': repaired_pdf_bytes},
                            FeatureTypes=['TABLES', 'FORMS']
                        )
                        
                        pdf_text = ""
                        for item in response['Blocks']:
                            if item['BlockType'] == 'LINE':
                                pdf_text += item['Text'] + "\n"
                        
                        if pdf_text.strip():
                            logger.info("Textract with repaired PDF (AnalyzeDocument) succeeded")
                            return pdf_text.strip()
                        
            except Exception as repair_error:
                logger.info(f"PDF repair attempt failed: {str(repair_error)}")
            
            # Strategy 5: Convert to image as last resort
            logger.info("Attempting PDF to image conversion for Textract")
            try:
                success_text = process_pdf_as_images(file_content, textract)
                if success_text.strip():
                    logger.info("PDF to image conversion succeeded")
                    return success_text.strip()
                    
            except Exception as image_error:
                logger.info(f"PDF to image conversion failed: {str(image_error)}")

    # Final fallback to PyMuPDF
    logger.info("All Textract strategies failed, falling back to PyMuPDF")
    return pymupdf_fallback(file_content)

def process_pdf_pages_individually(file_content: bytes, textract_client) -> str:
    """
    Process PDF page by page, converting each to image for Textract.
    This helps with PDFs that have compatibility issues.
    """
    try:
        pdf_doc = fitz.open(stream=file_content, filetype="pdf")
        all_text = []
        
        # Limit to first 5 pages to avoid timeout/size issues
        max_pages = min(5, pdf_doc.page_count)
        logger.info(f"Processing first {max_pages} pages individually")
        
        for page_num in range(max_pages):
            try:
                page = pdf_doc[page_num]
                
                # Convert page to high-quality image
                mat = fitz.Matrix(200/72, 200/72)  # 200 DPI (balance quality/size)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                pix = None  # Free memory
                
                # Check image size
                img_size_mb = len(img_bytes) / (1024 * 1024)
                if img_size_mb > 9:  # Skip if too large
                    logger.warning(f"Page {page_num + 1} image too large ({img_size_mb:.2f}MB), skipping")
                    continue
                
                # Process with Textract
                response = textract_client.detect_document_text(
                    Document={'Bytes': img_bytes}
                )
                
                page_text = ""
                for item in response['Blocks']:
                    if item['BlockType'] == 'LINE':
                        page_text += item['Text'] + "\n"
                
                if page_text.strip():
                    all_text.append(f"--- Page {page_num + 1} ---\n{page_text.strip()}")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as page_error:
                logger.warning(f"Failed to process page {page_num + 1}: {page_error}")
                continue
        
        pdf_doc.close()
        
        if all_text:
            return "\n\n".join(all_text)
        else:
            raise ValueError("No pages processed successfully")
            
    except Exception as e:
        logger.error(f"Page-by-page processing failed: {e}")
        raise

def process_pdf_as_images(file_content: bytes, textract_client) -> str:
    """
    Convert entire PDF to images and process with Textract.
    """
    try:
        pdf_doc = fitz.open(stream=file_content, filetype="pdf")
        all_text = []
        
        # Limit pages to avoid size/timeout issues
        max_pages = min(3, pdf_doc.page_count)
        logger.info(f"Converting first {max_pages} pages to images")
        
        for page_num in range(max_pages):
            try:
                page = pdf_doc[page_num]
                
                # Create medium quality image (balance between quality and size)
                mat = fitz.Matrix(150/72, 150/72)  # 150 DPI
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                pix = None
                
                # Size check
                img_size_mb = len(img_bytes) / (1024 * 1024)
                if img_size_mb > 8:  # Conservative limit
                    logger.warning(f"Page {page_num + 1} image too large, skipping")
                    continue
                
                # Process with Textract
                response = textract_client.detect_document_text(
                    Document={'Bytes': img_bytes}
                )
                
                page_text = ""
                for item in response['Blocks']:
                    if item['BlockType'] == 'LINE':
                        page_text += item['Text'] + "\n"
                
                if page_text.strip():
                    all_text.append(page_text.strip())
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as page_error:
                logger.warning(f"Image processing failed for page {page_num + 1}: {page_error}")
                continue
        
        pdf_doc.close()
        return "\n\n".join(all_text) if all_text else ""
        
    except Exception as e:
        logger.error(f"PDF to image processing failed: {e}")
        raise

def repair_pdf_for_textract(file_content: bytes) -> bytes:
    """
    Attempt to repair/sanitize PDF for Textract compatibility.
    Focus on creating the most basic, compatible PDF structure.
    """
    try:
        logger.info("Attempting minimal PDF repair for Textract compatibility")
        
        source_pdf = fitz.open(stream=file_content, filetype="pdf")
        
        if source_pdf.needs_pass:
            logger.warning("PDF is password protected, cannot repair")
            source_pdf.close()
            return file_content
        
        # Create minimal PDF with basic text only
        repaired_pdf = fitz.open()
        
        # Process only first few pages to avoid issues
        max_pages = min(10, source_pdf.page_count)
        
        for page_num in range(max_pages):
            try:
                source_page = source_pdf[page_num]
                new_page = repaired_pdf.new_page(
                    width=source_page.rect.width, 
                    height=source_page.rect.height
                )
                
                # Extract text in simplest possible way
                simple_text = source_page.get_text()
                
                if simple_text.strip():
                    # Insert as plain text with minimal formatting
                    text_rect = fitz.Rect(
                        50, 50, 
                        source_page.rect.width - 50, 
                        source_page.rect.height - 50
                    )
                    new_page.insert_textbox(
                        text_rect, 
                        simple_text, 
                        fontsize=10,
                        color=(0, 0, 0),
                        fontname="helv"  # Use standard font
                    )
                
            except Exception as page_error:
                logger.warning(f"Failed to repair page {page_num + 1}: {page_error}")
                continue
        
        # Generate minimal PDF
        repaired_pdf_bytes = repaired_pdf.tobytes(
            garbage=4,
            deflate=True,
            clean=True,
            ascii=False,
            expand=0,
            linear=False,
            pretty=False,
            encryption=fitz.PDF_ENCRYPT_NONE
        )
        
        source_pdf.close()
        repaired_pdf.close()
        
        # Basic validation
        if len(repaired_pdf_bytes) > 1000:
            logger.info("PDF repair completed")
            return repaired_pdf_bytes
        else:
            logger.warning("Repaired PDF too small, using original")
            return file_content
        
    except Exception as e:
        logger.warning(f"PDF repair failed: {str(e)}")
        return file_content

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
