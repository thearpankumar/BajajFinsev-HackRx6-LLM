import logging
import io
import email
from email.policy import default
from docx import Document

logger = logging.getLogger(__name__)

def parse_docx(file_content: bytes) -> str:
    """
    Parses the content of a .docx file and returns its text.
    """
    logger.info("Parsing DOCX file content.")
    try:
        doc = Document(io.BytesIO(file_content))
        full_text = [para.text for para in doc.paragraphs]
        return "\n".join(full_text)
    except Exception as e:
        logger.error(f"Failed to parse DOCX file: {e}")
        return "" # Return empty string on failure

def parse_eml(file_content: bytes) -> str:
    """
    Parses the content of a .eml file and returns its text body.
    """
    logger.info("Parsing EML file content.")
    try:
        msg = email.message_from_bytes(file_content, policy=default)
        
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
        else:
            if msg.get_content_type() == "text/plain":
                body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')

        return body.strip()
    except Exception as e:
        logger.error(f"Failed to parse EML file: {e}")
        return "" # Return empty string on failure

def parse_txt(file_content: bytes) -> str:
    """
    Parses the content of a .txt file and returns its text with proper UTF-8 encoding handling.
    """
    logger.info("Parsing TXT file content.")
    try:
        # Try UTF-8 first (most common)
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback to other common encodings
            for encoding in ['latin-1', 'cp1252', 'ascii']:
                try:
                    return file_content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # Final fallback with error handling
            return file_content.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Failed to parse TXT file: {e}")
        return "" # Return empty string on failure
