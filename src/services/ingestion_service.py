import logging
from typing import Tuple, List
import pathlib
from urllib.parse import urlparse
import aiohttp

from src.utils.document_parsers import parse_docx, parse_eml
from src.services.text_extraction_service import text_extraction_service
from src.core.config import settings

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Updated service to download and process documents using text extraction and chunking
    instead of uploading to Gemini API. Returns text chunks for embedding-based RAG.
    """

    async def download_document(self, url: str) -> bytes:
        """Downloads document content from a URL asynchronously."""
        logger.info(f"Downloading document from: {url}")
        timeout = aiohttp.ClientTimeout(total=180)  # Increased timeout for large files
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    return await response.read()
            except aiohttp.ClientError as e:
                logger.error(f"Network error downloading document: {e}")
                raise

    def _get_file_info(self, url: str) -> Tuple[str, str]:
        """Extracts file extension and name from a URL."""
        path = urlparse(url).path
        extension = pathlib.Path(path).suffix.lower()
        file_name = pathlib.Path(path).name
        return extension, file_name

    async def _process_pdf(self, file_content: bytes, file_name: str) -> List[str]:
        """
        Process PDF by extracting text and splitting into chunks.
        """
        logger.info(f"Processing PDF: {file_name} ({len(file_content) / (1024*1024):.2f}MB)")
        
        try:
            # Extract text and chunk it
            text_chunks = await text_extraction_service.extract_and_chunk_pdf(
                file_content, 
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )
            
            logger.info(f"Processed PDF into {len(text_chunks)} text chunks")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_name}: {e}", exc_info=True)
            raise

    async def _process_text_document(self, file_content: bytes, extension: str, file_name: str) -> List[str]:
        """
        Process DOCX or EML documents by extracting text and chunking.
        """
        logger.info(f"Processing {extension} document: {file_name}")
        
        try:
            # Extract text using appropriate parser
            text_parser = parse_docx if extension == ".docx" else parse_eml
            text = text_parser(file_content)
            
            if not text.strip():
                logger.warning(f"No text extracted from {file_name}")
                return []
            
            # Chunk the text
            text_chunks = text_extraction_service.chunk_text(
                text,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )
            
            logger.info(f"Processed {extension} document into {len(text_chunks)} text chunks")
            return text_chunks
            
        except Exception as e:
            logger.error(f"Error processing {extension} document {file_name}: {e}", exc_info=True)
            raise

    async def process_and_extract(self, url: str) -> List[str]:
        """
        Main orchestration method. Downloads a document, extracts text, and returns chunks.
        This replaces the old process_and_upload method for embedding-based RAG.
        """
        extension, file_name = self._get_file_info(url)
        file_content = await self.download_document(url)

        if extension == ".pdf":
            return await self._process_pdf(file_content, file_name)
        
        elif extension in [".docx", ".eml"]:
            return await self._process_text_document(file_content, extension, file_name)
            
        else:
            logger.warning(f"Unsupported file type: {extension}. Attempting PDF extraction.")
            # Try to process as PDF (fallback for unknown extensions)
            try:
                return await self._process_pdf(file_content, file_name)
            except Exception as e:
                logger.error(f"Could not process file {file_name} as PDF: {e}")
                return []