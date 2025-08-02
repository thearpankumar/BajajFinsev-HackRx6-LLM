import logging
from typing import Tuple, List, Dict, Any
import pathlib
from urllib.parse import urlparse
import aiohttp

from src.utils.document_parsers import parse_docx, parse_eml, parse_txt
from src.services.text_extraction_service import text_extraction_service
from src.core.config import settings

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Service to download, process, and chunk documents with metadata.
    """

    async def download_document(self, url: str) -> bytes:
        """Downloads document content from a URL asynchronously."""
        logger.info(f"Downloading document from: {url}")
        timeout = aiohttp.ClientTimeout(total=180)
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

    async def process_and_extract(self, url: str) -> List[Dict[str, Any]]:
        """
        Main orchestration method. Downloads a document, extracts text, 
        and returns chunks with metadata.
        """
        extension, file_name = self._get_file_info(url)
        file_content = await self.download_document(url)

        if extension == ".pdf":
            logger.info(f"Processing PDF: {file_name}")
            return await text_extraction_service.extract_and_chunk_pdf(
                file_content,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )
        
        elif extension in [".docx", ".eml", ".txt"]:
            logger.info(f"Processing {extension} document: {file_name}")
            if extension == ".docx":
                text = parse_docx(file_content)
            elif extension == ".eml":
                text = parse_eml(file_content)
            else: # .txt
                text = parse_txt(file_content)
            
            # For non-PDF files, we don't have page numbers, so we pass a single tuple
            # representing the whole document.
            pages_with_metadata = [(text, 1)]
            return text_extraction_service.chunk_text(
                pages_with_metadata,
                chunk_size=settings.CHUNK_SIZE,
                overlap=settings.CHUNK_OVERLAP
            )
            
        else:
            logger.warning(f"Unsupported file type: {extension}. Attempting PDF extraction as fallback.")
            try:
                return await text_extraction_service.extract_and_chunk_pdf(
                    file_content,
                    chunk_size=settings.CHUNK_SIZE,
                    overlap=settings.CHUNK_OVERLAP
                )
            except Exception as e:
                logger.error(f"Could not process file {file_name} as PDF: {e}")
                return []

    async def extract_full_text(self, url: str) -> str:
        """
        Extracts the full text content from a document without chunking.
        (This method is used for the hierarchical workflow and can remain as is,
        as it doesn't need chunk-level metadata).
        """
        extension, _ = self._get_file_info(url)
        file_content = await self.download_document(url)
        
        if extension == ".pdf":
            # This will now return a list of tuples. We need to join the text.
            pages_with_metadata = await text_extraction_service.extract_text_from_pdf_with_metadata(file_content)
            return "\n\n".join([page[0] for page in pages_with_metadata])
        elif extension == ".docx":
            return parse_docx(file_content)
        elif extension == ".eml":
            return parse_eml(file_content)
        elif extension == ".txt":
            return parse_txt(file_content)
        else:
            logger.warning(f"Unknown extension {extension}, trying PDF extraction.")
            pages_with_metadata = await text_extraction_service.extract_text_from_pdf_with_metadata(file_content)
            return "\n\n".join([page[0] for page in pages_with_metadata])


