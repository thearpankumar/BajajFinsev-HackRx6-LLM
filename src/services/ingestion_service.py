import logging
import asyncio
import google.generativeai as genai
from typing import Any, Tuple
import tempfile
import pathlib
from urllib.parse import urlparse
import aiohttp

from src.utils.document_parsers import parse_docx, parse_eml

logger = logging.getLogger(__name__)

class IngestionService:
    """
    A service to download, process, and upload documents to the Gemini API
    based on their file type.
    """

    async def download_document(self, url: str) -> bytes:
        """Downloads document content from a URL asynchronously."""
        logger.info(f"Downloading document from: {url}")
        timeout = aiohttp.ClientTimeout(total=60)
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

    async def _upload_raw_file(self, file_content: bytes, display_name: str) -> Any:
        """Uploads a raw file (like a PDF) directly to Gemini."""
        logger.info(f"Performing direct upload for: {display_name}")
        # This flow uses a temporary file to satisfy the API's path requirement
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(display_name).suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            gemini_file = await asyncio.to_thread(
                genai.upload_file,
                path=temp_file_path,
                display_name=display_name
            )
            pathlib.Path(temp_file_path).unlink() # Clean up the temp file
            return gemini_file
        except Exception:
            if 'temp_file_path' in locals() and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            raise

    async def _upload_text_as_file(self, text_content: str, display_name: str) -> Any:
        """Uploads extracted text to Gemini as a new .txt file."""
        logger.info(f"Performing text-based upload for: {display_name}")
        # This flow also uses a temporary file to upload the extracted text
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt", encoding='utf-8') as temp_file:
                temp_file.write(text_content)
                temp_file_path = temp_file.name

            gemini_file = await asyncio.to_thread(
                genai.upload_file,
                path=temp_file_path,
                display_name=f"{display_name}.txt",
                mime_type="text/plain"
            )
            pathlib.Path(temp_file_path).unlink() # Clean up the temp file
            return gemini_file
        except Exception:
            if 'temp_file_path' in locals() and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            raise

    async def process_and_upload(self, url: str) -> Any:
        """
        Main orchestration method. Downloads a document, processes it based on
        its type, and uploads it to Gemini.
        """
        extension, file_name = self._get_file_info(url)
        file_content = await self.download_document(url)

        if extension == ".pdf":
            # For PDFs, upload the original file directly
            return await self._upload_raw_file(file_content, file_name)
        
        elif extension == ".docx":
            # For DOCX, extract text and upload as a .txt file
            text = parse_docx(file_content)
            return await self._upload_text_as_file(text, file_name)

        elif extension == ".eml":
            # For EML, extract text and upload as a .txt file
            text = parse_eml(file_content)
            return await self._upload_text_as_file(text, file_name)
            
        else:
            # Fallback for other file types: attempt a direct upload
            logger.warning(f"Unsupported file type '{extension}'. Attempting direct upload.")
            return await self._upload_raw_file(file_content, file_name)
