import logging
import aiohttp
import asyncio
import google.generativeai as genai
from typing import Any
import tempfile
import pathlib

logger = logging.getLogger(__name__)

class IngestionService:
    """A service for downloading documents and uploading them to Gemini."""
    async def download_document(self, url: str) -> bytes:
        """
        Downloads document content from a URL asynchronously.
        """
        logger.info(f"Downloading document from: {url}")
        timeout = aiohttp.ClientTimeout(total=60)  # 60-second timeout for download
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()  # Raise an exception for bad status codes
                    content = await response.read()
                    logger.info("Document downloaded successfully.")
                    return content
            except aiohttp.ClientError as e:
                logger.error(f"Network error downloading document: {e}")
                raise Exception(f"Failed to download document from URL: {url}")

    async def upload_to_gemini(self, file_content: bytes, display_name: str, mime_type: str) -> Any:
        """
        Uploads file content to the Gemini API by first saving it to a temporary file.
        """
        logger.info(f"Uploading document ('{display_name}') to Gemini API via temporary file...")
        try:
            # Create a temporary file to store the downloaded content
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            logger.info(f"Content written to temporary file: {temp_file_path}")

            # Upload the file from its temporary path, now with the mime_type
            gemini_file = await asyncio.to_thread(
                genai.upload_file,
                path=temp_file_path,
                display_name=display_name,
                mime_type=mime_type
            )
            
            # Clean up the temporary file
            pathlib.Path(temp_file_path).unlink()
            
            logger.info(f"Successfully uploaded file to Gemini: {gemini_file.name}")
            return gemini_file
            
        except Exception as e:
            logger.error(f"Failed to upload file to Gemini: {e}", exc_info=True)
            # Ensure cleanup happens on error too
            if 'temp_file_path' in locals() and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
            raise