import logging
import asyncio
import google.generativeai as genai
from typing import Any, Tuple, List
import tempfile
import pathlib
from urllib.parse import urlparse
import aiohttp
import fitz  # PyMuPDF

from src.utils.document_parsers import parse_docx, parse_eml

logger = logging.getLogger(__name__)

class IngestionService:
    """
    A service to download, process, and upload documents to the Gemini API.
    Large PDFs (>30MB) are split into chunks of pages, each under 25MB.
    """
    MB_30 = 30 * 1024 * 1024
    MB_25 = 25 * 1024 * 1024

    async def download_document(self, url: str) -> bytes:
        """Downloads document content from a URL asynchronously."""
        logger.info(f"Downloading document from: {url}")
        timeout = aiohttp.ClientTimeout(total=180) # Increased timeout for very large files
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

    async def _upload_pdf_in_chunks(self, file_content: bytes, display_name: str) -> List[Any]:
        """
        Splits a large PDF into chunks of pages, each aiming to be under 25MB,
        and uploads each chunk as a separate file.
        """
        logger.info(f"PDF is large. Splitting and uploading in chunks: {display_name}")
        all_uploaded_files = []
        
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            current_chunk_pages = []
            current_chunk_size = 0
            chunk_num = 1

            for page_num in range(len(pdf_document)):
                # Create a new single-page PDF for accurate size measurement
                single_page_pdf = fitz.open()
                single_page_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
                page_bytes = single_page_pdf.tobytes()
                single_page_pdf.close()

                # If adding this page exceeds the chunk size, process the current chunk first
                if current_chunk_pages and current_chunk_size + len(page_bytes) > self.MB_25:
                    logger.info(f"Finalizing chunk {chunk_num} with size {current_chunk_size / (1024*1024):.2f}MB...")
                    # Combine pages in the current chunk into a single PDF
                    chunk_pdf = fitz.open()
                    for p_num in current_chunk_pages:
                        chunk_pdf.insert_pdf(pdf_document, from_page=p_num, to_page=p_num)
                    
                    chunk_bytes = chunk_pdf.tobytes()
                    chunk_pdf.close()
                    
                    chunk_display_name = f"{display_name}_chunk_{chunk_num}"
                    gemini_file = await self._upload_raw_file(chunk_bytes, chunk_display_name, "application/pdf")
                    all_uploaded_files.append(gemini_file)
                    
                    # Reset for the next chunk
                    current_chunk_pages = []
                    current_chunk_size = 0
                    chunk_num += 1

                # Add the current page to the new chunk
                current_chunk_pages.append(page_num)
                current_chunk_size += len(page_bytes)

            # Process the last remaining chunk
            if current_chunk_pages:
                logger.info(f"Finalizing final chunk {chunk_num} with size {current_chunk_size / (1024*1024):.2f}MB...")
                chunk_pdf = fitz.open()
                for p_num in current_chunk_pages:
                    chunk_pdf.insert_pdf(pdf_document, from_page=p_num, to_page=p_num)
                
                chunk_bytes = chunk_pdf.tobytes()
                chunk_pdf.close()

                chunk_display_name = f"{display_name}_chunk_{chunk_num}"
                gemini_file = await self._upload_raw_file(chunk_bytes, chunk_display_name, "application/pdf")
                all_uploaded_files.append(gemini_file)

            logger.info(f"Successfully uploaded {len(all_uploaded_files)} chunks for the PDF.")
            return all_uploaded_files

        except Exception as e:
            logger.error(f"Failed to split and upload PDF chunks: {e}", exc_info=True)
            # Attempt to clean up any files that were uploaded before the error
            for f in all_uploaded_files:
                await asyncio.to_thread(genai.delete_file, f.name)
            raise

    async def _upload_raw_file(self, file_content: bytes, display_name: str, mime_type: str = None) -> Any:
        """Uploads raw file content by writing it to a temporary file first."""
        logger.info(f"Uploading: {display_name} ({len(file_content) / (1024*1024):.2f}MB)")
        temp_file_path = None
        try:
            # Create a temporary file to hold the in-memory bytes
            with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(display_name).suffix) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            # Upload the file using its path
            gemini_file = await asyncio.to_thread(
                genai.upload_file,
                path=temp_file_path,
                display_name=display_name,
                mime_type=mime_type
            )
            return gemini_file
        except Exception as e:
            logger.error(f"Error during file upload for {display_name}: {e}", exc_info=True)
            raise
        finally:
            # Ensure the temporary file is always cleaned up
            if temp_file_path and pathlib.Path(temp_file_path).exists():
                pathlib.Path(temp_file_path).unlink()
                logger.info(f"Cleaned up temporary file: {temp_file_path}")

    async def _upload_text_as_file(self, text_content: str, display_name: str) -> List[Any]:
        """Uploads extracted text to Gemini as a new .txt file."""
        logger.info(f"Performing text-based upload for: {display_name}")
        file_bytes = text_content.encode('utf-8')
        gemini_file = await self._upload_raw_file(
            file_bytes, f"{display_name}.txt", "text/plain"
        )
        return [gemini_file]

    async def process_and_upload(self, url: str) -> List[Any]:
        """
        Main orchestration method. Downloads a document, processes it, and uploads it.
        Large PDFs are split into chunks. Returns a list of Gemini file objects.
        """
        extension, file_name = self._get_file_info(url)
        file_content = await self.download_document(url)

        if extension == ".pdf" and len(file_content) > self.MB_30:
            return await self._upload_pdf_in_chunks(file_content, file_name)
        
        elif extension in [".docx", ".eml"]:
            text_parser = parse_docx if extension == ".docx" else parse_eml
            text = text_parser(file_content)
            return await self._upload_text_as_file(text, file_name)
            
        else:
            # For small PDFs and other file types, upload directly
            file = await self._upload_raw_file(file_content, file_name)
            return [file]