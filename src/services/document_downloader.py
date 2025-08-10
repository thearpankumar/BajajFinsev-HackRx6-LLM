"""
Document Downloader Service
Handles downloading documents from HTTP/HTTPS URLs with validation and progress tracking
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Callable, Union
from urllib.parse import unquote, urlparse

try:
    import aiofiles
    import aiohttp
except ImportError:
    raise ImportError("Required packages missing. Install with: pip install aiohttp aiofiles")

from src.core.config import config

logger = logging.getLogger(__name__)


class DocumentDownloader:
    """
    Asynchronous document downloader with progress tracking and validation
    Integrated with centralized configuration
    """

    def __init__(self):
        self.max_file_size_mb = config.max_document_size_mb
        self.supported_formats = config.supported_formats
        self.download_timeout = config.query_timeout_seconds
        self.session: Union[aiohttp.ClientSession, None] = None

        # Create downloads directory
        self.downloads_dir = Path("./downloads")
        self.downloads_dir.mkdir(exist_ok=True)

        logger.info(f"DocumentDownloader initialized: max_size={self.max_file_size_mb}MB")

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.download_timeout),
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=5)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _validate_url(self, url: str) -> tuple[bool, str]:
        """Validate URL format and scheme"""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format"

            if parsed.scheme not in ['http', 'https']:
                return False, f"Unsupported scheme: {parsed.scheme}"

            return True, "URL is valid"
        except Exception as e:
            return False, f"URL validation error: {str(e)}"

    def _extract_filename_from_url(self, url: str, file_type: str = 'txt') -> str:
        """Extract filename from URL, using file_type to generate a better default."""
        try:
            parsed = urlparse(url)
            filename = unquote(Path(parsed.path).name)

            if not filename or '.' not in filename:
                # Generate filename from URL hash with the correct extension
                url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
                filename = f"document_{url_hash}.{file_type}"

            return filename
        except Exception:
            # Fallback filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            return f"document_{url_hash}.{file_type}"

    def _detect_file_type(self, url: str, headers: dict[str, str]) -> Union[str, None]:
        """Detect file type from URL and headers"""
        try:
            # Check Content-Type header first
            content_type = headers.get('content-type', '').lower()
            if content_type:
                # Map common MIME types
                mime_map = {
                    'application/pdf': 'pdf',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                    'application/msword': 'doc',
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                    'application/vnd.ms-excel': 'xls',
                    'text/csv': 'csv',
                    'image/jpeg': 'jpg',
                    'image/jpg': 'jpg',
                    'image/png': 'png',
                    'image/bmp': 'bmp',
                    'image/tiff': 'tiff',
                    'image/webp': 'webp',
                    'text/plain': 'txt',
                    'application/json': 'json',
                    'text/html': 'html'  # Added HTML detection
                }

                for mime_type, ext in mime_map.items():
                    if mime_type in content_type:
                        return ext
                
                # Flexible checks for text-based content
                if 'text' in content_type:
                    return 'txt'
                if 'json' in content_type:
                    return 'json'

            # Fall back to URL extension
            parsed = urlparse(url)
            filename = unquote(Path(parsed.path).name)
            if '.' in filename:
                ext = filename.split('.')[-1].lower()
                if ext in self.supported_formats:
                    return ext

            # If no extension and no content type, default to text
            return 'txt'
        except Exception as e:
            logger.warning(f"File type detection failed: {str(e)}")
            return 'txt'

    def _validate_file_type(self, file_type: Union[str, None]) -> tuple[bool, str]:
        """Validate if file type is supported"""
        if not file_type:
            return False, "Unknown file type"

        if file_type not in self.supported_formats:
            return False, f"Unsupported file type: {file_type}. Supported: {', '.join(self.supported_formats)}"

        return True, f"File type {file_type} is supported"

    async def get_file_info(self, url: str) -> dict[str, Any]:
        """Get file information without downloading"""
        # Validate URL
        is_valid, validation_message = self._validate_url(url)
        if not is_valid:
            return {
                "status": "error",
                "error": validation_message,
                "url": url
            }

        try:
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")

            # HEAD request to get file info
            async with self.session.head(url) as response:
                headers = dict(response.headers)

                # Get file size
                content_length = headers.get('content-length')
                file_size_bytes = int(content_length) if content_length else None
                file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes else None

                # Detect file type
                file_type = self._detect_file_type(url, headers)

                # Validate file type
                type_valid, type_message = self._validate_file_type(file_type)

                # Validate file size
                size_valid = True
                size_message = "File size OK"
                if file_size_mb and file_size_mb > self.max_file_size_mb:
                    size_valid = False
                    size_message = f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB"

                return {
                    "status": "success",
                    "url": url,
                    "filename": self._extract_filename_from_url(url, file_type),
                    "file_type": file_type,
                    "file_size_bytes": file_size_bytes,
                    "file_size_mb": round(file_size_mb, 2) if file_size_mb else None,
                    "content_type": headers.get('content-type'),
                    "last_modified": headers.get('last-modified'),
                    "validation": {
                        "type_valid": type_valid,
                        "type_message": type_message,
                        "size_valid": size_valid,
                        "size_message": size_message,
                        "overall_valid": type_valid and size_valid
                    },
                    "headers": headers
                }

        except TimeoutError:
            return {
                "status": "error",
                "error": f"Timeout after {self.download_timeout}s",
                "url": url
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "url": url
            }

    async def download_document(
        self,
        url: str,
        custom_filename: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> dict[str, Any]:
        """
        Download document from URL with validation and progress tracking
        
        Args:
            url: Document URL to download
            custom_filename: Optional custom filename
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with download results
        """
        logger.info(f"🔄 Starting document download from: {url}")

        try:
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")

            # Download with progress tracking using GET request
            async with self.session.get(url) as response:
                response.raise_for_status()
                
                # Get headers from the actual download response
                headers = dict(response.headers)
                
                # Now detect file type and validate
                file_type = self._detect_file_type(url, headers)
                type_valid, type_message = self._validate_file_type(file_type)
                
                content_length = headers.get('content-length')
                file_size_bytes = int(content_length) if content_length else None
                file_size_mb = file_size_bytes / (1024 * 1024) if file_size_bytes else None
                
                size_valid = True
                size_message = "File size OK"
                if file_size_mb and file_size_mb > self.max_file_size_mb:
                    size_valid = False
                    size_message = f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB"

                if not (type_valid and size_valid):
                    error_msg = f"Validation failed: {type_message}, {size_message}"
                    logger.error(f"❌ {error_msg}")
                    return {"status": "error", "error": error_msg, "url": url}

                # Determine filename
                filename = custom_filename or self._extract_filename_from_url(url, file_type)
                filepath = self.downloads_dir / filename
                
                logger.info(f"🔄 Downloading to: {filepath}")

                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0

                async with aiofiles.open(filepath, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        downloaded_size += len(chunk)

                        # Progress callback
                        if progress_callback and total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            await progress_callback(progress, downloaded_size, total_size)

            # Verify downloaded file
            actual_size = filepath.stat().st_size

            # Calculate file hash for integrity
            file_hash = hashlib.md5()
            async with aiofiles.open(filepath, 'rb') as f:
                while chunk := await f.read(8192):
                    file_hash.update(chunk)

            result = {
                "status": "success",
                "message": "Document downloaded successfully",
                "url": url,
                "filepath": str(filepath),
                "filename": filename,
                "file_type": file_type,
                "file_size_bytes": actual_size,
                "file_size_mb": round(actual_size / (1024 * 1024), 2),
                "file_hash": file_hash.hexdigest(),
                "download_info": {
                    "content_type": headers.get('content-type'),
                    "last_modified": headers.get('last-modified'),
                }
            }

            logger.info(f"✅ Document downloaded successfully: {filename} ({result['file_size_mb']}MB)")
            return result

        except TimeoutError:
            error_msg = f"Download timeout after {self.download_timeout}s"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "url": url
            }
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # Clean up partial download
            if 'filepath' in locals() and filepath.exists():
                try:
                    filepath.unlink()
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up partial download: {cleanup_error}")

            return {
                "status": "error",
                "error": error_msg,
                "url": url
            }

    async def download_with_retry(
        self,
        url: str,
        max_retries: int = 3,
        custom_filename: Union[str, None] = None
    ) -> dict[str, Any]:
        """Download document with retry logic"""

        for attempt in range(max_retries):
            logger.info(f"📥 Download attempt {attempt + 1}/{max_retries} for: {url}")

            result = await self.download_document(url, custom_filename)

            if result["status"] == "success":
                return result

            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                logger.info(f"⏳ Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        logger.error(f"❌ All download attempts failed for: {url}")
        return result  # Return the last failed attempt

    def cleanup_downloads(self, older_than_hours: int = 24) -> dict[str, Any]:
        """Clean up old downloaded files"""
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (older_than_hours * 3600)

            cleaned_files = []
            total_size_freed = 0

            for filepath in self.downloads_dir.iterdir():
                if filepath.is_file():
                    file_mtime = filepath.stat().st_mtime
                    if file_mtime < cutoff_time:
                        file_size = filepath.stat().st_size
                        filepath.unlink()
                        cleaned_files.append(str(filepath))
                        total_size_freed += file_size

            return {
                "status": "success",
                "cleaned_files_count": len(cleaned_files),
                "total_size_freed_mb": round(total_size_freed / (1024 * 1024), 2),
                "cleaned_files": cleaned_files
            }

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
