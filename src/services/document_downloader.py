"""
Document Downloader Service for BajajFinsev Hybrid RAG System
Downloads documents from URLs with support for various file formats
Handles HTTP/HTTPS, file validation, progress tracking, and error handling
"""

import os
import asyncio
import aiohttp
import aiofiles
import hashlib
import logging
import mimetypes
from urllib.parse import urlparse, unquote
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Result of document download operation"""
    success: bool
    file_path: Optional[str] = None
    file_size: int = 0
    mime_type: Optional[str] = None
    original_filename: Optional[str] = None
    download_time: float = 0.0
    error_message: Optional[str] = None
    file_hash: Optional[str] = None


class DocumentDownloader:
    """
    Advanced document downloader with support for multiple formats
    Features: progress tracking, file validation, caching, error handling
    """
    
    # Supported file formats and their MIME types
    SUPPORTED_FORMATS = {
        # Document formats
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.xls': 'application/vnd.ms-excel',
        '.csv': 'text/csv',
        '.txt': 'text/plain',
        '.rtf': 'application/rtf',
        
        # Image formats (for OCR)
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg', 
        '.png': 'image/png',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.gif': 'image/gif',
        
        # Presentation formats
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.ppt': 'application/vnd.ms-powerpoint'
    }
    
    def __init__(
        self,
        download_dir: Optional[str] = None,
        max_file_size_mb: int = 500,
        timeout_seconds: int = 300,
        chunk_size: int = 8192,
        enable_caching: bool = True
    ):
        """
        Initialize document downloader
        
        Args:
            download_dir: Directory to store downloaded files (default: temp)
            max_file_size_mb: Maximum file size in MB
            timeout_seconds: Download timeout in seconds
            chunk_size: Download chunk size in bytes
            enable_caching: Whether to cache downloaded files
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.timeout_seconds = timeout_seconds
        self.chunk_size = chunk_size
        self.enable_caching = enable_caching
        
        # Setup download directory
        if download_dir:
            self.download_dir = Path(download_dir)
        else:
            self.download_dir = Path(tempfile.gettempdir()) / "rag_downloads"
        
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Download statistics
        self.stats = {
            'total_downloads': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_bytes_downloaded': 0,
            'cache_hits': 0,
            'avg_download_time': 0.0
        }
        
        logger.info(f"DocumentDownloader initialized")
        logger.info(f"Download directory: {self.download_dir}")
        logger.info(f"Max file size: {max_file_size_mb} MB")
        logger.info(f"Supported formats: {len(self.SUPPORTED_FORMATS)}")
    
    def _get_file_hash(self, url: str) -> str:
        """Generate hash for URL to use as cache key"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL"""
        try:
            parsed = urlparse(url)
            filename = unquote(os.path.basename(parsed.path))
            
            if not filename or '.' not in filename:
                # Generate filename from hash if none found
                file_hash = self._get_file_hash(url)[:12]
                filename = f"document_{file_hash}.pdf"  # Default to PDF
                
            return filename
            
        except Exception as e:
            logger.warning(f"Failed to extract filename from URL: {e}")
            file_hash = self._get_file_hash(url)[:12]
            return f"document_{file_hash}.pdf"
    
    def _validate_file_type(self, filename: str, content_type: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate if file type is supported
        
        Returns:
            (is_supported, detected_extension)
        """
        # Get extension from filename
        file_ext = Path(filename).suffix.lower()
        
        # Check if extension is supported
        if file_ext in self.SUPPORTED_FORMATS:
            return True, file_ext
        
        # Try to detect from content type
        if content_type:
            for ext, mime_type in self.SUPPORTED_FORMATS.items():
                if mime_type in content_type.lower():
                    return True, ext
        
        return False, file_ext
    
    def _get_cached_file_path(self, url: str) -> Optional[str]:
        """Check if file is already cached"""
        if not self.enable_caching:
            return None
            
        file_hash = self._get_file_hash(url)
        
        # Look for any file with this hash prefix
        for file_path in self.download_dir.glob(f"*{file_hash}*"):
            if file_path.is_file():
                logger.debug(f"Found cached file: {file_path}")
                self.stats['cache_hits'] += 1
                return str(file_path)
        
        return None
    
    async def download_document(
        self, 
        url: str, 
        progress_callback: Optional[callable] = None
    ) -> DownloadResult:
        """
        Download document from URL with validation and caching
        
        Args:
            url: URL to download from
            progress_callback: Optional callback for progress updates
            
        Returns:
            DownloadResult with download status and file info
        """
        start_time = time.time()
        self.stats['total_downloads'] += 1
        
        logger.info(f"📥 Starting download: {url}")
        
        try:
            # Check cache first
            cached_path = self._get_cached_file_path(url)
            if cached_path and os.path.exists(cached_path):
                logger.info(f"✅ Using cached file: {cached_path}")
                
                # Get file info
                file_size = os.path.getsize(cached_path)
                mime_type = mimetypes.guess_type(cached_path)[0]
                
                return DownloadResult(
                    success=True,
                    file_path=cached_path,
                    file_size=file_size,
                    mime_type=mime_type,
                    original_filename=os.path.basename(cached_path),
                    download_time=0.0,  # Cached, no download time
                    file_hash=self._get_file_hash(url)
                )
            
            # Prepare for download
            filename = self._extract_filename_from_url(url)
            file_hash = self._get_file_hash(url)
            
            # Add hash to filename for uniqueness
            name, ext = os.path.splitext(filename)
            unique_filename = f"{name}_{file_hash}{ext}"
            file_path = self.download_dir / unique_filename
            
            # Download with aiohttp
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            ) as session:
                
                logger.debug(f"📡 Making HTTP request to: {url}")
                
                async with session.get(url) as response:
                    # Check HTTP status
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}: {response.reason}"
                        logger.error(f"❌ Download failed: {error_msg}")
                        self.stats['failed_downloads'] += 1
                        
                        return DownloadResult(
                            success=False,
                            error_message=error_msg,
                            download_time=time.time() - start_time
                        )
                    
                    # Get content info
                    content_length = response.headers.get('Content-Length')
                    content_type = response.headers.get('Content-Type', '')
                    
                    # Validate file size
                    if content_length and int(content_length) > self.max_file_size_bytes:
                        error_msg = f"File too large: {int(content_length)} bytes > {self.max_file_size_bytes} bytes"
                        logger.error(f"❌ {error_msg}")
                        self.stats['failed_downloads'] += 1
                        
                        return DownloadResult(
                            success=False,
                            error_message=error_msg,
                            download_time=time.time() - start_time
                        )
                    
                    # Validate file type
                    is_supported, detected_ext = self._validate_file_type(filename, content_type)
                    if not is_supported:
                        error_msg = f"Unsupported file type: {detected_ext} (Content-Type: {content_type})"
                        logger.error(f"❌ {error_msg}")
                        self.stats['failed_downloads'] += 1
                        
                        return DownloadResult(
                            success=False,
                            error_message=error_msg,
                            download_time=time.time() - start_time
                        )
                    
                    # Download file
                    logger.info(f"💾 Downloading to: {file_path}")
                    
                    downloaded_bytes = 0
                    async with aiofiles.open(file_path, 'wb') as file:
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            await file.write(chunk)
                            downloaded_bytes += len(chunk)
                            
                            # Check size limit during download
                            if downloaded_bytes > self.max_file_size_bytes:
                                await file.close()
                                file_path.unlink(missing_ok=True)  # Delete partial file
                                
                                error_msg = f"File size exceeded limit during download: {downloaded_bytes} bytes"
                                logger.error(f"❌ {error_msg}")
                                self.stats['failed_downloads'] += 1
                                
                                return DownloadResult(
                                    success=False,
                                    error_message=error_msg,
                                    download_time=time.time() - start_time
                                )
                            
                            # Progress callback
                            if progress_callback and content_length:
                                progress = (downloaded_bytes / int(content_length)) * 100
                                await progress_callback(progress, downloaded_bytes, int(content_length))
            
            # Verify file was downloaded
            if not file_path.exists() or file_path.stat().st_size == 0:
                error_msg = "Download failed: file not created or empty"
                logger.error(f"❌ {error_msg}")
                self.stats['failed_downloads'] += 1
                
                return DownloadResult(
                    success=False,
                    error_message=error_msg,
                    download_time=time.time() - start_time
                )
            
            # Final validation
            actual_size = file_path.stat().st_size
            actual_mime = mimetypes.guess_type(str(file_path))[0]
            
            download_time = time.time() - start_time
            
            # Update stats
            self.stats['successful_downloads'] += 1
            self.stats['total_bytes_downloaded'] += actual_size
            self._update_avg_download_time(download_time)
            
            logger.info(f"✅ Download successful: {actual_size:,} bytes in {download_time:.2f}s")
            
            return DownloadResult(
                success=True,
                file_path=str(file_path),
                file_size=actual_size,
                mime_type=actual_mime,
                original_filename=filename,
                download_time=download_time,
                file_hash=file_hash
            )
            
        except asyncio.TimeoutError:
            error_msg = f"Download timeout after {self.timeout_seconds} seconds"
            logger.error(f"❌ {error_msg}")
            self.stats['failed_downloads'] += 1
            
            return DownloadResult(
                success=False,
                error_message=error_msg,
                download_time=time.time() - start_time
            )
            
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.stats['failed_downloads'] += 1
            
            return DownloadResult(
                success=False,
                error_message=error_msg,
                download_time=time.time() - start_time
            )
    
    def _update_avg_download_time(self, download_time: float):
        """Update running average of download times"""
        successful_downloads = self.stats['successful_downloads']
        current_avg = self.stats['avg_download_time']
        
        self.stats['avg_download_time'] = (
            (current_avg * (successful_downloads - 1) + download_time) / successful_downloads
        )
    
    async def batch_download(
        self, 
        urls: list[str], 
        max_concurrent: int = 3
    ) -> list[DownloadResult]:
        """
        Download multiple documents concurrently
        
        Args:
            urls: List of URLs to download
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            List of DownloadResult objects
        """
        logger.info(f"📦 Starting batch download: {len(urls)} files, max concurrent: {max_concurrent}")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(url):
            async with semaphore:
                return await self.download_document(url)
        
        tasks = [download_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Batch download failed for {urls[i]}: {result}")
                final_results.append(DownloadResult(
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        successful = sum(1 for r in final_results if r.success)
        logger.info(f"📦 Batch download completed: {successful}/{len(urls)} successful")
        
        return final_results
    
    def cleanup_old_files(self, max_age_days: int = 7) -> int:
        """
        Clean up old downloaded files
        
        Args:
            max_age_days: Maximum age of files to keep
            
        Returns:
            Number of files deleted
        """
        if not self.download_dir.exists():
            return 0
        
        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        deleted_count = 0
        
        try:
            for file_path in self.download_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"🗑️ Deleted old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} old files")
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        
        return deleted_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get download statistics"""
        return {
            **self.stats,
            "download_dir": str(self.download_dir),
            "max_file_size_mb": self.max_file_size_bytes // (1024 * 1024),
            "timeout_seconds": self.timeout_seconds,
            "supported_formats": list(self.SUPPORTED_FORMATS.keys()),
            "cache_enabled": self.enable_caching,
            "success_rate": (
                self.stats['successful_downloads'] / self.stats['total_downloads'] * 100 
                if self.stats['total_downloads'] > 0 else 0
            )
        }


# Global document downloader instance
document_downloader: Optional[DocumentDownloader] = None


def get_document_downloader() -> DocumentDownloader:
    """Get or create global document downloader instance"""
    global document_downloader
    
    if document_downloader is None:
        document_downloader = DocumentDownloader()
    
    return document_downloader