"""
Enhanced Document Processor for BajajFinsev Hybrid RAG System
Integrates document downloading, text extraction, and preprocessing
Supports multiple formats with intelligent format detection and processing
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import tempfile
import json

# Service integrations
from src.services.document_downloader import get_document_downloader, DownloadResult
from src.services.text_extractor import get_text_extractor, ExtractionResult
from src.services.embedding_service import get_embedding_service, EmbeddingResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Result of complete document processing"""
    success: bool
    document_url: str
    document_hash: str = ""
    file_path: Optional[str] = None
    file_format: str = ""
    file_size: int = 0
    
    # Content
    raw_text: str = ""
    word_count: int = 0
    page_count: int = 0
    
    # Processing metadata
    download_time: float = 0.0
    extraction_time: float = 0.0
    total_processing_time: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    partial_success: bool = False
    
    # Structured data
    metadata: Optional[Dict[str, Any]] = None
    structured_content: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    text_quality_score: float = 0.0
    processing_warnings: List[str] = None


@dataclass
class ProcessingStats:
    """Document processing statistics"""
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    partially_processed: int = 0
    
    # Format breakdown
    format_stats: Dict[str, int] = None
    
    # Performance metrics
    avg_download_time: float = 0.0
    avg_extraction_time: float = 0.0
    avg_total_processing_time: float = 0.0
    
    # Quality metrics
    avg_text_quality_score: float = 0.0
    total_words_extracted: int = 0
    total_pages_processed: int = 0


class EnhancedDocumentProcessor:
    """
    Enhanced document processor with multi-format support
    Orchestrates document downloading, text extraction, and quality assessment
    """
    
    # Format-specific processing configurations
    PROCESSING_CONFIGS = {
        ".pdf": {
            "ocr_fallback": True,
            "page_analysis": True,
            "table_extraction": True,
            "image_extraction": True,
            "quality_threshold": 0.7
        },
        ".docx": {
            "table_extraction": True,
            "formatting_preservation": True,
            "embedded_objects": True,
            "quality_threshold": 0.8
        },
        ".xlsx": {
            "sheet_analysis": True,
            "data_type_detection": True,
            "formula_extraction": False,
            "quality_threshold": 0.9
        },
        ".txt": {
            "encoding_detection": True,
            "structure_analysis": False,
            "quality_threshold": 0.6
        },
        # Image formats
        ".jpg": {"ocr_preprocessing": True, "quality_threshold": 0.5},
        ".jpeg": {"ocr_preprocessing": True, "quality_threshold": 0.5},
        ".png": {"ocr_preprocessing": True, "quality_threshold": 0.6},
        ".bmp": {"ocr_preprocessing": True, "quality_threshold": 0.4},
        ".tiff": {"ocr_preprocessing": True, "quality_threshold": 0.5}
    }
    
    def __init__(self,
                 enable_caching: bool = True,
                 temp_dir: Optional[str] = None,
                 cleanup_temp_files: bool = True,
                 max_concurrent_downloads: int = 3,
                 quality_assessment: bool = True):
        """
        Initialize enhanced document processor
        
        Args:
            enable_caching: Enable document processing cache
            temp_dir: Custom temporary directory
            cleanup_temp_files: Auto-cleanup temporary files
            max_concurrent_downloads: Max concurrent document downloads
            quality_assessment: Enable text quality assessment
        """
        self.enable_caching = enable_caching
        self.cleanup_temp_files = cleanup_temp_files
        self.max_concurrent_downloads = max_concurrent_downloads
        self.quality_assessment = quality_assessment
        
        # Setup temporary directory
        if temp_dir:
            self.temp_dir = Path(temp_dir)
        else:
            self.temp_dir = Path(tempfile.gettempdir()) / "enhanced_doc_processor"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.document_downloader = get_document_downloader()
        self.text_extractor = get_text_extractor()
        self.embedding_service = get_embedding_service()
        
        # Processing cache
        self.processing_cache: Dict[str, ProcessedDocument] = {} if enable_caching else None
        
        # Statistics tracking
        self.stats = ProcessingStats(format_stats={})
        
        # Processing history
        self.processing_history: List[Dict[str, Any]] = []
        
        logger.info("EnhancedDocumentProcessor initialized")
        logger.info(f"Temporary directory: {self.temp_dir}")
        logger.info(f"Caching enabled: {enable_caching}")
        logger.info(f"Quality assessment: {quality_assessment}")
        logger.info(f"Supported formats: {len(self.PROCESSING_CONFIGS)}")
    
    async def process_document(self, document_url: str, **kwargs) -> ProcessedDocument:
        """
        Process a single document with enhanced multi-format support
        
        Args:
            document_url: URL of document to process
            **kwargs: Additional processing options
            
        Returns:
            ProcessedDocument with all extracted content and metadata
        """
        start_time = time.time()
        document_hash = self._generate_document_hash(document_url)
        
        logger.info(f"📄 Processing document: {document_url}")
        logger.info(f"Document hash: {document_hash}")
        
        try:
            # Check cache first
            if self.enable_caching and document_hash in self.processing_cache:
                cached_result = self.processing_cache[document_hash]
                logger.info(f"✅ Using cached processed document: {cached_result.file_format}")
                return cached_result
            
            # Initialize result
            result = ProcessedDocument(
                success=False,
                document_url=document_url,
                document_hash=document_hash,
                processing_warnings=[]
            )
            
            # Step 1: Download document
            logger.debug("⬇️ Step 1: Downloading document")
            download_start = time.time()
            
            download_result = await self.document_downloader.download_document(document_url)
            result.download_time = time.time() - download_start
            
            if not download_result.success:
                result.error_message = f"Download failed: {download_result.error_message}"
                self._update_stats(result, "failed")
                return result
            
            # Update result with download info
            result.file_path = download_result.file_path
            result.file_format = Path(download_result.file_path).suffix.lower()
            result.file_size = download_result.file_size
            
            logger.info(f"✅ Downloaded: {result.file_format} ({result.file_size:,} bytes)")
            
            # Step 2: Extract text with format-specific processing
            logger.debug("📝 Step 2: Extracting text")
            extraction_start = time.time()
            
            extraction_result = await self._extract_text_enhanced(
                result.file_path, 
                result.file_format,
                **kwargs
            )
            result.extraction_time = time.time() - extraction_start
            
            if not extraction_result.success:
                result.error_message = f"Text extraction failed: {extraction_result.error_message}"
                result.partial_success = True  # We have the file, just no text
                self._update_stats(result, "partial")
                return result
            
            # Update result with extraction info
            result.raw_text = extraction_result.text
            result.word_count = extraction_result.word_count
            result.page_count = extraction_result.page_count
            result.metadata = extraction_result.metadata
            result.structured_content = extraction_result.structured_data
            
            logger.info(f"✅ Extracted: {result.word_count:,} words, {result.page_count} pages")
            
            # Step 3: Quality assessment
            if self.quality_assessment:
                logger.debug("🎯 Step 3: Assessing text quality")
                result.text_quality_score = await self._assess_text_quality(
                    result.raw_text, 
                    result.file_format
                )
                
                if result.text_quality_score < 0.3:
                    result.processing_warnings.append(f"Low text quality score: {result.text_quality_score:.2f}")
                
                logger.debug(f"Quality score: {result.text_quality_score:.2f}")
            
            # Step 4: Format-specific post-processing
            logger.debug("⚙️ Step 4: Format-specific processing")
            await self._apply_format_specific_processing(result)
            
            # Step 5: Finalize processing
            result.total_processing_time = time.time() - start_time
            result.success = True
            
            # Cache result
            if self.enable_caching:
                self.processing_cache[document_hash] = result
                logger.debug(f"📦 Cached processed document: {document_hash}")
            
            # Update statistics
            self._update_stats(result, "success")
            
            # Add to processing history
            self._add_to_history(result)
            
            # Cleanup temp files if enabled
            if self.cleanup_temp_files and result.file_path:
                await self._cleanup_temp_file(result.file_path)
            
            logger.info(f"✅ Document processed successfully in {result.total_processing_time:.2f}s")
            logger.info(f"Quality: {result.text_quality_score:.2f}, Words: {result.word_count:,}")
            
            return result
            
        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            result.error_message = error_msg
            result.total_processing_time = time.time() - start_time
            self._update_stats(result, "failed")
            
            return result
    
    async def _extract_text_enhanced(self, 
                                   file_path: str, 
                                   file_format: str,
                                   **kwargs) -> ExtractionResult:
        """Enhanced text extraction with format-specific optimizations"""
        
        # Get format-specific config
        config = self.PROCESSING_CONFIGS.get(file_format, {})
        
        # Apply format-specific optimizations to text extractor
        if file_format == ".pdf" and config.get("ocr_fallback"):
            # Enable aggressive OCR for PDFs
            self.text_extractor.enable_ocr = True
            
        elif file_format in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            # Image-specific OCR preprocessing
            if config.get("ocr_preprocessing"):
                self.text_extractor.enable_ocr = True
        
        # Extract text
        extraction_result = await self.text_extractor.extract_text(file_path)
        
        return extraction_result
    
    async def _assess_text_quality(self, text: str, file_format: str) -> float:
        """
        Assess text quality based on various metrics
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0
        
        quality_factors = []
        
        # Length factor
        text_length = len(text.strip())
        if text_length < 50:
            length_score = 0.1
        elif text_length < 200:
            length_score = 0.5
        elif text_length < 1000:
            length_score = 0.8
        else:
            length_score = 1.0
        
        quality_factors.append(('length', length_score, 0.2))
        
        # Word count factor
        words = text.split()
        word_count = len(words)
        if word_count < 10:
            word_score = 0.1
        elif word_count < 50:
            word_score = 0.5
        elif word_count < 200:
            word_score = 0.8
        else:
            word_score = 1.0
        
        quality_factors.append(('word_count', word_score, 0.2))
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        total_chars = len(text)
        diversity_ratio = unique_chars / total_chars if total_chars > 0 else 0
        
        if diversity_ratio > 0.05:
            diversity_score = 1.0
        elif diversity_ratio > 0.02:
            diversity_score = 0.7
        else:
            diversity_score = 0.3
        
        quality_factors.append(('diversity', diversity_score, 0.15))
        
        # Sentence structure (basic)
        sentence_endings = text.count('.') + text.count('!') + text.count('?')
        if sentence_endings > 0:
            sentence_score = min(1.0, sentence_endings / (word_count / 10))
        else:
            sentence_score = 0.2
        
        quality_factors.append(('sentences', sentence_score, 0.15))
        
        # Format-specific quality assessment
        format_config = self.PROCESSING_CONFIGS.get(file_format, {})
        format_threshold = format_config.get('quality_threshold', 0.5)
        
        # OCR quality check for image formats
        if file_format in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Check for OCR artifacts
            ocr_artifacts = ['|', '\\', '~', '^', '`']
            artifact_count = sum(text.count(artifact) for artifact in ocr_artifacts)
            ocr_score = max(0.0, 1.0 - (artifact_count / text_length * 10))
        else:
            ocr_score = 1.0
        
        quality_factors.append(('ocr_quality', ocr_score, 0.2))
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for factor_name, score, weight in quality_factors:
            total_score += score * weight
            total_weight += weight
            logger.debug(f"Quality factor {factor_name}: {score:.2f} (weight: {weight})")
        
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Apply format-specific adjustment
        final_score = min(1.0, final_score * (1.0 + (format_threshold - 0.5)))
        
        return final_score
    
    async def _apply_format_specific_processing(self, result: ProcessedDocument):
        """Apply format-specific post-processing enhancements"""
        
        config = self.PROCESSING_CONFIGS.get(result.file_format, {})
        
        if result.file_format == ".pdf":
            # PDF-specific processing
            if config.get("page_analysis") and result.structured_content:
                pages = result.structured_content.get("pages", [])
                if pages:
                    result.processing_warnings.append(f"PDF has {len(pages)} pages with varying content density")
        
        elif result.file_format == ".xlsx":
            # Excel-specific processing
            if config.get("sheet_analysis") and result.structured_content:
                sheets = result.structured_content.get("sheets", [])
                if len(sheets) > 5:
                    result.processing_warnings.append(f"Excel file has {len(sheets)} sheets - consider splitting")
        
        elif result.file_format == ".docx":
            # Word document processing
            if config.get("table_extraction") and result.structured_content:
                tables = result.structured_content.get("tables", [])
                if tables:
                    result.processing_warnings.append(f"Document contains {len(tables)} tables")
        
        # Add format-specific metadata
        if not result.metadata:
            result.metadata = {}
        
        result.metadata["processing_config"] = config
        result.metadata["format_specific_processing"] = True
    
    async def batch_process_documents(self, 
                                    document_urls: List[str], 
                                    **kwargs) -> List[ProcessedDocument]:
        """
        Process multiple documents concurrently
        
        Args:
            document_urls: List of document URLs to process
            **kwargs: Additional processing options
            
        Returns:
            List of ProcessedDocument results
        """
        logger.info(f"📦 Starting batch document processing: {len(document_urls)} documents")
        
        # Control concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        async def process_with_semaphore(url):
            async with semaphore:
                return await self.process_document(url, **kwargs)
        
        # Process all documents
        start_time = time.time()
        tasks = [process_with_semaphore(url) for url in document_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Batch processing failed for {document_urls[i]}: {result}")
                error_result = ProcessedDocument(
                    success=False,
                    document_url=document_urls[i],
                    error_message=str(result),
                    document_hash=self._generate_document_hash(document_urls[i])
                )
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        # Log batch results
        successful = sum(1 for r in final_results if r.success)
        partial = sum(1 for r in final_results if r.partial_success)
        total_time = time.time() - start_time
        
        logger.info(f"📦 Batch processing complete: {successful}/{len(document_urls)} successful, "
                   f"{partial} partial, {total_time:.2f}s total")
        
        return final_results
    
    def _generate_document_hash(self, document_url: str) -> str:
        """Generate unique hash for document URL"""
        return hashlib.md5(document_url.encode()).hexdigest()
    
    def _update_stats(self, result: ProcessedDocument, outcome: str):
        """Update processing statistics"""
        self.stats.total_documents += 1
        
        if outcome == "success":
            self.stats.successful_documents += 1
            self.stats.total_words_extracted += result.word_count
            self.stats.total_pages_processed += result.page_count
            
            # Update averages
            self._update_average('avg_download_time', result.download_time)
            self._update_average('avg_extraction_time', result.extraction_time)
            self._update_average('avg_total_processing_time', result.total_processing_time)
            self._update_average('avg_text_quality_score', result.text_quality_score)
            
        elif outcome == "partial":
            self.stats.partially_processed += 1
        else:
            self.stats.failed_documents += 1
        
        # Update format stats
        if result.file_format:
            if result.file_format not in self.stats.format_stats:
                self.stats.format_stats[result.file_format] = 0
            self.stats.format_stats[result.file_format] += 1
    
    def _update_average(self, stat_name: str, new_value: float):
        """Update running average for a statistic"""
        current_avg = getattr(self.stats, stat_name)
        successful_count = self.stats.successful_documents
        
        if successful_count > 0:
            new_avg = ((current_avg * (successful_count - 1)) + new_value) / successful_count
            setattr(self.stats, stat_name, new_avg)
    
    def _add_to_history(self, result: ProcessedDocument):
        """Add processing result to history"""
        history_entry = {
            "timestamp": time.time(),
            "document_url": result.document_url,
            "document_hash": result.document_hash,
            "success": result.success,
            "file_format": result.file_format,
            "word_count": result.word_count,
            "processing_time": result.total_processing_time,
            "quality_score": result.text_quality_score
        }
        
        self.processing_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]
    
    async def _cleanup_temp_file(self, file_path: str):
        """Clean up temporary file"""
        try:
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
                logger.debug(f"🗑️ Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup temp file {file_path}: {e}")
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            "stats": {
                "total_documents": self.stats.total_documents,
                "successful_documents": self.stats.successful_documents,
                "failed_documents": self.stats.failed_documents,
                "partially_processed": self.stats.partially_processed,
                "success_rate": (
                    self.stats.successful_documents / self.stats.total_documents * 100
                    if self.stats.total_documents > 0 else 0
                ),
                "format_breakdown": self.stats.format_stats,
                "performance": {
                    "avg_download_time": self.stats.avg_download_time,
                    "avg_extraction_time": self.stats.avg_extraction_time,
                    "avg_total_processing_time": self.stats.avg_total_processing_time,
                    "avg_text_quality_score": self.stats.avg_text_quality_score
                },
                "content": {
                    "total_words_extracted": self.stats.total_words_extracted,
                    "total_pages_processed": self.stats.total_pages_processed
                }
            },
            "configuration": {
                "supported_formats": list(self.PROCESSING_CONFIGS.keys()),
                "caching_enabled": self.enable_caching,
                "quality_assessment": self.quality_assessment,
                "max_concurrent_downloads": self.max_concurrent_downloads,
                "temp_directory": str(self.temp_dir)
            },
            "cache_info": {
                "cached_documents": len(self.processing_cache) if self.processing_cache else 0,
                "cache_enabled": self.enable_caching
            },
            "service_status": {
                "document_downloader": "initialized",
                "text_extractor": "initialized",
                "embedding_service": "initialized"
            }
        }
    
    def clear_cache(self) -> int:
        """Clear processing cache"""
        if not self.enable_caching or not self.processing_cache:
            return 0
        
        cache_size = len(self.processing_cache)
        self.processing_cache.clear()
        logger.info(f"🧹 Cleared {cache_size} cached processed documents")
        return cache_size
    
    async def cleanup(self):
        """Cleanup processor resources"""
        logger.info("🧹 Cleaning up document processor resources")
        
        try:
            # Clear cache
            if self.processing_cache:
                self.processing_cache.clear()
            
            # Clear processing history
            self.processing_history.clear()
            
            # Cleanup services
            if hasattr(self.embedding_service, 'cleanup'):
                await self.embedding_service.cleanup()
            
            # Cleanup temp directory if empty
            try:
                if self.temp_dir.exists() and not any(self.temp_dir.iterdir()):
                    self.temp_dir.rmdir()
                    logger.debug(f"🗑️ Removed empty temp directory: {self.temp_dir}")
            except Exception as e:
                logger.debug(f"Could not remove temp directory: {e}")
            
            logger.info("✅ Document processor cleanup complete")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")


# Global document processor instance
document_processor: Optional[EnhancedDocumentProcessor] = None


def get_document_processor(**kwargs) -> EnhancedDocumentProcessor:
    """Get or create global document processor instance"""
    global document_processor
    
    if document_processor is None:
        document_processor = EnhancedDocumentProcessor(**kwargs)
    
    return document_processor


async def initialize_document_processor(**kwargs) -> EnhancedDocumentProcessor:
    """Initialize and return document processor"""
    processor = get_document_processor(**kwargs)
    
    # Initialize embedding service
    await processor.embedding_service.initialize()
    
    # Log initialization summary
    stats = await processor.get_processing_stats()
    logger.info("📄 Enhanced Document Processor Summary:")
    logger.info(f"  Supported formats: {len(stats['configuration']['supported_formats'])}")
    logger.info(f"  Caching enabled: {stats['configuration']['caching_enabled']}")
    logger.info(f"  Quality assessment: {stats['configuration']['quality_assessment']}")
    logger.info(f"  Max concurrent: {stats['configuration']['max_concurrent_downloads']}")
    logger.info(f"  Temp directory: {stats['configuration']['temp_directory']}")
    
    return processor