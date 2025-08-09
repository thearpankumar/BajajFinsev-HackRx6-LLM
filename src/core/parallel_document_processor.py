"""
Parallel Document Processor
8-worker async architecture for high-speed document processing with load balancing
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
from dataclasses import dataclass

from src.core.config import config
from src.services.document_downloader import DocumentDownloader
from src.services.pdf_processor import PDFProcessor
from src.services.office_processor import OfficeProcessor
from src.services.basic_text_extractor import BasicTextExtractor
from src.services.language_detector import LanguageDetector
from src.services.redis_cache import redis_manager

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Data class for processing tasks"""
    task_id: str
    document_url: str
    file_path: str
    file_type: str
    priority: int = 1
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingResult:
    """Data class for processing results"""
    task_id: str
    status: str
    document_url: str
    file_path: str
    processing_time: float
    content: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    worker_id: Optional[int] = None


class ParallelDocumentProcessor:
    """
    High-performance parallel document processor with 8-worker architecture
    Supports load balancing, task distribution, and memory-efficient streaming
    """
    
    def __init__(self):
        # Configuration from central config
        self.max_workers = config.max_workers
        self.max_concurrent_operations = config.max_concurrent_operations
        self.max_document_size_mb = config.max_document_size_mb
        self.supported_formats = config.supported_formats
        
        # Worker management
        self.workers_active = 0
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.worker_stats: Dict[int, Dict[str, Any]] = {}
        
        # Processing services
        self.document_downloader = DocumentDownloader()
        self.pdf_processor = PDFProcessor()
        self.office_processor = OfficeProcessor()
        self.text_extractor = BasicTextExtractor()
        self.language_detector = LanguageDetector()
        
        # Performance tracking
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_processing_time = 0.0
        self.peak_concurrent_tasks = 0
        
        # Cache integration
        self.redis_manager = redis_manager
        self.enable_caching = config.enable_embedding_cache
        
        logger.info(f"ParallelDocumentProcessor initialized: {self.max_workers} workers, "
                   f"max concurrent: {self.max_concurrent_operations}")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the parallel processing system"""
        try:
            logger.info("🔄 Initializing Parallel Document Processor...")
            start_time = time.time()
            
            # Initialize worker stats
            for i in range(self.max_workers):
                self.worker_stats[i] = {
                    "worker_id": i,
                    "tasks_processed": 0,
                    "total_processing_time": 0.0,
                    "current_task": None,
                    "status": "idle",
                    "errors": 0
                }
            
            # Initialize cache if enabled
            if self.enable_caching and not self.redis_manager.is_connected:
                await self.redis_manager.initialize()
            
            initialization_time = time.time() - start_time
            
            result = {
                "status": "success",
                "message": f"Parallel processor initialized in {initialization_time:.2f}s",
                "max_workers": self.max_workers,
                "max_concurrent_operations": self.max_concurrent_operations,
                "supported_formats": self.supported_formats,
                "cache_enabled": self.enable_caching and self.redis_manager.is_connected,
                "initialization_time": initialization_time
            }
            
            logger.info(f"✅ {result['message']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Parallel processor initialization failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def process_documents(
        self,
        document_urls: List[str],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process multiple documents in parallel with optimal load distribution
        
        Args:
            document_urls: List of document URLs to process
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results with detailed statistics
        """
        logger.info(f"🚀 Starting parallel processing of {len(document_urls)} documents")
        start_time = time.time()
        
        try:
            # Create processing tasks
            tasks = []
            for i, url in enumerate(document_urls):
                task_id = f"task_{int(time.time())}_{i}"
                task = ProcessingTask(
                    task_id=task_id,
                    document_url=url,
                    file_path="",  # Will be set after download
                    file_type="",  # Will be detected
                    priority=1,
                    metadata={"url_index": i}
                )
                tasks.append(task)
            
            self.total_tasks += len(tasks)
            
            # Process tasks with workers
            results = await self._process_tasks_parallel(tasks, progress_callback)
            
            # Calculate final statistics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            successful_results = [r for r in results if r.status == "success"]
            failed_results = [r for r in results if r.status == "error"]
            
            self.completed_tasks += len(successful_results)
            self.failed_tasks += len(failed_results)
            
            # Aggregate all content
            aggregated_content = self._aggregate_results(successful_results)
            
            final_result = {
                "status": "success",
                "processing_summary": {
                    "total_documents": len(document_urls),
                    "successful": len(successful_results),
                    "failed": len(failed_results),
                    "success_rate": round((len(successful_results) / len(document_urls)) * 100, 1),
                    "total_processing_time": round(processing_time, 2),
                    "average_time_per_document": round(processing_time / len(document_urls), 2)
                },
                "aggregated_content": aggregated_content,
                "detailed_results": [self._result_to_dict(r) for r in results],
                "worker_performance": self._get_worker_performance(),
                "system_metrics": self._get_system_metrics()
            }
            
            logger.info(f"✅ Parallel processing completed: {len(successful_results)}/{len(document_urls)} "
                       f"successful in {processing_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            error_msg = f"Parallel processing failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "processing_time": time.time() - start_time
            }
    
    async def _process_tasks_parallel(
        self,
        tasks: List[ProcessingTask],
        progress_callback: Optional[Callable]
    ) -> List[ProcessingResult]:
        """Process tasks using parallel workers"""
        
        # Create semaphore for concurrent operation limiting
        semaphore = asyncio.Semaphore(self.max_concurrent_operations)
        
        # Create worker coroutines
        worker_tasks = []
        for worker_id in range(self.max_workers):
            worker_task = asyncio.create_task(
                self._worker_loop(worker_id, semaphore)
            )
            worker_tasks.append(worker_task)
        
        # Add tasks to queue
        for task in tasks:
            await self.task_queue.put(task)
        
        # Add sentinel values to stop workers
        for _ in range(self.max_workers):
            await self.task_queue.put(None)
        
        # Collect results with progress tracking
        results = []
        completed_count = 0
        
        while completed_count < len(tasks):
            try:
                # Wait for result with timeout
                result = await asyncio.wait_for(self.result_queue.get(), timeout=30.0)
                results.append(result)
                completed_count += 1
                
                # Update progress
                if progress_callback:
                    progress = (completed_count / len(tasks)) * 100
                    await progress_callback(progress, completed_count, len(tasks))
                
            except asyncio.TimeoutError:
                logger.warning("⏰ Timeout waiting for worker results")
                break
        
        # Cancel remaining worker tasks
        for worker_task in worker_tasks:
            worker_task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*worker_tasks, return_exceptions=True)
        
        return results
    
    async def _worker_loop(self, worker_id: int, semaphore: asyncio.Semaphore):
        """Worker loop for processing tasks"""
        logger.debug(f"👷 Worker {worker_id} started")
        
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                if task is None:  # Sentinel value to stop worker
                    logger.debug(f"👷 Worker {worker_id} stopping")
                    break
                
                # Process task with semaphore control
                async with semaphore:
                    result = await self._process_single_task(task, worker_id)
                    await self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"❌ Worker {worker_id} error: {str(e)}")
                # Create error result
                error_result = ProcessingResult(
                    task_id=task.task_id if 'task' in locals() else "unknown",
                    status="error",
                    document_url=task.document_url if 'task' in locals() else "unknown",
                    file_path="",
                    processing_time=0.0,
                    error=str(e),
                    worker_id=worker_id
                )
                await self.result_queue.put(error_result)
    
    async def _process_single_task(self, task: ProcessingTask, worker_id: int) -> ProcessingResult:
        """Process a single document task"""
        logger.debug(f"📄 Worker {worker_id} processing: {task.document_url}")
        start_time = time.time()
        
        # Update worker stats
        self.worker_stats[worker_id]["current_task"] = task.task_id
        self.worker_stats[worker_id]["status"] = "processing"
        
        try:
            # Step 1: Download document
            download_result = await self._download_document(task)
            if download_result["status"] != "success":
                raise Exception(f"Download failed: {download_result['error']}")
            
            task.file_path = download_result["filepath"]
            task.file_type = download_result["file_type"]
            
            # Step 2: Process document based on type
            content_result = await self._extract_content(task)
            if content_result["status"] != "success":
                raise Exception(f"Content extraction failed: {content_result['error']}")
            
            # Step 3: Language detection
            language_result = await self._detect_language(content_result["content"]["full_text"])
            
            # Step 4: Combine results
            final_content = {
                **content_result["content"],
                "language_detection": language_result,
                "file_info": download_result
            }
            
            processing_time = time.time() - start_time
            
            # Update worker stats
            self.worker_stats[worker_id]["tasks_processed"] += 1
            self.worker_stats[worker_id]["total_processing_time"] += processing_time
            self.worker_stats[worker_id]["current_task"] = None
            self.worker_stats[worker_id]["status"] = "idle"
            
            # Cache result if enabled
            if self.enable_caching and self.redis_manager.is_connected:
                await self._cache_result(task, final_content)
            
            result = ProcessingResult(
                task_id=task.task_id,
                status="success",
                document_url=task.document_url,
                file_path=task.file_path,
                processing_time=processing_time,
                content=final_content,
                worker_id=worker_id
            )
            
            logger.debug(f"✅ Worker {worker_id} completed task in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Update worker stats
            self.worker_stats[worker_id]["errors"] += 1
            self.worker_stats[worker_id]["current_task"] = None
            self.worker_stats[worker_id]["status"] = "idle"
            
            logger.error(f"❌ Worker {worker_id} task failed: {str(e)}")
            
            return ProcessingResult(
                task_id=task.task_id,
                status="error",
                document_url=task.document_url,
                file_path=task.file_path,
                processing_time=processing_time,
                error=str(e),
                worker_id=worker_id
            )
    
    async def _download_document(self, task: ProcessingTask) -> Dict[str, Any]:
        """Download document for processing"""
        try:
            async with self.document_downloader as downloader:
                result = await downloader.download_with_retry(task.document_url, max_retries=2)
                return result
        except Exception as e:
            return {
                "status": "error",
                "error": f"Download error: {str(e)}"
            }
    
    async def _extract_content(self, task: ProcessingTask) -> Dict[str, Any]:
        """Extract content based on file type"""
        try:
            file_type = task.file_type.lower()
            
            if file_type == 'pdf':
                return await self.pdf_processor.process_pdf(
                    task.file_path, 
                    extract_tables=True, 
                    extract_images=True
                )
            elif file_type in ['docx', 'doc', 'xlsx', 'xls', 'csv']:
                return await self.office_processor.process_document(task.file_path, file_type)
            elif file_type in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp']:
                return await self.text_extractor.extract_text(task.file_path, file_type)
            else:
                return await self.text_extractor.extract_text(task.file_path, file_type)
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Content extraction error: {str(e)}"
            }
    
    async def _detect_language(self, text: str) -> Dict[str, Any]:
        """Detect language of extracted text"""
        try:
            if not text or len(text.strip()) < 10:
                return {
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "message": "Insufficient text for language detection"
                }
            
            # Use first 1000 characters for language detection
            sample_text = text[:1000]
            result = self.language_detector.detect_language(sample_text)
            return result
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return {
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    async def _cache_result(self, task: ProcessingTask, content: Dict[str, Any]):
        """Cache processing result"""
        try:
            cache_key = f"doc_processed:{hashlib.md5(task.document_url.encode()).hexdigest()}"
            cache_data = {
                "task_id": task.task_id,
                "document_url": task.document_url,
                "content": content,
                "processed_at": time.time(),
                "file_type": task.file_type
            }
            
            await self.redis_manager.set_json(
                cache_key, 
                cache_data, 
                ex=config.cache_ttl_hours * 3600
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")
    
    def _aggregate_results(self, successful_results: List[ProcessingResult]) -> Dict[str, Any]:
        """Aggregate content from all successful results"""
        try:
            all_text_parts = []
            all_tables = []
            total_pages = 0
            languages_detected = {}
            file_types = {}
            
            for result in successful_results:
                if result.content and "full_text" in result.content:
                    # Add document separator
                    doc_header = f"\n=== DOCUMENT: {result.document_url} ===\n"
                    all_text_parts.append(doc_header + result.content["full_text"])
                    
                    # Aggregate tables
                    if "tables" in result.content:
                        all_tables.extend(result.content["tables"])
                    
                    # Count pages
                    if "page_count" in result.content:
                        total_pages += result.content["page_count"]
                    
                    # Track languages
                    if "language_detection" in result.content:
                        lang = result.content["language_detection"].get("detected_language", "unknown")
                        languages_detected[lang] = languages_detected.get(lang, 0) + 1
                    
                    # Track file types
                    file_ext = Path(result.file_path).suffix.lower().lstrip('.')
                    file_types[file_ext] = file_types.get(file_ext, 0) + 1
            
            combined_text = "\n\n".join(all_text_parts)
            
            return {
                "combined_full_text": combined_text,
                "total_character_count": len(combined_text),
                "total_word_count": len(combined_text.split()),
                "total_pages_processed": total_pages,
                "total_tables_extracted": len(all_tables),
                "languages_detected": languages_detected,
                "file_types_processed": file_types,
                "document_count": len(successful_results),
                "aggregation_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {str(e)}")
            return {
                "aggregation_error": str(e),
                "document_count": len(successful_results)
            }
    
    def _result_to_dict(self, result: ProcessingResult) -> Dict[str, Any]:
        """Convert ProcessingResult to dictionary"""
        return {
            "task_id": result.task_id,
            "status": result.status,
            "document_url": result.document_url,
            "file_path": result.file_path,
            "processing_time": result.processing_time,
            "worker_id": result.worker_id,
            "error": result.error,
            "has_content": result.content is not None,
            "content_summary": {
                "char_count": len(result.content.get("full_text", "")) if result.content else 0,
                "word_count": len(result.content.get("full_text", "").split()) if result.content else 0,
                "language": result.content.get("language_detection", {}).get("detected_language", "unknown") if result.content else "unknown"
            } if result.content else None
        }
    
    def _get_worker_performance(self) -> Dict[str, Any]:
        """Get worker performance statistics"""
        performance = {}
        
        for worker_id, stats in self.worker_stats.items():
            avg_time = (
                stats["total_processing_time"] / stats["tasks_processed"]
                if stats["tasks_processed"] > 0 else 0.0
            )
            
            performance[f"worker_{worker_id}"] = {
                "tasks_processed": stats["tasks_processed"],
                "total_processing_time": round(stats["total_processing_time"], 2),
                "average_processing_time": round(avg_time, 2),
                "current_status": stats["status"],
                "error_count": stats["errors"],
                "efficiency": round(
                    (stats["tasks_processed"] / (stats["tasks_processed"] + stats["errors"])) * 100
                    if (stats["tasks_processed"] + stats["errors"]) > 0 else 0.0, 1
                )
            }
        
        return performance
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get overall system performance metrics"""
        success_rate = (
            (self.completed_tasks / self.total_tasks) * 100
            if self.total_tasks > 0 else 0.0
        )
        
        avg_processing_time = (
            self.total_processing_time / self.completed_tasks
            if self.completed_tasks > 0 else 0.0
        )
        
        return {
            "total_tasks_processed": self.total_tasks,
            "successful_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate_percent": round(success_rate, 1),
            "total_processing_time": round(self.total_processing_time, 2),
            "average_processing_time": round(avg_processing_time, 2),
            "peak_concurrent_tasks": self.peak_concurrent_tasks,
            "configured_workers": self.max_workers,
            "configured_max_concurrent": self.max_concurrent_operations,
            "cache_enabled": self.enable_caching and self.redis_manager.is_connected
        }