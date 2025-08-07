"""
Document Indexing Pipeline for BajajFinsev Hybrid RAG System
Orchestrates the complete document processing workflow:
URL → Download → Extract → Chunk → Embed → Store → Index
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

# Core components
from src.core.enhanced_document_processor import (
    get_document_processor, ProcessedDocument, EnhancedDocumentProcessor
)
from src.core.hierarchical_chunker import (
    get_hierarchical_chunker, ChunkingResult, DocumentChunk, HierarchicalChunker
)
from src.services.embedding_service import (
    get_embedding_service, EmbeddingResult, BasicEmbeddingService
)
from src.services.vector_storage import (
    get_vector_storage, VectorDocument, VectorStorageService
)
from src.services.redis_cache import get_redis_cache

logger = logging.getLogger(__name__)


class IndexingStatus(Enum):
    """Document indexing status"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORING = "storing"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class IndexingJob:
    """Individual document indexing job"""
    job_id: str
    document_url: str
    document_hash: str
    
    # Status tracking
    status: IndexingStatus = IndexingStatus.PENDING
    progress_percentage: float = 0.0
    current_step: str = ""
    
    # Processing results
    processed_document: Optional[ProcessedDocument] = None
    chunking_result: Optional[ChunkingResult] = None
    embedding_results: List[EmbeddingResult] = field(default_factory=list)
    stored_document_ids: List[str] = field(default_factory=list)
    
    # Timing information
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    total_processing_time: float = 0.0
    
    # Step timings
    download_time: float = 0.0
    extraction_time: float = 0.0
    chunking_time: float = 0.0
    embedding_time: float = 0.0
    storage_time: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    retry_count: int = 0
    
    # Statistics
    chunks_created: int = 0
    embeddings_generated: int = 0
    documents_stored: int = 0


@dataclass
class PipelineResult:
    """Complete pipeline processing result"""
    success: bool
    job: IndexingJob
    
    # Content summary
    document_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    processing_breakdown: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineStats:
    """Pipeline performance statistics"""
    total_jobs: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    cached_jobs: int = 0
    
    # Processing metrics
    total_documents_processed: int = 0
    total_chunks_created: int = 0
    total_embeddings_generated: int = 0
    
    # Performance metrics
    avg_processing_time: float = 0.0
    avg_download_time: float = 0.0
    avg_extraction_time: float = 0.0
    avg_chunking_time: float = 0.0
    avg_embedding_time: float = 0.0
    avg_storage_time: float = 0.0
    
    # Quality metrics
    avg_text_quality: float = 0.0
    avg_chunk_quality: float = 0.0
    
    # Resource usage
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0


class DocumentIndexingPipeline:
    """
    Complete document indexing pipeline
    Orchestrates document processing from URL to searchable vector storage
    """
    
    def __init__(self,
                 max_concurrent_jobs: int = 3,
                 enable_caching: bool = True,
                 cache_ttl: int = 3600,
                 retry_limit: int = 2,
                 quality_threshold: float = 0.3):
        """
        Initialize document indexing pipeline
        
        Args:
            max_concurrent_jobs: Maximum concurrent indexing jobs
            enable_caching: Enable Redis caching for processed documents
            cache_ttl: Cache TTL in seconds
            retry_limit: Maximum retry attempts for failed jobs
            quality_threshold: Minimum quality threshold for chunks
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.retry_limit = retry_limit
        self.quality_threshold = quality_threshold
        
        # Initialize pipeline components
        self.document_processor = get_document_processor()
        self.hierarchical_chunker = get_hierarchical_chunker()
        self.embedding_service = get_embedding_service()
        self.vector_storage = get_vector_storage()
        
        # Caching
        self.redis_cache = get_redis_cache() if enable_caching else None
        
        # Job management
        self.active_jobs: Dict[str, IndexingJob] = {}
        self.job_history: List[IndexingJob] = []
        self.job_semaphore = asyncio.Semaphore(max_concurrent_jobs)
        
        # Statistics
        self.stats = PipelineStats()
        
        # Pipeline configuration
        self.pipeline_config = {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "embedding_batch_size": 16,
            "hierarchical_chunking": True,
            "quality_assessment": True
        }
        
        logger.info("DocumentIndexingPipeline initialized")
        logger.info(f"Max concurrent jobs: {max_concurrent_jobs}")
        logger.info(f"Caching enabled: {enable_caching}")
        logger.info(f"Quality threshold: {quality_threshold}")
    
    async def initialize(self) -> bool:
        """Initialize all pipeline components"""
        logger.info("🔄 Initializing document indexing pipeline...")
        
        try:
            # Initialize components
            await self.embedding_service.initialize()
            await self.vector_storage.initialize()
            
            logger.info("✅ Document indexing pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {str(e)}")
            return False
    
    async def index_document(self, document_url: str, **kwargs) -> PipelineResult:
        """
        Index a single document through the complete pipeline
        
        Args:
            document_url: URL of document to index
            **kwargs: Additional processing options
            
        Returns:
            PipelineResult with indexing status and results
        """
        # Create indexing job
        job = IndexingJob(
            job_id=self._generate_job_id(document_url),
            document_url=document_url,
            document_hash=self._generate_document_hash(document_url)
        )
        
        logger.info(f"📋 Starting indexing job: {job.job_id}")
        logger.info(f"Document URL: {document_url}")
        
        # Check cache first
        if self.enable_caching:
            cached_result = await self._check_document_cache(job.document_hash)
            if cached_result:
                job.status = IndexingStatus.CACHED
                job.completed_at = time.time()
                job.total_processing_time = 0.0
                
                self.stats.cached_jobs += 1
                self.stats.total_jobs += 1
                
                logger.info(f"✅ Using cached result for: {job.job_id}")
                return PipelineResult(success=True, job=job)
        
        # Add to active jobs
        self.active_jobs[job.job_id] = job
        self.stats.total_jobs += 1
        
        try:
            # Process through pipeline with concurrency control
            async with self.job_semaphore:
                result = await self._process_job(job, **kwargs)
            
            # Update job history
            self._add_to_job_history(job)
            
            return result
            
        except Exception as e:
            job.error_message = f"Pipeline error: {str(e)}"
            job.status = IndexingStatus.FAILED
            job.completed_at = time.time()
            job.total_processing_time = time.time() - job.started_at
            
            self.stats.failed_jobs += 1
            logger.error(f"❌ Indexing failed for {job.job_id}: {str(e)}")
            
            return PipelineResult(success=False, job=job)
        
        finally:
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
    
    async def _process_job(self, job: IndexingJob, **kwargs) -> PipelineResult:
        """Process individual indexing job through pipeline stages"""
        
        try:
            # Stage 1: Document Processing (Download + Extract)
            job.status = IndexingStatus.DOWNLOADING
            job.current_step = "Downloading and extracting document"
            job.progress_percentage = 10.0
            
            logger.debug(f"📄 Stage 1: Processing document for {job.job_id}")
            processing_start = time.time()
            
            job.processed_document = await self.document_processor.process_document(
                job.document_url, **kwargs
            )
            
            processing_time = time.time() - processing_start
            job.download_time = job.processed_document.download_time
            job.extraction_time = job.processed_document.extraction_time
            
            if not job.processed_document.success:
                raise Exception(f"Document processing failed: {job.processed_document.error_message}")
            
            job.progress_percentage = 30.0
            logger.debug(f"✅ Document processed: {job.processed_document.word_count:,} words")
            
            # Stage 2: Hierarchical Chunking
            job.status = IndexingStatus.CHUNKING
            job.current_step = "Creating hierarchical chunks"
            job.progress_percentage = 40.0
            
            logger.debug(f"🔪 Stage 2: Chunking document for {job.job_id}")
            chunking_start = time.time()
            
            job.chunking_result = await self.hierarchical_chunker.chunk_document(
                text=job.processed_document.raw_text,
                document_url=job.document_url,
                file_format=job.processed_document.file_format,
                structured_content=job.processed_document.structured_content,
                metadata=job.processed_document.metadata
            )
            
            job.chunking_time = time.time() - chunking_start
            
            if not job.chunking_result.success:
                raise Exception(f"Chunking failed: {job.chunking_result.error_message}")
            
            job.chunks_created = len(job.chunking_result.chunks)
            job.progress_percentage = 50.0
            logger.debug(f"✅ Created {job.chunks_created} chunks")
            
            # Stage 3: Generate Embeddings
            job.status = IndexingStatus.EMBEDDING
            job.current_step = "Generating embeddings"
            job.progress_percentage = 60.0
            
            logger.debug(f"🎯 Stage 3: Generating embeddings for {job.job_id}")
            embedding_start = time.time()
            
            # Extract chunk texts for batch embedding
            chunk_texts = [chunk.content for chunk in job.chunking_result.chunks]
            
            # Generate embeddings in batch
            embedding_result = await self.embedding_service.generate_embeddings(chunk_texts)
            
            job.embedding_time = time.time() - embedding_start
            
            if not embedding_result.success:
                raise Exception(f"Embedding generation failed: {embedding_result.error_message}")
            
            job.embedding_results.append(embedding_result)
            job.embeddings_generated = embedding_result.text_count
            job.progress_percentage = 80.0
            logger.debug(f"✅ Generated {job.embeddings_generated} embeddings")
            
            # Stage 4: Store in Vector Database
            job.status = IndexingStatus.STORING
            job.current_step = "Storing in vector database"
            job.progress_percentage = 90.0
            
            logger.debug(f"🗄️ Stage 4: Storing vectors for {job.job_id}")
            storage_start = time.time()
            
            # Create VectorDocument objects for storage
            vector_documents = []
            for i, (chunk, embedding_vec) in enumerate(zip(
                job.chunking_result.chunks, 
                embedding_result.embeddings
            )):
                vector_doc = VectorDocument(
                    id=f"{job.document_hash}_{i}",
                    content=chunk.content,
                    embedding=embedding_vec,
                    metadata={
                        **chunk.metadata.__dict__,
                        "document_url": job.document_url,
                        "document_hash": job.document_hash,
                        "file_format": job.processed_document.file_format,
                        "chunk_quality": chunk.content_quality,
                        "text_quality": job.processed_document.text_quality_score
                    },
                    document_url=job.document_url,
                    document_hash=job.document_hash,
                    chunk_index=i
                )
                vector_documents.append(vector_doc)
            
            # Batch store documents
            storage_results = await self.vector_storage.batch_add_documents(vector_documents)
            
            job.storage_time = time.time() - storage_start
            
            # Track successful storage
            job.stored_document_ids = [
                doc_id for doc_id, success in storage_results.items() if success
            ]
            job.documents_stored = len(job.stored_document_ids)
            
            if job.documents_stored == 0:
                raise Exception("No documents were successfully stored in vector database")
            
            job.progress_percentage = 100.0
            logger.debug(f"✅ Stored {job.documents_stored} vector documents")
            
            # Finalize job
            job.status = IndexingStatus.COMPLETED
            job.current_step = "Completed"
            job.completed_at = time.time()
            job.total_processing_time = job.completed_at - job.started_at
            
            # Update statistics
            self._update_pipeline_stats(job)
            
            # Cache result
            if self.enable_caching:
                await self._cache_document_result(job)
            
            # Create result summary
            result = PipelineResult(
                success=True,
                job=job,
                document_summary=self._create_document_summary(job),
                processing_breakdown=self._create_processing_breakdown(job),
                quality_metrics=self._create_quality_metrics(job)
            )
            
            logger.info(f"✅ Indexing completed for {job.job_id} in {job.total_processing_time:.2f}s")
            logger.info(f"   Chunks: {job.chunks_created}, Embeddings: {job.embeddings_generated}, Stored: {job.documents_stored}")
            
            return result
            
        except Exception as e:
            # Handle job failure with retry logic
            job.error_message = str(e)
            job.retry_count += 1
            
            if job.retry_count <= self.retry_limit:
                logger.warning(f"⚠️ Job {job.job_id} failed, retrying ({job.retry_count}/{self.retry_limit}): {str(e)}")
                # Reset job status for retry
                job.status = IndexingStatus.PENDING
                job.progress_percentage = 0.0
                return await self._process_job(job, **kwargs)
            else:
                job.status = IndexingStatus.FAILED
                job.completed_at = time.time()
                job.total_processing_time = job.completed_at - job.started_at
                self.stats.failed_jobs += 1
                raise e
    
    async def batch_index_documents(self, 
                                  document_urls: List[str], 
                                  **kwargs) -> List[PipelineResult]:
        """
        Index multiple documents concurrently
        
        Args:
            document_urls: List of document URLs to index
            **kwargs: Additional processing options
            
        Returns:
            List of PipelineResult objects
        """
        logger.info(f"📦 Starting batch indexing: {len(document_urls)} documents")
        
        start_time = time.time()
        
        # Create indexing tasks
        tasks = [self.index_document(url, **kwargs) for url in document_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Batch indexing failed for {document_urls[i]}: {result}")
                # Create failed result
                failed_job = IndexingJob(
                    job_id=self._generate_job_id(document_urls[i]),
                    document_url=document_urls[i],
                    document_hash=self._generate_document_hash(document_urls[i]),
                    status=IndexingStatus.FAILED,
                    error_message=str(result)
                )
                final_results.append(PipelineResult(success=False, job=failed_job))
            else:
                final_results.append(result)
        
        # Log batch results
        successful = sum(1 for r in final_results if r.success)
        total_time = time.time() - start_time
        
        logger.info(f"📦 Batch indexing completed: {successful}/{len(document_urls)} successful in {total_time:.2f}s")
        
        return final_results
    
    async def get_job_status(self, job_id: str) -> Optional[IndexingJob]:
        """Get current status of indexing job"""
        return self.active_jobs.get(job_id)
    
    async def list_active_jobs(self) -> List[IndexingJob]:
        """Get list of currently active indexing jobs"""
        return list(self.active_jobs.values())
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an active indexing job"""
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = IndexingStatus.FAILED
            job.error_message = "Job cancelled by user"
            job.completed_at = time.time()
            job.total_processing_time = job.completed_at - job.started_at
            
            del self.active_jobs[job_id]
            logger.info(f"🚫 Cancelled job: {job_id}")
            return True
        
        return False
    
    def _generate_job_id(self, document_url: str) -> str:
        """Generate unique job ID"""
        timestamp = int(time.time() * 1000)
        url_hash = hashlib.md5(document_url.encode()).hexdigest()[:8]
        return f"job_{timestamp}_{url_hash}"
    
    def _generate_document_hash(self, document_url: str) -> str:
        """Generate document hash for caching"""
        return hashlib.md5(document_url.encode()).hexdigest()
    
    async def _check_document_cache(self, document_hash: str) -> Optional[PipelineResult]:
        """Check if document is already cached"""
        if not self.redis_cache:
            return None
        
        try:
            cache_key = f"indexed_document:{document_hash}"
            cached_data = await self.redis_cache.get(cache_key)
            
            if cached_data:
                logger.debug(f"📦 Found cached document: {document_hash}")
                # Would need proper deserialization
                return None  # TODO: Implement proper caching
        
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        
        return None
    
    async def _cache_document_result(self, job: IndexingJob):
        """Cache document processing result"""
        if not self.redis_cache:
            return
        
        try:
            cache_key = f"indexed_document:{job.document_hash}"
            cache_data = {
                "job_id": job.job_id,
                "document_url": job.document_url,
                "status": job.status.value,
                "chunks_created": job.chunks_created,
                "embeddings_generated": job.embeddings_generated,
                "documents_stored": job.documents_stored,
                "cached_at": time.time()
            }
            
            await self.redis_cache.setex(
                cache_key, 
                self.cache_ttl, 
                json.dumps(cache_data)
            )
            
            logger.debug(f"💾 Cached document result: {job.document_hash}")
            
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def _create_document_summary(self, job: IndexingJob) -> Dict[str, Any]:
        """Create document processing summary"""
        return {
            "document_url": job.document_url,
            "file_format": job.processed_document.file_format if job.processed_document else "",
            "file_size": job.processed_document.file_size if job.processed_document else 0,
            "word_count": job.processed_document.word_count if job.processed_document else 0,
            "page_count": job.processed_document.page_count if job.processed_document else 0,
            "chunks_created": job.chunks_created,
            "embeddings_generated": job.embeddings_generated,
            "documents_stored": job.documents_stored,
            "text_quality_score": job.processed_document.text_quality_score if job.processed_document else 0.0
        }
    
    def _create_processing_breakdown(self, job: IndexingJob) -> Dict[str, float]:
        """Create processing time breakdown"""
        return {
            "download_time": job.download_time,
            "extraction_time": job.extraction_time,
            "chunking_time": job.chunking_time,
            "embedding_time": job.embedding_time,
            "storage_time": job.storage_time,
            "total_time": job.total_processing_time
        }
    
    def _create_quality_metrics(self, job: IndexingJob) -> Dict[str, float]:
        """Create quality metrics summary"""
        text_quality = job.processed_document.text_quality_score if job.processed_document else 0.0
        
        # Calculate average chunk quality
        chunk_quality = 0.0
        if job.chunking_result and job.chunking_result.chunks:
            chunk_qualities = [chunk.content_quality for chunk in job.chunking_result.chunks]
            chunk_quality = sum(chunk_qualities) / len(chunk_qualities)
        
        return {
            "text_quality_score": text_quality,
            "avg_chunk_quality": chunk_quality,
            "embedding_dimensions": job.embedding_results[0].embedding_dimensions if job.embedding_results else 0,
            "storage_success_rate": job.documents_stored / job.chunks_created if job.chunks_created > 0 else 0.0
        }
    
    def _update_pipeline_stats(self, job: IndexingJob):
        """Update pipeline statistics"""
        self.stats.successful_jobs += 1
        self.stats.total_documents_processed += 1
        self.stats.total_chunks_created += job.chunks_created
        self.stats.total_embeddings_generated += job.embeddings_generated
        
        # Update averages
        self._update_average('avg_processing_time', job.total_processing_time)
        self._update_average('avg_download_time', job.download_time)
        self._update_average('avg_extraction_time', job.extraction_time)
        self._update_average('avg_chunking_time', job.chunking_time)
        self._update_average('avg_embedding_time', job.embedding_time)
        self._update_average('avg_storage_time', job.storage_time)
        
        if job.processed_document:
            self._update_average('avg_text_quality', job.processed_document.text_quality_score)
        
        # Calculate rates
        total_jobs = self.stats.successful_jobs + self.stats.failed_jobs + self.stats.cached_jobs
        if total_jobs > 0:
            self.stats.cache_hit_rate = self.stats.cached_jobs / total_jobs * 100
            self.stats.error_rate = self.stats.failed_jobs / total_jobs * 100
    
    def _update_average(self, stat_name: str, new_value: float):
        """Update running average for a statistic"""
        current_avg = getattr(self.stats, stat_name)
        successful_count = self.stats.successful_jobs
        
        if successful_count > 0:
            new_avg = ((current_avg * (successful_count - 1)) + new_value) / successful_count
            setattr(self.stats, stat_name, new_avg)
    
    def _add_to_job_history(self, job: IndexingJob):
        """Add completed job to history"""
        self.job_history.append(job)
        
        # Keep only last 100 jobs
        if len(self.job_history) > 100:
            self.job_history = self.job_history[-100:]
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            "pipeline_status": {
                "active_jobs": len(self.active_jobs),
                "max_concurrent": self.max_concurrent_jobs,
                "components_initialized": all([
                    self.embedding_service.is_initialized,
                    self.vector_storage.is_initialized
                ])
            },
            "processing_stats": {
                "total_jobs": self.stats.total_jobs,
                "successful_jobs": self.stats.successful_jobs,
                "failed_jobs": self.stats.failed_jobs,
                "cached_jobs": self.stats.cached_jobs,
                "success_rate": (
                    self.stats.successful_jobs / self.stats.total_jobs * 100
                    if self.stats.total_jobs > 0 else 0
                ),
                "cache_hit_rate": self.stats.cache_hit_rate,
                "error_rate": self.stats.error_rate
            },
            "content_stats": {
                "total_documents_processed": self.stats.total_documents_processed,
                "total_chunks_created": self.stats.total_chunks_created,
                "total_embeddings_generated": self.stats.total_embeddings_generated,
                "avg_chunks_per_document": (
                    self.stats.total_chunks_created / self.stats.total_documents_processed
                    if self.stats.total_documents_processed > 0 else 0
                )
            },
            "performance_stats": {
                "avg_processing_time": self.stats.avg_processing_time,
                "avg_download_time": self.stats.avg_download_time,
                "avg_extraction_time": self.stats.avg_extraction_time,
                "avg_chunking_time": self.stats.avg_chunking_time,
                "avg_embedding_time": self.stats.avg_embedding_time,
                "avg_storage_time": self.stats.avg_storage_time
            },
            "quality_stats": {
                "avg_text_quality": self.stats.avg_text_quality,
                "avg_chunk_quality": self.stats.avg_chunk_quality,
                "quality_threshold": self.quality_threshold
            },
            "configuration": {
                "max_concurrent_jobs": self.max_concurrent_jobs,
                "enable_caching": self.enable_caching,
                "cache_ttl": self.cache_ttl,
                "retry_limit": self.retry_limit,
                "pipeline_config": self.pipeline_config
            }
        }
    
    async def cleanup(self):
        """Cleanup pipeline resources"""
        logger.info("🧹 Cleaning up document indexing pipeline")
        
        try:
            # Cancel active jobs
            for job_id in list(self.active_jobs.keys()):
                await self.cancel_job(job_id)
            
            # Cleanup components
            await self.embedding_service.cleanup()
            await self.vector_storage.cleanup()
            
            # Clear history
            self.job_history.clear()
            
            logger.info("✅ Pipeline cleanup complete")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during pipeline cleanup: {e}")


# Global pipeline instance
document_indexing_pipeline: Optional[DocumentIndexingPipeline] = None


def get_document_indexing_pipeline(**kwargs) -> DocumentIndexingPipeline:
    """Get or create global document indexing pipeline instance"""
    global document_indexing_pipeline
    
    if document_indexing_pipeline is None:
        document_indexing_pipeline = DocumentIndexingPipeline(**kwargs)
    
    return document_indexing_pipeline


async def initialize_document_indexing_pipeline(**kwargs) -> DocumentIndexingPipeline:
    """Initialize and return document indexing pipeline"""
    pipeline = get_document_indexing_pipeline(**kwargs)
    
    # Initialize pipeline
    await pipeline.initialize()
    
    # Log initialization summary
    stats = await pipeline.get_pipeline_stats()
    logger.info("🏭 Document Indexing Pipeline Summary:")
    logger.info(f"  Max concurrent jobs: {stats['pipeline_status']['max_concurrent']}")
    logger.info(f"  Components initialized: {stats['pipeline_status']['components_initialized']}")
    logger.info(f"  Caching enabled: {stats['configuration']['enable_caching']}")
    logger.info(f"  Quality threshold: {stats['configuration']['quality_threshold']}")
    
    return pipeline