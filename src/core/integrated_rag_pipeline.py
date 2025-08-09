"""
Integrated RAG Pipeline
Comprehensive pipeline integrating all components for end-to-end document processing and retrieval
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Union

from src.core.config import config
from src.core.gpu_service import GPUService
from src.core.hierarchical_chunker import HierarchicalChunker
from src.core.parallel_document_processor import ParallelDocumentProcessor
from src.core.parallel_vector_store import ParallelVectorStore, VectorDocument
from src.services.embedding_service import EmbeddingService
from src.services.redis_cache import redis_manager

logger = logging.getLogger(__name__)


@dataclass
class RAGQuery:
    """Data class for RAG queries"""
    query_text: str
    query_id: Union[str, None] = None
    max_results: int = 10
    filter_metadata: dict[str, Any] | None = None
    include_embeddings: bool = False
    retrieval_strategy: str = "similarity"  # similarity, hybrid, rerank


@dataclass
class RAGResult:
    """Data class for RAG results"""
    query_id: str
    query_text: str
    retrieved_chunks: list[dict[str, Any]]
    total_results: int
    retrieval_time: float
    pipeline_metadata: dict[str, Any]


@dataclass
class DocumentIngestionResult:
    """Data class for document ingestion results"""
    status: str
    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    processing_time: float
    pipeline_metadata: dict[str, Any]
    errors: list[str] = None


class IntegratedRAGPipeline:
    """
    Comprehensive RAG pipeline integrating all processing components
    Handles document ingestion, chunking, embedding, storage, and retrieval
    """

    def __init__(self):
        # Core services
        self.gpu_service = GPUService()
        self.embedding_service = EmbeddingService(self.gpu_service)
        self.parallel_processor = ParallelDocumentProcessor()
        self.hierarchical_chunker = HierarchicalChunker()
        self.vector_store = ParallelVectorStore(self.embedding_service, self.gpu_service)

        # Redis cache manager
        self.redis_manager = redis_manager

        # Pipeline state
        self.is_initialized = False
        self.initialization_time = 0.0

        # Performance tracking
        self.total_documents_ingested = 0
        self.total_chunks_created = 0
        self.total_embeddings_generated = 0
        self.total_queries_processed = 0
        self.total_ingestion_time = 0.0
        self.total_query_time = 0.0

        # Configuration from central config
        self.enable_cache = config.enable_embedding_cache
        self.batch_size = config.batch_size
        self.chunking_strategy = getattr(config, 'chunking_strategy', 'hierarchical')

        logger.info("IntegratedRAGPipeline created with all components")

    async def initialize(self) -> dict[str, Any]:
        """Initialize the complete RAG pipeline"""
        try:
            logger.info("🚀 Initializing Integrated RAG Pipeline...")
            start_time = time.time()

            initialization_results = {}

            # Initialize GPU service first
            logger.info("🔄 Initializing GPU Service...")
            gpu_result = self.gpu_service.initialize()
            initialization_results["gpu_service"] = gpu_result

            # Initialize embedding service
            logger.info("🔄 Initializing Embedding Service...")
            embedding_result = await self.embedding_service.initialize()
            initialization_results["embedding_service"] = embedding_result

            if embedding_result["status"] != "success":
                return {
                    "status": "error",
                    "error": "Embedding service initialization failed",
                    "details": embedding_result
                }

            # Initialize parallel document processor
            logger.info("🔄 Initializing Parallel Document Processor...")
            processor_result = await self.parallel_processor.initialize()
            initialization_results["document_processor"] = processor_result

            if processor_result["status"] != "success":
                return {
                    "status": "error",
                    "error": "Document processor initialization failed",
                    "details": processor_result
                }

            # Initialize vector store
            logger.info("🔄 Initializing Vector Store...")
            vector_result = await self.vector_store.initialize()
            initialization_results["vector_store"] = vector_result

            if vector_result["status"] != "success":
                return {
                    "status": "error",
                    "error": "Vector store initialization failed",
                    "details": vector_result
                }

            # Initialize cache if enabled
            if self.enable_cache:
                logger.info("🔄 Initializing Cache...")
                if not self.redis_manager.is_connected:
                    cache_result = await self.redis_manager.initialize()
                    initialization_results["cache"] = cache_result

            self.is_initialized = True
            self.initialization_time = time.time() - start_time

            # Create comprehensive result
            result = {
                "status": "success",
                "message": f"RAG Pipeline initialized successfully in {self.initialization_time:.2f}s",
                "initialization_time": round(self.initialization_time, 2),
                "components_initialized": {
                    "gpu_service": gpu_result.get("status", "unknown"),
                    "embedding_service": embedding_result["status"],
                    "document_processor": processor_result["status"],
                    "vector_store": vector_result["status"],
                    "cache": "enabled" if self.enable_cache else "disabled"
                },
                "configuration": {
                    "embedding_model": config.embedding_model,
                    "vector_db_type": config.vector_db_type,
                    "chunk_size": config.chunk_size,
                    "batch_size": self.batch_size,
                    "gpu_provider": config.gpu_provider,
                    "max_workers": config.max_workers
                },
                "capabilities": {
                    "document_formats": config.supported_formats,
                    "gpu_acceleration": gpu_result.get("gpu_available", False),
                    "multilingual_support": True,
                    "parallel_processing": True,
                    "caching": self.enable_cache,
                    "batch_operations": True
                },
                "initialization_details": initialization_results
            }

            logger.info(f"✅ RAG Pipeline initialized successfully in {self.initialization_time:.2f}s")
            logger.info(f"📊 GPU: {gpu_result.get('gpu_available', False)}, "
                       f"Model: {config.embedding_model}, "
                       f"Cache: {'enabled' if self.enable_cache else 'disabled'}")

            return result

        except Exception as e:
            error_msg = f"RAG Pipeline initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "initialization_time": time.time() - start_time if 'start_time' in locals() else 0
            }

    async def ingest_documents(
        self,
        document_urls: list[str],
        progress_callback: Union[callable, None] = None,
        chunking_strategy: Union[str, None] = None
    ) -> DocumentIngestionResult:
        """
        Complete document ingestion pipeline: download -> process -> chunk -> embed -> store
        
        Args:
            document_urls: List of document URLs to ingest
            progress_callback: Optional progress callback function
            chunking_strategy: Optional chunking strategy override
            
        Returns:
            Comprehensive ingestion results
        """
        if not self.is_initialized:
            return DocumentIngestionResult(
                status="error",
                documents_processed=0,
                chunks_created=0,
                embeddings_generated=0,
                processing_time=0.0,
                pipeline_metadata={},
                errors=["Pipeline not initialized"]
            )

        logger.info(f"📥 Starting document ingestion pipeline for {len(document_urls)} documents")
        start_time = time.time()
        errors = []

        try:
            # Step 1: Document Processing (Download + Extract Content)
            logger.info("📄 Step 1: Processing documents...")

            async def processing_progress(progress, completed, total):
                if progress_callback:
                    await progress_callback(f"Processing documents: {completed}/{total}", progress * 0.4)

            processing_result = await self.parallel_processor.process_documents(
                document_urls,
                processing_progress
            )

            if processing_result["status"] != "success":
                return DocumentIngestionResult(
                    status="error",
                    documents_processed=0,
                    chunks_created=0,
                    embeddings_generated=0,
                    processing_time=time.time() - start_time,
                    pipeline_metadata=processing_result,
                    errors=[f"Document processing failed: {processing_result.get('error', 'Unknown error')}"]
                )

            successful_results = [r for r in processing_result["detailed_results"] if r["status"] == "success"]

            if not successful_results:
                return DocumentIngestionResult(
                    status="error",
                    documents_processed=0,
                    chunks_created=0,
                    embeddings_generated=0,
                    processing_time=time.time() - start_time,
                    pipeline_metadata=processing_result,
                    errors=["No documents processed successfully"]
                )

            # Step 2: Document Chunking
            logger.info("🔪 Step 2: Chunking documents...")

            all_chunks = []
            chunking_results = []

            for i, doc_result in enumerate(successful_results):
                if progress_callback:
                    progress = 40 + (i / len(successful_results)) * 30
                    await progress_callback(f"Chunking document {i+1}/{len(successful_results)}", progress)

                if not doc_result.get("has_content") or not doc_result.get("content_summary"):
                    continue

                # Get document text
                # In a real scenario, you'd extract from doc_result based on your document processor structure
                document_text = doc_result.get("aggregated_content", {}).get("combined_full_text", "")

                if not document_text:
                    errors.append(f"No text content for document: {doc_result['document_url']}")
                    continue

                # Create source info for chunking
                source_info = {
                    "document_url": doc_result["document_url"],
                    "file_path": doc_result["file_path"],
                    "processing_time": doc_result["processing_time"],
                    "worker_id": doc_result.get("worker_id"),
                    "content_summary": doc_result.get("content_summary")
                }

                # Chunk the document
                chunk_result = await self.hierarchical_chunker.chunk_document(
                    document_text,
                    source_info,
                    chunking_strategy or self.chunking_strategy
                )

                if chunk_result.chunks:
                    all_chunks.extend(chunk_result.chunks)
                    chunking_results.append(chunk_result)
                else:
                    errors.append(f"No chunks created for document: {doc_result['document_url']}")

            if not all_chunks:
                return DocumentIngestionResult(
                    status="error",
                    documents_processed=len(successful_results),
                    chunks_created=0,
                    embeddings_generated=0,
                    processing_time=time.time() - start_time,
                    pipeline_metadata={
                        "document_processing": processing_result,
                        "chunking_results": chunking_results
                    },
                    errors=errors + ["No chunks created from any documents"]
                )

            logger.info(f"✅ Created {len(all_chunks)} chunks from {len(successful_results)} documents")

            # Step 3: Generate Embeddings
            logger.info("🔢 Step 3: Generating embeddings...")

            if progress_callback:
                await progress_callback("Generating embeddings for chunks", 70)

            chunk_texts = [chunk.text for chunk in all_chunks]

            embedding_result = await self.embedding_service.encode_texts(
                chunk_texts,
                batch_size=self.batch_size,
                normalize_embeddings=True
            )

            if embedding_result["status"] != "success":
                return DocumentIngestionResult(
                    status="error",
                    documents_processed=len(successful_results),
                    chunks_created=len(all_chunks),
                    embeddings_generated=0,
                    processing_time=time.time() - start_time,
                    pipeline_metadata={
                        "document_processing": processing_result,
                        "chunking_results": chunking_results,
                        "embedding_error": embedding_result
                    },
                    errors=errors + [f"Embedding generation failed: {embedding_result.get('error')}"]
                )

            embeddings = embedding_result["embeddings"]
            logger.info(f"✅ Generated {len(embeddings)} embeddings")

            # Step 4: Create Vector Documents
            logger.info("📦 Step 4: Creating vector documents...")

            vector_documents = []
            for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings, strict=False)):
                vector_doc = VectorDocument(
                    doc_id=chunk.chunk_id,
                    embedding=embedding,
                    metadata={
                        "chunk_type": chunk.chunk_type,
                        "hierarchy_level": chunk.hierarchy_level,
                        "token_count": chunk.token_count,
                        "char_count": chunk.char_count,
                        "source_url": chunk.source_info.get("document_url"),
                        "file_path": chunk.source_info.get("file_path"),
                        "created_at": chunk.metadata.get("created_at") if chunk.metadata else time.time(),
                        "language": chunk.metadata.get("language") if chunk.metadata else "unknown",
                        "chunking_method": chunk.metadata.get("chunking_method") if chunk.metadata else "hierarchical"
                    },
                    text_content=chunk.text,
                    chunk_id=chunk.chunk_id,
                    source_info=chunk.source_info
                )
                vector_documents.append(vector_doc)

            # Step 5: Store in Vector Database
            logger.info("💾 Step 5: Storing in vector database...")

            if progress_callback:
                await progress_callback("Storing vectors in database", 90)

            storage_result = await self.vector_store.add_documents(
                vector_documents,
                batch_size=self.batch_size
            )

            if storage_result["status"] != "success":
                return DocumentIngestionResult(
                    status="error",
                    documents_processed=len(successful_results),
                    chunks_created=len(all_chunks),
                    embeddings_generated=len(embeddings),
                    processing_time=time.time() - start_time,
                    pipeline_metadata={
                        "document_processing": processing_result,
                        "chunking_results": chunking_results,
                        "embedding_result": embedding_result,
                        "storage_error": storage_result
                    },
                    errors=errors + [f"Vector storage failed: {storage_result.get('error')}"]
                )

            # Update pipeline statistics
            total_time = time.time() - start_time
            self.total_documents_ingested += len(successful_results)
            self.total_chunks_created += len(all_chunks)
            self.total_embeddings_generated += len(embeddings)
            self.total_ingestion_time += total_time

            if progress_callback:
                await progress_callback("Ingestion completed successfully", 100)

            # Create comprehensive result
            result = DocumentIngestionResult(
                status="success",
                documents_processed=len(successful_results),
                chunks_created=len(all_chunks),
                embeddings_generated=len(embeddings),
                processing_time=round(total_time, 2),
                pipeline_metadata={
                    "document_processing": {
                        "total_documents": len(document_urls),
                        "successful_documents": len(successful_results),
                        "processing_time": processing_result["processing_summary"]["total_processing_time"],
                        "success_rate": processing_result["processing_summary"]["success_rate"]
                    },
                    "chunking": {
                        "total_chunks": len(all_chunks),
                        "chunking_strategy": chunking_strategy or self.chunking_strategy,
                        "average_chunk_size": sum(c.token_count for c in all_chunks) / len(all_chunks) if all_chunks else 0
                    },
                    "embedding": {
                        "model_name": embedding_result["model_name"],
                        "dimension": embedding_result["dimension"],
                        "processing_time": embedding_result["processing_time"],
                        "cache_hits": embedding_result.get("cache_hits", 0)
                    },
                    "vector_storage": {
                        "documents_stored": storage_result["documents_added"],
                        "total_documents_in_store": storage_result["total_documents"],
                        "index_size": storage_result["index_size"]
                    }
                },
                errors=errors if errors else None
            )

            logger.info("✅ Document ingestion completed successfully!")
            logger.info(f"📊 Processed: {len(successful_results)} docs, "
                       f"Created: {len(all_chunks)} chunks, "
                       f"Generated: {len(embeddings)} embeddings in {total_time:.2f}s")

            return result

        except Exception as e:
            error_msg = f"Document ingestion pipeline failed: {str(e)}"
            logger.error(f"❌ {error_msg}")

            return DocumentIngestionResult(
                status="error",
                documents_processed=0,
                chunks_created=0,
                embeddings_generated=0,
                processing_time=time.time() - start_time,
                pipeline_metadata={},
                errors=[error_msg]
            )

    async def query(self, query: RAGQuery) -> RAGResult:
        """
        Execute RAG query pipeline: embed query -> search -> retrieve -> format results
        
        Args:
            query: RAGQuery object with query parameters
            
        Returns:
            RAGResult with retrieved chunks and metadata
        """
        if not self.is_initialized:
            return RAGResult(
                query_id=query.query_id or "unknown",
                query_text=query.query_text,
                retrieved_chunks=[],
                total_results=0,
                retrieval_time=0.0,
                pipeline_metadata={"error": "Pipeline not initialized"}
            )

        logger.info(f"🔍 Processing RAG query: '{query.query_text[:100]}...'")
        start_time = time.time()

        try:
            # Search vector store
            search_result = await self.vector_store.search(
                query_text=query.query_text,
                k=query.max_results,
                filter_metadata=query.filter_metadata,
                return_embeddings=query.include_embeddings
            )

            if search_result["status"] != "success":
                return RAGResult(
                    query_id=query.query_id or f"query_{int(time.time())}",
                    query_text=query.query_text,
                    retrieved_chunks=[],
                    total_results=0,
                    retrieval_time=time.time() - start_time,
                    pipeline_metadata={"search_error": search_result}
                )

            # Format retrieved chunks
            retrieved_chunks = []
            for result in search_result["results"]:
                chunk_data = {
                    "doc_id": result.doc_id,
                    "chunk_id": result.chunk_id,
                    "text": result.text_content,
                    "score": result.score,
                    "metadata": result.metadata,
                    "source_url": result.metadata.get("source_url"),
                    "chunk_type": result.metadata.get("chunk_type"),
                    "hierarchy_level": result.metadata.get("hierarchy_level"),
                    "token_count": result.metadata.get("token_count"),
                    "language": result.metadata.get("language")
                }

                if query.include_embeddings and result.embedding is not None:
                    chunk_data["embedding"] = result.embedding.tolist()

                retrieved_chunks.append(chunk_data)

            retrieval_time = time.time() - start_time
            self.total_queries_processed += 1
            self.total_query_time += retrieval_time

            result = RAGResult(
                query_id=query.query_id or f"query_{int(time.time())}",
                query_text=query.query_text,
                retrieved_chunks=retrieved_chunks,
                total_results=len(retrieved_chunks),
                retrieval_time=round(retrieval_time, 3),
                pipeline_metadata={
                    "search_time": search_result["search_time"],
                    "index_size": search_result["index_size"],
                    "retrieval_strategy": query.retrieval_strategy,
                    "max_results_requested": query.max_results,
                    "filters_applied": query.filter_metadata is not None
                }
            )

            logger.info(f"✅ Retrieved {len(retrieved_chunks)} chunks in {retrieval_time:.3f}s")
            return result

        except Exception as e:
            error_msg = f"RAG query failed: {str(e)}"
            logger.error(f"❌ {error_msg}")

            return RAGResult(
                query_id=query.query_id or "error",
                query_text=query.query_text,
                retrieved_chunks=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                pipeline_metadata={"error": error_msg}
            )

    def get_pipeline_stats(self) -> dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        avg_ingestion_time = (
            self.total_ingestion_time / self.total_documents_ingested
            if self.total_documents_ingested > 0 else 0.0
        )

        avg_query_time = (
            self.total_query_time / self.total_queries_processed
            if self.total_queries_processed > 0 else 0.0
        )

        return {
            "pipeline_status": "initialized" if self.is_initialized else "not_initialized",
            "initialization_time": round(self.initialization_time, 2),
            "performance_metrics": {
                "total_documents_ingested": self.total_documents_ingested,
                "total_chunks_created": self.total_chunks_created,
                "total_embeddings_generated": self.total_embeddings_generated,
                "total_queries_processed": self.total_queries_processed,
                "total_ingestion_time": round(self.total_ingestion_time, 2),
                "total_query_time": round(self.total_query_time, 3),
                "average_ingestion_time": round(avg_ingestion_time, 2),
                "average_query_time": round(avg_query_time, 4)
            },
            "component_stats": {
                "gpu_service": self.gpu_service.get_device_info(),
                "embedding_service": self.embedding_service.get_embedding_stats(),
                "document_processor": self.parallel_processor._get_system_metrics(),
                "chunker": self.hierarchical_chunker.get_chunking_stats(),
                "vector_store": self.vector_store.get_store_stats()
            },
            "configuration_summary": {
                "embedding_model": config.embedding_model,
                "vector_db_type": config.vector_db_type,
                "chunk_size": config.chunk_size,
                "batch_size": self.batch_size,
                "max_workers": config.max_workers,
                "gpu_provider": config.gpu_provider
            }
        }

    async def clear_pipeline(self) -> dict[str, Any]:
        """Clear all pipeline data and reset state"""
        try:
            logger.info("🧹 Clearing RAG pipeline...")

            # Clear vector store
            clear_result = await self.vector_store.clear_store()

            # Clear caches if enabled
            cache_results = {}
            if self.enable_cache:
                embedding_cache = await self.embedding_service.clear_embedding_cache()
                cache_results["embedding_cache"] = embedding_cache

            # Reset statistics
            self.total_documents_ingested = 0
            self.total_chunks_created = 0
            self.total_embeddings_generated = 0
            self.total_queries_processed = 0
            self.total_ingestion_time = 0.0
            self.total_query_time = 0.0

            return {
                "status": "success",
                "message": "Pipeline cleared successfully",
                "vector_store": clear_result,
                "cache": cache_results
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"Pipeline clearing failed: {str(e)}"
            }
