"""
Parallel FAISS Vector Store with Batch Operations
High-performance vector storage with GPU acceleration and batch processing
"""

import logging
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import time
import pickle
import hashlib
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from src.core.config import config
from src.core.gpu_service import GPUService
from src.services.embedding_service import EmbeddingService
from src.services.redis_cache import redis_manager

logger = logging.getLogger(__name__)


@dataclass
class VectorDocument:
    """Data class for vector documents"""
    doc_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    text_content: str
    chunk_id: Optional[str] = None
    source_info: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Data class for search results"""
    doc_id: str
    score: float
    metadata: Dict[str, Any]
    text_content: str
    chunk_id: Optional[str] = None
    embedding: Optional[np.ndarray] = None


class ParallelVectorStore:
    """
    High-performance FAISS vector store with parallel batch operations
    Supports GPU acceleration, batch indexing, and efficient similarity search
    """
    
    def __init__(self, embedding_service: Optional[EmbeddingService] = None, gpu_service: Optional[GPUService] = None):
        # Configuration from central config
        self.embedding_dimension = config.embedding_dimension
        self.index_type = config.vector_db_type.value  # faiss_hnsw, faiss_ivf, etc.
        self.batch_size = config.batch_size
        self.max_batch_size = config.max_batch_size
        self.nprobe = getattr(config, 'faiss_nprobe', 32)  # Search parameter
        self.ef_search = getattr(config, 'faiss_ef_search', 128)  # HNSW search parameter
        self.m_hnsw = getattr(config, 'faiss_m_hnsw', 32)  # HNSW construction parameter
        self.enable_cache = config.enable_embedding_cache
        
        # Services
        self.embedding_service = embedding_service
        self.gpu_service = gpu_service or GPUService()
        self.redis_manager = redis_manager
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.gpu_index: Optional[faiss.Index] = None
        self.index_initialized = False
        self.use_gpu = False
        
        # Document storage
        self.documents: Dict[str, VectorDocument] = {}
        self.doc_id_to_index: Dict[str, int] = {}  # Maps doc_id to FAISS index
        self.index_to_doc_id: Dict[int, str] = {}  # Maps FAISS index to doc_id
        self.next_index = 0
        
        # Performance tracking
        self.total_documents = 0
        self.total_search_queries = 0
        self.total_indexing_time = 0.0
        self.total_search_time = 0.0
        self.batch_operations = 0
        
        # Storage paths
        self.storage_dir = Path(config.storage_dir if hasattr(config, 'storage_dir') else './vector_storage')
        self.storage_dir.mkdir(exist_ok=True)
        
        logger.info(f"ParallelVectorStore initialized: {self.embedding_dimension}D, index_type={self.index_type}")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the vector store with GPU optimization"""
        try:
            logger.info("🔄 Initializing Parallel Vector Store...")
            start_time = time.time()
            
            # Check FAISS availability
            if not HAS_FAISS:
                return {
                    "status": "error",
                    "error": "FAISS not available. Install with: pip install faiss-gpu faiss-cpu",
                    "message": "Vector store requires FAISS library"
                }
            
            # Initialize GPU service if available
            if self.gpu_service and self.gpu_service.is_gpu_available:
                gpu_info = self.gpu_service.initialize()
                self.use_gpu = faiss.get_num_gpus() > 0
                logger.info(f"🎯 GPU support available: {self.use_gpu} (FAISS GPUs: {faiss.get_num_gpus()})")
            
            # Initialize the FAISS index
            index_result = self._initialize_faiss_index()
            
            if index_result["status"] != "success":
                return index_result
            
            # Initialize cache if enabled
            if self.enable_cache:
                await self._initialize_vector_cache()
            
            # Try to load existing index
            await self._load_existing_index()
            
            self.index_initialized = True
            initialization_time = time.time() - start_time
            
            result = {
                "status": "success",
                "message": f"Vector store initialized in {initialization_time:.2f}s",
                "index_type": self.index_type,
                "embedding_dimension": self.embedding_dimension,
                "gpu_enabled": self.use_gpu,
                "documents_loaded": self.total_documents,
                "cache_enabled": self.enable_cache,
                "initialization_time": initialization_time,
                "faiss_version": faiss.__version__ if hasattr(faiss, '__version__') else "unknown"
            }
            
            logger.info(f"✅ {result['message']}")
            return result
            
        except Exception as e:
            error_msg = f"Vector store initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }
    
    def _initialize_faiss_index(self) -> Dict[str, Any]:
        """Initialize FAISS index based on configuration"""
        try:
            # Create the appropriate index based on type
            if self.index_type == "faiss_hnsw":
                # HNSW index for fast similarity search
                self.index = faiss.IndexHNSWFlat(self.embedding_dimension, self.m_hnsw)
                self.index.hnsw.efConstruction = 200  # Construction parameter
                self.index.hnsw.efSearch = self.ef_search  # Search parameter
                
            elif self.index_type == "faiss_ivf":
                # IVF index with clustering
                nlist = min(4096, max(16, int(np.sqrt(10000))))  # Number of clusters
                quantizer = faiss.IndexFlatL2(self.embedding_dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, nlist)
                
            elif self.index_type == "faiss_flat":
                # Simple flat index for exact search
                self.index = faiss.IndexFlatL2(self.embedding_dimension)
                
            else:
                # Default to HNSW
                self.index = faiss.IndexHNSWFlat(self.embedding_dimension, self.m_hnsw)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = self.ef_search
            
            # Configure for GPU if available
            if self.use_gpu:
                try:
                    # Move index to GPU
                    gpu_resources = faiss.StandardGpuResources()
                    gpu_config = faiss.GpuIndexFlatConfig()
                    gpu_config.device = 0  # Use first GPU
                    
                    if isinstance(self.index, faiss.IndexHNSWFlat):
                        # HNSW doesn't support GPU directly, keep on CPU
                        logger.info("📌 HNSW index kept on CPU (GPU not supported for HNSW)")
                    else:
                        self.gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                        logger.info("🎯 Index moved to GPU")
                        
                except Exception as e:
                    logger.warning(f"GPU index creation failed: {str(e)}, using CPU")
                    self.use_gpu = False
            
            logger.info(f"✅ FAISS {self.index_type} index created: {self.embedding_dimension}D")
            
            return {"status": "success"}
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"FAISS index initialization failed: {str(e)}"
            }
    
    async def _initialize_vector_cache(self):
        """Initialize vector search cache"""
        if not self.redis_manager.is_connected:
            await self.redis_manager.initialize()
    
    async def _load_existing_index(self):
        """Load existing index and documents if available"""
        try:
            index_path = self.storage_dir / "faiss_index.bin"
            docs_path = self.storage_dir / "documents.pkl"
            
            if index_path.exists() and docs_path.exists():
                logger.info("📂 Loading existing index and documents...")
                
                # Load FAISS index
                loaded_index = faiss.read_index(str(index_path))
                
                # Verify dimension compatibility
                if loaded_index.d == self.embedding_dimension:
                    self.index = loaded_index
                    
                    # Move to GPU if needed
                    if self.use_gpu and not isinstance(self.index, faiss.IndexHNSWFlat):
                        gpu_resources = faiss.StandardGpuResources()
                        self.gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                    
                    # Load documents
                    with open(docs_path, 'rb') as f:
                        saved_data = pickle.load(f)
                        self.documents = saved_data.get('documents', {})
                        self.doc_id_to_index = saved_data.get('doc_id_to_index', {})
                        self.index_to_doc_id = saved_data.get('index_to_doc_id', {})
                        self.next_index = saved_data.get('next_index', 0)
                    
                    self.total_documents = len(self.documents)
                    logger.info(f"✅ Loaded {self.total_documents} documents from storage")
                else:
                    logger.warning(f"⚠️ Dimension mismatch: {loaded_index.d} != {self.embedding_dimension}")
                    
        except Exception as e:
            logger.warning(f"Failed to load existing index: {str(e)}")
    
    async def add_documents(self, documents: List[VectorDocument], batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Add documents to the vector store in parallel batches
        
        Args:
            documents: List of VectorDocument objects to add
            batch_size: Optional batch size for processing
            
        Returns:
            Results of the batch indexing operation
        """
        if not self.index_initialized:
            return {
                "status": "error",
                "error": "Vector store not initialized"
            }
        
        if not documents:
            return {
                "status": "success",
                "documents_added": 0,
                "message": "No documents to add"
            }
        
        logger.info(f"📥 Adding {len(documents)} documents to vector store")
        start_time = time.time()
        
        try:
            # Use configured batch size if not provided
            if batch_size is None:
                batch_size = min(self.max_batch_size, len(documents))
            
            added_count = 0
            failed_count = 0
            
            # Process documents in batches
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Process batch
                batch_result = await self._add_document_batch(batch_docs)
                
                added_count += batch_result["added"]
                failed_count += batch_result["failed"]
                
                self.batch_operations += 1
                
                logger.debug(f"📦 Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
            
            # Train index if needed (for IVF indexes)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained and self.index.ntotal > 0:
                await self._train_index()
            
            processing_time = time.time() - start_time
            self.total_indexing_time += processing_time
            
            # Save index periodically
            if added_count > 0:
                await self._save_index()
            
            result = {
                "status": "success",
                "documents_added": added_count,
                "documents_failed": failed_count,
                "total_documents": self.total_documents,
                "processing_time": round(processing_time, 2),
                "batch_count": (len(documents) + batch_size - 1) // batch_size,
                "index_size": self.index.ntotal if self.index else 0
            }
            
            logger.info(f"✅ Added {added_count} documents in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            error_msg = f"Batch document addition failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "documents_processed": added_count if 'added_count' in locals() else 0
            }
    
    async def _add_document_batch(self, batch_docs: List[VectorDocument]) -> Dict[str, Any]:
        """Add a batch of documents to the index"""
        try:
            embeddings = []
            valid_docs = []
            
            for doc in batch_docs:
                if doc.doc_id in self.documents:
                    logger.debug(f"⚠️ Document {doc.doc_id} already exists, skipping")
                    continue
                
                # Validate embedding
                if doc.embedding.shape[0] != self.embedding_dimension:
                    logger.warning(f"⚠️ Invalid embedding dimension for {doc.doc_id}")
                    continue
                
                embeddings.append(doc.embedding)
                valid_docs.append(doc)
            
            if not valid_docs:
                return {"added": 0, "failed": len(batch_docs) - len(valid_docs)}
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize embeddings
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            start_idx = self.next_index
            
            # Use GPU index if available
            active_index = self.gpu_index if self.gpu_index else self.index
            active_index.add(embeddings_array)
            
            # Update mappings and storage
            for i, doc in enumerate(valid_docs):
                doc_index = start_idx + i
                
                # Store document
                self.documents[doc.doc_id] = doc
                self.doc_id_to_index[doc.doc_id] = doc_index
                self.index_to_doc_id[doc_index] = doc.doc_id
            
            self.next_index += len(valid_docs)
            self.total_documents += len(valid_docs)
            
            return {
                "added": len(valid_docs),
                "failed": len(batch_docs) - len(valid_docs)
            }
            
        except Exception as e:
            logger.error(f"Batch addition failed: {str(e)}")
            return {
                "added": 0,
                "failed": len(batch_docs)
            }
    
    async def _train_index(self):
        """Train index for IVF-based indexes"""
        try:
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                logger.info("🎯 Training FAISS index...")
                
                # Get all embeddings for training
                all_embeddings = []
                for doc in self.documents.values():
                    all_embeddings.append(doc.embedding)
                
                if all_embeddings:
                    training_data = np.array(all_embeddings, dtype=np.float32)
                    faiss.normalize_L2(training_data)
                    
                    self.index.train(training_data)
                    
                    # Retrain GPU index if needed
                    if self.gpu_index:
                        gpu_resources = faiss.StandardGpuResources()
                        self.gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, self.index)
                    
                    logger.info("✅ Index training completed")
                    
        except Exception as e:
            logger.warning(f"Index training failed: {str(e)}")
    
    async def search(
        self, 
        query_embedding: Optional[np.ndarray] = None,
        query_text: Optional[str] = None,
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_embeddings: bool = False
    ) -> Dict[str, Any]:
        """
        Search for similar documents using embedding or text query
        
        Args:
            query_embedding: Pre-computed query embedding
            query_text: Text query (will be embedded automatically)
            k: Number of results to return
            filter_metadata: Optional metadata filters
            return_embeddings: Whether to return embeddings in results
            
        Returns:
            Search results with scores and metadata
        """
        if not self.index_initialized:
            return {
                "status": "error",
                "error": "Vector store not initialized"
            }
        
        if not query_embedding and not query_text:
            return {
                "status": "error",
                "error": "Either query_embedding or query_text must be provided"
            }
        
        logger.info(f"🔍 Searching for top-{k} similar documents")
        start_time = time.time()
        
        try:
            # Generate embedding if text query provided
            if query_text and not query_embedding:
                if not self.embedding_service:
                    return {
                        "status": "error",
                        "error": "Embedding service required for text queries"
                    }
                
                embed_result = await self.embedding_service.encode_single(query_text)
                if embed_result["status"] != "success":
                    return embed_result
                
                query_embedding = embed_result["embedding"]
            
            # Validate embedding
            if query_embedding.shape[0] != self.embedding_dimension:
                return {
                    "status": "error",
                    "error": f"Query embedding dimension mismatch: {query_embedding.shape[0]} != {self.embedding_dimension}"
                }
            
            # Prepare query embedding
            query_vector = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query_vector)
            
            # Set search parameters for HNSW
            if isinstance(self.index, faiss.IndexHNSWFlat):
                self.index.hnsw.efSearch = max(k, self.ef_search)
            elif hasattr(self.index, 'nprobe'):
                self.index.nprobe = self.nprobe
            
            # Perform search
            active_index = self.gpu_index if self.gpu_index else self.index
            scores, indices = active_index.search(query_vector, k)
            
            # Process results
            results = await self._process_search_results(
                scores[0], indices[0], filter_metadata, return_embeddings
            )
            
            search_time = time.time() - start_time
            self.total_search_queries += 1
            self.total_search_time += search_time
            
            final_result = {
                "status": "success",
                "results": results,
                "total_results": len(results),
                "search_time": round(search_time, 3),
                "query_type": "embedding" if query_embedding is not None else "text",
                "index_size": self.index.ntotal
            }
            
            logger.info(f"✅ Found {len(results)} results in {search_time:.3f}s")
            return final_result
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }
    
    async def _process_search_results(
        self, 
        scores: np.ndarray, 
        indices: np.ndarray,
        filter_metadata: Optional[Dict[str, Any]],
        return_embeddings: bool
    ) -> List[SearchResult]:
        """Process and filter search results"""
        results = []
        
        for score, idx in zip(scores, indices):
            # Skip invalid indices
            if idx == -1 or idx not in self.index_to_doc_id:
                continue
            
            doc_id = self.index_to_doc_id[idx]
            if doc_id not in self.documents:
                continue
            
            doc = self.documents[doc_id]
            
            # Apply metadata filtering
            if filter_metadata:
                if not self._matches_filter(doc.metadata, filter_metadata):
                    continue
            
            # Create search result
            result = SearchResult(
                doc_id=doc.doc_id,
                score=float(score),
                metadata=doc.metadata,
                text_content=doc.text_content,
                chunk_id=doc.chunk_id,
                embedding=doc.embedding if return_embeddings else None
            )
            
            results.append(result)
        
        return results
    
    def _matches_filter(self, doc_metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if document metadata matches filter criteria"""
        for key, value in filter_metadata.items():
            if key not in doc_metadata:
                return False
            
            if isinstance(value, list):
                if doc_metadata[key] not in value:
                    return False
            else:
                if doc_metadata[key] != value:
                    return False
        
        return True
    
    async def _save_index(self):
        """Save FAISS index and documents to storage"""
        try:
            logger.debug("💾 Saving vector store to disk...")
            
            # Save FAISS index (always save CPU version)
            cpu_index = faiss.index_gpu_to_cpu(self.gpu_index) if self.gpu_index else self.index
            faiss.write_index(cpu_index, str(self.storage_dir / "faiss_index.bin"))
            
            # Save documents and mappings
            save_data = {
                'documents': self.documents,
                'doc_id_to_index': self.doc_id_to_index,
                'index_to_doc_id': self.index_to_doc_id,
                'next_index': self.next_index,
                'total_documents': self.total_documents
            }
            
            with open(self.storage_dir / "documents.pkl", 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.debug("✅ Vector store saved successfully")
            
        except Exception as e:
            logger.warning(f"Failed to save vector store: {str(e)}")
    
    def get_store_stats(self) -> Dict[str, Any]:
        """Get comprehensive vector store statistics"""
        avg_indexing_time = (
            self.total_indexing_time / self.batch_operations
            if self.batch_operations > 0 else 0.0
        )
        
        avg_search_time = (
            self.total_search_time / self.total_search_queries
            if self.total_search_queries > 0 else 0.0
        )
        
        return {
            "store_status": "initialized" if self.index_initialized else "not_initialized",
            "index_type": self.index_type,
            "total_documents": self.total_documents,
            "embedding_dimension": self.embedding_dimension,
            "gpu_enabled": self.use_gpu,
            "index_size": self.index.ntotal if self.index else 0,
            "performance": {
                "total_search_queries": self.total_search_queries,
                "total_indexing_time": round(self.total_indexing_time, 2),
                "total_search_time": round(self.total_search_time, 3),
                "average_indexing_time": round(avg_indexing_time, 3),
                "average_search_time": round(avg_search_time, 4),
                "batch_operations": self.batch_operations
            },
            "configuration": {
                "batch_size": self.batch_size,
                "max_batch_size": self.max_batch_size,
                "nprobe": self.nprobe,
                "ef_search": self.ef_search,
                "m_hnsw": self.m_hnsw
            },
            "storage": {
                "storage_directory": str(self.storage_dir),
                "cache_enabled": self.enable_cache
            },
            "capabilities": {
                "batch_indexing": True,
                "gpu_acceleration": self.use_gpu,
                "metadata_filtering": True,
                "text_queries": self.embedding_service is not None,
                "persistent_storage": True
            }
        }
    
    async def clear_store(self) -> Dict[str, Any]:
        """Clear all documents from the vector store"""
        try:
            logger.info("🧹 Clearing vector store...")
            
            # Reset index
            self.index.reset()
            if self.gpu_index:
                self.gpu_index.reset()
            
            # Clear storage
            self.documents.clear()
            self.doc_id_to_index.clear()
            self.index_to_doc_id.clear()
            self.next_index = 0
            self.total_documents = 0
            
            # Clear saved files
            for file_path in [self.storage_dir / "faiss_index.bin", self.storage_dir / "documents.pkl"]:
                if file_path.exists():
                    file_path.unlink()
            
            return {
                "status": "success",
                "message": "Vector store cleared successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to clear vector store: {str(e)}"
            }