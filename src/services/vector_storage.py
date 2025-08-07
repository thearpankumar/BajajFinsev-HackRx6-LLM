"""
Vector Storage Service for BajajFinsev Hybrid RAG System
Manages vector database operations for document embeddings
Supports multiple backends with similarity search and metadata filtering
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle
from pathlib import Path
import tempfile

# Vector database backends
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

# Redis integration for caching
from src.services.redis_cache import get_redis_cache

logger = logging.getLogger(__name__)


class VectorBackend(Enum):
    """Vector storage backend types"""
    FAISS = "faiss"
    CHROMADB = "chromadb"
    IN_MEMORY = "in_memory"


@dataclass
class VectorDocument:
    """Document representation for vector storage"""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    
    # Document context
    document_url: str = ""
    document_hash: str = ""
    chunk_index: int = 0
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class SimilarityResult:
    """Result from similarity search"""
    document: VectorDocument
    score: float
    distance: float


@dataclass
class SearchResult:
    """Complete search result with metadata"""
    query: str
    results: List[SimilarityResult]
    total_results: int
    search_time: float
    backend_used: str
    filter_applied: Optional[Dict[str, Any]] = None


@dataclass
class StorageStats:
    """Vector storage statistics"""
    total_documents: int = 0
    total_embeddings: int = 0
    index_size_mb: float = 0.0
    
    # Performance metrics
    avg_search_time: float = 0.0
    total_searches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Backend specific
    backend_info: Dict[str, Any] = field(default_factory=dict)


class VectorStorageService:
    """
    Vector storage service with multiple backend support
    Handles document embeddings, similarity search, and metadata filtering
    """
    
    def __init__(self,
                 backend: VectorBackend = VectorBackend.FAISS,
                 embedding_dimension: int = 1024,
                 storage_path: Optional[str] = None,
                 enable_caching: bool = True,
                 cache_ttl: int = 3600,
                 similarity_metric: str = "cosine"):
        """
        Initialize vector storage service
        
        Args:
            backend: Vector storage backend to use
            embedding_dimension: Dimension of embeddings
            storage_path: Path for persistent storage
            enable_caching: Enable Redis caching
            cache_ttl: Cache TTL in seconds
            similarity_metric: Similarity metric (cosine, euclidean, dot_product)
        """
        self.backend = backend
        self.embedding_dimension = embedding_dimension
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.similarity_metric = similarity_metric
        
        # Setup storage path
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path(tempfile.gettempdir()) / "vector_storage"
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend
        self.index = None
        self.documents: Dict[str, VectorDocument] = {}
        self.is_initialized = False
        
        # Caching
        self.redis_cache = get_redis_cache() if enable_caching else None
        
        # Statistics
        self.stats = StorageStats()
        
        # Backend-specific configurations
        self.backend_config = self._get_backend_config()
        
        logger.info(f"VectorStorageService initialized")
        logger.info(f"Backend: {backend.value}")
        logger.info(f"Embedding dimension: {embedding_dimension}")
        logger.info(f"Storage path: {self.storage_path}")
        logger.info(f"Similarity metric: {similarity_metric}")
    
    def _get_backend_config(self) -> Dict[str, Any]:
        """Get backend-specific configuration"""
        if self.backend == VectorBackend.FAISS:
            if not FAISS_AVAILABLE:
                logger.warning("⚠️ FAISS not available, falling back to in-memory")
                self.backend = VectorBackend.IN_MEMORY
                return {}
            
            return {
                "index_type": "IVFFlat" if self.embedding_dimension > 100 else "Flat",
                "nlist": 100,  # Number of clusters for IVF
                "nprobe": 10   # Number of clusters to search
            }
            
        elif self.backend == VectorBackend.CHROMADB:
            if not CHROMADB_AVAILABLE:
                logger.warning("⚠️ ChromaDB not available, falling back to in-memory")
                self.backend = VectorBackend.IN_MEMORY
                return {}
            
            return {
                "collection_name": "bajaj_rag_documents",
                "distance_function": "cosine",
                "persist_directory": str(self.storage_path / "chromadb")
            }
        
        else:
            return {"max_documents": 10000}
    
    async def initialize(self) -> bool:
        """Initialize vector storage backend"""
        if self.is_initialized:
            return True
        
        logger.info(f"🔄 Initializing vector storage: {self.backend.value}")
        
        try:
            if self.backend == VectorBackend.FAISS:
                await self._initialize_faiss()
            elif self.backend == VectorBackend.CHROMADB:
                await self._initialize_chromadb()
            else:
                await self._initialize_in_memory()
            
            self.is_initialized = True
            logger.info(f"✅ Vector storage initialized: {self.backend.value}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector storage: {str(e)}")
            
            # Fallback to in-memory
            if self.backend != VectorBackend.IN_MEMORY:
                logger.info("🔄 Falling back to in-memory storage")
                self.backend = VectorBackend.IN_MEMORY
                await self._initialize_in_memory()
                self.is_initialized = True
                return True
            
            return False
    
    async def _initialize_faiss(self):
        """Initialize FAISS backend"""
        config = self.backend_config
        
        if config["index_type"] == "IVFFlat":
            # Use IVF (Inverted File) index for large datasets
            quantizer = faiss.IndexFlatL2(self.embedding_dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dimension, config["nlist"])
            
            # Create dummy data for training if no existing index
            index_path = self.storage_path / "faiss_index.bin"
            if not index_path.exists():
                logger.info("🔄 Training FAISS IVF index...")
                # Generate random training data
                training_data = np.random.random((1000, self.embedding_dimension)).astype('float32')
                self.index.train(training_data)
        else:
            # Use flat index for smaller datasets
            if self.similarity_metric == "cosine":
                self.index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dimension)  # L2 distance
        
        # Load existing index if available
        await self._load_faiss_index()
        
        self.stats.backend_info = {
            "index_type": config["index_type"],
            "total_vectors": self.index.ntotal,
            "is_trained": getattr(self.index, 'is_trained', True)
        }
    
    async def _initialize_chromadb(self):
        """Initialize ChromaDB backend"""
        config = self.backend_config
        
        # Create ChromaDB client
        settings = Settings(
            persist_directory=config["persist_directory"],
            anonymized_telemetry=False
        )
        
        self.chroma_client = chromadb.PersistentClient(settings=settings)
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=config["collection_name"]
            )
            logger.info(f"📦 Using existing ChromaDB collection: {config['collection_name']}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=config["collection_name"],
                metadata={"description": "BajajFinsev RAG document embeddings"}
            )
            logger.info(f"🆕 Created new ChromaDB collection: {config['collection_name']}")
        
        self.stats.backend_info = {
            "collection_name": config["collection_name"],
            "distance_function": config["distance_function"],
            "total_vectors": self.collection.count()
        }
    
    async def _initialize_in_memory(self):
        """Initialize in-memory backend"""
        self.index = {}  # Simple dictionary-based storage
        self.embeddings_matrix = None  # Will be built when needed
        
        self.stats.backend_info = {
            "storage_type": "dictionary",
            "max_documents": self.backend_config.get("max_documents", 10000)
        }
    
    async def add_document(self, document: VectorDocument) -> bool:
        """
        Add document with embedding to vector storage
        
        Args:
            document: VectorDocument to add
            
        Returns:
            True if successful
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            if self.backend == VectorBackend.FAISS:
                return await self._add_document_faiss(document)
            elif self.backend == VectorBackend.CHROMADB:
                return await self._add_document_chromadb(document)
            else:
                return await self._add_document_in_memory(document)
                
        except Exception as e:
            logger.error(f"❌ Failed to add document {document.id}: {str(e)}")
            return False
    
    async def _add_document_faiss(self, document: VectorDocument) -> bool:
        """Add document to FAISS index"""
        try:
            # Normalize embedding for cosine similarity
            embedding = document.embedding.astype('float32')
            if self.similarity_metric == "cosine":
                embedding = embedding / np.linalg.norm(embedding)
            
            # Add to index
            self.index.add(embedding.reshape(1, -1))
            
            # Store document metadata
            self.documents[document.id] = document
            
            # Update stats
            self.stats.total_documents += 1
            self.stats.total_embeddings += 1
            
            logger.debug(f"✅ Added document to FAISS: {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ FAISS add failed: {str(e)}")
            return False
    
    async def _add_document_chromadb(self, document: VectorDocument) -> bool:
        """Add document to ChromaDB"""
        try:
            # Prepare document for ChromaDB
            self.collection.add(
                ids=[document.id],
                embeddings=[document.embedding.tolist()],
                documents=[document.content],
                metadatas=[{
                    **document.metadata,
                    "document_url": document.document_url,
                    "document_hash": document.document_hash,
                    "chunk_index": document.chunk_index,
                    "created_at": document.created_at
                }]
            )
            
            # Store local copy for quick access
            self.documents[document.id] = document
            
            # Update stats
            self.stats.total_documents += 1
            self.stats.total_embeddings += 1
            
            logger.debug(f"✅ Added document to ChromaDB: {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ ChromaDB add failed: {str(e)}")
            return False
    
    async def _add_document_in_memory(self, document: VectorDocument) -> bool:
        """Add document to in-memory storage"""
        try:
            # Check capacity
            max_docs = self.backend_config.get("max_documents", 10000)
            if len(self.documents) >= max_docs:
                logger.warning(f"⚠️ In-memory storage at capacity: {max_docs}")
                return False
            
            # Store document
            self.documents[document.id] = document
            
            # Rebuild embeddings matrix
            await self._rebuild_embeddings_matrix()
            
            # Update stats
            self.stats.total_documents += 1
            self.stats.total_embeddings += 1
            
            logger.debug(f"✅ Added document to memory: {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ In-memory add failed: {str(e)}")
            return False
    
    async def _rebuild_embeddings_matrix(self):
        """Rebuild embeddings matrix for in-memory backend"""
        if not self.documents:
            self.embeddings_matrix = None
            return
        
        embeddings = []
        self.doc_id_to_index = {}
        
        for i, (doc_id, doc) in enumerate(self.documents.items()):
            embeddings.append(doc.embedding)
            self.doc_id_to_index[doc_id] = i
        
        self.embeddings_matrix = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        if self.similarity_metric == "cosine":
            norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            self.embeddings_matrix = self.embeddings_matrix / (norms + 1e-8)
    
    async def search_similar(self,
                           query_embedding: np.ndarray,
                           top_k: int = 10,
                           filter_metadata: Optional[Dict[str, Any]] = None,
                           min_similarity: float = 0.0) -> SearchResult:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            min_similarity: Minimum similarity threshold
            
        Returns:
            SearchResult with similar documents
        """
        start_time = time.time()
        self.stats.total_searches += 1
        
        if not self.is_initialized:
            await self.initialize()
        
        # Check cache first
        cache_key = None
        if self.enable_caching and self.redis_cache:
            cache_key = self._generate_search_cache_key(
                query_embedding, top_k, filter_metadata, min_similarity
            )
            
            cached_result = await self._get_cached_search_result(cache_key)
            if cached_result:
                self.stats.cache_hits += 1
                logger.debug("✅ Using cached search result")
                return cached_result
        
        self.stats.cache_misses += 1
        
        try:
            if self.backend == VectorBackend.FAISS:
                results = await self._search_faiss(
                    query_embedding, top_k, filter_metadata, min_similarity
                )
            elif self.backend == VectorBackend.CHROMADB:
                results = await self._search_chromadb(
                    query_embedding, top_k, filter_metadata, min_similarity
                )
            else:
                results = await self._search_in_memory(
                    query_embedding, top_k, filter_metadata, min_similarity
                )
            
            search_time = time.time() - start_time
            
            # Update stats
            self._update_avg_search_time(search_time)
            
            search_result = SearchResult(
                query="embedding_search",
                results=results,
                total_results=len(results),
                search_time=search_time,
                backend_used=self.backend.value,
                filter_applied=filter_metadata
            )
            
            # Cache result
            if self.enable_caching and cache_key:
                await self._cache_search_result(cache_key, search_result)
            
            logger.debug(f"🔍 Search completed: {len(results)} results in {search_time:.3f}s")
            
            return search_result
            
        except Exception as e:
            logger.error(f"❌ Search failed: {str(e)}")
            return SearchResult(
                query="embedding_search",
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                backend_used=self.backend.value
            )
    
    async def _search_faiss(self,
                          query_embedding: np.ndarray,
                          top_k: int,
                          filter_metadata: Optional[Dict[str, Any]],
                          min_similarity: float) -> List[SimilarityResult]:
        """Search using FAISS index"""
        # Normalize query for cosine similarity
        query_vec = query_embedding.astype('float32')
        if self.similarity_metric == "cosine":
            query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Search index
        distances, indices = self.index.search(query_vec.reshape(1, -1), top_k)
        
        results = []
        doc_list = list(self.documents.values())
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # No more results
                break
            
            if idx >= len(doc_list):
                continue
            
            document = doc_list[idx]
            
            # Apply metadata filter
            if filter_metadata and not self._matches_filter(document.metadata, filter_metadata):
                continue
            
            # Convert distance to similarity
            if self.similarity_metric == "cosine":
                similarity = float(1.0 - distance)  # For IP, higher = more similar
            else:
                similarity = float(1.0 / (1.0 + distance))  # Convert L2 to similarity
            
            # Apply similarity threshold
            if similarity < min_similarity:
                continue
            
            results.append(SimilarityResult(
                document=document,
                score=similarity,
                distance=float(distance)
            ))
        
        return results
    
    async def _search_chromadb(self,
                             query_embedding: np.ndarray,
                             top_k: int,
                             filter_metadata: Optional[Dict[str, Any]],
                             min_similarity: float) -> List[SimilarityResult]:
        """Search using ChromaDB"""
        # Prepare where filter
        where_filter = None
        if filter_metadata:
            where_filter = {}
            for key, value in filter_metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    where_filter[key] = value
        
        # Search collection
        search_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter
        )
        
        results = []
        if search_results['ids'] and len(search_results['ids']) > 0:
            ids = search_results['ids'][0]
            distances = search_results['distances'][0]
            documents = search_results['documents'][0]
            metadatas = search_results['metadatas'][0]
            
            for doc_id, distance, content, metadata in zip(ids, distances, documents, metadatas):
                # Get full document from local storage
                document = self.documents.get(doc_id)
                if not document:
                    # Reconstruct document from ChromaDB data
                    document = VectorDocument(
                        id=doc_id,
                        content=content,
                        embedding=query_embedding,  # Placeholder
                        metadata=metadata,
                        document_url=metadata.get('document_url', ''),
                        document_hash=metadata.get('document_hash', ''),
                        chunk_index=metadata.get('chunk_index', 0)
                    )
                
                # Convert distance to similarity (ChromaDB uses cosine distance)
                similarity = float(1.0 - distance)
                
                # Apply similarity threshold
                if similarity < min_similarity:
                    continue
                
                results.append(SimilarityResult(
                    document=document,
                    score=similarity,
                    distance=float(distance)
                ))
        
        return results
    
    async def _search_in_memory(self,
                              query_embedding: np.ndarray,
                              top_k: int,
                              filter_metadata: Optional[Dict[str, Any]],
                              min_similarity: float) -> List[SimilarityResult]:
        """Search using in-memory storage"""
        if self.embeddings_matrix is None:
            return []
        
        # Normalize query
        query_vec = query_embedding.astype('float32')
        if self.similarity_metric == "cosine":
            query_vec = query_vec / np.linalg.norm(query_vec)
        
        # Calculate similarities
        if self.similarity_metric == "cosine":
            similarities = np.dot(self.embeddings_matrix, query_vec)
        else:
            # L2 distance
            distances = np.linalg.norm(self.embeddings_matrix - query_vec, axis=1)
            similarities = 1.0 / (1.0 + distances)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more to account for filtering
        
        results = []
        doc_list = list(self.documents.values())
        
        for idx in top_indices:
            if len(results) >= top_k:
                break
            
            if idx >= len(doc_list):
                continue
            
            document = doc_list[idx]
            similarity = float(similarities[idx])
            
            # Apply similarity threshold
            if similarity < min_similarity:
                continue
            
            # Apply metadata filter
            if filter_metadata and not self._matches_filter(document.metadata, filter_metadata):
                continue
            
            distance = 1.0 - similarity if self.similarity_metric == "cosine" else float(np.linalg.norm(self.embeddings_matrix[idx] - query_vec))
            
            results.append(SimilarityResult(
                document=document,
                score=similarity,
                distance=distance
            ))
        
        return results
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        for key, expected_value in filter_criteria.items():
            if key not in metadata:
                return False
            
            actual_value = metadata[key]
            
            # Handle different value types
            if isinstance(expected_value, dict):
                # Handle operators like {"$gt": 0.5}
                for op, val in expected_value.items():
                    if op == "$gt" and actual_value <= val:
                        return False
                    elif op == "$gte" and actual_value < val:
                        return False
                    elif op == "$lt" and actual_value >= val:
                        return False
                    elif op == "$lte" and actual_value > val:
                        return False
                    elif op == "$eq" and actual_value != val:
                        return False
                    elif op == "$ne" and actual_value == val:
                        return False
            else:
                # Direct comparison
                if actual_value != expected_value:
                    return False
        
        return True
    
    async def batch_add_documents(self, documents: List[VectorDocument]) -> Dict[str, bool]:
        """
        Add multiple documents in batch
        
        Args:
            documents: List of VectorDocument objects
            
        Returns:
            Dict mapping document IDs to success status
        """
        logger.info(f"📦 Batch adding {len(documents)} documents")
        
        results = {}
        
        # Process in chunks for memory efficiency
        chunk_size = 100
        for i in range(0, len(documents), chunk_size):
            chunk = documents[i:i + chunk_size]
            
            # Add documents concurrently within chunk
            tasks = [self.add_document(doc) for doc in chunk]
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Record results
            for doc, result in zip(chunk, chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Failed to add document {doc.id}: {result}")
                    results[doc.id] = False
                else:
                    results[doc.id] = result
        
        successful = sum(1 for success in results.values() if success)
        logger.info(f"📦 Batch add completed: {successful}/{len(documents)} successful")
        
        return results
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete document from vector storage
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if successful
        """
        try:
            if document_id not in self.documents:
                logger.warning(f"⚠️ Document not found for deletion: {document_id}")
                return False
            
            if self.backend == VectorBackend.CHROMADB:
                self.collection.delete(ids=[document_id])
            
            # Remove from local storage
            del self.documents[document_id]
            
            # Update stats
            self.stats.total_documents -= 1
            self.stats.total_embeddings -= 1
            
            # Rebuild in-memory structures if needed
            if self.backend == VectorBackend.IN_MEMORY:
                await self._rebuild_embeddings_matrix()
            
            logger.debug(f"🗑️ Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document {document_id}: {str(e)}")
            return False
    
    async def get_document(self, document_id: str) -> Optional[VectorDocument]:
        """Get document by ID"""
        return self.documents.get(document_id)
    
    async def list_documents(self, 
                           limit: int = 100, 
                           offset: int = 0,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[VectorDocument]:
        """List documents with optional filtering"""
        documents = list(self.documents.values())
        
        # Apply metadata filter
        if filter_metadata:
            documents = [
                doc for doc in documents 
                if self._matches_filter(doc.metadata, filter_metadata)
            ]
        
        # Apply pagination
        return documents[offset:offset + limit]
    
    async def _load_faiss_index(self):
        """Load FAISS index from disk"""
        index_path = self.storage_path / "faiss_index.bin"
        metadata_path = self.storage_path / "faiss_metadata.pkl"
        
        try:
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"📦 Loaded FAISS index: {self.index.ntotal} vectors")
            
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"📦 Loaded {len(self.documents)} document metadata")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load existing FAISS data: {e}")
    
    async def save_index(self) -> bool:
        """Save index to persistent storage"""
        try:
            if self.backend == VectorBackend.FAISS and self.index:
                index_path = self.storage_path / "faiss_index.bin"
                metadata_path = self.storage_path / "faiss_metadata.pkl"
                
                # Save FAISS index
                faiss.write_index(self.index, str(index_path))
                
                # Save metadata
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.documents, f)
                
                logger.info(f"💾 Saved FAISS index: {self.index.ntotal} vectors")
                
            elif self.backend == VectorBackend.CHROMADB:
                # ChromaDB auto-persists
                logger.debug("💾 ChromaDB auto-persisted")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save index: {str(e)}")
            return False
    
    def _generate_search_cache_key(self, 
                                 query_embedding: np.ndarray,
                                 top_k: int,
                                 filter_metadata: Optional[Dict[str, Any]],
                                 min_similarity: float) -> str:
        """Generate cache key for search"""
        # Create hash from query parameters
        key_data = {
            "embedding_hash": hashlib.md5(query_embedding.tobytes()).hexdigest()[:16],
            "top_k": top_k,
            "filter": filter_metadata,
            "min_similarity": min_similarity,
            "backend": self.backend.value
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"vector_search:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _get_cached_search_result(self, cache_key: str) -> Optional[SearchResult]:
        """Get cached search result"""
        if not self.redis_cache:
            return None
        
        try:
            cached_data = await self.redis_cache.get(cache_key)
            if cached_data:
                # Deserialize result (simplified - would need proper serialization)
                return None  # TODO: Implement proper serialization
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        
        return None
    
    async def _cache_search_result(self, cache_key: str, result: SearchResult):
        """Cache search result"""
        if not self.redis_cache:
            return
        
        try:
            # Serialize result (simplified - would need proper serialization)
            # await self.redis_cache.setex(cache_key, self.cache_ttl, serialized_result)
            pass  # TODO: Implement proper serialization
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def _update_avg_search_time(self, search_time: float):
        """Update average search time"""
        total_searches = self.stats.total_searches
        current_avg = self.stats.avg_search_time
        
        self.stats.avg_search_time = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        # Calculate index size
        index_size_mb = 0.0
        if self.backend == VectorBackend.FAISS and self.index:
            # Estimate FAISS index size
            index_size_mb = (self.index.ntotal * self.embedding_dimension * 4) / (1024 * 1024)
        elif self.embeddings_matrix is not None:
            index_size_mb = self.embeddings_matrix.nbytes / (1024 * 1024)
        
        self.stats.index_size_mb = index_size_mb
        
        return {
            "backend": self.backend.value,
            "embedding_dimension": self.embedding_dimension,
            "similarity_metric": self.similarity_metric,
            "storage_path": str(self.storage_path),
            "is_initialized": self.is_initialized,
            "stats": {
                "total_documents": self.stats.total_documents,
                "total_embeddings": self.stats.total_embeddings,
                "index_size_mb": self.stats.index_size_mb,
                "total_searches": self.stats.total_searches,
                "avg_search_time": self.stats.avg_search_time,
                "cache_hit_rate": (
                    self.stats.cache_hits / (self.stats.cache_hits + self.stats.cache_misses) * 100
                    if (self.stats.cache_hits + self.stats.cache_misses) > 0 else 0
                ),
                "backend_info": self.stats.backend_info
            },
            "configuration": {
                "enable_caching": self.enable_caching,
                "cache_ttl": self.cache_ttl,
                "backend_config": self.backend_config
            }
        }
    
    async def cleanup(self):
        """Cleanup storage resources"""
        logger.info("🧹 Cleaning up vector storage resources")
        
        try:
            # Save index before cleanup
            await self.save_index()
            
            # Clear in-memory data
            self.documents.clear()
            self.index = None
            self.embeddings_matrix = None
            
            # Close ChromaDB client
            if hasattr(self, 'chroma_client'):
                del self.chroma_client
                del self.collection
            
            # Clear cache references
            self.redis_cache = None
            
            self.is_initialized = False
            
            logger.info("✅ Vector storage cleanup complete")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")


# Global storage service instance
vector_storage: Optional[VectorStorageService] = None


def get_vector_storage(**kwargs) -> VectorStorageService:
    """Get or create global vector storage service instance"""
    global vector_storage
    
    if vector_storage is None:
        vector_storage = VectorStorageService(**kwargs)
    
    return vector_storage


async def initialize_vector_storage(**kwargs) -> VectorStorageService:
    """Initialize and return vector storage service"""
    service = get_vector_storage(**kwargs)
    
    if not service.is_initialized:
        await service.initialize()
    
    # Log initialization summary
    stats = await service.get_storage_stats()
    logger.info("🗄️ Vector Storage Summary:")
    logger.info(f"  Backend: {stats['backend']}")
    logger.info(f"  Embedding dimension: {stats['embedding_dimension']}")
    logger.info(f"  Storage path: {stats['storage_path']}")
    logger.info(f"  Documents: {stats['stats']['total_documents']}")
    logger.info(f"  Index size: {stats['stats']['index_size_mb']:.2f} MB")
    
    return service