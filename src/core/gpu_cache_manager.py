"""
GPU Cache Manager - High Performance GPU Operations with Memory Caching
Optimized for RTX 3050 with intelligent caching and memory management
"""

import hashlib
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss

from src.core.config import config

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Cache performance statistics"""
    gpu_cache_hits: int = 0
    gpu_cache_misses: int = 0
    memory_cache_hits: int = 0
    memory_cache_misses: int = 0
    embeddings_computed: int = 0
    total_gpu_time: float = 0.0
    cache_hit_rate: float = 0.0

class GPUCacheManager:
    """
    High-performance GPU operations with intelligent caching
    - GPU memory pooling for embeddings
    - Multi-level caching (GPU memory → CPU memory → Disk)
    - Batch optimization for maximum throughput
    - RTX 3050 memory management
    """
    
    def __init__(self):
        # GPU Configuration
        self.device = None
        self.model: Optional[SentenceTransformer] = None
        self.is_initialized = False
        
        # Memory Management
        self.gpu_memory_pool: Dict[str, torch.Tensor] = {}
        self.cpu_embedding_cache: Dict[str, np.ndarray] = {}
        self.batch_queue: List[str] = []
        
        # FAISS GPU Index
        self.gpu_index: Optional[faiss.Index] = None
        self.embedding_dim = 768  # e5-base dimension
        
        # Performance Optimization
        self.batch_size = 64  # Optimized for RTX 3050
        self.max_cache_size = 10000  # Max embeddings in memory
        self.gpu_memory_threshold = 0.8  # Use 80% of GPU memory
        
        # Threading for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        self.cache_lock = threading.RLock()
        
        # Statistics
        self.stats = CacheStats()
        
        logger.info("GPU Cache Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize GPU, model, and caches"""
        try:
            logger.info("🚀 Initializing GPU Cache Manager...")
            
            # Setup GPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_per_process_memory_fraction(0.8)  # RTX 3050 optimization
                torch.cuda.empty_cache()
                logger.info(f"✅ GPU initialized: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                logger.warning("⚠️ GPU not available, using CPU")
            
            # Load embedding model once
            logger.info("Loading embedding model...")
            self.model = SentenceTransformer(
                'intfloat/multilingual-e5-base',
                device=self.device
            )
            
            # Enable mixed precision for RTX 3050
            if self.device.type == 'cuda':
                self.model.half()  # FP16 for memory efficiency
                logger.info("✅ Mixed precision (FP16) enabled")
            
            # Initialize FAISS GPU index
            await self._initialize_faiss_gpu()
            
            self.is_initialized = True
            logger.info("✅ GPU Cache Manager fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ GPU Cache Manager initialization failed: {e}")
            return False
    
    async def _initialize_faiss_gpu(self):
        """Initialize FAISS with GPU acceleration"""
        try:
            # Create FAISS index
            self.gpu_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            
            if self.device.type == 'cuda':
                # Move FAISS to GPU
                res = faiss.StandardGpuResources()
                res.setTempMemory(512 * 1024 * 1024)  # 512MB temp memory
                self.gpu_index = faiss.index_cpu_to_gpu(res, 0, self.gpu_index)
                logger.info("✅ FAISS GPU index initialized")
            else:
                logger.info("✅ FAISS CPU index initialized")
                
        except Exception as e:
            logger.warning(f"FAISS GPU initialization failed, using CPU: {e}")
            self.gpu_index = faiss.IndexFlatIP(self.embedding_dim)
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    async def get_embeddings(self, texts: List[str], use_cache: bool = True) -> np.ndarray:
        """
        Get embeddings with multi-level caching
        Priority: GPU cache → CPU cache → Compute new
        """
        if not self.is_initialized:
            raise RuntimeError("GPU Cache Manager not initialized")
        
        start_time = time.time()
        
        # Separate cached and uncached texts
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            with self.cache_lock:
                for i, text in enumerate(texts):
                    cache_key = self._get_cache_key(text)
                    
                    # Check GPU memory cache first
                    if cache_key in self.gpu_memory_pool:
                        cached_embeddings[i] = self.gpu_memory_pool[cache_key].cpu().numpy()
                        self.stats.gpu_cache_hits += 1
                        continue
                    
                    # Check CPU memory cache
                    if cache_key in self.cpu_embedding_cache:
                        cached_embeddings[i] = self.cpu_embedding_cache[cache_key]
                        self.stats.memory_cache_hits += 1
                        continue
                    
                    # Need to compute
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    self.stats.gpu_cache_misses += 1
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Compute embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            new_embeddings = await self._compute_embeddings_batch(uncached_texts, use_cache)
            self.stats.embeddings_computed += len(uncached_texts)
        
        # Combine cached and new embeddings
        result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        # Fill cached embeddings
        for i, embedding in cached_embeddings.items():
            result[i] = embedding
        
        # Fill new embeddings
        for i, embedding in zip(uncached_indices, new_embeddings):
            result[i] = embedding
        
        # Update statistics
        total_time = time.time() - start_time
        self.stats.total_gpu_time += total_time
        self._update_cache_hit_rate()
        
        logger.debug(f"Got {len(texts)} embeddings in {total_time:.3f}s (cached: {len(cached_embeddings)}, computed: {len(new_embeddings)})")
        
        return result
    
    async def _compute_embeddings_batch(self, texts: List[str], cache_results: bool = True) -> List[np.ndarray]:
        """Compute embeddings in optimized batches"""
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in optimized batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Compute embeddings on GPU
            with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                
                # Normalize for cosine similarity
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            # Cache results
            if cache_results:
                self._cache_batch_embeddings(batch_texts, batch_embeddings)
            
            # Convert to numpy and add to results
            batch_numpy = batch_embeddings.cpu().numpy().astype(np.float32)
            all_embeddings.extend(batch_numpy)
            
            # GPU memory management
            del batch_embeddings
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return all_embeddings
    
    def _cache_batch_embeddings(self, texts: List[str], embeddings: torch.Tensor):
        """Cache embeddings in both GPU and CPU memory"""
        with self.cache_lock:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                
                # Cache in GPU memory (limited by GPU memory)
                if len(self.gpu_memory_pool) < 1000:  # Limit GPU cache size
                    self.gpu_memory_pool[cache_key] = embeddings[i].clone().detach()
                
                # Always cache in CPU memory
                if len(self.cpu_embedding_cache) >= self.max_cache_size:
                    # Remove oldest entries (simple LRU)
                    keys_to_remove = list(self.cpu_embedding_cache.keys())[:100]
                    for key in keys_to_remove:
                        del self.cpu_embedding_cache[key]
                
                self.cpu_embedding_cache[cache_key] = embeddings[i].cpu().numpy()
    
    async def add_to_vector_index(self, embeddings: np.ndarray) -> bool:
        """Add embeddings to FAISS GPU index"""
        try:
            if self.gpu_index is None:
                return False
            
            # Ensure embeddings are float32
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            # Add to index
            self.gpu_index.add(embeddings)
            logger.debug(f"Added {len(embeddings)} vectors to GPU index (total: {self.gpu_index.ntotal})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to index: {e}")
            return False
    
    async def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Search similar vectors using GPU-accelerated FAISS"""
        try:
            if self.gpu_index is None or self.gpu_index.ntotal == 0:
                return np.array([]), np.array([])
            
            # Ensure query is float32 and 2D
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
            
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Search
            k = min(k, self.gpu_index.ntotal)
            scores, indices = self.gpu_index.search(query_embedding, k)
            
            return scores[0], indices[0]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return np.array([]), np.array([])
    
    def _update_cache_hit_rate(self):
        """Update cache hit rate statistics"""
        total_requests = self.stats.gpu_cache_hits + self.stats.gpu_cache_misses
        if total_requests > 0:
            self.stats.cache_hit_rate = self.stats.gpu_cache_hits / total_requests
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        gpu_memory_used = 0
        if self.device and self.device.type == 'cuda':
            gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
        
        return {
            'gpu_cache_size': len(self.gpu_memory_pool),
            'cpu_cache_size': len(self.cpu_embedding_cache),
            'gpu_cache_hits': self.stats.gpu_cache_hits,
            'gpu_cache_misses': self.stats.gpu_cache_misses,
            'memory_cache_hits': self.stats.memory_cache_hits,
            'memory_cache_misses': self.stats.memory_cache_misses,
            'cache_hit_rate': self.stats.cache_hit_rate,
            'embeddings_computed': self.stats.embeddings_computed,
            'total_gpu_time': self.stats.total_gpu_time,
            'average_gpu_time_per_embedding': self.stats.total_gpu_time / max(1, self.stats.embeddings_computed),
            'gpu_memory_used_mb': gpu_memory_used,
            'faiss_index_size': self.gpu_index.ntotal if self.gpu_index else 0,
            'device': str(self.device)
        }
    
    def clear_cache(self, clear_gpu: bool = True, clear_cpu: bool = True):
        """Clear caches to free memory"""
        with self.cache_lock:
            if clear_gpu:
                self.gpu_memory_pool.clear()
                if self.device and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                logger.info("🗑️ GPU cache cleared")
            
            if clear_cpu:
                self.cpu_embedding_cache.clear()
                logger.info("🗑️ CPU cache cleared")
    
    def optimize_for_speed(self):
        """Apply speed optimizations for RTX 3050"""
        if self.device and self.device.type == 'cuda':
            # Increase batch size for better GPU utilization
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory_gb <= 4:  # RTX 3050
                self.batch_size = 128  # Aggressive batching
                logger.info("🎯 RTX 3050 speed optimization applied (batch_size=128)")
            else:
                self.batch_size = 256
                logger.info("🚀 High-end GPU speed optimization applied")
    
    async def preload_common_embeddings(self, texts: List[str]):
        """Preload frequently used embeddings"""
        logger.info(f"🔄 Preloading {len(texts)} common embeddings...")
        await self.get_embeddings(texts, use_cache=True)
        logger.info("✅ Common embeddings preloaded")


# Global GPU Cache Manager instance
gpu_cache_manager = GPUCacheManager()