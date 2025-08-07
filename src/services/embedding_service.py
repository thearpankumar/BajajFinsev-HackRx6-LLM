"""
Basic Embedding Service for BajajFinsev Hybrid RAG System
GPU-accelerated embedding generation using BAAI/bge-m3 model
Optimized for RTX 3050 with mixed precision and batch processing
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# Core dependencies
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    torch = None

# GPU service integration
from src.core.gpu_service import get_gpu_service, GPUService

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding operation"""
    success: bool
    embeddings: Optional[np.ndarray] = None
    text_count: int = 0
    embedding_dimensions: int = 0
    processing_time: float = 0.0
    model_used: str = ""
    device_used: str = ""
    error_message: Optional[str] = None
    batch_info: Optional[Dict[str, Any]] = None


class BasicEmbeddingService:
    """
    Basic embedding service optimized for RTX 3050
    Features: BAAI/bge-m3 model, mixed precision, batch processing, GPU acceleration
    """
    
    # Model configurations
    EMBEDDING_MODELS = {
        "bge-m3": {
            "name": "BAAI/bge-m3",
            "dimensions": 1024,
            "max_length": 8192,
            "batch_size_gpu": 16,  # RTX 3050 optimized
            "batch_size_cpu": 4
        },
        "bge-small": {
            "name": "BAAI/bge-small-en-v1.5", 
            "dimensions": 384,
            "max_length": 512,
            "batch_size_gpu": 32,
            "batch_size_cpu": 8
        }
    }
    
    def __init__(self, 
                 model_name: str = "bge-m3",
                 enable_caching: bool = True,
                 max_sequence_length: Optional[int] = None):
        """
        Initialize embedding service
        
        Args:
            model_name: Model to use ("bge-m3" or "bge-small")
            enable_caching: Whether to enable embedding caching
            max_sequence_length: Override default max sequence length
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch and transformers are required for embedding service")
            
        self.model_name = model_name
        self.enable_caching = enable_caching
        
        # Model configuration
        if model_name not in self.EMBEDDING_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Available: {list(self.EMBEDDING_MODELS.keys())}")
            
        self.model_config = self.EMBEDDING_MODELS[model_name]
        self.max_sequence_length = max_sequence_length or self.model_config["max_length"]
        
        # Initialize GPU service
        self.gpu_service = get_gpu_service()
        self.device_config = self.gpu_service.get_device_config()
        
        # Model components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[Union[AutoModel, SentenceTransformer]] = None
        self.is_initialized = False
        
        # Caching
        self.embedding_cache: Dict[str, np.ndarray] = {} if enable_caching else None
        
        # Statistics
        self.stats = {
            'total_embeddings': 0,
            'successful_embeddings': 0,
            'failed_embeddings': 0,
            'cache_hits': 0,
            'total_texts_processed': 0,
            'avg_processing_time': 0.0,
            'gpu_processing_time': 0.0,
            'cpu_processing_time': 0.0,
            'batch_processing_count': 0,
            'model_loading_time': 0.0
        }
        
        logger.info(f"BasicEmbeddingService initialized")
        logger.info(f"Model: {self.model_config['name']}")
        logger.info(f"Device: {self.device_config.device if self.device_config else 'cpu'}")
        logger.info(f"Mixed precision: {self.device_config.mixed_precision if self.device_config else False}")
    
    async def initialize(self) -> bool:
        """Initialize embedding model with GPU optimizations"""
        if self.is_initialized:
            return True
            
        start_time = time.time()
        logger.info(f"🔄 Initializing embedding model: {self.model_config['name']}")
        
        try:
            # Determine device and batch size
            device = self.device_config.device if self.device_config else "cpu"
            is_gpu = device != "cpu"
            
            # Get RTX 3050 optimized batch size
            if is_gpu and self.device_config.optimization_level == "rtx_3050_optimized":
                batch_size = self.model_config["batch_size_gpu"]
                logger.info(f"🎮 RTX 3050 optimization: batch_size={batch_size}")
            elif is_gpu:
                batch_size = self.model_config["batch_size_gpu"]
            else:
                batch_size = self.model_config["batch_size_cpu"]
            
            # Store optimized batch size
            self.optimal_batch_size = batch_size
            
            # Load tokenizer
            logger.debug(f"Loading tokenizer: {self.model_config['name']}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['name'],
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            
            # Load model based on type
            if self.model_name == "bge-m3":
                # Use sentence-transformers for BGE-M3 (better optimization)
                logger.debug("Loading BGE-M3 with sentence-transformers")
                self.model = SentenceTransformer(
                    self.model_config['name'],
                    device=device,
                    cache_folder="./model_cache"
                )
                
                # Apply RTX 3050 optimizations
                if is_gpu and hasattr(self.model[0], 'auto_model'):
                    base_model = self.model[0].auto_model
                    
                    # Mixed precision for RTX 3050
                    if self.device_config and self.device_config.mixed_precision:
                        logger.info("🎮 Enabling mixed precision (FP16) for RTX 3050")
                        base_model = base_model.half()
                    
                    # Memory optimization
                    if self.device_config and self.device_config.optimization_level == "rtx_3050_optimized":
                        # Enable gradient checkpointing for memory efficiency
                        if hasattr(base_model.config, 'use_cache'):
                            base_model.config.use_cache = False
                        if hasattr(base_model, 'gradient_checkpointing_enable'):
                            base_model.gradient_checkpointing_enable()
                        
                        logger.info("🎮 RTX 3050 memory optimizations applied")
            
            else:
                # Use transformers for other models
                logger.debug(f"Loading model with transformers: {self.model_config['name']}")
                self.model = AutoModel.from_pretrained(
                    self.model_config['name'],
                    trust_remote_code=True,
                    cache_dir="./model_cache"
                ).to(device)
                
                # Mixed precision
                if is_gpu and self.device_config and self.device_config.mixed_precision:
                    self.model = self.model.half()
                    logger.info("🎮 Mixed precision (FP16) enabled")
            
            # Set model to evaluation mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            loading_time = time.time() - start_time
            self.stats['model_loading_time'] = loading_time
            
            self.is_initialized = True
            
            logger.info(f"✅ Embedding model initialized in {loading_time:.2f}s")
            logger.info(f"Device: {device}")
            logger.info(f"Optimal batch size: {self.optimal_batch_size}")
            logger.info(f"Max sequence length: {self.max_sequence_length}")
            logger.info(f"Embedding dimensions: {self.model_config['dimensions']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {str(e)}")
            self.is_initialized = False
            return False
    
    async def generate_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for list of texts with GPU optimization
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()
        self.stats['total_embeddings'] += 1
        self.stats['total_texts_processed'] += len(texts)
        
        logger.debug(f"🔄 Generating embeddings for {len(texts)} texts")
        
        try:
            # Ensure model is initialized
            if not self.is_initialized:
                initialized = await self.initialize()
                if not initialized:
                    return self._create_error_result(
                        "Failed to initialize embedding model",
                        time.time() - start_time
                    )
            
            # Check for cached embeddings
            if self.enable_caching:
                cached_embeddings, uncached_texts, cache_indices = self._check_cache(texts)
                if len(uncached_texts) == 0:
                    # All embeddings cached
                    self.stats['cache_hits'] += len(texts)
                    logger.debug(f"✅ All {len(texts)} embeddings found in cache")
                    
                    return EmbeddingResult(
                        success=True,
                        embeddings=cached_embeddings,
                        text_count=len(texts),
                        embedding_dimensions=cached_embeddings.shape[1],
                        processing_time=time.time() - start_time,
                        model_used=self.model_config['name'],
                        device_used=self.device_config.device if self.device_config else "cpu"
                    )
                else:
                    texts_to_process = uncached_texts
                    logger.debug(f"📦 Cache hit: {len(texts) - len(uncached_texts)}, Processing: {len(uncached_texts)}")
            else:
                texts_to_process = texts
                cached_embeddings = None
                cache_indices = None
            
            # Process texts in batches
            embeddings = await self._process_texts_batched(texts_to_process)
            
            # Combine cached and new embeddings
            if cached_embeddings is not None and len(cached_embeddings) > 0:
                final_embeddings = self._combine_cached_and_new_embeddings(
                    cached_embeddings, embeddings, cache_indices
                )
            else:
                final_embeddings = embeddings
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['successful_embeddings'] += 1
            self._update_avg_processing_time(processing_time)
            
            # Update device-specific timing
            device_type = "gpu" if self.device_config and self.device_config.device != "cpu" else "cpu"
            if device_type == "gpu":
                self.stats['gpu_processing_time'] += processing_time
            else:
                self.stats['cpu_processing_time'] += processing_time
            
            logger.info(f"✅ Generated {final_embeddings.shape[0]} embeddings "
                       f"({final_embeddings.shape[1]}D) in {processing_time:.2f}s")
            
            return EmbeddingResult(
                success=True,
                embeddings=final_embeddings,
                text_count=len(texts),
                embedding_dimensions=final_embeddings.shape[1],
                processing_time=processing_time,
                model_used=self.model_config['name'],
                device_used=self.device_config.device if self.device_config else "cpu",
                batch_info={
                    "total_texts": len(texts),
                    "cached_texts": len(texts) - len(texts_to_process) if self.enable_caching else 0,
                    "processed_texts": len(texts_to_process),
                    "optimal_batch_size": self.optimal_batch_size
                }
            )
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            self.stats['failed_embeddings'] += 1
            
            return self._create_error_result(error_msg, time.time() - start_time)
    
    async def _process_texts_batched(self, texts: List[str]) -> np.ndarray:
        """Process texts in optimized batches"""
        all_embeddings = []
        device = self.device_config.device if self.device_config else "cpu"
        
        # Process in batches
        for i in range(0, len(texts), self.optimal_batch_size):
            batch_texts = texts[i:i + self.optimal_batch_size]
            batch_start = time.time()
            
            logger.debug(f"Processing batch {i//self.optimal_batch_size + 1}: {len(batch_texts)} texts")
            
            try:
                if isinstance(self.model, SentenceTransformer):
                    # Use sentence-transformers encode method
                    with torch.no_grad():
                        batch_embeddings = self.model.encode(
                            batch_texts,
                            batch_size=len(batch_texts),
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,  # L2 normalization
                            device=device
                        )
                else:
                    # Manual processing with transformers
                    batch_embeddings = await self._encode_with_transformers(batch_texts, device)
                
                all_embeddings.append(batch_embeddings)
                
                # Cache new embeddings
                if self.enable_caching:
                    for j, text in enumerate(batch_texts):
                        cache_key = self._get_cache_key(text)
                        self.embedding_cache[cache_key] = batch_embeddings[j]
                
                self.stats['batch_processing_count'] += 1
                
                # GPU memory cleanup for RTX 3050
                if device != "cpu" and self.device_config and self.device_config.optimization_level == "rtx_3050_optimized":
                    self.gpu_service.clear_gpu_cache()
                
                batch_time = time.time() - batch_start
                logger.debug(f"Batch processed in {batch_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}")
                raise
        
        # Combine all batch results
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            raise RuntimeError("No embeddings generated")
    
    async def _encode_with_transformers(self, texts: List[str], device: str) -> np.ndarray:
        """Manual encoding using transformers (for non-sentence-transformers models)"""
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors='pt'
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            if self.device_config and self.device_config.mixed_precision:
                # Mixed precision inference
                with torch.autocast(device_type=device.split(':')[0] if ':' in device else device):
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)
            
            # Pool embeddings (mean pooling)
            embeddings = self._mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply mean pooling to get sentence embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _check_cache(self, texts: List[str]) -> Tuple[Optional[np.ndarray], List[str], Optional[List[int]]]:
        """Check cache for existing embeddings"""
        if not self.enable_caching or not self.embedding_cache:
            return None, texts, None
        
        cached_embeddings = []
        uncached_texts = []
        cache_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                cached_embeddings.append(self.embedding_cache[cache_key])
                cache_indices.append(i)
            else:
                uncached_texts.append(text)
        
        if cached_embeddings:
            return np.array(cached_embeddings), uncached_texts, cache_indices
        else:
            return None, texts, None
    
    def _combine_cached_and_new_embeddings(
        self, 
        cached_embeddings: np.ndarray, 
        new_embeddings: np.ndarray, 
        cache_indices: List[int]
    ) -> np.ndarray:
        """Combine cached and newly generated embeddings in correct order"""
        total_count = len(cache_indices) + len(new_embeddings)
        embedding_dim = cached_embeddings.shape[1] if len(cached_embeddings) > 0 else new_embeddings.shape[1]
        
        final_embeddings = np.zeros((total_count, embedding_dim))
        
        # Place cached embeddings
        cached_idx = 0
        new_idx = 0
        
        for i in range(total_count):
            if cached_idx < len(cache_indices) and i == cache_indices[cached_idx]:
                final_embeddings[i] = cached_embeddings[cached_idx]
                cached_idx += 1
            else:
                final_embeddings[i] = new_embeddings[new_idx]
                new_idx += 1
        
        return final_embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use hash of text + model name for uniqueness
        import hashlib
        text_hash = hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()
        return text_hash
    
    def _create_error_result(self, error_message: str, processing_time: float) -> EmbeddingResult:
        """Create error result"""
        return EmbeddingResult(
            success=False,
            error_message=error_message,
            processing_time=processing_time,
            model_used=self.model_config['name'],
            device_used=self.device_config.device if self.device_config else "unknown"
        )
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update running average processing time"""
        successful_count = self.stats['successful_embeddings']
        current_avg = self.stats['avg_processing_time']
        
        self.stats['avg_processing_time'] = (
            (current_avg * (successful_count - 1) + processing_time) / successful_count
        )
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """Get comprehensive embedding statistics"""
        gpu_stats = {}
        if self.gpu_service.is_gpu_available():
            gpu_stats = await self.gpu_service.get_gpu_stats()
        
        return {
            **self.stats,
            "model_info": {
                "name": self.model_config['name'],
                "dimensions": self.model_config['dimensions'],
                "max_length": self.max_sequence_length,
                "optimal_batch_size": getattr(self, 'optimal_batch_size', 0)
            },
            "device_info": {
                "device": self.device_config.device if self.device_config else "cpu",
                "mixed_precision": self.device_config.mixed_precision if self.device_config else False,
                "optimization_level": self.device_config.optimization_level if self.device_config else "unknown"
            },
            "cache_info": {
                "enabled": self.enable_caching,
                "cache_size": len(self.embedding_cache) if self.embedding_cache else 0,
                "cache_hit_rate": (
                    self.stats['cache_hits'] / self.stats['total_texts_processed'] * 100
                    if self.stats['total_texts_processed'] > 0 else 0
                )
            },
            "performance": {
                "success_rate": (
                    self.stats['successful_embeddings'] / self.stats['total_embeddings'] * 100
                    if self.stats['total_embeddings'] > 0 else 0
                ),
                "avg_processing_time": self.stats['avg_processing_time'],
                "total_batches_processed": self.stats['batch_processing_count']
            },
            "gpu_stats": gpu_stats,
            "is_initialized": self.is_initialized
        }
    
    def clear_cache(self) -> int:
        """Clear embedding cache and return number of entries cleared"""
        if not self.enable_caching or not self.embedding_cache:
            return 0
        
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"🧹 Cleared {cache_size} cached embeddings")
        return cache_size
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up embedding service resources")
        
        try:
            # Clear cache
            if self.embedding_cache:
                self.embedding_cache.clear()
            
            # Clear GPU memory
            if self.gpu_service.is_gpu_available():
                self.gpu_service.clear_gpu_cache()
            
            # Delete model references
            self.model = None
            self.tokenizer = None
            self.is_initialized = False
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("✅ Embedding service cleanup complete")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")


# Global embedding service instance
embedding_service: Optional[BasicEmbeddingService] = None


def get_embedding_service(model_name: str = "bge-m3") -> BasicEmbeddingService:
    """Get or create global embedding service instance"""
    global embedding_service
    
    if embedding_service is None or embedding_service.model_name != model_name:
        embedding_service = BasicEmbeddingService(model_name=model_name)
    
    return embedding_service


async def initialize_embedding_service(model_name: str = "bge-m3") -> BasicEmbeddingService:
    """Initialize and return embedding service"""
    service = get_embedding_service(model_name)
    
    if not service.is_initialized:
        await service.initialize()
    
    # Log initialization summary
    stats = await service.get_embedding_stats()
    logger.info("🎯 Embedding Service Summary:")
    logger.info(f"  Model: {stats['model_info']['name']}")
    logger.info(f"  Device: {stats['device_info']['device']}")
    logger.info(f"  Mixed Precision: {stats['device_info']['mixed_precision']}")
    logger.info(f"  Dimensions: {stats['model_info']['dimensions']}")
    logger.info(f"  Batch Size: {stats['model_info']['optimal_batch_size']}")
    
    return service