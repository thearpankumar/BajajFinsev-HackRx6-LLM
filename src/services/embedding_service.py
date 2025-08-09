"""
Configurable Multilingual Embedding Service
RTX 3050 optimized with multilingual support and centralized configuration
"""

import logging
import time
from pathlib import Path
from typing import Any, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from src.core.config import config
from src.core.gpu_service import GPUService
from src.services.redis_cache import redis_manager

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Configurable multilingual embedding service with GPU optimization
    Supports multiple embedding models with RTX 3050 optimization
    """

    def __init__(self, gpu_service: Union[GPUService, None] = None):
        # Configuration from central config
        self.embedding_model_name = config.embedding_model
        self.embedding_dimension = config.embedding_dimension
        self.max_length = config.embedding_max_length
        self.batch_size = config.batch_size
        self.max_batch_size = config.max_batch_size
        self.mixed_precision = config.enable_mixed_precision
        self.enable_cache = config.enable_embedding_cache

        # GPU service integration
        self.gpu_service = gpu_service or GPUService()

        # Model state
        self.model: Union[SentenceTransformer, None] = None
        self.device = None
        self.is_initialized = False
        self.model_info = {}

        # Performance tracking
        self.total_embeddings = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        # Redis cache manager
        self.redis_manager = redis_manager

        logger.info(f"EmbeddingService initialized with model: {self.embedding_model_name}")

    async def initialize(self) -> dict[str, Any]:
        """Initialize the embedding service with GPU optimization"""
        try:
            logger.info("🔄 Initializing Embedding Service...")
            start_time = time.time()

            # Check dependencies
            if not HAS_TORCH or not HAS_SENTENCE_TRANSFORMERS:
                missing = []
                if not HAS_TORCH:
                    missing.append("torch")
                if not HAS_SENTENCE_TRANSFORMERS:
                    missing.append("sentence-transformers")

                return {
                    "status": "error",
                    "error": f"Missing dependencies: {', '.join(missing)}",
                    "message": "Install with: pip install torch sentence-transformers"
                }

            # Initialize GPU service first
            gpu_info = self.gpu_service.initialize()
            self.device = self.gpu_service.device

            # Load embedding model
            model_result = await self._load_model()

            if model_result["status"] != "success":
                return model_result

            # Configure model for GPU
            self._configure_model_for_gpu()

            # Initialize cache if enabled
            if self.enable_cache:
                cache_init = await self._initialize_cache()
                if cache_init["status"] != "success":
                    logger.warning("⚠️ Cache initialization failed, continuing without cache")
                    self.enable_cache = False

            self.is_initialized = True
            initialization_time = time.time() - start_time

            result = {
                "status": "success",
                "message": f"Embedding service initialized in {initialization_time:.2f}s",
                "model_name": self.embedding_model_name,
                "device": str(self.device),
                "embedding_dimension": self.embedding_dimension,
                "max_sequence_length": self.max_length,
                "gpu_info": gpu_info,
                "cache_enabled": self.enable_cache,
                "initialization_time": initialization_time
            }

            logger.info(f"✅ {result['message']}")
            return result

        except Exception as e:
            logger.error(f"❌ Embedding service initialization failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _load_model(self) -> dict[str, Any]:
        """Load the configured embedding model"""
        try:
            logger.info(f"📦 Loading embedding model: {self.embedding_model_name}")

            # Load model with optimizations
            device_str = str(self.device) if self.device else 'cpu'
            model_kwargs = {
                'trust_remote_code': True
            }

            # Add cache directory for models
            cache_dir = Path("./models_cache")
            cache_dir.mkdir(exist_ok=True)
            model_kwargs['cache_folder'] = str(cache_dir)

            # Load the model initially on CPU to avoid CUDA memory issues during loading
            self.model = SentenceTransformer(
                self.embedding_model_name,
                **model_kwargs
            )
            
            # Move to GPU after loading if GPU is available
            if self.device and str(self.device) != 'cpu':
                logger.info(f"🎯 Moving model to {device_str}")
                self.model = self.model.to(self.device)

            # Get model information
            self.model_info = {
                "model_name": self.embedding_model_name,
                "max_seq_length": getattr(self.model, 'max_seq_length', self.max_length),
                "embedding_dimension": self.model.get_sentence_embedding_dimension(),
                "tokenizer_name": self.model.tokenizer.__class__.__name__ if hasattr(self.model, 'tokenizer') else "unknown"
            }

            # Update embedding dimension from model
            self.embedding_dimension = self.model_info["embedding_dimension"]

            logger.info(f"✅ Model loaded: {self.embedding_dimension}D embeddings")

            return {
                "status": "success",
                "model_info": self.model_info
            }

        except Exception as e:
            error_msg = f"Model loading failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }

    def _configure_model_for_gpu(self):
        """Configure model for GPU optimization"""
        if not self.model or not self.gpu_service.is_gpu_available:
            return

        try:
            # Model is already moved to GPU in _load_model, just apply optimizations
            
            # Enable mixed precision if configured
            if self.mixed_precision and self.gpu_service.gpu_provider == "cuda":
                # Enable autocast for inference
                self.model.half()  # Convert to FP16
                logger.info("🔄 Model converted to FP16 for memory efficiency")

            # Set model to evaluation mode
            self.model.eval()

            # Disable gradient computation for inference
            for param in self.model.parameters():
                param.requires_grad = False

            logger.info(f"✅ Model optimized for {self.device} inference")

        except Exception as e:
            logger.warning(f"Model GPU configuration failed: {str(e)}")

    async def _initialize_cache(self) -> dict[str, Any]:
        """Initialize embedding cache"""
        try:
            if not self.redis_manager.is_connected:
                await self.redis_manager.initialize()

            if self.redis_manager.is_connected:
                # Test cache with a simple embedding
                test_key = "embedding_cache_test"
                test_value = {"test": "embedding_cache"}

                success = await self.redis_manager.set_json(test_key, test_value, ex=10)
                if success:
                    await self.redis_manager.delete(test_key)
                    return {"status": "success", "message": "Embedding cache initialized"}

            return {"status": "error", "message": "Cache connection failed"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _generate_cache_key(self, text: Union[str, list[str]], model_name: str) -> str:
        """Generate cache key for embeddings"""
        import hashlib

        if isinstance(text, list):
            text_content = "|".join(text)
        else:
            text_content = text

        # Create hash from text and model name
        content = f"{model_name}:{text_content}"
        return f"embedding:{hashlib.md5(content.encode()).hexdigest()}"

    async def _get_cached_embeddings(self, texts: list[str]) -> dict[str, Any]:
        """Get embeddings from cache if available"""
        if not self.enable_cache or not self.redis_manager.is_connected:
            return {"cached": [], "missing": list(range(len(texts)))}

        try:
            cached_embeddings = {}
            missing_indices = []

            for i, text in enumerate(texts):
                cache_key = self._generate_cache_key(text, self.embedding_model_name)
                cached_data = await self.redis_manager.get_json(cache_key)

                if cached_data and "embedding" in cached_data:
                    cached_embeddings[i] = np.array(cached_data["embedding"])
                    self.cache_hits += 1
                else:
                    missing_indices.append(i)
                    self.cache_misses += 1

            return {
                "cached": cached_embeddings,
                "missing": missing_indices
            }

        except Exception as e:
            logger.warning(f"Cache retrieval failed: {str(e)}")
            return {"cached": {}, "missing": list(range(len(texts)))}

    async def _cache_embeddings(self, texts: list[str], embeddings: np.ndarray, indices: list[int]):
        """Cache embeddings for future use"""
        if not self.enable_cache or not self.redis_manager.is_connected:
            return

        try:
            cache_ttl = config.cache_ttl_hours * 3600

            for i, text_idx in enumerate(indices):
                text = texts[text_idx]
                embedding = embeddings[i]

                cache_key = self._generate_cache_key(text, self.embedding_model_name)
                cache_data = {
                    "embedding": embedding.tolist(),
                    "model_name": self.embedding_model_name,
                    "dimension": len(embedding),
                    "cached_at": time.time()
                }

                await self.redis_manager.set_json(cache_key, cache_data, ex=cache_ttl)

        except Exception as e:
            logger.warning(f"Embedding caching failed: {str(e)}")

    async def encode_texts(
        self,
        texts: Union[str, list[str]],
        batch_size: Union[int, None] = None,
        normalize_embeddings: bool = True
    ) -> dict[str, Any]:
        """
        Generate embeddings for input texts with caching and optimization
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Optional batch size override
            normalize_embeddings: Whether to normalize embeddings to unit vectors
            
        Returns:
            Dictionary with embeddings and metadata
        """
        if not self.is_initialized:
            return {
                "status": "error",
                "error": "Embedding service not initialized"
            }

        start_time = time.time()

        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        if not texts:
            return {
                "status": "error",
                "error": "No texts provided"
            }

        logger.info(f"🔢 Encoding {len(texts)} texts with {self.embedding_model_name}")

        try:
            # Check cache for existing embeddings
            cache_result = await self._get_cached_embeddings(texts)
            cached_embeddings = cache_result["cached"]
            missing_indices = cache_result["missing"]

            # Initialize results array
            all_embeddings = np.zeros((len(texts), self.embedding_dimension))

            # Fill in cached embeddings
            for idx, embedding in cached_embeddings.items():
                all_embeddings[idx] = embedding

            # Generate embeddings for missing texts
            if missing_indices:
                missing_texts = [texts[i] for i in missing_indices]

                # Get optimal batch size
                if batch_size is None:
                    batch_size = self.gpu_service.get_optimal_batch_size(self.batch_size)

                logger.info(f"🔄 Generating {len(missing_texts)} new embeddings (batch_size={batch_size})")

                # Generate embeddings in batches
                new_embeddings = await self._generate_embeddings_batch(
                    missing_texts,
                    batch_size,
                    normalize_embeddings
                )

                if new_embeddings["status"] != "success":
                    return new_embeddings

                # Fill in new embeddings
                for i, idx in enumerate(missing_indices):
                    all_embeddings[idx] = new_embeddings["embeddings"][i]

                # Cache new embeddings
                await self._cache_embeddings(texts, new_embeddings["embeddings"], missing_indices)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_embeddings += len(texts)
            self.total_processing_time += processing_time
            self.gpu_service.increment_operations()

            # Return results
            result = {
                "status": "success",
                "embeddings": all_embeddings[0] if single_text else all_embeddings,
                "dimension": self.embedding_dimension,
                "text_count": len(texts),
                "processing_time": round(processing_time, 3),
                "cache_hits": len(cached_embeddings),
                "cache_misses": len(missing_indices),
                "model_name": self.embedding_model_name,
                "device": str(self.device),
                "batch_size_used": batch_size if missing_indices else 0
            }

            logger.info(f"✅ Embeddings generated in {processing_time:.3f}s "
                       f"(cache hits: {len(cached_embeddings)}, misses: {len(missing_indices)})")

            return result

        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "text_count": len(texts)
            }

    async def _generate_embeddings_batch(
        self,
        texts: list[str],
        batch_size: int,
        normalize_embeddings: bool = True
    ) -> dict[str, Any]:
        """Generate embeddings in batches with GPU optimization"""
        try:
            all_embeddings = []

            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Check GPU memory before processing
                if self.gpu_service.is_gpu_available:
                    memory_info = self.gpu_service.monitor_gpu_usage()
                    if "torch_memory" in memory_info:
                        allocated_mb = memory_info["torch_memory"]["allocated_memory_mb"]
                        if allocated_mb > (self.gpu_service.total_memory * 0.9):
                            logger.warning("⚠️ GPU memory usage high, forcing cleanup")
                            self.gpu_service.cleanup_gpu_memory(force=True)

                # Generate embeddings for batch
                if self.mixed_precision and self.gpu_service.gpu_provider == "cuda":
                    # Use autocast for mixed precision inference
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        batch_embeddings = self.model.encode(
                            batch_texts,
                            convert_to_numpy=True,
                            normalize_embeddings=normalize_embeddings,
                            show_progress_bar=False
                        )
                else:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=False
                    )

                all_embeddings.append(batch_embeddings)

                # Log batch progress
                logger.debug(f"📦 Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

            # Combine all batches
            final_embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

            return {
                "status": "success",
                "embeddings": final_embeddings
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"Batch embedding generation failed: {str(e)}"
            }

    async def encode_single(self, text: str, normalize: bool = True) -> dict[str, Any]:
        """Convenience method for encoding single text"""
        result = await self.encode_texts(text, normalize_embeddings=normalize)
        if result["status"] == "success":
            result["embedding"] = result["embeddings"]  # Single embedding
        return result

    def get_embedding_stats(self) -> dict[str, Any]:
        """Get comprehensive embedding service statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_embeddings
            if self.total_embeddings > 0 else 0.0
        )

        cache_hit_rate = (
            (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        )

        stats = {
            "service_status": "initialized" if self.is_initialized else "not_initialized",
            "model_info": self.model_info,
            "performance": {
                "total_embeddings": self.total_embeddings,
                "total_processing_time": round(self.total_processing_time, 2),
                "average_processing_time": round(avg_processing_time, 4),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate_percent": round(cache_hit_rate, 2)
            },
            "configuration": {
                "model_name": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension,
                "max_sequence_length": self.max_length,
                "batch_size": self.batch_size,
                "max_batch_size": self.max_batch_size,
                "mixed_precision": self.mixed_precision,
                "cache_enabled": self.enable_cache
            },
            "device_info": self.gpu_service.get_device_info() if self.gpu_service else {},
            "cache_status": "connected" if (self.enable_cache and self.redis_manager.is_connected) else "disabled"
        }

        return stats

    async def clear_embedding_cache(self) -> dict[str, Any]:
        """Clear embedding cache"""
        if not self.enable_cache or not self.redis_manager.is_connected:
            return {"status": "disabled", "message": "Cache not available"}

        try:
            cleared_count = await self.redis_manager.clear_cache("embedding:*")
            return {
                "status": "success",
                "cleared_entries": cleared_count,
                "message": f"Cleared {cleared_count} embedding cache entries"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def get_model_info(self) -> dict[str, Any]:
        """Get detailed model information"""
        return {
            "is_initialized": self.is_initialized,
            "model_name": self.embedding_model_name,
            "model_info": self.model_info,
            "device": str(self.device) if self.device else None,
            "gpu_available": self.gpu_service.is_gpu_available,
            "mixed_precision": self.mixed_precision,
            "supports_languages": ["en", "ml", "hi", "ta", "te", "kn", "bn", "gu"],  # Common for multilingual models
            "optimized_for": "RTX_3050" if self.gpu_service.total_memory <= 4096 else "High_End_GPU"
        }
