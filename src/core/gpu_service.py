"""
GPU Service with Centralized Configuration
Handles GPU detection, memory management, and RTX 3050 optimization
"""

import gc
import logging
import time
from typing import Any, Union

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

from src.core.config import config

logger = logging.getLogger(__name__)


class GPUService:
    """
    GPU service with RTX 3050 optimization and centralized configuration
    Handles device detection, memory management, and performance monitoring
    """

    def __init__(self):
        # Configuration from central config
        self.gpu_provider = config.gpu_provider
        self.memory_fraction = config.gpu_memory_fraction
        self.batch_size = config.batch_size
        self.max_batch_size = config.max_batch_size
        self.mixed_precision = config.enable_mixed_precision
        self.cleanup_interval = config.gpu_memory_cleanup_interval
        self.skip_gpu_check = config.skip_gpu_check

        # GPU state
        self.device = None
        self.device_name = None
        self.total_memory = 0
        self.available_memory = 0
        self.is_gpu_available = False
        self.operations_count = 0

        logger.info(f"GPUService initialized for provider: {self.gpu_provider}")

    def initialize(self) -> dict[str, Any]:
        """Initialize GPU service and detect available devices"""
        try:
            logger.info("🔄 Initializing GPU Service...")

            if self.skip_gpu_check:
                logger.info("⏭️ Skipping GPU check (configured)")
                return self._setup_cpu_fallback()

            if not HAS_TORCH:
                logger.warning("❌ PyTorch not available, falling back to CPU")
                return self._setup_cpu_fallback()

            # Detect and configure GPU
            gpu_info = self._detect_gpu()

            if gpu_info["status"] == "success":
                self._configure_gpu_memory()
                logger.info(f"✅ GPU Service initialized: {self.device_name}")
            else:
                logger.warning(f"⚠️ GPU initialization failed: {gpu_info['message']}")
                return self._setup_cpu_fallback()

            return gpu_info

        except Exception as e:
            logger.error(f"❌ GPU Service initialization failed: {str(e)}")
            return self._setup_cpu_fallback()

    def _detect_gpu(self) -> dict[str, Any]:
        """Detect available GPU and configure based on provider"""

        if self.gpu_provider == "cpu":
            return self._setup_cpu_fallback()

        try:
            # Check CUDA availability
            if self.gpu_provider == "cuda":
                if not torch.cuda.is_available():
                    return {
                        "status": "fallback",
                        "message": "CUDA not available, falling back to CPU",
                        "device": "cpu"
                    }

                # Get GPU information
                gpu_count = torch.cuda.device_count()
                if gpu_count == 0:
                    return {
                        "status": "fallback",
                        "message": "No CUDA devices found",
                        "device": "cpu"
                    }

                # Use first GPU (typically GPU 0)
                self.device = torch.device("cuda:0")
                self.device_name = torch.cuda.get_device_name(0)
                self.is_gpu_available = True

                # Get memory info
                if HAS_PYNVML:
                    memory_info = self._get_gpu_memory_info()
                    self.total_memory = memory_info.get("total_memory_mb", 0)
                    self.available_memory = memory_info.get("available_memory_mb", 0)
                else:
                    # Fallback memory detection
                    self.total_memory = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                    torch.cuda.empty_cache()
                    self.available_memory = self.total_memory

                return {
                    "status": "success",
                    "message": f"CUDA GPU detected: {self.device_name}",
                    "device": str(self.device),
                    "device_name": self.device_name,
                    "total_memory_mb": self.total_memory,
                    "available_memory_mb": self.available_memory,
                    "gpu_count": gpu_count
                }

            # Check MPS (Apple Silicon) availability
            elif self.gpu_provider == "mps":
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    self.device_name = "Apple Silicon GPU (MPS)"
                    self.is_gpu_available = True

                    # MPS doesn't provide direct memory info
                    self.total_memory = 8192  # Assume 8GB shared memory
                    self.available_memory = self.total_memory

                    return {
                        "status": "success",
                        "message": "MPS GPU detected",
                        "device": str(self.device),
                        "device_name": self.device_name,
                        "total_memory_mb": self.total_memory,
                        "available_memory_mb": self.available_memory
                    }
                else:
                    return {
                        "status": "fallback",
                        "message": "MPS not available, falling back to CPU",
                        "device": "cpu"
                    }

            else:
                return {
                    "status": "fallback",
                    "message": f"Unknown GPU provider: {self.gpu_provider}",
                    "device": "cpu"
                }

        except Exception as e:
            logger.error(f"GPU detection failed: {str(e)}")
            return {
                "status": "fallback",
                "message": f"GPU detection error: {str(e)}",
                "device": "cpu"
            }

    def _setup_cpu_fallback(self) -> dict[str, Any]:
        """Setup CPU fallback configuration"""
        self.device = torch.device("cpu") if HAS_TORCH else None
        self.device_name = "CPU"
        self.is_gpu_available = False
        self.total_memory = 0
        self.available_memory = 0

        logger.info("🖥️ Using CPU for processing")

        return {
            "status": "success",
            "message": "CPU fallback configured",
            "device": "cpu",
            "device_name": "CPU",
            "is_gpu": False
        }

    def _configure_gpu_memory(self):
        """Configure GPU memory settings for RTX 3050 optimization"""
        if not self.is_gpu_available or self.gpu_provider != "cuda":
            return

        try:
            # Set memory fraction for RTX 3050 (4GB VRAM)
            if self.total_memory <= 4096:  # RTX 3050 or similar
                logger.info("🎯 Configuring for RTX 3050 (4GB VRAM)")
                # Use configured memory fraction (default 80% = 3.2GB)
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                logger.info(f"📊 GPU memory fraction set to {self.memory_fraction} ({self.memory_fraction * self.total_memory:.0f}MB)")

            # Enable memory caching for efficiency
            torch.cuda.empty_cache()

            # Configure for mixed precision if enabled
            if self.mixed_precision:
                logger.info("🔄 Mixed precision (FP16) enabled for memory efficiency")

        except Exception as e:
            logger.warning(f"GPU memory configuration failed: {str(e)}")

    def _get_gpu_memory_info(self) -> dict[str, Any]:
        """Get detailed GPU memory information using pynvml"""
        if not HAS_PYNVML or not self.is_gpu_available:
            return {}

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory = memory_info.total // (1024 * 1024)  # Convert to MB
            used_memory = memory_info.used // (1024 * 1024)
            available_memory = total_memory - used_memory

            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0

            return {
                "total_memory_mb": total_memory,
                "used_memory_mb": used_memory,
                "available_memory_mb": available_memory,
                "memory_utilization": (used_memory / total_memory) * 100,
                "gpu_utilization": util.gpu,
                "memory_utilization_percent": util.memory,
                "temperature_celsius": temp
            }

        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {str(e)}")
            return {}

    def get_optimal_batch_size(self, base_size: Union[int, None] = None) -> int:
        """Get optimal batch size based on available GPU memory"""
        if not self.is_gpu_available:
            return min(self.batch_size, 8)  # Smaller batches for CPU

        if base_size is None:
            base_size = self.batch_size

        try:
            # Get current memory usage
            if self.gpu_provider == "cuda" and HAS_PYNVML:
                memory_info = self._get_gpu_memory_info()
                available_mb = memory_info.get("available_memory_mb", 0)

                # RTX 3050 optimization - Increased batch sizes for faster processing
                if self.total_memory <= 4096:  # RTX 3050
                    if available_mb > 2000:  # Plenty of memory
                        return min(self.max_batch_size, 64)  # Doubled from 32
                    elif available_mb > 1000:  # Moderate memory
                        return min(base_size * 2, 48)  # Doubled from 24
                    else:  # Low memory
                        return max(base_size // 2, 16)  # Doubled from 8

                # For other GPUs, use standard logic
                if available_mb > 6000:
                    return min(self.max_batch_size, 64)
                elif available_mb > 3000:
                    return min(base_size * 2, 48)
                else:
                    return max(base_size, 16)

            # Fallback to configured batch size
            return base_size

        except Exception as e:
            logger.warning(f"Batch size optimization failed: {str(e)}")
            return base_size

    def monitor_gpu_usage(self) -> dict[str, Any]:
        """Monitor current GPU usage and performance"""
        if not self.is_gpu_available:
            return {
                "device": "cpu",
                "message": "GPU monitoring not available"
            }

        try:
            if self.gpu_provider == "cuda":
                # PyTorch memory stats
                torch_stats = {
                    "allocated_memory_mb": torch.cuda.memory_allocated() // (1024 * 1024),
                    "cached_memory_mb": torch.cuda.memory_reserved() // (1024 * 1024),
                    "max_allocated_mb": torch.cuda.max_memory_allocated() // (1024 * 1024)
                }

                # NVML stats if available
                nvml_stats = self._get_gpu_memory_info()

                return {
                    "device": str(self.device),
                    "device_name": self.device_name,
                    "torch_memory": torch_stats,
                    "system_memory": nvml_stats,
                    "operations_count": self.operations_count,
                    "timestamp": time.time()
                }

            elif self.gpu_provider == "mps":
                return {
                    "device": str(self.device),
                    "device_name": self.device_name,
                    "message": "MPS monitoring limited",
                    "operations_count": self.operations_count,
                    "timestamp": time.time()
                }

        except Exception as e:
            logger.warning(f"GPU monitoring failed: {str(e)}")
            return {
                "device": str(self.device),
                "error": str(e),
                "timestamp": time.time()
            }

    def cleanup_gpu_memory(self, force: bool = False):
        """Clean up GPU memory and run garbage collection"""
        try:
            if force or (self.operations_count % self.cleanup_interval == 0):
                if self.is_gpu_available and self.gpu_provider == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Run Python garbage collection
                gc.collect()

                logger.debug(f"🧹 GPU memory cleanup completed (operations: {self.operations_count})")

        except Exception as e:
            logger.warning(f"GPU memory cleanup failed: {str(e)}")

    def increment_operations(self):
        """Increment operations counter and trigger cleanup if needed"""
        self.operations_count += 1
        self.cleanup_gpu_memory()

    def get_device_info(self) -> dict[str, Any]:
        """Get comprehensive device information"""
        info = {
            "device": str(self.device) if self.device else "none",
            "device_name": self.device_name,
            "is_gpu_available": self.is_gpu_available,
            "gpu_provider": self.gpu_provider,
            "total_memory_mb": self.total_memory,
            "configured_memory_fraction": self.memory_fraction,
            "batch_size": self.batch_size,
            "max_batch_size": self.max_batch_size,
            "mixed_precision_enabled": self.mixed_precision,
            "operations_count": self.operations_count,
            "cleanup_interval": self.cleanup_interval
        }

        # Add current usage if available
        if self.is_gpu_available:
            usage = self.monitor_gpu_usage()
            if "torch_memory" in usage:
                info["current_usage"] = usage["torch_memory"]
            if "system_memory" in usage:
                info["system_memory"] = usage["system_memory"]

        return info

    def is_memory_available(self, required_mb: int) -> bool:
        """Check if sufficient GPU memory is available"""
        if not self.is_gpu_available:
            return True  # CPU has different memory constraints

        try:
            if self.gpu_provider == "cuda" and HAS_PYNVML:
                memory_info = self._get_gpu_memory_info()
                available_mb = memory_info.get("available_memory_mb", 0)
                return available_mb >= required_mb

            # Fallback check using PyTorch
            if self.gpu_provider == "cuda":
                allocated_mb = torch.cuda.memory_allocated() // (1024 * 1024)
                total_allowed_mb = self.total_memory * self.memory_fraction
                return (total_allowed_mb - allocated_mb) >= required_mb

            return True  # For MPS and other providers

        except Exception as e:
            logger.warning(f"Memory availability check failed: {str(e)}")
            return False

    def get_recommended_settings(self) -> dict[str, Any]:
        """Get recommended settings based on detected hardware"""
        if not self.is_gpu_available:
            return {
                "device": "cpu",
                "recommended_batch_size": 4,
                "mixed_precision": False,
                "max_sequence_length": 256,
                "note": "CPU processing - use smaller batches"
            }

        # RTX 3050 specific recommendations
        if self.total_memory <= 4096:  # RTX 3050 or similar
            return {
                "device": str(self.device),
                "device_name": self.device_name,
                "recommended_batch_size": self.get_optimal_batch_size(),
                "mixed_precision": True,
                "max_sequence_length": 512,
                "memory_fraction": 0.8,
                "note": "RTX 3050 optimized settings",
                "model_recommendations": {
                    "embedding_model": "intfloat/multilingual-e5-base",
                    "max_model_size": "1GB",
                    "concurrent_operations": "limited"
                }
            }

        # Higher-end GPU recommendations
        else:
            return {
                "device": str(self.device),
                "device_name": self.device_name,
                "recommended_batch_size": min(self.max_batch_size, 64),
                "mixed_precision": True,
                "max_sequence_length": 512,
                "memory_fraction": 0.9,
                "note": "High-performance GPU settings",
                "model_recommendations": {
                    "embedding_model": "intfloat/multilingual-e5-large",
                    "max_model_size": "2GB",
                    "concurrent_operations": "full"
                }
            }
