"""
GPU Service for BajajFinsev Hybrid RAG System
RTX 3050 optimized GPU detection, configuration, and management
Handles device detection, memory management, and performance monitoring
"""

import logging
import time
import psutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio

# GPU libraries (with fallbacks)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_GPU_AVAILABLE = True
except (ImportError, Exception):
    NVIDIA_GPU_AVAILABLE = False
    pynvml = None

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information structure"""
    device_id: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    memory_used_gb: float
    compute_capability: Optional[Tuple[int, int]] = None
    temperature: Optional[float] = None
    power_draw: Optional[float] = None
    utilization: Optional[float] = None


@dataclass
class GPUConfig:
    """GPU configuration for RAG processing"""
    device: str
    batch_size: int
    memory_fraction: float
    mixed_precision: bool
    optimization_level: str


class GPUService:
    """
    GPU Service optimized for RTX 3050 (4GB VRAM)
    Provides device detection, memory management, and performance optimization
    """
    
    def __init__(self):
        """Initialize GPU service with RTX 3050 optimizations"""
        self.torch_available = TORCH_AVAILABLE
        self.nvidia_available = NVIDIA_GPU_AVAILABLE
        self.device_info: Optional[GPUInfo] = None
        self.config: Optional[GPUConfig] = None
        self.performance_stats = {
            'initialization_time': 0.0,
            'device_switches': 0,
            'memory_warnings': 0,
            'performance_optimizations': 0,
            'fallback_activations': 0
        }
        
        logger.info("🎮 GPU Service initializing...")
        logger.info(f"PyTorch available: {self.torch_available}")
        logger.info(f"NVIDIA GPU monitoring available: {self.nvidia_available}")
        
        # Initialize device detection
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize and detect optimal GPU device"""
        start_time = time.time()
        
        try:
            if not self.torch_available:
                logger.warning("⚠️ PyTorch not available, GPU acceleration disabled")
                self.config = GPUConfig(
                    device="cpu",
                    batch_size=8,
                    memory_fraction=0.0,
                    mixed_precision=False,
                    optimization_level="cpu_optimized"
                )
                return
            
            # Detect optimal device
            device = self._detect_optimal_device()
            
            # Configure for detected device
            if device.startswith("cuda"):
                self._configure_cuda_device(device)
            elif device == "mps":
                self._configure_mps_device()
            else:
                self._configure_cpu_device()
            
            self.performance_stats['initialization_time'] = time.time() - start_time
            
            logger.info(f"✅ GPU Service initialized in {self.performance_stats['initialization_time']:.2f}s")
            logger.info(f"Device: {self.config.device}")
            logger.info(f"Batch size: {self.config.batch_size}")
            logger.info(f"Mixed precision: {self.config.mixed_precision}")
            
        except Exception as e:
            logger.error(f"❌ GPU Service initialization failed: {str(e)}")
            # Fallback to CPU
            self.config = GPUConfig(
                device="cpu",
                batch_size=8,
                memory_fraction=0.0,
                mixed_precision=False,
                optimization_level="cpu_fallback"
            )
    
    def _detect_optimal_device(self) -> str:
        """Detect and return optimal compute device"""
        if not torch:
            return "cpu"
        
        # Check CUDA availability
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"🎮 Found {gpu_count} CUDA GPU(s)")
            
            # Get best GPU (most memory)
            best_gpu = 0
            best_memory = 0
            
            for i in range(gpu_count):
                try:
                    props = torch.cuda.get_device_properties(i)
                    memory_gb = props.total_memory / (1024**3)
                    gpu_name = props.name
                    
                    logger.info(f"  GPU {i}: {gpu_name} ({memory_gb:.1f}GB VRAM)")
                    
                    # Store device info for RTX 3050 or similar
                    if i == 0:  # Use first GPU for now
                        self.device_info = GPUInfo(
                            device_id=i,
                            name=gpu_name,
                            memory_total_gb=memory_gb,
                            memory_free_gb=memory_gb,  # Will be updated later
                            memory_used_gb=0.0,
                            compute_capability=(props.major, props.minor)
                        )
                    
                    if memory_gb > best_memory:
                        best_memory = memory_gb
                        best_gpu = i
                
                except Exception as e:
                    logger.warning(f"Failed to get properties for GPU {i}: {e}")
            
            return f"cuda:{best_gpu}"
        
        # Check Apple Silicon MPS
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("🍎 Using Apple M1/M2 GPU acceleration (MPS)")
            return "mps"
        
        else:
            logger.info("💻 No GPU available, using CPU with optimizations")
            return "cpu"
    
    def _configure_cuda_device(self, device: str):
        """Configure CUDA device with RTX 3050 optimizations"""
        gpu_id = int(device.split(":")[-1])
        
        try:
            # Get GPU memory info
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(gpu_id)
                total_memory_gb = props.total_memory / (1024**3)
                
                # RTX 3050 specific optimizations
                if total_memory_gb <= 4.5:  # RTX 3050 or similar low VRAM
                    batch_size = 16  # Conservative for 4GB VRAM
                    memory_fraction = 0.8  # Use 80% of VRAM
                    mixed_precision = True  # Essential for RTX 3050
                    optimization_level = "rtx_3050_optimized"
                    
                    logger.info("🎮 RTX 3050 (4GB) optimizations applied")
                    
                elif total_memory_gb <= 6.5:  # RTX 4060 Ti, RTX 3060 Ti
                    batch_size = 32
                    memory_fraction = 0.85
                    mixed_precision = True
                    optimization_level = "mid_tier_gpu"
                    
                elif total_memory_gb <= 8.5:  # RTX 4070, RTX 3080
                    batch_size = 48
                    memory_fraction = 0.9
                    mixed_precision = True
                    optimization_level = "high_tier_gpu"
                    
                else:  # RTX 4080, RTX 4090, etc.
                    batch_size = 64
                    memory_fraction = 0.9
                    mixed_precision = True
                    optimization_level = "enthusiast_gpu"
                
                # Update device info with current memory
                if self.device_info:
                    self.device_info.memory_free_gb = total_memory_gb * memory_fraction
                
                # Apply PyTorch CUDA optimizations
                torch.cuda.set_device(gpu_id)
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for speed
                
                self.config = GPUConfig(
                    device=device,
                    batch_size=batch_size,
                    memory_fraction=memory_fraction,
                    mixed_precision=mixed_precision,
                    optimization_level=optimization_level
                )
                
                logger.info(f"🎮 CUDA configured: {props.name} ({total_memory_gb:.1f}GB)")
                
            else:
                raise RuntimeError("CUDA not available")
                
        except Exception as e:
            logger.error(f"❌ CUDA configuration failed: {str(e)}")
            self._configure_cpu_device()
    
    def _configure_mps_device(self):
        """Configure Apple MPS device"""
        try:
            self.config = GPUConfig(
                device="mps",
                batch_size=32,  # MPS can handle larger batches
                memory_fraction=0.8,
                mixed_precision=True,
                optimization_level="apple_silicon"
            )
            
            logger.info("🍎 MPS (Apple Silicon) configured")
            
        except Exception as e:
            logger.error(f"❌ MPS configuration failed: {str(e)}")
            self._configure_cpu_device()
    
    def _configure_cpu_device(self):
        """Configure CPU device with optimizations"""
        # Get CPU info
        cpu_count = psutil.cpu_count(logical=False)  # Physical cores
        logical_count = psutil.cpu_count(logical=True)  # Logical cores
        
        # Optimize thread count for embeddings
        optimal_threads = min(8, logical_count)  # Cap at 8 threads
        
        if torch:
            torch.set_num_threads(optimal_threads)
        
        self.config = GPUConfig(
            device="cpu",
            batch_size=8,  # Smaller batches for CPU
            memory_fraction=0.0,  # Not applicable for CPU
            mixed_precision=False,  # Not beneficial for CPU
            optimization_level="cpu_optimized"
        )
        
        logger.info(f"💻 CPU configured: {cpu_count} cores, {optimal_threads} threads")
    
    async def get_gpu_stats(self) -> Dict[str, Any]:
        """Get comprehensive GPU statistics"""
        stats = {
            "torch_available": self.torch_available,
            "nvidia_monitoring_available": self.nvidia_available,
            "device": self.config.device if self.config else "unknown",
            "batch_size": self.config.batch_size if self.config else 0,
            "mixed_precision": self.config.mixed_precision if self.config else False,
            "optimization_level": self.config.optimization_level if self.config else "unknown",
            "performance_stats": self.performance_stats.copy()
        }
        
        # Add device-specific stats
        if self.config and self.config.device.startswith("cuda") and torch and torch.cuda.is_available():
            stats.update(await self._get_cuda_stats())
        elif self.device_info:
            stats["device_info"] = {
                "name": self.device_info.name,
                "memory_total_gb": self.device_info.memory_total_gb,
                "compute_capability": self.device_info.compute_capability
            }
        
        return stats
    
    async def _get_cuda_stats(self) -> Dict[str, Any]:
        """Get detailed CUDA statistics"""
        if not torch or not torch.cuda.is_available():
            return {}
        
        try:
            device_id = int(self.config.device.split(":")[-1])
            
            # PyTorch memory info
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
            memory_cached = torch.cuda.memory_cached(device_id) / (1024**3) if hasattr(torch.cuda, 'memory_cached') else 0
            
            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory / (1024**3)
            
            cuda_stats = {
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(total_memory, 2),
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_reserved_gb": round(memory_reserved, 2),
                "memory_cached_gb": round(memory_cached, 2),
                "memory_free_gb": round(total_memory - memory_reserved, 2),
                "memory_utilization_percent": round((memory_reserved / total_memory) * 100, 1)
            }
            
            # Add NVIDIA GPU monitoring if available
            if self.nvidia_available and pynvml:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    cuda_stats["temperature_c"] = temp
                    
                    # Power
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Watts
                    cuda_stats["power_draw_watts"] = round(power, 1)
                    
                    # Utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    cuda_stats["gpu_utilization_percent"] = util.gpu
                    cuda_stats["memory_controller_utilization_percent"] = util.memory
                    
                except Exception as e:
                    logger.debug(f"NVIDIA monitoring failed: {e}")
            
            return cuda_stats
            
        except Exception as e:
            logger.error(f"Failed to get CUDA stats: {e}")
            return {}
    
    def optimize_for_batch_size(self, target_batch_size: int) -> int:
        """
        Optimize batch size based on available GPU memory
        Returns adjusted batch size that fits in memory
        """
        if not self.config:
            return target_batch_size
        
        # For RTX 3050 (4GB VRAM), be conservative
        if self.config.optimization_level == "rtx_3050_optimized":
            max_safe_batch = 16
            adjusted_batch = min(target_batch_size, max_safe_batch)
            
            if adjusted_batch < target_batch_size:
                logger.info(f"🎮 RTX 3050: Batch size reduced {target_batch_size} → {adjusted_batch}")
                self.performance_stats['performance_optimizations'] += 1
            
            return adjusted_batch
        
        # For other devices, use configured batch size as limit
        max_batch = self.config.batch_size
        return min(target_batch_size, max_batch)
    
    async def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage"""
        if not self.config or not self.config.device.startswith("cuda"):
            return {"cpu_memory_percent": psutil.virtual_memory().percent}
        
        if not torch or not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        try:
            device_id = int(self.config.device.split(":")[-1])
            
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory / (1024**3)
            
            usage_percent = (memory_reserved / total_memory) * 100
            
            # Warning for RTX 3050 if usage is high
            if self.config.optimization_level == "rtx_3050_optimized" and usage_percent > 90:
                logger.warning(f"⚠️ RTX 3050 memory usage high: {usage_percent:.1f}%")
                self.performance_stats['memory_warnings'] += 1
            
            return {
                "gpu_memory_allocated_gb": round(memory_allocated, 2),
                "gpu_memory_reserved_gb": round(memory_reserved, 2),
                "gpu_memory_total_gb": round(total_memory, 2),
                "gpu_memory_utilization_percent": round(usage_percent, 1),
                "cpu_memory_percent": psutil.virtual_memory().percent
            }
            
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            return {"error": str(e)}
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache"""
        if self.config and self.config.device.startswith("cuda") and torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.debug("🧹 GPU cache cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {e}")
    
    def get_device_config(self) -> Optional[GPUConfig]:
        """Get current device configuration"""
        return self.config
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and configured"""
        return (
            self.config is not None and 
            self.config.device != "cpu" and
            self.torch_available
        )
    
    def get_recommended_settings(self) -> Dict[str, Any]:
        """Get recommended settings for current GPU"""
        if not self.config:
            return {}
        
        settings = {
            "batch_size": self.config.batch_size,
            "mixed_precision": self.config.mixed_precision,
            "memory_fraction": self.config.memory_fraction,
            "optimization_level": self.config.optimization_level
        }
        
        # Add device-specific recommendations
        if self.config.optimization_level == "rtx_3050_optimized":
            settings.update({
                "chunk_size": 256,  # Smaller chunks for limited VRAM
                "max_sequence_length": 512,
                "gradient_checkpointing": True,
                "pin_memory": True,
                "prefetch_factor": 2
            })
        
        return settings


# Global GPU service instance
gpu_service: Optional[GPUService] = None


def get_gpu_service() -> GPUService:
    """Get or create global GPU service instance"""
    global gpu_service
    
    if gpu_service is None:
        gpu_service = GPUService()
    
    return gpu_service


async def initialize_gpu_service() -> GPUService:
    """Initialize and return GPU service"""
    service = get_gpu_service()
    
    # Log initialization summary
    stats = await service.get_gpu_stats()
    logger.info("🎮 GPU Service Summary:")
    logger.info(f"  Device: {stats['device']}")
    logger.info(f"  Batch Size: {stats['batch_size']}")
    logger.info(f"  Mixed Precision: {stats['mixed_precision']}")
    logger.info(f"  Optimization: {stats['optimization_level']}")
    
    return service