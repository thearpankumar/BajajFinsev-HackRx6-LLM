"""
Advanced GPU Memory Manager
Enhanced GPU management with memory optimization, profiling, and dynamic scaling
"""

import gc
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psutil

try:
    import nvidia_ml_py3 as nvml
    import torch
    HAS_TORCH = True
    HAS_NVML = True
except ImportError:
    HAS_TORCH = False
    HAS_NVML = False

from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory snapshot data"""
    timestamp: float
    gpu_memory_used: int
    gpu_memory_total: int
    gpu_memory_cached: int
    gpu_utilization: float
    cpu_memory_percent: float
    active_tensors: int
    memory_efficiency: float


@dataclass
class PerformanceProfile:
    """Performance profiling data"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    peak_memory_mb: int
    memory_growth_mb: int
    gpu_utilization_avg: float
    batch_size: int
    throughput: float


class AdvancedGPUManager:
    """
    Advanced GPU memory manager with profiling, optimization, and dynamic scaling
    Specifically optimized for RTX 3050 4GB constraints
    """

    def __init__(self):
        # Configuration
        # Use proper config access with default instead of getattr
        self.memory_limit_mb = config.gpu_memory_limit_mb if hasattr(config, 'gpu_memory_limit_mb') else 3500  # Reserve 500MB for system
        self.memory_warning_threshold = 0.85  # 85% of available memory
        self.memory_critical_threshold = 0.95  # 95% of available memory
        self.cleanup_frequency = 50  # Operations between cleanups
        self.profiling_enabled = config.enable_gpu_profiling if hasattr(config, 'enable_gpu_profiling') else True

        # State tracking
        self.device = None
        self.is_initialized = False
        self.operations_count = 0
        self.last_cleanup_time = time.time()

        # Memory tracking
        self.memory_snapshots: list[MemorySnapshot] = []
        self.max_snapshots = 1000

        # Performance profiling
        self.performance_profiles: list[PerformanceProfile] = []
        self.max_profiles = 500
        self.active_profiles: dict[str, dict[str, Any]] = {}

        # Optimization parameters
        self.optimal_batch_sizes: dict[str, int] = {}
        self.memory_usage_patterns: dict[str, list[int]] = {}

        # RTX 3050 specific optimizations
        self.rtx_3050_optimizations = {
            'enable_memory_pool': True,
            'use_mixed_precision': True,
            'enable_gradient_checkpointing': True,
            'optimize_for_inference': True,
            'chunk_large_operations': True
        }

        logger.info("AdvancedGPUManager initialized with RTX 3050 optimizations")

    async def initialize(self) -> dict[str, Any]:
        """Initialize advanced GPU manager"""
        try:
            logger.info("🔄 Initializing Advanced GPU Manager...")
            start_time = time.time()

            # Initialize NVML for detailed GPU monitoring
            if HAS_NVML:
                try:
                    nvml.nvmlInit()
                    self.nvml_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    logger.info("✅ NVML initialized for detailed GPU monitoring")
                except Exception as e:
                    logger.warning(f"NVML initialization failed: {str(e)}")
                    self.nvml_handle = None

            # Initialize PyTorch if available
            if HAS_TORCH:
                # Detect device
                if torch.cuda.is_available():
                    self.device = torch.device("cuda:0")
                    device_name = torch.cuda.get_device_name(0)

                    # RTX 3050 specific setup
                    if "RTX 3050" in device_name or "3050" in device_name:
                        await self._setup_rtx_3050_optimizations()

                    # Set memory fraction
                    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                        memory_fraction = min(0.9, self.memory_limit_mb / 4096)  # 4GB total for RTX 3050
                        torch.cuda.set_per_process_memory_fraction(memory_fraction)
                        logger.info(f"🎯 GPU memory fraction set to {memory_fraction:.2f}")

                elif torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                    device_name = "Apple Metal Performance Shaders"
                else:
                    self.device = torch.device("cpu")
                    device_name = "CPU"
            else:
                self.device = "cpu"
                device_name = "CPU (PyTorch not available)"

            # Initial memory snapshot
            await self._take_memory_snapshot("initialization")

            self.is_initialized = True
            initialization_time = time.time() - start_time

            result = {
                "status": "success",
                "message": f"Advanced GPU Manager initialized in {initialization_time:.2f}s",
                "device": str(self.device),
                "device_name": device_name,
                "memory_limit_mb": self.memory_limit_mb,
                "rtx_3050_optimized": "3050" in device_name,
                "nvml_available": HAS_NVML and hasattr(self, 'nvml_handle'),
                "profiling_enabled": self.profiling_enabled,
                "initialization_time": initialization_time
            }

            logger.info(f"✅ Advanced GPU Manager ready: {device_name}")
            return result

        except Exception as e:
            error_msg = f"Advanced GPU Manager initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }

    async def _setup_rtx_3050_optimizations(self):
        """Setup RTX 3050 specific optimizations"""
        try:
            logger.info("🎯 Applying RTX 3050 specific optimizations...")

            # Enable memory pool for better allocation
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.empty_cache()
                torch.cuda.set_per_process_memory_fraction(0.85)  # Leave some headroom

            # Enable cuDNN benchmarking for consistent workloads
            if hasattr(torch.backends.cudnn, 'benchmark'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False

            # Set optimal memory growth
            if hasattr(torch.cuda, 'memory_stats'):
                # Enable memory pool
                torch.cuda.empty_cache()

            logger.info("✅ RTX 3050 optimizations applied")

        except Exception as e:
            logger.warning(f"RTX 3050 optimization setup failed: {str(e)}")

    @contextmanager
    def profile_operation(self, operation_name: str, batch_size: int = 1):
        """Context manager for profiling GPU operations"""
        if not self.profiling_enabled:
            yield
            return

        profile_id = f"{operation_name}_{int(time.time())}"
        start_time = time.time()
        start_memory = self._get_gpu_memory_usage()

        # Store operation start info
        self.active_profiles[profile_id] = {
            "operation_name": operation_name,
            "start_time": start_time,
            "start_memory": start_memory,
            "batch_size": batch_size,
            "peak_memory": start_memory
        }

        try:
            yield profile_id
        finally:
            # Complete profiling
            end_time = time.time()
            end_memory = self._get_gpu_memory_usage()

            if profile_id in self.active_profiles:
                profile_data = self.active_profiles[profile_id]

                # Calculate metrics
                duration = end_time - start_time
                memory_growth = end_memory - profile_data["start_memory"]
                throughput = batch_size / duration if duration > 0 else 0

                # Create performance profile
                profile = PerformanceProfile(
                    operation_name=operation_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    peak_memory_mb=profile_data["peak_memory"],
                    memory_growth_mb=memory_growth,
                    gpu_utilization_avg=self._get_gpu_utilization(),
                    batch_size=batch_size,
                    throughput=throughput
                )

                # Store profile
                self.performance_profiles.append(profile)

                # Limit profile history
                if len(self.performance_profiles) > self.max_profiles:
                    self.performance_profiles = self.performance_profiles[-self.max_profiles//2:]

                # Update optimal batch sizes
                await self._update_optimal_batch_size(operation_name, batch_size, throughput, memory_growth)

                # Clean up
                del self.active_profiles[profile_id]

    async def _update_optimal_batch_size(self, operation: str, batch_size: int, throughput: float, memory_growth: int):
        """Update optimal batch size based on performance data"""
        try:
            if operation not in self.optimal_batch_sizes:
                self.optimal_batch_sizes[operation] = batch_size
                self.memory_usage_patterns[operation] = [memory_growth]
                return

            # Get recent performance data for this operation
            recent_profiles = [p for p in self.performance_profiles[-20:] if p.operation_name == operation]

            if len(recent_profiles) >= 3:
                # Calculate efficiency score (throughput / memory_growth)
                current_efficiency = throughput / max(1, memory_growth)

                # Find best efficiency batch size
                best_batch_size = batch_size
                best_efficiency = current_efficiency

                for profile in recent_profiles:
                    profile_efficiency = profile.throughput / max(1, profile.memory_growth_mb)
                    if profile_efficiency > best_efficiency and profile.peak_memory_mb < self.memory_limit_mb * 0.8:
                        best_efficiency = profile_efficiency
                        best_batch_size = profile.batch_size

                self.optimal_batch_sizes[operation] = best_batch_size

            # Update memory usage patterns
            self.memory_usage_patterns[operation].append(memory_growth)
            if len(self.memory_usage_patterns[operation]) > 50:
                self.memory_usage_patterns[operation] = self.memory_usage_patterns[operation][-25:]

        except Exception as e:
            logger.warning(f"Optimal batch size update failed: {str(e)}")

    def get_optimal_batch_size(self, operation: str, default_batch_size: int) -> int:
        """Get optimal batch size for operation"""
        if operation in self.optimal_batch_sizes:
            optimal_size = self.optimal_batch_sizes[operation]

            # Check current memory availability
            current_memory_usage = self._get_gpu_memory_usage()
            memory_available = self.memory_limit_mb - current_memory_usage

            # Scale down if memory is limited
            if memory_available < self.memory_limit_mb * 0.3:  # Less than 30% available
                optimal_size = max(1, optimal_size // 2)
            elif memory_available < self.memory_limit_mb * 0.5:  # Less than 50% available
                optimal_size = max(1, int(optimal_size * 0.7))

            return min(optimal_size, default_batch_size * 2)  # Don't exceed 2x default

        return default_batch_size

    async def _take_memory_snapshot(self, context: str = "general"):
        """Take a memory snapshot for monitoring"""
        try:
            if not HAS_TORCH or self.device == "cpu":
                return

            # GPU memory stats
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() // (1024 ** 2)  # MB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
                gpu_memory_cached = torch.cuda.memory_reserved() // (1024 ** 2)
            else:
                gpu_memory_used = 0
                gpu_memory_total = 0
                gpu_memory_cached = 0

            # GPU utilization
            gpu_utilization = self._get_gpu_utilization()

            # CPU memory
            cpu_memory_percent = psutil.virtual_memory().percent

            # Active tensors (approximation)
            active_tensors = len([obj for obj in gc.get_objects() if torch.is_tensor(obj)])

            # Memory efficiency
            memory_efficiency = (gpu_memory_used / max(1, gpu_memory_cached)) * 100

            snapshot = MemorySnapshot(
                timestamp=time.time(),
                gpu_memory_used=gpu_memory_used,
                gpu_memory_total=gpu_memory_total,
                gpu_memory_cached=gpu_memory_cached,
                gpu_utilization=gpu_utilization,
                cpu_memory_percent=cpu_memory_percent,
                active_tensors=active_tensors,
                memory_efficiency=memory_efficiency
            )

            self.memory_snapshots.append(snapshot)

            # Limit snapshot history
            if len(self.memory_snapshots) > self.max_snapshots:
                self.memory_snapshots = self.memory_snapshots[-self.max_snapshots//2:]

            # Check for memory warnings
            memory_usage_ratio = gpu_memory_used / max(1, self.memory_limit_mb)

            if memory_usage_ratio > self.memory_critical_threshold:
                logger.error(f"🚨 CRITICAL GPU memory usage: {gpu_memory_used}MB ({memory_usage_ratio:.1%})")
                await self.emergency_cleanup()
            elif memory_usage_ratio > self.memory_warning_threshold:
                logger.warning(f"⚠️ High GPU memory usage: {gpu_memory_used}MB ({memory_usage_ratio:.1%})")
                await self.smart_cleanup()

        except Exception as e:
            logger.warning(f"Memory snapshot failed: {str(e)}")

    def _get_gpu_memory_usage(self) -> int:
        """Get current GPU memory usage in MB"""
        try:
            if HAS_TORCH and torch.cuda.is_available():
                return torch.cuda.memory_allocated() // (1024 ** 2)
            return 0
        except Exception:
            return 0

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage"""
        try:
            if HAS_NVML and hasattr(self, 'nvml_handle') and self.nvml_handle:
                utilization = nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                return utilization.gpu
            return 0.0
        except Exception:
            return 0.0

    async def smart_cleanup(self):
        """Smart GPU memory cleanup"""
        try:
            logger.info("🧹 Starting smart GPU cleanup...")

            if not HAS_TORCH or self.device == "cpu":
                return

            initial_memory = self._get_gpu_memory_usage()

            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Python garbage collection
            gc.collect()

            # Clear any cached computations
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()

            final_memory = self._get_gpu_memory_usage()
            freed_mb = initial_memory - final_memory

            logger.info(f"✅ Smart cleanup freed {freed_mb}MB GPU memory")

        except Exception as e:
            logger.warning(f"Smart cleanup failed: {str(e)}")

    async def emergency_cleanup(self):
        """Emergency GPU memory cleanup for critical situations"""
        try:
            logger.error("🚨 Starting emergency GPU cleanup...")

            if not HAS_TORCH or self.device == "cpu":
                return

            initial_memory = self._get_gpu_memory_usage()

            # Aggressive cleanup
            if torch.cuda.is_available():
                # Clear all caches
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                # Reset memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

            # Multiple garbage collection rounds
            for _ in range(3):
                gc.collect()
                time.sleep(0.1)

            final_memory = self._get_gpu_memory_usage()
            freed_mb = initial_memory - final_memory

            logger.error(f"🚨 Emergency cleanup freed {freed_mb}MB GPU memory")

            # Update last cleanup time
            self.last_cleanup_time = time.time()

        except Exception as e:
            logger.error(f"Emergency cleanup failed: {str(e)}")

    async def periodic_maintenance(self):
        """Periodic maintenance and optimization"""
        try:
            self.operations_count += 1

            # Take memory snapshot
            await self._take_memory_snapshot("periodic")

            # Periodic cleanup
            if self.operations_count % self.cleanup_frequency == 0:
                await self.smart_cleanup()

            # Memory pressure check
            current_memory = self._get_gpu_memory_usage()
            memory_ratio = current_memory / max(1, self.memory_limit_mb)

            if memory_ratio > 0.7:  # 70% usage
                await self.smart_cleanup()

        except Exception as e:
            logger.warning(f"Periodic maintenance failed: {str(e)}")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            current_memory = self._get_gpu_memory_usage()

            if self.memory_snapshots:
                recent_snapshots = self.memory_snapshots[-10:]
                avg_memory = sum(s.gpu_memory_used for s in recent_snapshots) / len(recent_snapshots)
                peak_memory = max(s.gpu_memory_used for s in recent_snapshots)
                memory_trend = "stable"

                if len(recent_snapshots) >= 5:
                    recent_avg = sum(s.gpu_memory_used for s in recent_snapshots[-3:]) / 3
                    older_avg = sum(s.gpu_memory_used for s in recent_snapshots[:3]) / 3

                    if recent_avg > older_avg * 1.2:
                        memory_trend = "increasing"
                    elif recent_avg < older_avg * 0.8:
                        memory_trend = "decreasing"
            else:
                avg_memory = current_memory
                peak_memory = current_memory
                memory_trend = "unknown"

            return {
                "current_memory_mb": current_memory,
                "memory_limit_mb": self.memory_limit_mb,
                "memory_usage_percent": (current_memory / max(1, self.memory_limit_mb)) * 100,
                "average_memory_mb": round(avg_memory, 1),
                "peak_memory_mb": peak_memory,
                "memory_trend": memory_trend,
                "total_snapshots": len(self.memory_snapshots),
                "gpu_utilization": self._get_gpu_utilization(),
                "operations_count": self.operations_count,
                "last_cleanup": time.time() - self.last_cleanup_time
            }

        except Exception as e:
            logger.warning(f"Memory stats collection failed: {str(e)}")
            return {"error": str(e)}

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance profiling statistics"""
        try:
            if not self.performance_profiles:
                return {"message": "No performance data available"}

            # Aggregate performance data
            operations = {}
            for profile in self.performance_profiles[-100:]:  # Last 100 operations
                op_name = profile.operation_name
                if op_name not in operations:
                    operations[op_name] = []
                operations[op_name].append(profile)

            # Calculate statistics per operation
            operation_stats = {}
            for op_name, profiles in operations.items():
                if profiles:
                    durations = [p.duration for p in profiles]
                    throughputs = [p.throughput for p in profiles]
                    memory_growths = [p.memory_growth_mb for p in profiles]

                    operation_stats[op_name] = {
                        "count": len(profiles),
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "avg_throughput": sum(throughputs) / len(throughputs),
                        "avg_memory_growth": sum(memory_growths) / len(memory_growths),
                        "optimal_batch_size": self.optimal_batch_sizes.get(op_name, "unknown")
                    }

            return {
                "total_profiles": len(self.performance_profiles),
                "operations_profiled": len(operations),
                "operation_statistics": operation_stats,
                "profiling_enabled": self.profiling_enabled
            }

        except Exception as e:
            logger.warning(f"Performance stats collection failed: {str(e)}")
            return {"error": str(e)}

    def get_optimization_recommendations(self) -> list[str]:
        """Get optimization recommendations based on profiling data"""
        recommendations = []

        try:
            # Memory usage recommendations
            current_memory = self._get_gpu_memory_usage()
            memory_ratio = current_memory / max(1, self.memory_limit_mb)

            if memory_ratio > 0.8:
                recommendations.append("Consider reducing batch sizes to prevent OOM errors")
                recommendations.append("Enable gradient checkpointing for memory efficiency")

            if memory_ratio < 0.3:
                recommendations.append("GPU memory underutilized - consider increasing batch sizes")

            # Performance recommendations
            if self.performance_profiles:
                recent_profiles = self.performance_profiles[-50:]
                avg_gpu_util = sum(p.gpu_utilization_avg for p in recent_profiles) / len(recent_profiles)

                if avg_gpu_util < 50:
                    recommendations.append("Low GPU utilization - check for CPU bottlenecks")
                    recommendations.append("Consider increasing computational intensity per batch")

                # Check for memory leaks
                memory_growths = [p.memory_growth_mb for p in recent_profiles]
                avg_growth = sum(memory_growths) / len(memory_growths)

                if avg_growth > 10:  # Growing more than 10MB per operation
                    recommendations.append("Potential memory leak detected - review tensor cleanup")

            # RTX 3050 specific recommendations
            if "3050" in str(self.device):
                recommendations.append("RTX 3050 detected - ensure mixed precision is enabled")
                recommendations.append("Consider using gradient accumulation for larger effective batch sizes")

            if not recommendations:
                recommendations.append("System is running optimally")

        except Exception as e:
            logger.warning(f"Optimization recommendations failed: {str(e)}")
            recommendations.append(f"Unable to generate recommendations: {str(e)}")

        return recommendations
