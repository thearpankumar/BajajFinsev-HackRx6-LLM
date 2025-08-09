"""
Comprehensive Performance Profiler
Advanced profiling system for RAG pipeline performance monitoring and optimization
"""

import asyncio
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Union

import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import nvidia_ml_py3 as nvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """System performance snapshot"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: int
    gpu_memory_used_mb: int
    gpu_memory_total_mb: int
    gpu_utilization: float
    gpu_temperature: float
    active_threads: int
    io_read_mb: float
    io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float


@dataclass
class OperationProfile:
    """Individual operation performance profile"""
    operation_id: str
    operation_name: str
    component: str
    start_time: float
    end_time: float
    duration: float
    cpu_usage_avg: float
    memory_peak_mb: int
    memory_growth_mb: int
    gpu_memory_peak_mb: int
    gpu_memory_growth_mb: int
    gpu_utilization_avg: float
    io_operations: int
    network_bytes: int
    success: bool
    error_message: Union[str, None]
    custom_metrics: dict[str, Any]


@dataclass
class ComponentProfile:
    """Component-level performance profile"""
    component_name: str
    total_operations: int
    successful_operations: int
    failed_operations: int
    total_duration: float
    average_duration: float
    min_duration: float
    max_duration: float
    average_cpu_usage: float
    peak_memory_mb: int
    average_memory_growth: float
    gpu_utilization_avg: float
    throughput_ops_per_sec: float
    error_rate: float


@dataclass
class SystemBottleneck:
    """System bottleneck identification"""
    bottleneck_type: str  # cpu, memory, gpu_memory, gpu_compute, io, network
    severity: str  # low, medium, high, critical
    description: str
    affected_components: list[str]
    recommendations: list[str]
    metrics: dict[str, Any]


class PerformanceProfiler:
    """
    Comprehensive performance profiler for RAG pipeline
    Monitors CPU, memory, GPU, I/O, and network performance
    """

    def __init__(self):
        # Configuration
        self.monitoring_enabled = getattr(config, 'enable_performance_monitoring', True)
        self.snapshot_interval = 1.0  # seconds
        self.max_snapshots = 1000
        self.max_profiles = 1000
        self.profile_retention_hours = 24

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Union[threading.Thread, None] = None

        # Data storage
        self.snapshots: deque = deque(maxlen=self.max_snapshots)
        self.operation_profiles: deque = deque(maxlen=self.max_profiles)
        self.component_profiles: dict[str, ComponentProfile] = {}

        # Active operation tracking
        self.active_operations: dict[str, dict[str, Any]] = {}

        # System baseline
        self.baseline_snapshot: Union[PerformanceSnapshot, None] = None
        self.baseline_established = False

        # Bottleneck detection
        self.bottleneck_thresholds = {
            "cpu_high": 85.0,
            "memory_high": 85.0,
            "gpu_memory_high": 90.0,
            "gpu_utilization_low": 30.0,
            "io_high": 100.0,  # MB/s
            "error_rate_high": 5.0  # percent
        }

        # Performance alerts
        self.alert_callbacks: list[Callable] = []
        self.last_alert_times: dict[str, float] = {}
        self.alert_cooldown = 300  # 5 minutes

        # Export settings
        self.export_dir = Path("./performance_profiles")
        self.export_dir.mkdir(exist_ok=True)

        logger.info("PerformanceProfiler initialized")

    async def initialize(self) -> dict[str, Any]:
        """Initialize performance profiler"""
        try:
            logger.info("🔄 Initializing Performance Profiler...")
            start_time = time.time()

            # Initialize NVML if available
            if HAS_NVML:
                try:
                    nvml.nvmlInit()
                    self.nvml_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    logger.info("✅ NVML initialized for GPU monitoring")
                except Exception as e:
                    logger.warning(f"NVML initialization failed: {str(e)}")
                    self.nvml_handle = None

            # Establish baseline
            await self._establish_baseline()

            # Start monitoring if enabled
            if self.monitoring_enabled:
                await self.start_monitoring()

            initialization_time = time.time() - start_time

            result = {
                "status": "success",
                "message": f"Performance Profiler initialized in {initialization_time:.2f}s",
                "monitoring_enabled": self.monitoring_enabled,
                "snapshot_interval": self.snapshot_interval,
                "gpu_monitoring": HAS_NVML and hasattr(self, 'nvml_handle'),
                "baseline_established": self.baseline_established,
                "initialization_time": initialization_time
            }

            logger.info("✅ Performance Profiler ready")
            return result

        except Exception as e:
            error_msg = f"Performance Profiler initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }

    async def _establish_baseline(self):
        """Establish system performance baseline"""
        try:
            logger.info("📊 Establishing performance baseline...")

            # Take multiple snapshots to establish baseline
            baseline_snapshots = []
            for _ in range(5):
                snapshot = await self._take_snapshot()
                baseline_snapshots.append(snapshot)
                await asyncio.sleep(0.5)

            # Calculate baseline averages
            if baseline_snapshots:
                self.baseline_snapshot = PerformanceSnapshot(
                    timestamp=time.time(),
                    cpu_percent=sum(s.cpu_percent for s in baseline_snapshots) / len(baseline_snapshots),
                    memory_percent=sum(s.memory_percent for s in baseline_snapshots) / len(baseline_snapshots),
                    memory_used_mb=sum(s.memory_used_mb for s in baseline_snapshots) // len(baseline_snapshots),
                    gpu_memory_used_mb=sum(s.gpu_memory_used_mb for s in baseline_snapshots) // len(baseline_snapshots),
                    gpu_memory_total_mb=baseline_snapshots[0].gpu_memory_total_mb,
                    gpu_utilization=sum(s.gpu_utilization for s in baseline_snapshots) / len(baseline_snapshots),
                    gpu_temperature=sum(s.gpu_temperature for s in baseline_snapshots) / len(baseline_snapshots),
                    active_threads=sum(s.active_threads for s in baseline_snapshots) // len(baseline_snapshots),
                    io_read_mb=sum(s.io_read_mb for s in baseline_snapshots) / len(baseline_snapshots),
                    io_write_mb=sum(s.io_write_mb for s in baseline_snapshots) / len(baseline_snapshots),
                    network_sent_mb=sum(s.network_sent_mb for s in baseline_snapshots) / len(baseline_snapshots),
                    network_recv_mb=sum(s.network_recv_mb for s in baseline_snapshots) / len(baseline_snapshots)
                )

                self.baseline_established = True
                logger.info(f"✅ Baseline established: CPU {self.baseline_snapshot.cpu_percent:.1f}%, "
                           f"Memory {self.baseline_snapshot.memory_percent:.1f}%, "
                           f"GPU {self.baseline_snapshot.gpu_utilization:.1f}%")

        except Exception as e:
            logger.warning(f"Baseline establishment failed: {str(e)}")
            self.baseline_established = False

    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        if self.is_monitoring:
            return

        logger.info("🔄 Starting continuous performance monitoring...")
        self.is_monitoring = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("✅ Performance monitoring started")

    async def stop_monitoring(self):
        """Stop continuous performance monitoring"""
        if not self.is_monitoring:
            return

        logger.info("🔄 Stopping performance monitoring...")
        self.is_monitoring = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        logger.info("✅ Performance monitoring stopped")

    def _monitoring_loop(self):
        """Monitoring loop running in separate thread"""
        while self.is_monitoring:
            try:
                # Take snapshot
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                snapshot = loop.run_until_complete(self._take_snapshot())
                self.snapshots.append(snapshot)

                # Check for bottlenecks
                bottlenecks = self._detect_bottlenecks(snapshot)
                if bottlenecks:
                    loop.run_until_complete(self._handle_bottlenecks(bottlenecks))

                loop.close()

                time.sleep(self.snapshot_interval)

            except Exception as e:
                logger.warning(f"Monitoring loop error: {str(e)}")
                time.sleep(self.snapshot_interval)

    async def _take_snapshot(self) -> PerformanceSnapshot:
        """Take a comprehensive system performance snapshot"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            # I/O stats
            io_stats = psutil.disk_io_counters()
            io_read_mb = io_stats.read_bytes / (1024 * 1024) if io_stats else 0
            io_write_mb = io_stats.write_bytes / (1024 * 1024) if io_stats else 0

            # Network stats
            net_stats = psutil.net_io_counters()
            network_sent_mb = net_stats.bytes_sent / (1024 * 1024) if net_stats else 0
            network_recv_mb = net_stats.bytes_recv / (1024 * 1024) if net_stats else 0

            # GPU stats
            gpu_memory_used_mb = 0
            gpu_memory_total_mb = 0
            gpu_utilization = 0.0
            gpu_temperature = 0.0

            if HAS_TORCH and torch.cuda.is_available():
                gpu_memory_used_mb = torch.cuda.memory_allocated() // (1024 ** 2)
                gpu_memory_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)

            if HAS_NVML and hasattr(self, 'nvml_handle') and self.nvml_handle:
                try:
                    utilization = nvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    gpu_utilization = utilization.gpu

                    temperature = nvml.nvmlDeviceGetTemperature(self.nvml_handle, nvml.NVML_TEMPERATURE_GPU)
                    gpu_temperature = temperature
                except:
                    pass

            return PerformanceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used // (1024 ** 2),
                gpu_memory_used_mb=gpu_memory_used_mb,
                gpu_memory_total_mb=gpu_memory_total_mb,
                gpu_utilization=gpu_utilization,
                gpu_temperature=gpu_temperature,
                active_threads=threading.active_count(),
                io_read_mb=io_read_mb,
                io_write_mb=io_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )

        except Exception as e:
            logger.warning(f"Snapshot creation failed: {str(e)}")
            return PerformanceSnapshot(
                timestamp=time.time(),
                cpu_percent=0, memory_percent=0, memory_used_mb=0,
                gpu_memory_used_mb=0, gpu_memory_total_mb=0, gpu_utilization=0,
                gpu_temperature=0, active_threads=0,
                io_read_mb=0, io_write_mb=0, network_sent_mb=0, network_recv_mb=0
            )

    async def profile_operation(self, operation_name: str, component: str = "unknown") -> str:
        """Start profiling an operation"""
        operation_id = f"{component}_{operation_name}_{int(time.time() * 1000)}"

        # Get initial snapshot
        initial_snapshot = await self._take_snapshot()

        self.active_operations[operation_id] = {
            "operation_name": operation_name,
            "component": component,
            "start_time": time.time(),
            "initial_snapshot": initial_snapshot,
            "peak_memory_mb": initial_snapshot.memory_used_mb,
            "peak_gpu_memory_mb": initial_snapshot.gpu_memory_used_mb,
            "cpu_samples": [initial_snapshot.cpu_percent],
            "gpu_samples": [initial_snapshot.gpu_utilization],
            "io_operations": 0,
            "network_bytes": 0
        }

        return operation_id

    async def complete_operation(self, operation_id: str, success: bool = True,
                               error_message: Union[str, None] = None,
                               custom_metrics: dict[str, Any] | None = None):
        """Complete operation profiling"""
        if operation_id not in self.active_operations:
            return

        try:
            op_data = self.active_operations[operation_id]
            end_time = time.time()

            # Get final snapshot
            final_snapshot = await self._take_snapshot()

            # Calculate metrics
            duration = end_time - op_data["start_time"]
            cpu_usage_avg = sum(op_data["cpu_samples"]) / len(op_data["cpu_samples"])
            gpu_utilization_avg = sum(op_data["gpu_samples"]) / len(op_data["gpu_samples"])

            memory_growth_mb = final_snapshot.memory_used_mb - op_data["initial_snapshot"].memory_used_mb
            gpu_memory_growth_mb = final_snapshot.gpu_memory_used_mb - op_data["initial_snapshot"].gpu_memory_used_mb

            # Create operation profile
            profile = OperationProfile(
                operation_id=operation_id,
                operation_name=op_data["operation_name"],
                component=op_data["component"],
                start_time=op_data["start_time"],
                end_time=end_time,
                duration=duration,
                cpu_usage_avg=cpu_usage_avg,
                memory_peak_mb=op_data["peak_memory_mb"],
                memory_growth_mb=memory_growth_mb,
                gpu_memory_peak_mb=op_data["peak_gpu_memory_mb"],
                gpu_memory_growth_mb=gpu_memory_growth_mb,
                gpu_utilization_avg=gpu_utilization_avg,
                io_operations=op_data["io_operations"],
                network_bytes=op_data["network_bytes"],
                success=success,
                error_message=error_message,
                custom_metrics=custom_metrics or {}
            )

            # Store profile
            self.operation_profiles.append(profile)

            # Update component profile
            await self._update_component_profile(profile)

            # Clean up
            del self.active_operations[operation_id]

        except Exception as e:
            logger.warning(f"Operation completion failed: {str(e)}")
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]

    async def _update_component_profile(self, operation_profile: OperationProfile):
        """Update component-level profile statistics"""
        try:
            component = operation_profile.component

            if component not in self.component_profiles:
                self.component_profiles[component] = ComponentProfile(
                    component_name=component,
                    total_operations=0, successful_operations=0, failed_operations=0,
                    total_duration=0.0, average_duration=0.0,
                    min_duration=float('inf'), max_duration=0.0,
                    average_cpu_usage=0.0, peak_memory_mb=0,
                    average_memory_growth=0.0, gpu_utilization_avg=0.0,
                    throughput_ops_per_sec=0.0, error_rate=0.0
                )

            profile = self.component_profiles[component]

            # Update counters
            profile.total_operations += 1
            if operation_profile.success:
                profile.successful_operations += 1
            else:
                profile.failed_operations += 1

            # Update duration stats
            profile.total_duration += operation_profile.duration
            profile.average_duration = profile.total_duration / profile.total_operations
            profile.min_duration = min(profile.min_duration, operation_profile.duration)
            profile.max_duration = max(profile.max_duration, operation_profile.duration)

            # Update resource usage (running average)
            alpha = 0.1  # Smoothing factor
            profile.average_cpu_usage = (1 - alpha) * profile.average_cpu_usage + alpha * operation_profile.cpu_usage_avg
            profile.peak_memory_mb = max(profile.peak_memory_mb, operation_profile.memory_peak_mb)
            profile.average_memory_growth = (1 - alpha) * profile.average_memory_growth + alpha * operation_profile.memory_growth_mb
            profile.gpu_utilization_avg = (1 - alpha) * profile.gpu_utilization_avg + alpha * operation_profile.gpu_utilization_avg

            # Calculate throughput
            if profile.total_duration > 0:
                profile.throughput_ops_per_sec = profile.total_operations / profile.total_duration

            # Calculate error rate
            profile.error_rate = (profile.failed_operations / profile.total_operations) * 100

        except Exception as e:
            logger.warning(f"Component profile update failed: {str(e)}")

    def _detect_bottlenecks(self, snapshot: PerformanceSnapshot) -> list[SystemBottleneck]:
        """Detect system bottlenecks from performance snapshot"""
        bottlenecks = []

        try:
            # CPU bottleneck
            if snapshot.cpu_percent > self.bottleneck_thresholds["cpu_high"]:
                bottlenecks.append(SystemBottleneck(
                    bottleneck_type="cpu",
                    severity="high" if snapshot.cpu_percent > 95 else "medium",
                    description=f"High CPU utilization: {snapshot.cpu_percent:.1f}%",
                    affected_components=["all"],
                    recommendations=[
                        "Consider reducing batch sizes",
                        "Enable parallel processing",
                        "Check for inefficient algorithms"
                    ],
                    metrics={"cpu_percent": snapshot.cpu_percent}
                ))

            # Memory bottleneck
            if snapshot.memory_percent > self.bottleneck_thresholds["memory_high"]:
                bottlenecks.append(SystemBottleneck(
                    bottleneck_type="memory",
                    severity="high" if snapshot.memory_percent > 95 else "medium",
                    description=f"High memory usage: {snapshot.memory_percent:.1f}%",
                    affected_components=["embedding_service", "vector_store"],
                    recommendations=[
                        "Clear caches periodically",
                        "Reduce batch sizes",
                        "Enable memory-efficient processing"
                    ],
                    metrics={"memory_percent": snapshot.memory_percent, "memory_used_mb": snapshot.memory_used_mb}
                ))

            # GPU memory bottleneck
            if snapshot.gpu_memory_total_mb > 0:
                gpu_memory_percent = (snapshot.gpu_memory_used_mb / snapshot.gpu_memory_total_mb) * 100
                if gpu_memory_percent > self.bottleneck_thresholds["gpu_memory_high"]:
                    bottlenecks.append(SystemBottleneck(
                        bottleneck_type="gpu_memory",
                        severity="critical" if gpu_memory_percent > 98 else "high",
                        description=f"High GPU memory usage: {gpu_memory_percent:.1f}%",
                        affected_components=["embedding_service", "gpu_service"],
                        recommendations=[
                            "Reduce embedding batch sizes",
                            "Enable mixed precision (FP16)",
                            "Clear GPU cache frequently"
                        ],
                        metrics={"gpu_memory_percent": gpu_memory_percent, "gpu_memory_used_mb": snapshot.gpu_memory_used_mb}
                    ))

            # Low GPU utilization (inefficiency)
            if snapshot.gpu_utilization < self.bottleneck_thresholds["gpu_utilization_low"] and snapshot.gpu_memory_used_mb > 100:
                bottlenecks.append(SystemBottleneck(
                    bottleneck_type="gpu_compute",
                    severity="medium",
                    description=f"Low GPU utilization: {snapshot.gpu_utilization:.1f}%",
                    affected_components=["embedding_service"],
                    recommendations=[
                        "Increase batch sizes",
                        "Check for CPU-GPU transfer bottlenecks",
                        "Optimize data pipeline"
                    ],
                    metrics={"gpu_utilization": snapshot.gpu_utilization}
                ))

        except Exception as e:
            logger.warning(f"Bottleneck detection failed: {str(e)}")

        return bottlenecks

    async def _handle_bottlenecks(self, bottlenecks: list[SystemBottleneck]):
        """Handle detected bottlenecks"""
        for bottleneck in bottlenecks:
            # Check cooldown
            alert_key = f"{bottleneck.bottleneck_type}_{bottleneck.severity}"
            current_time = time.time()

            if alert_key in self.last_alert_times:
                if current_time - self.last_alert_times[alert_key] < self.alert_cooldown:
                    continue

            # Log bottleneck
            severity_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}
            emoji = severity_emoji.get(bottleneck.severity, "⚠️")

            logger.warning(f"{emoji} BOTTLENECK DETECTED: {bottleneck.description}")
            logger.warning(f"Affected components: {', '.join(bottleneck.affected_components)}")
            logger.warning(f"Recommendations: {'; '.join(bottleneck.recommendations)}")

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    await callback(bottleneck)
                except Exception as e:
                    logger.warning(f"Alert callback failed: {str(e)}")

            # Update last alert time
            self.last_alert_times[alert_key] = current_time

    def add_alert_callback(self, callback: Callable):
        """Add callback for performance alerts"""
        self.alert_callbacks.append(callback)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary"""
        try:
            if not self.snapshots:
                return {"message": "No performance data available"}

            recent_snapshots = list(self.snapshots)[-60:]  # Last 60 snapshots

            # Calculate averages
            avg_cpu = sum(s.cpu_percent for s in recent_snapshots) / len(recent_snapshots)
            avg_memory = sum(s.memory_percent for s in recent_snapshots) / len(recent_snapshots)
            avg_gpu_util = sum(s.gpu_utilization for s in recent_snapshots) / len(recent_snapshots)

            # Component summaries
            component_summary = {}
            for comp_name, comp_profile in self.component_profiles.items():
                component_summary[comp_name] = {
                    "total_operations": comp_profile.total_operations,
                    "success_rate": ((comp_profile.successful_operations / max(1, comp_profile.total_operations)) * 100),
                    "average_duration": comp_profile.average_duration,
                    "throughput_ops_per_sec": comp_profile.throughput_ops_per_sec,
                    "error_rate": comp_profile.error_rate
                }

            # Recent bottlenecks
            current_snapshot = recent_snapshots[-1]
            current_bottlenecks = self._detect_bottlenecks(current_snapshot)

            return {
                "system_performance": {
                    "cpu_percent_avg": round(avg_cpu, 1),
                    "memory_percent_avg": round(avg_memory, 1),
                    "gpu_utilization_avg": round(avg_gpu_util, 1),
                    "current_cpu": current_snapshot.cpu_percent,
                    "current_memory": current_snapshot.memory_percent,
                    "current_gpu_utilization": current_snapshot.gpu_utilization,
                    "gpu_memory_used_mb": current_snapshot.gpu_memory_used_mb,
                    "gpu_memory_total_mb": current_snapshot.gpu_memory_total_mb
                },
                "component_performance": component_summary,
                "monitoring_status": {
                    "is_monitoring": self.is_monitoring,
                    "snapshots_collected": len(self.snapshots),
                    "operations_profiled": len(self.operation_profiles),
                    "active_operations": len(self.active_operations),
                    "baseline_established": self.baseline_established
                },
                "current_bottlenecks": [
                    {
                        "type": b.bottleneck_type,
                        "severity": b.severity,
                        "description": b.description,
                        "recommendations": b.recommendations
                    }
                    for b in current_bottlenecks
                ]
            }

        except Exception as e:
            logger.warning(f"Performance summary generation failed: {str(e)}")
            return {"error": str(e)}

    async def export_profile_data(self, format: str = "json") -> str:
        """Export profile data for analysis"""
        try:
            timestamp = int(time.time())

            if format == "json":
                export_file = self.export_dir / f"performance_profile_{timestamp}.json"

                data = {
                    "export_timestamp": timestamp,
                    "monitoring_duration": len(self.snapshots) * self.snapshot_interval,
                    "system_snapshots": [asdict(s) for s in list(self.snapshots)],
                    "operation_profiles": [asdict(p) for p in list(self.operation_profiles)],
                    "component_profiles": {name: asdict(profile) for name, profile in self.component_profiles.items()},
                    "baseline_snapshot": asdict(self.baseline_snapshot) if self.baseline_snapshot else None
                }

                with open(export_file, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"📊 Performance data exported to {export_file}")
                return str(export_file)

            else:
                return "Unsupported export format"

        except Exception as e:
            logger.error(f"Profile data export failed: {str(e)}")
            return f"Export failed: {str(e)}"
