"""
Performance monitoring and metrics collection for the RAG system
"""

import time
import psutil
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request"""

    timestamp: float
    processing_time: float
    document_size: int
    num_questions: int
    success: bool
    error: Optional[str] = None


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.start_time = time.time()

        # Request tracking
        self.request_history = deque(maxlen=max_history)
        self.active_requests = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

        # Performance metrics
        self.processing_times = deque(maxlen=max_history)
        self.cache_hits = 0
        self.cache_misses = 0

        # System metrics
        self.process = psutil.Process()

        # Thread safety
        self.lock = threading.Lock()

    def start_request(self):
        """Mark the start of a new request"""
        with self.lock:
            self.active_requests += 1
            self.total_requests += 1

    def log_request(
        self,
        processing_time: float,
        document_size: int = 0,
        num_questions: int = 1,
        success: bool = True,
        error: Optional[str] = None,
    ):
        """
        Log a completed request

        Args:
            processing_time: Time taken to process the request
            document_size: Size of the processed document
            num_questions: Number of questions processed
            success: Whether the request was successful
            error: Error message if request failed
        """
        with self.lock:
            # Update counters
            self.active_requests = max(0, self.active_requests - 1)

            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1

            # Record metrics
            metrics = RequestMetrics(
                timestamp=time.time(),
                processing_time=processing_time,
                document_size=document_size,
                num_questions=num_questions,
                success=success,
                error=error,
            )

            self.request_history.append(metrics)
            self.processing_times.append(processing_time)

            logger.info(f"Request logged: {processing_time:.2f}s, success={success}")

    def log_cache_hit(self):
        """Log a cache hit"""
        with self.lock:
            self.cache_hits += 1

    def log_cache_miss(self):
        """Log a cache miss"""
        with self.lock:
            self.cache_misses += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics

        Returns:
            Dictionary containing all performance metrics
        """
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.start_time

            # Calculate averages
            avg_processing_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times
                else 0
            )

            # Calculate cache hit rate
            total_cache_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                (self.cache_hits / total_cache_requests * 100)
                if total_cache_requests > 0
                else 0
            )

            # Get system metrics
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()

            # Calculate request rate (requests per minute)
            recent_requests = [
                r
                for r in self.request_history
                if current_time - r.timestamp <= 60  # Last minute
            ]
            requests_per_minute = len(recent_requests)

            # Calculate success rate
            success_rate = (
                (self.successful_requests / self.total_requests * 100)
                if self.total_requests > 0
                else 0
            )

            # Get recent error summary
            recent_errors = defaultdict(int)
            for request in list(self.request_history)[-100:]:  # Last 100 requests
                if not request.success and request.error:
                    recent_errors[request.error] += 1

            return {
                # Request metrics
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "active_requests": self.active_requests,
                "success_rate": round(success_rate, 2),
                "requests_per_minute": requests_per_minute,
                # Performance metrics
                "average_processing_time": round(avg_processing_time, 3),
                "min_processing_time": round(min(self.processing_times), 3)
                if self.processing_times
                else 0,
                "max_processing_time": round(max(self.processing_times), 3)
                if self.processing_times
                else 0,
                # Cache metrics
                "cache_hit_rate": round(cache_hit_rate, 2),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                # System metrics
                "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": round(cpu_percent, 2),
                "uptime_seconds": round(uptime, 2),
                # Error summary
                "recent_errors": dict(recent_errors),
                # Timestamps
                "last_updated": current_time,
                "start_time": self.start_time,
            }

    def get_recent_performance(self, minutes: int = 5) -> Dict[str, Any]:
        """
        Get performance metrics for recent time period

        Args:
            minutes: Number of minutes to look back

        Returns:
            Performance metrics for the specified time period
        """
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - (minutes * 60)

            # Filter recent requests
            recent_requests = [
                r for r in self.request_history if r.timestamp >= cutoff_time
            ]

            if not recent_requests:
                return {
                    "period_minutes": minutes,
                    "requests": 0,
                    "average_processing_time": 0,
                    "success_rate": 0,
                    "errors": {},
                }

            # Calculate metrics
            successful = sum(1 for r in recent_requests if r.success)
            total = len(recent_requests)
            avg_time = sum(r.processing_time for r in recent_requests) / total

            # Error breakdown
            errors = defaultdict(int)
            for request in recent_requests:
                if not request.success and request.error:
                    errors[request.error] += 1

            return {
                "period_minutes": minutes,
                "requests": total,
                "successful_requests": successful,
                "failed_requests": total - successful,
                "success_rate": round((successful / total * 100), 2),
                "average_processing_time": round(avg_time, 3),
                "errors": dict(errors),
            }

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health status

        Returns:
            System health metrics
        """
        try:
            # Get system metrics
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()

            # Get system-wide metrics
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=1)

            # Determine health status
            health_status = "healthy"
            issues = []

            # Check memory usage
            memory_usage_percent = (memory_info.rss / system_memory.total) * 100
            if memory_usage_percent > 80:
                health_status = "warning"
                issues.append(f"High memory usage: {memory_usage_percent:.1f}%")

            # Check CPU usage
            if cpu_percent > 80:
                health_status = "warning"
                issues.append(f"High CPU usage: {cpu_percent:.1f}%")

            # Check error rate
            recent_metrics = self.get_recent_performance(5)
            if recent_metrics["requests"] > 0 and recent_metrics["success_rate"] < 90:
                health_status = "critical"
                issues.append(
                    f"Low success rate: {recent_metrics['success_rate']:.1f}%"
                )

            return {
                "status": health_status,
                "issues": issues,
                "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
                "memory_usage_percent": round(memory_usage_percent, 2),
                "cpu_percent": round(cpu_percent, 2),
                "system_cpu_percent": round(system_cpu, 2),
                "system_memory_percent": round(system_memory.percent, 2),
                "active_requests": self.active_requests,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {"status": "error", "error": str(e), "timestamp": time.time()}

    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        with self.lock:
            self.request_history.clear()
            self.processing_times.clear()

            self.active_requests = 0
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.cache_hits = 0
            self.cache_misses = 0

            self.start_time = time.time()

            logger.info("Performance metrics reset")

    def export_metrics(self, filepath: str):
        """
        Export metrics to a file for analysis

        Args:
            filepath: Path to save the metrics file
        """
        try:
            import json

            metrics = self.get_metrics()

            # Add detailed request history
            metrics["request_history"] = [
                {
                    "timestamp": r.timestamp,
                    "processing_time": r.processing_time,
                    "document_size": r.document_size,
                    "num_questions": r.num_questions,
                    "success": r.success,
                    "error": r.error,
                }
                for r in self.request_history
            ]

            with open(filepath, "w") as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Metrics exported to {filepath}")

        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            raise
