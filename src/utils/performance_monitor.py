import logging
import time
import asyncio
import psutil
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for document processing"""
    timestamp: str
    operation: str
    duration: float
    document_size_chars: int
    document_size_tokens: Optional[int]
    chunks_processed: int
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hits: int
    cache_misses: int
    api_calls: int
    error_count: int
    success: bool
    metadata: Dict[str, Any]

class PerformanceMonitor:
    """
    Monitor and log performance metrics for large document processing
    """
    
    def __init__(self):
        self.logger = logger
        self.process = psutil.Process(os.getpid())
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_start_times: Dict[str, float] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self.api_call_count = 0
        self.error_count = 0
    
    def start_operation(self, operation_name: str) -> None:
        """Start timing an operation"""
        self.operation_start_times[operation_name] = time.time()
        self.logger.debug(f"Started operation: {operation_name}")
    
    def end_operation(
        self, 
        operation_name: str, 
        document_size: int = 0,
        chunks_processed: int = 0,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerformanceMetrics:
        """End timing an operation and record metrics"""
        
        if operation_name not in self.operation_start_times:
            self.logger.warning(f"Operation {operation_name} was not started")
            return None
        
        duration = time.time() - self.operation_start_times[operation_name]
        del self.operation_start_times[operation_name]
        
        # Gather system metrics
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        # Estimate token count (rough approximation: 4 chars per token)
        estimated_tokens = document_size // 4 if document_size > 0 else None
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            operation=operation_name,
            duration=duration,
            document_size_chars=document_size,
            document_size_tokens=estimated_tokens,
            chunks_processed=chunks_processed,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            cache_hits=self.cache_stats["hits"],
            cache_misses=self.cache_stats["misses"],
            api_calls=self.api_call_count,
            error_count=self.error_count,
            success=success,
            metadata=metadata or {}
        )
        
        self.metrics_history.append(metrics)
        
        # Log performance summary
        self.logger.info(f"Operation '{operation_name}' completed in {duration:.2f}s")
        self.logger.info(f"Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, Chunks: {chunks_processed}")
        
        if document_size > 0:
            chars_per_second = document_size / duration
            self.logger.info(f"Processing speed: {chars_per_second:.0f} chars/second")
        
        return metrics
    
    def record_cache_hit(self) -> None:
        """Record a cache hit"""
        self.cache_stats["hits"] += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss"""
        self.cache_stats["misses"] += 1
    
    def record_api_call(self) -> None:
        """Record an API call"""
        self.api_call_count += 1
    
    def record_error(self) -> None:
        """Record an error"""
        self.error_count += 1
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage"""
        total = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total == 0:
            return 0.0
        return (self.cache_stats["hits"] / total) * 100
    
    def get_recent_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent performance metrics"""
        recent_metrics = self.metrics_history[-limit:] if self.metrics_history else []
        return [asdict(metric) for metric in recent_metrics]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        if not self.metrics_history:
            return {"message": "No metrics recorded yet"}
        
        # Calculate averages
        total_operations = len(self.metrics_history)
        avg_duration = sum(m.duration for m in self.metrics_history) / total_operations
        avg_memory = sum(m.memory_usage_mb for m in self.metrics_history) / total_operations
        avg_cpu = sum(m.cpu_usage_percent for m in self.metrics_history) / total_operations
        
        # Find largest document processed
        largest_doc = max(self.metrics_history, key=lambda m: m.document_size_chars)
        
        # Calculate success rate
        successful_ops = sum(1 for m in self.metrics_history if m.success)
        success_rate = (successful_ops / total_operations) * 100
        
        # Group by operation type
        operation_stats = {}
        for metric in self.metrics_history:
            op = metric.operation
            if op not in operation_stats:
                operation_stats[op] = {
                    'count': 0,
                    'total_duration': 0,
                    'avg_duration': 0,
                    'success_count': 0
                }
            
            operation_stats[op]['count'] += 1
            operation_stats[op]['total_duration'] += metric.duration
            operation_stats[op]['success_count'] += 1 if metric.success else 0
        
        # Calculate averages for each operation
        for op_stat in operation_stats.values():
            op_stat['avg_duration'] = op_stat['total_duration'] / op_stat['count']
            op_stat['success_rate'] = (op_stat['success_count'] / op_stat['count']) * 100
        
        return {
            "total_operations": total_operations,
            "avg_duration_seconds": round(avg_duration, 2),
            "avg_memory_usage_mb": round(avg_memory, 1),
            "avg_cpu_usage_percent": round(avg_cpu, 1),
            "cache_hit_rate_percent": round(self.get_cache_hit_rate(), 1),
            "total_api_calls": self.api_call_count,
            "total_errors": self.error_count,
            "success_rate_percent": round(success_rate, 1),
            "largest_document_chars": largest_doc.document_size_chars,
            "largest_document_tokens": largest_doc.document_size_tokens,
            "operation_breakdown": operation_stats
        }
    
    def log_performance_summary(self) -> None:
        """Log a comprehensive performance summary"""
        summary = self.get_performance_summary()
        
        self.logger.info("=== PERFORMANCE SUMMARY ===")
        self.logger.info(f"Total Operations: {summary.get('total_operations', 0)}")
        self.logger.info(f"Average Duration: {summary.get('avg_duration_seconds', 0)}s")
        self.logger.info(f"Cache Hit Rate: {summary.get('cache_hit_rate_percent', 0)}%")
        self.logger.info(f"Success Rate: {summary.get('success_rate_percent', 0)}%")
        self.logger.info(f"Memory Usage: {summary.get('avg_memory_usage_mb', 0)}MB")
        self.logger.info(f"API Calls: {summary.get('total_api_calls', 0)}")
        
        if summary.get('largest_document_chars', 0) > 0:
            self.logger.info(f"Largest Document: {summary['largest_document_chars']:,} chars ({summary.get('largest_document_tokens', 0):,} tokens)")
        
        self.logger.info("=== END SUMMARY ===")
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to JSON file"""
        try:
            metrics_data = {
                "summary": self.get_performance_summary(),
                "recent_metrics": self.get_recent_metrics(50),  # Last 50 operations
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            self.logger.info(f"Performance metrics exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics and counters"""
        self.metrics_history.clear()
        self.operation_start_times.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        self.api_call_count = 0
        self.error_count = 0
        self.logger.info("Performance metrics reset")

class PerformanceDecorator:
    """Decorator for automatic performance monitoring"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
    
    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                self.monitor.start_operation(self.operation_name)
                try:
                    result = await func(*args, **kwargs)
                    self.monitor.end_operation(self.operation_name, success=True)
                    return result
                except Exception as e:
                    self.monitor.record_error()
                    self.monitor.end_operation(self.operation_name, success=False)
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                self.monitor.start_operation(self.operation_name)
                try:
                    result = func(*args, **kwargs)
                    self.monitor.end_operation(self.operation_name, success=True)
                    return result
                except Exception as e:
                    self.monitor.record_error()
                    self.monitor.end_operation(self.operation_name, success=False)
                    raise
            return sync_wrapper

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str):
    """Decorator factory for performance monitoring"""
    return PerformanceDecorator(performance_monitor, operation_name)