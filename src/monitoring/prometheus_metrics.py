"""
Prometheus metrics and monitoring for BajajFinsev RAG System
"""

import time
import psutil
import functools
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, make_wsgi_app
from prometheus_client.exposition import generate_latest
from prometheus_fastapi_instrumentator import Instrumentator


class RAGMetrics:
    """RAG-specific Prometheus metrics"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry
        
        # Document processing metrics
        self.documents_processed = Counter(
            'bajajfinsev_rag_documents_processed_total',
            'Total documents processed by RAG system',
            ['document_type'],
            registry=registry
        )
        
        # Question processing metrics
        self.questions_processed = Counter(
            'bajajfinsev_rag_questions_processed_total',
            'Total questions processed by RAG system',
            registry=registry
        )
        
        # Performance metrics
        self.processing_duration = Histogram(
            'bajajfinsev_rag_processing_duration_seconds',
            'Time spent processing RAG operations',
            ['operation'],
            registry=registry
        )
        
        # Error tracking
        self.errors = Counter(
            'bajajfinsev_rag_errors_total',
            'Total errors in RAG system',
            ['error_type', 'component'],
            registry=registry
        )
        
        # GPU metrics
        self.gpu_utilization = Gauge(
            'bajajfinsev_rag_gpu_utilization_percent',
            'GPU utilization percentage',
            registry=registry
        )
        
        # Memory metrics
        self.memory_usage = Gauge(
            'bajajfinsev_rag_memory_usage_bytes',
            'Memory usage by component',
            ['component'],
            registry=registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'bajajfinsev_rag_cache_hits_total',
            'Cache hits by type',
            ['cache_type'],
            registry=registry
        )
        
        # Answer quality metrics
        self.answer_quality = Gauge(
            'bajajfinsev_rag_answer_quality_score',
            'Quality score of generated answers',
            registry=registry
        )
        
        # Retrieval metrics
        self.chunks_created = Counter(
            'bajajfinsev_rag_chunks_created_total',
            'Total chunks created',
            registry=registry
        )
        
        self.embeddings_generated = Counter(
            'bajajfinsev_rag_embeddings_generated_total',
            'Total embeddings generated',
            registry=registry
        )
        
        self.query_duration = Histogram(
            'bajajfinsev_rag_query_duration_seconds',
            'Time spent on query processing',
            registry=registry
        )
        
        self.retrieval_accuracy = Gauge(
            'bajajfinsev_rag_retrieval_accuracy_score',
            'Accuracy score of document retrieval',
            registry=registry
        )
        
        # Analysis metrics
        self.successful_analyses = Counter(
            'bajajfinsev_rag_successful_analyses_total',
            'Total successful analyses',
            registry=registry
        )
        
        self.failed_analyses = Counter(
            'bajajfinsev_rag_failed_analyses_total',
            'Total failed analyses',
            ['failure_reason'],
            registry=registry
        )
    
    def record_document_processed(self, document_type: str):
        """Record a document processing event"""
        self.documents_processed.labels(document_type=document_type).inc()
    
    def record_question_processed(self):
        """Record a question processing event"""
        self.questions_processed.inc()
    
    def record_processing_duration(self, operation: str, duration: float):
        """Record processing duration"""
        self.processing_duration.labels(operation=operation).observe(duration)
    
    def record_error(self, error_type: str, component: str):
        """Record an error event"""
        self.errors.labels(error_type=error_type, component=component).inc()
    
    def set_gpu_utilization(self, utilization: float):
        """Set GPU utilization percentage"""
        self.gpu_utilization.set(utilization)
    
    def set_memory_usage(self, component: str, bytes_used: int):
        """Set memory usage for a component"""
        self.memory_usage.labels(component=component).set(bytes_used)
    
    def record_cache_hit(self, cache_type: str):
        """Record a cache hit"""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_answer_quality(self, score: float):
        """Record answer quality score"""
        self.answer_quality.set(score)
    
    def record_chunks_created(self, count: int):
        """Record number of chunks created"""
        self.chunks_created.inc(count)
    
    def record_embeddings_generated(self, count: int):
        """Record number of embeddings generated"""
        self.embeddings_generated.inc(count)
    
    def record_query_duration(self, duration: float):
        """Record query duration"""
        self.query_duration.observe(duration)
    
    def record_retrieval_accuracy(self, score: float):
        """Record retrieval accuracy score"""
        self.retrieval_accuracy.set(score)
    
    def record_successful_analysis(self):
        """Record a successful analysis"""
        self.successful_analyses.inc()
    
    def record_failed_analysis(self, reason: str):
        """Record a failed analysis"""
        self.failed_analyses.labels(failure_reason=reason).inc()


# Global metrics instance
rag_metrics = RAGMetrics()


def monitor_rag_operation(operation_name: str):
    """Decorator to monitor RAG operations"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                rag_metrics.record_processing_duration(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                rag_metrics.record_processing_duration(operation_name, duration)
                rag_metrics.record_error(type(e).__name__, operation_name)
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                rag_metrics.record_processing_duration(operation_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                rag_metrics.record_processing_duration(operation_name, duration)
                rag_metrics.record_error(type(e).__name__, operation_name)
                raise
        
        # Return appropriate wrapper based on whether function is async
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def setup_prometheus_instrumentation(app):
    """Setup Prometheus instrumentation for FastAPI app"""
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[".*admin.*", "/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="bajajfinsev_inprogress",
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app)
    return instrumentator


def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    try:
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # GPU metrics (basic - would need nvidia-ml-py for detailed GPU stats)
        gpu_info = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'temperature': gpu.temperature
                }
        except ImportError:
            # Fallback GPU info
            gpu_info = {
                'utilization': 0,
                'memory_used': 0,
                'memory_total': 4096,  # Assume 4GB
                'memory_percent': 0,
                'temperature': 0
            }
        
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_bytes': memory.used,
            'memory_total_bytes': memory.total,
            'disk_percent': disk.percent,
            'disk_used_bytes': disk.used,
            'disk_total_bytes': disk.total,
            'gpu_info': gpu_info
        }
    
    except Exception as e:
        return {
            'timestamp': time.time(),
            'error': str(e),
            'cpu_percent': 0,
            'memory_percent': 0,
            'gpu_info': None
        }


def set_gpu_memory_used(memory_mb: float):
    """Set GPU memory usage metric"""
    rag_metrics.set_memory_usage('gpu', memory_mb * 1024 * 1024)  # Convert MB to bytes


def update_system_metrics():
    """Update system metrics gauges"""
    try:
        metrics = get_system_metrics()
        
        # Update memory metrics
        rag_metrics.set_memory_usage('system', metrics.get('memory_used_bytes', 0))
        
        # Update GPU metrics if available
        if metrics.get('gpu_info'):
            gpu_info = metrics['gpu_info']
            rag_metrics.set_gpu_utilization(gpu_info.get('utilization', 0))
            set_gpu_memory_used(gpu_info.get('memory_used', 0))
    
    except Exception as e:
        rag_metrics.record_error('SystemMetricsError', 'update_system_metrics')