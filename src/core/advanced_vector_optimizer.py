"""
Advanced Vector Database Optimizer
High-performance optimization for FAISS with dynamic tuning and intelligent indexing
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from src.core.gpu_service import GPUService
from src.core.config import config

logger = logging.getLogger(__name__)


@dataclass
class IndexPerformanceMetrics:
    """Index performance metrics"""
    index_type: str
    search_time: float
    build_time: float
    memory_usage_mb: int
    recall_score: float
    throughput_qps: float
    dimension: int
    num_vectors: int


@dataclass
class OptimizationResult:
    """Optimization result data"""
    original_performance: IndexPerformanceMetrics
    optimized_performance: IndexPerformanceMetrics
    improvement_percentage: float
    optimization_strategy: str
    parameters_changed: dict[str, Any]


class AdvancedVectorOptimizer:
    """
    Advanced vector database optimizer with dynamic parameter tuning
    Specifically optimized for RTX 3050 and high-performance retrieval
    """

    def __init__(self, gpu_service: Union[GPUService, None] = None):
        self.gpu_service = gpu_service or GPUService()

        # Configuration
        self.dimension = config.embedding_dimension
        self.optimization_cache_dir = Path("./optimization_cache")
        self.optimization_cache_dir.mkdir(exist_ok=True)

        # Performance tracking
        self.performance_history: list[IndexPerformanceMetrics] = []
        self.optimization_history: list[OptimizationResult] = []

        # Index configurations for different use cases
        self.index_configurations = {
            "speed_optimized": {
                "index_factory": "HNSW32",
                "parameters": {"efConstruction": 200, "efSearch": 128, "M": 32}
            },
            "memory_optimized": {
                "index_factory": "IVF4096,Flat",
                "parameters": {"nprobe": 64}
            },
            "accuracy_optimized": {
                "index_factory": "IVF1024,PQ64",
                "parameters": {"nprobe": 128}
            },
            "rtx_3050_optimized": {
                "index_factory": "HNSW16",
                "parameters": {"efConstruction": 128, "efSearch": 64, "M": 16}
            }
        }

        # Dynamic optimization parameters
        self.auto_optimization_enabled = True
        self.performance_threshold = 0.8  # Trigger optimization if performance drops below 80%
        self.optimization_frequency = 100  # Operations between optimization checks
        self.operations_since_optimization = 0

        # Current optimal configuration
        self.current_config = "rtx_3050_optimized"
        self.optimal_parameters = self.index_configurations[self.current_config]["parameters"].copy()

        logger.info("AdvancedVectorOptimizer initialized with RTX 3050 focus")

    async def initialize(self) -> dict[str, Any]:
        """Initialize the vector optimizer"""
        try:
            logger.info("🔄 Initializing Advanced Vector Optimizer...")
            start_time = time.time()

            if not HAS_FAISS:
                return {
                    "status": "error",
                    "error": "FAISS not available for vector optimization"
                }

            # Initialize GPU service
            if not hasattr(self.gpu_service, 'is_gpu_available') or not self.gpu_service.is_gpu_available:
                gpu_result = self.gpu_service.initialize()
                if gpu_result.get('status') != 'success':
                    logger.warning("GPU initialization failed, using CPU fallback")

            # Load optimization history if available
            await self._load_optimization_cache()

            # Determine optimal configuration based on hardware
            await self._detect_optimal_configuration()

            initialization_time = time.time() - start_time

            result = {
                "status": "success",
                "message": f"Vector Optimizer initialized in {initialization_time:.2f}s",
                "current_config": self.current_config,
                "optimal_parameters": self.optimal_parameters,
                "auto_optimization": self.auto_optimization_enabled,
                "cached_optimizations": len(self.optimization_history),
                "initialization_time": initialization_time
            }

            logger.info(f"✅ Vector Optimizer ready with {self.current_config} configuration")
            return result

        except Exception as e:
            error_msg = f"Vector Optimizer initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }

    async def _detect_optimal_configuration(self):
        """Detect optimal configuration based on hardware and usage patterns"""
        try:
            # Check GPU memory availability
            device_info = self.gpu_service.get_device_info()
            available_memory = device_info.get("total_memory_mb", 4000)

            # Detect hardware type from GPU service
            gpu_name = device_info.get("device_name", "unknown")

            if available_memory < 2000:  # Less than 2GB
                self.current_config = "memory_optimized"
                logger.info("🎯 Selected memory_optimized configuration for low memory")
            elif "3050" in gpu_name or available_memory < 4500:  # RTX 3050 or similar
                self.current_config = "rtx_3050_optimized"
                logger.info("🎯 Selected rtx_3050_optimized configuration")
            elif available_memory > 6000:  # High-end GPU
                self.current_config = "accuracy_optimized"
                logger.info("🎯 Selected accuracy_optimized configuration")
            else:
                self.current_config = "speed_optimized"
                logger.info("🎯 Selected speed_optimized configuration")

            # Update optimal parameters
            self.optimal_parameters = self.index_configurations[self.current_config]["parameters"].copy()

        except Exception as e:
            logger.warning(f"Configuration detection failed: {str(e)}")
            self.current_config = "rtx_3050_optimized"  # Safe default

    async def optimize_index_parameters(
        self,
        sample_vectors: np.ndarray,
        query_vectors: np.ndarray,
        current_index: Union['faiss.Index', None] = None
    ) -> dict[str, Any]:
        """
        Optimize index parameters based on sample data
        
        Args:
            sample_vectors: Representative sample of vectors for optimization
            query_vectors: Sample queries for testing
            current_index: Current index to optimize (if any)
            
        Returns:
            Optimization results with improved parameters
        """
        logger.info(f"🎯 Starting index optimization with {len(sample_vectors)} samples")
        start_time = time.time()

        try:
            # Benchmark current configuration
            current_metrics = await self._benchmark_configuration(
                sample_vectors, query_vectors, self.current_config, current_index
            )

            logger.info(f"📊 Current performance: {current_metrics.throughput_qps:.1f} QPS, "
                       f"{current_metrics.recall_score:.3f} recall")

            # Test different configurations
            best_config = self.current_config
            best_metrics = current_metrics
            best_parameters = self.optimal_parameters.copy()

            for config_name, config_data in self.index_configurations.items():
                if config_name == self.current_config:
                    continue  # Skip current configuration

                try:
                    logger.info(f"🔍 Testing {config_name} configuration...")

                    test_metrics = await self._benchmark_configuration(
                        sample_vectors, query_vectors, config_name
                    )

                    # Calculate improvement score (weighted combination of metrics)
                    improvement_score = self._calculate_improvement_score(
                        current_metrics, test_metrics
                    )

                    if improvement_score > 0.1:  # 10% improvement threshold
                        logger.info(f"✅ {config_name}: {improvement_score:.1%} improvement")

                        current_score = self._calculate_improvement_score(current_metrics, best_metrics)
                        if improvement_score > current_score:
                            best_config = config_name
                            best_metrics = test_metrics
                            best_parameters = config_data["parameters"].copy()
                    else:
                        logger.info(f"⚪ {config_name}: {improvement_score:.1%} change")

                except Exception as e:
                    logger.warning(f"Configuration {config_name} testing failed: {str(e)}")
                    continue

            # Fine-tune parameters for best configuration
            if best_config != self.current_config:
                logger.info(f"🔧 Fine-tuning {best_config} parameters...")
                fine_tuned_params = await self._fine_tune_parameters(
                    sample_vectors, query_vectors, best_config, best_parameters
                )
                best_parameters.update(fine_tuned_params)

            # Create optimization result
            improvement_percentage = self._calculate_improvement_score(current_metrics, best_metrics) * 100

            optimization_result = OptimizationResult(
                original_performance=current_metrics,
                optimized_performance=best_metrics,
                improvement_percentage=improvement_percentage,
                optimization_strategy=best_config,
                parameters_changed=best_parameters
            )

            # Update current configuration if significantly better
            if improvement_percentage > 5:  # 5% improvement threshold for switching
                self.current_config = best_config
                self.optimal_parameters = best_parameters
                logger.info(f"🚀 Adopted {best_config} with {improvement_percentage:.1f}% improvement")
            else:
                logger.info(f"💡 Current configuration remains optimal ({improvement_percentage:.1f}% change)")

            # Store optimization result
            self.optimization_history.append(optimization_result)
            await self._save_optimization_cache()

            optimization_time = time.time() - start_time

            return {
                "status": "success",
                "optimization_result": optimization_result,
                "new_configuration": self.current_config,
                "new_parameters": self.optimal_parameters,
                "improvement_percentage": improvement_percentage,
                "optimization_time": optimization_time,
                "tested_configurations": len(self.index_configurations)
            }

        except Exception as e:
            error_msg = f"Index optimization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "optimization_time": time.time() - start_time
            }

    async def _benchmark_configuration(
        self,
        sample_vectors: np.ndarray,
        query_vectors: np.ndarray,
        config_name: str,
        existing_index: Union['faiss.Index', None] = None
    ) -> IndexPerformanceMetrics:
        """Benchmark a specific configuration"""

        config_data = self.index_configurations[config_name]

        with self.gpu_service.memory_profiler(f"benchmark_{config_name}"):
            # Build index
            build_start = time.time()

            if existing_index is not None and config_name == self.current_config:
                index = existing_index
                build_time = 0.0  # Using existing index
            else:
                index = await self._build_index_with_config(sample_vectors, config_data)
                build_time = time.time() - build_start

            # Measure memory usage
            memory_before = self._get_current_memory_usage()

            # Benchmark search performance
            search_times = []
            all_distances = []
            all_indices = []

            k = min(10, len(sample_vectors) // 10)  # Search for 10 neighbors or 10% of dataset

            for i in range(min(100, len(query_vectors))):  # Test with up to 100 queries
                query = query_vectors[i:i+1]

                search_start = time.time()
                distances, indices = index.search(query, k)
                search_time = time.time() - search_start

                search_times.append(search_time)
                all_distances.append(distances)
                all_indices.append(indices)

            memory_after = self._get_current_memory_usage()

            # Calculate metrics
            avg_search_time = sum(search_times) / len(search_times)
            throughput_qps = len(search_times) / sum(search_times)
            memory_usage_mb = memory_after - memory_before

            # Calculate recall (simplified - using distance thresholds)
            recall_score = self._calculate_recall_score(all_distances, all_indices)

            return IndexPerformanceMetrics(
                index_type=config_name,
                search_time=avg_search_time,
                build_time=build_time,
                memory_usage_mb=memory_usage_mb,
                recall_score=recall_score,
                throughput_qps=throughput_qps,
                dimension=sample_vectors.shape[1],
                num_vectors=len(sample_vectors)
            )

    async def _build_index_with_config(self, vectors: np.ndarray, config_data: dict[str, Any]) -> 'faiss.Index':
        """Build FAISS index with specific configuration"""
        try:
            index_factory = config_data["index_factory"]
            parameters = config_data["parameters"]

            # Create index
            if "HNSW" in index_factory:
                M = parameters.get("M", 16)
                index = faiss.IndexHNSWFlat(vectors.shape[1], M)
                index.hnsw.efConstruction = parameters.get("efConstruction", 200)
                index.hnsw.efSearch = parameters.get("efSearch", 128)

            elif "IVF" in index_factory:
                # Parse IVF configuration
                parts = index_factory.split(",")
                nlist = int(parts[0].replace("IVF", ""))

                quantizer = faiss.IndexFlatL2(vectors.shape[1])

                if "PQ" in parts[1]:
                    # Product Quantization
                    pq_segments = int(parts[1].replace("PQ", ""))
                    index = faiss.IndexIVFPQ(quantizer, vectors.shape[1], nlist, pq_segments, 8)
                else:
                    # Flat
                    index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist)

                # Train index
                index.train(vectors)
                index.nprobe = parameters.get("nprobe", 32)

            else:
                # Default to flat index
                index = faiss.IndexFlatL2(vectors.shape[1])

            # Add vectors
            index.add(vectors)

            return index

        except Exception as e:
            logger.error(f"Index building failed: {str(e)}")
            # Fallback to flat index
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(vectors)
            return index

    def _calculate_recall_score(self, all_distances: list, all_indices: list) -> float:
        """Calculate recall score (simplified implementation)"""
        try:
            if not all_distances:
                return 0.0

            # Simple recall calculation based on distance distributions
            # In practice, this would need ground truth data
            total_queries = len(all_distances)
            valid_results = 0

            for distances in all_distances:
                # Consider results valid if we have reasonable distances
                if len(distances[0]) > 0 and distances[0][0] >= 0:
                    valid_results += 1

            return valid_results / max(1, total_queries)

        except Exception:
            return 0.5  # Default recall score

    def _calculate_improvement_score(
        self,
        baseline: IndexPerformanceMetrics,
        candidate: IndexPerformanceMetrics
    ) -> float:
        """Calculate improvement score between two configurations"""
        try:
            # Weighted combination of improvements
            weights = {
                "throughput": 0.4,    # 40% weight on throughput
                "recall": 0.3,        # 30% weight on recall
                "memory": 0.2,        # 20% weight on memory efficiency
                "search_time": 0.1    # 10% weight on search time
            }

            # Calculate relative improvements
            throughput_improvement = (candidate.throughput_qps - baseline.throughput_qps) / max(1, baseline.throughput_qps)
            recall_improvement = (candidate.recall_score - baseline.recall_score) / max(0.1, baseline.recall_score)

            # Memory improvement (negative memory usage is better)
            memory_improvement = (baseline.memory_usage_mb - candidate.memory_usage_mb) / max(1, baseline.memory_usage_mb)

            # Search time improvement (lower is better)
            search_time_improvement = (baseline.search_time - candidate.search_time) / max(0.001, baseline.search_time)

            # Weighted score
            improvement_score = (
                weights["throughput"] * throughput_improvement +
                weights["recall"] * recall_improvement +
                weights["memory"] * memory_improvement +
                weights["search_time"] * search_time_improvement
            )

            return improvement_score

        except Exception as e:
            logger.warning(f"Improvement score calculation failed: {str(e)}")
            return 0.0

    async def _fine_tune_parameters(
        self,
        sample_vectors: np.ndarray,
        query_vectors: np.ndarray,
        config_name: str,
        base_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Fine-tune parameters for optimal performance"""
        try:
            logger.info(f"🔧 Fine-tuning parameters for {config_name}")

            fine_tuned = base_parameters.copy()

            if "HNSW" in config_name:
                # Fine-tune HNSW parameters
                best_ef_search = await self._optimize_ef_search(
                    sample_vectors, query_vectors, fine_tuned
                )
                fine_tuned["efSearch"] = best_ef_search

            elif "IVF" in config_name:
                # Fine-tune IVF parameters
                best_nprobe = await self._optimize_nprobe(
                    sample_vectors, query_vectors, fine_tuned
                )
                fine_tuned["nprobe"] = best_nprobe

            return fine_tuned

        except Exception as e:
            logger.warning(f"Parameter fine-tuning failed: {str(e)}")
            return base_parameters

    async def _optimize_ef_search(
        self,
        sample_vectors: np.ndarray,
        query_vectors: np.ndarray,
        parameters: dict[str, Any]
    ) -> int:
        """Optimize efSearch parameter for HNSW"""
        try:
            current_ef = parameters.get("efSearch", 128)
            candidates = [current_ef // 2, current_ef, current_ef * 2]

            best_ef = current_ef
            best_score = 0.0

            for ef_candidate in candidates:
                if ef_candidate < 16 or ef_candidate > 512:  # Reasonable bounds
                    continue

                # Quick benchmark
                test_params = parameters.copy()
                test_params["efSearch"] = ef_candidate

                # Build small test index
                test_vectors = sample_vectors[:min(1000, len(sample_vectors))]
                test_config = {"index_factory": "HNSW16", "parameters": test_params}

                test_index = await self._build_index_with_config(test_vectors, test_config)

                # Test performance
                start_time = time.time()
                test_index.search(query_vectors[:10], 5)
                search_time = time.time() - start_time

                # Score based on speed (lower is better)
                score = 1.0 / max(0.001, search_time)

                if score > best_score:
                    best_score = score
                    best_ef = ef_candidate

            return best_ef

        except Exception as e:
            logger.warning(f"efSearch optimization failed: {str(e)}")
            return parameters.get("efSearch", 128)

    async def _optimize_nprobe(
        self,
        sample_vectors: np.ndarray,
        query_vectors: np.ndarray,
        parameters: dict[str, Any]
    ) -> int:
        """Optimize nprobe parameter for IVF"""
        try:
            current_nprobe = parameters.get("nprobe", 32)
            candidates = [current_nprobe // 2, current_nprobe, current_nprobe * 2]

            best_nprobe = current_nprobe
            best_score = 0.0

            for nprobe_candidate in candidates:
                if nprobe_candidate < 8 or nprobe_candidate > 256:  # Reasonable bounds
                    continue

                # Quick benchmark
                test_params = parameters.copy()
                test_params["nprobe"] = nprobe_candidate

                # Build small test index
                test_vectors = sample_vectors[:min(1000, len(sample_vectors))]
                test_config = {"index_factory": "IVF256,Flat", "parameters": test_params}

                test_index = await self._build_index_with_config(test_vectors, test_config)

                # Test performance
                start_time = time.time()
                test_index.search(query_vectors[:10], 5)
                search_time = time.time() - start_time

                # Score based on speed
                score = 1.0 / max(0.001, search_time)

                if score > best_score:
                    best_score = score
                    best_nprobe = nprobe_candidate

            return best_nprobe

        except Exception as e:
            logger.warning(f"nprobe optimization failed: {str(e)}")
            return parameters.get("nprobe", 32)

    async def _load_optimization_cache(self):
        """Load optimization history from cache"""
        try:
            cache_file = self.optimization_cache_dir / "optimization_history.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    data = json.load(f)

                # Reconstruct optimization history
                for opt_data in data.get("optimizations", []):
                    # This is a simplified version - in practice you'd need full reconstruction
                    self.optimization_history.append({
                        "config": opt_data.get("config"),
                        "improvement": opt_data.get("improvement", 0),
                        "timestamp": opt_data.get("timestamp", time.time())
                    })

                logger.info(f"📚 Loaded {len(self.optimization_history)} optimization records from cache")

        except Exception as e:
            logger.warning(f"Optimization cache loading failed: {str(e)}")

    async def _save_optimization_cache(self):
        """Save optimization history to cache"""
        try:
            cache_file = self.optimization_cache_dir / "optimization_history.json"

            # Prepare data for serialization
            cache_data = {
                "last_updated": time.time(),
                "current_config": self.current_config,
                "optimal_parameters": self.optimal_parameters,
                "optimizations": []
            }

            # Save recent optimizations
            for opt in self.optimization_history[-50:]:  # Keep last 50
                cache_data["optimizations"].append({
                    "config": opt.optimization_strategy,
                    "improvement": opt.improvement_percentage,
                    "timestamp": time.time()
                })

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

            logger.debug("💾 Optimization cache saved")

        except Exception as e:
            logger.warning(f"Optimization cache saving failed: {str(e)}")

    async def auto_optimize_if_needed(self, current_performance: dict[str, Any]) -> bool:
        """Automatically optimize if performance has degraded"""
        try:
            if not self.auto_optimization_enabled:
                return False

            self.operations_since_optimization += 1

            # Check if optimization is needed
            should_optimize = False

            # Performance-based trigger
            current_throughput = current_performance.get("throughput_qps", 0)
            if self.performance_history:
                recent_avg = sum(p.throughput_qps for p in self.performance_history[-10:]) / min(10, len(self.performance_history))
                if current_throughput < recent_avg * self.performance_threshold:
                    should_optimize = True
                    logger.info(f"📉 Performance degradation detected: {current_throughput:.1f} < {recent_avg * self.performance_threshold:.1f}")

            # Frequency-based trigger
            if self.operations_since_optimization >= self.optimization_frequency:
                should_optimize = True
                logger.info(f"⏰ Optimization frequency reached: {self.operations_since_optimization} operations")

            if should_optimize:
                logger.info("🚀 Triggering automatic optimization...")
                self.operations_since_optimization = 0
                return True

            return False

        except Exception as e:
            logger.warning(f"Auto-optimization check failed: {str(e)}")
            return False

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics"""
        try:
            recent_optimizations = self.optimization_history[-10:]

            if recent_optimizations:
                avg_improvement = sum(opt.improvement_percentage for opt in recent_optimizations) / len(recent_optimizations)
                best_improvement = max(opt.improvement_percentage for opt in recent_optimizations)
            else:
                avg_improvement = 0.0
                best_improvement = 0.0

            return {
                "current_configuration": self.current_config,
                "optimal_parameters": self.optimal_parameters,
                "total_optimizations": len(self.optimization_history),
                "average_improvement": round(avg_improvement, 2),
                "best_improvement": round(best_improvement, 2),
                "auto_optimization_enabled": self.auto_optimization_enabled,
                "operations_since_optimization": self.operations_since_optimization,
                "optimization_frequency": self.optimization_frequency,
                "available_configurations": list(self.index_configurations.keys()),
                "performance_history_count": len(self.performance_history)
            }

        except Exception as e:
            logger.warning(f"Optimization stats collection failed: {str(e)}")
            return {"error": str(e)}

    def _get_current_memory_usage(self) -> int:
        """Get current memory usage in MB"""
        try:
            if self.gpu_service.is_gpu_available:
                usage_info = self.gpu_service.monitor_gpu_usage()
                if "torch_memory" in usage_info:
                    return usage_info["torch_memory"].get("allocated_memory_mb", 0)
                elif "system_memory" in usage_info:
                    return usage_info["system_memory"].get("used_memory_mb", 0)
            return 0
        except Exception:
            return 0
