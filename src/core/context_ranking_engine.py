"""
Context Ranking and Reranking Engine for BajajFinsev Hybrid RAG System
Advanced reranking with cross-encoder models, relevance scoring, and diversity optimization
GPU-accelerated with RTX 3050 optimizations for efficient inference
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict
import hashlib

# ML libraries for reranking
try:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import CrossEncoder
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Ranking algorithms
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Core integrations
from src.core.hybrid_retrieval_engine import HybridRetrievalResult, RetrievalResult
from src.core.query_enhancement_engine import EnhancedQuery, QueryIntent, QueryComplexity
from src.core.gpu_service import get_gpu_service
from src.services.redis_cache import get_redis_cache

logger = logging.getLogger(__name__)


class RankingStrategy(Enum):
    """Available ranking strategies"""
    RELEVANCE_SCORE = "relevance_score"
    CROSS_ENCODER = "cross_encoder"
    DIVERSITY_MMR = "diversity_mmr"          # Maximal Marginal Relevance
    LISTWISE_RANKING = "listwise_ranking"
    QUERY_LIKELIHOOD = "query_likelihood"
    BM25_RERANKING = "bm25_reranking"
    NEURAL_RERANKING = "neural_reranking"


class DiversityMethod(Enum):
    """Methods for ensuring result diversity"""
    MMR = "maximal_marginal_relevance"
    CLUSTERING = "clustering_based"
    GREEDY_SELECTION = "greedy_selection"
    SUBMODULAR = "submodular_optimization"


@dataclass
class RankingResult:
    """Individual ranking result with detailed scoring"""
    document_id: str
    content: str
    original_rank: int
    reranked_position: int
    
    # Scoring breakdown
    relevance_score: float
    diversity_score: float
    quality_score: float
    final_score: float
    
    # Ranking metadata
    ranking_strategy: RankingStrategy
    confidence: float
    explanation: str
    
    # Cross-encoder specific
    cross_encoder_score: Optional[float] = None
    query_document_similarity: Optional[float] = None
    
    # Processing info
    gpu_processed: bool = False
    processing_time: float = 0.0


@dataclass
class ContextRankingResult:
    """Complete context ranking result"""
    query: str
    enhanced_query: EnhancedQuery
    original_results: List[RetrievalResult]
    
    # Reranked results
    ranked_results: List[RankingResult] = field(default_factory=list)
    ranking_strategy: RankingStrategy = RankingStrategy.RELEVANCE_SCORE
    diversity_method: Optional[DiversityMethod] = None
    
    # Performance metrics
    total_processing_time: float = 0.0
    ranking_improvement: float = 0.0  # Quality improvement from reranking
    diversity_improvement: float = 0.0
    
    # Quality metrics
    result_coherence: float = 0.0
    coverage_breadth: float = 0.0
    ranking_confidence: float = 0.0
    
    # GPU utilization
    gpu_acceleration_used: bool = False
    cross_encoder_time: float = 0.0


@dataclass
class RankingStats:
    """Ranking engine statistics"""
    total_ranking_operations: int = 0
    successful_operations: int = 0
    
    # Strategy usage
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    strategy_performance: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    avg_ranking_time: float = 0.0
    avg_cross_encoder_time: float = 0.0
    avg_results_processed: float = 0.0
    gpu_usage_rate: float = 0.0
    
    # Quality improvements
    avg_ranking_improvement: float = 0.0
    avg_diversity_improvement: float = 0.0
    avg_coherence_score: float = 0.0


class ContextRankingEngine:
    """
    Advanced context ranking and reranking engine
    Supports multiple ranking strategies with GPU acceleration
    """
    
    # Cross-encoder models for reranking
    CROSS_ENCODER_MODELS = {
        "ms-marco-MiniLM": {
            "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_length": 512,
            "batch_size": 16,  # RTX 3050 optimized
            "description": "Fast cross-encoder for passage ranking"
        },
        "ms-marco-distilbert": {
            "name": "cross-encoder/ms-marco-DistilBERT-L-2-v2",
            "max_length": 512,
            "batch_size": 12,  # Slightly smaller for memory
            "description": "Balanced speed/accuracy cross-encoder"
        }
    }
    
    # Ranking strategy configurations
    RANKING_CONFIGS = {
        RankingStrategy.RELEVANCE_SCORE: {
            "weight_relevance": 0.7,
            "weight_diversity": 0.2,
            "weight_quality": 0.1,
            "description": "Basic relevance-based ranking"
        },
        RankingStrategy.CROSS_ENCODER: {
            "model": "ms-marco-MiniLM",
            "batch_size": 16,
            "enable_gpu": True,
            "description": "Neural cross-encoder reranking"
        },
        RankingStrategy.DIVERSITY_MMR: {
            "lambda_param": 0.7,  # Trade-off between relevance and diversity
            "similarity_threshold": 0.8,
            "description": "Maximal Marginal Relevance for diversity"
        },
        RankingStrategy.BM25_RERANKING: {
            "k1": 1.5,
            "b": 0.75,
            "description": "BM25 statistical ranking"
        }
    }
    
    def __init__(self,
                 default_strategy: RankingStrategy = RankingStrategy.CROSS_ENCODER,
                 cross_encoder_model: str = "ms-marco-MiniLM",
                 enable_gpu_acceleration: bool = True,
                 enable_diversity: bool = True,
                 diversity_method: DiversityMethod = DiversityMethod.MMR,
                 max_rerank_size: int = 100,
                 enable_caching: bool = True,
                 cache_ttl: int = 1800):
        """
        Initialize context ranking engine
        
        Args:
            default_strategy: Default ranking strategy
            cross_encoder_model: Cross-encoder model to use
            enable_gpu_acceleration: Enable GPU for neural models
            enable_diversity: Enable diversity optimization
            diversity_method: Method for ensuring diversity
            max_rerank_size: Maximum number of results to rerank
            enable_caching: Enable ranking result caching
            cache_ttl: Cache TTL in seconds
        """
        self.default_strategy = default_strategy
        self.cross_encoder_model_name = cross_encoder_model
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.enable_diversity = enable_diversity
        self.diversity_method = diversity_method
        self.max_rerank_size = max_rerank_size
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Initialize services
        self.gpu_service = get_gpu_service() if enable_gpu_acceleration else None
        self.redis_cache = get_redis_cache() if enable_caching else None
        
        # Cross-encoder model
        self.cross_encoder = None
        self.cross_encoder_device = "cpu"
        self.is_initialized = False
        
        # Statistics
        self.stats = RankingStats(
            strategy_usage=defaultdict(int),
            strategy_performance=defaultdict(float)
        )
        
        # Caching
        self.ranking_cache = {}  # In-memory cache for small results
        
        logger.info("ContextRankingEngine initialized")
        logger.info(f"Default strategy: {default_strategy.value}")
        logger.info(f"Cross-encoder model: {cross_encoder_model}")
        logger.info(f"GPU acceleration: {enable_gpu_acceleration}")
        logger.info(f"Diversity enabled: {enable_diversity}")
    
    async def initialize(self) -> bool:
        """Initialize ranking engine and models"""
        logger.info("🔄 Initializing context ranking engine...")
        
        try:
            # Initialize cross-encoder if needed
            if (self.default_strategy == RankingStrategy.CROSS_ENCODER or 
                self.default_strategy == RankingStrategy.NEURAL_RERANKING):
                await self._initialize_cross_encoder()
            
            self.is_initialized = True
            logger.info("✅ Context ranking engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize ranking engine: {str(e)}")
            return False
    
    async def _initialize_cross_encoder(self):
        """Initialize cross-encoder model with GPU optimization"""
        if not TORCH_AVAILABLE:
            logger.warning("⚠️ PyTorch not available, cross-encoder disabled")
            return
        
        try:
            model_config = self.CROSS_ENCODER_MODELS[self.cross_encoder_model_name]
            model_name = model_config["name"]
            
            logger.info(f"🔄 Loading cross-encoder: {model_name}")
            
            # Load model
            self.cross_encoder = CrossEncoder(model_name, max_length=model_config["max_length"])
            
            # Configure device
            if self.enable_gpu_acceleration and self.gpu_service:
                gpu_config = self.gpu_service.get_device_config()
                if gpu_config and gpu_config.device != "cpu":
                    self.cross_encoder_device = gpu_config.device
                    self.cross_encoder.model.to(self.cross_encoder_device)
                    
                    # Enable mixed precision for RTX 3050
                    if gpu_config.mixed_precision:
                        self.cross_encoder.model = self.cross_encoder.model.half()
                        logger.info("🎮 Cross-encoder using mixed precision (FP16)")
                    
                    logger.info(f"🎮 Cross-encoder loaded on: {self.cross_encoder_device}")
                else:
                    self.cross_encoder_device = "cpu"
                    logger.info("💻 Cross-encoder loaded on CPU")
            else:
                self.cross_encoder_device = "cpu"
                logger.info("💻 Cross-encoder loaded on CPU")
            
            # Set to evaluation mode
            self.cross_encoder.model.eval()
            
            logger.info(f"✅ Cross-encoder initialized: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Cross-encoder initialization failed: {str(e)}")
            self.cross_encoder = None
    
    async def rank_results(self,
                          retrieval_result: HybridRetrievalResult,
                          strategy: Optional[RankingStrategy] = None,
                          top_k: Optional[int] = None,
                          **kwargs) -> ContextRankingResult:
        """
        Rank and rerank retrieval results for better relevance
        
        Args:
            retrieval_result: Results from hybrid retrieval
            strategy: Ranking strategy to use (None = default)
            top_k: Number of top results to return (None = all)
            **kwargs: Additional ranking parameters
            
        Returns:
            ContextRankingResult with reranked results
        """
        start_time = time.time()
        self.stats.total_ranking_operations += 1
        
        query = retrieval_result.query
        logger.debug(f"📊 Ranking results for: {query}")
        logger.debug(f"Original results: {len(retrieval_result.final_results)}")
        
        # Check cache
        cache_key = None
        if self.enable_caching and self.redis_cache:
            cache_key = self._generate_cache_key(retrieval_result, strategy, top_k)
            cached_result = await self._get_cached_ranking(cache_key)
            if cached_result:
                logger.debug("✅ Using cached ranking result")
                return cached_result
        
        try:
            # Use default strategy if not specified
            if strategy is None:
                strategy = self._select_ranking_strategy(retrieval_result.enhanced_query)
            
            # Limit results for performance
            results_to_rank = retrieval_result.final_results[:self.max_rerank_size]
            
            if not results_to_rank:
                logger.warning("⚠️ No results to rank")
                return ContextRankingResult(
                    query=query,
                    enhanced_query=retrieval_result.enhanced_query,
                    original_results=retrieval_result.final_results,
                    total_processing_time=time.time() - start_time
                )
            
            # Initialize ranking result
            ranking_result = ContextRankingResult(
                query=query,
                enhanced_query=retrieval_result.enhanced_query,
                original_results=results_to_rank,
                ranking_strategy=strategy
            )
            
            # Execute ranking strategy
            if strategy == RankingStrategy.CROSS_ENCODER:
                ranked_results = await self._cross_encoder_ranking(
                    results_to_rank, retrieval_result.enhanced_query, **kwargs
                )
                ranking_result.gpu_acceleration_used = self.cross_encoder_device != "cpu"
                
            elif strategy == RankingStrategy.DIVERSITY_MMR:
                ranked_results = await self._mmr_ranking(
                    results_to_rank, retrieval_result.enhanced_query, **kwargs
                )
                ranking_result.diversity_method = DiversityMethod.MMR
                
            elif strategy == RankingStrategy.NEURAL_RERANKING:
                ranked_results = await self._neural_reranking(
                    results_to_rank, retrieval_result.enhanced_query, **kwargs
                )
                ranking_result.gpu_acceleration_used = True
                
            elif strategy == RankingStrategy.BM25_RERANKING:
                ranked_results = await self._bm25_ranking(
                    results_to_rank, retrieval_result.enhanced_query, **kwargs
                )
                
            else:
                # Default relevance score ranking
                ranked_results = await self._relevance_score_ranking(
                    results_to_rank, retrieval_result.enhanced_query, **kwargs
                )
            
            # Apply diversity optimization if enabled
            if self.enable_diversity and strategy != RankingStrategy.DIVERSITY_MMR:
                ranked_results = await self._apply_diversity_optimization(
                    ranked_results, retrieval_result.enhanced_query
                )
                ranking_result.diversity_method = self.diversity_method
            
            # Apply top-k filtering
            if top_k:
                ranked_results = ranked_results[:top_k]
            
            ranking_result.ranked_results = ranked_results
            
            # Calculate quality metrics
            await self._calculate_ranking_metrics(ranking_result)
            
            # Finalize result
            ranking_result.total_processing_time = time.time() - start_time
            
            # Update statistics
            self.stats.successful_operations += 1
            self._update_ranking_stats(ranking_result, strategy)
            
            # Cache result
            if self.enable_caching and cache_key:
                await self._cache_ranking_result(cache_key, ranking_result)
            
            logger.info(f"✅ Ranking completed: {len(ranked_results)} results in {ranking_result.total_processing_time:.3f}s")
            
            return ranking_result
            
        except Exception as e:
            logger.error(f"❌ Ranking failed: {str(e)}")
            
            # Return original results as fallback
            return ContextRankingResult(
                query=query,
                enhanced_query=retrieval_result.enhanced_query,
                original_results=retrieval_result.final_results,
                total_processing_time=time.time() - start_time
            )
    
    def _select_ranking_strategy(self, enhanced_query: EnhancedQuery) -> RankingStrategy:
        """Select optimal ranking strategy based on query analysis"""
        analysis = enhanced_query.analysis
        
        # High complexity queries benefit from cross-encoder
        if analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            if self.cross_encoder:
                return RankingStrategy.CROSS_ENCODER
            else:
                return RankingStrategy.NEURAL_RERANKING
        
        # Comparative queries need diversity
        if analysis.intent == QueryIntent.COMPARATIVE:
            return RankingStrategy.DIVERSITY_MMR
        
        # Numerical/factual queries can use BM25
        if analysis.intent in [QueryIntent.NUMERICAL, QueryIntent.FACTUAL]:
            return RankingStrategy.BM25_RERANKING
        
        # Default to cross-encoder if available
        if self.cross_encoder:
            return RankingStrategy.CROSS_ENCODER
        else:
            return RankingStrategy.RELEVANCE_SCORE
    
    async def _cross_encoder_ranking(self,
                                   results: List[RetrievalResult],
                                   enhanced_query: EnhancedQuery,
                                   **kwargs) -> List[RankingResult]:
        """Cross-encoder neural reranking"""
        if not self.cross_encoder:
            logger.warning("⚠️ Cross-encoder not available, falling back to relevance scoring")
            return await self._relevance_score_ranking(results, enhanced_query, **kwargs)
        
        query_text = enhanced_query.analysis.expanded_query or enhanced_query.analysis.original_query
        
        # Prepare query-document pairs
        pairs = []
        for result in results:
            # Truncate document content for cross-encoder
            doc_content = result.document.content[:500]  # Limit content length
            pairs.append([query_text, doc_content])
        
        try:
            cross_encoder_start = time.time()
            
            # Batch processing for GPU efficiency
            model_config = self.CROSS_ENCODER_MODELS[self.cross_encoder_model_name]
            batch_size = model_config["batch_size"]
            
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # GPU memory management for RTX 3050
                if self.cross_encoder_device != "cpu" and self.gpu_service:
                    self.gpu_service.clear_gpu_cache()
                
                # Predict scores
                batch_scores = self.cross_encoder.predict(batch_pairs)
                all_scores.extend(batch_scores)
            
            cross_encoder_time = time.time() - cross_encoder_start
            
            # Create ranking results
            ranked_results = []
            score_rank_pairs = list(zip(all_scores, range(len(results))))
            score_rank_pairs.sort(key=lambda x: x[0], reverse=True)
            
            for new_rank, (score, original_rank) in enumerate(score_rank_pairs, 1):
                original_result = results[original_rank]
                
                ranking_result = RankingResult(
                    document_id=original_result.document.id,
                    content=original_result.document.content,
                    original_rank=original_rank + 1,
                    reranked_position=new_rank,
                    relevance_score=float(score),
                    diversity_score=0.0,  # Not calculated in cross-encoder
                    quality_score=original_result.score,
                    final_score=float(score),
                    ranking_strategy=RankingStrategy.CROSS_ENCODER,
                    confidence=min(1.0, abs(float(score))),
                    explanation=f"Cross-encoder score: {score:.3f}",
                    cross_encoder_score=float(score),
                    gpu_processed=self.cross_encoder_device != "cpu",
                    processing_time=cross_encoder_time / len(results)
                )
                
                ranked_results.append(ranking_result)
            
            logger.debug(f"✅ Cross-encoder ranking: {len(ranked_results)} results in {cross_encoder_time:.3f}s")
            
            return ranked_results
            
        except Exception as e:
            logger.error(f"❌ Cross-encoder ranking failed: {str(e)}")
            # Fallback to relevance scoring
            return await self._relevance_score_ranking(results, enhanced_query, **kwargs)
    
    async def _mmr_ranking(self,
                          results: List[RetrievalResult],
                          enhanced_query: EnhancedQuery,
                          **kwargs) -> List[RankingResult]:
        """Maximal Marginal Relevance (MMR) ranking for diversity"""
        config = self.RANKING_CONFIGS[RankingStrategy.DIVERSITY_MMR]
        lambda_param = kwargs.get('lambda_param', config['lambda_param'])
        
        if not results:
            return []
        
        # Use embeddings if available
        if not enhanced_query.original_embedding:
            logger.warning("⚠️ No query embedding available for MMR")
            return await self._relevance_score_ranking(results, enhanced_query, **kwargs)
        
        query_embedding = np.array(enhanced_query.original_embedding)
        
        # Extract document embeddings (simplified - using scores as proxy)
        doc_similarities = [result.score for result in results]
        
        # MMR algorithm
        selected_indices = []
        remaining_indices = list(range(len(results)))
        
        while remaining_indices and len(selected_indices) < len(results):
            mmr_scores = []
            
            for i in remaining_indices:
                relevance = doc_similarities[i]
                
                # Calculate diversity (maximum similarity to already selected)
                if not selected_indices:
                    diversity = 0.0
                else:
                    # Simplified diversity calculation using content overlap
                    max_similarity = 0.0
                    current_content = results[i].document.content[:200].lower()
                    
                    for selected_idx in selected_indices:
                        selected_content = results[selected_idx].document.content[:200].lower()
                        
                        # Simple word overlap similarity
                        current_words = set(current_content.split())
                        selected_words = set(selected_content.split())
                        
                        if current_words and selected_words:
                            overlap = len(current_words.intersection(selected_words))
                            total = len(current_words.union(selected_words))
                            similarity = overlap / total if total > 0 else 0.0
                            max_similarity = max(max_similarity, similarity)
                    
                    diversity = max_similarity
                
                # MMR score: λ * relevance - (1-λ) * diversity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((mmr_score, i))
            
            # Select best MMR score
            best_mmr_score, best_idx = max(mmr_scores)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Create ranking results
        ranked_results = []
        for new_rank, original_idx in enumerate(selected_indices, 1):
            original_result = results[original_idx]
            
            relevance_score = doc_similarities[original_idx]
            diversity_score = 1.0 - (new_rank - 1) / len(selected_indices)  # Higher for earlier selections
            
            ranking_result = RankingResult(
                document_id=original_result.document.id,
                content=original_result.document.content,
                original_rank=original_idx + 1,
                reranked_position=new_rank,
                relevance_score=relevance_score,
                diversity_score=diversity_score,
                quality_score=original_result.score,
                final_score=lambda_param * relevance_score + (1 - lambda_param) * diversity_score,
                ranking_strategy=RankingStrategy.DIVERSITY_MMR,
                confidence=relevance_score,
                explanation=f"MMR score: {relevance_score:.3f} (λ={lambda_param})",
                gpu_processed=False
            )
            
            ranked_results.append(ranking_result)
        
        return ranked_results
    
    async def _neural_reranking(self,
                               results: List[RetrievalResult],
                               enhanced_query: EnhancedQuery,
                               **kwargs) -> List[RankingResult]:
        """Neural reranking using embeddings and similarity scores"""
        
        # Use embedding-based scoring if available
        if enhanced_query.original_embedding and SKLEARN_AVAILABLE:
            query_embedding = np.array(enhanced_query.original_embedding).reshape(1, -1)
            
            # Collect document embeddings (if available from vector storage results)
            doc_embeddings = []
            embedding_available = []
            
            for result in results:
                # Check if document has embedding stored
                if hasattr(result.document, 'embedding') and result.document.embedding is not None:
                    doc_embeddings.append(result.document.embedding)
                    embedding_available.append(True)
                else:
                    doc_embeddings.append(np.zeros_like(query_embedding[0]))  # Placeholder
                    embedding_available.append(False)
            
            if any(embedding_available):
                doc_embeddings_matrix = np.array(doc_embeddings)
                
                # Calculate cosine similarities
                similarities = cosine_similarity(query_embedding, doc_embeddings_matrix)[0]
                
                # Combine with original scores
                combined_scores = []
                for i, (similarity, result, has_embedding) in enumerate(zip(similarities, results, embedding_available)):
                    if has_embedding:
                        # Weighted combination of original score and embedding similarity
                        combined_score = 0.6 * similarity + 0.4 * result.score
                    else:
                        combined_score = result.score
                    
                    combined_scores.append((combined_score, i))
                
                # Sort by combined scores
                combined_scores.sort(key=lambda x: x[0], reverse=True)
                
                # Create ranking results
                ranked_results = []
                for new_rank, (score, original_idx) in enumerate(combined_scores, 1):
                    original_result = results[original_idx]
                    
                    ranking_result = RankingResult(
                        document_id=original_result.document.id,
                        content=original_result.document.content,
                        original_rank=original_idx + 1,
                        reranked_position=new_rank,
                        relevance_score=float(score),
                        diversity_score=0.0,
                        quality_score=original_result.score,
                        final_score=float(score),
                        ranking_strategy=RankingStrategy.NEURAL_RERANKING,
                        confidence=float(score),
                        explanation=f"Neural reranking score: {score:.3f}",
                        query_document_similarity=float(similarities[original_idx]) if embedding_available[original_idx] else None,
                        gpu_processed=False
                    )
                    
                    ranked_results.append(ranking_result)
                
                return ranked_results
        
        # Fallback to relevance scoring
        return await self._relevance_score_ranking(results, enhanced_query, **kwargs)
    
    async def _bm25_ranking(self,
                           results: List[RetrievalResult],
                           enhanced_query: EnhancedQuery,
                           **kwargs) -> List[RankingResult]:
        """BM25 statistical ranking"""
        config = self.RANKING_CONFIGS[RankingStrategy.BM25_RERANKING]
        k1 = kwargs.get('k1', config['k1'])
        b = kwargs.get('b', config['b'])
        
        query_text = enhanced_query.analysis.original_query.lower()
        query_terms = query_text.split()
        
        if not query_terms:
            return await self._relevance_score_ranking(results, enhanced_query, **kwargs)
        
        # Calculate BM25 scores
        doc_lengths = [len(result.document.content.split()) for result in results]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        bm25_scores = []
        for i, result in enumerate(results):
            doc_text = result.document.content.lower()
            doc_terms = doc_text.split()
            doc_length = doc_lengths[i]
            
            score = 0.0
            for term in query_terms:
                # Term frequency in document
                tf = doc_terms.count(term)
                
                if tf > 0:
                    # Document frequency (simplified - assume all docs have the term)
                    df = 1  # Simplified
                    N = len(results)
                    
                    # IDF calculation
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
                    
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                    
                    score += idf * (numerator / denominator)
            
            bm25_scores.append((score, i))
        
        # Sort by BM25 scores
        bm25_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Create ranking results
        ranked_results = []
        for new_rank, (score, original_idx) in enumerate(bm25_scores, 1):
            original_result = results[original_idx]
            
            ranking_result = RankingResult(
                document_id=original_result.document.id,
                content=original_result.document.content,
                original_rank=original_idx + 1,
                reranked_position=new_rank,
                relevance_score=float(score),
                diversity_score=0.0,
                quality_score=original_result.score,
                final_score=float(score),
                ranking_strategy=RankingStrategy.BM25_RERANKING,
                confidence=min(1.0, float(score) / 10.0),  # Normalize confidence
                explanation=f"BM25 score: {score:.3f} (k1={k1}, b={b})",
                gpu_processed=False
            )
            
            ranked_results.append(ranking_result)
        
        return ranked_results
    
    async def _relevance_score_ranking(self,
                                      results: List[RetrievalResult],
                                      enhanced_query: EnhancedQuery,
                                      **kwargs) -> List[RankingResult]:
        """Basic relevance score ranking (fallback method)"""
        config = self.RANKING_CONFIGS[RankingStrategy.RELEVANCE_SCORE]
        
        # Sort by original scores
        scored_results = [(result.score, i, result) for i, result in enumerate(results)]
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Create ranking results
        ranked_results = []
        for new_rank, (score, original_idx, original_result) in enumerate(scored_results, 1):
            
            # Calculate composite score
            relevance_score = score
            quality_score = original_result.confidence if hasattr(original_result, 'confidence') else score
            diversity_score = 1.0 - (new_rank - 1) / len(results)  # Simple position-based diversity
            
            final_score = (
                config['weight_relevance'] * relevance_score +
                config['weight_quality'] * quality_score +
                config['weight_diversity'] * diversity_score
            )
            
            ranking_result = RankingResult(
                document_id=original_result.document.id,
                content=original_result.document.content,
                original_rank=original_idx + 1,
                reranked_position=new_rank,
                relevance_score=relevance_score,
                diversity_score=diversity_score,
                quality_score=quality_score,
                final_score=final_score,
                ranking_strategy=RankingStrategy.RELEVANCE_SCORE,
                confidence=relevance_score,
                explanation=f"Composite score: {final_score:.3f} (R:{relevance_score:.3f}, D:{diversity_score:.3f})",
                gpu_processed=False
            )
            
            ranked_results.append(ranking_result)
        
        return ranked_results
    
    async def _apply_diversity_optimization(self,
                                          ranked_results: List[RankingResult],
                                          enhanced_query: EnhancedQuery) -> List[RankingResult]:
        """Apply diversity optimization to ranking results"""
        
        if self.diversity_method == DiversityMethod.MMR:
            return await self._apply_mmr_diversity(ranked_results, enhanced_query)
        elif self.diversity_method == DiversityMethod.CLUSTERING:
            return await self._apply_clustering_diversity(ranked_results)
        elif self.diversity_method == DiversityMethod.GREEDY_SELECTION:
            return await self._apply_greedy_diversity(ranked_results)
        else:
            return ranked_results  # No additional diversity optimization
    
    async def _apply_mmr_diversity(self,
                                  ranked_results: List[RankingResult],
                                  enhanced_query: EnhancedQuery,
                                  lambda_param: float = 0.7) -> List[RankingResult]:
        """Apply MMR-based diversity to already ranked results"""
        
        if len(ranked_results) <= 1:
            return ranked_results
        
        # Re-rank using MMR with current relevance scores
        selected_indices = []
        remaining_indices = list(range(len(ranked_results)))
        
        while remaining_indices and len(selected_indices) < len(ranked_results):
            mmr_scores = []
            
            for i in remaining_indices:
                relevance = ranked_results[i].relevance_score
                
                # Calculate diversity
                if not selected_indices:
                    diversity = 0.0
                else:
                    max_similarity = 0.0
                    current_content = ranked_results[i].content[:200].lower()
                    
                    for selected_idx in selected_indices:
                        selected_content = ranked_results[selected_idx].content[:200].lower()
                        
                        # Calculate content similarity
                        current_words = set(current_content.split())
                        selected_words = set(selected_content.split())
                        
                        if current_words and selected_words:
                            overlap = len(current_words.intersection(selected_words))
                            total = len(current_words.union(selected_words))
                            similarity = overlap / total if total > 0 else 0.0
                            max_similarity = max(max_similarity, similarity)
                    
                    diversity = max_similarity
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((mmr_score, i))
            
            # Select best MMR score
            best_mmr_score, best_idx = max(mmr_scores)
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Reorder results based on MMR selection
        diversified_results = []
        for new_rank, original_idx in enumerate(selected_indices, 1):
            result = ranked_results[original_idx]
            result.reranked_position = new_rank
            result.diversity_score = 1.0 - (new_rank - 1) / len(selected_indices)
            diversified_results.append(result)
        
        return diversified_results
    
    async def _apply_clustering_diversity(self, ranked_results: List[RankingResult]) -> List[RankingResult]:
        """Apply clustering-based diversity"""
        if not SKLEARN_AVAILABLE or len(ranked_results) <= 3:
            return ranked_results
        
        try:
            # Simple clustering based on content similarity
            contents = [result.content[:300] for result in ranked_results]
            
            # Create simple feature vectors (word counts)
            all_words = set()
            for content in contents:
                all_words.update(content.lower().split())
            
            word_list = list(all_words)[:100]  # Limit vocabulary
            
            # Create feature vectors
            feature_vectors = []
            for content in contents:
                content_words = content.lower().split()
                vector = [content_words.count(word) for word in word_list]
                feature_vectors.append(vector)
            
            if len(feature_vectors) == 0:
                return ranked_results
            
            feature_matrix = np.array(feature_vectors)
            
            # Perform clustering
            n_clusters = min(5, len(ranked_results) // 2)  # Reasonable number of clusters
            if n_clusters < 2:
                return ranked_results
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(feature_matrix)
            
            # Select diverse results from different clusters
            diversified_results = []
            cluster_representatives = {}
            
            # Find best result from each cluster
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_representatives:
                    cluster_representatives[cluster_id] = []
                cluster_representatives[cluster_id].append((ranked_results[i], i))
            
            # Sort clusters by their best result's score
            cluster_order = []
            for cluster_id, cluster_results in cluster_representatives.items():
                best_result = max(cluster_results, key=lambda x: x[0].relevance_score)
                cluster_order.append((best_result[0].relevance_score, cluster_id))
            
            cluster_order.sort(reverse=True)
            
            # Select results from clusters in order
            new_rank = 1
            for _, cluster_id in cluster_order:
                cluster_results = sorted(
                    cluster_representatives[cluster_id],
                    key=lambda x: x[0].relevance_score,
                    reverse=True
                )
                
                # Add top results from this cluster
                for result, _ in cluster_results:
                    result.reranked_position = new_rank
                    result.diversity_score = 1.0 - (new_rank - 1) / len(ranked_results)
                    diversified_results.append(result)
                    new_rank += 1
            
            return diversified_results
            
        except Exception as e:
            logger.warning(f"⚠️ Clustering diversity failed: {e}")
            return ranked_results
    
    async def _apply_greedy_diversity(self, ranked_results: List[RankingResult]) -> List[RankingResult]:
        """Apply greedy diversity selection"""
        if len(ranked_results) <= 1:
            return ranked_results
        
        selected_results = []
        remaining_results = ranked_results.copy()
        
        # Start with the highest scoring result
        best_result = max(remaining_results, key=lambda x: x.relevance_score)
        selected_results.append(best_result)
        remaining_results.remove(best_result)
        
        # Greedily select diverse results
        while remaining_results:
            best_diversity_score = -1
            best_result = None
            
            for candidate in remaining_results:
                # Calculate diversity relative to selected results
                min_similarity = float('inf')
                candidate_words = set(candidate.content[:200].lower().split())
                
                for selected in selected_results:
                    selected_words = set(selected.content[:200].lower().split())
                    
                    if candidate_words and selected_words:
                        overlap = len(candidate_words.intersection(selected_words))
                        total = len(candidate_words.union(selected_words))
                        similarity = overlap / total if total > 0 else 0.0
                        min_similarity = min(min_similarity, similarity)
                
                # Diversity score combines relevance and dissimilarity
                diversity_score = 0.6 * candidate.relevance_score + 0.4 * (1.0 - min_similarity)
                
                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_result = candidate
            
            if best_result:
                selected_results.append(best_result)
                remaining_results.remove(best_result)
        
        # Update positions and diversity scores
        for i, result in enumerate(selected_results, 1):
            result.reranked_position = i
            result.diversity_score = 1.0 - (i - 1) / len(selected_results)
        
        return selected_results
    
    async def _calculate_ranking_metrics(self, ranking_result: ContextRankingResult):
        """Calculate quality metrics for ranking result"""
        if not ranking_result.ranked_results:
            return
        
        # Ranking improvement (compare original vs reranked positions)
        position_improvements = []
        for result in ranking_result.ranked_results:
            if result.original_rank != result.reranked_position:
                improvement = result.original_rank - result.reranked_position
                position_improvements.append(improvement)
        
        if position_improvements:
            ranking_result.ranking_improvement = sum(position_improvements) / len(position_improvements)
        
        # Diversity improvement (measure content diversity in top results)
        top_results = ranking_result.ranked_results[:10]
        if len(top_results) > 1:
            diversity_scores = []
            
            for i, result1 in enumerate(top_results):
                for result2 in top_results[i+1:]:
                    content1 = set(result1.content[:200].lower().split())
                    content2 = set(result2.content[:200].lower().split())
                    
                    if content1 and content2:
                        overlap = len(content1.intersection(content2))
                        total = len(content1.union(content2))
                        similarity = overlap / total if total > 0 else 0.0
                        diversity_scores.append(1.0 - similarity)
            
            if diversity_scores:
                ranking_result.diversity_improvement = sum(diversity_scores) / len(diversity_scores)
        
        # Result coherence (consistency of scores in top results)
        if len(ranking_result.ranked_results) >= 3:
            top_scores = [result.final_score for result in ranking_result.ranked_results[:5]]
            score_variance = np.var(top_scores) if len(top_scores) > 1 else 0.0
            ranking_result.result_coherence = max(0.0, 1.0 - score_variance)
        
        # Coverage breadth (how many different strategies contributed)
        strategies_used = set(result.ranking_strategy for result in ranking_result.ranked_results)
        ranking_result.coverage_breadth = len(strategies_used) / len(RankingStrategy)
        
        # Ranking confidence (average confidence of top results)
        if ranking_result.ranked_results:
            top_confidences = [result.confidence for result in ranking_result.ranked_results[:5]]
            ranking_result.ranking_confidence = sum(top_confidences) / len(top_confidences)
    
    def _generate_cache_key(self,
                           retrieval_result: HybridRetrievalResult,
                           strategy: Optional[RankingStrategy],
                           top_k: Optional[int]) -> str:
        """Generate cache key for ranking result"""
        # Use query and result hashes for caching
        query_hash = hashlib.md5(retrieval_result.query.encode()).hexdigest()[:16]
        
        # Create results signature
        result_sig = "|".join([
            f"{r.document.id}:{r.score:.3f}" 
            for r in retrieval_result.final_results[:20]  # Limit for cache key size
        ])
        result_hash = hashlib.md5(result_sig.encode()).hexdigest()[:16]
        
        strategy_str = strategy.value if strategy else "default"
        top_k_str = str(top_k) if top_k else "all"
        
        return f"ranking:{query_hash}:{result_hash}:{strategy_str}:{top_k_str}"
    
    async def _get_cached_ranking(self, cache_key: str) -> Optional[ContextRankingResult]:
        """Get cached ranking result"""
        # Check in-memory cache first
        if cache_key in self.ranking_cache:
            return self.ranking_cache[cache_key]
        
        # Check Redis cache
        if self.redis_cache:
            try:
                cached_data = await self.redis_cache.get(cache_key)
                if cached_data:
                    # Would need proper deserialization
                    return None  # TODO: Implement proper serialization
            except Exception as e:
                logger.debug(f"Cache get failed: {e}")
        
        return None
    
    async def _cache_ranking_result(self, cache_key: str, result: ContextRankingResult):
        """Cache ranking result"""
        # Store in in-memory cache (limited size)
        if len(self.ranking_cache) < 100:  # Limit memory usage
            self.ranking_cache[cache_key] = result
        
        # Store in Redis cache
        if self.redis_cache:
            try:
                # Would need proper serialization
                # await self.redis_cache.setex(cache_key, self.cache_ttl, serialized_result)
                pass  # TODO: Implement proper serialization
            except Exception as e:
                logger.debug(f"Cache set failed: {e}")
    
    def _update_ranking_stats(self, result: ContextRankingResult, strategy: RankingStrategy):
        """Update ranking statistics"""
        # Update strategy usage
        self.stats.strategy_usage[strategy.value] += 1
        
        # Update performance metrics
        current_avg = self.stats.strategy_performance[strategy.value]
        usage_count = self.stats.strategy_usage[strategy.value]
        
        new_avg = ((current_avg * (usage_count - 1)) + result.total_processing_time) / usage_count
        self.stats.strategy_performance[strategy.value] = new_avg
        
        # Update overall statistics
        total_ops = self.stats.total_ranking_operations
        
        # Average ranking time
        current_time_avg = self.stats.avg_ranking_time
        new_time_avg = ((current_time_avg * (total_ops - 1)) + result.total_processing_time) / total_ops
        self.stats.avg_ranking_time = new_time_avg
        
        # Average results processed
        current_results_avg = self.stats.avg_results_processed
        new_results_avg = ((current_results_avg * (total_ops - 1)) + len(result.ranked_results)) / total_ops
        self.stats.avg_results_processed = new_results_avg
        
        # Quality improvements
        if result.ranking_improvement > 0:
            current_improvement = self.stats.avg_ranking_improvement
            new_improvement = ((current_improvement * (total_ops - 1)) + result.ranking_improvement) / total_ops
            self.stats.avg_ranking_improvement = new_improvement
        
        if result.diversity_improvement > 0:
            current_diversity = self.stats.avg_diversity_improvement
            new_diversity = ((current_diversity * (total_ops - 1)) + result.diversity_improvement) / total_ops
            self.stats.avg_diversity_improvement = new_diversity
        
        if result.result_coherence > 0:
            current_coherence = self.stats.avg_coherence_score
            new_coherence = ((current_coherence * (total_ops - 1)) + result.result_coherence) / total_ops
            self.stats.avg_coherence_score = new_coherence
        
        # GPU usage
        if result.gpu_acceleration_used:
            gpu_usage = (self.stats.gpu_usage_rate * (total_ops - 1) + 1) / total_ops
            self.stats.gpu_usage_rate = gpu_usage
        
        # Cross-encoder time
        if result.cross_encoder_time > 0:
            current_ce_time = self.stats.avg_cross_encoder_time
            new_ce_time = ((current_ce_time * (total_ops - 1)) + result.cross_encoder_time) / total_ops
            self.stats.avg_cross_encoder_time = new_ce_time
    
    async def get_ranking_stats(self) -> Dict[str, Any]:
        """Get comprehensive ranking statistics"""
        return {
            "engine_config": {
                "default_strategy": self.default_strategy.value,
                "cross_encoder_model": self.cross_encoder_model_name,
                "gpu_acceleration": self.enable_gpu_acceleration,
                "diversity_enabled": self.enable_diversity,
                "diversity_method": self.diversity_method.value if self.diversity_method else None,
                "max_rerank_size": self.max_rerank_size,
                "caching_enabled": self.enable_caching
            },
            "processing_stats": {
                "total_operations": self.stats.total_ranking_operations,
                "successful_operations": self.stats.successful_operations,
                "success_rate": (
                    self.stats.successful_operations / self.stats.total_ranking_operations * 100
                    if self.stats.total_ranking_operations > 0 else 0
                ),
                "avg_ranking_time": self.stats.avg_ranking_time,
                "avg_cross_encoder_time": self.stats.avg_cross_encoder_time,
                "avg_results_processed": self.stats.avg_results_processed,
                "gpu_usage_rate": self.stats.gpu_usage_rate * 100
            },
            "strategy_stats": {
                "strategy_usage": dict(self.stats.strategy_usage),
                "strategy_performance": dict(self.stats.strategy_performance),
                "available_strategies": [s.value for s in RankingStrategy]
            },
            "quality_stats": {
                "avg_ranking_improvement": self.stats.avg_ranking_improvement,
                "avg_diversity_improvement": self.stats.avg_diversity_improvement,
                "avg_coherence_score": self.stats.avg_coherence_score
            },
            "model_info": {
                "cross_encoder_available": self.cross_encoder is not None,
                "cross_encoder_device": self.cross_encoder_device,
                "available_models": list(self.CROSS_ENCODER_MODELS.keys())
            }
        }


# Global ranking engine instance
context_ranking_engine: Optional[ContextRankingEngine] = None


def get_context_ranking_engine(**kwargs) -> ContextRankingEngine:
    """Get or create global context ranking engine instance"""
    global context_ranking_engine
    
    if context_ranking_engine is None:
        context_ranking_engine = ContextRankingEngine(**kwargs)
    
    return context_ranking_engine


async def initialize_context_ranking_engine(**kwargs) -> ContextRankingEngine:
    """Initialize and return context ranking engine"""
    engine = get_context_ranking_engine(**kwargs)
    
    # Initialize engine
    await engine.initialize()
    
    # Log initialization summary
    stats = await engine.get_ranking_stats()
    logger.info("📊 Context Ranking Engine Summary:")
    logger.info(f"  Default strategy: {stats['engine_config']['default_strategy']}")
    logger.info(f"  Cross-encoder: {stats['model_info']['cross_encoder_available']}")
    logger.info(f"  GPU acceleration: {stats['engine_config']['gpu_acceleration']}")
    logger.info(f"  Device: {stats['model_info']['cross_encoder_device']}")
    logger.info(f"  Diversity enabled: {stats['engine_config']['diversity_enabled']}")
    
    return engine