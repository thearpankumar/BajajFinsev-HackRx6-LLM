"""
Hybrid Retrieval Engine for BajajFinsev Hybrid RAG System
Combines multiple retrieval strategies: semantic, keyword, exact match, temporal
GPU-accelerated with RTX 3050 optimizations and intelligent strategy fusion
"""

import asyncio
import logging
import time
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from collections import defaultdict
import math

# Search and ranking libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Core integrations
from src.core.query_enhancement_engine import EnhancedQuery, QueryIntent, QueryComplexity
from src.services.vector_storage import get_vector_storage, SimilarityResult, VectorDocument
from src.services.embedding_service import get_embedding_service
from src.core.gpu_service import get_gpu_service
from src.services.redis_cache import get_redis_cache

logger = logging.getLogger(__name__)


class RetrievalStrategy(Enum):
    """Available retrieval strategies"""
    SEMANTIC_SEARCH = "semantic_search"
    KEYWORD_SEARCH = "keyword_search"
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    TEMPORAL_SEARCH = "temporal_search"
    COMPARATIVE_SEARCH = "comparative_search"
    HYBRID_SEARCH = "hybrid_search"
    MULTI_STAGE_RETRIEVAL = "multi_stage_retrieval"


class FusionMethod(Enum):
    """Methods for combining retrieval results"""
    RRF = "reciprocal_rank_fusion"      # Reciprocal Rank Fusion
    WEIGHTED_SUM = "weighted_sum"        # Weighted score combination
    BORDA_COUNT = "borda_count"          # Borda count ranking
    CONDORCET = "condorcet"              # Condorcet winner
    BAYESIAN_FUSION = "bayesian_fusion"  # Bayesian score fusion


@dataclass
class RetrievalResult:
    """Individual retrieval result with metadata"""
    document: VectorDocument
    score: float
    retrieval_strategy: RetrievalStrategy
    rank: int
    
    # Additional metadata
    match_type: str = ""
    confidence: float = 0.0
    explanation: str = ""
    
    # GPU processing info
    gpu_processed: bool = False
    processing_time: float = 0.0


@dataclass
class HybridRetrievalResult:
    """Complete hybrid retrieval result"""
    query: str
    enhanced_query: EnhancedQuery
    
    # Individual strategy results
    strategy_results: Dict[RetrievalStrategy, List[RetrievalResult]] = field(default_factory=dict)
    
    # Fused final results
    final_results: List[RetrievalResult] = field(default_factory=list)
    fusion_method: FusionMethod = FusionMethod.RRF
    
    # Performance metrics
    total_candidates: int = 0
    total_processing_time: float = 0.0
    strategy_timings: Dict[str, float] = field(default_factory=dict)
    gpu_acceleration_used: bool = False
    
    # Quality metrics
    result_diversity: float = 0.0
    semantic_coherence: float = 0.0
    coverage_score: float = 0.0


@dataclass
class RetrievalStats:
    """Retrieval engine statistics"""
    total_queries: int = 0
    successful_queries: int = 0
    
    # Strategy usage
    strategy_usage: Dict[str, int] = field(default_factory=dict)
    strategy_performance: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    avg_retrieval_time: float = 0.0
    avg_results_per_query: float = 0.0
    gpu_usage_rate: float = 0.0
    
    # Quality metrics
    avg_result_diversity: float = 0.0
    avg_semantic_coherence: float = 0.0
    avg_coverage_score: float = 0.0


class HybridRetrievalEngine:
    """
    Advanced hybrid retrieval engine with multiple strategies
    GPU-accelerated semantic search with intelligent strategy fusion
    """
    
    # Strategy configurations
    STRATEGY_CONFIGS = {
        RetrievalStrategy.SEMANTIC_SEARCH: {
            "weight": 0.4,
            "min_similarity": 0.3,
            "top_k": 20,
            "gpu_accelerated": True,
            "description": "Vector similarity search using embeddings"
        },
        RetrievalStrategy.KEYWORD_SEARCH: {
            "weight": 0.3,
            "min_similarity": 0.2,
            "top_k": 15,
            "gpu_accelerated": False,
            "description": "TF-IDF based keyword matching"
        },
        RetrievalStrategy.EXACT_MATCH: {
            "weight": 0.8,
            "min_similarity": 0.9,
            "top_k": 5,
            "gpu_accelerated": False,
            "description": "Exact phrase and entity matching"
        },
        RetrievalStrategy.FUZZY_MATCH: {
            "weight": 0.2,
            "min_similarity": 0.4,
            "top_k": 10,
            "gpu_accelerated": False,
            "description": "Approximate string matching"
        },
        RetrievalStrategy.TEMPORAL_SEARCH: {
            "weight": 0.6,
            "min_similarity": 0.5,
            "top_k": 10,
            "gpu_accelerated": False,
            "description": "Time-aware document retrieval"
        }
    }
    
    # Fusion method configurations
    FUSION_CONFIGS = {
        FusionMethod.RRF: {
            "k": 60,  # RRF parameter
            "description": "Reciprocal rank fusion with parameter k"
        },
        FusionMethod.WEIGHTED_SUM: {
            "normalization": "min_max",
            "description": "Weighted sum with score normalization"
        },
        FusionMethod.BORDA_COUNT: {
            "tie_breaking": "score",
            "description": "Borda count with score-based tie breaking"
        }
    }
    
    def __init__(self,
                 default_fusion_method: FusionMethod = FusionMethod.RRF,
                 enable_gpu_acceleration: bool = True,
                 max_results: int = 50,
                 enable_caching: bool = True,
                 cache_ttl: int = 1800):
        """
        Initialize hybrid retrieval engine
        
        Args:
            default_fusion_method: Default method for combining results
            enable_gpu_acceleration: Enable GPU-accelerated operations
            max_results: Maximum results to return
            enable_caching: Enable result caching
            cache_ttl: Cache TTL in seconds
        """
        self.default_fusion_method = default_fusion_method
        self.enable_gpu_acceleration = enable_gpu_acceleration
        self.max_results = max_results
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Initialize services
        self.vector_storage = get_vector_storage()
        self.embedding_service = get_embedding_service()
        self.gpu_service = get_gpu_service() if enable_gpu_acceleration else None
        self.redis_cache = get_redis_cache() if enable_caching else None
        
        # TF-IDF vectorizer for keyword search
        self.tfidf_vectorizer = None
        self.document_tfidf_matrix = None
        self.document_texts = []
        self.document_ids = []
        
        # Strategy weights (can be adjusted dynamically)
        self.strategy_weights = {
            strategy: config["weight"] 
            for strategy, config in self.STRATEGY_CONFIGS.items()
        }
        
        # Statistics
        self.stats = RetrievalStats(
            strategy_usage=defaultdict(int),
            strategy_performance=defaultdict(float)
        )
        
        # Performance optimization
        self.batch_size = 32  # RTX 3050 optimized
        self.similarity_cache = {}  # In-memory similarity cache
        
        logger.info("HybridRetrievalEngine initialized")
        logger.info(f"Fusion method: {default_fusion_method.value}")
        logger.info(f"GPU acceleration: {enable_gpu_acceleration}")
        logger.info(f"Available strategies: {len(self.STRATEGY_CONFIGS)}")
    
    async def initialize(self) -> bool:
        """Initialize retrieval engine and build indices"""
        logger.info("🔄 Initializing hybrid retrieval engine...")
        
        try:
            # Initialize vector storage and embedding service
            await self.vector_storage.initialize()
            await self.embedding_service.initialize()
            
            # Build TF-IDF index for keyword search
            await self._build_tfidf_index()
            
            logger.info("✅ Hybrid retrieval engine initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize retrieval engine: {str(e)}")
            return False
    
    async def retrieve(self, 
                      enhanced_query: EnhancedQuery,
                      strategies: Optional[List[RetrievalStrategy]] = None,
                      fusion_method: Optional[FusionMethod] = None,
                      **kwargs) -> HybridRetrievalResult:
        """
        Perform hybrid retrieval using multiple strategies
        
        Args:
            enhanced_query: Enhanced query with analysis and embeddings
            strategies: List of strategies to use (None = auto-select)
            fusion_method: Method for combining results (None = default)
            **kwargs: Additional parameters
            
        Returns:
            HybridRetrievalResult with combined results
        """
        start_time = time.time()
        self.stats.total_queries += 1
        
        query = enhanced_query.analysis.original_query
        logger.debug(f"🔍 Hybrid retrieval for: {query}")
        
        # Check cache first
        cache_key = None
        if self.enable_caching and self.redis_cache:
            cache_key = self._generate_cache_key(enhanced_query, strategies, fusion_method)
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                logger.debug("✅ Using cached retrieval result")
                return cached_result
        
        try:
            # Auto-select strategies if not provided
            if strategies is None:
                strategies = self._select_strategies(enhanced_query)
            
            # Use default fusion method if not specified
            if fusion_method is None:
                fusion_method = self.default_fusion_method
            
            # Initialize result
            result = HybridRetrievalResult(
                query=query,
                enhanced_query=enhanced_query,
                fusion_method=fusion_method
            )
            
            # Execute retrieval strategies
            strategy_tasks = []
            for strategy in strategies:
                task = self._execute_strategy(strategy, enhanced_query, **kwargs)
                strategy_tasks.append((strategy, task))
            
            # Run strategies concurrently
            strategy_results = {}
            for strategy, task in strategy_tasks:
                strategy_start = time.time()
                
                try:
                    strategy_result = await task
                    strategy_results[strategy] = strategy_result
                    
                    strategy_time = time.time() - strategy_start
                    result.strategy_timings[strategy.value] = strategy_time
                    
                    self.stats.strategy_usage[strategy.value] += 1
                    self._update_strategy_performance(strategy, strategy_time)
                    
                    logger.debug(f"✅ {strategy.value}: {len(strategy_result)} results in {strategy_time:.3f}s")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Strategy {strategy.value} failed: {str(e)}")
                    strategy_results[strategy] = []
            
            result.strategy_results = strategy_results
            
            # Fuse results
            result.final_results = await self._fuse_results(
                strategy_results, fusion_method, enhanced_query
            )
            
            # Calculate quality metrics
            await self._calculate_quality_metrics(result)
            
            # Finalize result
            result.total_candidates = sum(len(results) for results in strategy_results.values())
            result.total_processing_time = time.time() - start_time
            result.gpu_acceleration_used = any(
                self.STRATEGY_CONFIGS.get(strategy, {}).get("gpu_accelerated", False)
                for strategy in strategies
            )
            
            # Update statistics
            self.stats.successful_queries += 1
            self._update_performance_stats(result)
            
            # Cache result
            if self.enable_caching and cache_key:
                await self._cache_result(cache_key, result)
            
            logger.info(f"✅ Hybrid retrieval completed: {len(result.final_results)} results in {result.total_processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Hybrid retrieval failed: {str(e)}")
            
            # Return empty result
            return HybridRetrievalResult(
                query=query,
                enhanced_query=enhanced_query,
                total_processing_time=time.time() - start_time
            )
    
    def _select_strategies(self, enhanced_query: EnhancedQuery) -> List[RetrievalStrategy]:
        """Auto-select retrieval strategies based on query analysis"""
        analysis = enhanced_query.analysis
        selected_strategies = []
        
        # Always include semantic search
        selected_strategies.append(RetrievalStrategy.SEMANTIC_SEARCH)
        
        # Intent-based strategy selection
        if analysis.intent == QueryIntent.FACTUAL:
            selected_strategies.extend([
                RetrievalStrategy.KEYWORD_SEARCH,
                RetrievalStrategy.EXACT_MATCH
            ])
        elif analysis.intent == QueryIntent.NUMERICAL:
            selected_strategies.extend([
                RetrievalStrategy.EXACT_MATCH,
                RetrievalStrategy.KEYWORD_SEARCH
            ])
        elif analysis.intent == QueryIntent.TEMPORAL:
            selected_strategies.extend([
                RetrievalStrategy.TEMPORAL_SEARCH,
                RetrievalStrategy.KEYWORD_SEARCH
            ])
        elif analysis.intent == QueryIntent.COMPARATIVE:
            selected_strategies.extend([
                RetrievalStrategy.COMPARATIVE_SEARCH,
                RetrievalStrategy.KEYWORD_SEARCH
            ])
        else:
            # Default strategies
            selected_strategies.extend([
                RetrievalStrategy.KEYWORD_SEARCH,
                RetrievalStrategy.FUZZY_MATCH
            ])
        
        # Complexity-based additions
        if analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            if RetrievalStrategy.MULTI_STAGE_RETRIEVAL not in selected_strategies:
                selected_strategies.append(RetrievalStrategy.MULTI_STAGE_RETRIEVAL)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_strategies = []
        for strategy in selected_strategies:
            if strategy not in seen:
                seen.add(strategy)
                unique_strategies.append(strategy)
        
        logger.debug(f"Selected strategies: {[s.value for s in unique_strategies]}")
        return unique_strategies
    
    async def _execute_strategy(self, 
                               strategy: RetrievalStrategy,
                               enhanced_query: EnhancedQuery,
                               **kwargs) -> List[RetrievalResult]:
        """Execute individual retrieval strategy"""
        
        if strategy == RetrievalStrategy.SEMANTIC_SEARCH:
            return await self._semantic_search(enhanced_query, **kwargs)
        elif strategy == RetrievalStrategy.KEYWORD_SEARCH:
            return await self._keyword_search(enhanced_query, **kwargs)
        elif strategy == RetrievalStrategy.EXACT_MATCH:
            return await self._exact_match_search(enhanced_query, **kwargs)
        elif strategy == RetrievalStrategy.FUZZY_MATCH:
            return await self._fuzzy_match_search(enhanced_query, **kwargs)
        elif strategy == RetrievalStrategy.TEMPORAL_SEARCH:
            return await self._temporal_search(enhanced_query, **kwargs)
        elif strategy == RetrievalStrategy.COMPARATIVE_SEARCH:
            return await self._comparative_search(enhanced_query, **kwargs)
        elif strategy == RetrievalStrategy.MULTI_STAGE_RETRIEVAL:
            return await self._multi_stage_retrieval(enhanced_query, **kwargs)
        else:
            logger.warning(f"⚠️ Unknown strategy: {strategy}")
            return []
    
    async def _semantic_search(self, enhanced_query: EnhancedQuery, **kwargs) -> List[RetrievalResult]:
        """Semantic similarity search using embeddings"""
        config = self.STRATEGY_CONFIGS[RetrievalStrategy.SEMANTIC_SEARCH]
        
        # Use expanded query embedding if available, otherwise original
        query_embedding = None
        if enhanced_query.expanded_embedding:
            query_embedding = np.array(enhanced_query.expanded_embedding)
        elif enhanced_query.original_embedding:
            query_embedding = np.array(enhanced_query.original_embedding)
        else:
            # Generate embedding on the fly
            embedding_result = await self.embedding_service.generate_embeddings([
                enhanced_query.analysis.expanded_query or enhanced_query.analysis.original_query
            ])
            if embedding_result.success:
                query_embedding = embedding_result.embeddings[0]
        
        if query_embedding is None:
            logger.warning("⚠️ No embedding available for semantic search")
            return []
        
        # Perform vector search
        search_result = await self.vector_storage.search_similar(
            query_embedding=query_embedding,
            top_k=config["top_k"],
            min_similarity=config["min_similarity"]
        )
        
        # Convert to RetrievalResult format
        results = []
        for i, similarity_result in enumerate(search_result.results):
            result = RetrievalResult(
                document=similarity_result.document,
                score=similarity_result.score,
                retrieval_strategy=RetrievalStrategy.SEMANTIC_SEARCH,
                rank=i + 1,
                match_type="semantic_similarity",
                confidence=similarity_result.score,
                explanation=f"Semantic similarity score: {similarity_result.score:.3f}",
                gpu_processed=search_result.backend_used != "cpu"
            )
            results.append(result)
        
        return results
    
    async def _keyword_search(self, enhanced_query: EnhancedQuery, **kwargs) -> List[RetrievalResult]:
        """TF-IDF based keyword search"""
        if not SKLEARN_AVAILABLE or self.tfidf_vectorizer is None:
            logger.warning("⚠️ TF-IDF search not available")
            return []
        
        config = self.STRATEGY_CONFIGS[RetrievalStrategy.KEYWORD_SEARCH]
        query_text = enhanced_query.analysis.expanded_query or enhanced_query.analysis.original_query
        
        try:
            # Transform query to TF-IDF vector
            query_vector = self.tfidf_vectorizer.transform([query_text])
            
            # Calculate cosine similarity with all documents
            similarities = cosine_similarity(query_vector, self.document_tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:config["top_k"]]
            
            results = []
            for i, doc_idx in enumerate(top_indices):
                similarity_score = similarities[doc_idx]
                
                if similarity_score < config["min_similarity"]:
                    break
                
                # Get document by ID
                doc_id = self.document_ids[doc_idx]
                document = await self.vector_storage.get_document(doc_id)
                
                if document:
                    result = RetrievalResult(
                        document=document,
                        score=float(similarity_score),
                        retrieval_strategy=RetrievalStrategy.KEYWORD_SEARCH,
                        rank=i + 1,
                        match_type="keyword_similarity",
                        confidence=float(similarity_score),
                        explanation=f"TF-IDF similarity score: {similarity_score:.3f}",
                        gpu_processed=False
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ Keyword search failed: {str(e)}")
            return []
    
    async def _exact_match_search(self, enhanced_query: EnhancedQuery, **kwargs) -> List[RetrievalResult]:
        """Exact phrase and entity matching"""
        config = self.STRATEGY_CONFIGS[RetrievalStrategy.EXACT_MATCH]
        
        query_text = enhanced_query.analysis.original_query.lower()
        key_phrases = enhanced_query.analysis.key_concepts
        entities = enhanced_query.analysis.entities
        
        # Build search patterns
        search_patterns = []
        
        # Add key phrases
        for phrase in key_phrases:
            if len(phrase) > 2:  # Skip very short phrases
                search_patterns.append(phrase.lower())
        
        # Add entities
        for entity in entities:
            if len(entity) > 2:
                search_patterns.append(entity.lower())
        
        # Add quoted phrases from query
        quoted_phrases = re.findall(r'"([^"]*)"', query_text)
        search_patterns.extend(quoted_phrases)
        
        if not search_patterns:
            return []
        
        # Search for exact matches in documents
        results = []
        documents = await self.vector_storage.list_documents(limit=1000)  # Limit search scope
        
        for doc in documents:
            doc_text = doc.content.lower()
            
            # Count exact matches
            match_count = 0
            matched_phrases = []
            
            for pattern in search_patterns:
                if pattern in doc_text:
                    match_count += 1
                    matched_phrases.append(pattern)
            
            if match_count > 0:
                # Calculate match score
                match_score = match_count / len(search_patterns)
                
                if match_score >= config["min_similarity"]:
                    result = RetrievalResult(
                        document=doc,
                        score=match_score,
                        retrieval_strategy=RetrievalStrategy.EXACT_MATCH,
                        rank=len(results) + 1,
                        match_type="exact_phrase",
                        confidence=match_score,
                        explanation=f"Exact matches: {', '.join(matched_phrases[:3])}",
                        gpu_processed=False
                    )
                    results.append(result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:config["top_k"]]
    
    async def _fuzzy_match_search(self, enhanced_query: EnhancedQuery, **kwargs) -> List[RetrievalResult]:
        """Fuzzy string matching for approximate matches"""
        config = self.STRATEGY_CONFIGS[RetrievalStrategy.FUZZY_MATCH]
        
        # Simple fuzzy matching using edit distance concepts
        query_text = enhanced_query.analysis.original_query.lower()
        query_words = set(query_text.split())
        
        results = []
        documents = await self.vector_storage.list_documents(limit=500)  # Limit for performance
        
        for doc in documents:
            doc_text = doc.content.lower()
            doc_words = set(doc_text.split())
            
            # Calculate word overlap ratio
            common_words = query_words.intersection(doc_words)
            if not query_words:
                continue
                
            overlap_ratio = len(common_words) / len(query_words)
            
            # Calculate partial matches (simplified fuzzy)
            partial_matches = 0
            for query_word in query_words:
                if len(query_word) > 3:  # Only for longer words
                    for doc_word in doc_words:
                        if query_word[:3] == doc_word[:3] or query_word[-3:] == doc_word[-3:]:
                            partial_matches += 1
                            break
            
            partial_ratio = partial_matches / len(query_words) if query_words else 0
            
            # Combine exact and partial matches
            fuzzy_score = 0.7 * overlap_ratio + 0.3 * partial_ratio
            
            if fuzzy_score >= config["min_similarity"]:
                result = RetrievalResult(
                    document=doc,
                    score=fuzzy_score,
                    retrieval_strategy=RetrievalStrategy.FUZZY_MATCH,
                    rank=len(results) + 1,
                    match_type="fuzzy_match",
                    confidence=fuzzy_score,
                    explanation=f"Fuzzy match score: {fuzzy_score:.3f} (exact: {overlap_ratio:.2f}, partial: {partial_ratio:.2f})",
                    gpu_processed=False
                )
                results.append(result)
        
        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:config["top_k"]]
    
    async def _temporal_search(self, enhanced_query: EnhancedQuery, **kwargs) -> List[RetrievalResult]:
        """Time-aware document search"""
        config = self.STRATEGY_CONFIGS[RetrievalStrategy.TEMPORAL_SEARCH]
        
        temporal_indicators = enhanced_query.analysis.temporal_indicators
        if not temporal_indicators:
            return []
        
        # Extract time-related patterns
        time_patterns = []
        for indicator in temporal_indicators:
            # Years
            if re.match(r'\d{4}', indicator):
                time_patterns.append(indicator)
            # Month names
            elif re.match(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', indicator.lower()):
                time_patterns.append(indicator)
            # Time keywords
            elif indicator.lower() in ['recent', 'current', 'latest', 'new', 'old', 'previous']:
                time_patterns.append(indicator)
        
        if not time_patterns:
            return []
        
        # Search for temporal matches
        results = []
        documents = await self.vector_storage.list_documents(limit=1000)
        
        for doc in documents:
            doc_text = doc.content.lower()
            
            # Count temporal matches
            temporal_matches = 0
            matched_patterns = []
            
            for pattern in time_patterns:
                if pattern.lower() in doc_text:
                    temporal_matches += 1
                    matched_patterns.append(pattern)
            
            if temporal_matches > 0:
                # Calculate temporal relevance score
                temporal_score = temporal_matches / len(time_patterns)
                
                # Boost score for recent documents (if metadata available)
                if hasattr(doc, 'created_at') and doc.created_at:
                    # Boost recent documents
                    recency_boost = min(0.2, (time.time() - doc.created_at) / (30 * 24 * 3600))  # 30 days
                    temporal_score += recency_boost
                
                temporal_score = min(1.0, temporal_score)
                
                if temporal_score >= config["min_similarity"]:
                    result = RetrievalResult(
                        document=doc,
                        score=temporal_score,
                        retrieval_strategy=RetrievalStrategy.TEMPORAL_SEARCH,
                        rank=len(results) + 1,
                        match_type="temporal_relevance",
                        confidence=temporal_score,
                        explanation=f"Temporal matches: {', '.join(matched_patterns[:3])}",
                        gpu_processed=False
                    )
                    results.append(result)
        
        # Sort by temporal relevance
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:config["top_k"]]
    
    async def _comparative_search(self, enhanced_query: EnhancedQuery, **kwargs) -> List[RetrievalResult]:
        """Search for comparative content"""
        # Look for comparison keywords and patterns
        comparison_keywords = ['compare', 'versus', 'vs', 'difference', 'better', 'worse', 'advantage', 'disadvantage']
        
        query_text = enhanced_query.analysis.original_query.lower()
        has_comparison = any(keyword in query_text for keyword in comparison_keywords)
        
        if not has_comparison:
            return []
        
        # Use keyword search with comparison-focused scoring
        keyword_results = await self._keyword_search(enhanced_query, **kwargs)
        
        # Boost documents that contain comparative language
        comparative_results = []
        for result in keyword_results:
            doc_text = result.document.content.lower()
            
            # Count comparative terms
            comp_count = sum(1 for keyword in comparison_keywords if keyword in doc_text)
            
            # Boost score based on comparative content
            boost_factor = 1.0 + (comp_count * 0.1)
            boosted_score = min(1.0, result.score * boost_factor)
            
            comparative_result = RetrievalResult(
                document=result.document,
                score=boosted_score,
                retrieval_strategy=RetrievalStrategy.COMPARATIVE_SEARCH,
                rank=result.rank,
                match_type="comparative_content",
                confidence=boosted_score,
                explanation=f"Comparative content score: {boosted_score:.3f} (boost: {boost_factor:.2f})",
                gpu_processed=result.gpu_processed
            )
            comparative_results.append(comparative_result)
        
        # Re-sort by boosted scores
        comparative_results.sort(key=lambda x: x.score, reverse=True)
        return comparative_results[:10]
    
    async def _multi_stage_retrieval(self, enhanced_query: EnhancedQuery, **kwargs) -> List[RetrievalResult]:
        """Multi-stage retrieval for complex queries"""
        # Stage 1: Broad semantic search
        stage1_results = await self._semantic_search(enhanced_query, **{**kwargs, "top_k": 100})
        
        if not stage1_results:
            return []
        
        # Stage 2: Keyword refinement
        stage2_candidates = [result.document for result in stage1_results[:50]]
        
        # Create temporary enhanced query for stage 2
        stage2_enhanced = enhanced_query
        stage2_enhanced.analysis.expanded_query = enhanced_query.analysis.original_query
        
        stage2_results = await self._keyword_search(stage2_enhanced, **kwargs)
        
        # Stage 3: Exact match verification
        stage3_results = await self._exact_match_search(enhanced_query, **kwargs)
        
        # Combine and deduplicate results
        all_results = {}  # Use document ID as key to avoid duplicates
        
        # Add results with stage-based scoring
        for result in stage1_results[:20]:  # Top 20 from semantic
            doc_id = result.document.id
            if doc_id not in all_results or all_results[doc_id].score < result.score:
                result.score *= 0.8  # Slightly reduce semantic scores
                result.explanation = f"Multi-stage: {result.explanation}"
                all_results[doc_id] = result
        
        for result in stage2_results[:15]:  # Top 15 from keyword
            doc_id = result.document.id
            if doc_id not in all_results or all_results[doc_id].score < result.score:
                result.score *= 0.9  # Slightly reduce keyword scores
                result.explanation = f"Multi-stage: {result.explanation}"
                all_results[doc_id] = result
        
        for result in stage3_results[:10]:  # Top 10 from exact match
            doc_id = result.document.id
            if doc_id not in all_results or all_results[doc_id].score < result.score:
                result.score *= 1.1  # Boost exact match scores
                result.explanation = f"Multi-stage: {result.explanation}"
                all_results[doc_id] = result
        
        # Convert back to list and sort
        final_results = list(all_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update retrieval strategy
        for result in final_results:
            result.retrieval_strategy = RetrievalStrategy.MULTI_STAGE_RETRIEVAL
            result.match_type = "multi_stage"
        
        return final_results[:20]
    
    async def _fuse_results(self, 
                           strategy_results: Dict[RetrievalStrategy, List[RetrievalResult]],
                           fusion_method: FusionMethod,
                           enhanced_query: EnhancedQuery) -> List[RetrievalResult]:
        """Fuse results from multiple strategies"""
        
        if fusion_method == FusionMethod.RRF:
            return await self._reciprocal_rank_fusion(strategy_results)
        elif fusion_method == FusionMethod.WEIGHTED_SUM:
            return await self._weighted_sum_fusion(strategy_results, enhanced_query)
        elif fusion_method == FusionMethod.BORDA_COUNT:
            return await self._borda_count_fusion(strategy_results)
        else:
            # Default to RRF
            return await self._reciprocal_rank_fusion(strategy_results)
    
    async def _reciprocal_rank_fusion(self, 
                                     strategy_results: Dict[RetrievalStrategy, List[RetrievalResult]]) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion (RRF) for combining ranked lists"""
        k = self.FUSION_CONFIGS[FusionMethod.RRF]["k"]
        
        # Collect all unique documents
        all_documents = {}
        
        for strategy, results in strategy_results.items():
            for rank, result in enumerate(results, 1):
                doc_id = result.document.id
                
                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        'document': result.document,
                        'rrf_score': 0.0,
                        'strategy_scores': {},
                        'strategies': set(),
                        'best_explanation': result.explanation,
                        'gpu_processed': result.gpu_processed
                    }
                
                # Add RRF score: 1/(k + rank)
                rrf_contribution = 1.0 / (k + rank)
                all_documents[doc_id]['rrf_score'] += rrf_contribution
                all_documents[doc_id]['strategy_scores'][strategy.value] = result.score
                all_documents[doc_id]['strategies'].add(strategy.value)
                
                # Keep best explanation
                if result.score > 0.8:
                    all_documents[doc_id]['best_explanation'] = result.explanation
        
        # Convert to RetrievalResult objects
        fused_results = []
        for rank, (doc_id, doc_info) in enumerate(
            sorted(all_documents.items(), key=lambda x: x[1]['rrf_score'], reverse=True), 1
        ):
            
            # Create explanation
            strategies_used = list(doc_info['strategies'])
            explanation = f"RRF fusion (strategies: {', '.join(strategies_used)}, score: {doc_info['rrf_score']:.3f})"
            
            result = RetrievalResult(
                document=doc_info['document'],
                score=doc_info['rrf_score'],
                retrieval_strategy=RetrievalStrategy.HYBRID_SEARCH,
                rank=rank,
                match_type="rrf_fusion",
                confidence=min(1.0, doc_info['rrf_score'] * 2),  # Normalize confidence
                explanation=explanation,
                gpu_processed=doc_info['gpu_processed']
            )
            fused_results.append(result)
        
        return fused_results[:self.max_results]
    
    async def _weighted_sum_fusion(self,
                                  strategy_results: Dict[RetrievalStrategy, List[RetrievalResult]],
                                  enhanced_query: EnhancedQuery) -> List[RetrievalResult]:
        """Weighted sum fusion with adaptive weights"""
        all_documents = {}
        
        # Adjust weights based on query analysis
        adjusted_weights = self._adjust_strategy_weights(enhanced_query)
        
        for strategy, results in strategy_results.items():
            weight = adjusted_weights.get(strategy, self.strategy_weights.get(strategy, 0.3))
            
            for result in results:
                doc_id = result.document.id
                
                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        'document': result.document,
                        'weighted_score': 0.0,
                        'strategy_contributions': {},
                        'strategies': set(),
                        'best_explanation': result.explanation,
                        'gpu_processed': result.gpu_processed
                    }
                
                # Add weighted contribution
                weighted_contribution = result.score * weight
                all_documents[doc_id]['weighted_score'] += weighted_contribution
                all_documents[doc_id]['strategy_contributions'][strategy.value] = weighted_contribution
                all_documents[doc_id]['strategies'].add(strategy.value)
        
        # Normalize scores
        max_score = max((doc['weighted_score'] for doc in all_documents.values()), default=1.0)
        
        fused_results = []
        for rank, (doc_id, doc_info) in enumerate(
            sorted(all_documents.items(), key=lambda x: x[1]['weighted_score'], reverse=True), 1
        ):
            
            normalized_score = doc_info['weighted_score'] / max_score
            
            # Create explanation
            strategies_used = list(doc_info['strategies'])
            explanation = f"Weighted fusion (strategies: {', '.join(strategies_used)}, score: {normalized_score:.3f})"
            
            result = RetrievalResult(
                document=doc_info['document'],
                score=normalized_score,
                retrieval_strategy=RetrievalStrategy.HYBRID_SEARCH,
                rank=rank,
                match_type="weighted_fusion",
                confidence=normalized_score,
                explanation=explanation,
                gpu_processed=doc_info['gpu_processed']
            )
            fused_results.append(result)
        
        return fused_results[:self.max_results]
    
    async def _borda_count_fusion(self,
                                 strategy_results: Dict[RetrievalStrategy, List[RetrievalResult]]) -> List[RetrievalResult]:
        """Borda count fusion method"""
        all_documents = {}
        
        for strategy, results in strategy_results.items():
            # Borda points: n-rank for n total results
            n = len(results)
            
            for rank, result in enumerate(results, 1):
                doc_id = result.document.id
                
                if doc_id not in all_documents:
                    all_documents[doc_id] = {
                        'document': result.document,
                        'borda_points': 0,
                        'strategies': set(),
                        'best_explanation': result.explanation,
                        'gpu_processed': result.gpu_processed
                    }
                
                # Add Borda points
                borda_points = n - rank + 1
                all_documents[doc_id]['borda_points'] += borda_points
                all_documents[doc_id]['strategies'].add(strategy.value)
        
        # Convert to results
        fused_results = []
        for rank, (doc_id, doc_info) in enumerate(
            sorted(all_documents.items(), key=lambda x: x[1]['borda_points'], reverse=True), 1
        ):
            
            # Normalize Borda points to 0-1 score
            max_possible_points = sum(len(results) for results in strategy_results.values())
            normalized_score = doc_info['borda_points'] / max_possible_points
            
            strategies_used = list(doc_info['strategies'])
            explanation = f"Borda count (strategies: {', '.join(strategies_used)}, points: {doc_info['borda_points']})"
            
            result = RetrievalResult(
                document=doc_info['document'],
                score=normalized_score,
                retrieval_strategy=RetrievalStrategy.HYBRID_SEARCH,
                rank=rank,
                match_type="borda_fusion",
                confidence=normalized_score,
                explanation=explanation,
                gpu_processed=doc_info['gpu_processed']
            )
            fused_results.append(result)
        
        return fused_results[:self.max_results]
    
    def _adjust_strategy_weights(self, enhanced_query: EnhancedQuery) -> Dict[RetrievalStrategy, float]:
        """Adjust strategy weights based on query analysis"""
        adjusted_weights = self.strategy_weights.copy()
        analysis = enhanced_query.analysis
        
        # Adjust based on intent
        if analysis.intent == QueryIntent.NUMERICAL:
            adjusted_weights[RetrievalStrategy.EXACT_MATCH] *= 1.5
            adjusted_weights[RetrievalStrategy.SEMANTIC_SEARCH] *= 0.8
        elif analysis.intent == QueryIntent.TEMPORAL:
            adjusted_weights[RetrievalStrategy.TEMPORAL_SEARCH] *= 1.3
        elif analysis.intent == QueryIntent.COMPARATIVE:
            adjusted_weights[RetrievalStrategy.COMPARATIVE_SEARCH] *= 1.2
        
        # Adjust based on complexity
        if analysis.complexity == QueryComplexity.SIMPLE:
            adjusted_weights[RetrievalStrategy.EXACT_MATCH] *= 1.2
        elif analysis.complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            adjusted_weights[RetrievalStrategy.SEMANTIC_SEARCH] *= 1.3
            adjusted_weights[RetrievalStrategy.KEYWORD_SEARCH] *= 1.1
        
        return adjusted_weights
    
    async def _calculate_quality_metrics(self, result: HybridRetrievalResult):
        """Calculate quality metrics for retrieval result"""
        if not result.final_results:
            return
        
        # Result diversity - measure content diversity
        if len(result.final_results) > 1:
            # Simple diversity based on document content overlap
            contents = [doc.document.content[:500] for doc in result.final_results[:10]]
            diversity_scores = []
            
            for i, content1 in enumerate(contents):
                for j, content2 in enumerate(contents[i+1:], i+1):
                    words1 = set(content1.lower().split())
                    words2 = set(content2.lower().split())
                    
                    if words1 and words2:
                        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                        diversity_scores.append(1.0 - overlap)
            
            result.result_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0
        
        # Semantic coherence - consistency of results with query
        if result.enhanced_query.original_embedding:
            query_embedding = np.array(result.enhanced_query.original_embedding)
            coherence_scores = []
            
            for res in result.final_results[:5]:  # Check top 5 results
                # Would need document embeddings for proper coherence calculation
                # Using score as proxy for now
                coherence_scores.append(res.score)
            
            result.semantic_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        # Coverage score - how well results cover different aspects
        strategy_coverage = len(set(res.retrieval_strategy for res in result.final_results))
        max_strategies = len(self.STRATEGY_CONFIGS)
        result.coverage_score = strategy_coverage / max_strategies
    
    async def _build_tfidf_index(self):
        """Build TF-IDF index for keyword search"""
        if not SKLEARN_AVAILABLE:
            logger.warning("⚠️ Scikit-learn not available, keyword search disabled")
            return
        
        try:
            # Get all documents
            documents = await self.vector_storage.list_documents(limit=10000)
            
            if not documents:
                logger.info("📋 No documents found for TF-IDF indexing")
                return
            
            # Extract text and IDs
            self.document_texts = [doc.content for doc in documents]
            self.document_ids = [doc.id for doc in documents]
            
            # Build TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),  # Unigrams and bigrams
                lowercase=True,
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.8  # Ignore terms that appear in more than 80% of documents
            )
            
            # Fit and transform documents
            self.document_tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.document_texts)
            
            logger.info(f"✅ TF-IDF index built: {len(documents)} documents, {self.tfidf_vectorizer.vocabulary_.__len__()} features")
            
        except Exception as e:
            logger.error(f"❌ Failed to build TF-IDF index: {str(e)}")
    
    def _generate_cache_key(self, 
                           enhanced_query: EnhancedQuery,
                           strategies: Optional[List[RetrievalStrategy]],
                           fusion_method: Optional[FusionMethod]) -> str:
        """Generate cache key for retrieval result"""
        key_parts = [
            enhanced_query.analysis.original_query,
            str(sorted([s.value for s in strategies] if strategies else [])),
            fusion_method.value if fusion_method else self.default_fusion_method.value
        ]
        
        key_string = "|".join(key_parts)
        return f"hybrid_retrieval:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[HybridRetrievalResult]:
        """Get cached retrieval result"""
        if not self.redis_cache:
            return None
        
        try:
            cached_data = await self.redis_cache.get(cache_key)
            if cached_data:
                # Would need proper deserialization
                return None  # TODO: Implement proper serialization
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: HybridRetrievalResult):
        """Cache retrieval result"""
        if not self.redis_cache:
            return
        
        try:
            # Would need proper serialization
            # await self.redis_cache.setex(cache_key, self.cache_ttl, serialized_result)
            pass  # TODO: Implement proper serialization
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def _update_strategy_performance(self, strategy: RetrievalStrategy, processing_time: float):
        """Update strategy performance metrics"""
        current_avg = self.stats.strategy_performance[strategy.value]
        usage_count = self.stats.strategy_usage[strategy.value]
        
        # Update running average
        new_avg = ((current_avg * (usage_count - 1)) + processing_time) / usage_count
        self.stats.strategy_performance[strategy.value] = new_avg
    
    def _update_performance_stats(self, result: HybridRetrievalResult):
        """Update overall performance statistics"""
        # Update averages
        total_queries = self.stats.total_queries
        
        current_time_avg = self.stats.avg_retrieval_time
        new_time_avg = ((current_time_avg * (total_queries - 1)) + result.total_processing_time) / total_queries
        self.stats.avg_retrieval_time = new_time_avg
        
        current_results_avg = self.stats.avg_results_per_query
        new_results_avg = ((current_results_avg * (total_queries - 1)) + len(result.final_results)) / total_queries
        self.stats.avg_results_per_query = new_results_avg
        
        # Update quality metrics
        if result.result_diversity > 0:
            current_diversity = self.stats.avg_result_diversity
            new_diversity = ((current_diversity * (total_queries - 1)) + result.result_diversity) / total_queries
            self.stats.avg_result_diversity = new_diversity
        
        if result.semantic_coherence > 0:
            current_coherence = self.stats.avg_semantic_coherence
            new_coherence = ((current_coherence * (total_queries - 1)) + result.semantic_coherence) / total_queries
            self.stats.avg_semantic_coherence = new_coherence
        
        if result.coverage_score > 0:
            current_coverage = self.stats.avg_coverage_score
            new_coverage = ((current_coverage * (total_queries - 1)) + result.coverage_score) / total_queries
            self.stats.avg_coverage_score = new_coverage
        
        # GPU usage rate
        if result.gpu_acceleration_used:
            gpu_usage = (self.stats.gpu_usage_rate * (total_queries - 1) + 1) / total_queries
            self.stats.gpu_usage_rate = gpu_usage
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get comprehensive retrieval statistics"""
        return {
            "engine_config": {
                "fusion_method": self.default_fusion_method.value,
                "gpu_acceleration": self.enable_gpu_acceleration,
                "max_results": self.max_results,
                "caching_enabled": self.enable_caching,
                "cache_ttl": self.cache_ttl
            },
            "processing_stats": {
                "total_queries": self.stats.total_queries,
                "successful_queries": self.stats.successful_queries,
                "success_rate": (
                    self.stats.successful_queries / self.stats.total_queries * 100
                    if self.stats.total_queries > 0 else 0
                ),
                "avg_retrieval_time": self.stats.avg_retrieval_time,
                "avg_results_per_query": self.stats.avg_results_per_query,
                "gpu_usage_rate": self.stats.gpu_usage_rate * 100
            },
            "strategy_stats": {
                "strategy_usage": dict(self.stats.strategy_usage),
                "strategy_performance": dict(self.stats.strategy_performance),
                "available_strategies": [s.value for s in RetrievalStrategy]
            },
            "quality_stats": {
                "avg_result_diversity": self.stats.avg_result_diversity,
                "avg_semantic_coherence": self.stats.avg_semantic_coherence,
                "avg_coverage_score": self.stats.avg_coverage_score
            },
            "index_stats": {
                "tfidf_available": self.tfidf_vectorizer is not None,
                "indexed_documents": len(self.document_ids) if self.document_ids else 0,
                "vocabulary_size": (
                    len(self.tfidf_vectorizer.vocabulary_) 
                    if self.tfidf_vectorizer else 0
                )
            }
        }


# Global retrieval engine instance
hybrid_retrieval_engine: Optional[HybridRetrievalEngine] = None


def get_hybrid_retrieval_engine(**kwargs) -> HybridRetrievalEngine:
    """Get or create global hybrid retrieval engine instance"""
    global hybrid_retrieval_engine
    
    if hybrid_retrieval_engine is None:
        hybrid_retrieval_engine = HybridRetrievalEngine(**kwargs)
    
    return hybrid_retrieval_engine


async def initialize_hybrid_retrieval_engine(**kwargs) -> HybridRetrievalEngine:
    """Initialize and return hybrid retrieval engine"""
    engine = get_hybrid_retrieval_engine(**kwargs)
    
    # Initialize engine
    await engine.initialize()
    
    # Log initialization summary
    stats = await engine.get_retrieval_stats()
    logger.info("🔍 Hybrid Retrieval Engine Summary:")
    logger.info(f"  Fusion method: {stats['engine_config']['fusion_method']}")
    logger.info(f"  GPU acceleration: {stats['engine_config']['gpu_acceleration']}")
    logger.info(f"  Available strategies: {len(stats['strategy_stats']['available_strategies'])}")
    logger.info(f"  TF-IDF available: {stats['index_stats']['tfidf_available']}")
    logger.info(f"  Indexed documents: {stats['index_stats']['indexed_documents']}")
    
    return engine