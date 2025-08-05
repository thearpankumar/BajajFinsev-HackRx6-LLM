"""
High-Performance RAG Engine optimized for 600+ page documents
Features: Enhanced parallel processing, intelligent caching, optimized for <30 second processing
"""

import asyncio
import time
import logging
import hashlib
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

import pickle
import aiofiles

from openai import AsyncOpenAI
import google.generativeai as genai

from src.core.config import settings
try:
    from src.core.advanced_document_processor import AdvancedDocumentProcessor
    from src.core.document_processor import DocumentChunk
    USE_ADVANCED_PROCESSOR = True
except ImportError:
    from src.core.document_processor import DocumentProcessor as AdvancedDocumentProcessor
    from src.core.document_processor import DocumentChunk
    USE_ADVANCED_PROCESSOR = False
from src.core.advanced_retrieval_engine import AdvancedRetrievalEngine, SearchResult
from src.core.vector_store import VectorStore
from src.core.prompt_templates import PromptTemplates

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    task_id: str
    question: str
    priority: int = 0
    estimated_time: float = 0.0
    context_requirements: List[str] = None


@dataclass
class CacheEntry:
    key: str
    value: Any
    timestamp: float
    ttl: float
    size_bytes: int


class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"  # In-memory cache for immediate access
    L2_REDIS = "l2_redis"    # Redis cache for session persistence
    L3_DISK = "l3_disk"      # Disk cache for long-term storage


class HighPerformanceRAGEngine:
    """
    Ultra-high-performance RAG engine optimized for 600+ page documents
    Target: 95% accuracy in <30 seconds for insurance/legal/HR/compliance domains
    """

    def __init__(self):
        # Core components
        self.document_processor = AdvancedDocumentProcessor()
        self.vector_store = None
        self.retrieval_engine = None
        self.prompt_templates = PromptTemplates()
        
        # AI clients
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.google_model = genai.GenerativeModel(settings.GOOGLE_MODEL)
        
        # Performance optimization
        self.max_workers = min(16, settings.MAX_PARALLEL_QUESTIONS)
        self.embedding_semaphore = asyncio.Semaphore(8)  # Limit embedding API calls
        self.generation_semaphore = asyncio.Semaphore(6)  # Limit generation API calls
        
        # Multi-level caching system
        self.l1_cache = {}  # Memory cache
        self.l2_cache = None  # Redis cache (if available)
        self.l3_cache_path = "./cache"  # Disk cache
        self.cache_stats = {
            "l1_hits": 0, "l1_misses": 0,
            "l2_hits": 0, "l2_misses": 0,
            "l3_hits": 0, "l3_misses": 0
        }
        
        # Performance tracking
        self.performance_metrics = {
            "total_processing_time": 0,
            "document_processing_time": 0,
            "embedding_generation_time": 0,
            "retrieval_time": 0,
            "generation_time": 0,
            "questions_processed": 0,
            "cache_hit_rate": 0.0,
            "parallel_efficiency": 0.0
        }
        
        # Intelligent batching
        self.batch_optimizer = BatchOptimizer()
        
        # Pre-computed embeddings for common queries
        self.query_embeddings_cache = {}

    async def initialize(self):
        """Initialize all components with performance optimizations"""
        logger.info("Initializing High-Performance RAG Engine...")
        
        start_time = time.time()
        
        try:
            # Initialize vector store
            self.vector_store = VectorStore()
            await self.vector_store.initialize()
            
            # Initialize retrieval engine
            self.retrieval_engine = AdvancedRetrievalEngine(self.vector_store)
            
            # Initialize caching system
            await self._initialize_caching_system()
            
            # Warm up models (pre-load to reduce first-call latency)
            await self._warmup_models()
            
            initialization_time = time.time() - start_time
            logger.info(f"High-Performance RAG Engine initialized in {initialization_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to initialize High-Performance RAG Engine: {str(e)}")
            raise

    async def _initialize_caching_system(self):
        """Initialize multi-level caching system"""
        try:
            # Try to initialize Redis cache
            if REDIS_AVAILABLE:
                try:
                    import redis.asyncio as redis_async
                    self.l2_cache = redis_async.Redis(host='localhost', port=6379, db=0)
                    await self.l2_cache.ping()
                    logger.info("Redis L2 cache initialized")
                except Exception as e:
                    logger.warning(f"Redis not available, using memory-only caching: {str(e)}")
                    self.l2_cache = None
            else:
                logger.info("Redis not installed, using memory-only caching")
                self.l2_cache = None
            
            # Initialize disk cache directory
            import os
            os.makedirs(self.l3_cache_path, exist_ok=True)
            logger.info("Disk L3 cache initialized")
            
        except Exception as e:
            logger.warning(f"Cache initialization issues: {str(e)}")

    async def _warmup_models(self):
        """Warm up AI models to reduce first-call latency"""
        try:
            # Warm up OpenAI embedding model
            await self.openai_client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=["warmup text"]
            )
            
            # Warm up OpenAI generation model
            await self.openai_client.chat.completions.create(
                model=settings.OPENAI_GENERATION_MODEL,
                messages=[{"role": "user", "content": "warmup"}],
                max_tokens=10
            )
            
            logger.info("AI models warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {str(e)}")
            logger.info("Continuing without warmup - models will be initialized on first use")

    async def analyze_document_ultra_fast(
        self, document_url: str, questions: List[str]
    ) -> Dict[str, Any]:
        """
        Ultra-fast document analysis optimized for 600+ page documents
        Target: <30 seconds processing time with 95% accuracy
        """
        overall_start = time.time()
        
        try:
            # Check document cache first
            doc_cache_key = self._get_document_cache_key(document_url)
            cached_doc_data = await self._get_from_cache(doc_cache_key)
            
            if cached_doc_data:
                logger.info("Using cached document processing results")
                chunks, doc_metadata = cached_doc_data
                self.cache_stats["l1_hits"] += 1
            else:
                # Process document with parallel optimizations
                processing_start = time.time()
                chunks, doc_metadata = await self.document_processor.process_document(document_url)
                processing_time = time.time() - processing_start
                
                # Cache document processing results
                await self._set_cache(doc_cache_key, (chunks, doc_metadata), ttl=3600)
                
                self.performance_metrics["document_processing_time"] = processing_time
                logger.info(f"Document processed in {processing_time:.2f}s, {len(chunks)} chunks")
            
            # Initialize retrieval engine with chunks
            await self.retrieval_engine.initialize(chunks, document_url)
            
            # Index chunks in vector store (with caching)
            await self._index_chunks_optimized(chunks, document_url)
            
            # Process questions with intelligent parallelization
            if len(questions) > 1:
                answers = await self._process_questions_ultra_parallel(questions, chunks, document_url)
            else:
                # Single question optimization
                answers = [await self._answer_single_question_optimized(questions[0], chunks, document_url)]
            
            total_time = time.time() - overall_start
            
            # Update performance metrics
            self._update_performance_metrics(total_time, len(questions))
            
            logger.info(f"Ultra-fast analysis completed in {total_time:.2f}s for {len(questions)} questions")
            
            return {
                "answers": answers,
                "document_size": doc_metadata.get("size", 0),
                "metadata": {
                    "processing_time": total_time,
                    "num_chunks": len(chunks),
                    "questions_processed": len(questions),
                    "cache_hit_rate": self._calculate_cache_hit_rate(),
                    "performance_metrics": self.performance_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error in ultra-fast analysis: {str(e)}")
            raise

    async def _index_chunks_optimized(self, chunks: List[DocumentChunk], document_url: str):
        """Optimized chunk indexing with parallel embedding generation"""
        index_cache_key = f"index_{self._get_document_cache_key(document_url)}"
        
        if await self._get_from_cache(index_cache_key):
            logger.info("Using cached index data")
            return
        
        try:
            # Parallel embedding generation
            embedding_start = time.time()
            
            # Batch texts for optimal API usage
            texts = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    "chunk_id": chunk.chunk_id,
                    "page_num": chunk.page_num,
                    "word_count": chunk.word_count,
                    "source_url": document_url,
                    **chunk.metadata
                }
                for chunk in chunks
            ]
            
            # Process in optimal batches
            batch_size = 50  # Optimal for OpenAI API
            embedding_tasks = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                task = self._generate_embeddings_batch(batch_texts, batch_metadatas)
                embedding_tasks.append(task)
            
            # Execute all embedding tasks in parallel
            embedding_results = await asyncio.gather(*embedding_tasks)
            
            # Combine results
            all_embeddings = []
            all_metadatas = []
            for embeddings, metas in embedding_results:
                all_embeddings.extend(embeddings)
                all_metadatas.extend(metas)
            
            # Add to vector store
            if all_embeddings:
                combined_texts = [texts[i] for i in range(len(all_embeddings))]
                await self.vector_store.add_texts(combined_texts, all_metadatas)
            
            embedding_time = time.time() - embedding_start
            self.performance_metrics["embedding_generation_time"] = embedding_time
            
            # Cache indexing completion
            await self._set_cache(index_cache_key, True, ttl=3600)
            
            logger.info(f"Optimized indexing completed in {embedding_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in optimized indexing: {str(e)}")
            raise

    async def _generate_embeddings_batch(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> Tuple[List[List[float]], List[Dict[str, Any]]]:
        """Generate embeddings for a batch of texts with rate limiting"""
        async with self.embedding_semaphore:
            try:
                response = await self.openai_client.embeddings.create(
                    model=settings.OPENAI_EMBEDDING_MODEL,
                    input=texts
                )
                
                embeddings = [item.embedding for item in response.data]
                return embeddings, metadatas
                
            except Exception as e:
                logger.error(f"Error generating embeddings batch: {str(e)}")
                return [], []

    async def _process_questions_ultra_parallel(
        self, questions: List[str], chunks: List[DocumentChunk], document_url: str
    ) -> List[str]:
        """Ultra-parallel question processing with intelligent batching"""
        
        # Optimize batch configuration
        optimal_config = self.batch_optimizer.optimize_for_questions(questions)
        
        logger.info(
            f"Processing {len(questions)} questions with config: "
            f"batch_size={optimal_config['batch_size']}, "
            f"max_parallel={optimal_config['max_parallel']}"
        )
        
        # Create processing tasks with priorities
        tasks = []
        for i, question in enumerate(questions):
            task = ProcessingTask(
                task_id=f"q_{i}",
                question=question,
                priority=self._calculate_question_priority(question),
                estimated_time=self._estimate_processing_time(question)
            )
            tasks.append(task)
        
        # Sort by priority (high priority first)
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        # Process in optimized batches
        semaphore = asyncio.Semaphore(optimal_config['max_parallel'])
        
        async def process_single_task(task: ProcessingTask) -> Tuple[int, str]:
            async with semaphore:
                try:
                    start_time = time.time()
                    answer = await self._answer_single_question_optimized(
                        task.question, chunks, document_url
                    )
                    processing_time = time.time() - start_time
                    
                    logger.debug(f"Task {task.task_id} completed in {processing_time:.2f}s")
                    return (int(task.task_id.split('_')[1]), answer)
                    
                except Exception as e:
                    logger.error(f"Error processing task {task.task_id}: {str(e)}")
                    return (int(task.task_id.split('_')[1]), f"Error processing question: {str(e)}")
        
        # Execute all tasks
        parallel_start = time.time()
        task_coroutines = [process_single_task(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        parallel_time = time.time() - parallel_start
        
        # Sort results back to original order
        answers = [""] * len(questions)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue
            
            index, answer = result
            answers[index] = answer
        
        # Calculate parallel efficiency
        estimated_sequential_time = sum(task.estimated_time for task in tasks)
        efficiency = estimated_sequential_time / parallel_time if parallel_time > 0 else 1.0
        self.performance_metrics["parallel_efficiency"] = efficiency
        
        logger.info(f"Parallel processing completed in {parallel_time:.2f}s (efficiency: {efficiency:.2f}x)")
        
        return answers

    async def _answer_single_question_optimized(
        self, question: str, chunks: List[DocumentChunk], document_url: str
    ) -> str:
        """Optimized single question answering with caching"""
        
        # Check answer cache
        question_hash = hashlib.md5(f"{question}_{document_url}".encode()).hexdigest()
        cached_answer = await self._get_from_cache(f"answer_{question_hash}")
        
        if cached_answer:
            self.cache_stats["l1_hits"] += 1
            return cached_answer
        
        try:
            # Retrieval phase
            retrieval_start = time.time()
            search_results = await self.retrieval_engine.search(
                question, k=settings.TOP_K_RETRIEVAL, document_url=document_url
            )
            retrieval_time = time.time() - retrieval_start
            
            # Generation phase
            generation_start = time.time()
            answer = await self._generate_answer_optimized(question, search_results)
            generation_time = time.time() - generation_start
            
            # Update metrics
            self.performance_metrics["retrieval_time"] += retrieval_time
            self.performance_metrics["generation_time"] += generation_time
            
            # Cache the answer
            await self._set_cache(f"answer_{question_hash}", answer, ttl=1800)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question '{question}': {str(e)}")
            return "Unable to process the question due to an error."

    async def _generate_answer_optimized(
        self, question: str, search_results: List[SearchResult]
    ) -> str:
        """Optimized answer generation with context management"""
        
        async with self.generation_semaphore:
            if not search_results:
                return "I couldn't find any relevant information in the document to answer this question."
            
            # Select best results with context optimization
            relevant_chunks = [result.chunk for result in search_results if result.chunk][:settings.MAX_CHUNKS_FOR_GENERATION]
            
            # Prepare optimized context
            context_parts = []
            total_context_length = 0
            max_context_length = 4000  # Optimized for fast processing
            
            for i, chunk in enumerate(relevant_chunks):
                chunk_text = chunk.text
                
                # Add section context if available
                section_context = ""
                if "section_title" in chunk.metadata:
                    section_context = f"[{chunk.metadata['section_title']}] "
                
                chunk_with_context = f"[Context {i + 1}] {section_context}{chunk_text}"
                
                if total_context_length + len(chunk_with_context) > max_context_length:
                    chunk_with_context = chunk_with_context[:max_context_length - total_context_length] + "..."
                
                context_parts.append(chunk_with_context)
                total_context_length += len(chunk_with_context)
                
                if total_context_length >= max_context_length:
                    break
            
            context = "\n\n".join(context_parts)
            
            # Optimized prompt for fast, accurate responses
            prompt = f"""Based on the document sections below, provide a direct and precise answer to the question in 2-3 sentences maximum.

DOCUMENT SECTIONS:
{context}

QUESTION: {question}

Provide a clear, factual answer in 2-3 sentences using specific information from the document:"""
            
            try:
                # Use optimized generation parameters
                response = await self.openai_client.chat.completions.create(
                    model=settings.OPENAI_GENERATION_MODEL,
                    messages=[
                        {"role": "system", "content": self._get_optimized_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=120,  # Optimized for concise responses
                    temperature=0.1,
                    timeout=10  # Fast timeout for performance
                )
                
                answer = response.choices[0].message.content.strip()
                return answer
                
            except Exception as e:
                logger.error(f"Error generating optimized answer: {str(e)}")
                return "Unable to generate an answer due to a processing error."

    def _get_optimized_system_prompt(self) -> str:
        """Optimized system prompt for fast, accurate responses"""
        return """You are a document analyst that provides precise, factual answers in exactly 2-3 sentences.

REQUIREMENTS:
- Answer in exactly 2-3 sentences maximum
- Use specific facts, numbers, and terms from the document
- Be direct and factual
- Include exact figures, percentages, timeframes when mentioned
- No introductory phrases or explanations"""

    def _calculate_question_priority(self, question: str) -> int:
        """Calculate question priority for optimal batching"""
        priority = 0
        
        # Higher priority for questions with specific terms
        high_priority_terms = ["amount", "cost", "premium", "limit", "percentage", "when", "how much"]
        for term in high_priority_terms:
            if term in question.lower():
                priority += 10
        
        # Higher priority for shorter questions (faster to process)
        if len(question.split()) < 10:
            priority += 5
        
        return priority

    def _estimate_processing_time(self, question: str) -> float:
        """Estimate processing time for a question"""
        # Base time
        base_time = 2.0
        
        # Adjust based on complexity
        if len(question.split()) > 15:
            base_time += 1.0
        
        if any(term in question.lower() for term in ["compare", "difference", "versus"]):
            base_time += 1.5
        
        return base_time

    async def _get_from_cache(self, key: str) -> Any:
        """Get value from multi-level cache"""
        
        # L1: Memory cache
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if time.time() - entry.timestamp < entry.ttl:
                self.cache_stats["l1_hits"] += 1
                return entry.value
            else:
                del self.l1_cache[key]
        
        # L2: Redis cache
        if self.l2_cache:
            try:
                cached_data = await self.l2_cache.get(key)
                if cached_data:
                    value = pickle.loads(cached_data)
                    # Promote to L1 cache
                    self.l1_cache[key] = CacheEntry(
                        key=key, value=value, timestamp=time.time(), ttl=300, size_bytes=len(cached_data)
                    )
                    self.cache_stats["l2_hits"] += 1
                    return value
            except Exception as e:
                logger.warning(f"Redis cache error: {str(e)}")
        
        # L3: Disk cache
        try:
            cache_file = f"{self.l3_cache_path}/{hashlib.md5(key.encode()).hexdigest()}.pkl"
            async with aiofiles.open(cache_file, 'rb') as f:
                cached_data = await f.read()
                value = pickle.loads(cached_data)
                # Promote to L1 cache
                self.l1_cache[key] = CacheEntry(
                    key=key, value=value, timestamp=time.time(), ttl=300, size_bytes=len(cached_data)
                )
                self.cache_stats["l3_hits"] += 1
                return value
        except Exception:
            pass
        
        # Cache miss
        self.cache_stats["l1_misses"] += 1
        return None

    async def _set_cache(self, key: str, value: Any, ttl: float = 300):
        """Set value in multi-level cache"""
        
        # Serialize value
        serialized = pickle.dumps(value)
        size_bytes = len(serialized)
        
        # L1: Memory cache (with size limit)
        if size_bytes < 1024 * 1024:  # 1MB limit for memory cache
            self.l1_cache[key] = CacheEntry(
                key=key, value=value, timestamp=time.time(), ttl=ttl, size_bytes=size_bytes
            )
        
        # L2: Redis cache
        if self.l2_cache:
            try:
                await self.l2_cache.setex(key, int(ttl), serialized)
            except Exception as e:
                logger.warning(f"Redis cache set error: {str(e)}")
        
        # L3: Disk cache
        try:
            cache_file = f"{self.l3_cache_path}/{hashlib.md5(key.encode()).hexdigest()}.pkl"
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(serialized)
        except Exception as e:
            logger.warning(f"Disk cache set error: {str(e)}")

    def _get_document_cache_key(self, document_url: str) -> str:
        """Generate cache key for document"""
        return f"doc_{hashlib.md5(document_url.encode()).hexdigest()}"

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        total_hits = self.cache_stats["l1_hits"] + self.cache_stats["l2_hits"] + self.cache_stats["l3_hits"]
        total_requests = total_hits + self.cache_stats["l1_misses"]
        
        if total_requests == 0:
            return 0.0
        
        return total_hits / total_requests

    def _update_performance_metrics(self, total_time: float, num_questions: int):
        """Update performance metrics"""
        self.performance_metrics["total_processing_time"] = total_time
        self.performance_metrics["questions_processed"] += num_questions
        self.performance_metrics["cache_hit_rate"] = self._calculate_cache_hit_rate()

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            "performance_metrics": self.performance_metrics,
            "cache_stats": self.cache_stats,
            "retrieval_stats": self.retrieval_engine.get_search_stats() if self.retrieval_engine else {},
            "batch_optimizer_stats": self.batch_optimizer.get_stats()
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.vector_store:
            await self.vector_store.close()
        
        if self.l2_cache:
            await self.l2_cache.close()
        
        # Clear memory caches
        self.l1_cache.clear()
        self.query_embeddings_cache.clear()
        
        logger.info("High-Performance RAG Engine cleanup completed")


class BatchOptimizer:
    """Intelligent batch optimization for parallel processing"""
    
    def __init__(self):
        self.stats = {
            "optimizations_performed": 0,
            "avg_efficiency_gain": 0.0
        }
    
    def optimize_for_questions(self, questions: List[str]) -> Dict[str, int]:
        """Optimize batch configuration based on question characteristics"""
        
        num_questions = len(questions)
        
        # Analyze question complexity
        total_complexity = sum(self._calculate_complexity(q) for q in questions)
        avg_complexity = total_complexity / num_questions if num_questions > 0 else 1.0
        
        # Optimize batch size based on complexity and number of questions
        if avg_complexity > 3.0:  # High complexity
            batch_size = min(4, num_questions)
            max_parallel = min(8, num_questions)
        elif avg_complexity > 2.0:  # Medium complexity
            batch_size = min(8, num_questions)
            max_parallel = min(12, num_questions)
        else:  # Low complexity
            batch_size = min(12, num_questions)
            max_parallel = min(16, num_questions)
        
        self.stats["optimizations_performed"] += 1
        
        return {
            "batch_size": batch_size,
            "max_parallel": max_parallel,
            "estimated_complexity": avg_complexity
        }
    
    def _calculate_complexity(self, question: str) -> float:
        """Calculate question complexity score"""
        complexity = 1.0
        
        # Length factor
        word_count = len(question.split())
        if word_count > 20:
            complexity += 1.0
        elif word_count > 15:
            complexity += 0.5
        
        # Complexity indicators
        complex_terms = ["compare", "difference", "versus", "analyze", "explain"]
        for term in complex_terms:
            if term in question.lower():
                complexity += 0.5
        
        # Multiple concepts
        if "and" in question.lower() or "or" in question.lower():
            complexity += 0.5
        
        return complexity
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch optimizer statistics"""
        return self.stats