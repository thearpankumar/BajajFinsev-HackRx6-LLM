"""
Advanced RAG Engine with hybrid search and high accuracy focus
Combines dense and sparse retrieval with re-ranking for maximum accuracy
Now using Qdrant for improved Docker compatibility and performance
"""

import asyncio
import time
import logging
import json
from typing import List, Dict, Any, Tuple
import numpy as np
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    CrossEncoder = None
from openai import AsyncOpenAI
import google.generativeai as genai
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import re

from src.core.config import settings
from src.core.enhanced_document_processor import EnhancedDocumentProcessor, DocumentChunk
from src.core.qdrant_vector_store import QdrantVectorStore  # NEW: Using Qdrant instead of LanceDB
from src.core.document_cache_manager import DocumentCacheManager  # NEW: Document caching
from src.core.prompt_templates import PromptTemplates

# Initialize OpenAI client after importing settings
# Note: Client will be initialized in the RAGEngine.__init__ method

# Download required NLTK data
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    High-accuracy RAG engine with hybrid retrieval and advanced re-ranking
    """

    def __init__(self):
        self.document_processor = EnhancedDocumentProcessor()
        self.vector_store = None
        self.document_cache_manager = DocumentCacheManager()  # NEW: Document cache manager
        self.bm25_index = None
        self.cross_encoder = None
        self.prompt_templates = PromptTemplates()

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # Initialize Google AI
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.google_model = genai.GenerativeModel(settings.GOOGLE_MODEL)

        # Cache for processed documents
        self.document_cache = {}
        self.chunk_cache = {}

        # Performance tracking
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "avg_retrieval_time": 0,
            "avg_generation_time": 0,
        }
        
        # Resource management - use a simpler approach
        self._shutdown = False

    async def _rebuild_caches_from_qdrant(self):
        """Rebuild in-memory caches from existing Qdrant data"""
        try:
            logger.info("🔄 Rebuilding in-memory caches from Qdrant data...")
            
            # Get all points from Qdrant
            try:
                # Scroll through all points in the collection
                scroll_result = self.vector_store.qdrant_client.scroll(
                    collection_name=self.vector_store.collection_name,
                    limit=10000,  # Adjust based on your data size
                    with_payload=True,
                    with_vectors=False
                )
                
                # Handle the scroll result properly
                if isinstance(scroll_result, tuple):
                    points = scroll_result[0]  # First element is the points list
                else:
                    points = scroll_result
                    
            except Exception as e:
                logger.error(f"Failed to scroll Qdrant collection: {str(e)}")
                return
            
            if not points:
                logger.warning("No points found in Qdrant collection")
                return
            
            logger.info(f"Found {len(points)} points in Qdrant")
            
            # Group points by source URL
            documents_by_source = {}
            for point in points:
                try:
                    payload = point.payload
                    source_url = payload.get('source_url', 'unknown')
                    
                    if source_url not in documents_by_source:
                        documents_by_source[source_url] = []
                    
                    # Recreate DocumentChunk from payload
                    chunk = DocumentChunk(
                        text=payload.get('text', ''),
                        page_num=payload.get('page_num', 0),
                        chunk_id=payload.get('chunk_id', ''),
                        metadata=json.loads(payload.get('metadata', '{}'))
                    )
                    documents_by_source[source_url].append(chunk)
                    
                except Exception as e:
                    logger.warning(f"Error processing point: {str(e)}")
                    continue
            
            if not documents_by_source:
                logger.warning("No valid documents found in points")
                return
            
            # Rebuild caches for each document
            for source_url, chunks in documents_by_source.items():
                logger.info(f"Rebuilding cache for {source_url} ({len(chunks)} chunks)")
                
                # Rebuild chunk cache
                texts = [chunk.text for chunk in chunks]
                tokenized_texts = [self._tokenize_for_bm25(text) for text in texts]
                
                self.chunk_cache[source_url] = {
                    "chunks": chunks,
                    "texts": texts,
                    "tokenized_texts": tokenized_texts,
                }
                
                # Rebuild document cache (basic metadata)
                self.document_cache[source_url] = (chunks, {
                    "type": "rebuilt_from_qdrant",
                    "source_url": source_url,
                    "num_chunks": len(chunks),
                    "size": sum(len(chunk.text) for chunk in chunks)
                })
            
            # Rebuild BM25 index for the most recent document (or just pick the first one)
            if documents_by_source:
                # Use the first document for BM25 index
                first_source = list(documents_by_source.keys())[0]
                first_tokenized = self.chunk_cache[first_source]["tokenized_texts"]
                
                if first_tokenized and all(first_tokenized):  # Make sure we have valid tokens
                    self.bm25_index = BM25Okapi(first_tokenized)
                    logger.info(f"Rebuilt BM25 index for {first_source}")
                else:
                    logger.warning("Could not rebuild BM25 index - no valid tokens")
            
            logger.info(f"✅ Successfully rebuilt caches for {len(documents_by_source)} documents")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild caches: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't let this break the initialization
            logger.info("Continuing without cache rebuild...")

    async def cleanup(self):
        """
        Cleanup resources to prevent semaphore leaks
        """
        try:
            self._shutdown = True
            
            # Close OpenAI client gracefully
            try:
                if hasattr(self.openai_client, 'close'):
                    await self.openai_client.close()
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {str(e)}")
            
            # Cleanup vector store
            try:
                if self.vector_store and hasattr(self.vector_store, 'close'):
                    await self.vector_store.close()
            except Exception as e:
                logger.warning(f"Error closing vector store: {str(e)}")
            
            # Clear caches
            self.document_cache.clear()
            self.chunk_cache.clear()
            
            logger.info("RAG Engine cleanup completed")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    async def initialize(self):
        """Initialize the RAG engine components"""
        logger.info("Initializing RAG Engine...")

        # Initialize document cache manager first
        logger.info("📁 Initializing document cache manager...")
        await self.document_cache_manager.initialize()
        logger.info("✅ Document cache manager initialized")

        # Initialize Qdrant vector store (NEW: Using Qdrant instead of LanceDB)
        self.vector_store = QdrantVectorStore()
        await self.vector_store.initialize()

        # Skip cache rebuild for now - it will be built when documents are processed
        logger.info("Skipping cache rebuild - will be built on first document processing")

        # Don't initialize cross-encoder here - it's slow and optional
        # It will be lazy-loaded only if re-ranking is needed
        logger.info(
            "RAG Engine initialized successfully with Qdrant and document caching! (Cross-encoder will be loaded on demand)"
        )

    def _get_cross_encoder(self):
        """Lazy-load cross-encoder only when needed"""
        if not CROSS_ENCODER_AVAILABLE:
            logger.warning("Cross-encoder not available (sentence-transformers not installed)")
            return None
            
        if self.cross_encoder is None:
            logger.info("Loading cross-encoder model (this may take a moment)...")
            try:
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("Cross-encoder loaded successfully!")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                return None
        return self.cross_encoder

    async def _decompose_complex_question(self, question: str) -> List[str]:
        """
        Fast decomposition of complex multi-part questions
        Uses rule-based approach for speed with better pattern recognition
        """
        # Quick rule-based decomposition for common patterns
        question_lower = question.lower()
        
        # Pattern 1: "while X, also Y, and Z" -> Extract each distinct part
        if "while" in question_lower and ("also" in question_lower or "simultaneously" in question_lower):
            parts = []
            
            # Extract the main "while" clause
            while_part = question.split("while")[1].split(",")[0].strip() if "while" in question else ""
            if while_part and len(while_part) > 15:
                parts.append(while_part + "?")
            
            # Extract "also" parts
            remaining = question.lower()
            if "also" in remaining:
                also_parts = question.split("also")[1:]
                for part in also_parts:
                    clean_part = part.split(",")[0].split("and")[0].strip()
                    if clean_part and len(clean_part) > 15:
                        parts.append(clean_part + "?")
            
            # Extract "provide" or "confirm" parts
            if "provide" in question_lower:
                provide_parts = [p.strip() for p in question.split("provide")[1:]]
                for part in provide_parts:
                    clean_part = part.split(",")[0].split(".")[0].strip()
                    if clean_part and len(clean_part) > 10:
                        parts.append(f"What is {clean_part}?")
            
            if len(parts) >= 2:
                return parts[:3]  # Limit to 3 for speed
        
        # Pattern 2: "For X, what Y, how Z, and can W" -> Split on question words
        question_words = ["what", "how", "can", "when", "where", "why", "which"]
        if any(f", {word}" in question_lower for word in question_words):
            parts = []
            
            # Split on commas and look for question patterns
            comma_parts = [p.strip() for p in question.split(",")]
            current_question = ""
            
            for part in comma_parts:
                part_lower = part.lower()
                if any(part_lower.startswith(word) for word in question_words):
                    if current_question:
                        parts.append(current_question + "?")
                    current_question = part
                else:
                    if current_question:
                        current_question += ", " + part
                    else:
                        current_question = part
            
            if current_question:
                parts.append(current_question + "?")
            
            if len(parts) >= 2:
                return parts[:3]
        
        # Pattern 3: "simultaneously" or "at the same time" patterns
        if "simultaneously" in question_lower or "at the same time" in question_lower:
            # Split on these phrases
            split_phrases = ["simultaneously", "at the same time"]
            for phrase in split_phrases:
                if phrase in question_lower:
                    parts = question.split(phrase)
                    if len(parts) == 2:
                        return [parts[0].strip() + "?", parts[1].strip() + "?"]
        
        # Pattern 4: Multiple questions with "and" - more sophisticated splitting
        if question.count(" and ") >= 2 and len(question) > 100:
            # Look for natural break points
            parts = []
            sentences = question.replace(" and ", " AND ").split(" AND ")
            
            current_part = sentences[0]
            for i, sentence in enumerate(sentences[1:], 1):
                if len(current_part) > 50 and any(word in sentence.lower() for word in ["what", "how", "can", "provide", "confirm"]):
                    parts.append(current_part.strip() + "?")
                    current_part = sentence
                else:
                    current_part += " and " + sentence
            
            if current_part:
                parts.append(current_part.strip() + "?")
            
            if len(parts) >= 2:
                return parts[:3]
        
        # If no patterns match, return original question
        return [question]

    async def analyze_document(
        self, document_url: str, questions: List[str]
    ) -> Dict[str, Any]:
        """
        Main analysis function - processes document and answers questions with parallel processing

        Args:
            document_url: URL to the document
            questions: List of questions to answer

        Returns:
            Dictionary with answers and metadata
        """
        start_time = time.time()

        try:
            # Process document (with caching)
            chunks, doc_metadata = await self._get_or_process_document(document_url)

            # Index chunks in vector store
            await self._index_chunks(chunks, document_url)

            # Process questions in parallel if enabled and we have multiple questions
            if settings.PARALLEL_PROCESSING and len(questions) > 1:
                logger.info(f"Processing {len(questions)} questions in parallel...")
                answers = await self._answer_questions_parallel(
                    questions, chunks, document_url
                )
            else:
                # Sequential processing for single questions or when parallel is disabled
                logger.info(f"Processing {len(questions)} questions sequentially...")
                answers = []
                for question in questions:
                    answer = await self._answer_question(question, chunks, document_url)
                    answers.append(answer)

            processing_time = time.time() - start_time

            # Update stats
            self.stats["total_queries"] += len(questions)

            return {
                "answers": answers,
                "document_size": doc_metadata.get("size", 0),
                "metadata": {
                    "processing_time": processing_time,
                    "num_chunks": len(chunks),
                    "doc_metadata": doc_metadata,
                    "parallel_processing": settings.PARALLEL_PROCESSING
                    and len(questions) > 1,
                    "questions_processed": len(questions),
                },
            }

        except Exception as e:
            logger.error(f"Error in analyze_document: {str(e)}")
            raise

    async def _answer_questions_parallel(
        self, questions: List[str], chunks: List[DocumentChunk], document_url: str
    ) -> List[str]:
        """
        Process multiple questions in parallel with batching and concurrency control

        Args:
            questions: List of questions to answer
            chunks: Document chunks
            document_url: URL of the document

        Returns:
            List of answers in the same order as questions
        """
        # Limit the number of questions to process
        max_questions = min(len(questions), settings.MAX_PARALLEL_QUESTIONS)
        questions_to_process = questions[:max_questions]

        logger.info(
            f"Processing {len(questions_to_process)} questions with parallel processing"
        )

        # Create semaphore to limit concurrent operations - use 15 for better performance
        semaphore = asyncio.Semaphore(min(settings.QUESTION_BATCH_SIZE, 15))  # Limit to 15 concurrent

        async def process_single_question_with_semaphore(
            question: str, index: int
        ) -> Tuple[int, str]:
            """Process a single question with semaphore control"""
            try:
                async with semaphore:
                    if self._shutdown:
                        return (index, "Processing interrupted")
                        
                    logger.info(
                        f"Processing question {index + 1}/{len(questions_to_process)}: {question[:50]}..."
                    )
                    answer = await self._answer_question(question, chunks, document_url)
                    logger.info(f"Completed question {index + 1}")
                    return (index, answer)
            except asyncio.CancelledError:
                return (index, "Question processing was cancelled")
            except Exception as e:
                logger.error(f"Error processing question {index + 1}: {str(e)}")
                return (index, f"Error processing question: {str(e)}")

        # Create tasks for all questions
        tasks = [
            process_single_question_with_semaphore(question, i)
            for i, question in enumerate(questions_to_process)
        ]

        # Execute all tasks concurrently with timeout
        start_time = time.time()
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300.0  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            logger.error("Parallel processing timed out, falling back to sequential")
            # Fallback to sequential processing
            results = []
            for i, question in enumerate(questions_to_process):
                try:
                    answer = await self._answer_question(question, chunks, document_url)
                    results.append((i, answer))
                except Exception as ex:
                    results.append((i, f"Error: {str(ex)}"))
        
        parallel_time = time.time() - start_time

        logger.info(f"Parallel processing completed in {parallel_time:.2f} seconds")

        # Sort results by original question order and extract answers
        answers = [""] * len(questions_to_process)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                continue

            index, answer = result
            answers[index] = answer

        # If we had more questions than the limit, add error messages for the rest
        if len(questions) > max_questions:
            remaining_count = len(questions) - max_questions
            for i in range(remaining_count):
                answers.append(
                    f"Question limit exceeded. Maximum {settings.MAX_PARALLEL_QUESTIONS} questions can be processed in parallel."
                )

        return answers

    async def stream_analyze(
        self, document_url: str, questions: List[str]
    ) -> Dict[str, Any]:
        """
        Streaming analysis for faster initial responses
        """
        # For now, return quick placeholder answers
        # In production, this would start processing and return initial results
        initial_answers = [f"Processing question: {q[:50]}..." for q in questions]

        return {
            "initial_answers": initial_answers,
            "eta": min(30, len(questions) * 5),  # Estimate 5 seconds per question
        }

    async def _get_or_process_document(
        self, document_url: str
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Get document from cache or process it with intelligent caching strategy
        
        Priority order:
        1. Memory cache (fastest)
        2. Persistent document cache (fast)
        3. Vector database reconstruction (medium)
        4. Download and process (slowest)
        """
        logger.info(f"🔍 Getting document: {document_url}")
        
        # Step 1: Check memory cache first (fastest)
        if document_url in self.document_cache:
            logger.info("✅ Found document in memory cache")
            self.stats["cache_hits"] += 1
            return self.document_cache[document_url]

        # Step 2: Check persistent document cache (fast)
        if settings.ENABLE_PERSISTENT_DOCUMENT_CACHE:
            logger.info("🔍 Checking persistent document cache...")
            cached_result = await self.document_cache_manager.get_cached_document(document_url)
            if cached_result:
                chunks, metadata = cached_result
                logger.info(f"✅ Found document in persistent cache with {len(chunks)} chunks")
                
                # Store in memory cache for faster future access
                self.document_cache[document_url] = (chunks, metadata)
                self.stats["cache_hits"] += 1
                return chunks, metadata

        # Step 3: Check if document exists in vector database (medium speed)
        if settings.CHECK_VECTOR_DB_BEFORE_DOWNLOAD:
            logger.info("🔍 Checking if document exists in vector database...")
            if await self.vector_store.document_exists(document_url):
                logger.info("✅ Found document in vector database, reconstructing...")
                
                # Get chunks from vector database
                chunks_with_scores = await self.vector_store.get_document_chunks_from_db(document_url)
                if chunks_with_scores:
                    chunks = [chunk for chunk, _ in chunks_with_scores]
                    metadata = await self.vector_store.get_document_metadata_from_db(document_url)
                    
                    logger.info(f"✅ Reconstructed document from vector DB with {len(chunks)} chunks")
                    
                    # Cache in memory and persistent storage
                    self.document_cache[document_url] = (chunks, metadata)
                    if settings.ENABLE_PERSISTENT_DOCUMENT_CACHE:
                        await self.document_cache_manager.cache_document(document_url, chunks, metadata)
                    
                    self.stats["cache_hits"] += 1
                    return chunks, metadata
                else:
                    logger.warning("⚠️ Document exists in vector DB but couldn't retrieve chunks")

        # Step 4: Download and process document (slowest)
        logger.info("📥 Document not found in any cache, downloading and processing...")
        
        # Check if we should skip duplicate processing
        if settings.SKIP_DUPLICATE_DOCUMENTS and document_url in self.document_cache:
            logger.info("⚠️ Document processing already in progress, using existing result")
            return self.document_cache[document_url]

        try:
            # Process document from URL
            chunks, metadata = await self.document_processor.process_document(document_url)
            logger.info(f"✅ Processed document with {len(chunks)} chunks")

            # Add processing metadata
            metadata.update({
                'source': 'downloaded_and_processed',
                'cache_hit': False,
                'processing_method': 'full_download'
            })

            # Cache in memory
            self.document_cache[document_url] = (chunks, metadata)

            # Cache in persistent storage
            if settings.ENABLE_PERSISTENT_DOCUMENT_CACHE:
                cache_success = await self.document_cache_manager.cache_document(document_url, chunks, metadata)
                if cache_success:
                    logger.info("✅ Document cached persistently")
                else:
                    logger.warning("⚠️ Failed to cache document persistently")

            return chunks, metadata

        except Exception as e:
            logger.error(f"❌ Failed to process document {document_url}: {str(e)}")
            raise

    async def _index_chunks(self, chunks: List[DocumentChunk], document_url: str):
        """Index chunks in both vector store and BM25 with smart duplicate detection"""
        logger.info(f"🔍 Checking indexing status for document: {document_url}")
        
        # Check memory cache first
        if document_url in self.chunk_cache:
            logger.info(f"✅ Document {document_url} already indexed in memory, skipping...")
            return

        # Check if document already exists in vector database
        if settings.CHECK_VECTOR_DB_BEFORE_DOWNLOAD:
            if await self.vector_store.document_exists(document_url):
                logger.info(f"✅ Document {document_url} already exists in vector database")
                
                # Just rebuild memory caches without re-adding to vector store
                logger.info("🔄 Rebuilding memory caches from existing data...")
                await self._rebuild_memory_cache_for_document(document_url, chunks)
                return

        logger.info(f"📝 Indexing {len(chunks)} chunks for document: {document_url}")

        # Prepare texts for indexing
        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "chunk_id": chunk.chunk_id,
                "page_num": chunk.page_num,
                "word_count": chunk.word_count,
                "source_url": document_url,
                **chunk.metadata,
            }
            for chunk in chunks
        ]

        # Add to vector store (only if not already there)
        try:
            existing_stats = await self.vector_store.get_stats()
            logger.info(f"Vector store has {existing_stats.get('total_vectors', 0)} existing vectors")
            
            logger.info("📤 Adding texts to vector store...")
            await self.vector_store.add_texts(texts, metadatas)
            logger.info("✅ Successfully added texts to vector store")
            
        except Exception as e:
            logger.error(f"❌ Error adding texts to vector store: {str(e)}")
            # Continue with BM25 even if vector store fails
            pass

        # Build memory caches
        await self._rebuild_memory_cache_for_document(document_url, chunks)

        logger.info(f"✅ Indexed {len(chunks)} chunks for {document_url}")

    async def _rebuild_memory_cache_for_document(self, document_url: str, chunks: List[DocumentChunk]):
        """Rebuild memory caches for a document"""
        try:
            logger.info(f"🔄 Rebuilding memory cache for {document_url}")
            
            # Prepare texts for BM25
            texts = [chunk.text for chunk in chunks]
            tokenized_texts = [self._tokenize_for_bm25(text) for text in texts]
            
            # Create BM25 index (always create for new documents)
            if tokenized_texts and all(tokenized_texts):
                self.bm25_index = BM25Okapi(tokenized_texts)
                logger.info("✅ BM25 index created successfully")
            else:
                logger.warning("⚠️ Could not create BM25 index - no valid tokens")

            # Cache chunk information in memory
            self.chunk_cache[document_url] = {
                "chunks": chunks,
                "texts": texts,
                "tokenized_texts": tokenized_texts,
            }

            logger.info(f"✅ Memory cache rebuilt for {document_url} with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"❌ Failed to rebuild memory cache for {document_url}: {str(e)}")
            raise

    def _tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing"""
        # Convert to lowercase and split
        tokens = re.findall(r"\b\w+\b", text.lower())

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [
            token for token in tokens if token not in stop_words and len(token) > 2
        ]

        return tokens

    async def _answer_question(
        self, question: str, chunks: List[DocumentChunk], document_url: str
    ) -> str:
        """
        Answer a single question using hybrid retrieval and generation
        Handles complex multi-part questions by decomposing them
        """
        retrieval_start = time.time()

        # Step 0: Check if this is a complex multi-part question (if enabled)
        if settings.ENABLE_QUESTION_DECOMPOSITION:
            sub_questions = await self._decompose_complex_question(question)
        else:
            sub_questions = [question]
        
        if len(sub_questions) > 1 and settings.FAST_COMPLEX_QUESTIONS:
            # Use fast processing for complex questions
            return await self._fast_process_complex_question(question, sub_questions, document_url, retrieval_start)
        elif len(sub_questions) > 1:
            # Handle multi-part question with optimized processing
            logger.info(f"Processing multi-part question with {len(sub_questions)} parts")
            
            # Process sub-questions in parallel for speed
            if settings.PARALLEL_PROCESSING and len(sub_questions) > 2:
                logger.info("Processing sub-questions in parallel for speed")
                
                async def process_sub_question(sub_q):
                    relevant_chunks = await self._hybrid_retrieval(sub_q, document_url)
                    reranked_chunks = await self._rerank_chunks(
                        sub_q, relevant_chunks, use_reranking=False  # Skip reranking for speed
                    )
                    return await self._generate_answer(sub_q, reranked_chunks)
                
                # Process all sub-questions in parallel
                sub_answers = await asyncio.gather(*[
                    process_sub_question(sub_q) for sub_q in sub_questions
                ])
                
            else:
                # Sequential processing for smaller sets
                sub_answers = []
                for i, sub_question in enumerate(sub_questions):
                    logger.info(f"Processing sub-question {i+1}: {sub_question[:50]}...")
                    
                    relevant_chunks = await self._hybrid_retrieval(sub_question, document_url)
                    reranked_chunks = await self._rerank_chunks(
                        sub_question, relevant_chunks, use_reranking=False  # Skip reranking for speed
                    )
                    sub_answer = await self._generate_answer(sub_question, reranked_chunks)
                    sub_answers.append(sub_answer)
            
            # Fast combination - just join with connecting words
            combined_answer = await self._fast_combine_answers(question, sub_questions, sub_answers)
            
            retrieval_time = time.time() - retrieval_start
            self.stats["avg_retrieval_time"] = (
                self.stats["avg_retrieval_time"] + retrieval_time
            ) / 2
            
            return combined_answer
        
        else:
            # Handle single-part question (original logic)
            # Step 1: Hybrid Retrieval
            relevant_chunks = await self._hybrid_retrieval(question, document_url)

            retrieval_time = time.time() - retrieval_start

            # Step 2: Re-ranking (configurable via settings)
            reranked_chunks = await self._rerank_chunks(
                question, relevant_chunks, use_reranking=settings.ENABLE_RERANKING
            )

            # Step 3: Generate answer
            generation_start = time.time()
            answer = await self._generate_answer(question, reranked_chunks)
            generation_time = time.time() - generation_start

            # Update stats
            self.stats["avg_retrieval_time"] = (
                self.stats["avg_retrieval_time"] + retrieval_time
            ) / 2
            self.stats["avg_generation_time"] = (
                self.stats["avg_generation_time"] + generation_time
            ) / 2

        return answer

    async def _combine_sub_answers(
        self, original_question: str, sub_questions: List[str], sub_answers: List[str]
    ) -> str:
        """
        Combine multiple sub-answers into a comprehensive response for complex questions
        """
        try:
            # Create a structured response that addresses all parts
            combination_prompt = f"""
            You are an insurance policy expert. You were asked a complex question that was broken down into parts. 
            Combine the individual answers into one comprehensive, well-structured response that fully addresses the original question.
            
            Original question: {original_question}
            
            Sub-questions and their answers:
            """
            
            for i, (sub_q, sub_a) in enumerate(zip(sub_questions, sub_answers), 1):
                combination_prompt += f"\n{i}. {sub_q}\nAnswer: {sub_a}\n"
            
            combination_prompt += """
            
            Instructions for combining:
            1. Provide a comprehensive answer that addresses ALL aspects of the original question
            2. Maintain the specific policy details and references from each sub-answer
            3. Structure the response logically to cover each part of the original question
            4. Keep it detailed but concise (3-4 sentences maximum)
            5. If some information is not available in the policy, clearly state that
            6. Use specific policy section references when available
            7. Ensure the answer flows naturally and addresses the question completely
            
            Combined comprehensive answer:
            """
            
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_GENERATION_MODEL,
                messages=[{"role": "user", "content": combination_prompt}],
                max_tokens=settings.COMPLEX_QUESTION_MAX_TOKENS,  # Use dedicated setting for complex questions
                temperature=settings.GENERATION_TEMPERATURE
            )
            
            combined_answer = response.choices[0].message.content.strip()
            logger.info("Successfully combined sub-answers into comprehensive response")
            return combined_answer
            
        except Exception as e:
            logger.warning(f"Failed to combine sub-answers: {str(e)}")
            # Fallback: create a structured combination manually
            combined_parts = []
            for i, (sub_q, sub_a) in enumerate(zip(sub_questions, sub_answers)):
                if i == 0:
                    combined_parts.append(sub_a)
                else:
                    # Add connecting words for subsequent parts
                    combined_parts.append(f"Additionally, {sub_a.lower()}")
            
            return " ".join(combined_parts)

    async def _fast_combine_answers(
        self, original_question: str, sub_questions: List[str], sub_answers: List[str]
    ) -> str:
        """
        Fast combination of sub-answers without AI processing for speed
        Creates natural flow without repetitive connecting words
        """
        if len(sub_answers) == 1:
            return sub_answers[0]
        
        # Analyze the original question to determine combination style
        question_lower = original_question.lower()
        
        # For questions asking multiple specific things, use structured approach
        if "also" in question_lower or "simultaneously" in question_lower or "and" in question_lower:
            combined_parts = []
            
            for i, answer in enumerate(sub_answers):
                if i == 0:
                    combined_parts.append(answer)
                elif i == 1:
                    # Use different connectors based on content
                    if "document does not contain" in answer.lower():
                        combined_parts.append(f"Regarding the second part, {answer.lower()}")
                    else:
                        combined_parts.append(f"For the second aspect, {answer}")
                elif i == 2:
                    if "document does not contain" in answer.lower():
                        combined_parts.append(f"As for the third part, {answer.lower()}")
                    else:
                        combined_parts.append(f"Finally, {answer}")
                else:
                    combined_parts.append(answer)
            
            return " ".join(combined_parts)
        
        # For other complex questions, use simple joining
        else:
            # Remove redundant phrases and create smooth flow
            cleaned_answers = []
            for answer in sub_answers:
                # Remove common redundant starts
                cleaned = answer
                if cleaned.startswith("The document does not contain information about"):
                    cleaned = "The document doesn't specify " + cleaned[47:]
                elif cleaned.startswith("According to the document"):
                    cleaned = cleaned[25:]
                
                cleaned_answers.append(cleaned)
            
            return " ".join(cleaned_answers)

    async def _fast_process_complex_question(
        self, question: str, sub_questions: List[str], document_url: str, retrieval_start: float
    ) -> str:
        """
        Fast processing for complex questions - optimized for speed over perfect accuracy
        """
        logger.info(f"Fast processing {len(sub_questions)} sub-questions")
        
        # Process sub-questions in parallel for maximum speed
        async def fast_process_sub_question(sub_q):
            if self._shutdown:
                return "Processing interrupted"
                
            # Enhance query for better retrieval if enabled
            if settings.ENABLE_QUERY_ENHANCEMENT:
                enhanced_query = await self._enhance_query_for_retrieval(sub_q)
            else:
                enhanced_query = sub_q
            
            # Temporarily lower similarity threshold for complex questions
            original_threshold = settings.SIMILARITY_THRESHOLD
            settings.SIMILARITY_THRESHOLD = max(0.05, original_threshold * 0.5)  # Lower threshold
            
            try:
                # Use fewer chunks for speed but better query
                relevant_chunks = await self._hybrid_retrieval(enhanced_query, document_url)
                # Skip reranking for speed
                top_chunks = [chunk for chunk, _ in relevant_chunks[:5]]  # Take top 5 only
                return await self._generate_answer(sub_q, top_chunks)
            finally:
                # Restore original threshold
                settings.SIMILARITY_THRESHOLD = original_threshold
        
        # Process all sub-questions in parallel
        sub_answers = await asyncio.gather(*[
            fast_process_sub_question(sub_q) for sub_q in sub_questions
        ], return_exceptions=True)
        
        # Handle any exceptions in sub-answers
        processed_answers = []
        for answer in sub_answers:
            if isinstance(answer, Exception):
                processed_answers.append(f"Error processing sub-question: {str(answer)}")
            else:
                processed_answers.append(answer)
        
        # Fast combination without AI
        combined_answer = await self._fast_combine_answers(question, sub_questions, processed_answers)
        
        retrieval_time = time.time() - retrieval_start
        self.stats["avg_retrieval_time"] = (
            self.stats["avg_retrieval_time"] + retrieval_time
        ) / 2
        
        return combined_answer

    async def _enhance_query_for_retrieval(self, query: str) -> str:
        """
        Fast query enhancement for better retrieval without AI calls
        """
        query_lower = query.lower()
        
        # Add relevant insurance terms based on query content
        enhancements = []
        
        # Dental-related queries
        if "dental" in query_lower:
            enhancements.extend(["dental treatment", "dental surgery", "dental claim", "OPD dental"])
        
        # Claim-related queries
        if any(word in query_lower for word in ["claim", "submit", "documents"]):
            enhancements.extend(["claim process", "documentation", "TPA", "reimbursement"])
        
        # Dependent-related queries
        if any(word in query_lower for word in ["dependent", "child", "spouse", "parent"]):
            enhancements.extend(["dependent eligibility", "family member", "coverage"])
        
        # Process-related queries
        if any(word in query_lower for word in ["process", "procedure", "steps"]):
            enhancements.extend(["procedure", "requirements", "notification"])
        
        # Hospital-related queries
        if any(word in query_lower for word in ["hospital", "network", "cashless"]):
            enhancements.extend(["network hospital", "cashless facility", "pre-authorization"])
        
        # Contact-related queries
        if any(word in query_lower for word in ["email", "contact", "phone", "grievance"]):
            enhancements.extend(["contact details", "customer service", "grievance redressal"])
        
        # Combine original query with enhancements
        if enhancements:
            enhanced = f"{query} {' '.join(enhancements[:3])}"  # Limit to 3 enhancements
            return enhanced
        
        return query

    async def _hybrid_retrieval(
        self, query: str, document_url: str
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Enhanced hybrid retrieval combining dense (vector) and sparse (BM25) search
        Optimized for large documents with better query processing
        """
        logger.info(f"🔍 Starting hybrid retrieval for query: '{query}'")
        
        # Enhanced query processing for better retrieval (configurable)
        if settings.FAST_MODE and settings.USE_ENHANCED_QUERY:
            search_query = self._enhance_query_for_large_docs(query)
        elif not settings.FAST_MODE:
            search_query = self._expand_query(query)
        else:
            search_query = query  # No enhancement
        
        logger.info(f"Enhanced query: '{search_query}'")

        # Dense retrieval using vector similarity with higher k for large docs
        retrieval_k = min(settings.TOP_K_RETRIEVAL * 2, 50)  # Increase for large docs
        logger.info(f"Performing dense retrieval with k={retrieval_k}")
        
        dense_results = await self.vector_store.similarity_search(
            search_query,
            k=retrieval_k,
        )
        
        logger.info(f"Dense retrieval found {len(dense_results)} results")
        if dense_results:
            logger.info(f"Top dense result score: {dense_results[0][1]:.4f}")
            logger.info(f"Top dense result preview: {dense_results[0][0].text[:200]}...")
        else:
            logger.warning("❌ No dense results found!")

        # Sparse retrieval using BM25 (only if we have BM25 index)
        sparse_results = []
        if self.bm25_index and document_url in self.chunk_cache:
            logger.info("Performing sparse (BM25) retrieval...")
            
            # Use enhanced query for BM25 as well
            enhanced_tokens = self._tokenize_for_bm25(search_query)
            enhanced_scores = self.bm25_index.get_scores(enhanced_tokens)

            chunks = self.chunk_cache[document_url]["chunks"]

            # Get top BM25 results with lower threshold for specific terms
            top_indices = np.argsort(enhanced_scores)[::-1][:retrieval_k]
            min_score_threshold = 0.5  # Lower threshold to catch more results

            for idx in top_indices:
                if enhanced_scores[idx] > min_score_threshold:
                    sparse_results.append((chunks[idx], float(enhanced_scores[idx])))
            
            logger.info(f"Sparse retrieval found {len(sparse_results)} results above threshold")
            if sparse_results:
                logger.info(f"Top sparse result score: {sparse_results[0][1]:.4f}")
                logger.info(f"Top sparse result preview: {sparse_results[0][0].text[:200]}...")
            else:
                logger.warning("❌ No sparse results found above threshold!")
                # Show top results even if below threshold for debugging
                if enhanced_scores.max() > 0:
                    best_idx = np.argmax(enhanced_scores)
                    logger.info(f"Best BM25 score: {enhanced_scores[best_idx]:.4f}")
                    logger.info(f"Best BM25 text: {chunks[best_idx].text[:200]}...")
        else:
            logger.info("⚠️ BM25 index not available - using vector search only")

        # Use enhanced RRF for large documents (configurable)
        if settings.USE_ENHANCED_RRF and sparse_results:
            # Only use enhanced RRF if we have sparse results
            combined_results = self._enhanced_reciprocal_rank_fusion_for_large_docs(
                dense_results, sparse_results, query
            )
        else:
            # Use dense results only when no sparse results
            logger.info("Using dense results only (no BM25 available)")
            combined_results = [(chunk, score) for chunk, score in dense_results]

        # Return top results with better filtering
        final_results = combined_results[:settings.TOP_K_RETRIEVAL]
        logger.info(f"Final hybrid retrieval: {len(final_results)} results")
        
        # Fallback: If no good results, try with lower thresholds
        if len(final_results) < 3:
            logger.warning("⚠️ Few results found, trying fallback search with lower thresholds...")
            
            # Try with much lower similarity threshold and more results
            fallback_dense = await self.vector_store.similarity_search(
                query,  # Use original query
                k=50,  # Get more results
            )
            
            logger.info(f"Fallback search found {len(fallback_dense)} results")
            
            # Add fallback results if we still don't have enough
            for chunk, score in fallback_dense[:15]:  # Take top 15
                if len(final_results) >= settings.TOP_K_RETRIEVAL:
                    break
                # Avoid duplicates
                if not any(existing_chunk.chunk_id == chunk.chunk_id for existing_chunk, _ in final_results):
                    final_results.append((chunk, score * 0.8))  # Slightly lower score for fallback
            
            logger.info(f"Added {len(fallback_dense)} fallback results")
        
        # If still no results, there's a real problem
        if not final_results:
            logger.error("❌ No results found even with fallback search!")
            logger.error("This suggests the vector database might be empty or disconnected")
        
        # Debug: Show final results
        for i, (chunk, score) in enumerate(final_results[:3]):
            logger.info(f"Final result {i+1}: Score={score:.4f}")
            logger.info(f"  Text: {chunk.text[:150]}...")
        
        return final_results

    def _simple_reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Simple and fast RRF implementation
        """
        # Create chunk ID to result mapping
        chunk_scores = {}
        chunk_map = {}

        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            rrf_score = settings.DENSE_WEIGHT / (60 + rank + 1)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            rrf_score = settings.SPARSE_WEIGHT / (60 + rank + 1)
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

        # Create combined results
        combined_results = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in sorted(
                chunk_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return combined_results

    def _enhance_query_for_large_docs(self, query: str) -> str:
        """Enhanced query processing for multi-domain large documents"""
        key_terms = []
        query_lower = query.lower()
        
        # Multi-domain expansions
        domain_expansions = {
            # Insurance domain - Enhanced for specific terms
            "grace period": "grace period premium payment delay extension time allowed",
            "grace": "grace period premium payment delay extension",
            "premium payment": "premium payment due date grace period installment",
            "policy": "policy insurance coverage plan benefit document",
            "premium": "premium payment cost fee amount installment",
            "coverage": "coverage benefit protection insurance plan",
            "claim": "claim request application reimbursement settlement",
            "deductible": "deductible excess copay out-of-pocket amount",
            "waiting": "waiting period delay time eligibility qualification",
            "exclusion": "exclusion exception limitation restriction not covered",
            "maternity": "maternity pregnancy childbirth delivery benefit",
            "pre-existing": "pre-existing prior existing previous condition disease",
            "mediclaim": "mediclaim medical insurance health coverage policy",
            "national": "national insurance company policy provider",
            "parivar": "parivar family insurance policy coverage plan",
            
            # Legal domain
            "contract": "contract agreement document terms conditions",
            "jurisdiction": "jurisdiction court legal authority",
            "liability": "liability responsibility obligation duty",
            "breach": "breach violation infringement default",
            "damages": "damages compensation remedy relief",
            "clause": "clause provision section article term",
            
            # HR domain
            "employee": "employee worker staff personnel",
            "employment": "employment job work position role",
            "benefits": "benefits compensation package perks",
            "leave": "leave absence vacation sick time",
            "performance": "performance evaluation review assessment",
            "termination": "termination dismissal separation end",
            
            # Compliance domain
            "regulation": "regulation rule requirement standard",
            "audit": "audit review inspection examination",
            "compliance": "compliance adherence conformity following",
            "violation": "violation breach infringement non-compliance",
            "reporting": "reporting disclosure notification filing",
            
            # Philosophy domain
            "ethics": "ethics moral philosophy principles values",
            "logic": "logic reasoning argument inference",
            "truth": "truth reality fact knowledge belief",
            "justice": "justice fairness equity rights",
            "virtue": "virtue character excellence moral",
            
            # Science domain
            "newton": "newton isaac principia physics mechanics",
            "gravity": "gravity gravitational force attraction",
            "motion": "motion movement dynamics kinematics",
            "force": "force forces mechanics dynamics",
            "law": "law laws principle theorem rule",
            "experiment": "experiment test observation study",
            "theory": "theory hypothesis explanation model",
            "evidence": "evidence proof data observation",
        }
        
        # Add relevant expansions based on query content
        for term, expansion in domain_expansions.items():
            if term in query_lower:
                key_terms.append(expansion)
        
        # Also add general academic/professional terms
        academic_terms = {
            "definition": "definition meaning explanation description",
            "procedure": "procedure process method steps protocol",
            "requirement": "requirement condition necessity obligation",
            "document": "document paper record file form",
            "section": "section part chapter clause provision",
            "amount": "amount sum total cost price value",
            "time": "time period duration timeframe deadline",
            "condition": "condition requirement state situation",
        }
        
        for term, expansion in academic_terms.items():
            if term in query_lower:
                key_terms.append(expansion)
        
        # Combine original query with relevant expansions
        if key_terms:
            # Limit expansions to avoid query bloat
            selected_expansions = key_terms[:3]  # Max 3 expansions
            enhanced_query = query + " " + " ".join(selected_expansions)
            return enhanced_query
        
        return query

    def _enhanced_reciprocal_rank_fusion_for_large_docs(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        original_query: str,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Enhanced RRF optimized for large multi-domain documents
        Works across insurance, legal, HR, compliance, philosophy, and science domains
        """
        chunk_scores = {}
        chunk_map = {}
        
        # Extract query keywords for relevance boosting (domain-agnostic)
        query_keywords = set(word.lower() for word in original_query.split() if len(word) > 2)
        
        # Process dense results with relevance boosting
        for rank, (chunk, score) in enumerate(dense_results):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            
            # Base RRF score with higher weight for dense results in large docs
            rrf_score = 0.8 / (60 + rank + 1)  # Increased weight for vector similarity
            
            # Domain-agnostic keyword matching boost
            chunk_words = set(word.lower() for word in chunk.text.split() if len(word) > 2)
            keyword_matches = len(query_keywords.intersection(chunk_words))
            if keyword_matches > 0:
                # Scale boost based on query length to avoid over-boosting short queries
                boost_factor = min(0.3, 0.1 * keyword_matches)
                rrf_score *= (1 + boost_factor)
            
            # Boost for informative chunks (domain-agnostic)
            if chunk.word_count > 50:  # Reasonable threshold for all domains
                rrf_score *= 1.1
            
            # Boost for chunks with professional/academic language patterns
            professional_indicators = [
                'shall', 'pursuant', 'accordance', 'provision', 'section',
                'definition', 'requirement', 'condition', 'procedure', 'policy',
                'regulation', 'compliance', 'standard', 'principle', 'theory'
            ]
            
            chunk_lower = chunk.text.lower()
            professional_matches = sum(1 for indicator in professional_indicators if indicator in chunk_lower)
            if professional_matches > 0:
                rrf_score *= (1 + 0.05 * professional_matches)  # Small boost for professional language
            
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

        # Process sparse results with keyword focus
        for rank, (chunk, score) in enumerate(sparse_results):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk
            
            # Base RRF score for sparse results
            rrf_score = 0.4 / (60 + rank + 1)  # Lower base weight for BM25
            
            # Boost based on BM25 score strength (domain-agnostic)
            if score > 10.0:  # Very high BM25 score
                rrf_score *= 1.8
            elif score > 5.0:  # High BM25 score
                rrf_score *= 1.4
            elif score > 2.0:  # Moderate BM25 score
                rrf_score *= 1.2
            
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

        # Create combined results sorted by enhanced scores
        combined_results = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in sorted(
                chunk_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return combined_results

    def _expand_query(self, query: str) -> str:
        """Expand query with domain-specific synonyms and related terms"""
        # Common insurance/legal/HR synonyms
        expansions = {
            "grace period": "grace period waiting time allowance",
            "premium": "premium payment amount fee",
            "coverage": "coverage benefit protection",
            "waiting period": "waiting period delay time",
            "claim": "claim request application",
            "policy": "policy document contract",
            "benefit": "benefit advantage coverage",
            "exclusion": "exclusion exception limitation",
            "deductible": "deductible excess amount",
            "maternity": "maternity pregnancy childbirth",
            "pre-existing": "pre-existing prior existing previous",
            "renewal": "renewal continuation extension",
            "discount": "discount reduction benefit",
            "hospital": "hospital medical facility institution",
            "treatment": "treatment therapy care medical",
        }

        expanded_terms = []
        query_lower = query.lower()

        for term, expansion in expansions.items():
            if term in query_lower:
                expanded_terms.append(expansion)

        if expanded_terms:
            return query + " " + " ".join(expanded_terms)

        return query

    def _enhanced_reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
        original_query: str,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Enhanced RRF that considers query relevance and chunk quality
        """
        # Create chunk ID to result mapping with enhanced scoring
        chunk_scores = {}
        chunk_map = {}

        # Process dense results with query relevance boost
        for rank, (chunk, score) in enumerate(dense_results):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk

            # Base RRF score
            rrf_score = settings.DENSE_WEIGHT / (60 + rank + 1)

            # Boost score if chunk contains exact query terms
            query_terms = set(original_query.lower().split())
            chunk_terms = set(chunk.text.lower().split())
            exact_matches = len(query_terms.intersection(chunk_terms))

            if exact_matches > 0:
                rrf_score *= 1 + 0.1 * exact_matches  # 10% boost per exact match

            # Boost score for longer, more informative chunks
            if chunk.word_count > 50:
                rrf_score *= 1.1

            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

        # Process sparse results with similar enhancements
        for rank, (chunk, score) in enumerate(sparse_results):
            chunk_id = chunk.chunk_id
            chunk_map[chunk_id] = chunk

            # Base RRF score
            rrf_score = settings.SPARSE_WEIGHT / (60 + rank + 1)

            # Boost for high BM25 scores
            if score > 5.0:  # High BM25 score threshold
                rrf_score *= 1.2

            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

        # Create combined results sorted by enhanced scores
        combined_results = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in sorted(
                chunk_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return combined_results

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[DocumentChunk, float]],
        sparse_results: List[Tuple[DocumentChunk, float]],
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Combine dense and sparse results using Reciprocal Rank Fusion
        """
        # Create chunk ID to result mapping
        chunk_scores = {}

        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results):
            chunk_id = chunk.chunk_id
            rrf_score = settings.DENSE_WEIGHT / (60 + rank + 1)  # RRF with k=60
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results):
            chunk_id = chunk.chunk_id
            rrf_score = settings.SPARSE_WEIGHT / (60 + rank + 1)  # RRF with k=60
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score

        # Create combined results
        chunk_map = {}
        for chunk, _ in dense_results + sparse_results:
            chunk_map[chunk.chunk_id] = chunk

        combined_results = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in sorted(
                chunk_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]

        return combined_results

    async def _rerank_chunks(
        self,
        query: str,
        chunks: List[Tuple[DocumentChunk, float]],
        use_reranking: bool = False,  # Make re-ranking optional
    ) -> List[DocumentChunk]:
        """
        Re-rank chunks using cross-encoder for maximum relevance
        """
        if not chunks:
            return []

        # If re-ranking is disabled or we have few chunks, skip expensive re-ranking
        if not use_reranking or len(chunks) <= 5:
            logger.info("Skipping re-ranking for faster response")
            # Just return top chunks sorted by original scores
            sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in sorted_chunks[: settings.RERANK_TOP_K]]

        try:
            # Lazy-load cross-encoder
            cross_encoder = self._get_cross_encoder()
            
            if cross_encoder is None:
                logger.warning("Cross-encoder not available, using original scores")
                sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
                return [chunk for chunk, _ in sorted_chunks[: settings.RERANK_TOP_K]]

            # Prepare query-chunk pairs for cross-encoder
            pairs = [(query, chunk.text) for chunk, _ in chunks]

            logger.info(f"Re-ranking {len(pairs)} chunks with cross-encoder...")

            # Get cross-encoder scores
            cross_scores = cross_encoder.predict(pairs)

            # Combine with original scores and re-rank
            reranked = []
            for i, (chunk, original_score) in enumerate(chunks):
                combined_score = 0.7 * cross_scores[i] + 0.3 * original_score
                reranked.append((chunk, combined_score))

            # Sort by combined score
            reranked.sort(key=lambda x: x[1], reverse=True)

            logger.info("Re-ranking completed")

            # Return top chunks
            return [chunk for chunk, _ in reranked[: settings.RERANK_TOP_K]]

        except Exception as e:
            logger.warning(f"Re-ranking failed, using original scores: {str(e)}")
            # Fallback to original ranking
            sorted_chunks = sorted(chunks, key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in sorted_chunks[: settings.RERANK_TOP_K]]

    async def _generate_answer(
        self, question: str, relevant_chunks: List[DocumentChunk]
    ) -> str:
        """
        Generate concise 2-3 sentence answers using OpenAI GPT-4o-mini
        """
        if not relevant_chunks:
            return "I couldn't find any relevant information in the document to answer this question."

        # Limit chunks for faster processing (configurable)
        max_chunks = (
            settings.MAX_CHUNKS_FOR_GENERATION
            if settings.FAST_MODE
            else len(relevant_chunks)
        )
        chunks_to_use = relevant_chunks[:max_chunks]

        # Prepare context from relevant chunks
        context_parts = []
        for i, chunk in enumerate(chunks_to_use):
            # Limit chunk text length for faster processing
            chunk_text = chunk.text
            if len(chunk_text) > 600:  # Shorter chunks for concise answers
                chunk_text = chunk_text[:600] + "..."
            context_parts.append(f"[Section {i + 1}]\n{chunk_text}")

        context = "\n\n".join(context_parts)

        # Enhanced prompt for better accuracy while keeping 2-3 sentences
        prompt = f"""Based on the document sections below, provide a precise answer to the question in exactly 2-3 sentences.

DOCUMENT SECTIONS:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer in exactly 2-3 sentences like a knowledgeable human would in conversation
- Use ONLY information from the document sections above
- If the answer is not in the sections, say "The document does not contain information about [specific topic]"
- Include specific numbers, dates, and terms from the document
- Be precise, factual, and conversational in tone
- Write as if you're explaining to a colleague or friend

Answer:"""

        try:
            # Generate answer using OpenAI (configurable parameters)
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": self._get_concise_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=settings.MAX_GENERATION_TOKENS,  # Configurable
                temperature=settings.GENERATION_TEMPERATURE,  # Configurable
                timeout=15,
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Unable to process the question due to an error."

    def _get_concise_system_prompt(self) -> str:
        """System prompt optimized for concise responses"""
        return """You are a knowledgeable document analysis expert who explains complex information in a clear, human-like manner. 

REQUIREMENTS:
- Answer in exactly 2-3 sentences like a helpful colleague would in conversation
- Use specific facts, numbers, and terms from the document
- Be conversational yet professional - write as if explaining to a friend or colleague
- Include exact figures, percentages, timeframes when mentioned
- Use natural, human language while maintaining accuracy"""

    def _get_enhanced_system_prompt(self) -> str:
        """Enhanced system prompt for human-like, accurate responses"""
        return """You are an expert document analyst who provides clear, accurate, and human-like explanations. Your role is to help people understand complex documents by giving comprehensive answers in a natural, conversational tone.

CORE PRINCIPLES:
1. ALWAYS provide a complete answer using the available information
2. Write naturally, as if explaining to a colleague or friend
3. Be thorough and include all relevant details from the document
4. Use specific numbers, dates, terms, and conditions exactly as stated
5. Organize complex information in an easy-to-understand way
6. Never say "the document doesn't contain" - instead work with what's available

RESPONSE STYLE:
- Start with a direct, clear answer
- Provide supporting details and context
- Use bullet points or numbered lists for complex information
- Include examples or scenarios when mentioned in the document
- Explain technical terms in simple language while keeping the exact terminology
- Make it conversational but professional

ACCURACY REQUIREMENTS:
- Quote exact figures, percentages, and timeframes
- Use precise terminology from the document
- Include all conditions and exceptions mentioned
- Reference specific sections or clauses when relevant"""

    def _enhance_answer_quality(
        self, answer: str, question: str, chunks: List[DocumentChunk]
    ) -> str:
        """Enhance answer quality to be more human-like and comprehensive"""

        # Remove robotic prefixes
        prefixes_to_remove = [
            "Based on the document sections provided,",
            "According to the information given,",
            "The document states that",
            "From the provided context,",
            "Based on the available information,",
        ]

        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix) :].strip()

        # Ensure proper capitalization
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]

        # Add natural flow if answer seems too abrupt
        if len(answer.split(".")) == 1 and len(answer) < 100:
            # For short answers, try to add more context from chunks
            additional_context = self._extract_additional_context(
                question, chunks, answer
            )
            if additional_context:
                answer = f"{answer} {additional_context}"

        # Ensure proper ending
        if answer and not answer.endswith((".", "!", "?")):
            answer += "."

        return answer.strip()

    def _extract_additional_context(
        self, question: str, chunks: List[DocumentChunk], current_answer: str
    ) -> str:
        """Extract additional relevant context to make answers more comprehensive"""

        # Look for related information in chunks that might add value
        question_keywords = set(question.lower().split())
        additional_info = []

        for chunk in chunks[:3]:  # Check top 3 chunks
            # Find sentences with question keywords that aren't in current answer
            sentences = chunk.text.split(".")
            for sentence in sentences:
                sentence = sentence.strip()
                if (
                    len(sentence) > 20
                    and any(
                        keyword in sentence.lower() for keyword in question_keywords
                    )
                    and sentence.lower() not in current_answer.lower()
                ):
                    # Check if this adds new information
                    if len(additional_info) < 2:  # Limit additional context
                        additional_info.append(sentence)

        if additional_info:
            return "Additionally, " + ". ".join(additional_info) + "."

        return ""

    def _create_intelligent_fallback(
        self, question: str, chunks: List[DocumentChunk]
    ) -> str:
        """Create intelligent fallback answers that are still human-like"""

        if not chunks:
            return "I don't see any relevant information in the document that directly addresses this question."

        # Extract the most relevant information from chunks
        best_chunk = chunks[0]

        # Try to find the most relevant sentences
        sentences = best_chunk.text.split(".")
        question_words = set(question.lower().split())

        relevant_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and any(word in sentence.lower() for word in question_words):
                relevant_sentences.append(sentence)

        if relevant_sentences:
            # Create a natural response using the relevant sentences
            if len(relevant_sentences) == 1:
                return f"From what I can see in the document: {relevant_sentences[0]}."
            else:
                main_info = relevant_sentences[0]
                additional = ". ".join(
                    relevant_sentences[1:2]
                )  # Limit to avoid too long responses
                return f"Based on the document information: {main_info}. {additional}."
        else:
            # Use the chunk content but make it more natural
            chunk_preview = (
                best_chunk.text[:300] + "..."
                if len(best_chunk.text) > 300
                else best_chunk.text
            )
            return f"Here's what I found in the document that may be relevant: {chunk_preview}"

    async def check_vector_db_health(self) -> bool:
        """Check vector database health"""
        try:
            if self.vector_store:
                return await self.vector_store.health_check()
            return False
        except Exception:
            return False

    async def check_model_health(self) -> bool:
        """Check model health"""
        try:
            # Test OpenAI API
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_GENERATION_MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                timeout=5,
            )
            return bool(response.choices)
        except Exception:
            return False
