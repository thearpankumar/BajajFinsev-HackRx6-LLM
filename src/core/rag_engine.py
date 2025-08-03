"""
Advanced RAG Engine with hybrid search and high accuracy focus
Combines dense and sparse retrieval with re-ranking for maximum accuracy
"""

import asyncio
import time
import logging
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
from src.core.document_processor import DocumentProcessor, DocumentChunk
from src.core.vector_store import VectorStore
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
        self.document_processor = DocumentProcessor()
        self.vector_store = None
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

    async def initialize(self):
        """Initialize the RAG engine components"""
        logger.info("Initializing RAG Engine...")

        # Initialize vector store
        self.vector_store = VectorStore()
        await self.vector_store.initialize()

        # Don't initialize cross-encoder here - it's slow and optional
        # It will be lazy-loaded only if re-ranking is needed
        logger.info(
            "RAG Engine initialized successfully! (Cross-encoder will be loaded on demand)"
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

        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(settings.QUESTION_BATCH_SIZE)

        async def process_single_question_with_semaphore(
            question: str, index: int
        ) -> Tuple[int, str]:
            """Process a single question with semaphore control"""
            async with semaphore:
                try:
                    logger.info(
                        f"Processing question {index + 1}/{len(questions_to_process)}: {question[:50]}..."
                    )
                    answer = await self._answer_question(question, chunks, document_url)
                    logger.info(f"Completed question {index + 1}")
                    return (index, answer)
                except Exception as e:
                    logger.error(f"Error processing question {index + 1}: {str(e)}")
                    return (index, f"Error processing question: {str(e)}")

        # Create tasks for all questions
        tasks = [
            process_single_question_with_semaphore(question, i)
            for i, question in enumerate(questions_to_process)
        ]

        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
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
        """Get document from cache or process it"""
        if document_url in self.document_cache:
            self.stats["cache_hits"] += 1
            return self.document_cache[document_url]

        # Process document
        chunks, metadata = await self.document_processor.process_document(document_url)

        # Cache results
        self.document_cache[document_url] = (chunks, metadata)

        return chunks, metadata

    async def _index_chunks(self, chunks: List[DocumentChunk], document_url: str):
        """Index chunks in both vector store and BM25"""
        if document_url in self.chunk_cache:
            logger.info(f"Document {document_url} already indexed, skipping...")
            return  # Already indexed

        logger.info(f"Indexing {len(chunks)} chunks for document: {document_url}")

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

        # Check if vector store already has data for this document
        try:
            existing_stats = await self.vector_store.get_stats()
            if existing_stats.get("total_vectors", 0) > 0:
                logger.info(f"Vector store already has {existing_stats['total_vectors']} vectors, checking if document exists...")
                
                # Try a test search to see if the table is working
                test_results = await self.vector_store.similarity_search("test query", k=1)
                if test_results:
                    logger.info("Vector store is working, document may already be indexed")
                    # Still proceed with BM25 indexing for this session
                    tokenized_texts = [self._tokenize_for_bm25(text) for text in texts]
                    self.bm25_index = BM25Okapi(tokenized_texts)
                    
                    # Cache chunk information
                    self.chunk_cache[document_url] = {
                        "chunks": chunks,
                        "texts": texts,
                        "tokenized_texts": tokenized_texts,
                        "metadatas": metadatas,
                    }
                    return
        except Exception as e:
            logger.warning(f"Error checking existing vectors: {str(e)}")

        # Index in vector store
        try:
            logger.info("Adding texts to vector store...")
            await self.vector_store.add_texts(texts, metadatas)
            logger.info("Successfully added texts to vector store")
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {str(e)}")
            # Continue with BM25 even if vector store fails
            pass

        # Create BM25 index
        logger.info("Creating BM25 index...")
        tokenized_texts = [self._tokenize_for_bm25(text) for text in texts]
        self.bm25_index = BM25Okapi(tokenized_texts)

        # Cache chunk information
        self.chunk_cache[document_url] = {
            "chunks": chunks,
            "texts": texts,
            "tokenized_texts": tokenized_texts,
        }

        logger.info(f"Indexed {len(chunks)} chunks for {document_url}")

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
        """
        retrieval_start = time.time()

        # Step 1: Hybrid Retrieval
        relevant_chunks = await self._hybrid_retrieval(question, document_url)

        retrieval_time = time.time() - retrieval_start

        # Step 2: Re-ranking (disabled by default for speed)
        reranked_chunks = await self._rerank_chunks(
            question, relevant_chunks, use_reranking=False
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

    async def _hybrid_retrieval(
        self, query: str, document_url: str
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Enhanced hybrid retrieval combining dense (vector) and sparse (BM25) search
        Optimized for speed while maintaining good accuracy
        """
        # In fast mode, use simpler query processing
        if settings.FAST_MODE:
            search_query = query  # Skip expensive query expansion
        else:
            search_query = self._expand_query(query)

        # Dense retrieval using vector similarity
        dense_results = await self.vector_store.similarity_search(
            search_query,
            k=settings.TOP_K_RETRIEVAL,  # Use configured limit
        )

        # Sparse retrieval using BM25 (only if we have BM25 index)
        sparse_results = []
        if self.bm25_index and document_url in self.chunk_cache:
            # Use original query for BM25 (faster than expanded)
            original_tokens = self._tokenize_for_bm25(query)
            original_scores = self.bm25_index.get_scores(original_tokens)

            chunks = self.chunk_cache[document_url]["chunks"]

            # Get top BM25 results
            top_indices = np.argsort(original_scores)[::-1][: settings.TOP_K_RETRIEVAL]

            for idx in top_indices:
                if original_scores[idx] > 0:  # Only include positive scores
                    sparse_results.append((chunks[idx], float(original_scores[idx])))

        # Use simpler RRF in fast mode
        if settings.FAST_MODE:
            combined_results = self._simple_reciprocal_rank_fusion(
                dense_results, sparse_results
            )
        else:
            combined_results = self._enhanced_reciprocal_rank_fusion(
                dense_results, sparse_results, query
            )

        return combined_results[: settings.TOP_K_RETRIEVAL]

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

        # Limit chunks for faster processing
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

        # Concise prompt for short answers
        prompt = f"""Based on the document sections below, provide a direct and concise answer to the question in 2-3 sentences maximum.

DOCUMENT SECTIONS:
{context}

QUESTION: {question}

Provide a clear, factual answer in 2-3 sentences using specific information from the document:"""

        try:
            # Generate answer using OpenAI
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": self._get_concise_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,  # Limit tokens for concise responses
                temperature=0.1,
                timeout=15,
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Unable to process the question due to an error."

    def _get_concise_system_prompt(self) -> str:
        """System prompt optimized for concise responses"""
        return """You are a document analyst that provides precise, factual answers in 2-3 sentences maximum. 

REQUIREMENTS:
- Answer in exactly 2-3 sentences
- Use specific facts, numbers, and terms from the document
- Be direct and factual
- No introductory phrases or explanations
- Include exact figures, percentages, timeframes when mentioned"""

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

    async def cleanup(self):
        """Cleanup resources"""
        if self.vector_store:
            await self.vector_store.close()

        # Clear caches
        self.document_cache.clear()
        self.chunk_cache.clear()

        logger.info("RAG Engine cleanup completed")
