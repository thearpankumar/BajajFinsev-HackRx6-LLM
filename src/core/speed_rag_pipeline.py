"""
Speed-Optimized RAG Pipeline
High-performance RAG using GPU Cache Manager for maximum throughput
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib

import requests
import numpy as np
from groq import Groq

from src.core.config import config
from src.core.gpu_cache_manager import gpu_cache_manager

logger = logging.getLogger(__name__)

@dataclass
class SpeedDocument:
    """Lightweight document representation"""
    id: str
    url: str
    text: str
    chunks: List[str]
    chunk_embeddings: Optional[np.ndarray] = None

@dataclass
class SpeedQueryResult:
    """Fast query result"""
    query: str
    answer: str
    relevant_chunks: List[str]
    similarity_scores: List[float]
    processing_time: float
    cache_used: bool

class SpeedRAGPipeline:
    """
    Ultra-fast RAG pipeline optimized for speed and GPU utilization
    - Single embedding model instance
    - GPU-accelerated similarity search
    - Multi-level caching
    - Minimal abstraction layers
    """
    
    def __init__(self):
        self.documents: Dict[str, SpeedDocument] = {}
        self.groq_client: Optional[Groq] = None
        self.is_initialized = False
        
        # Performance tracking
        self.total_documents = 0
        self.total_queries = 0
        self.total_chunks = 0
        self.average_query_time = 0.0
        
        logger.info("Speed RAG Pipeline initialized")
    
    async def initialize(self) -> bool:
        """Initialize the speed-optimized RAG pipeline"""
        try:
            logger.info("🚀 Initializing Speed RAG Pipeline...")
            
            # Initialize GPU cache manager
            if not await gpu_cache_manager.initialize():
                logger.error("Failed to initialize GPU cache manager")
                return False
            
            # Apply speed optimizations
            gpu_cache_manager.optimize_for_speed()
            
            # Initialize Groq client
            if config.groq_api_key:
                self.groq_client = Groq(api_key=config.groq_api_key)
                logger.info("✅ Groq client initialized")
            else:
                logger.warning("⚠️ No Groq API key - using fallback responses")
            
            self.is_initialized = True
            logger.info("✅ Speed RAG Pipeline fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"❌ Speed RAG Pipeline initialization failed: {e}")
            return False
    
    async def add_document_url(self, url: str, progress_callback=None) -> bool:
        """Add document from URL with speed optimizations"""
        try:
            if progress_callback:
                await progress_callback("Downloading document...", 10)
            
            # Download document
            logger.info(f"📥 Processing document: {url}")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Extract text (simplified for speed)
            text = response.text
            if len(text) < 100:
                logger.warning(f"Document too short: {len(text)} chars")
                return False
            
            if progress_callback:
                await progress_callback("Chunking document...", 30)
            
            # Fast chunking (optimized for speed)
            chunks = self._fast_chunk(text)
            
            if progress_callback:
                await progress_callback("Generating embeddings...", 50)
            
            # Generate embeddings using GPU cache manager
            embeddings = await gpu_cache_manager.get_embeddings(chunks, use_cache=True)
            
            if progress_callback:
                await progress_callback("Adding to vector index...", 80)
            
            # Add to FAISS index
            await gpu_cache_manager.add_to_vector_index(embeddings)
            
            # Create document
            doc_id = hashlib.md5(url.encode()).hexdigest()
            document = SpeedDocument(
                id=doc_id,
                url=url,
                text=text,
                chunks=chunks,
                chunk_embeddings=embeddings
            )
            
            self.documents[doc_id] = document
            self.total_documents += 1
            self.total_chunks += len(chunks)
            
            if progress_callback:
                await progress_callback("Document processing complete", 100)
            
            logger.info(f"✅ Document processed: {len(chunks)} chunks in {len(text)} chars")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            return False
    
    def _fast_chunk(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Fast chunking optimized for speed"""
        chunks = []
        words = text.split()
        
        # Estimate words per chunk
        avg_word_length = 5  # Average word length
        words_per_chunk = chunk_size // avg_word_length
        overlap_words = overlap // avg_word_length
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Skip tiny chunks
                chunks.append(chunk_text.strip())
        
        return chunks
    
    async def query_fast(self, question: str, max_results: int = 5) -> SpeedQueryResult:
        """Ultra-fast query processing with GPU acceleration"""
        start_time = time.time()
        cache_used = False
        
        try:
            if not self.is_initialized:
                raise RuntimeError("Pipeline not initialized")
            
            # Generate query embedding (cached automatically)
            query_embeddings = await gpu_cache_manager.get_embeddings([question], use_cache=True)
            query_embedding = query_embeddings[0]
            
            # Fast similarity search using GPU FAISS
            scores, indices = await gpu_cache_manager.search_similar(query_embedding, k=max_results)
            
            # Get relevant chunks
            relevant_chunks = []
            similarity_scores = []
            
            if len(indices) > 0:
                chunk_idx = 0
                for doc in self.documents.values():
                    for chunk in doc.chunks:
                        if chunk_idx in indices:
                            relevant_chunks.append(chunk)
                            # Get corresponding score
                            score_idx = np.where(indices == chunk_idx)[0]
                            if len(score_idx) > 0:
                                similarity_scores.append(float(scores[score_idx[0]]))
                        chunk_idx += 1
            
            # Generate answer
            answer = await self._generate_fast_answer(question, relevant_chunks)
            
            processing_time = time.time() - start_time
            self.total_queries += 1
            self._update_average_query_time(processing_time)
            
            return SpeedQueryResult(
                query=question,
                answer=answer,
                relevant_chunks=relevant_chunks,
                similarity_scores=similarity_scores,
                processing_time=processing_time,
                cache_used=cache_used
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Query processing failed: {e}")
            
            return SpeedQueryResult(
                query=question,
                answer=f"Sorry, I encountered an error: {str(e)}",
                relevant_chunks=[],
                similarity_scores=[],
                processing_time=processing_time,
                cache_used=False
            )
    
    async def _generate_fast_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate answer with speed optimizations"""
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."
        
        if not self.groq_client:
            # Fast fallback - return best chunk
            return f"Based on the available information: {context_chunks[0][:400]}..."
        
        try:
            # Build concise context
            context = "\n\n".join(context_chunks[:3])  # Limit context for speed
            
            prompt = f"""Answer this question concisely based on the context:

Context: {context[:2000]}

Question: {question}

Answer:"""
            
            # Use fastest Groq model
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Fastest model
                max_tokens=200,  # Limit for speed
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.warning(f"Groq API failed: {e}")
            return f"Based on the information: {context_chunks[0][:300]}..."
    
    def _update_average_query_time(self, new_time: float):
        """Update running average query time"""
        if self.total_queries == 1:
            self.average_query_time = new_time
        else:
            self.average_query_time = ((self.average_query_time * (self.total_queries - 1)) + new_time) / self.total_queries
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        gpu_stats = gpu_cache_manager.get_cache_stats()
        
        return {
            # Pipeline stats
            'total_documents': self.total_documents,
            'total_chunks': self.total_chunks,
            'total_queries': self.total_queries,
            'average_query_time': self.average_query_time,
            'chunks_per_document': self.total_chunks / max(1, self.total_documents),
            
            # GPU & Cache stats
            **gpu_stats,
            
            # Speed metrics
            'embeddings_per_second': gpu_stats['embeddings_computed'] / max(0.001, gpu_stats['total_gpu_time']),
            'queries_per_minute': (self.total_queries / max(0.001, self.average_query_time)) * 60,
            
            # System info
            'is_initialized': self.is_initialized,
            'pipeline_type': 'speed_optimized'
        }
    
    async def preload_for_speed(self):
        """Preload common operations for maximum speed"""
        logger.info("🔄 Preloading for speed optimization...")
        
        # Preload common question patterns
        common_questions = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key findings?",
            "Explain the important details"
        ]
        
        await gpu_cache_manager.preload_common_embeddings(common_questions)
        logger.info("✅ Speed preloading complete")
    
    def clear_all_caches(self):
        """Clear all caches to free memory"""
        gpu_cache_manager.clear_cache(clear_gpu=True, clear_cpu=True)
        logger.info("🗑️ All caches cleared")


# Global Speed RAG Pipeline instance
speed_rag_pipeline = SpeedRAGPipeline()