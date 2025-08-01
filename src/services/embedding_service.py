import logging
import asyncio
import hashlib
import pickle
from typing import List, Optional, Tuple
import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile
import aiofiles

from src.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating and managing embeddings using OpenAI's API.
    Includes caching and similarity search capabilities.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.logger = logger
        self.cache_dir = os.path.join(tempfile.gettempdir(), "embedding_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate a cache key for the text and model combination."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    async def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache if it exists."""
        cache_path = self._get_cache_path(cache_key)
        try:
            if os.path.exists(cache_path):
                async with aiofiles.open(cache_path, 'rb') as f:
                    content = await f.read()
                    embedding = pickle.loads(content)
                    self.logger.debug(f"Loaded embedding from cache: {cache_key}")
                    return embedding
        except Exception as e:
            self.logger.warning(f"Error loading from cache: {e}")
        return None
    
    async def _save_to_cache(self, cache_key: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            async with aiofiles.open(cache_path, 'wb') as f:
                content = pickle.dumps(embedding)
                await f.write(content)
                self.logger.debug(f"Saved embedding to cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {e}")
    
    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Generate embedding for a single text using OpenAI API.
        Uses caching to avoid redundant API calls.
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if model is None:
            model = settings.EMBEDDING_MODEL
        
        # Check cache first
        cache_key = self._get_cache_key(text, model)
        cached_embedding = await self._load_from_cache(cache_key)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            self.logger.debug(f"Generating embedding for text (length: {len(text)})")
            
            response = await self.client.embeddings.create(
                model=model,
                input=text.strip(),
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Cache the result
            await self._save_to_cache(cache_key, embedding)
            
            self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise
    
    async def generate_embeddings_batch(self, texts: List[str], model: str = None, batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        Uses caching and batching for efficiency.
        """
        if not texts:
            return []
        
        if model is None:
            model = settings.EMBEDDING_MODEL
        
        all_embeddings = []
        
        # Process in batches to avoid API limits and improve performance
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache for each text in batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                if not text.strip():
                    batch_embeddings.append([0.0] * settings.EMBEDDING_DIMENSIONS)
                    continue
                
                cache_key = self._get_cache_key(text, model)
                cached_embedding = await self._load_from_cache(cache_key)
                
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text.strip())
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    self.logger.debug(f"Generating {len(uncached_texts)} embeddings in batch")
                    
                    response = await self.client.embeddings.create(
                        model=model,
                        input=uncached_texts,
                        encoding_format="float"
                    )
                    
                    # Fill in the batch results and cache them
                    for k, embedding_data in enumerate(response.data):
                        embedding = embedding_data.embedding
                        batch_index = uncached_indices[k]
                        batch_embeddings[batch_index] = embedding
                        
                        # Cache the result
                        text = uncached_texts[k]
                        cache_key = self._get_cache_key(text, model)
                        await self._save_to_cache(cache_key, embedding)
                
                except Exception as e:
                    self.logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
                    # Fill remaining None values with zero vectors
                    for j in range(len(batch_embeddings)):
                        if batch_embeddings[j] is None:
                            batch_embeddings[j] = [0.0] * settings.EMBEDDING_DIMENSIONS
            
            all_embeddings.extend(batch_embeddings)
            
            self.logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        self.logger.info(f"Generated {len(all_embeddings)} embeddings total")
        return all_embeddings
    
    def calculate_similarity(self, query_embedding: List[float], document_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarity between query embedding and document embeddings.
        Returns similarity scores (higher = more similar).
        """
        if not query_embedding or not document_embeddings:
            return []
        
        # Convert to numpy arrays
        query_vec = np.array(query_embedding).reshape(1, -1)
        doc_vecs = np.array(document_embeddings)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        
        return similarities.tolist()
    
    def find_most_similar_chunks(
        self, 
        query_embedding: List[float], 
        chunk_embeddings: List[List[float]], 
        chunks: List[str], 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar chunks to a query embedding.
        Returns list of (chunk_text, similarity_score) tuples.
        """
        if len(chunk_embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        if not query_embedding or not chunk_embeddings:
            return []
        
        # Calculate similarities
        similarities = self.calculate_similarity(query_embedding, chunk_embeddings)
        
        # Get top-k most similar chunks
        chunk_scores = list(zip(chunks, similarities))
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_chunks = chunk_scores[:top_k]
        
        self.logger.debug(f"Found {len(top_chunks)} similar chunks with scores: {[score for _, score in top_chunks[:3]]}")
        
        return top_chunks
    
    async def embed_and_search(
        self, 
        query: str, 
        document_chunks: List[str], 
        top_k: int = None
    ) -> List[Tuple[str, float]]:
        """
        Complete pipeline: embed query and documents, then find most similar chunks.
        """
        if top_k is None:
            top_k = settings.MAX_CHUNKS_PER_QUERY
        
        self.logger.info(f"Embedding query and {len(document_chunks)} chunks for similarity search")
        
        # Generate embeddings concurrently
        query_task = self.generate_embedding(query)
        chunks_task = self.generate_embeddings_batch(document_chunks)
        
        query_embedding, chunk_embeddings = await asyncio.gather(query_task, chunks_task)
        
        # Find most similar chunks
        similar_chunks = self.find_most_similar_chunks(
            query_embedding, 
            chunk_embeddings, 
            document_chunks, 
            top_k
        )
        
        self.logger.info(f"Found {len(similar_chunks)} similar chunks")
        return similar_chunks

# Global instance
embedding_service = EmbeddingService()