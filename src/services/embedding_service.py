import logging
import asyncio
import hashlib
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from src.core.config import settings
from src.services.pinecone_service import get_pinecone_service

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings using OpenAI and interacting with Pinecone.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.logger = logger
        self.pinecone_service = get_pinecone_service()
        # Ensure the index exists before performing operations
        self.pinecone_service.create_index_if_not_exists(
            index_name=settings.PINECONE_INDEX_NAME,
            dimension=settings.EMBEDDING_DIMENSIONS
        )

    async def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """
        Generate embedding for a single text using OpenAI API.
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        if model is None:
            model = settings.EMBEDDING_MODEL
        
        try:
            self.logger.debug(f"Generating embedding for text (length: {len(text)})")
            response = await self.client.embeddings.create(
                model=model,
                input=text.strip(),
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise

    async def generate_embeddings_batch(self, texts: List[str], model: str = None, batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        """
        if not texts:
            return []
        
        if model is None:
            model = settings.EMBEDDING_MODEL
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [t.strip() for t in texts[i:i + batch_size]]
            try:
                response = await self.client.embeddings.create(
                    model=model,
                    input=batch_texts,
                    encoding_format="float"
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                self.logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
                # Add placeholder for failed batch
                all_embeddings.extend([[0.0] * settings.EMBEDDING_DIMENSIONS] * len(batch_texts))
        
        self.logger.info(f"Generated {len(all_embeddings)} embeddings total")
        return all_embeddings

    def _generate_chunk_id(self, document_url: str, chunk_index: int) -> str:
        """Generate a unique and deterministic ID for a chunk."""
        return hashlib.sha256(f"{document_url}-{chunk_index}".encode()).hexdigest()

    async def embed_and_upsert_chunks(self, document_url: str, document_chunks: List[Dict[str, Any]]):
        """
        Generates embeddings for document chunks and upserts them into Pinecone.
        Expects chunks as dictionaries with 'text' and 'metadata' keys.
        """
        self.logger.info(f"Starting embedding and upsert for {len(document_chunks)} chunks from {document_url}")
        
        chunk_texts = [chunk['text'] for chunk in document_chunks]
        embeddings = await self.generate_embeddings_batch(chunk_texts)
        
        # Prepare chunks with metadata for upserting
        chunks_to_upsert = []
        for i, chunk_data in enumerate(document_chunks):
            chunk_id = self._generate_chunk_id(document_url, i)
            
            # Combine existing metadata with the document URL and full text
            metadata = chunk_data['metadata']
            metadata['document_url'] = document_url
            metadata['text'] = chunk_data['text']

            chunks_to_upsert.append({"id": chunk_id, "metadata": metadata})

        # Upsert to Pinecone
        self.pinecone_service.upsert_chunks(
            index_name=settings.PINECONE_INDEX_NAME,
            vectors=embeddings,
            chunks=chunks_to_upsert
        )
        self.logger.info(f"Completed upsert for {len(document_chunks)} chunks.")

    async def embed_and_search(self, query: str, document_url: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Embeds a query and searches for the most similar chunks in Pinecone,
        filtered by the document URL.
        """
        if top_k is None:
            top_k = settings.MAX_CHUNKS_PER_QUERY
        
        self.logger.info(f"Embedding query and searching for top {top_k} chunks for document: {document_url}")
        
        query_embedding = await self.generate_embedding(query)
        
        # Use metadata filtering to search only within the specified document
        filter_dict = {"document_url": {"$eq": document_url}}
        
        search_results = self.pinecone_service.query(
            index_name=settings.PINECONE_INDEX_NAME,
            vector=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )
        
        # The result from pinecone_service.query is already a list of dictionaries
        self.logger.info(f"Found {len(search_results)} similar chunks in Pinecone.")
        return search_results

# Global instance
embedding_service = EmbeddingService()
