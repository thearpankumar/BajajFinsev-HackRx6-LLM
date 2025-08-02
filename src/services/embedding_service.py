import logging
import asyncio
import hashlib
from typing import List, Optional, Dict, Any
from openai import AsyncOpenAI

from src.core.config import settings
from src.services.lancedb_service import get_lancedb_service

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Service for generating embeddings using OpenAI and interacting with LanceDB.
    """
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.logger = logger
        self.lancedb_service = get_lancedb_service()
        # Ensure the table exists before performing operations
        self.lancedb_service.create_table_if_not_exists(
            table_name=settings.LANCEDB_TABLE_NAME
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
            response = await self.client.embeddings.create(
                model=model, input=text.strip(), encoding_format="float"
            )
            return response.data[0].embedding
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
                    model=model, input=batch_texts, encoding_format="float"
                )
                all_embeddings.extend([item.embedding for item in response.data])
            except Exception as e:
                self.logger.error(f"Error generating batch embeddings: {e}", exc_info=True)
                all_embeddings.extend([[0.0] * settings.EMBEDDING_DIMENSIONS] * len(batch_texts))
        
        return all_embeddings

    async def embed_and_upsert_chunks(self, document_url: str, document_chunks: List[Dict[str, Any]]):
        """
        Generates embeddings for document chunks and upserts them into LanceDB.
        """
        self.logger.info(f"Starting embedding and upsert for {len(document_chunks)} chunks from {document_url}")
        
        chunk_texts = [chunk['text'] for chunk in document_chunks]
        embeddings = await self.generate_embeddings_batch(chunk_texts)
        
        # Add document_url to each chunk's metadata
        for chunk in document_chunks:
            chunk['metadata']['document_url'] = document_url
            chunk['metadata']['text'] = chunk['text']

        self.lancedb_service.upsert_chunks(
            table_name=settings.LANCEDB_TABLE_NAME,
            vectors=embeddings,
            chunks=document_chunks
        )
        self.logger.info(f"Completed upsert for {len(document_chunks)} chunks.")

    async def embed_and_search(self, query: str, document_url: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Embeds a query and searches for the most similar chunks in LanceDB,
        filtered by the document URL.
        """
        if top_k is None:
            top_k = settings.MAX_CHUNKS_PER_QUERY
        
        self.logger.info(f"Embedding query and searching in LanceDB for top {top_k} chunks for document: {document_url}")
        
        query_embedding = await self.generate_embedding(query)
        
        # LanceDB uses a SQL-like WHERE clause for filtering
        filter_str = f"document_url = '{document_url}'"
        
        search_results = self.lancedb_service.query(
            table_name=settings.LANCEDB_TABLE_NAME,
            vector=query_embedding,
            top_k=top_k,
            filter_str=filter_str
        )
        
        # The result from lancedb_service is already a list of dictionaries.
        # We need to reshape it slightly to match the expected format downstream.
        formatted_results = []
        for res in search_results:
            formatted_results.append({
                "metadata": {
                    "text": res.get("text", ""),
                    "page_number": res.get("page_number", -1),
                    "document_url": res.get("document_url", "")
                },
                "score": 1 - res.get("score", 1.0) # LanceDB distance is 0-2, convert to similarity score 1-0
            })

        self.logger.info(f"Found {len(formatted_results)} similar chunks in LanceDB.")
        return formatted_results

# Global instance
embedding_service = EmbeddingService()