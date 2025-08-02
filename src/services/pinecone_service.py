import logging
import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from src.core.config import settings

logger = logging.getLogger(__name__)

class PineconeService:
    """
    Service for interacting with the Pinecone vector database.
    Handles index creation, upserting, and querying.
    """
    def __init__(self, api_key: str, environment: str):
        if not api_key or not environment:
            raise ValueError("Pinecone API key and environment must be set.")
        self.pinecone = Pinecone(api_key=api_key)
        self.environment = environment
        self.logger = logger

    def _get_index(self, index_name: str):
        """Get a Pinecone index object."""
        return self.pinecone.Index(index_name)

    def create_index_if_not_exists(self, index_name: str, dimension: int):
        """
        Create a new Pinecone index if it doesn't already exist.
        Uses a serverless spec for cost-effective scaling.
        """
        if index_name not in self.pinecone.list_indexes().names():
            self.logger.info(f"Creating new Pinecone index: {index_name}")
            try:
                self.pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                self.logger.info(f"Index '{index_name}' created successfully.")
            except Exception as e:
                self.logger.error(f"Failed to create Pinecone index: {e}", exc_info=True)
                raise
        else:
            self.logger.info(f"Pinecone index '{index_name}' already exists.")

    def upsert_chunks(self, index_name: str, vectors: List[List[float]], chunks: List[Dict[str, Any]]) -> int:
        """
        Upsert document chunks and their embeddings into the Pinecone index.
        
        Args:
            index_name: The name of the index.
            vectors: A list of vector embeddings.
            chunks: A list of dictionaries, each containing chunk text and metadata.
        
        Returns:
            The number of vectors successfully upserted.
        """
        if not vectors or not chunks:
            self.logger.warning("Upsert called with empty vectors or chunks.")
            return 0

        index = self._get_index(index_name)
        
        # Prepare data in the format Pinecone expects
        vectors_to_upsert = []
        for i, (vector, chunk_data) in enumerate(zip(vectors, chunks)):
            vectors_to_upsert.append({
                "id": chunk_data['id'],  # Assuming each chunk has a unique ID
                "values": vector,
                "metadata": chunk_data['metadata']
            })

        try:
            # Upsert in batches for efficiency
            upsert_response = index.upsert(vectors=vectors_to_upsert, batch_size=100)
            self.logger.info(f"Successfully upserted {upsert_response.upserted_count} vectors into '{index_name}'.")
            return upsert_response.upserted_count
        except Exception as e:
            self.logger.error(f"Error upserting data to Pinecone: {e}", exc_info=True)
            raise

    def query(self, index_name: str, vector: List[float], top_k: int, filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        Query the Pinecone index to find the most similar vectors.

        Args:
            index_name: The name of the index.
            vector: The query vector.
            top_k: The number of top results to return.
            filter_dict: An optional dictionary for metadata filtering.

        Returns:
            A list of query results.
        """
        index = self._get_index(index_name)
        try:
            query_response = index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            self.logger.info(f"Query returned {len(query_response.matches)} results from '{index_name}'.")
            return query_response.matches
        except Exception as e:
            self.logger.error(f"Error querying Pinecone: {e}", exc_info=True)
            raise

    def delete_index(self, index_name: str):
        """Deletes a Pinecone index."""
        if index_name in self.pinecone.list_indexes().names():
            self.logger.info(f"Deleting Pinecone index: {index_name}")
            self.pinecone.delete_index(index_name)
            self.logger.info(f"Index '{index_name}' deleted.")
        else:
            self.logger.warning(f"Attempted to delete non-existent index: {index_name}")

# Global instance of the service
# It's better to instantiate this where it's needed to ensure config is loaded.
pinecone_service: Optional[PineconeService] = None

def get_pinecone_service():
    """Factory function to get a singleton instance of the Pinecone service."""
    global pinecone_service
    if pinecone_service is None:
        pinecone_service = PineconeService(
            api_key=settings.PINECONE_API_KEY,
            environment=settings.PINECONE_ENVIRONMENT
        )
    return pinecone_service
