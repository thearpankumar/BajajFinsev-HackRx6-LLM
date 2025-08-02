import logging
import lancedb
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import numpy as np
from lancedb.pydantic import pydantic_to_schema, Vector

from src.core.config import settings

logger = logging.getLogger(__name__)

class LanceDBSchema(BaseModel):
    vector: Vector(settings.EMBEDDING_DIMENSIONS)
    text: str
    document_url: str
    page_number: int
    chunk_index: int

class LanceDBService:
    """
    Service for interacting with the LanceDB local vector database.
    """
    def __init__(self, uri: str):
        self.db = lancedb.connect(uri)
        self.logger = logger

    def create_table_if_not_exists(self, table_name: str):
        """
        Create a new LanceDB table with an explicit schema if it doesn't already exist.
        """
        if table_name in self.db.table_names():
            self.logger.info(f"LanceDB table '{table_name}' already exists. Deleting and recreating.")
            self.db.drop_table(table_name)

        self.logger.info(f"Creating new LanceDB table: {table_name}")
        try:
            # Convert the Pydantic model to a PyArrow schema for table creation.
            schema = pydantic_to_schema(LanceDBSchema)
            self.db.create_table(table_name, schema=schema)
            self.logger.info(f"Table '{table_name}' created successfully.")
        except Exception as e:
            self.logger.error(f"Failed to create LanceDB table: {e}", exc_info=True)
            raise

    def upsert_chunks(self, table_name: str, vectors: List[List[float]], chunks: List[Dict[str, Any]]):
        """
        Upsert document chunks and their embeddings into the LanceDB table.
        """
        if not vectors or not chunks:
            self.logger.warning("Upsert called with empty vectors or chunks.")
            return

        tbl = self.db.open_table(table_name)
        
        data_to_upsert = []
        for i, (vector, chunk_data) in enumerate(zip(vectors, chunks)):
            data = {
                "vector": vector,
                "text": chunk_data['metadata'].get('text', ''),
                "document_url": chunk_data['metadata'].get('document_url', ''),
                "page_number": chunk_data['metadata'].get('page_number', -1),
                "chunk_index": i
            }
            data_to_upsert.append(data)

        try:
            tbl.add(data_to_upsert)
            self.logger.info(f"Successfully upserted {len(data_to_upsert)} vectors into '{table_name}'.")
        except Exception as e:
            self.logger.error(f"Error upserting data to LanceDB: {e}", exc_info=True)
            raise

    def query(self, table_name: str, vector: List[float], top_k: int, filter_str: Optional[str] = None) -> List[Dict]:
        """
        Query the LanceDB table to find the most similar vectors.
        """
        tbl = self.db.open_table(table_name)
        try:
            # Explicitly specify the vector column name in the search query
            query_builder = tbl.search(vector, vector_column_name="vector").limit(top_k)
            if filter_str:
                query_builder = query_builder.where(filter_str, prefilter=True)
            
            results = query_builder.to_df().to_dict('records')
            
            # Add a 'score' field to mimic the Pinecone result structure for easier integration
            for res in results:
                res['score'] = res.pop('_distance')

            self.logger.info(f"Query returned {len(results)} results from '{table_name}'.")
            return results
        except Exception as e:
            self.logger.error(f"Error querying LanceDB: {e}", exc_info=True)
            return []

# Global instance and factory function
lancedb_service: Optional[LanceDBService] = None

def get_lancedb_service():
    """Factory function to get a singleton instance of the LanceDB service."""
    global lancedb_service
    if lancedb_service is None:
        lancedb_service = LanceDBService(uri=settings.LANCEDB_PATH)
    return lancedb_service
