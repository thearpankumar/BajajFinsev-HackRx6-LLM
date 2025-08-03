"""
Vector store implementation using LanceDB for high-performance similarity search
Fixed for Docker container compatibility with proper vector schema
"""

import logging
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from openai import AsyncOpenAI
import lancedb
import pyarrow as pa
from pathlib import Path

from src.core.config import settings
from src.core.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    High-performance vector store using LanceDB with proper vector schema
    """

    def __init__(self):
        self.db = None
        self.table = None
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        self.dimension = settings.VECTOR_DIMENSION

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def initialize(self):
        """Initialize the vector database"""
        try:
            logger.info("Starting vector store initialization...")
            
            # Create database directory if it doesn't exist
            db_path = Path(settings.VECTOR_DB_PATH)
            db_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Database directory created/verified: {db_path}")

            # Connect to LanceDB
            logger.info("Connecting to LanceDB...")
            self.db = lancedb.connect(str(db_path))
            logger.info("LanceDB connection established")

            # Create or connect to table
            logger.info("Creating/connecting to vector table...")
            await self._create_or_connect_table()

            logger.info(f"Vector store initialized successfully at {db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _create_or_connect_table(self):
        """Create or connect to the vectors table with proper schema"""
        table_name = "document_vectors"

        try:
            # Try to open existing table
            self.table = self.db.open_table(table_name)
            logger.info(f"Connected to existing table: {table_name}")
            
            # Verify the table has the correct schema
            schema = self.table.schema
            logger.info(f"Table schema: {schema}")

        except Exception as e:
            logger.info(f"Table doesn't exist or has issues, creating new one: {str(e)}")
            
            try:
                # Create with proper schema using PyArrow
                logger.info("Creating table with proper vector schema...")
                
                # Define schema with fixed-size list for vectors
                schema = pa.schema([
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.dimension)),  # Fixed-size list
                    pa.field("metadata", pa.string()),
                    pa.field("chunk_id", pa.string()),
                    pa.field("page_num", pa.int32()),
                    pa.field("word_count", pa.int32()),
                    pa.field("source_url", pa.string()),
                ])
                
                # Create empty table with schema
                empty_table = pa.table([], schema=schema)
                self.table = self.db.create_table(table_name, empty_table)
                
                logger.info(f"Created new table with proper schema: {self.table.schema}")
                
            except Exception as creation_error:
                logger.error(f"Failed to create vector table: {creation_error}")
                raise RuntimeError(f"Failed to create vector table: {creation_error}")

    async def add_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add texts to the vector store with embeddings

        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries

        Returns:
            List of document IDs
        """
        if not texts:
            return []

        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = await self._get_embeddings(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Prepare data for insertion
            data = []
            doc_ids = []

            for i, (text, embedding, metadata) in enumerate(
                zip(texts, embeddings, metadatas)
            ):
                doc_id = f"doc_{len(data)}_{hash(text) % 1000000}"
                doc_ids.append(doc_id)

                # Validate embedding
                if not isinstance(embedding, (list, np.ndarray)):
                    raise ValueError(f"Invalid embedding type: {type(embedding)}")

                if len(embedding) != self.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}"
                    )

                # Ensure embedding is a proper numpy array with correct dtype and shape
                embedding_array = np.array(embedding, dtype=np.float32)
                if embedding_array.shape != (self.dimension,):
                    embedding_array = embedding_array.reshape(self.dimension)

                data.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "vector": embedding_array.tolist(),  # Convert to list for PyArrow
                        "metadata": json.dumps(metadata),  # Convert to JSON string
                        "chunk_id": metadata.get("chunk_id", ""),
                        "page_num": int(metadata.get("page_num", 0)),
                        "word_count": int(metadata.get("word_count", 0)),
                        "source_url": metadata.get("source_url", ""),
                    }
                )

            logger.info(f"Prepared {len(data)} records for insertion")

            # Create PyArrow table with proper schema
            ids = [item["id"] for item in data]
            texts_list = [item["text"] for item in data]
            vectors = [item["vector"] for item in data]
            metadatas_list = [item["metadata"] for item in data]
            chunk_ids = [item["chunk_id"] for item in data]
            page_nums = [item["page_num"] for item in data]
            word_counts = [item["word_count"] for item in data]
            source_urls = [item["source_url"] for item in data]

            logger.info("Creating PyArrow table with proper vector format...")

            # Create PyArrow table with explicit schema
            pa_table = pa.table({
                "id": pa.array(ids, type=pa.string()),
                "text": pa.array(texts_list, type=pa.string()),
                "vector": pa.array(vectors, type=pa.list_(pa.float32(), self.dimension)),
                "metadata": pa.array(metadatas_list, type=pa.string()),
                "chunk_id": pa.array(chunk_ids, type=pa.string()),
                "page_num": pa.array(page_nums, type=pa.int32()),
                "word_count": pa.array(word_counts, type=pa.int32()),
                "source_url": pa.array(source_urls, type=pa.string()),
            })

            logger.info("Adding data to LanceDB...")
            # Add to LanceDB
            self.table.add(pa_table)

            logger.info(f"Successfully added {len(texts)} texts to vector store")

            return doc_ids

        except Exception as e:
            logger.error(f"Error adding texts to vector store: {str(e)}")
            raise

    async def similarity_search(
        self, query: str, k: int = 10, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform similarity search

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filters

        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        try:
            # Check if table exists and has data
            if not self.table:
                logger.warning("Vector table not initialized")
                return []
                
            # Check if table has any data
            try:
                row_count = self.table.count_rows()
                if row_count == 0:
                    logger.warning("Vector table is empty, no data to search")
                    return []
            except Exception as e:
                logger.warning(f"Error checking table row count: {str(e)}")
                return []

            # Get query embedding
            query_embedding = await self._get_embeddings([query])
            query_vector = query_embedding[0]
            
            # Ensure query vector is proper numpy array
            query_vector = np.array(query_vector, dtype=np.float32)

            # Perform vector search with explicit vector column name
            logger.info(f"Performing vector search with query vector shape: {query_vector.shape}")
            results = self.table.search(query_vector, vector_column_name="vector").limit(k).to_pandas()

            # Convert results to DocumentChunk objects
            chunks_with_scores = []

            for _, row in results.iterrows():
                # Parse metadata
                try:
                    metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Create DocumentChunk
                chunk = DocumentChunk(
                    text=row["text"],
                    page_num=row["page_num"],
                    chunk_id=row["chunk_id"],
                    metadata=metadata,
                )

                # Get similarity score (LanceDB returns distance, convert to similarity)
                distance = row.get("_distance", 1.0)
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity

                chunks_with_scores.append((chunk, similarity))

            logger.info(f"Found {len(chunks_with_scores)} similar chunks for query")

            return chunks_with_scores

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            # Batch texts for efficiency (OpenAI has limits)
            batch_size = 100
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Get embeddings from OpenAI
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model, input=batch
                )

                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

            logger.info(f"Generated embeddings for {len(texts)} texts")

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def delete_by_source(self, source_url: str):
        """Delete all vectors from a specific source"""
        try:
            # LanceDB doesn't have direct delete by filter yet
            # This is a placeholder for future implementation
            logger.info(f"Delete by source not yet implemented for {source_url}")

        except Exception as e:
            logger.error(f"Error deleting by source: {str(e)}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            if not self.table:
                return {"error": "Table not initialized"}

            # Get table stats
            count = self.table.count_rows()

            return {
                "total_vectors": count,
                "dimension": self.dimension,
                "embedding_model": self.embedding_model,
            }

        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """Check if vector store is healthy"""
        try:
            if not self.table:
                logger.warning("Vector table not initialized for health check")
                return False

            # Try to get count - this should work even with empty table
            row_count = self.table.count_rows()
            logger.info(f"Vector store health check passed - {row_count} rows in table")
            return True

        except Exception as e:
            logger.error(f"Vector store health check failed: {str(e)}")
            return False

    async def close(self):
        """Close the vector store connection"""
        try:
            # LanceDB connections are automatically managed
            self.db = None
            self.table = None
            logger.info("Vector store connection closed")

        except Exception as e:
            logger.error(f"Error closing vector store: {str(e)}")
