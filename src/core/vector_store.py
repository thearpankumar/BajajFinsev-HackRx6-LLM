"""
Vector store implementation using LanceDB for high-performance similarity search
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

# Initialize OpenAI client after importing settings
# Note: Client will be initialized in the VectorStore.__init__ method

logger = logging.getLogger(__name__)


class VectorStore:
    """
    High-performance vector store using LanceDB
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
            # Create database directory if it doesn't exist
            db_path = Path(settings.VECTOR_DB_PATH)
            db_path.mkdir(parents=True, exist_ok=True)

            # Connect to LanceDB
            self.db = lancedb.connect(str(db_path))

            # Create or connect to table
            await self._create_or_connect_table()

            logger.info(f"Vector store initialized at {db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    async def _create_or_connect_table(self):
        """Create or connect to the vectors table"""
        table_name = "document_vectors"

        try:
            # Try to open existing table
            self.table = self.db.open_table(table_name)
            logger.info(f"Connected to existing table: {table_name}")

        except Exception:
            # Create new table
            # Create empty table
            empty_data = pa.table(
                {
                    "id": pa.array([], type=pa.string()),
                    "text": pa.array([], type=pa.string()),
                    "vector": pa.array([], type=pa.list_(pa.float32())),
                    "metadata": pa.array([], type=pa.string()),
                    "chunk_id": pa.array([], type=pa.string()),
                    "page_num": pa.array([], type=pa.int32()),
                    "word_count": pa.array([], type=pa.int32()),
                    "source_url": pa.array([], type=pa.string()),
                }
            )

            self.table = self.db.create_table(table_name, empty_data)
            logger.info(f"Created new table: {table_name}")

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

                data.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "vector": list(embedding),  # Ensure it's a Python list
                        "metadata": str(metadata),  # Convert to JSON string
                        "chunk_id": metadata.get("chunk_id", ""),
                        "page_num": int(
                            metadata.get("page_num", 0)
                        ),  # Ensure it's an int
                        "word_count": int(
                            metadata.get("word_count", 0)
                        ),  # Ensure it's an int
                        "source_url": metadata.get("source_url", ""),
                    }
                )

            logger.info(f"Prepared {len(data)} records for insertion")

            # Convert to PyArrow table with explicit schema
            # Extract data into separate lists for each column
            ids = [item["id"] for item in data]
            texts = [item["text"] for item in data]
            vectors = [item["vector"] for item in data]
            metadatas = [item["metadata"] for item in data]
            chunk_ids = [item["chunk_id"] for item in data]
            page_nums = [item["page_num"] for item in data]
            word_counts = [item["word_count"] for item in data]
            source_urls = [item["source_url"] for item in data]

            logger.info("Creating PyArrow table...")

            # Create PyArrow table with explicit column data
            pa_table = pa.table(
                {
                    "id": pa.array(ids, type=pa.string()),
                    "text": pa.array(texts, type=pa.string()),
                    "vector": pa.array(
                        vectors, type=pa.list_(pa.float32())
                    ),  # Remove dimension constraint
                    "metadata": pa.array(metadatas, type=pa.string()),
                    "chunk_id": pa.array(chunk_ids, type=pa.string()),
                    "page_num": pa.array(page_nums, type=pa.int32()),
                    "word_count": pa.array(word_counts, type=pa.int32()),
                    "source_url": pa.array(source_urls, type=pa.string()),
                }
            )

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
            # Get query embedding
            query_embedding = await self._get_embeddings([query])
            query_vector = query_embedding[0]

            # Perform vector search
            results = self.table.search(query_vector).limit(k).to_pandas()

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
                return False

            # Try to get count
            self.table.count_rows()
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
