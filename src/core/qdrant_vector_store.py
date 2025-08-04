"""
Qdrant Vector Store - Drop-in replacement for LanceDB
Maintains exact same interface as existing vector_store.py for seamless migration
"""

import logging
import json
import uuid
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, 
    MatchValue
)
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import settings
from src.core.enhanced_document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Qdrant Vector Store with EXACT same interface as existing VectorStore
    Drop-in replacement - no changes needed in RAG engine
    """

    def __init__(self):
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        self.dimension = settings.VECTOR_DIMENSION
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Initialize clients
        self.qdrant_client = None
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Connection settings
        self.host = settings.QDRANT_HOST
        self.port = settings.QDRANT_PORT
        self.timeout = settings.QDRANT_TIMEOUT

    async def initialize(self):
        """Initialize Qdrant - SAME interface as existing VectorStore.initialize()"""
        try:
            logger.info("Starting Qdrant vector store initialization...")
            
            # Connect to Qdrant
            logger.info(f"Connecting to Qdrant at {self.host}:{self.port}")
            self.qdrant_client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=settings.QDRANT_API_KEY,
                timeout=self.timeout
            )
            
            # Test connection
            logger.info("Testing Qdrant connection...")
            collections = self.qdrant_client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Found {len(collections.collections)} collections")
            
            # Create collection if it doesn't exist
            await self._create_collection_if_not_exists()
            
            logger.info("Qdrant vector store initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant vector store: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _create_collection_if_not_exists(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"Collection '{self.collection_name}' already exists with {collection_info.points_count} points")
                return
            except UnexpectedResponse as e:
                if "Not found" in str(e):
                    logger.info(f"Collection '{self.collection_name}' doesn't exist, creating...")
                else:
                    raise
            
            # Create collection with vector configuration
            logger.info(f"Creating collection '{self.collection_name}' with dimension {self.dimension}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE  # Use cosine similarity like your existing system
                )
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise

    async def add_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add texts to the vector store with embeddings
        SAME interface as existing VectorStore.add_texts()
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries

        Returns:
            List of document IDs
        """
        if not texts:
            return []

        try:
            logger.info(f"Adding {len(texts)} texts to Qdrant vector store...")
            
            # Generate embeddings in batches (memory efficient)
            logger.info("Generating embeddings...")
            embeddings = await self._get_embeddings(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # Prepare points for Qdrant
            points = []
            doc_ids = []

            for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                # Generate UUID for Qdrant (required format)
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)

                # Validate embedding
                if not isinstance(embedding, (list, np.ndarray)):
                    raise ValueError(f"Invalid embedding type: {type(embedding)}")

                if len(embedding) != self.dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(embedding)}, expected {self.dimension}"
                    )

                # Convert embedding to list (Qdrant expects list)
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                # Create point with payload (metadata)
                point = PointStruct(
                    id=doc_id,  # UUID string
                    vector=embedding,  # List of floats
                    payload={
                        "text": text,
                        "metadata": json.dumps(metadata),
                        "chunk_id": metadata.get("chunk_id", ""),
                        "page_num": int(metadata.get("page_num", 0)),
                        "word_count": int(metadata.get("word_count", 0)),
                        "source_url": metadata.get("source_url", ""),
                    }
                )
                
                # Validate point structure
                if not isinstance(point.id, str):
                    raise ValueError(f"Point ID must be string, got {type(point.id)}")
                if not isinstance(point.vector, list):
                    raise ValueError(f"Vector must be list, got {type(point.vector)}")
                if len(point.vector) != self.dimension:
                    raise ValueError(f"Vector dimension mismatch: {len(point.vector)} != {self.dimension}")
                
                points.append(point)

            # Upload points to Qdrant in batches (memory efficient)
            batch_size = 100  # Process in smaller batches
            logger.info(f"Uploading {len(points)} points to Qdrant in batches of {batch_size}")
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(points) + batch_size - 1)//batch_size
                
                logger.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} points)")
                
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    logger.info(f"✅ Successfully uploaded batch {batch_num}")
                    
                except Exception as batch_error:
                    logger.error(f"❌ Failed to upload batch {batch_num}: {str(batch_error)}")
                    # Log the first point in the batch for debugging
                    if batch:
                        sample_point = batch[0]
                        logger.error(f"Sample point ID: {sample_point.id} (type: {type(sample_point.id)})")
                        logger.error(f"Sample vector length: {len(sample_point.vector)}")
                    raise

            logger.info(f"Successfully added {len(texts)} texts to Qdrant vector store")
            return doc_ids

        except Exception as e:
            logger.error(f"Error adding texts to Qdrant vector store: {str(e)}")
            raise

    async def similarity_search(
        self, query: str, k: int = 10, filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Perform similarity search
        SAME interface as existing VectorStore.similarity_search()

        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional filters (not implemented yet)

        Returns:
            List of (DocumentChunk, similarity_score) tuples
        """
        try:
            # Check if collection exists and has data
            if not self.qdrant_client:
                logger.warning("Qdrant client not initialized")
                return []
                
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                if collection_info.points_count == 0:
                    logger.warning("Qdrant collection is empty, no data to search")
                    return []
            except Exception as e:
                logger.warning(f"Error checking collection: {str(e)}")
                return []

            # Get query embedding
            logger.info("Generating query embedding...")
            query_embedding = await self._get_embeddings([query])
            query_vector = query_embedding[0]
            
            # Convert to list if numpy array
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()

            # Perform vector search
            logger.info(f"Performing similarity search with k={k}")
            logger.info(f"Query: '{query[:100]}...'")
            
            # First try with threshold
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=True,
                with_vectors=False,
                score_threshold=settings.SIMILARITY_THRESHOLD
            )
            
            # If we get too few results, try without threshold
            if len(search_result) < 3:
                logger.warning(f"Only {len(search_result)} results with threshold {settings.SIMILARITY_THRESHOLD}, trying without threshold...")
                search_result = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=k,
                    with_payload=True,
                    with_vectors=False
                    # No threshold - get all results
                )
            
            logger.info(f"Qdrant returned {len(search_result)} results")

            # Convert results to DocumentChunk objects
            chunks_with_scores = []

            for idx, scored_point in enumerate(search_result):
                try:
                    # Extract payload
                    payload = scored_point.payload
                    
                    # Parse metadata
                    try:
                        metadata = json.loads(payload.get("metadata", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                    # Create DocumentChunk (same format as existing system)
                    chunk = DocumentChunk(
                        text=payload.get("text", ""),
                        page_num=payload.get("page_num", 0),
                        chunk_id=payload.get("chunk_id", ""),
                        metadata=metadata,
                    )

                    # Qdrant returns similarity score (higher is better)
                    similarity_score = float(scored_point.score)
                    
                    # Debug logging for large documents
                    if idx < 3:  # Log first 3 results for debugging
                        logger.info(f"Result {idx+1}: Score={similarity_score:.4f}")
                        logger.info(f"  Text preview: {chunk.text[:150]}...")
                        logger.info(f"  Chunk ID: {chunk.chunk_id}")
                        logger.info(f"  Page: {chunk.page_num}")

                    chunks_with_scores.append((chunk, similarity_score))

                except Exception as e:
                    logger.warning(f"Error processing search result {idx}: {str(e)}")
                    continue

            logger.info(f"Found {len(chunks_with_scores)} similar chunks for query")
            return chunks_with_scores

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts using OpenAI
        SAME implementation as existing VectorStore._get_embeddings()

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
                logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")

                # Get embeddings from OpenAI
                response = await self.openai_client.embeddings.create(
                    model=self.embedding_model, 
                    input=batch
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
            logger.info(f"Deleting vectors from source: {source_url}")
            
            # Create filter for source URL
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="source_url",
                        match=MatchValue(value=source_url)
                    )
                ]
            )
            
            # Delete points matching the filter
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=filter_condition
            )
            
            logger.info(f"Successfully deleted vectors from source: {source_url}")

        except Exception as e:
            logger.error(f"Error deleting by source: {str(e)}")

    async def document_exists(self, source_url: str) -> bool:
        """
        Check if document exists in vector database
        
        Args:
            source_url: URL of the document to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            if not self.qdrant_client:
                logger.warning("Qdrant client not initialized")
                return False
            
            # Check if collection exists
            try:
                collection_info = self.qdrant_client.get_collection(self.collection_name)
                if collection_info.points_count == 0:
                    return False
            except Exception:
                return False
            
            # Search for any point with this source URL
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="source_url",
                        match=MatchValue(value=source_url)
                    )
                ]
            )
            
            # Use scroll to check existence (more efficient than search)
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1,  # We only need to know if at least one exists
                with_payload=False,
                with_vectors=False
            )
            
            # Handle scroll result
            if isinstance(scroll_result, tuple):
                points = scroll_result[0]
            else:
                points = scroll_result
            
            exists = len(points) > 0
            logger.info(f"Document {source_url} {'exists' if exists else 'does not exist'} in vector database")
            return exists
            
        except Exception as e:
            logger.error(f"Error checking document existence for {source_url}: {str(e)}")
            return False

    async def get_document_chunks_from_db(self, source_url: str) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve all chunks for a document from vector database
        
        Args:
            source_url: URL of the document
            
        Returns:
            List of (DocumentChunk, score) tuples
        """
        try:
            if not self.qdrant_client:
                logger.warning("Qdrant client not initialized")
                return []
            
            # Create filter for source URL
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="source_url",
                        match=MatchValue(value=source_url)
                    )
                ]
            )
            
            # Scroll through all points for this document
            all_chunks = []
            offset = None
            
            while True:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_condition,
                    limit=100,  # Process in batches
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                # Handle scroll result
                if isinstance(scroll_result, tuple):
                    points, next_offset = scroll_result
                else:
                    points = scroll_result
                    next_offset = None
                
                if not points:
                    break
                
                # Convert points to DocumentChunk objects
                for point in points:
                    try:
                        payload = point.payload
                        
                        # Parse metadata
                        try:
                            metadata = json.loads(payload.get("metadata", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                        
                        # Create DocumentChunk
                        chunk = DocumentChunk(
                            text=payload.get("text", ""),
                            page_num=payload.get("page_num", 0),
                            chunk_id=payload.get("chunk_id", ""),
                            metadata=metadata,
                        )
                        
                        # Use a default score of 1.0 since we're not doing similarity search
                        all_chunks.append((chunk, 1.0))
                        
                    except Exception as e:
                        logger.warning(f"Error processing point: {str(e)}")
                        continue
                
                # Check if we have more data
                if not next_offset:
                    break
                offset = next_offset
            
            logger.info(f"Retrieved {len(all_chunks)} chunks for document {source_url} from vector database")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving document chunks for {source_url}: {str(e)}")
            return []

    async def get_document_metadata_from_db(self, source_url: str) -> Dict[str, Any]:
        """
        Get document metadata from vector database
        
        Args:
            source_url: URL of the document
            
        Returns:
            Dictionary with document metadata
        """
        try:
            chunks = await self.get_document_chunks_from_db(source_url)
            
            if not chunks:
                return {}
            
            # Calculate metadata from chunks
            total_chunks = len(chunks)
            total_size = sum(len(chunk.text) for chunk, _ in chunks)
            pages = set(chunk.page_num for chunk, _ in chunks)
            
            return {
                'source': 'vector_database',
                'type': 'reconstructed_from_vector_db',
                'num_chunks': total_chunks,
                'size': total_size,
                'pages': len(pages),
                'source_url': source_url,
                'reconstructed': True
            }
            
        except Exception as e:
            logger.error(f"Error getting document metadata for {source_url}: {str(e)}")
            return {}

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics
        SAME interface as existing VectorStore.get_stats()
        """
        try:
            if not self.qdrant_client:
                return {"error": "Qdrant client not initialized"}

            # Get collection info
            collection_info = self.qdrant_client.get_collection(self.collection_name)

            return {
                "total_vectors": collection_info.points_count,
                "dimension": self.dimension,
                "embedding_model": self.embedding_model,
                "collection_name": self.collection_name,
                "status": collection_info.status,
            }

        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}

    async def health_check(self) -> bool:
        """
        Check if vector store is healthy
        SAME interface as existing VectorStore.health_check()
        """
        try:
            if not self.qdrant_client:
                logger.warning("Qdrant client not initialized for health check")
                return False

            # Try to get collection info - this should work if Qdrant is healthy
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Qdrant health check passed - {collection_info.points_count} points in collection")
            return True

        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False

    async def close(self):
        """
        Close the vector store connection
        SAME interface as existing VectorStore.close()
        """
        try:
            if self.qdrant_client:
                # Qdrant client doesn't need explicit closing, but we can clear the reference
                self.qdrant_client = None
                logger.info("Qdrant vector store connection closed")

        except Exception as e:
            logger.error(f"Error closing Qdrant vector store: {str(e)}")
