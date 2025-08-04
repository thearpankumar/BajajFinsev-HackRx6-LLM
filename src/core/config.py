"""
Configuration settings for BajajFinsev RAG System
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_KEY: str = "123456"

    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_GENERATION_MODEL: str = "gpt-4o-mini"

    # Google AI Configuration
    GOOGLE_API_KEY: str
    GOOGLE_MODEL: str = "gemini-2.5-flash-lite"

    # Qdrant Vector Database Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "bajaj_documents"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_TIMEOUT: int = 60
    VECTOR_DIMENSION: int = 1536  # OpenAI text-embedding-3-small dimension

    # Document Processing Configuration
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE_MB: int = 100

    # Document Caching Configuration
    ENABLE_PERSISTENT_DOCUMENT_CACHE: bool = True
    DOCUMENT_CACHE_PATH: str = "/app/document_cache" if os.path.exists("/app") else "./document_cache"
    DOCUMENT_CACHE_TTL_HOURS: int = 168  # 7 days
    CHECK_VECTOR_DB_BEFORE_DOWNLOAD: bool = True
    SKIP_DUPLICATE_DOCUMENTS: bool = True
    DOCUMENT_HASH_ALGORITHM: str = "sha256"

    # Parallel Processing Configuration
    MAX_PARALLEL_QUESTIONS: int = 40
    QUESTION_BATCH_SIZE: int = 10
    PARALLEL_PROCESSING: bool = True
    MAX_CONCURRENT_OPERATIONS: int = 15  # NEW: Limit concurrent operations (increased for better performance)

    # Accuracy Control Settings
    ENABLE_RERANKING: bool = True
    FAST_MODE: bool = True
    MAX_CHUNKS_FOR_GENERATION: int = 10
    USE_ENHANCED_QUERY: bool = True
    USE_ENHANCED_RRF: bool = True
    ENABLE_QUESTION_DECOMPOSITION: bool = True  # NEW: Handle complex multi-part questions
    COMPLEX_QUESTION_MAX_TOKENS: int = 250  # NEW: More tokens for complex questions
    FAST_COMPLEX_QUESTIONS: bool = True  # NEW: Use fast processing for complex questions
    ENABLE_QUERY_ENHANCEMENT: bool = True  # NEW: Enhance queries for better retrieval
    
    # Generation Settings
    MAX_GENERATION_TOKENS: int = 180
    GENERATION_TEMPERATURE: float = 0.1

    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 25
    RERANK_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.1

    # Hybrid Search Configuration
    DENSE_WEIGHT: float = 0.8
    SPARSE_WEIGHT: float = 0.2

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()
