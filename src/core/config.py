"""
Configuration settings for BajajFinsev RAG System
"""

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
    GOOGLE_MODEL: str = "gemini-2.0-flash-exp"

    # Qdrant Vector Database Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "bajaj_documents"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_TIMEOUT: int = 60

    # Document Processing Configuration
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE_MB: int = 100

    # Parallel Processing Configuration
    MAX_PARALLEL_QUESTIONS: int = 40
    QUESTION_BATCH_SIZE: int = 10
    PARALLEL_PROCESSING: bool = True

    # Accuracy Control Settings
    ENABLE_RERANKING: bool = True
    FAST_MODE: bool = True
    MAX_CHUNKS_FOR_GENERATION: int = 8
    USE_ENHANCED_QUERY: bool = True
    USE_ENHANCED_RRF: bool = True
    
    # Generation Settings
    MAX_GENERATION_TOKENS: int = 180
    GENERATION_TEMPERATURE: float = 0.05

    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 20
    RERANK_TOP_K: int = 10
    SIMILARITY_THRESHOLD: float = 0.1

    # Hybrid Search Configuration
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()
