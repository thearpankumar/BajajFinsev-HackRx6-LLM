"""
Configuration settings for BajajFinsev RAG System
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_KEY: str = "12345678901"

    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_GENERATION_MODEL: str = "gpt-4o-mini"
    OPENAI_MAX_TOKENS: int = 4000
    OPENAI_TEMPERATURE: float = 0.1

    # Google AI Configuration
    GOOGLE_API_KEY: str
    GOOGLE_MODEL: str = "gemini-2.0-flash-exp"

    # Vector Database Configuration
    VECTOR_DB_PATH: str = "./vector_db"
    VECTOR_DIMENSION: int = 1536  # text-embedding-3-small dimension

    # Document Processing Configuration
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENT_SIZE_MB: int = 100

    # Performance Configuration
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 300  # 5 minutes
    CACHE_TTL: int = 3600  # 1 hour

    # Parallel Processing Configuration
    MAX_PARALLEL_QUESTIONS: int = 40  # Maximum parallel questions to process
    QUESTION_BATCH_SIZE: int = 10  # Process questions in batches
    PARALLEL_PROCESSING: bool = True  # Enable parallel question processing

    # Speed vs Accuracy Trade-offs
    ENABLE_RERANKING: bool = False  # Disable by default for speed
    FAST_MODE: bool = True  # Enable fast mode by default
    MAX_CHUNKS_FOR_GENERATION: int = 10  # Increased for better context

    # Retrieval Configuration (optimized for speed)
    TOP_K_RETRIEVAL: int = 20  # Increased for better coverage
    RERANK_TOP_K: int = 10  # Increased for better relevance
    SIMILARITY_THRESHOLD: float = 0.5  # Lowered for more inclusive retrieval

    # BM25 Configuration
    BM25_K1: float = 1.2
    BM25_B: float = 0.75

    # Hybrid Search Configuration
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()
