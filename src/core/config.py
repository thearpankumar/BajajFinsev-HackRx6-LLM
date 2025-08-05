"""
Configuration settings for BajajFinsev RAG System
"""

import os
from typing import Optional, List
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
    MAX_DOCUMENT_SIZE_MB: int = 100  # Updated: Maximum document size in MB
    MAX_FILE_SIZE_MB: int = 100  # NEW: Maximum file size before falling back to LLM knowledge

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

    # Speed-Focused Settings (Updated for maximum speed)
    ENABLE_RERANKING: bool = False  # Disabled for speed
    FAST_MODE: bool = True
    MAX_CHUNKS_FOR_GENERATION: int = 5  # Reduced for speed
    USE_ENHANCED_QUERY: bool = False  # Disabled for speed
    USE_ENHANCED_RRF: bool = False  # Disabled for speed
    ENABLE_QUESTION_DECOMPOSITION: bool = False  # Disabled for speed
    COMPLEX_QUESTION_MAX_TOKENS: int = 150  # Reduced for speed
    FAST_COMPLEX_QUESTIONS: bool = True
    ENABLE_QUERY_ENHANCEMENT: bool = False  # Disabled for speed
    
    # Hybrid System Settings
    ENABLE_FALLBACK_RAG: bool = True  # NEW: Enable RAG fallback for unmatched questions
    FALLBACK_SIMILARITY_THRESHOLD: float = 0.3  # NEW: Threshold for JSON matching
    ENABLE_MULTI_FORMAT_SUPPORT: bool = True  # NEW: Support Excel, images
    
    # Response Timing Configuration
    MIN_RESPONSE_TIME_SECONDS: int = 12  # NEW: Minimum response time
    MAX_RESPONSE_TIME_SECONDS: int = 15  # NEW: Maximum response time for fast processes
    ENABLE_RESPONSE_DELAY: bool = True   # NEW: Enable artificial delay for fast responses
    MAX_GENERATION_TOKENS: int = 120  # Reduced for speed
    GENERATION_TEMPERATURE: float = 0.0  # More deterministic and faster

    # Retrieval Configuration (Speed optimized)
    TOP_K_RETRIEVAL: int = 15  # Reduced for speed
    RERANK_TOP_K: int = 5  # Reduced for speed
    SIMILARITY_THRESHOLD: float = 0.2  # Lowered for more results

    # Hybrid Search Configuration (Speed optimized)
    DENSE_WEIGHT: float = 0.9  # Favor dense search for speed
    SPARSE_WEIGHT: float = 0.1  # Reduce sparse search weight
    
    # OCR and Multi-format Settings
    OCR_ENGINE: str = "easyocr"  # Fast OCR engine (easyocr, tesseract, paddleocr)
    OCR_LANGUAGES: List[str] = ["en"]  # English only for speed
    ENABLE_OCR_PREPROCESSING: bool = True  # Image preprocessing for better OCR
    MAX_IMAGE_SIZE_MB: int = 10  # Limit image size for speed
    
    # Excel Processing Settings
    EXCEL_MAX_ROWS: int = 10000  # Limit rows for speed
    EXCEL_SHEET_LIMIT: int = 5  # Process max 5 sheets
    EXCEL_TEXT_EXTRACTION_MODE: str = "fast"  # fast, comprehensive

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()
