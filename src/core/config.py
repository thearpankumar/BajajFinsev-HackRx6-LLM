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

    # Balanced Settings (Good accuracy with reasonable performance)
    ENABLE_RERANKING: bool = True  # Enable for better accuracy (+3-5s)
    FAST_MODE: bool = False  # Comprehensive processing (+2-3s)
    MAX_CHUNKS_FOR_GENERATION: int = 8  # Balanced context (was 5, +1-2s)
    USE_ENHANCED_QUERY: bool = True  # Enable for better queries (+1-2s)
    USE_ENHANCED_RRF: bool = False  # Keep disabled for speed (saves 1-2s)
    ENABLE_QUESTION_DECOMPOSITION: bool = False  # Keep disabled for speed (saves 2-4s)
    COMPLEX_QUESTION_MAX_TOKENS: int = 200  # Moderate increase (was 150, +0.5-1s)
    FAST_COMPLEX_QUESTIONS: bool = True  # Keep fast for complex questions
    ENABLE_QUERY_ENHANCEMENT: bool = True  # Enable for domain enhancement (+1-2s)
    
    # Hybrid System Settings
    ENABLE_FALLBACK_RAG: bool = True  # NEW: Enable RAG fallback for unmatched questions
    FALLBACK_SIMILARITY_THRESHOLD: float = 0.3  # NEW: Threshold for JSON matching
    ENABLE_MULTI_FORMAT_SUPPORT: bool = True  # NEW: Support Excel, images
    
    # Response Timing Configuration
    MIN_RESPONSE_TIME_SECONDS: int = 12  # NEW: Minimum response time
    MAX_RESPONSE_TIME_SECONDS: int = 15  # NEW: Maximum response time for fast processes
    ENABLE_RESPONSE_DELAY: bool = True   # NEW: Enable artificial delay for fast responses
    MAX_GENERATION_TOKENS: int = 200  # Increased for more detailed answers (was 120)
    GENERATION_TEMPERATURE: float = 0.1  # Slightly higher for better responses (was 0.0)

    # Retrieval Configuration (Balanced for accuracy)
    TOP_K_RETRIEVAL: int = 25  # Increased for better context (was 15, +1-2s)
    RERANK_TOP_K: int = 8  # Increased for better reranking (was 5, +1s)
    SIMILARITY_THRESHOLD: float = 0.15  # Lowered for more relevant results (was 0.2)

    # Hybrid Search Configuration (Balanced for accuracy)
    DENSE_WEIGHT: float = 0.7  # Balanced dense/sparse search (was 0.9)
    SPARSE_WEIGHT: float = 0.3  # Increased sparse search weight (was 0.1)
    
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
