"""
Centralized Configuration System for BajajFinsev RAG System
Comprehensive Pydantic-based configuration with environment variable support
"""

import os
from typing import Optional, List
from enum import Enum
from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings


class GPUProvider(str, Enum):
    """GPU provider options"""
    CUDA = "cuda"
    MPS = "mps" 
    CPU = "cpu"


class EmbeddingModel(str, Enum):
    """Available embedding models"""
    E5_BASE = "intfloat/multilingual-e5-base"
    E5_LARGE = "intfloat/multilingual-e5-large"
    BGE_M3 = "BAAI/bge-m3"
    OPENAI_SMALL = "text-embedding-3-small"


class LLMProvider(str, Enum):
    """Available LLM providers"""
    GROQ_LLAMA = "groq/llama-3.3-70b-versatile"
    GEMINI = "gemini-2.5-flash-lite"
    OPENAI = "gpt-4o-mini"


class VectorDBType(str, Enum):
    """Vector database options"""
    FAISS_GPU = "faiss-gpu"
    FAISS_CPU = "faiss-cpu"
    QDRANT = "qdrant"
    CHROMA = "chroma"


class SystemConfig(BaseSettings):
    """Centralized system configuration using Pydantic BaseSettings"""

    # ========== API Configuration ==========
    API_KEY: str = Field("123456", description="API authentication key")

    # ========== GPU Configuration ==========
    gpu_provider: GPUProvider = Field(GPUProvider.CUDA, description="GPU provider to use")
    gpu_memory_fraction: float = Field(0.8, description="RTX 3050: 80% of 4GB = 3.2GB usable")
    batch_size: int = Field(16, description="RTX 3050 optimized batch size")
    max_batch_size: int = Field(32, description="Maximum batch size for e5-base model")
    enable_mixed_precision: bool = Field(True, description="FP16 for memory efficiency")
    gpu_memory_cleanup_interval: int = Field(100, description="GPU cleanup every N operations")

    # ========== Embedding Configuration ==========
    embedding_model: EmbeddingModel = Field(EmbeddingModel.E5_BASE, description="Multilingual embedding model")
    embedding_dimension: int = Field(768, description="e5-base embedding dimension")
    embedding_max_length: int = Field(512, description="Maximum sequence length for embeddings")
    enable_embedding_cache: bool = Field(True, description="Enable embedding caching")

    # ========== LLM Configuration ==========
    query_llm: LLMProvider = Field(LLMProvider.GEMINI, description="LLM for query understanding")
    response_llm: LLMProvider = Field(LLMProvider.GROQ_LLAMA, description="LLM for response generation")
    
    # ========== API Keys ==========
    groq_api_key: Optional[str] = Field(None, env="GROQ_API_KEY", description="Groq API key")
    gemini_api_key: Optional[str] = Field(None, env="GEMINI_API_KEY", description="Google Gemini API key") 
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY", description="OpenAI API key")
    google_translate_key: Optional[str] = Field(None, env="GOOGLE_TRANSLATE_KEY", description="Google Translate API key")
    azure_translator_key: Optional[str] = Field(None, env="AZURE_TRANSLATOR_KEY", description="Azure Translator key")

    # ========== Processing Configuration ==========
    max_workers: int = Field(8, description="Parallel processing workers")
    chunk_size: int = Field(512, description="Document chunk size in tokens")
    chunk_overlap: int = Field(128, description="Overlap between chunks in tokens")
    max_document_size_mb: int = Field(100, description="Maximum document size in MB")
    max_concurrent_operations: int = Field(15, description="Maximum concurrent operations")
    
    # ========== Vector Database Configuration ==========
    vector_db_type: VectorDBType = Field(VectorDBType.FAISS_GPU, description="Vector database type")
    
    # FAISS Configuration
    faiss_index_type: str = Field("HNSW", description="FAISS index type")
    hnsw_m: int = Field(32, description="HNSW M parameter")
    hnsw_ef_construction: int = Field(200, description="HNSW efConstruction parameter")
    hnsw_ef_search: int = Field(100, description="HNSW efSearch parameter")
    
    # Qdrant Configuration (fallback)
    qdrant_host: str = Field("localhost", description="Qdrant host")
    qdrant_port: int = Field(6333, description="Qdrant port")
    qdrant_collection_name: str = Field("bajaj_documents", description="Qdrant collection name")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY", description="Qdrant API key")
    qdrant_timeout: int = Field(60, description="Qdrant timeout in seconds")

    # ========== Redis Configuration ==========
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port") 
    redis_db: int = Field(0, description="Redis database number")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD", description="Redis password")
    redis_max_connections: int = Field(20, description="Maximum Redis connections")
    redis_timeout: int = Field(30, description="Redis connection timeout")

    # ========== Performance Thresholds ==========
    query_timeout_seconds: int = Field(30, description="Maximum query processing time")
    min_response_time_seconds: int = Field(4, description="Minimum response time for UX")
    max_response_time_seconds: int = Field(6, description="Maximum response time target")
    cache_ttl_hours: int = Field(24, description="Default cache TTL in hours")
    
    # ========== Translation Settings ==========
    enable_translation: bool = Field(True, description="Enable Malayalam-English translation")
    translation_confidence_threshold: float = Field(0.7, description="Translation quality threshold")
    enable_parallel_translation: bool = Field(True, description="Enable parallel translation processing")
    translation_batch_size: int = Field(10, description="Translation batch size")

    # ========== Document Processing Settings ==========
    supported_formats: List[str] = Field(
        ["pdf", "docx", "doc", "xlsx", "xls", "csv", "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"],
        description="Supported document formats"
    )
    
    # OCR Settings
    ocr_engine: str = Field("easyocr", description="OCR engine (easyocr, tesseract)")
    ocr_languages: List[str] = Field(["en", "ml"], description="OCR supported languages")
    enable_ocr_preprocessing: bool = Field(True, description="Enable image preprocessing for OCR")
    max_image_size_mb: int = Field(10, description="Maximum image size for OCR")
    
    # ========== Human Response Settings ==========
    conversational_tone: bool = Field(True, description="Enable human-like conversational responses")
    response_length_preference: str = Field("medium", description="Response length: short/medium/detailed")
    include_source_attribution: bool = Field(True, description="Include source citations in responses")
    enable_response_streaming: bool = Field(True, description="Enable streaming responses")
    
    # ========== Retrieval Configuration ==========
    top_k_retrieval: int = Field(25, description="Number of chunks to retrieve")
    rerank_top_k: int = Field(8, description="Number of chunks after reranking")
    similarity_threshold: float = Field(0.15, description="Minimum similarity threshold")
    enable_reranking: bool = Field(True, description="Enable result reranking")
    dense_weight: float = Field(0.7, description="Weight for dense search in hybrid retrieval")
    sparse_weight: float = Field(0.3, description="Weight for sparse search in hybrid retrieval")

    # ========== MCP Integration Settings ==========
    enable_mcp_integration: bool = Field(True, description="Enable MCP tool integration")
    mcp_timeout_seconds: int = Field(30, description="MCP tool timeout")
    max_web_requests_per_document: int = Field(20, description="Maximum web requests per document")
    web_request_timeout: int = Field(10, description="Web request timeout in seconds")

    # ========== Monitoring and Logging ==========
    enable_performance_monitoring: bool = Field(True, description="Enable performance monitoring")
    log_level: str = Field("INFO", description="Logging level")
    enable_detailed_logging: bool = Field(False, description="Enable detailed debug logging")
    metrics_collection_interval: int = Field(60, description="Metrics collection interval in seconds")

    # ========== Development Settings ==========
    enable_debug_mode: bool = Field(False, env="DEBUG", description="Enable debug mode")
    enable_response_delay: bool = Field(True, description="Enable artificial response delays for UX")
    skip_gpu_check: bool = Field(False, description="Skip GPU availability check")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"
        use_enum_values = True


# Global configuration instance
config = SystemConfig()

# Legacy settings alias for backward compatibility
settings = config
