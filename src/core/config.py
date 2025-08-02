from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Authentication
    API_KEY: str = "12345678901"

    # Google Gemini Configuration
    GOOGLE_API_KEY: str

    # OpenAI Configuration
    OPENAI_API_KEY: str

    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # Fast and cost-effective
    EMBEDDING_DIMENSIONS: int = 1536
    CHUNK_SIZE: int = 2000  # Characters per chunk (optimized for large docs)
    CHUNK_OVERLAP: int = 600  # Overlap between chunks (enhanced for scientific texts)
    MAX_CHUNKS_PER_QUERY: int = 35  # Number of top chunks to retrieve (increased for complex scientific queries)
    
    # Large Document Processing Configuration
    EMBEDDING_BATCH_SIZE: int = 100  # Texts per parallel batch (reduced to avoid gRPC issues)
    PARALLEL_BATCHES: int = 3  # Max concurrent batches (reduced for stability)
    MAX_SECTIONS_PER_QUERY: int = 5  # Sections to analyze in hierarchical processing (increased for better coverage)
    ENABLE_HIERARCHICAL_PROCESSING: bool = True  # Use hierarchical chunking for large docs
    LARGE_DOCUMENT_THRESHOLD: int = 20971520  # 20MB in bytes threshold for large doc processing
    LARGE_DOCUMENT_CHAR_THRESHOLD: int = 500000  # 500K characters (~2MB text) threshold
    
    # Performance Optimization
    ENABLE_DOCUMENT_CACHE: bool = True  # Enable document-level caching
    CACHE_EXPIRY_HOURS: int = 24  # Cache expiry time
    ENABLE_STREAMING_RESPONSES: bool = False  # Enable streaming responses (future feature)

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
