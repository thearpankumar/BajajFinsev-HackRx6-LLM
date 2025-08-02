from pydantic_settings import BaseSettings
import json
from pydantic import field_validator

class Settings(BaseSettings):
    # API Authentication
    API_KEY: str = "12345678901"

    # Google Gemini Configuration
    GOOGLE_API_KEY: str

    # OpenAI Configuration
    OPENAI_API_KEY: str

    # LanceDB Configuration
    LANCEDB_PATH: str = "/tmp/lancedb"
    LANCEDB_TABLE_NAME: str = "hackrx_bajaj_rag"

    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # Fast and cost-effective
    EMBEDDING_DIMENSIONS: int = 1536
    CHUNK_SIZE: int = 2000  # Characters per chunk (optimized for large docs)
    CHUNK_OVERLAP: int = 200  # Reduced overlap to prevent infinite loops (was 600)
    MAX_CHUNKS_PER_QUERY: int = 50  # Increased from 35 to 50 for better coverage
    
    # Large Document Processing Configuration
    EMBEDDING_BATCH_SIZE: int = 100  # Texts per parallel batch (reduced to avoid gRPC issues)
    PARALLEL_BATCHES: int = 3  # Max concurrent batches (reduced for stability)
    MAX_SECTIONS_PER_QUERY: int = 10  # Increased from 5 to 10 for better coverage
    ENABLE_HIERARCHICAL_PROCESSING: bool = False  # Disable hierarchical chunking (expensive and usually unnecessary)
    HIERARCHICAL_THRESHOLD_MB: int = 50  # Only use hierarchical processing for 50MB+ documents
    LARGE_DOCUMENT_THRESHOLD: int = 20971520  # 20MB in bytes threshold for large doc processing (increased from 20MB)
    LARGE_DOCUMENT_CHAR_THRESHOLD: int = 2000000  # 2M characters (~8MB text) threshold (increased from 500K)
    
    # Performance Optimization
    ENABLE_DOCUMENT_CACHE: bool = True  # Enable document-level caching
    CACHE_EXPIRY_HOURS: int = 24  # Cache expiry time
    ENABLE_STREAMING_RESPONSES: bool = False  # Enable streaming responses (future feature)
    ENABLE_METADATA_EXTRACTION: bool = False  # Disable metadata extraction by default (slow)
    METADATA_EXTRACTION_PAGE_LIMIT: int = 10  # Only extract metadata for documents with <10 pages
    METADATA_EXTRACTION_CHUNK_LIMIT: int = 50  # Only extract metadata for first 50 chunks
    PDF_EXTRACTION_TIMEOUT: int = 600  # Timeout in seconds for PDF extraction (10 minutes)
    
    # Fast processing mode for 40-second target
    ENABLE_FAST_MODE: bool = True  # Skip embeddings for speed
    FAST_MODE_MAX_CHUNKS: int = 300  # Increased from 200 - use even more chunks
    FAST_MODE_CHUNK_SIZE: int = 4000  # Larger chunks for fast mode
    FAST_MODE_MAX_PAGES: int = 0  # 0 = process ALL pages (disabled page limit)
    FAST_MODE_SKIP_ANALYSIS: bool = True  # Skip document analysis in fast mode
    
    # Ultra-large document processing (20MB+)
    ULTRA_LARGE_DOC_THRESHOLD: int = 20 * 1024 * 1024  # 20MB threshold for ultra-fast processing
    LARGE_DOC_THRESHOLD: int = 5 * 1024 * 1024  # 5MB threshold for fast processing
    DISABLE_METADATA_FOR_LARGE_DOCS: bool = True  # Disable metadata extraction for large docs
    USE_FAST_HASH_FOR_LARGE_DOCS: bool = True  # Use faster hash function for large docs
    
    # Parallel PDF processing
    PARALLEL_PDF_THRESHOLD: int = 100  # Pages threshold for parallel processing
    ULTRA_PARALLEL_PDF_THRESHOLD: int = 500  # Pages threshold for process-based parallelism
    MAX_PDF_WORKER_THREADS: int = 8  # Maximum threads for PDF processing
    MAX_PDF_WORKER_PROCESSES: int = 8  # Maximum processes for PDF processing
    PDF_BATCH_SIZE: int = 50  # Pages per batch in parallel processing
    
    # Text Cleaning Configuration
    SPECIAL_CHAR_MAPPING: dict = {
        '–': '-',  # en dash to hyphen
        '—': '-',  # em dash to hyphen
        '―': '-',  # horizontal bar to hyphen
        '“': '"',  # left double quotation mark
        '”': '"',  # right double quotation mark
        '‘': "'",  # left single quotation mark
        '’': "'",  # right single quotation mark
        '…': '...',  # ellipsis
        '•': '*',  # bullet
        '′': "'",  # prime
        '″': '"',  # double prime
    }  # Configurable mapping for special characters

    @field_validator('SPECIAL_CHAR_MAPPING', mode='before')
    @classmethod
    def validate_special_char_mapping(cls, value):
        """Validate and parse special character mapping from environment variable."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Return empty dict if parsing fails
                return {}
        return value

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
