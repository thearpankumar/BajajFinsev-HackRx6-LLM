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
    CHUNK_OVERLAP: int = 600  # Overlap between chunks (enhanced for scientific texts)
    MAX_CHUNKS_PER_QUERY: int = 50  # Increased from 35 to 50 for better coverage
    
    # Large Document Processing Configuration
    EMBEDDING_BATCH_SIZE: int = 100  # Texts per parallel batch (reduced to avoid gRPC issues)
    PARALLEL_BATCHES: int = 3  # Max concurrent batches (reduced for stability)
    MAX_SECTIONS_PER_QUERY: int = 10  # Increased from 5 to 10 for better coverage
    ENABLE_HIERARCHICAL_PROCESSING: bool = True  # Use hierarchical chunking for large docs
    LARGE_DOCUMENT_THRESHOLD: int = 20971520  # 20MB in bytes threshold for large doc processing
    LARGE_DOCUMENT_CHAR_THRESHOLD: int = 500000  # 500K characters (~2MB text) threshold
    
    # Performance Optimization
    ENABLE_DOCUMENT_CACHE: bool = True  # Enable document-level caching
    CACHE_EXPIRY_HOURS: int = 24  # Cache expiry time
    ENABLE_STREAMING_RESPONSES: bool = False  # Enable streaming responses (future feature)
    ENABLE_METADATA_EXTRACTION: bool = True  # Enable metadata extraction for chunks
    METADATA_EXTRACTION_PAGE_LIMIT: int = 100  # Skip metadata extraction for documents larger than this
    PDF_EXTRACTION_TIMEOUT: int = 600  # Timeout in seconds for PDF extraction (10 minutes)
    
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
