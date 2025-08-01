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
    CHUNK_SIZE: int = 1000  # Characters per chunk
    CHUNK_OVERLAP: int = 200  # Overlap between chunks
    MAX_CHUNKS_PER_QUERY: int = 5  # Number of top chunks to retrieve

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
