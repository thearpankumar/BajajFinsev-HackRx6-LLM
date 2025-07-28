from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Authentication
    API_KEY: str = "12345678901"

    # Database Configuration
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/bajaj_finsev_db"

    # Google Gemini Configuration
    GOOGLE_API_KEY: str

    # Groq Configuration
    GROQ_API_KEY: str

    # Pinecone Configuration
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "bajaj-legal-docs"

    # AWS Configuration (Optional - for OCR fallback)
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()