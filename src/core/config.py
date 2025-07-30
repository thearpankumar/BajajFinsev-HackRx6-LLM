from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Authentication
    API_KEY: str = "12345678901"

    # Google Gemini Configuration
    GOOGLE_API_KEY: str

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
