import google.generativeai as genai
from openai import AsyncOpenAI
from src.core.config import settings

# Configure Google Gemini
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Configure OpenAI
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# --- Model Definitions ---
# Use OpenAI GPT-4o-mini for the main RAG task
OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_CLIENT = openai_client

# Use the fast, lightweight Gemini model for query clarification
GEMINI_FLASH_MODEL = genai.GenerativeModel('gemini-2.5-flash-lite')
