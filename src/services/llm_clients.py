import google.generativeai as genai
from src.core.config import settings

# Configure Google Gemini
genai.configure(api_key=settings.GOOGLE_API_KEY)

# --- Model Definitions ---
# Use the latest, most capable model for the main RAG task
GEMINI_PRO_MODEL = genai.GenerativeModel('gemini-2.5-pro')

# Use the fast, lightweight model for query clarification
GEMINI_FLASH_MODEL = genai.GenerativeModel('gemini-2.5-flash-lite')
