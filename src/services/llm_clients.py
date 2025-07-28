import google.generativeai as genai
from groq import AsyncGroq
from src.core.config import settings

# Configure Google Gemini
genai.configure(api_key=settings.GOOGLE_API_KEY)

# Initialize Google Gemini Model
gemini_flash_model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Initialize Groq Client
groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
