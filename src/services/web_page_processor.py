"""
Web Page Processor Service
Intelligently processes web page content to answer specific questions using an LLM.
"""

import logging
from typing import Any, Dict
import aiohttp
from bs4 import BeautifulSoup

from src.services.gemini_service import GeminiService, QueryContext

logger = logging.getLogger(__name__)

class WebPageProcessor:
    """
    Intelligently processes web page content to answer specific questions.
    """

    def __init__(self):
        self.gemini_service = GeminiService()
        self.is_initialized = False

    async def initialize(self) -> None:
        if not self.gemini_service.is_initialized:
            await self.gemini_service.initialize()
        self.is_initialized = True

    async def process_url(self, url: str, question: str) -> Dict[str, Any]:
        """
        Fetches a URL, extracts content, and uses an LLM to answer a question about it.
        """
        if not self.is_initialized:
            await self.initialize()

        try:
            # 1. Fetch URL content
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    html_content = await response.text()

            # 2. Extract clean text from HTML with targeted extraction
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements for better accuracy
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                element.decompose()
            
            # First try to find specific elements that might contain the answer
            page_text = ""
            
            # Check for specific token-related elements
            if "token" in question.lower():
                # Look for elements with 'token' in id, class, or text
                token_elements = soup.find_all(lambda tag: tag.name and (
                    'token' in str(tag.get('id', '')).lower() or 
                    'token' in str(tag.get('class', '')).lower() or
                    'secret' in str(tag.get('id', '')).lower()
                ))
                
                for element in token_elements:
                    element_text = element.get_text(strip=True)
                    if element_text and len(element_text) > 10:  # Non-empty meaningful text
                        page_text += element_text + " "
            
            # If no specific elements found or text is too short, get main content
            if len(page_text.strip()) < 50:
                # Try main content areas first
                main_content = soup.find(['main', 'article', 'section', 'div'])
                if main_content:
                    page_text = main_content.get_text(separator=' ', strip=True)
                else:
                    # Fallback to body content, excluding common noise
                    body = soup.find('body')
                    if body:
                        page_text = body.get_text(separator=' ', strip=True)
                    else:
                        page_text = soup.get_text(separator=' ', strip=True)
            
            # Clean up the extracted text
            page_text = ' '.join(page_text.split())  # Remove extra whitespace
            
            # Limit text size to avoid being too verbose for the LLM
            max_length = 10000  # Increased limit for better context
            if len(page_text) > max_length:
                # Smart truncation - try to keep complete sentences
                truncated = page_text[:max_length]
                last_period = truncated.rfind('.')
                if last_period > max_length * 0.8:  # If we can find a period in the last 20%
                    page_text = truncated[:last_period + 1] + "..."
                else:
                    page_text = truncated + "..."


            # 3. Use Gemini to answer the question based on the content
            prompt = f"""
            Based on the following text content from the URL {url}, please answer the user's question.
            Provide only the direct answer to the question, without any extra explanation.

            Content:
            ---
            {page_text}
            ---

            Question: {question}

            Answer:
            """

            context = QueryContext(
                user_id=None,
                session_id=None,
                query_type="web_extraction",
                domain_context="web",
                conversation_history=[],
                retrieved_documents=[],
                language="en"
            )

            gemini_response = await self.gemini_service.generate_response(
                prompt,
                context=context,
                response_type="factual"
            )

            if gemini_response.confidence_score > 0.5:
                return {
                    "status": "success",
                    "answer": gemini_response.response_text.strip()
                }
            else:
                return {
                    "status": "error",
                    "error": "LLM could not confidently answer the question from the page content."
                }

        except Exception as e:
            logger.error(f"Failed to process URL {url}: {e}")
            return {"status": "error", "error": str(e)}

web_page_processor = WebPageProcessor()
