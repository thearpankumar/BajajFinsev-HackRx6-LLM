"""
Gemini Flash Service
Google Gemini Flash integration for general query processing and fast responses
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Union

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    from google.api_core import exceptions as google_exceptions
    HAS_GOOGLE_EXCEPTIONS = True
except ImportError:
    HAS_GOOGLE_EXCEPTIONS = False

from src.core.config import config
from src.core.llm_config import llm_config_manager
from src.services.redis_cache import redis_manager

logger = logging.getLogger(__name__)


@dataclass
class GeminiResponse:
    """Gemini response with metadata"""
    response_text: str
    response_type: str
    confidence_score: float
    processing_time: float
    model_used: str
    token_count: dict[str, int]
    safety_ratings: dict[str, str]
    metadata: dict[str, Any] | None = None


@dataclass
class QueryContext:
    """Context for query processing"""
    user_id: Union[str, None]
    session_id: Union[str, None]
    query_type: str
    domain_context: Union[str, None]
    conversation_history: list[dict[str, str]]
    retrieved_documents: list[dict[str, Any]]
    language: str


class GeminiService:
    """
    Google Gemini Flash service for fast, intelligent query processing
    Optimized for quick responses and general conversational AI
    """

    def __init__(self):
        # Get proper configuration from centralized config manager
        provider, model_name, api_key = llm_config_manager.get_query_llm_config()
        
        if provider != "gemini":
            logger.warning(f"Expected Gemini provider but got {provider} for query LLM")
        
        # Configuration from centralized config
        self.api_key = api_key
        self.model_name = model_name
        self.enable_cache = config.enable_embedding_cache  # Use existing cache config
        self.max_tokens = 2048  # Reasonable default for Gemini
        self.temperature = 0.7   # Default temperature
        self.top_p = 0.9        # Default top_p

        # Model instance
        self.model = None
        self.is_initialized = False

        # Safety settings
        self.safety_settings = {
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        } if HAS_GEMINI else {}

        # Generation config
        self.generation_config = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        # Performance tracking
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.response_types = {
            'factual': 0,
            'conversational': 0,
            'analytical': 0,
            'creative': 0,
            'error': 0
        }

        # Rate limiting
        self.request_times = []
        self.rate_limit_per_minute = 60

        # Redis cache manager
        self.redis_manager = redis_manager

        logger.info("GeminiService initialized")

    async def initialize(self) -> dict[str, Any]:
        """Initialize Gemini service"""
        try:
            logger.info("🔄 Initializing Gemini Service...")
            start_time = time.time()

            if not HAS_GEMINI:
                return {
                    "status": "error",
                    "error": "Google Generative AI library not available. Install with: pip install google-generativeai"
                }

            if not self.api_key:
                return {
                    "status": "error",
                    "error": "Gemini API key not configured"
                }

            # Configure API
            genai.configure(api_key=self.api_key)

            # Initialize model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                safety_settings=self.safety_settings,
                generation_config=self.generation_config
            )

            # Test model connectivity
            test_response = await self._generate_response(
                "Hello, this is a connectivity test. Please respond with 'OK'.",
                response_type="test"
            )

            if not test_response or test_response.response_text.strip().lower() != "ok":
                logger.warning("Gemini connectivity test did not return expected response")

            # Initialize cache
            if self.enable_cache and not self.redis_manager.is_connected:
                await self.redis_manager.initialize()

            self.is_initialized = True
            initialization_time = time.time() - start_time

            result = {
                "status": "success",
                "message": f"Gemini service initialized in {initialization_time:.2f}s",
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "caching_enabled": self.enable_cache,
                "rate_limit_per_minute": self.rate_limit_per_minute,
                "initialization_time": initialization_time
            }

            logger.info(f"✅ Gemini Service ready with {self.model_name}")
            return result

        except Exception as e:
            error_msg = f"Gemini service initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }

    async def generate_response(
        self,
        query: str,
        context: Union[QueryContext, None] = None,
        response_type: str = "general",
        use_cache: bool = True
    ) -> GeminiResponse:
        """
        Generate response using Gemini Flash
        
        Args:
            query: User query text
            context: Query context with additional information
            response_type: Type of response (general, factual, conversational, etc.)
            use_cache: Whether to use cached responses
            
        Returns:
            GeminiResponse with generated text and metadata
        """
        logger.info(f"🤖 Generating Gemini response for: '{query[:50]}...'")
        start_time = time.time()

        try:
            if not self.is_initialized:
                await self.initialize()

            # Check rate limiting
            if not await self._check_rate_limit():
                return self._create_error_response(
                    query, "Rate limit exceeded. Please try again later.", start_time
                )

            # Check cache first
            if use_cache and self.enable_cache:
                cached_response = await self._get_cached_response(query, context, response_type)
                if cached_response:
                    self.cache_hits += 1
                    logger.info("✅ Gemini cache hit")
                    return cached_response

            self.cache_misses += 1

            # Prepare enhanced prompt
            enhanced_prompt = await self._prepare_enhanced_prompt(query, context, response_type)

            # Generate response
            response = await self._generate_response(enhanced_prompt, response_type)

            # Cache response if successful
            if use_cache and self.enable_cache and response.confidence_score > 0.7:
                await self._cache_response(query, context, response_type, response)

            # Update statistics
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            self.total_requests += 1
            self.total_processing_time += processing_time
            self.response_types[response_type] += 1

            logger.info(f"✅ Gemini response generated in {processing_time:.2f}s")
            return response

        except Exception as e:
            error_msg = f"Gemini response generation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return self._create_error_response(query, error_msg, start_time)

    async def _generate_response(self, prompt: str, response_type: str) -> GeminiResponse:
        """Generate response using Gemini model"""
        try:
            # Generate content
            response = self.model.generate_content(prompt)

            # Extract response text
            response_text = response.text if hasattr(response, 'text') else str(response)

            # Extract safety ratings
            safety_ratings = {}
            if hasattr(response, 'safety_ratings') and response.safety_ratings:
                for rating in response.safety_ratings:
                    category = str(rating.category).split('.')[-1]
                    probability = str(rating.probability).split('.')[-1]
                    safety_ratings[category] = probability

            # Extract token usage
            token_count = {
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0) if hasattr(response, 'usage_metadata') else 0
            }

            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(response_text, safety_ratings)

            return GeminiResponse(
                response_text=response_text,
                response_type=response_type,
                confidence_score=confidence_score,
                processing_time=0.0,  # Will be set by caller
                model_used=self.model_name,
                token_count=token_count,
                safety_ratings=safety_ratings,
                metadata={
                    "prompt_length": len(prompt),
                    "response_length": len(response_text)
                }
            )

        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            return GeminiResponse(
                response_text="I apologize, but I'm unable to generate a response at the moment. Please try again.",
                response_type="error",
                confidence_score=0.0,
                processing_time=0.0,
                model_used=self.model_name,
                token_count={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                safety_ratings={},
                metadata={"error": str(e)}
            )

    async def _prepare_enhanced_prompt(
        self,
        query: str,
        context: Union[QueryContext, None],
        response_type: str
    ) -> str:
        """Prepare enhanced prompt with context"""
        try:
            # Base prompt templates by type
            prompt_templates = {
                "factual": "Please provide a factual, accurate response to this query. Be concise and cite sources if available:\n\n",
                "conversational": "Please provide a friendly, conversational response to this query:\n\n",
                "analytical": "Please provide a detailed analytical response, breaking down the key points:\n\n",
                "creative": "Please provide a creative and engaging response to this query:\n\n",
                "general": "Please provide a helpful and informative response to this query:\n\n"
            }

            # Start with base template
            prompt = prompt_templates.get(response_type, prompt_templates["general"])

            # Add context if available
            if context:
                # Add conversation history
                if context.conversation_history:
                    prompt += "Previous conversation context:\n"
                    for turn in context.conversation_history[-3:]:  # Last 3 turns
                        if turn.get("role") == "user":
                            prompt += f"User: {turn.get('content', '')}\n"
                        elif turn.get("role") == "assistant":
                            prompt += f"Assistant: {turn.get('content', '')}\n"
                    prompt += "\n"

                # Add retrieved documents
                if context.retrieved_documents:
                    prompt += "Relevant information from knowledge base:\n"
                    for i, doc in enumerate(context.retrieved_documents[:3]):  # Top 3 docs
                        content = doc.get('content', '')[:500]  # Limit content length
                        prompt += f"{i+1}. {content}\n"
                    prompt += "\n"

                # Add domain context
                if context.domain_context:
                    prompt += f"Domain context: {context.domain_context}\n\n"

                # Add language preference
                if context.language and context.language != "en":
                    prompt += f"Please respond in {context.language} language.\n\n"

            # Add the actual query
            prompt += f"Query: {query}\n\n"

            # Add response guidelines
            prompt += "Please provide a helpful, accurate, and appropriately detailed response."

            return prompt

        except Exception as e:
            logger.warning(f"Enhanced prompt preparation failed: {str(e)}")
            return f"Please respond to this query: {query}"

    def _calculate_confidence_score(self, response_text: str, safety_ratings: dict[str, str]) -> float:
        """Calculate confidence score for the response"""
        try:
            confidence = 0.8  # Base confidence

            # Adjust based on response length and quality
            if len(response_text) < 10:
                confidence -= 0.3
            elif len(response_text) > 50:
                confidence += 0.1

            # Adjust based on safety ratings
            if safety_ratings:
                for category, probability in safety_ratings.items():
                    if probability in ['HIGH', 'MEDIUM']:
                        confidence -= 0.2

            # Check for common error patterns
            error_patterns = [
                "i apologize",
                "i'm sorry",
                "unable to",
                "cannot provide",
                "don't know"
            ]

            response_lower = response_text.lower()
            for pattern in error_patterns:
                if pattern in response_lower:
                    confidence -= 0.2
                    break

            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5

    async def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits"""
        try:
            current_time = time.time()

            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if current_time - t < 60]

            # Check if we're under the limit
            if len(self.request_times) >= self.rate_limit_per_minute:
                return False

            # Add current request time
            self.request_times.append(current_time)
            return True

        except Exception:
            return True  # Allow request if rate limit check fails

    def _create_error_response(self, query: str, error_msg: str, start_time: float) -> GeminiResponse:
        """Create error response"""
        return GeminiResponse(
            response_text="I apologize, but I encountered an error while processing your request. Please try again.",
            response_type="error",
            confidence_score=0.0,
            processing_time=time.time() - start_time,
            model_used=self.model_name,
            token_count={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            safety_ratings={},
            metadata={"error": error_msg, "original_query": query}
        )

    async def _get_cached_response(
        self,
        query: str,
        context: Union[QueryContext, None],
        response_type: str
    ) -> Union[GeminiResponse, None]:
        """Get cached response if available"""
        if not self.enable_cache or not self.redis_manager.is_connected:
            return None

        try:
            import hashlib

            # Create cache key
            context_str = ""
            if context:
                context_str = f"{context.domain_context}_{context.language}_{len(context.conversation_history)}"

            cache_key = f"gemini:{hashlib.md5(f'{query}:{response_type}:{context_str}'.encode()).hexdigest()}"
            cached_data = await self.redis_manager.get_json(cache_key)

            if cached_data:
                return GeminiResponse(**cached_data)

        except Exception as e:
            logger.warning(f"Gemini cache retrieval failed: {str(e)}")

        return None

    async def _cache_response(
        self,
        query: str,
        context: Union[QueryContext, None],
        response_type: str,
        response: GeminiResponse
    ):
        """Cache response for future use"""
        if not self.enable_cache or not self.redis_manager.is_connected:
            return

        try:
            import hashlib

            # Create cache key
            context_str = ""
            if context:
                context_str = f"{context.domain_context}_{context.language}_{len(context.conversation_history)}"

            cache_key = f"gemini:{hashlib.md5(f'{query}:{response_type}:{context_str}'.encode()).hexdigest()}"

            cache_data = {
                "response_text": response.response_text,
                "response_type": response.response_type,
                "confidence_score": response.confidence_score,
                "processing_time": response.processing_time,
                "model_used": response.model_used,
                "token_count": response.token_count,
                "safety_ratings": response.safety_ratings,
                "metadata": response.metadata,
                "cached_at": time.time()
            }

            # Cache for 1 hour for general responses, 24 hours for factual
            ttl = 86400 if response_type == "factual" else 3600
            await self.redis_manager.set_json(cache_key, cache_data, ex=ttl)

        except Exception as e:
            logger.warning(f"Gemini response caching failed: {str(e)}")

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        context: Union[QueryContext, None] = None
    ) -> GeminiResponse:
        """Multi-turn chat completion"""
        try:
            # Convert messages to a single prompt
            conversation_text = ""
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")

                if role == "user":
                    conversation_text += f"User: {content}\n"
                elif role == "assistant":
                    conversation_text += f"Assistant: {content}\n"
                elif role == "system":
                    conversation_text = f"System: {content}\n" + conversation_text

            conversation_text += "Assistant: "

            # Generate response
            return await self.generate_response(
                conversation_text,
                context=context,
                response_type="conversational"
            )

        except Exception as e:
            error_msg = f"Chat completion failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return self._create_error_response("", error_msg, time.time())

    def get_service_stats(self) -> dict[str, Any]:
        """Get comprehensive service statistics"""
        try:
            avg_processing_time = (
                self.total_processing_time / self.total_requests
                if self.total_requests > 0 else 0.0
            )

            cache_hit_rate = (
                (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            )

            return {
                "service_status": "active" if self.is_initialized else "inactive",
                "model_name": self.model_name,
                "total_requests": self.total_requests,
                "total_processing_time": round(self.total_processing_time, 2),
                "average_processing_time": round(avg_processing_time, 3),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "response_type_distribution": self.response_types,
                "rate_limit_per_minute": self.rate_limit_per_minute,
                "recent_request_count": len(self.request_times),
                "configuration": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "caching_enabled": self.enable_cache
                }
            }

        except Exception as e:
            logger.warning(f"Gemini service stats collection failed: {str(e)}")
            return {"error": str(e)}
