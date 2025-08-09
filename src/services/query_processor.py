"""
Advanced Query Processor
Handles query preprocessing, optimization, and result post-processing for RAG pipeline
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Union

from src.core.config import config
from src.services.language_detector import LanguageDetector
from src.services.redis_cache import redis_manager

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Data class for processed queries"""
    original_query: str
    processed_query: str
    query_type: str  # factual, conversational, analytical, etc.
    language: str
    keywords: list[str]
    entities: list[str]
    intent: str
    confidence: float
    preprocessing_time: float


@dataclass
class QueryContext:
    """Data class for query context"""
    user_id: Union[str, None] = None
    session_id: Union[str, None] = None
    conversation_history: list[str] = None
    domain_context: Union[str, None] = None
    preferred_language: str = "en"
    result_preferences: dict[str, Any] | None = None


class QueryProcessor:
    """
    Advanced query processor with language detection, intent analysis, and optimization
    """

    def __init__(self):
        # Language detector for cross-lingual support
        self.language_detector = LanguageDetector()

        # Redis cache for query processing
        self.redis_manager = redis_manager
        self.enable_cache = config.enable_embedding_cache

        # Query processing patterns
        self._init_processing_patterns()

        # Performance tracking
        self.total_queries_processed = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info("QueryProcessor initialized with cross-lingual support")

    def _init_processing_patterns(self):
        """Initialize query processing patterns"""

        # Question types
        self.question_patterns = {
            "what": r"\b(what|എന്താണ്|क्या)\b",
            "how": r"\b(how|എങ്ങനെ|कैसे)\b",
            "why": r"\b(why|എന്തുകൊണ്ട്|क्यों)\b",
            "when": r"\b(when|എപ്പോൾ|कब)\b",
            "where": r"\b(where|എവിടെ|कहाँ)\b",
            "who": r"\b(who|ആര്|कौन)\b"
        }

        # Intent patterns
        self.intent_patterns = {
            "search": r"\b(find|search|look for|കണ്ടെത്തുക|खोजना)\b",
            "explain": r"\b(explain|describe|tell me|വിശദീകരിക്കുക|समझाना)\b",
            "compare": r"\b(compare|difference|versus|तुलना|താരതമ്യം)\b",
            "list": r"\b(list|enumerate|show me|सूची|പട്ടിക)\b",
            "define": r"\b(define|meaning|definition|അർത്ഥം|परिभाषा)\b",
            "calculate": r"\b(calculate|compute|കണക്കാക്കുക|गणना)\b"
        }

        # Entity extraction patterns
        self.entity_patterns = {
            "number": r"\b\d+(?:\.\d+)?\b",
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "email": r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "url": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "currency": r"\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:rupees|dollars|₹|\$)"
        }

    async def process_query(
        self,
        query: str,
        context: Union[QueryContext, None] = None
    ) -> ProcessedQuery:
        """
        Process and analyze query for optimal retrieval
        
        Args:
            query: Raw query text
            context: Optional query context
            
        Returns:
            ProcessedQuery with analysis and preprocessing results
        """
        logger.info(f"🔄 Processing query: '{query[:100]}...'")
        start_time = time.time()

        try:
            # Check cache first
            if self.enable_cache:
                cached_result = await self._get_cached_query_processing(query)
                if cached_result:
                    self.cache_hits += 1
                    logger.debug("✅ Query processing cache hit")
                    return cached_result
                self.cache_misses += 1

            # Language detection
            language_info = await self._detect_query_language(query)

            # Query preprocessing
            processed_text = self._preprocess_query_text(query, language_info)

            # Query type classification
            query_type = self._classify_query_type(processed_text, language_info)

            # Intent analysis
            intent, intent_confidence = self._analyze_intent(processed_text, language_info)

            # Keyword extraction
            keywords = self._extract_keywords(processed_text, language_info)

            # Entity extraction
            entities = self._extract_entities(processed_text)

            # Query optimization
            optimized_query = await self._optimize_query(
                processed_text, query_type, intent, context
            )

            processing_time = time.time() - start_time
            self.total_queries_processed += 1
            self.total_processing_time += processing_time

            # Create result
            result = ProcessedQuery(
                original_query=query,
                processed_query=optimized_query,
                query_type=query_type,
                language=language_info["primary_language"],
                keywords=keywords,
                entities=entities,
                intent=intent,
                confidence=intent_confidence,
                preprocessing_time=round(processing_time, 4)
            )

            # Cache result
            if self.enable_cache:
                await self._cache_query_processing(query, result)

            logger.info(f"✅ Query processed in {processing_time:.4f}s: "
                       f"Type={query_type}, Intent={intent}, Language={language_info['primary_language']}")

            return result

        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            logger.error(f"❌ {error_msg}")

            # Return basic processed query on error
            return ProcessedQuery(
                original_query=query,
                processed_query=query,
                query_type="unknown",
                language="en",
                keywords=[],
                entities=[],
                intent="search",
                confidence=0.0,
                preprocessing_time=time.time() - start_time
            )

    async def _detect_query_language(self, query: str) -> dict[str, Any]:
        """Detect query language"""
        try:
            return self.language_detector.detect_language(query, detailed=True)
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return {
                "primary_language": "en",
                "confidence": 0.0,
                "error": str(e)
            }

    def _preprocess_query_text(self, query: str, language_info: dict[str, Any]) -> str:
        """Preprocess query text"""
        try:
            # Normalize unicode
            text = self.language_detector.normalize_text_encoding(query)

            # Language-specific preprocessing
            if language_info["primary_language"] == "ml":
                text = self._preprocess_malayalam_query(text)
            elif language_info["primary_language"] == "en":
                text = self._preprocess_english_query(text)

            # Common preprocessing
            text = self._common_query_preprocessing(text)

            return text

        except Exception as e:
            logger.warning(f"Query preprocessing failed: {str(e)}")
            return query

    def _preprocess_malayalam_query(self, text: str) -> str:
        """Malayalam-specific query preprocessing"""
        # Normalize Malayalam punctuation
        text = text.replace('।', '.')
        text = text.replace('॥', '.')

        # Remove common Malayalam stop words (basic set)
        malayalam_stopwords = ['എന്ന്', 'ഇത്', 'അത്', 'ഈ', 'ആ', 'ഒരു']
        words = text.split()
        filtered_words = [w for w in words if w not in malayalam_stopwords]

        return ' '.join(filtered_words)

    def _preprocess_english_query(self, text: str) -> str:
        """English-specific query preprocessing"""
        # Convert to lowercase
        text = text.lower()

        # Remove common English stop words (basic set)
        english_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        words = text.split()
        filtered_words = [w for w in words if w not in english_stopwords and len(w) > 2]

        return ' '.join(filtered_words)

    def _common_query_preprocessing(self, text: str) -> str:
        """Common preprocessing for all languages"""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\?\!\.\,\-]', ' ', text)

        # Normalize punctuation
        text = re.sub(r'[.!?]+', '.', text)
        text = re.sub(r'[,]+', ',', text)

        return text.strip()

    def _classify_query_type(self, query: str, language_info: dict[str, Any]) -> str:
        """Classify query type based on patterns"""
        query_lower = query.lower()

        # Check for question patterns
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                return f"question_{q_type}"

        # Check for specific query structures
        if '?' in query:
            return "question_general"
        elif re.search(r'\b(calculate|compute)\b', query_lower):
            return "computational"
        elif re.search(r'\b(compare|versus|vs)\b', query_lower):
            return "comparison"
        elif re.search(r'\b(list|show|enumerate)\b', query_lower):
            return "enumeration"
        else:
            return "factual"

    def _analyze_intent(self, query: str, language_info: dict[str, Any]) -> tuple[str, float]:
        """Analyze query intent"""
        query_lower = query.lower()
        intent_scores = {}

        # Calculate scores for each intent
        for intent, pattern in self.intent_patterns.items():
            matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
            if matches > 0:
                intent_scores[intent] = matches

        if intent_scores:
            # Get intent with highest score
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(1.0, intent_scores[best_intent] * 0.3)  # Basic confidence scoring
            return best_intent, confidence

        # Default intent
        return "search", 0.5

    def _extract_keywords(self, query: str, language_info: dict[str, Any]) -> list[str]:
        """Extract keywords from query"""
        try:
            # Simple keyword extraction
            words = query.split()

            # Filter by word length and remove common words
            keywords = []
            for word in words:
                word_clean = re.sub(r'[^\w]', '', word)
                if len(word_clean) > 2 and word_clean.isalpha():
                    keywords.append(word_clean)

            # Remove duplicates while preserving order
            unique_keywords = []
            seen = set()
            for keyword in keywords:
                if keyword.lower() not in seen:
                    unique_keywords.append(keyword)
                    seen.add(keyword.lower())

            return unique_keywords[:10]  # Limit to top 10 keywords

        except Exception as e:
            logger.warning(f"Keyword extraction failed: {str(e)}")
            return []

    def _extract_entities(self, query: str) -> list[str]:
        """Extract entities from query"""
        entities = []

        try:
            for entity_type, pattern in self.entity_patterns.items():
                matches = re.findall(pattern, query, re.IGNORECASE)
                for match in matches:
                    entities.append({
                        "text": match,
                        "type": entity_type
                    })

            # Return just the entity texts
            return [e["text"] for e in entities]

        except Exception as e:
            logger.warning(f"Entity extraction failed: {str(e)}")
            return []

    async def _optimize_query(
        self,
        query: str,
        query_type: str,
        intent: str,
        context: Union[QueryContext, None]
    ) -> str:
        """Optimize query for better retrieval"""
        try:
            optimized = query

            # Add context-based optimization
            if context and context.domain_context:
                # Could add domain-specific terms or modifications
                pass

            # Query expansion based on intent
            if intent == "compare":
                # Add comparison-related terms
                comparison_terms = ["difference", "versus", "comparison", "contrast"]
                for term in comparison_terms:
                    if term not in optimized.lower():
                        optimized += f" {term}"

            elif intent == "define":
                # Add definition-related terms
                definition_terms = ["meaning", "definition", "explanation"]
                for term in definition_terms:
                    if term not in optimized.lower():
                        optimized += f" {term}"

            # Ensure query isn't too long
            words = optimized.split()
            if len(words) > 20:
                optimized = ' '.join(words[:20])

            return optimized

        except Exception as e:
            logger.warning(f"Query optimization failed: {str(e)}")
            return query

    async def _get_cached_query_processing(self, query: str) -> Union[ProcessedQuery, None]:
        """Get cached query processing result"""
        if not self.redis_manager.is_connected:
            return None

        try:
            import hashlib
            cache_key = f"query_proc:{hashlib.md5(query.encode()).hexdigest()}"
            cached_data = await self.redis_manager.get_json(cache_key)

            if cached_data:
                return ProcessedQuery(**cached_data)

        except Exception as e:
            logger.warning(f"Query processing cache retrieval failed: {str(e)}")

        return None

    async def _cache_query_processing(self, query: str, result: ProcessedQuery):
        """Cache query processing result"""
        if not self.redis_manager.is_connected:
            return

        try:
            import hashlib
            cache_key = f"query_proc:{hashlib.md5(query.encode()).hexdigest()}"

            # Convert to dict for caching
            cache_data = {
                "original_query": result.original_query,
                "processed_query": result.processed_query,
                "query_type": result.query_type,
                "language": result.language,
                "keywords": result.keywords,
                "entities": result.entities,
                "intent": result.intent,
                "confidence": result.confidence,
                "preprocessing_time": result.preprocessing_time
            }

            # Cache for 1 hour
            await self.redis_manager.set_json(cache_key, cache_data, ex=3600)

        except Exception as e:
            logger.warning(f"Query processing caching failed: {str(e)}")

    def get_processing_stats(self) -> dict[str, Any]:
        """Get query processing statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_queries_processed
            if self.total_queries_processed > 0 else 0.0
        )

        cache_hit_rate = (
            (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100
            if (self.cache_hits + self.cache_misses) > 0 else 0.0
        )

        return {
            "total_queries_processed": self.total_queries_processed,
            "total_processing_time": round(self.total_processing_time, 3),
            "average_processing_time": round(avg_processing_time, 5),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "supported_languages": ["en", "ml", "hi"],
            "supported_intents": list(self.intent_patterns.keys()),
            "supported_query_types": list(self.question_patterns.keys()) + ["factual", "computational", "comparison"],
            "entity_types": list(self.entity_patterns.keys())
        }
