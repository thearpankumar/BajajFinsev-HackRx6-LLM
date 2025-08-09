"""
Bilingual Query Processor
Advanced query processing with cross-lingual support and translation
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

from src.services.query_processor import QueryProcessor, ProcessedQuery, QueryContext
from src.services.translation_service import TranslationService, TranslationResult
from src.services.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


@dataclass
class BilingualQuery:
    """Bilingual query with original and translated versions"""
    original_query: str
    original_language: str
    translated_query: str
    translated_language: str
    translation_confidence: float
    primary_query: str  # The query to use for search
    secondary_query: str  # Alternative query for expanded search
    cross_lingual_strategy: str


@dataclass
class BilingualResult:
    """Bilingual search result"""
    query_info: BilingualQuery
    search_results: List[Dict[str, Any]]
    translation_metadata: Dict[str, Any]
    processing_time: float
    confidence_score: float


class BilingualQueryProcessor:
    """
    Advanced bilingual query processor supporting Malayalam-English cross-lingual search
    Handles query translation, language detection, and result optimization
    """
    
    def __init__(self):
        # Initialize core services
        self.query_processor = QueryProcessor()
        self.translation_service = TranslationService()
        self.language_detector = LanguageDetector()
        
        # Cross-lingual strategies
        self.strategies = {
            "translate_query": {
                "description": "Translate query to target language",
                "confidence_threshold": 0.7,
                "use_cases": ["different_language_content", "multilingual_search"]
            },
            "parallel_search": {
                "description": "Search in both languages simultaneously",
                "confidence_threshold": 0.5,
                "use_cases": ["mixed_content", "uncertain_language"]
            },
            "adaptive_fusion": {
                "description": "Intelligently combine results from both languages",
                "confidence_threshold": 0.6,
                "use_cases": ["comprehensive_search", "quality_optimization"]
            },
            "language_detection": {
                "description": "Auto-detect and adapt to query language",
                "confidence_threshold": 0.8,
                "use_cases": ["unknown_input_language"]
            }
        }
        
        # Performance tracking
        self.total_bilingual_queries = 0
        self.total_processing_time = 0.0
        self.strategy_usage = {strategy: 0 for strategy in self.strategies.keys()}
        
        # Configuration
        self.default_strategy = "adaptive_fusion"
        self.parallel_search_threshold = 0.5
        self.translation_quality_threshold = 0.6
        
        logger.info("BilingualQueryProcessor initialized with Malayalam-English support")
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize bilingual query processor"""
        try:
            logger.info("🔄 Initializing Bilingual Query Processor...")
            start_time = time.time()
            
            initialization_results = {}
            
            # Initialize query processor
            # Note: QueryProcessor doesn't have an initialize method in the current implementation
            initialization_results["query_processor"] = "initialized"
            
            # Initialize translation service
            translation_init = await self.translation_service.initialize()
            initialization_results["translation_service"] = translation_init
            
            initialization_time = time.time() - start_time
            
            result = {
                "status": "success",
                "message": f"Bilingual Query Processor initialized in {initialization_time:.2f}s",
                "supported_languages": ["en", "ml"],
                "available_strategies": list(self.strategies.keys()),
                "default_strategy": self.default_strategy,
                "translation_service_status": translation_init.get("status"),
                "initialization_details": initialization_results,
                "initialization_time": initialization_time
            }
            
            logger.info("✅ Bilingual Query Processor ready")
            return result
            
        except Exception as e:
            error_msg = f"Bilingual Query Processor initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }
    
    async def process_bilingual_query(
        self,
        query: str,
        target_language: str = "auto",
        strategy: str = "auto",
        context: Optional[QueryContext] = None
    ) -> BilingualResult:
        """
        Process query with bilingual support
        
        Args:
            query: Input query text
            target_language: Target language for search (en, ml, auto)
            strategy: Cross-lingual strategy to use
            context: Query context information
            
        Returns:
            BilingualResult with processed query and metadata
        """
        logger.info(f"🌐 Processing bilingual query: '{query[:50]}...'")
        start_time = time.time()
        
        try:
            # Step 1: Language detection
            detection_result = self.language_detector.detect_language(query, detailed=True)
            source_language = detection_result.get("detected_language", "en")
            detection_confidence = detection_result.get("confidence", 0.0)
            
            logger.info(f"🔍 Detected language: {source_language} (confidence: {detection_confidence:.2f})")
            
            # Step 2: Determine target language
            if target_language == "auto":
                target_language = "en" if source_language == "ml" else "ml"
            
            # Step 3: Select optimal strategy
            if strategy == "auto":
                strategy = await self._select_optimal_strategy(
                    query, source_language, target_language, detection_confidence, context
                )
            
            logger.info(f"📋 Selected strategy: {strategy}")
            
            # Step 4: Create bilingual query
            bilingual_query = await self._create_bilingual_query(
                query, source_language, target_language, strategy, context
            )
            
            # Step 5: Process queries
            search_results = await self._execute_bilingual_search(bilingual_query, context)
            
            # Step 6: Calculate confidence and metadata
            processing_time = time.time() - start_time
            confidence_score = self._calculate_bilingual_confidence(bilingual_query, search_results)
            
            translation_metadata = {
                "source_language": source_language,
                "target_language": target_language,
                "detection_confidence": detection_confidence,
                "translation_confidence": bilingual_query.translation_confidence,
                "strategy_used": strategy,
                "translation_method": getattr(bilingual_query, 'translation_method', 'unknown')
            }
            
            # Update statistics
            self.total_bilingual_queries += 1
            self.total_processing_time += processing_time
            self.strategy_usage[strategy] += 1
            
            result = BilingualResult(
                query_info=bilingual_query,
                search_results=search_results,
                translation_metadata=translation_metadata,
                processing_time=processing_time,
                confidence_score=confidence_score
            )
            
            logger.info(f"✅ Bilingual query processed in {processing_time:.2f}s "
                       f"with {len(search_results)} results (confidence: {confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            error_msg = f"Bilingual query processing failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Return error result
            return BilingualResult(
                query_info=BilingualQuery(
                    original_query=query,
                    original_language="unknown",
                    translated_query=query,
                    translated_language="unknown",
                    translation_confidence=0.0,
                    primary_query=query,
                    secondary_query="",
                    cross_lingual_strategy="error"
                ),
                search_results=[],
                translation_metadata={"error": error_msg},
                processing_time=time.time() - start_time,
                confidence_score=0.0
            )
    
    async def _select_optimal_strategy(
        self,
        query: str,
        source_language: str,
        target_language: str,
        detection_confidence: float,
        context: Optional[QueryContext]
    ) -> str:
        """Select optimal cross-lingual strategy"""
        try:
            # If same language, no translation needed
            if source_language == target_language:
                return "language_detection"
            
            # If detection confidence is low, use parallel search
            if detection_confidence < 0.6:
                logger.info("📊 Low detection confidence -> parallel_search")
                return "parallel_search"
            
            # If query is very short, translation might be unreliable
            if len(query.split()) < 3:
                logger.info("📊 Short query -> parallel_search")
                return "parallel_search"
            
            # If context suggests mixed language content
            if context and hasattr(context, 'domain_context'):
                if "multilingual" in str(context.domain_context).lower():
                    logger.info("📊 Multilingual context -> adaptive_fusion")
                    return "adaptive_fusion"
            
            # For high-confidence detection, use translation
            if detection_confidence > 0.8:
                logger.info("📊 High detection confidence -> translate_query")
                return "translate_query"
            
            # Default to adaptive fusion for balanced approach
            logger.info("📊 Default selection -> adaptive_fusion")
            return "adaptive_fusion"
            
        except Exception as e:
            logger.warning(f"Strategy selection failed: {str(e)}")
            return self.default_strategy
    
    async def _create_bilingual_query(
        self,
        query: str,
        source_language: str,
        target_language: str,
        strategy: str,
        context: Optional[QueryContext]
    ) -> BilingualQuery:
        """Create bilingual query based on strategy"""
        try:
            if strategy == "language_detection" or source_language == target_language:
                # No translation needed
                return BilingualQuery(
                    original_query=query,
                    original_language=source_language,
                    translated_query=query,
                    translated_language=source_language,
                    translation_confidence=1.0,
                    primary_query=query,
                    secondary_query="",
                    cross_lingual_strategy=strategy
                )
            
            # Translate query
            translation_result = await self.translation_service.translate(
                text=query,
                target_language=target_language,
                source_language=source_language,
                quality_assessment=True
            )
            
            if strategy == "translate_query":
                # Use translated query as primary
                primary_query = translation_result.translated_text or query
                secondary_query = query
                
            elif strategy == "parallel_search":
                # Use both queries with equal weight
                primary_query = query
                secondary_query = translation_result.translated_text or ""
                
            elif strategy == "adaptive_fusion":
                # Choose primary based on translation quality
                if translation_result.confidence_score > self.translation_quality_threshold:
                    primary_query = translation_result.translated_text
                    secondary_query = query
                else:
                    primary_query = query
                    secondary_query = translation_result.translated_text or ""
            
            else:
                # Fallback
                primary_query = query
                secondary_query = translation_result.translated_text or ""
            
            return BilingualQuery(
                original_query=query,
                original_language=source_language,
                translated_query=translation_result.translated_text or query,
                translated_language=target_language,
                translation_confidence=translation_result.confidence_score,
                primary_query=primary_query,
                secondary_query=secondary_query,
                cross_lingual_strategy=strategy
            )
            
        except Exception as e:
            logger.warning(f"Bilingual query creation failed: {str(e)}")
            # Return fallback query
            return BilingualQuery(
                original_query=query,
                original_language=source_language,
                translated_query=query,
                translated_language=source_language,
                translation_confidence=0.0,
                primary_query=query,
                secondary_query="",
                cross_lingual_strategy="fallback"
            )
    
    async def _execute_bilingual_search(
        self,
        bilingual_query: BilingualQuery,
        context: Optional[QueryContext]
    ) -> List[Dict[str, Any]]:
        """Execute search using bilingual query"""
        try:
            search_results = []
            
            # Process primary query
            if bilingual_query.primary_query:
                primary_processed = await self.query_processor.process_query(
                    bilingual_query.primary_query, context
                )
                
                search_results.append({
                    "query": bilingual_query.primary_query,
                    "language": bilingual_query.original_language,
                    "type": "primary",
                    "processed_query": primary_processed,
                    "confidence": 1.0
                })
            
            # Process secondary query if available and strategy supports it
            if (bilingual_query.secondary_query and 
                bilingual_query.cross_lingual_strategy in ["parallel_search", "adaptive_fusion"]):
                
                secondary_processed = await self.query_processor.process_query(
                    bilingual_query.secondary_query, context
                )
                
                search_results.append({
                    "query": bilingual_query.secondary_query,
                    "language": bilingual_query.translated_language,
                    "type": "secondary",
                    "processed_query": secondary_processed,
                    "confidence": bilingual_query.translation_confidence
                })
            
            # Add query expansion if translation confidence is high
            if (bilingual_query.translation_confidence > 0.8 and 
                bilingual_query.cross_lingual_strategy == "adaptive_fusion"):
                
                # Create expanded queries using both languages
                expanded_queries = await self._generate_expanded_queries(bilingual_query)
                
                for expanded_query in expanded_queries:
                    expanded_processed = await self.query_processor.process_query(
                        expanded_query["query"], context
                    )
                    
                    search_results.append({
                        "query": expanded_query["query"],
                        "language": expanded_query["language"],
                        "type": "expanded",
                        "processed_query": expanded_processed,
                        "confidence": expanded_query["confidence"]
                    })
            
            return search_results
            
        except Exception as e:
            logger.warning(f"Bilingual search execution failed: {str(e)}")
            return []
    
    async def _generate_expanded_queries(self, bilingual_query: BilingualQuery) -> List[Dict[str, str]]:
        """Generate expanded queries for better cross-lingual search"""
        try:
            expanded_queries = []
            
            # Extract keywords from both original and translated queries
            original_keywords = bilingual_query.original_query.split()
            translated_keywords = bilingual_query.translated_query.split()
            
            # Create mixed-language queries (for documents containing both languages)
            if len(original_keywords) > 1 and len(translated_keywords) > 1:
                # Mix important keywords from both languages
                mixed_query = f"{original_keywords[0]} {translated_keywords[0]}"
                expanded_queries.append({
                    "query": mixed_query,
                    "language": "mixed",
                    "confidence": 0.7
                })
            
            # Create keyword-focused queries
            if len(original_keywords) > 2:
                # Take first and last words (often most important)
                keyword_query = f"{original_keywords[0]} {original_keywords[-1]}"
                expanded_queries.append({
                    "query": keyword_query,
                    "language": bilingual_query.original_language,
                    "confidence": 0.6
                })
            
            return expanded_queries[:2]  # Limit to 2 expanded queries
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {str(e)}")
            return []
    
    def _calculate_bilingual_confidence(
        self,
        bilingual_query: BilingualQuery,
        search_results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence for bilingual processing"""
        try:
            confidence_factors = []
            
            # Translation confidence
            if bilingual_query.translation_confidence > 0:
                confidence_factors.append(bilingual_query.translation_confidence)
            
            # Strategy appropriateness
            strategy_confidence = {
                "translate_query": 0.9,
                "parallel_search": 0.7,
                "adaptive_fusion": 0.8,
                "language_detection": 1.0,
                "fallback": 0.3
            }
            confidence_factors.append(strategy_confidence.get(bilingual_query.cross_lingual_strategy, 0.5))
            
            # Search results availability
            if search_results:
                result_confidence = sum(r.get("confidence", 0.5) for r in search_results) / len(search_results)
                confidence_factors.append(result_confidence)
            else:
                confidence_factors.append(0.1)
            
            # Query processing confidence
            for result in search_results:
                if "processed_query" in result:
                    processed = result["processed_query"]
                    if hasattr(processed, 'confidence'):
                        confidence_factors.append(processed.confidence)
            
            # Calculate weighted average
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5
    
    async def translate_query_results(
        self,
        results: List[Dict[str, Any]],
        target_language: str
    ) -> List[Dict[str, Any]]:
        """Translate query results to target language"""
        try:
            translated_results = []
            
            for result in results:
                translated_result = result.copy()
                
                # Translate text fields
                if "text" in result:
                    translation = await self.translation_service.translate(
                        text=result["text"],
                        target_language=target_language,
                        quality_assessment=False
                    )
                    translated_result["translated_text"] = translation.translated_text
                    translated_result["translation_confidence"] = translation.confidence_score
                
                # Translate summary if available
                if "summary" in result:
                    summary_translation = await self.translation_service.translate(
                        text=result["summary"],
                        target_language=target_language,
                        quality_assessment=False
                    )
                    translated_result["translated_summary"] = summary_translation.translated_text
                
                translated_results.append(translated_result)
            
            return translated_results
            
        except Exception as e:
            logger.warning(f"Result translation failed: {str(e)}")
            return results  # Return original results on failure
    
    def get_bilingual_stats(self) -> Dict[str, Any]:
        """Get comprehensive bilingual processing statistics"""
        try:
            avg_processing_time = (
                self.total_processing_time / self.total_bilingual_queries
                if self.total_bilingual_queries > 0 else 0.0
            )
            
            # Get translation service stats
            translation_stats = self.translation_service.get_translation_stats()
            
            # Get query processor stats
            query_stats = self.query_processor.get_processing_stats()
            
            return {
                "bilingual_processing": {
                    "total_bilingual_queries": self.total_bilingual_queries,
                    "total_processing_time": round(self.total_processing_time, 2),
                    "average_processing_time": round(avg_processing_time, 3),
                    "strategy_usage": self.strategy_usage,
                    "default_strategy": self.default_strategy
                },
                "translation_service": translation_stats,
                "query_processor": query_stats,
                "supported_features": {
                    "cross_lingual_search": True,
                    "automatic_language_detection": True,
                    "query_translation": True,
                    "result_translation": True,
                    "parallel_search": True,
                    "adaptive_fusion": True,
                    "quality_assessment": True
                },
                "supported_languages": ["en", "ml"],
                "strategies_available": list(self.strategies.keys())
            }
            
        except Exception as e:
            logger.warning(f"Bilingual stats collection failed: {str(e)}")
            return {"error": str(e)}
    
    async def optimize_for_language_pair(self, source_lang: str, target_lang: str):
        """Optimize processing for specific language pair"""
        try:
            logger.info(f"🎯 Optimizing for {source_lang} -> {target_lang} language pair")
            
            # Update strategy preferences based on language pair
            if source_lang == "ml" and target_lang == "en":
                # Malayalam to English typically needs translation
                self.default_strategy = "translate_query"
                self.translation_quality_threshold = 0.5  # Lower threshold for Malayalam
                
            elif source_lang == "en" and target_lang == "ml":
                # English to Malayalam might need parallel search
                self.default_strategy = "adaptive_fusion"
                self.translation_quality_threshold = 0.6
                
            else:
                # Mixed or unknown languages
                self.default_strategy = "parallel_search"
                self.translation_quality_threshold = 0.7
            
            logger.info(f"✅ Optimized: default_strategy={self.default_strategy}, "
                       f"quality_threshold={self.translation_quality_threshold}")
            
        except Exception as e:
            logger.warning(f"Language pair optimization failed: {str(e)}")
    
    async def evaluate_cross_lingual_performance(
        self,
        test_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate cross-lingual performance with test queries"""
        try:
            logger.info(f"🧪 Evaluating cross-lingual performance with {len(test_queries)} test queries")
            
            evaluation_results = {
                "total_queries": len(test_queries),
                "successful_translations": 0,
                "average_translation_confidence": 0.0,
                "strategy_performance": {strategy: {"count": 0, "avg_confidence": 0.0} for strategy in self.strategies.keys()},
                "language_pair_performance": {}
            }
            
            total_confidence = 0.0
            
            for i, test_query in enumerate(test_queries):
                query_text = test_query.get("query", "")
                expected_language = test_query.get("language", "auto")
                
                if not query_text:
                    continue
                
                # Process query
                result = await self.process_bilingual_query(
                    query=query_text,
                    target_language="auto",
                    strategy="auto"
                )
                
                # Update statistics
                confidence = result.confidence_score
                total_confidence += confidence
                
                if confidence > 0.5:
                    evaluation_results["successful_translations"] += 1
                
                # Strategy performance
                strategy = result.query_info.cross_lingual_strategy
                if strategy in evaluation_results["strategy_performance"]:
                    strategy_stats = evaluation_results["strategy_performance"][strategy]
                    strategy_stats["count"] += 1
                    strategy_stats["avg_confidence"] = (
                        (strategy_stats["avg_confidence"] * (strategy_stats["count"] - 1) + confidence) / 
                        strategy_stats["count"]
                    )
                
                # Language pair performance
                lang_pair = f"{result.translation_metadata.get('source_language', 'unknown')}->{result.translation_metadata.get('target_language', 'unknown')}"
                if lang_pair not in evaluation_results["language_pair_performance"]:
                    evaluation_results["language_pair_performance"][lang_pair] = {"count": 0, "avg_confidence": 0.0}
                
                pair_stats = evaluation_results["language_pair_performance"][lang_pair]
                pair_stats["count"] += 1
                pair_stats["avg_confidence"] = (
                    (pair_stats["avg_confidence"] * (pair_stats["count"] - 1) + confidence) /
                    pair_stats["count"]
                )
            
            # Calculate averages
            if len(test_queries) > 0:
                evaluation_results["average_translation_confidence"] = total_confidence / len(test_queries)
                evaluation_results["success_rate"] = (evaluation_results["successful_translations"] / len(test_queries)) * 100
            
            logger.info(f"✅ Evaluation completed: {evaluation_results['success_rate']:.1f}% success rate")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Cross-lingual performance evaluation failed: {str(e)}")
            return {"error": str(e)}