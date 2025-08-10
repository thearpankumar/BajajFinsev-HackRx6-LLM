"""
Retrieval Orchestrator
Coordinates retrieval strategies, result ranking, and response formatting
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Union

from src.core.config import config
from src.core.integrated_rag_pipeline import IntegratedRAGPipeline, RAGQuery, RAGResult
from src.services.query_processor import ProcessedQuery, QueryContext, QueryProcessor
from src.services.legal_query_processor import DomainQueryProcessor
from src.services.gemini_query_enhancer import gemini_query_enhancer, EnhancedQuery
from src.services.bilingual_query_processor import BilingualQueryProcessor
from src.services.translation_service import TranslationService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalStrategy:
    """Data class for retrieval strategies"""
    name: str
    weight: float
    enabled: bool
    parameters: dict[str, Any]


@dataclass
class RankedResult:
    """Data class for ranked retrieval results"""
    chunk_id: str
    text: str
    score: float
    ranking_score: float
    metadata: dict[str, Any]
    source_url: str
    relevance_explanation: str


@dataclass
class FormattedResponse:
    """Data class for formatted responses"""
    query_id: str
    original_query: str
    processed_query: str
    ranked_results: list[RankedResult]
    total_results: int
    retrieval_time: float
    processing_metadata: dict[str, Any]
    response_summary: str
    confidence_score: float


class RetrievalOrchestrator:
    """
    Advanced retrieval orchestrator that coordinates different retrieval strategies,
    ranks results, and formats comprehensive responses
    """

    def __init__(self, rag_pipeline: IntegratedRAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.query_processor = QueryProcessor()
        self.domain_processor = DomainQueryProcessor()
        
        # Multilingual support components
        self.bilingual_processor = BilingualQueryProcessor()
        self.translation_service = TranslationService()

        # Retrieval strategies configuration with bilingual support
        self.strategies = {
            "semantic_similarity": RetrievalStrategy(
                name="semantic_similarity",
                weight=0.7,
                enabled=True,
                parameters={"similarity_threshold": 0.6}
            ),
            "keyword_matching": RetrievalStrategy(
                name="keyword_matching",
                weight=0.2,
                enabled=True,
                parameters={"keyword_boost": 1.5}
            ),
            "metadata_filtering": RetrievalStrategy(
                name="metadata_filtering",
                weight=0.1,
                enabled=True,
                parameters={"filter_boost": 1.2}
            ),
            # Bilingual strategy variants (dynamically added)
            "semantic_similarity_bilingual": RetrievalStrategy(
                name="semantic_similarity_bilingual",
                weight=0.5,  # Slightly lower weight for cross-language matches
                enabled=True,
                parameters={"similarity_threshold": 0.5, "cross_language_boost": 1.1}
            ),
            "keyword_matching_bilingual": RetrievalStrategy(
                name="keyword_matching_bilingual", 
                weight=0.15,  # Lower weight for cross-language keyword matches
                enabled=True,
                parameters={"keyword_boost": 1.3, "cross_language_boost": 1.0}
            ),
            "semantic_similarity_secondary": RetrievalStrategy(
                name="semantic_similarity_secondary",
                weight=0.3,  # Lower weight for secondary language matches
                enabled=True,
                parameters={"similarity_threshold": 0.4, "secondary_language_boost": 0.9}
            )
        }

        # Performance tracking
        self.total_retrievals = 0
        self.total_retrieval_time = 0.0
        self.total_ranking_time = 0.0

        # Configuration with proper config access instead of getattr
        self.max_results = config.max_retrieval_results if hasattr(config, 'max_retrieval_results') else 20
        self.min_confidence_threshold = config.min_confidence_threshold if hasattr(config, 'min_confidence_threshold') else 0.3
        self.enable_reranking = config.enable_reranking if hasattr(config, 'enable_reranking') else True

        logger.info("RetrievalOrchestrator initialized with advanced ranking")

    async def initialize(self) -> dict[str, Any]:
        """Initialize the retrieval orchestrator and all its components"""
        try:
            logger.info("🔄 Initializing Retrieval Orchestrator...")
            start_time = time.time()
            
            # Initialize multilingual components
            logger.info("🌐 Initializing translation service...")
            translation_result = await self.translation_service.initialize()
            
            logger.info("🔄 Initializing bilingual query processor...")
            bilingual_result = await self.bilingual_processor.initialize()
            
            if translation_result.get("status") != "success":
                logger.warning(f"⚠️ Translation service initialization failed: {translation_result.get('error')}")
            
            if bilingual_result.get("status") != "success":
                logger.warning(f"⚠️ Bilingual processor initialization failed: {bilingual_result.get('error')}")
            
            # Initialize Gemini query enhancer
            gemini_result = await gemini_query_enhancer.initialize()
            
            if gemini_result["status"] != "success":
                logger.warning(f"⚠️ Gemini query enhancement initialization failed: {gemini_result.get('error')}")
                logger.info("📋 Continuing with rule-based query enhancement only")
            
            initialization_time = time.time() - start_time
            
            result = {
                "status": "success",
                "message": f"Retrieval Orchestrator initialized in {initialization_time:.2f}s",
                "components": {
                    "query_processor": "initialized",
                    "domain_processor": "initialized", 
                    "translation_service": translation_result.get("status", "unknown"),
                    "bilingual_processor": bilingual_result.get("status", "unknown"),
                    "gemini_enhancer": gemini_result["status"],
                    "rag_pipeline": "external"
                },
                "retrieval_strategies": list(self.strategies.keys()),
                "multilingual_support": {
                    "translation_enabled": translation_result.get("status") == "success",
                    "bilingual_processing": bilingual_result.get("status") == "success",
                    "supported_languages": ["en", "ml"] if translation_result.get("status") == "success" else ["en"]
                },
                "gemini_enhancement": gemini_result if gemini_result["status"] == "success" else None,
                "initialization_time": initialization_time
            }
            
            logger.info("✅ Retrieval Orchestrator ready with Gemini query enhancement")
            return result
            
        except Exception as e:
            error_msg = f"Retrieval Orchestrator initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}

    async def retrieve_and_rank(
        self,
        query: str,
        max_results: Union[int, None] = None,
        context: Union[QueryContext, None] = None,
        strategies: list[str] | None = None
    ) -> FormattedResponse:
        """
        Execute comprehensive retrieval with ranking and formatting
        
        Args:
            query: User query text
            max_results: Maximum number of results to return
            context: Optional query context
            strategies: Optional list of strategies to use
            
        Returns:
            FormattedResponse with ranked and formatted results
        """
        logger.info(f"🎯 Starting orchestrated retrieval for: '{query[:100]}...'")
        start_time = time.time()

        try:
            # Step 1: Process query with domain awareness
            logger.info("🔄 Step 1: Processing query with domain awareness...")
            processed_query = await self.query_processor.process_query(query, context)
            
            # Step 1.2: Bilingual query processing (Malayalam-English support)
            logger.info("🌐 Step 1.2: Processing bilingual query...")
            try:
                bilingual_result = await self.bilingual_processor.process_bilingual_query(
                    query=query,
                    target_language="en",  # Always translate to English for processing
                    strategy="auto",
                    context=context
                )
                
                # If bilingual processing was successful and produced a translation
                if (bilingual_result and 
                    bilingual_result.confidence_score > 0.3 and
                    bilingual_result.query_info.primary_query != query):
                    
                    logger.info(f"🔤 Using bilingual query: '{query[:50]}...' -> '{bilingual_result.query_info.primary_query[:50]}...'")
                    
                    # Update processed query with the bilingual result
                    processed_query.processed_query = bilingual_result.query_info.primary_query
                    processed_query.metadata = processed_query.metadata or {}
                    processed_query.metadata.update({
                        "original_query": query,
                        "bilingual_processing": {
                            "source_language": bilingual_result.translation_metadata.get("source_language"),
                            "target_language": bilingual_result.translation_metadata.get("target_language"),
                            "translation_confidence": bilingual_result.translation_metadata.get("translation_confidence"),
                            "strategy_used": bilingual_result.translation_metadata.get("strategy_used"),
                            "processing_time": bilingual_result.processing_time,
                            "confidence_score": bilingual_result.confidence_score
                        }
                    })
                    
                    # If there's a secondary query, add it to keywords for expanded search
                    if bilingual_result.query_info.secondary_query:
                        secondary_keywords = bilingual_result.query_info.secondary_query.split()[:3]  # Take first 3 words
                        processed_query.keywords = list(processed_query.keywords) + secondary_keywords
                        
                else:
                    logger.info("🌐 Bilingual processing completed - using original query")
                    
            except Exception as e:
                logger.warning(f"⚠️ Bilingual processing failed: {str(e)} - continuing with original query")
            
            # Step 1.3: Detect query domain for context-aware enhancement
            logger.info("🧠 Step 1.3: Enhancing query with domain-specific context...")
            domain_enhancement = self.domain_processor.enhance_domain_query(query)
            detected_domain = domain_enhancement["detected_domain"]
            
            logger.info(f"🎯 Detected domain: {detected_domain}")
            
            # Step 1.5: Use Gemini for LLM-based query enhancement with domain keywords
            gemini_enhancement_data = None
            final_enhanced_query = domain_enhancement["enhanced_query"]  # Default fallback
            
            # Try Gemini enhancement if initialized
            if gemini_query_enhancer.is_initialized:
                try:
                    logger.info("🤖 Step 1.5: Enhancing query with Gemini LLM...")
                    gemini_enhanced_query = await gemini_query_enhancer.enhance_query(
                        query=query,
                        detected_domain=detected_domain,
                        context_info={
                            "processed_query": processed_query,
                            "domain_enhancement": domain_enhancement
                        }
                    )
                    
                    # Use Gemini result if confidence is good
                    if gemini_enhanced_query.confidence > 0.3:
                        logger.info(f"✨ Gemini enhanced query: {gemini_enhanced_query.enhanced_query[:100]}...")
                        logger.info(f"📎 Added keywords: {gemini_enhanced_query.added_keywords[:5]}")
                        final_enhanced_query = gemini_enhanced_query.enhanced_query
                        gemini_enhancement_data = {
                            "enhanced_query": gemini_enhanced_query.enhanced_query,
                            "added_keywords": gemini_enhanced_query.added_keywords,
                            "confidence": gemini_enhanced_query.confidence,
                            "processing_time": gemini_enhanced_query.processing_time
                        }
                    else:
                        logger.warning(f"⚠️ Low confidence Gemini enhancement ({gemini_enhanced_query.confidence:.2f}), using rule-based fallback")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Gemini query enhancement failed: {str(e)}, using rule-based fallback")
            else:
                logger.info("📋 Gemini not available, using rule-based query enhancement")
            
            # Merge all enhancement information
            processed_query.processed_query = final_enhanced_query
            
            # Combine keywords from all sources
            all_keywords = list(processed_query.keywords)
            all_keywords.extend(domain_enhancement["expanded_terms"][:5])  # Add top 5 domain terms
            if gemini_enhancement_data and gemini_enhancement_data["added_keywords"]:
                all_keywords.extend(gemini_enhancement_data["added_keywords"][:5])  # Add Gemini keywords
            processed_query.keywords = list(dict.fromkeys(all_keywords))  # Remove duplicates while preserving order
            
            processed_query.metadata = {
                "domain_info": domain_enhancement,
                "detected_domain": detected_domain,
                "gemini_enhancement": gemini_enhancement_data
            }
            
            logger.info(f"📝 Final enhanced query: {processed_query.processed_query[:100]}...")

            # Step 2: Execute retrieval strategies
            logger.info("🔍 Step 2: Executing retrieval strategies...")
            retrieval_results = await self._execute_retrieval_strategies(
                processed_query, max_results or self.max_results, strategies
            )

            if not retrieval_results:
                return self._create_empty_response(query, processed_query, start_time)

            # Step 3: Rank and merge results
            logger.info("📊 Step 3: Ranking and merging results...")
            ranked_results = await self._rank_and_merge_results(
                retrieval_results, processed_query
            )

            # Step 4: Apply final filtering and formatting
            logger.info("✨ Step 4: Final filtering and formatting...")
            final_results = self._apply_final_filtering(ranked_results)

            # Step 5: Generate response summary
            response_summary = self._generate_response_summary(
                processed_query, final_results
            )

            # Step 6: Calculate confidence score
            confidence_score = self._calculate_overall_confidence(final_results, processed_query)

            total_time = time.time() - start_time
            self.total_retrievals += 1
            self.total_retrieval_time += total_time

            # Create formatted response
            response = FormattedResponse(
                query_id=f"retr_{int(time.time())}",
                original_query=query,
                processed_query=processed_query.processed_query,
                ranked_results=final_results,
                total_results=len(final_results),
                retrieval_time=round(total_time, 3),
                processing_metadata={
                    "query_processing": {
                        "language": processed_query.language,
                        "intent": processed_query.intent,
                        "query_type": processed_query.query_type,
                        "keywords": processed_query.keywords,
                        "entities": processed_query.entities
                    },
                    "retrieval_strategies": [s.name for s in self.strategies.values() if s.enabled],
                    "ranking_applied": self.enable_reranking,
                    "filters_applied": len([s for s in self.strategies.values() if s.enabled and "filter" in s.name]),
                    "detected_domain": processed_query.metadata.get("detected_domain", "general") if processed_query.metadata else "general",
                    "domain_info": processed_query.metadata.get("domain_info", {}) if processed_query.metadata else {}
                },
                response_summary=response_summary,
                confidence_score=confidence_score
            )

            logger.info(f"✅ Orchestrated retrieval completed in {total_time:.3f}s: "
                       f"{len(final_results)} results, confidence={confidence_score:.2f}")

            return response

        except Exception as e:
            error_msg = f"Orchestrated retrieval failed: {str(e)}"
            logger.error(f"❌ {error_msg}")

            return FormattedResponse(
                query_id="error",
                original_query=query,
                processed_query=query,
                ranked_results=[],
                total_results=0,
                retrieval_time=time.time() - start_time,
                processing_metadata={"error": error_msg},
                response_summary="Retrieval failed due to an error.",
                confidence_score=0.0
            )

    async def _execute_retrieval_strategies(
        self,
        processed_query: ProcessedQuery,
        max_results: int,
        strategies: list[str] | None
    ) -> list[tuple[str, RAGResult]]:
        """Execute enabled retrieval strategies with bilingual search support"""
        results = []

        # Filter strategies to use
        active_strategies = [
            (name, strategy) for name, strategy in self.strategies.items()
            if strategy.enabled and (not strategies or name in strategies)
        ]

        for strategy_name, strategy in active_strategies:
            try:
                logger.debug(f"🔄 Executing strategy: {strategy_name}")

                # Create RAG query based on strategy
                rag_query = self._create_rag_query_for_strategy(
                    processed_query, strategy, max_results
                )

                # Execute primary retrieval
                retrieval_result = await self.rag_pipeline.query(rag_query)

                if retrieval_result.total_results > 0:
                    results.append((strategy_name, retrieval_result))
                    logger.debug(f"✅ Strategy {strategy_name}: {retrieval_result.total_results} results")
                else:
                    logger.debug(f"⚠️ Strategy {strategy_name}: No results")

                # Enhanced Bilingual Cross-Language Search
                # If we have bilingual processing capability and the original query had translation data
                if (hasattr(processed_query, 'metadata') and 
                    processed_query.metadata and 
                    processed_query.metadata.get("bilingual_processing") and
                    self.translation_service):
                    
                    try:
                        bilingual_data = processed_query.metadata["bilingual_processing"]
                        original_query = processed_query.metadata.get("original_query", "")
                        
                        # If query was translated, also search with original language version
                        if (original_query and 
                            original_query != processed_query.processed_query and
                            bilingual_data.get("translation_confidence", 0) > 0.3):
                            
                            logger.debug(f"🌐 Executing bilingual search for strategy: {strategy_name}")
                            
                            # Create bilingual RAG query using original language
                            bilingual_rag_query = self._create_rag_query_for_strategy(
                                processed_query, strategy, max_results // 2  # Limit results to avoid duplication
                            )
                            # Use original query for bilingual search
                            bilingual_rag_query.query_text = original_query
                            
                            # Execute bilingual retrieval
                            bilingual_result = await self.rag_pipeline.query(bilingual_rag_query)
                            
                            if bilingual_result.total_results > 0:
                                # Add bilingual results as a separate strategy variant
                                bilingual_strategy_name = f"{strategy_name}_bilingual"
                                results.append((bilingual_strategy_name, bilingual_result))
                                logger.debug(f"🔤 Bilingual strategy {bilingual_strategy_name}: {bilingual_result.total_results} results")
                                
                        # Additional cross-language keyword search if we have secondary queries
                        secondary_query = ""
                        if hasattr(processed_query, 'metadata') and processed_query.metadata:
                            secondary_query = processed_query.metadata.get("secondary_query", "")
                        
                        if secondary_query and secondary_query != processed_query.processed_query:
                            logger.debug(f"🔍 Executing secondary language search for strategy: {strategy_name}")
                            
                            # Create secondary language RAG query
                            secondary_rag_query = self._create_rag_query_for_strategy(
                                processed_query, strategy, max_results // 3
                            )
                            secondary_rag_query.query_text = secondary_query
                            
                            # Execute secondary language retrieval
                            secondary_result = await self.rag_pipeline.query(secondary_rag_query)
                            
                            if secondary_result.total_results > 0:
                                # Add secondary results as a separate strategy variant
                                secondary_strategy_name = f"{strategy_name}_secondary"
                                results.append((secondary_strategy_name, secondary_result))
                                logger.debug(f"🔎 Secondary strategy {secondary_strategy_name}: {secondary_result.total_results} results")
                                
                    except Exception as e:
                        logger.warning(f"Bilingual search for strategy {strategy_name} failed: {str(e)}")
                        continue

            except Exception as e:
                logger.warning(f"Strategy {strategy_name} failed: {str(e)}")
                continue

        return results

    def _create_rag_query_for_strategy(
        self,
        processed_query: ProcessedQuery,
        strategy: RetrievalStrategy,
        max_results: int
    ) -> RAGQuery:
        """Create RAG query optimized for specific strategy"""

        if strategy.name == "semantic_similarity":
            # Use processed query for semantic similarity
            query_text = processed_query.processed_query

        elif strategy.name == "keyword_matching":
            # Use keywords for keyword-based matching
            if processed_query.keywords:
                query_text = " ".join(processed_query.keywords)
            else:
                query_text = processed_query.processed_query

        else:
            # Default to processed query
            query_text = processed_query.processed_query

        # Create filter metadata based on strategy
        filter_metadata = None
        if strategy.name == "metadata_filtering":
            filter_metadata = {
                "language": processed_query.language
            }

            # Add intent-based filtering
            if processed_query.intent in ["define", "explain"]:
                filter_metadata["chunk_type"] = "paragraph"

        return RAGQuery(
            query_text=query_text,
            max_results=max_results,
            filter_metadata=filter_metadata,
            retrieval_strategy=strategy.name
        )

    async def _rank_and_merge_results(
        self,
        retrieval_results: list[tuple[str, RAGResult]],
        processed_query: ProcessedQuery
    ) -> list[RankedResult]:
        """Rank and merge results from different strategies"""
        start_time = time.time()

        try:
            # Collect all unique chunks with strategy scores
            chunk_scores = {}  # chunk_id -> {strategy_name: score, metadata: ..., text: ...}

            for strategy_name, rag_result in retrieval_results:
                strategy_weight = self.strategies[strategy_name].weight

                for chunk_data in rag_result.retrieved_chunks:
                    chunk_id = chunk_data["chunk_id"]
                    similarity_score = chunk_data["score"]

                    if chunk_id not in chunk_scores:
                        chunk_scores[chunk_id] = {
                            "strategies": {},
                            "text": chunk_data["text"],
                            "metadata": chunk_data["metadata"],
                            "source_url": chunk_data.get("source_url", ""),
                            "max_score": 0.0
                        }

                    # Apply strategy-specific scoring
                    weighted_score = similarity_score * strategy_weight

                    # Apply strategy-specific boosts
                    if strategy_name == "keyword_matching":
                        keyword_boost = self._calculate_keyword_boost(
                            chunk_data["text"], processed_query.keywords
                        )
                        weighted_score *= keyword_boost

                    chunk_scores[chunk_id]["strategies"][strategy_name] = weighted_score
                    chunk_scores[chunk_id]["max_score"] = max(
                        chunk_scores[chunk_id]["max_score"],
                        similarity_score
                    )

            # Calculate final ranking scores
            ranked_results = []
            for chunk_id, chunk_data in chunk_scores.items():
                # Combine scores from different strategies
                final_score = sum(chunk_data["strategies"].values())

                # Apply additional ranking factors
                ranking_score = self._calculate_ranking_score(
                    chunk_data, processed_query, final_score
                )

                # Generate relevance explanation
                relevance_explanation = self._generate_relevance_explanation(
                    chunk_data, processed_query
                )

                ranked_result = RankedResult(
                    chunk_id=chunk_id,
                    text=chunk_data["text"],
                    score=chunk_data["max_score"],
                    ranking_score=ranking_score,
                    metadata=chunk_data["metadata"],
                    source_url=chunk_data["source_url"],
                    relevance_explanation=relevance_explanation
                )

                ranked_results.append(ranked_result)

            # Sort by ranking score
            ranked_results.sort(key=lambda x: x.ranking_score, reverse=True)

            ranking_time = time.time() - start_time
            self.total_ranking_time += ranking_time

            logger.debug(f"🏆 Ranked {len(ranked_results)} unique results in {ranking_time:.3f}s")

            return ranked_results

        except Exception as e:
            logger.error(f"Result ranking failed: {str(e)}")
            return []

    def _calculate_keyword_boost(self, text: str, keywords: list[str]) -> float:
        """Calculate keyword matching boost"""
        if not keywords:
            return 1.0

        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)

        if matches == 0:
            return 0.8  # Slight penalty for no keyword matches

        # Boost based on keyword match ratio
        match_ratio = matches / len(keywords)
        return 1.0 + (match_ratio * 0.5)  # Up to 50% boost

    def _calculate_ranking_score(
        self,
        chunk_data: dict[str, Any],
        processed_query: ProcessedQuery,
        base_score: float
    ) -> float:
        """Calculate final ranking score with various factors including bilingual support"""
        ranking_score = base_score

        # Chunk type preference based on query intent
        chunk_type = chunk_data["metadata"].get("chunk_type", "paragraph")
        if processed_query.intent == "define" and chunk_type == "section":
            ranking_score *= 1.2  # Boost sections for definitions
        elif processed_query.intent == "list" and "table" in chunk_type.lower():
            ranking_score *= 1.3  # Boost tables for lists

        # Enhanced Language matching and cross-language support
        chunk_language = chunk_data["metadata"].get("language", "unknown")
        original_language = chunk_data["metadata"].get("original_language")
        is_translated = chunk_data["metadata"].get("is_translated", False)
        
        # Handle bilingual processing context
        bilingual_context = None
        if hasattr(processed_query, 'metadata') and processed_query.metadata:
            bilingual_context = processed_query.metadata.get("bilingual_processing")
        
        if bilingual_context:
            query_source_lang = bilingual_context.get("source_language", processed_query.language)
            query_target_lang = bilingual_context.get("target_language", "en")
            
            # Prioritize chunks in the original query language
            if chunk_language == query_source_lang:
                ranking_score *= 1.3  # Strong boost for original language match
            elif chunk_language == query_target_lang:
                ranking_score *= 1.15  # Moderate boost for translated language match
            elif is_translated and original_language == query_source_lang:
                ranking_score *= 1.2  # Good boost for translated content from original language
            elif chunk_language in ["ml", "en"]:  # Both supported languages
                ranking_score *= 1.1  # Small boost for supported languages
        else:
            # Standard language matching
            if chunk_language == processed_query.language:
                ranking_score *= 1.1  # Boost for language match

        # Translation quality scoring for bilingual chunks
        if is_translated:
            translation_confidence = chunk_data["metadata"].get("translation_confidence", 0.5)
            if translation_confidence > 0.7:
                ranking_score *= 1.05  # Boost for high-quality translations
            elif translation_confidence < 0.4:
                ranking_score *= 0.9   # Slight penalty for low-quality translations
            
            # Boost for chunks that have both original and translated versions available
            original_chunk_id = chunk_data["metadata"].get("original_chunk_id")
            if original_chunk_id:
                ranking_score *= 1.08  # Small boost for having bilingual pair

        # Cross-language strategy scoring
        strategies_used = list(chunk_data["strategies"].keys())
        bilingual_strategies = [s for s in strategies_used if "bilingual" in s or "secondary" in s]
        
        if bilingual_strategies:
            # Apply cross-language boost for bilingual retrieval results
            for strategy_name in bilingual_strategies:
                if "bilingual" in strategy_name:
                    cross_lang_boost = self.strategies[strategy_name].parameters.get("cross_language_boost", 1.0)
                    ranking_score *= cross_lang_boost
                elif "secondary" in strategy_name:
                    secondary_boost = self.strategies[strategy_name].parameters.get("secondary_language_boost", 0.9)
                    ranking_score *= secondary_boost

        # Content length consideration
        token_count = chunk_data["metadata"].get("token_count", 0)
        if processed_query.intent == "explain" and token_count > 100:
            ranking_score *= 1.1  # Boost longer chunks for explanations
        elif processed_query.intent == "define" and 50 <= token_count <= 200:
            ranking_score *= 1.15  # Boost medium chunks for definitions

        # Recency boost (if timestamp available)
        created_at = chunk_data["metadata"].get("created_at", 0)
        if created_at > 0:
            # Simple recency boost (newer content gets slight preference)
            current_time = time.time()
            age_days = (current_time - created_at) / (24 * 3600)
            if age_days < 30:  # Content less than 30 days old
                ranking_score *= 1.05

        return ranking_score

    def _generate_relevance_explanation(
        self,
        chunk_data: dict[str, Any],
        processed_query: ProcessedQuery
    ) -> str:
        """Generate explanation of why result is relevant including bilingual context"""
        explanations = []

        # Strategy-based explanations
        strategies_used = list(chunk_data["strategies"].keys())
        if "semantic_similarity" in strategies_used:
            explanations.append("semantically similar content")
        if "keyword_matching" in strategies_used:
            explanations.append("contains relevant keywords")
        if "metadata_filtering" in strategies_used:
            explanations.append("matches contextual filters")
        
        # Bilingual strategy explanations
        if any("bilingual" in s for s in strategies_used):
            explanations.append("cross-language semantic match")
        if any("secondary" in s for s in strategies_used):
            explanations.append("secondary language search result")

        # Content type explanation
        chunk_type = chunk_data["metadata"].get("chunk_type", "content")
        if chunk_type == "section":
            explanations.append("section heading or summary")
        elif "table" in chunk_type.lower():
            explanations.append("structured data table")

        # Enhanced language and translation explanations
        chunk_language = chunk_data["metadata"].get("language", "unknown")
        is_translated = chunk_data["metadata"].get("is_translated", False)
        original_language = chunk_data["metadata"].get("original_language")
        
        # Handle bilingual context
        bilingual_context = None
        if hasattr(processed_query, 'metadata') and processed_query.metadata:
            bilingual_context = processed_query.metadata.get("bilingual_processing")
        
        if bilingual_context:
            query_source_lang = bilingual_context.get("source_language", processed_query.language)
            
            if chunk_language == query_source_lang:
                explanations.append(f"matches original query language ({chunk_language})")
            elif is_translated and original_language == query_source_lang:
                explanations.append(f"translated from query language ({original_language} → {chunk_language})")
            elif is_translated:
                translation_confidence = chunk_data["metadata"].get("translation_confidence", 0.5)
                explanations.append(f"high-quality translation (confidence: {translation_confidence:.1f})")
        else:
            # Standard language matching
            if chunk_language == processed_query.language:
                explanations.append(f"matches query language ({chunk_language})")
            elif is_translated:
                explanations.append(f"translated content ({original_language} → {chunk_language})")

        if explanations:
            return "Relevant because: " + ", ".join(explanations)
        else:
            return "Content matches query criteria"

    def _apply_final_filtering(self, ranked_results: list[RankedResult]) -> list[RankedResult]:
        """Apply final filtering to ranked results"""
        filtered_results = []

        for result in ranked_results:
            # Skip results below confidence threshold
            if result.ranking_score < self.min_confidence_threshold:
                continue

            # Skip very short results unless they're definitions
            if len(result.text.split()) < 5:
                continue

            # Skip duplicate content (basic deduplication)
            is_duplicate = False
            for existing in filtered_results:
                if self._calculate_text_similarity(result.text, existing.text) > 0.9:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_results.append(result)

            # Limit total results
            if len(filtered_results) >= self.max_results:
                break

        return filtered_results

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity for deduplication"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _generate_response_summary(
        self,
        processed_query: ProcessedQuery,
        results: list[RankedResult]
    ) -> str:
        """Generate summary of the response"""
        if not results:
            return "No relevant results found for your query."

        # Generate summary based on query intent and results
        if processed_query.intent == "define":
            return f"Found {len(results)} definitions and explanations related to your query."
        elif processed_query.intent == "compare":
            return f"Found {len(results)} comparison points and related information."
        elif processed_query.intent == "list":
            return f"Found {len(results)} items and lists matching your request."
        elif processed_query.intent == "explain":
            return f"Found {len(results)} detailed explanations for your question."
        else:
            return f"Found {len(results)} relevant results matching your search."

    def _calculate_overall_confidence(
        self,
        results: list[RankedResult],
        processed_query: ProcessedQuery
    ) -> float:
        """Calculate overall confidence in the response"""
        if not results:
            return 0.0

        # Base confidence on top result's score
        top_score = results[0].ranking_score if results else 0.0

        # Adjust based on query processing confidence
        query_confidence = processed_query.confidence

        # Adjust based on number of results
        result_count_factor = min(1.0, len(results) / 5)  # More results = higher confidence

        # Combine factors
        overall_confidence = (top_score * 0.5 + query_confidence * 0.3 + result_count_factor * 0.2)

        return min(1.0, max(0.0, overall_confidence))

    def _create_empty_response(
        self,
        query: str,
        processed_query: ProcessedQuery,
        start_time: float
    ) -> FormattedResponse:
        """Create empty response when no results found"""
        return FormattedResponse(
            query_id=f"empty_{int(time.time())}",
            original_query=query,
            processed_query=processed_query.processed_query,
            ranked_results=[],
            total_results=0,
            retrieval_time=round(time.time() - start_time, 3),
            processing_metadata={
                "query_processing": {
                    "language": processed_query.language,
                    "intent": processed_query.intent,
                    "query_type": processed_query.query_type
                },
                "no_results_reason": "No matching content found in the knowledge base"
            },
            response_summary="No relevant results found for your query. Try rephrasing or using different keywords.",
            confidence_score=0.0
        )

    def get_orchestrator_stats(self) -> dict[str, Any]:
        """Get orchestrator statistics"""
        avg_retrieval_time = (
            self.total_retrieval_time / self.total_retrievals
            if self.total_retrievals > 0 else 0.0
        )

        avg_ranking_time = (
            self.total_ranking_time / self.total_retrievals
            if self.total_retrievals > 0 else 0.0
        )

        return {
            "total_retrievals": self.total_retrievals,
            "total_retrieval_time": round(self.total_retrieval_time, 2),
            "total_ranking_time": round(self.total_ranking_time, 3),
            "average_retrieval_time": round(avg_retrieval_time, 3),
            "average_ranking_time": round(avg_ranking_time, 4),
            "strategies_configured": {
                name: {
                    "enabled": strategy.enabled,
                    "weight": strategy.weight,
                    "parameters": strategy.parameters
                }
                for name, strategy in self.strategies.items()
            },
            "configuration": {
                "max_results": self.max_results,
                "min_confidence_threshold": self.min_confidence_threshold,
                "enable_reranking": self.enable_reranking
            },
            "query_processor_stats": self.query_processor.get_processing_stats()
        }
