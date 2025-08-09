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

        # Retrieval strategies configuration
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
            )
        }

        # Performance tracking
        self.total_retrievals = 0
        self.total_retrieval_time = 0.0
        self.total_ranking_time = 0.0

        # Configuration
        self.max_results = getattr(config, 'max_retrieval_results', 20)
        self.min_confidence_threshold = getattr(config, 'min_confidence_threshold', 0.3)
        self.enable_reranking = getattr(config, 'enable_reranking', True)

        logger.info("RetrievalOrchestrator initialized with advanced ranking")

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
            # Step 1: Process query
            logger.info("🔄 Step 1: Processing query...")
            processed_query = await self.query_processor.process_query(query, context)

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
                    "filters_applied": len([s for s in self.strategies.values() if s.enabled and "filter" in s.name])
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
        """Execute enabled retrieval strategies"""
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

                # Execute retrieval
                retrieval_result = await self.rag_pipeline.query(rag_query)

                if retrieval_result.total_results > 0:
                    results.append((strategy_name, retrieval_result))
                    logger.debug(f"✅ Strategy {strategy_name}: {retrieval_result.total_results} results")
                else:
                    logger.debug(f"⚠️ Strategy {strategy_name}: No results")

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
        """Calculate final ranking score with various factors"""
        ranking_score = base_score

        # Chunk type preference based on query intent
        chunk_type = chunk_data["metadata"].get("chunk_type", "paragraph")
        if processed_query.intent == "define" and chunk_type == "section":
            ranking_score *= 1.2  # Boost sections for definitions
        elif processed_query.intent == "list" and "table" in chunk_type.lower():
            ranking_score *= 1.3  # Boost tables for lists

        # Language matching
        chunk_language = chunk_data["metadata"].get("language", "unknown")
        if chunk_language == processed_query.language:
            ranking_score *= 1.1  # Boost for language match

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
        """Generate explanation of why result is relevant"""
        explanations = []

        # Strategy-based explanations
        strategies_used = list(chunk_data["strategies"].keys())
        if "semantic_similarity" in strategies_used:
            explanations.append("semantically similar content")
        if "keyword_matching" in strategies_used:
            explanations.append("contains relevant keywords")
        if "metadata_filtering" in strategies_used:
            explanations.append("matches contextual filters")

        # Content type explanation
        chunk_type = chunk_data["metadata"].get("chunk_type", "content")
        if chunk_type == "section":
            explanations.append("section heading or summary")
        elif "table" in chunk_type.lower():
            explanations.append("structured data table")

        # Language match
        chunk_language = chunk_data["metadata"].get("language", "unknown")
        if chunk_language == processed_query.language:
            explanations.append(f"matches query language ({chunk_language})")

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
