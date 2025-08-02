import logging
import asyncio
from typing import List, Tuple, Dict, Any
from datetime import datetime
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.services.llm_clients import OPENAI_CLIENT, OPENAI_MODEL_NAME, GEMINI_FLASH_MODEL
from src.services.embedding_service import embedding_service
from src.services.hierarchical_chunking_service import hierarchical_chunking_service
from src.services.prompt_registry import prompt_registry
from src.services.contradiction_service import contradiction_service
from src.services.knowledge_base_service import knowledge_base_service
from src.core.config import settings

# NLTK imports for metadata extraction
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.data import find
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# SpaCy imports for metadata extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGWorkflowService:
    """
    Advanced RAG workflow service with hybrid search, re-ranking, and Pinecone integration.
    """
    
    def __init__(self):
        self.embedding_service = embedding_service
        self.hierarchical_service = hierarchical_chunking_service
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        self.logger = logger

    async def clarify_query(self, query: str) -> str:
        """
        Clarifies and enhances a user's query using the fast Gemini Flash model.
        """
        logger.info(f"Clarifying query: '{query}'")
        prompt = prompt_registry.get_prompt('query_clarification').format(query=query)
        
        try:
            response = await GEMINI_FLASH_MODEL.generate_content_async(prompt)
            clarified_query = response.text.strip()
            logger.info(f"Clarified query to: '{clarified_query}'")
            return clarified_query
        except Exception as e:
            logger.error(f"Error during query clarification: {e}")
            return query # Fallback to original query

    async def retrieve_and_rerank_chunks(self, query: str, document_url: str, document_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Performs hybrid search (BM25 + LanceDB) with Reciprocal Rank Fusion (RRF)
        and re-ranks the results.
        """
        self.logger.info(f"Starting hybrid search and re-ranking for query: '{query[:50]}...'")
        
        chunk_texts = [chunk['text'] for chunk in document_chunks]

        # 1. Sparse Search (BM25)
        tokenized_corpus = [doc.split(" ") for doc in chunk_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_results = sorted(
            [(score, i) for i, score in enumerate(bm25_scores)], 
            key=lambda x: x[0], 
            reverse=True
        )[:30]  # Increased from 20 to 30 for better coverage

        # 2. Dense Search (LanceDB)
        pinecone_results = await self.embedding_service.embed_and_search(
            query=query, document_url=document_url, top_k=30  # Increased from 20 to 30
        )
        
        # 3. Reciprocal Rank Fusion (RRF)
        fused_scores = {}
        k = 60  # RRF ranking constant

        # Process BM25 results
        for rank, (score, doc_index) in enumerate(bm25_results):
            doc_text = chunk_texts[doc_index]
            if doc_text not in fused_scores:
                fused_scores[doc_text] = 0
            fused_scores[doc_text] += 1 / (k + rank + 1)

        # Process Dense search results
        for rank, result in enumerate(pinecone_results):
            doc_text = result['metadata']['text']
            if doc_text not in fused_scores:
                fused_scores[doc_text] = 0
            fused_scores[doc_text] += 1 / (k + rank + 1)

        # Sort fused results
        sorted_fused_results = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Get the chunk objects for the top fused results
        combined_chunks = []
        for text, score in sorted_fused_results[:30]: # Take top 30 fused results for re-ranking
            # Find the original chunk data
            original_chunk = next((chunk for chunk in document_chunks if chunk['text'] == text), None)
            if original_chunk:
                combined_chunks.append(original_chunk)

        self.logger.info(f"Fused search returned {len(combined_chunks)} unique candidates for re-ranking.")

        # 4. Re-ranking with CrossEncoder
        if not combined_chunks:
            return []
            
        rerank_pairs = [[query, chunk['text']] for chunk in combined_chunks]
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        reranked_results = sorted(zip(combined_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
        
        # 5. Format final results
        final_chunks = []
        top_k = settings.MAX_CHUNKS_PER_QUERY * 2  # Increase number of chunks for better coverage
        for chunk_data, score in reranked_results[:top_k]:
            # Ensure metadata includes the text
            metadata = chunk_data.get('metadata', {})
            if 'text' not in metadata:
                metadata['text'] = chunk_data.get('text', '')
            
            final_chunks.append({
                "metadata": metadata,
                "score": float(score)
            })
        
        self.logger.info(f"Re-ranking complete. Returning top {len(final_chunks)} chunks.")
        return final_chunks

    async def generate_answer_from_chunks(self, original_query: str, clarified_query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Generates an answer from the relevant chunks.
        """
        logger.info(f"Generating answer for: '{original_query}'")
        
        if not relevant_chunks:
            return f"No relevant information was found in the document to answer the question: '{original_query}'."

        context_parts = []
        metadata_parts = []
        for i, chunk in enumerate(relevant_chunks):
            text = chunk['metadata'].get('text', '')
            score = chunk.get('score', 0.0)
            context_parts.append(f"--- Excerpt {i+1} (relevance: {score:.3f}) ---\n{text}")
            
            # Add metadata information
            entities = chunk['metadata'].get('entities', [])
            concepts = chunk['metadata'].get('concepts', [])
            categories = chunk['metadata'].get('categories', [])
            keywords = chunk['metadata'].get('keywords', [])
            
            if entities or concepts or categories or keywords:
                metadata_info = []
                if entities:
                    entity_names = [e['name'] for e in entities[:5]]  # Increased from 3 to 5 entities
                    metadata_info.append(f"Entities: {', '.join(entity_names)}")
                if concepts:
                    metadata_info.append(f"Concepts: {', '.join(concepts[:10])}")  # Increased from 5 to 10 concepts
                if categories:
                    metadata_info.append(f"Categories: {', '.join(categories)}")
                if keywords:
                    metadata_info.append(f"Keywords: {', '.join(keywords[:15])}")  # Increased from 10 to 15 keywords
                metadata_parts.append(f"--- Excerpt {i+1} Metadata ---\n" + "\n".join(metadata_info))
        
        context = "\n\n".join(context_parts)
        metadata_context = "\n\n".join(metadata_parts) if metadata_parts else ""
        
        system_prompt = prompt_registry.get_prompt('answer_generation')

        user_prompt = f"""Document Excerpts:
{context}

Metadata Information:
{metadata_context}

Question: {clarified_query}

CONCISE ANSWER:"""

        try:
            # import time
            # start_time = time.time()
            
            response = await OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=250,  # Increased from 150 to 250 for more detailed answers
                top_p=0.9
            )
            answer = response.choices[0].message.content.strip()
            # end_time = time.time()
            
            # Record A/B testing result
            # TODO: Implement A/B testing recording
            # response_time = end_time - start_time
            # result_data = {
            #     'response_time': response_time,
            #     'answer_length': len(answer),
            #     'model': OPENAI_MODEL_NAME
            # }
            
            # Check for contradictions if multiple chunks were used
            if len(relevant_chunks) > 1:
                # Extract answers from contexts for contradiction detection
                context_texts = []
                for chunk in relevant_chunks:
                    # In a real implementation, we might extract potential answers from each context
                    # For now, we'll just use the main answer and check for contradictions in the contexts
                    context_texts.append(chunk['metadata'].get('text', ''))
                
                # Generate uncertainty response if needed
                # TODO: Use uncertainty response in contradiction detection
                # uncertainty_response = contradiction_service.generate_uncertainty_response(
                #     original_query, confidence=0.9
                # )
                
                # For now, we'll just log that we detected potential contradictions
                # A full implementation would involve more sophisticated contradiction detection
                if context_texts:
                    self.logger.info("Potential contradictions detected in contexts")
            
            
            logger.info(f"Answer generated: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            return "Error: Could not generate an answer for this question."
    
    def _calculate_answer_confidence(self, relevant_chunks: List[Dict[str, Any]],
                                      answer: str) -> float:
        """
        Calculate confidence score for an answer based on retrieved chunks and answer quality.
        
        Args:
            relevant_chunks: List of relevant chunks with scores
            answer: Generated answer
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not relevant_chunks:
            return 0.0
        
        # Calculate average chunk relevance score
        avg_relevance = sum(chunk.get('score', 0.0) for chunk in relevant_chunks) / len(relevant_chunks)
        
        # Boost confidence for high relevance scores
        relevance_confidence = min(avg_relevance * 2.0, 1.0)  # Scale to 0-1 range
        
        # Calculate answer quality score
        answer_quality = self._calculate_answer_quality(answer)
        
        # Combine relevance and quality scores
        combined_confidence = (relevance_confidence * 0.7) + (answer_quality * 0.3)
        
        return combined_confidence
    
    def _calculate_answer_quality(self, answer: str) -> float:
        """
        Calculate answer quality based on length, specificity, and structure.
        
        Args:
            answer: Answer string
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not answer or len(answer.strip()) == 0:
            return 0.0
        
        # Score based on word count (optimal range 10-50 words)
        words = answer.split()
        word_count = len(words)
        word_score = max(0.0, min(1.0, (word_count - 5) / 45.0))  # Normalize to 0-1
        
        # Score based on specific terms (numbers, dates, etc.)
        import re
        number_count = len(re.findall(r'\d+', answer))
        number_score = min(number_count * 0.1, 0.5)  # Cap at 0.5
        
        # Score based on structured response (Yes/No at start)
        structured_score = 0.0
        answer_lower = answer.lower().strip()
        if answer_lower.startswith(('yes', 'no')):
            structured_score = 0.3
        
        # Total quality score
        total_score = word_score + number_score + structured_score
        return min(total_score, 1.0)
    
    async def process_single_question(self, question: str, document_url: str, document_chunks: List[Dict[str, Any]], question_index: int) -> Dict[str, Any]:
        """
        Processes a single question and returns the answer and the retrieved contexts.
        """
        try:
            self.logger.info(f"Processing question {question_index + 1}: {question[:50]}...")
            clarified_query = await self.clarify_query(question)
            
            relevant_chunks = await self.retrieve_and_rerank_chunks(
                query=clarified_query,
                document_url=document_url,
                document_chunks=document_chunks
            )
            
            answer = await self.generate_answer_from_chunks(question, clarified_query, relevant_chunks)
            
            contexts = [chunk['metadata'].get('text', '') for chunk in relevant_chunks]
            
            # Extract metadata from relevant chunks for additional context
            chunk_metadata = []
            for chunk in relevant_chunks:
                metadata = {
                    "entities": chunk['metadata'].get('entities', []),
                    "concepts": chunk['metadata'].get('concepts', []),
                    "categories": chunk['metadata'].get('categories', []),
                    "keywords": chunk['metadata'].get('keywords', [])
                }
                chunk_metadata.append(metadata)
            
            # Record A/B testing result
            # TODO: Implement A/B testing recording
            # result_data = {
            #     'accuracy': 0.0,  # This would be set based on evaluation
            #     'response_time': 0.0,  # This would be measured
            #     'answer_length': len(answer)
            # }
            
            self.logger.info(f"Question {question_index + 1} completed.")
            # Track knowledge base versioning
            if document_url:
                # Create a document entry for versioning
                document_entry = {
                    'id': document_url,
                    'content': ' '.join(contexts[:100]),  # First 100 contexts
                    'metadata': {
                        'question': question,
                        'answer': answer,
                        'timestamp': datetime.now().isoformat()
                    }
                }
                
                # Add to current knowledge base version
                try:
                    knowledge_base_service.add_document_to_version(document_entry)
                except Exception as e:
                    self.logger.warning(f"Failed to add document to knowledge base versioning: {e}")
            
            return {"answer": answer, "contexts": contexts, "metadata": chunk_metadata}
        except Exception as e:
            self.logger.error(f"Error processing question {question_index + 1}: {e}", exc_info=True)
            return {
                "answer": f"Error: Could not process this question - {str(e)}",
                "contexts": [],
                "metadata": []
            }
    
    async def process_single_question_with_confidence(self, question: str, document_url: str,
                                                      document_chunks: List[Dict[str, Any]],
                                                      question_index: int) -> Dict[str, Any]:
        """
        Processes a single question with confidence calculation and advanced uncertainty expression.
        
        Args:
            question: Question to process
            document_url: Document URL
            document_chunks: Document chunks
            question_index: Question index for logging
            
        Returns:
            Dictionary with answer, contexts, metadata, and confidence information
        """
        try:
            self.logger.info(f"Processing question {question_index + 1} with confidence: {question[:50]}...")
            clarified_query = await self.clarify_query(question)
            
            relevant_chunks = await self.retrieve_and_rerank_chunks(
                query=clarified_query,
                document_url=document_url,
                document_chunks=document_chunks
            )
            
            answer = await self.generate_answer_from_chunks(question, clarified_query, relevant_chunks)
            
            # Calculate confidence in the answer
            confidence = self._calculate_answer_confidence(relevant_chunks, answer)
            
            # Enhance answer with uncertainty expression if confidence is low
            if confidence < 0.3:
                uncertainty_expression = contradiction_service.generate_uncertainty_response(
                    question, confidence=confidence
                )
                enhanced_answer = f"{answer} {uncertainty_expression}"
            else:
                enhanced_answer = answer
            
            contexts = [chunk['metadata'].get('text', '') for chunk in relevant_chunks]
            
            # Extract metadata from relevant chunks for additional context
            chunk_metadata = []
            for chunk in relevant_chunks:
                metadata = {
                    "entities": chunk['metadata'].get('entities', []),
                    "concepts": chunk['metadata'].get('concepts', []),
                    "categories": chunk['metadata'].get('categories', []),
                    "keywords": chunk['metadata'].get('keywords', [])
                }
                chunk_metadata.append(metadata)
            
            self.logger.info(f"Question {question_index + 1} completed with confidence {confidence:.2f}.")
            
            return {
                "answer": enhanced_answer,
                "contexts": contexts,
                "metadata": chunk_metadata,
                "confidence": confidence,
                "original_answer": answer
            }
        except Exception as e:
            self.logger.error(f"Error processing question {question_index + 1}: {e}", exc_info=True)
            return {
                "answer": f"Error: Could not process this question - {str(e)}",
                "contexts": [],
                "confidence": 0.0
            }
    
    async def run_parallel_workflow(self, document_url: str, questions: List[str], document_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Standard workflow: Indexes the document, then processes all questions in parallel.
        """
        logger.info(f"🚀 Starting standard parallel workflow for {len(questions)} questions on {document_url}")
        await self.embedding_service.embed_and_upsert_chunks(document_url, document_chunks)
        tasks = [self.process_single_question(q, document_url, document_chunks, i) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks)
        logger.info(f"🎉 Standard parallel workflow completed for {document_url}")
        return results
    
    async def run_hierarchical_workflow(self, document_url: str, questions: List[str], document_text: str) -> Tuple[List[Dict[str, Any]], Dict]:
        """
        Hierarchical workflow: Indexes relevant chunks, then processes questions in parallel.
        """
        logger.info(f"🚀 Starting hierarchical workflow for {len(questions)} questions on {document_url}")
        
        all_relevant_chunks = []
        for question in questions:
            relevant_chunks_for_q, _ = await self.hierarchical_service.process_large_document(
                document=document_text,
                query=question,
                max_sections=settings.MAX_SECTIONS_PER_QUERY * 2  # Increase sections for better coverage
            )
            # Add chunks with their metadata
            for chunk in relevant_chunks_for_q:
                if isinstance(chunk, dict) and "text" in chunk:
                    all_relevant_chunks.append(chunk)
                else:
                    # Handle legacy string format
                    all_relevant_chunks.append({"text": chunk, "metadata": {}})
        
        # Remove duplicates while preserving metadata
        unique_chunks = []
        seen_texts = set()
        for chunk in all_relevant_chunks:
            chunk_text = chunk["text"]
            if chunk_text not in seen_texts:
                unique_chunks.append(chunk)
                seen_texts.add(chunk_text)
        
        logger.info(f"Hierarchical processing identified {len(unique_chunks)} unique relevant chunks.")

        await self.embedding_service.embed_and_upsert_chunks(document_url, unique_chunks)
        
        tasks = [self.process_single_question(q, document_url, unique_chunks, i) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks)
        
        logger.info(f"🎉 Hierarchical workflow completed for {document_url}")
        
        metrics = {
            'hierarchical_used': True,
            'total_questions': len(questions),
            'unique_chunks_indexed': len(unique_chunks)
        }
        
        return results, metrics
    
