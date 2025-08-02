import logging
import asyncio
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.services.llm_clients import OPENAI_CLIENT, OPENAI_MODEL_NAME, GEMINI_FLASH_MODEL
from src.services.embedding_service import embedding_service
from src.services.hierarchical_chunking_service import hierarchical_chunking_service
from src.core.config import settings

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
        prompt = f"""You are a precision-focused document analyst specializing in multiple domains: insurance, legal, HR, compliance, and scientific/historical texts. Your task is to refine user queries for precise information extraction.

Your Task:
Transform user queries into focused, specific queries that target exact information in documents. Adapt your approach based on the document domain.

Output Format: Provide ONLY a refined query. No explanations or additional text.

Refinement Guidelines:
- Focus on specific, exact information being requested
- Include precise terminology appropriate to the domain
- For scientific texts: Target definitions, laws, principles, and demonstrations
- For business documents: Target clauses, sections, and policy provisions
- Emphasize exact values, timeframes, conditions, and methodologies

---
User Query: '{query}'"""
        
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
        Performs hybrid search (BM25 + Pinecone) with Reciprocal Rank Fusion (RRF)
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
        )[:20]

        # 2. Dense Search (LanceDB)
        pinecone_results = await self.embedding_service.embed_and_search(
            query=query, document_url=document_url, top_k=20
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
        for text, score in sorted_fused_results[:20]: # Take top 20 fused results for re-ranking
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
        top_k = settings.MAX_CHUNKS_PER_QUERY
        for chunk_data, score in reranked_results[:top_k]:
            final_chunks.append({
                "metadata": chunk_data.get('metadata', {}),
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
        for i, chunk in enumerate(relevant_chunks):
            text = chunk['metadata'].get('text', '')
            score = chunk.get('score', 0.0)
            context_parts.append(f"--- Excerpt {i+1} (relevance: {score:.3f}) ---\n{text}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a helpful document analyst. Provide concise, conversational answers.
RULES:
- Maximum 1-2 sentences.
- Start with Yes/No for existence/coverage questions.
- Use simple, direct language.
- Include key numbers/timeframes only.
- Summarize main points, skip detailed conditions.
- If the excerpts do not contain the answer, state that the information was not found in the provided text."""

        user_prompt = f"""Document Excerpts:
{context}

Question: {clarified_query}

CONCISE ANSWER:"""

        try:
            response = await OPENAI_CLIENT.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=150,
                top_p=0.9
            )
            answer = response.choices[0].message.content.strip()
            logger.info(f"Answer generated: {answer[:100]}...")
            return answer
        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            return "Error: Could not generate an answer for this question."

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
            
            self.logger.info(f"Question {question_index + 1} completed.")
            return {"answer": answer, "contexts": contexts}
        except Exception as e:
            self.logger.error(f"Error processing question {question_index + 1}: {e}", exc_info=True)
            return {
                "answer": f"Error: Could not process this question - {str(e)}",
                "contexts": []
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
        
        all_relevant_texts = set()
        for question in questions:
            relevant_chunks_for_q, _ = await self.hierarchical_service.process_large_document(
                document=document_text,
                query=question,
                max_sections=settings.MAX_SECTIONS_PER_QUERY
            )
            for chunk in relevant_chunks_for_q:
                all_relevant_texts.add(chunk)
        
        chunk_list = [{"text": text, "metadata": {}} for text in all_relevant_texts]
        logger.info(f"Hierarchical processing identified {len(chunk_list)} unique relevant chunks.")

        await self.embedding_service.embed_and_upsert_chunks(document_url, chunk_list)
        
        tasks = [self.process_single_question(q, document_url, chunk_list, i) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks)
        
        logger.info(f"🎉 Hierarchical workflow completed for {document_url}")
        
        metrics = {
            'hierarchical_used': True,
            'total_questions': len(questions),
            'unique_chunks_indexed': len(chunk_list)
        }
        
        return results, metrics
