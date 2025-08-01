import logging
import asyncio
from typing import List, Tuple

from src.services.llm_clients import GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL
from src.services.embedding_service import embedding_service
from src.core.config import settings

logger = logging.getLogger(__name__)

class RAGWorkflowService:
    """
    Updated RAG workflow service that uses embeddings for document retrieval
    instead of uploading entire documents to Gemini.
    """
    
    def __init__(self):
        self.embedding_service = embedding_service
    
    async def clarify_query(self, query: str) -> str:
        """
        Clarifies and enhances a user's query using the fast Gemini Flash model.
        """
        logger.info(f"Clarifying query: '{query}'")
        prompt = f"""You are a query enhancement specialist. Transform vague user queries into specific, detailed prompts for document analysis.

Your Task:
Convert user queries into comprehensive prompts that extract information exclusively from provided document excerpts.

Output Format: Provide ONLY an enhanced query prompt. No explanations or additional text.

Enhancement Guidelines:
- Add specific terminology and concepts relevant to the query topic
- Target specific document sections and references
- Focus on extracting information only from provided document excerpts
- Include related concepts that might be in the documents

Examples:
Input: "What about car insurance claims?"
Output: "Extract from document excerpts: automobile insurance claims procedures, filing requirements, documentation needed, processing timelines, coverage limits, deductibles, approval criteria, and related processes."

Input: "Employee termination process"
Output: "Identify from document excerpts: employee termination procedures, documentation requirements, compliance steps, final pay processes, benefits termination, notifications required, and related policies."

Input: "Research methodology"
Output: "Locate from document excerpts: research methodology procedures, validation requirements, review processes, data validation methods, analysis frameworks, and quality assurance measures."

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

    async def retrieve_relevant_chunks(self, query: str, document_chunks: List[str]) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant document chunks using embedding similarity.
        """
        logger.info(f"Retrieving relevant chunks for query: '{query[:50]}...'")
        
        try:
            # Use embedding service to find similar chunks
            similar_chunks = await self.embedding_service.embed_and_search(
                query=query,
                document_chunks=document_chunks,
                top_k=settings.MAX_CHUNKS_PER_QUERY
            )
            
            logger.info(f"Retrieved {len(similar_chunks)} relevant chunks with similarities: {[f'{score:.3f}' for _, score in similar_chunks[:3]]}")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            # Fallback: return first few chunks
            fallback_chunks = [(chunk, 0.0) for chunk in document_chunks[:settings.MAX_CHUNKS_PER_QUERY]]
            return fallback_chunks

    async def generate_answer_from_chunks(self, original_query: str, clarified_query: str, relevant_chunks: List[Tuple[str, float]]) -> str:
        """
        Generates a final answer using Gemini Pro, based on the most relevant document chunks.
        """
        logger.info(f"Generating answer for: '{original_query}'")
        
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        # Prepare context from relevant chunks
        context_parts = []
        for i, (chunk, score) in enumerate(relevant_chunks):
            context_parts.append(f"--- Relevant Excerpt {i+1} (similarity: {score:.3f}) ---\n{chunk}")
        
        context = "\n\n".join(context_parts)
        
        system_prompt = """You are a document analysis specialist. Your task is to answer questions based exclusively on the provided document excerpts.

Response Requirements:
- Maximum 1-2 sentences only
- Include specific details from excerpts (amounts, percentages, timeframes, section references)
- State "Information not found in provided excerpts" if unavailable
- Only use information explicitly stated in the provided document excerpts
- Reference the excerpt number when possible (e.g., "According to Excerpt 1...")

Examples:
Query: "What is the waiting period for pre-existing diseases?"
Response: "According to Excerpt 1, the waiting period for pre-existing diseases is 2 years from the policy commencement date."

Query: "What are the room rent limits?"
Response: "Excerpt 2 states that room rent is limited to 1% of the sum insured per day with ICU charges at 2% of sum insured per day."
"""
        
        user_prompt = f"""Document Excerpts:
{context}

Question: {clarified_query}

Based on the above excerpts, provide a concise answer:"""

        try:
            response = await GEMINI_PRO_MODEL.generate_content_async(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config={"temperature": 0.1}  # Lower temperature for more factual responses
            )
            
            answer = response.text.strip()
            logger.debug(f"Generated answer: {answer[:100]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer from chunks: {e}")
            return "Error: Could not generate an answer for this question."

    async def run_workflow_for_question(self, question: str, document_chunks: List[str]) -> str:
        """
        Runs the full embedding-based RAG workflow for a single question.
        """
        # Step 1: Clarify the query
        clarified_query = await self.clarify_query(question)
        
        # Step 2: Retrieve relevant chunks using embeddings
        relevant_chunks = await self.retrieve_relevant_chunks(clarified_query, document_chunks)
        
        # Step 3: Generate answer from relevant chunks
        answer = await self.generate_answer_from_chunks(question, clarified_query, relevant_chunks)
        
        return answer

    async def run_parallel_workflow(self, questions: List[str], document_chunks: List[str]) -> List[str]:
        """
        Runs the entire embedding-based RAG workflow in parallel for a list of questions.
        """
        logger.info(f"Starting parallel embedding-based workflow for {len(questions)} questions.")
        
        # Pre-generate embeddings for all document chunks once to avoid redundant work
        logger.info("Pre-generating embeddings for document chunks...")
        await self.embedding_service.generate_embeddings_batch(document_chunks)
        
        # Create a task for each question to run fully in parallel
        tasks = [self.run_workflow_for_question(q, document_chunks) for q in questions]
        
        # Run all tasks concurrently
        answers = await asyncio.gather(*tasks)
        
        logger.info("Parallel embedding-based workflow completed.")
        return answers

# Do not instantiate here; it will be handled in the __init__.py
# rag_workflow_service = RAGWorkflowService()
