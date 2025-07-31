import logging
import asyncio
from typing import List, Any

from src.services.llm_clients import GEMINI_PRO_MODEL, GEMINI_FLASH_MODEL

logger = logging.getLogger(__name__)

class RAGWorkflowService:
    async def clarify_query(self, query: str) -> str:
        """
        Clarifies and enhances a user's query using the fast Gemini Flash model.
        """
        logger.info(f"Clarifying query: '{query}'")
        prompt = f"""You are a query enhancement specialist. Transform vague user queries into specific, detailed prompts for document analysis.
Your Task:
Convert user queries into comprehensive prompts that extract information exclusively from uploaded documents.
Output Format: Provide ONLY an enhanced query prompt. No explanations or additional text.
Enhancement Guidelines:

Add specific terminology and concepts relevant to the query topic
Target specific document sections and references
Focus on extracting information only from provided documents
Include related concepts that might be in the documents

Examples:
Input: "What about car insurance claims?"
Output: "Extract from uploaded documents: automobile insurance claims procedures, filing requirements, documentation needed, processing timelines, coverage limits, deductibles, approval criteria, and related processes."
Input: "Employee termination process"
Output: "Identify from uploaded documents: employee termination procedures, documentation requirements, compliance steps, final pay processes, benefits termination, notifications required, and related policies."
Input: "Research methodology"
Output: "Locate from uploaded documents: research methodology procedures, validation requirements, review processes, data validation methods, analysis frameworks, and quality assurance measures."
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

    async def generate_answer_from_document(self, original_query: str, clarified_query: str, document_files: List[Any]) -> str:
        """
        Generates a final answer using Gemini Pro, based on the uploaded document chunks.
        """
        logger.info(f"Generating answer for: '{original_query}'")
        
        system_prompt = """You are a document analysis specialist.
Your Task:

Analyze uploaded documents to extract specific information
Provide concise responses based exclusively on document content
Reference only information found in the provided documents

Response Requirements:

Maximum 1-2 sentences only
Include specific details from documents (amounts, percentages, timeframes, section references)
State "Information not found in provided documents" if unavailable
Only use information explicitly stated in the uploaded documents

Examples:
Query: "Extract automobile insurance claims procedures..."
Response: "According to Policy Section 4.2, claims must be filed within 30 days with police report and are processed within 15 business days with $500 standard deductible."
Query: "Identify employee termination procedures..."
Response: "HR Policy 8.1 requires 72-hour advance notice to payroll, completion of Form HR-205, and IT access revocation within 24 hours of separation."
Query: "Locate research methodology validation..."
Response: "Protocol Section 3.4 requires peer review by three qualified reviewers and statistical analysis with α=0.05 significance level as outlined in the methodology guidelines."
"""
        
        try:
            # The system prompt is the first turn, and the user's query and files are the second.
            contents = [
                {
                    "role": "model",
                    "parts": [system_prompt]
                },
                {
                    "role": "user",
                    "parts": [clarified_query] + document_files
                }
            ]
            
            # Use the centrally defined GEMINI_PRO_MODEL
            response = await GEMINI_PRO_MODEL.generate_content_async(
                contents,
                generation_config={"temperature": 0.2}
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer from document: {e}")
            return "Error: Could not generate an answer for this question."

    async def run_workflow_for_question(self, question: str, document_files: List[Any]) -> str:
        """
        Runs the full workflow for a single question against a list of document chunks.
        """
        clarified_query = await self.clarify_query(question)
        answer = await self.generate_answer_from_document(question, clarified_query, document_files)
        return answer

    async def run_parallel_workflow(self, questions: List[str], document_files: List[Any]) -> List[str]:
        """
        Runs the entire RAG workflow in parallel for a list of questions.
        """
        logger.info(f"Starting parallel workflow for {len(questions)} questions.")
        
        # Create a task for each question to run fully in parallel
        tasks = [self.run_workflow_for_question(q, document_files) for q in questions]
        
        # Run all tasks concurrently
        answers = await asyncio.gather(*tasks)
        
        logger.info("Parallel workflow completed.")
        return answers

# Do not instantiate here; it will be handled in the __init__.py
# rag_workflow_service = RAGWorkflowService()
