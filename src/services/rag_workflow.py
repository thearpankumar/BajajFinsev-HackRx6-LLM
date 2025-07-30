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
        prompt = f"""You are an insurance domain expert specializing in query interpretation and enhancement. Your role is to transform user queries into detailed, comprehensive prompts for document analysis.
Your Task:

Receive user queries that may be vague, incomplete, or in plain English
Transform them into detailed, specific queries that capture the user's intent
Focus on insurance-related contexts including policies, claims, coverage, premiums, underwriting, regulations, and compliance
Enhance queries with relevant insurance terminology and specific details needed for accurate document retrieval

Output Format:
Provide ONLY an enhanced, detailed query prompt. Do not include explanations, introductions, or additional text.
Enhancement Guidelines:

Add specific insurance terms and contexts
Include relevant policy types (life, health, auto, property, liability, etc.)
Specify document types if applicable (policy documents, claim forms, regulatory filings, etc.)
Clarify time periods, coverage amounts, or geographical regions when relevant
Include related concepts that might be in the documents

Examples:

Input: "What about car insurance claims?"
Output: "Provide detailed information about automobile insurance claims procedures, including filing requirements, documentation needed, claim processing timelines, coverage limitations, deductible applications, and approval criteria from the relevant policy documents and claims processing guidelines."
Input: "Premium changes"
Output: "Explain the factors that influence insurance premium adjustments, including risk assessment changes, policy modifications, regulatory updates, claims history impact, and renewal terms, with specific details on calculation methods and notification requirements from underwriting and policy administration documents."

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

    async def generate_answer_from_document(self, original_query: str, clarified_query: str, document_file: Any) -> str:
        """
        Generates a final answer using Gemini Pro, based on the uploaded document.
        """
        logger.info(f"Generating answer for: '{original_query}'")
        
        # The system prompt is the first part of the conversation history
        prompt_parts = [
            """You are an insurance document analysis specialist. Your role is to process detailed queries and extract precise, relevant information from insurance sector documents.
                                                    Your Task:

                                                    Receive detailed, enhanced queries from Agent 1
                                                    Analyze insurance documents to find specific, accurate information
                                                    Provide concise, actionable responses
                                                    Focus on factual information from the documents

                                                    Response Requirements:

                                                    Maximum 1-2 sentences only
                                                    Be precise and factual
                                                    Include specific details (amounts, percentages, timeframes) when available
                                                    Reference the source document type if relevant
                                                    No explanations or elaborations beyond the core answer

                                                    Response Format:
                                                    Provide direct answers based on document content. If information spans multiple aspects, prioritize the most critical points within the sentence limit.
                                                    Examples:

                                                    Query: "Provide detailed information about automobile insurance claims procedures..."
                                                    Response: "Auto insurance claims must be filed within 30 days of the incident with police report, photos, and repair estimates, and are processed within 15 business days with a $500 standard deductible."
                                                    Query: "Explain the factors that influence insurance premium adjustments..."
                                                    Response: "Premium adjustments are based on claims history (up to 25% increase), credit score changes, and annual risk reassessment, with 30-day advance notice required for any changes exceeding 10%.""",
            clarified_query,
            document_file
        ]
        
        try:
            response = await GEMINI_PRO_MODEL.generate_content_async(
                prompt_parts,
                generation_config={"temperature": 0.2}
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating answer from document: {e}")
            return "Error: Could not generate an answer for this question."

    async def run_workflow_for_question(self, question: str, document_file: Any) -> str:
        """
        Runs the full workflow for a single question.
        """
        clarified_query = await self.clarify_query(question)
        answer = await self.generate_answer_from_document(question, clarified_query, document_file)
        return answer

    async def run_parallel_workflow(self, questions: List[str], document_file: Any) -> List[str]:
        """
        Runs the entire RAG workflow in parallel for a list of questions.
        """
        logger.info(f"Starting parallel workflow for {len(questions)} questions.")
        
        # Create a task for each question
        tasks = [self.run_workflow_for_question(q, document_file) for q in questions]
        
        # Run all tasks concurrently
        answers = await asyncio.gather(*tasks)
        
        logger.info("Parallel workflow completed.")
        return answers

# Do not instantiate here; it will be handled in the __init__.py
# rag_workflow_service = RAGWorkflowService()