import logging
from src.services.llm_clients import gemini_flash_model, groq_client, GROQ_MODEL_NAME
from pinecone import Pinecone
from src.core.config import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class RAGWorkflowService:
    def __init__(self):
        self.embedding_model = SentenceTransformer("nlpaueb/legal-bert-base-uncased")
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

    async def clarify_query(self, query: str) -> str:
        """Clarifies a user's query using Gemini 1.5 Flash."""
        logger.info(f"Clarifying query with Gemini: '{query}'")
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
            response = await gemini_flash_model.generate_content_async(prompt)
            clarified_query = response.text.strip()
            logger.info(f"Gemini clarified query: '{clarified_query}'")
            return clarified_query
        except Exception as e:
            logger.error(f"Error during Gemini query clarification: {e}")
            # Fallback to the original query if clarification fails
            return query

    def retrieve_relevant_clauses(self, query: str, document_ids: list[int]) -> list[str]:
        """Retrieves relevant clauses from Pinecone."""
        logger.info(f"Retrieving clauses for query: '{query}'")
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.pinecone_index.query(
            vector=query_embedding,
            top_k=10, # Increased top_k for better context
            filter={"document_id": {"$in": document_ids}},
            include_metadata=True
        )
        
        clauses = [match.metadata['clause_text'] for match in results.matches]
        logger.info(f"Retrieved {len(clauses)} relevant clauses.")
        return clauses

    async def generate_final_answer(self, original_query: str, context: list[str]) -> str:
        """Generates the final answer using the specified Groq model."""
        logger.info("Generating final answer with Groq...")
        context_str = "\n\n".join(context)
        prompt = f"Based on the following context, please provide a detailed and accurate answer to the user's original question.\n\nContext:\n{context_str}\n\nOriginal Question: {original_query}"
        
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": """You are an insurance document analysis specialist. Your role is to process detailed queries and extract precise, relevant information from insurance sector documents.
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
                                                    Response: "Premium adjustments are based on claims history (up to 25% increase), credit score changes, and annual risk reassessment, with 30-day advance notice required for any changes exceeding 10%."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        final_answer = response.choices[0].message.content
        logger.info("Successfully generated final answer.")
        return final_answer

    async def run_workflow(self, questions: list[str], document_ids: list[int]) -> list[str]:
        """Runs the full RAG workflow for a list of questions."""
        answers = []
        for question in questions:
            # Step 1: Clarify the query with Gemini
            clarified_query = await self.clarify_query(question)
            
            # Step 2: Retrieve clauses using the *clarified* query
            relevant_clauses = self.retrieve_relevant_clauses(clarified_query, document_ids)
            
            # Step 3: Generate the final answer using the *original* question for context
            final_answer = await self.generate_final_answer(question, relevant_clauses)
            
            answers.append(final_answer)
        return answers

rag_workflow_service = RAGWorkflowService()
