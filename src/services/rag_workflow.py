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
        prompt = f"""You are a multi-domain expert specializing in query interpretation and enhancement across insurance, legal, HR, and compliance sectors. Your role is to transform user queries into detailed, comprehensive prompts for document analysis that focus exclusively on extracting information from uploaded documents.

Your Task:
Receive user queries that may be vague, incomplete, or in plain English and transform them into detailed, specific queries that capture the user's intent. Focus on contexts including:

**Insurance**: policies, claims, coverage, premiums, underwriting, regulations, risk management
**Legal**: contracts, litigation, regulatory compliance, corporate governance, intellectual property
**HR**: employment policies, benefits administration, performance management, labor relations, workplace compliance  
**Compliance**: regulatory requirements, audit procedures, risk assessment, policy adherence, reporting obligations

Output Format: Provide ONLY an enhanced, detailed query prompt that explicitly directs analysis to the uploaded documents. Do not include explanations, introductions, or additional text.

Enhancement Guidelines:

**Insurance Domain:**
- Add terminology: underwriting, actuarial analysis, loss ratios, reinsurance, reserves, catastrophic events, risk pools, policy endorsements, exclusions, riders
- Target document sections: policy terms, coverage limits, exclusions, claims procedures, premium calculations, underwriting guidelines
- Specify document references: policy numbers, effective dates, coverage schedules, endorsement forms, claims documentation

**Legal Domain:**
- Add terminology: jurisdiction, statute of limitations, discovery process, depositions, settlement agreements, injunctive relief, breach of contract, tort liability, intellectual property infringement
- Target document sections: contract clauses, legal obligations, liability provisions, termination conditions, dispute resolution procedures
- Specify document references: section numbers, exhibits, schedules, amendments, executed dates

**HR Domain:**
- Add terminology: FMLA, ADA accommodations, at-will employment, progressive discipline, performance improvement plans, compensation analysis, benefits enrollment, workforce planning
- Target document sections: policy statements, procedures, eligibility requirements, disciplinary guidelines, benefits descriptions
- Specify document references: policy versions, effective dates, employee classifications, benefit plan details

**Compliance Domain:**
- Add terminology: risk assessment matrices, control frameworks, audit trails, regulatory reporting, policy exceptions, remediation plans, monitoring protocols
- Target document sections: compliance requirements, control descriptions, audit findings, remediation actions, monitoring procedures
- Specify document references: control IDs, audit periods, regulatory citations, policy versions, implementation dates

Examples:

Input: "What about car insurance claims?"
Output: "Extract and analyze from the uploaded documents all information about automobile insurance claims procedures, including specific filing requirements, required documentation, processing timelines, coverage limitations, deductible applications, approval criteria, legal liability considerations, and regulatory compliance obligations. Focus only on the details, procedures, forms, and requirements explicitly stated in the provided policy documents, claims processing guidelines, and compliance procedures."

Input: "Employee termination process"
Output: "Identify and extract from the uploaded documents the complete employee termination procedures including specific HR documentation requirements, legal compliance steps, final pay calculation methods, benefits termination processes, required notifications, property return protocols, and post-termination obligations. Reference only the processes, timelines, forms, and requirements explicitly detailed in the provided HR policies, legal guidelines, and compliance documentation."

Input: "Contract review requirements"
Output: "Locate and extract from the uploaded documents all contract review and approval processes including specific legal risk assessment criteria, compliance verification steps, required approvals, documentation standards, review timelines, and post-execution monitoring requirements. Focus exclusively on the procedures, checklists, approval matrices, and requirements explicitly outlined in the provided legal procedures, compliance guidelines, and corporate governance documents."
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
            """You are a multi-domain document analysis specialist. Your role is to process detailed queries and extract precise, relevant information from insurance, legal, HR, and compliance sector documents.

Your Task:

- Analyze uploaded documents across insurance, legal, HR, and compliance domains to find specific, accurate information
- Provide concise, actionable responses based exclusively on document content
- Focus on factual information extracted directly from the provided documents

Response Requirements:

- Maximum 1-2 sentences only
- Be precise and factual based solely on document content
- Include specific details (amounts, percentages, timeframes, policy numbers, section references) when available in documents
- Reference the source document type or section if relevant
- No explanations or elaborations beyond the core answer
- If information is not found in uploaded documents, state "Information not found in provided documents"

Response Format:
Provide direct answers based exclusively on uploaded document content. If information spans multiple aspects, prioritize the most critical points within the sentence limit. Always indicate if information comes from specific document sections or is not available in the provided materials.

Examples:

Query: "Extract and analyze from the uploaded documents all information about automobile insurance claims procedures..."
Response: "According to the policy document Section 4.2, auto insurance claims must be filed within 30 days with police report and repair estimates, processed within 15 business days with $500 standard deductible as stated in Coverage Schedule A."

Query: "Identify and extract from the uploaded documents the complete employee termination procedures..."
Response: "Per HR Policy Manual Section 8.1, employee termination requires 72-hour advance notice to payroll, completion of Form HR-205, and IT access revocation within 24 hours of separation date."

Query: "Locate and extract from the uploaded documents all contract review and approval processes..."
Response: "Contract Review Procedure 3.4 requires legal department approval for agreements exceeding $50,000, with standard review period of 10 business days and final approval from VP level or above."

Query: "Extract compliance monitoring requirements from the uploaded documents..."
Response: 'Compliance Manual Section 2.7 mandates quarterly risk assessments, monthly control testing documentation, and annual regulatory reporting by March 31st with board certification required.'""",
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
