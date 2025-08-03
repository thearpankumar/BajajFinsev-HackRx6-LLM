"""
Prompt templates optimized for high-accuracy document analysis
Domain-specific prompts for Insurance, Legal, HR, and Compliance
"""

from typing import Dict


class PromptTemplates:
    """
    High-quality prompt templates for accurate document analysis
    """

    def __init__(self):
        self.system_prompt = self._create_system_prompt()
        self.domain_prompts = self._create_domain_prompts()

    def _create_system_prompt(self) -> str:
        """Create the main system prompt for accurate, human-like analysis"""
        return """You are a helpful document expert who explains complex information in a clear, natural way. People come to you when they need to understand important documents, and you always provide complete, accurate answers based on what's available.

YOUR APPROACH:
1. Always give a complete answer using the information you have
2. Speak naturally, like you're helping a colleague understand something important
3. Include all the specific details - numbers, dates, conditions, exceptions
4. Break down complex information into easy-to-understand parts
5. Use the exact terms from the document but explain them when needed
6. Be thorough but conversational

RESPONSE STYLE:
- Start with the main answer right away
- Add supporting details and context
- Use lists or bullet points for multiple items
- Include examples or scenarios mentioned in the document
- Explain technical terms simply while keeping the original language
- Sound helpful and knowledgeable, not robotic

ACCURACY FOCUS:
- Use exact numbers, percentages, and timeframes from the document
- Include all conditions, limitations, and exceptions mentioned
- Reference specific sections when it helps clarify
- Never make up information - only use what's in the document
- If something has multiple parts or conditions, explain them all"""

    def _create_domain_prompts(self) -> Dict[str, str]:
        """Create domain-specific prompt enhancements"""
        return {
            "insurance": """
INSURANCE DOMAIN EXPERTISE:
- Focus on policy terms, coverage limits, exclusions, and conditions
- Pay attention to waiting periods, deductibles, and claim procedures
- Identify beneficiaries, sum insured amounts, and premium details
- Note renewal conditions and no-claim bonuses
- Highlight any special provisions or riders
""",
            "legal": """
LEGAL DOMAIN EXPERTISE:
- Focus on rights, obligations, and legal procedures
- Identify key dates, deadlines, and statutory requirements
- Pay attention to definitions and legal terminology
- Note jurisdiction and applicable laws
- Highlight penalties, consequences, and remedies
""",
            "hr": """
HR DOMAIN EXPERTISE:
- Focus on employee rights, benefits, and policies
- Pay attention to leave policies, compensation, and performance criteria
- Identify reporting structures and procedures
- Note compliance requirements and disciplinary actions
- Highlight training and development provisions
""",
            "compliance": """
COMPLIANCE DOMAIN EXPERTISE:
- Focus on regulatory requirements and standards
- Pay attention to audit procedures and reporting obligations
- Identify risk management and control measures
- Note documentation and record-keeping requirements
- Highlight penalties for non-compliance
""",
        }

    def get_system_prompt(self) -> str:
        """Get the main system prompt"""
        return self.system_prompt

    def get_qa_prompt(
        self, question: str, context: str, domain: str = "general"
    ) -> str:
        """
        Generate a question-answering prompt with context for human-like responses

        Args:
            question: The question to answer
            context: Relevant document context
            domain: Document domain (insurance, legal, hr, compliance)
        """
        domain_guidance = self.domain_prompts.get(domain, "")

        prompt = f"""{domain_guidance}

Here's the relevant information from the document:

{context}

Question: {question}

Please provide a complete, natural answer based on this information. Write as if you're explaining this to someone who really needs to understand it clearly. Include all the important details, numbers, and conditions mentioned in the document sections above.

Your answer:"""

        return prompt

    def get_extraction_prompt(self, text: str, fields: list) -> str:
        """
        Generate a prompt for extracting specific fields from text

        Args:
            text: Text to extract from
            fields: List of fields to extract
        """
        fields_str = ", ".join(fields)

        prompt = f"""Extract the following information from the provided text: {fields_str}

TEXT:
{text}

INSTRUCTIONS:
1. Extract only information that is explicitly stated in the text
2. If a field is not mentioned, respond with "Not mentioned"
3. Provide exact quotes when possible
4. Format your response as a clear list

EXTRACTED INFORMATION:"""

        return prompt

    def get_summary_prompt(self, text: str, max_length: int = 200) -> str:
        """
        Generate a prompt for summarizing text

        Args:
            text: Text to summarize
            max_length: Maximum length of summary
        """
        prompt = f"""Provide a concise summary of the following text in no more than {max_length} words. Focus on the key points, important details, and main conclusions.

TEXT:
{text}

SUMMARY:"""

        return prompt

    def get_comparison_prompt(self, text1: str, text2: str, aspect: str) -> str:
        """
        Generate a prompt for comparing two text sections

        Args:
            text1: First text to compare
            text2: Second text to compare
            aspect: Specific aspect to compare
        """
        prompt = f"""Compare the following two text sections focusing on: {aspect}

TEXT 1:
{text1}

TEXT 2:
{text2}

INSTRUCTIONS:
1. Identify similarities and differences regarding {aspect}
2. Highlight any contradictions or inconsistencies
3. Provide specific examples from both texts
4. Be objective and factual in your comparison

COMPARISON:"""

        return prompt

    def get_validation_prompt(self, claim: str, context: str) -> str:
        """
        Generate a prompt for validating a claim against context

        Args:
            claim: Claim to validate
            context: Context to validate against
        """
        prompt = f"""Validate the following claim against the provided context:

CLAIM: {claim}

CONTEXT:
{context}

INSTRUCTIONS:
1. Determine if the claim is supported by the context
2. Identify specific evidence that supports or contradicts the claim
3. Note any conditions or limitations that apply
4. Provide a clear validation result: SUPPORTED, NOT SUPPORTED, or PARTIALLY SUPPORTED

VALIDATION RESULT:"""

        return prompt
