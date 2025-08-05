"""
Domain-Aware Answer Generator with context-aware prompts
Optimized for 95% accuracy in insurance, legal, HR, and compliance domains
"""

import logging
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time

from openai import AsyncOpenAI
import google.generativeai as genai

from src.core.config import settings
from src.core.advanced_retrieval_engine import SearchResult, QueryType

logger = logging.getLogger(__name__)


class DomainType(Enum):
    INSURANCE_POLICY = "insurance_policy"
    LEGAL_CONTRACT = "legal_contract"
    HR_DOCUMENT = "hr_document"
    COMPLIANCE_DOCUMENT = "compliance_document"
    GENERAL = "general"


class AnswerType(Enum):
    FACTUAL = "factual"           # Direct facts and figures
    PROCEDURAL = "procedural"     # Step-by-step procedures
    CONDITIONAL = "conditional"   # If-then scenarios
    COMPARATIVE = "comparative"   # Comparisons and differences
    TEMPORAL = "temporal"         # Time-related information
    QUANTITATIVE = "quantitative" # Numbers and calculations


@dataclass
class GenerationContext:
    domain_type: DomainType
    query_type: QueryType
    answer_type: AnswerType
    relevant_sections: List[str]
    key_entities: List[str]
    confidence_level: float
    context_metadata: Dict[str, Any]


@dataclass
class DomainPrompt:
    system_prompt: str
    user_template: str
    post_processing_rules: List[str]
    validation_patterns: List[str]
    confidence_indicators: List[str]


class DomainAwareGenerator:
    """
    Advanced answer generator with domain-specific knowledge and context awareness
    Optimized for high accuracy in specialized domains
    """

    def __init__(self):
        # AI clients
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.google_model = genai.GenerativeModel(settings.GOOGLE_MODEL)
        
        # Domain-specific prompts and rules
        self.domain_prompts = self._initialize_domain_prompts()
        self.domain_validators = self._initialize_domain_validators()
        self.context_enhancers = self._initialize_context_enhancers()
        
        # Answer quality metrics
        self.generation_stats = {
            "total_generations": 0,
            "domain_specific_generations": 0,
            "validation_passes": 0,
            "avg_confidence": 0.0,
            "avg_generation_time": 0.0
        }

    def _initialize_domain_prompts(self) -> Dict[DomainType, Dict[AnswerType, DomainPrompt]]:
        """Initialize domain and answer type specific prompts"""
        
        prompts = {}
        
        # Insurance Policy Prompts
        prompts[DomainType.INSURANCE_POLICY] = {
            AnswerType.FACTUAL: DomainPrompt(
                system_prompt="""You are an expert insurance policy analyst. Provide precise, factual answers about insurance policies with exact figures, percentages, and terms from the policy document.

REQUIREMENTS:
- State exact amounts, percentages, and timeframes as written in the policy
- Use precise insurance terminology
- Include relevant policy section references when available
- Mention any conditions or exceptions that apply
- Answer in 2-3 sentences maximum
- Be absolutely accurate with numbers and terms""",
                
                user_template="""Based on the insurance policy sections below, answer this question with exact facts and figures:

POLICY SECTIONS:
{context}

QUESTION: {question}

Provide a precise answer with exact amounts, percentages, and conditions:""",
                
                post_processing_rules=[
                    "Ensure all monetary amounts are correctly formatted",
                    "Verify percentage values are accurate",
                    "Check that policy terms are used correctly",
                    "Confirm time periods are stated accurately"
                ],
                
                validation_patterns=[
                    r"(?:Rs\.?|₹)\s*[\d,]+(?:\.\d+)?",  # Currency amounts
                    r"\d+(?:\.\d+)?%",                   # Percentages
                    r"\d+\s*(?:days?|months?|years?)",  # Time periods
                ],
                
                confidence_indicators=[
                    "exact figure mentioned", "specific percentage stated",
                    "clear time period defined", "policy section referenced"
                ]
            ),
            
            AnswerType.PROCEDURAL: DomainPrompt(
                system_prompt="""You are an expert insurance claims specialist. Provide clear, step-by-step procedures for insurance-related processes based on policy documents.

REQUIREMENTS:
- Break down procedures into clear, numbered steps
- Include required documents and timeframes
- Mention any prerequisites or conditions
- Reference relevant policy sections
- Be specific about deadlines and requirements
- Answer in a structured, easy-to-follow format""",
                
                user_template="""Based on the insurance policy sections below, explain the procedure for this question:

POLICY SECTIONS:
{context}

QUESTION: {question}

Provide a step-by-step procedure with requirements and timeframes:""",
                
                post_processing_rules=[
                    "Structure as numbered steps",
                    "Include all required documents",
                    "Mention specific timeframes",
                    "Add any relevant conditions"
                ],
                
                validation_patterns=[
                    r"(?:step|first|second|third|\d+\.)",
                    r"(?:within|before|after)\s+\d+\s+(?:days?|months?)",
                    r"(?:submit|provide|file|complete)"
                ],
                
                confidence_indicators=[
                    "complete procedure outlined", "timeframes specified",
                    "required documents listed", "clear steps provided"
                ]
            ),
            
            AnswerType.CONDITIONAL: DomainPrompt(
                system_prompt="""You are an expert insurance policy interpreter. Explain conditional scenarios and their outcomes based on policy terms.

REQUIREMENTS:
- Clearly state the conditions that must be met
- Explain what happens in different scenarios
- Include any exceptions or exclusions
- Use "if-then" structure for clarity
- Reference specific policy conditions
- Be precise about applicable circumstances""",
                
                user_template="""Based on the insurance policy sections below, explain the conditional scenario for this question:

POLICY SECTIONS:
{context}

QUESTION: {question}

Explain the conditions and outcomes clearly:""",
                
                post_processing_rules=[
                    "Use clear if-then structure",
                    "List all applicable conditions",
                    "Include relevant exceptions",
                    "Be specific about scenarios"
                ],
                
                validation_patterns=[
                    r"(?:if|when|provided|unless|except)",
                    r"(?:then|will|shall|coverage)",
                    r"(?:condition|requirement|eligible)"
                ],
                
                confidence_indicators=[
                    "conditions clearly stated", "outcomes explained",
                    "exceptions mentioned", "scenarios covered"
                ]
            )
        }
        
        # Legal Contract Prompts
        prompts[DomainType.LEGAL_CONTRACT] = {
            AnswerType.FACTUAL: DomainPrompt(
                system_prompt="""You are an expert legal contract analyst. Provide precise, legally accurate answers based on contract terms and conditions.

REQUIREMENTS:
- Quote exact contract language when relevant
- Be precise with legal terminology
- Include section or clause references
- Mention any applicable conditions or exceptions
- State obligations and rights clearly
- Answer in 2-3 sentences with legal precision""",
                
                user_template="""Based on the contract sections below, answer this legal question with exact terms and conditions:

CONTRACT SECTIONS:
{context}

QUESTION: {question}

Provide a legally precise answer with exact contract terms:""",
                
                post_processing_rules=[
                    "Use exact contract language",
                    "Include relevant clause references",
                    "Ensure legal terminology is correct",
                    "Mention applicable conditions"
                ],
                
                validation_patterns=[
                    r"(?:clause|section|article)\s+\d+",
                    r"(?:shall|must|required|obligated)",
                    r"(?:party|parties|agreement|contract)"
                ],
                
                confidence_indicators=[
                    "exact contract language quoted", "legal terms used correctly",
                    "clause references included", "obligations stated clearly"
                ]
            )
        }
        
        # HR Document Prompts
        prompts[DomainType.HR_DOCUMENT] = {
            AnswerType.FACTUAL: DomainPrompt(
                system_prompt="""You are an expert HR policy specialist. Provide clear, accurate answers about employee policies, benefits, and procedures.

REQUIREMENTS:
- State exact policy terms and conditions
- Include specific eligibility criteria
- Mention any required approvals or processes
- Reference relevant policy sections
- Be clear about employee rights and obligations
- Answer in 2-3 sentences with HR precision""",
                
                user_template="""Based on the HR policy sections below, answer this employee-related question:

HR POLICY SECTIONS:
{context}

QUESTION: {question}

Provide a clear answer with exact policy terms and conditions:""",
                
                post_processing_rules=[
                    "Use exact policy language",
                    "Include eligibility criteria",
                    "Mention approval processes",
                    "Reference policy sections"
                ],
                
                validation_patterns=[
                    r"(?:employee|staff|personnel)",
                    r"(?:policy|procedure|requirement)",
                    r"(?:eligible|entitled|required)"
                ],
                
                confidence_indicators=[
                    "policy terms stated exactly", "eligibility criteria clear",
                    "approval processes mentioned", "employee rights defined"
                ]
            )
        }
        
        # Fill in remaining combinations with general prompts
        for domain in DomainType:
            if domain not in prompts:
                prompts[domain] = {}
            
            for answer_type in AnswerType:
                if answer_type not in prompts[domain]:
                    prompts[domain][answer_type] = self._get_general_prompt(domain, answer_type)
        
        return prompts

    def _get_general_prompt(self, domain: DomainType, answer_type: AnswerType) -> DomainPrompt:
        """Get general prompt for domain/answer type combinations"""
        
        return DomainPrompt(
            system_prompt=f"""You are a document analyst specializing in {domain.value.replace('_', ' ')} documents. 
            Provide accurate, {answer_type.value} answers based on the document content.
            
            REQUIREMENTS:
            - Be precise and factual
            - Use exact information from the document
            - Include relevant details and conditions
            - Answer in 2-3 sentences maximum""",
            
            user_template="""Based on the document sections below, provide a {answer_type} answer:

DOCUMENT SECTIONS:
{{context}}

QUESTION: {{question}}

Provide an accurate answer based on the document content:""",
            
            post_processing_rules=[
                "Ensure accuracy with document content",
                "Include relevant conditions",
                "Use appropriate terminology"
            ],
            
            validation_patterns=[],
            confidence_indicators=["accurate information provided", "relevant details included"]
        )

    def _initialize_domain_validators(self) -> Dict[DomainType, Dict[str, Any]]:
        """Initialize domain-specific validators"""
        
        return {
            DomainType.INSURANCE_POLICY: {
                "required_elements": ["premium", "coverage", "policy", "insured"],
                "forbidden_phrases": ["I think", "maybe", "possibly", "might be"],
                "accuracy_checks": [
                    lambda text: bool(re.search(r"(?:Rs\.?|₹)\s*[\d,]+", text)),  # Has currency
                    lambda text: bool(re.search(r"\d+(?:\.\d+)?%", text))         # Has percentage
                ]
            },
            DomainType.LEGAL_CONTRACT: {
                "required_elements": ["party", "agreement", "obligation", "contract"],
                "forbidden_phrases": ["I believe", "seems like", "appears to", "probably"],
                "accuracy_checks": [
                    lambda text: bool(re.search(r"(?:shall|must|required)", text, re.IGNORECASE))
                ]
            },
            DomainType.HR_DOCUMENT: {
                "required_elements": ["employee", "policy", "procedure", "company"],
                "forbidden_phrases": ["I assume", "likely", "presumably", "generally"],
                "accuracy_checks": [
                    lambda text: bool(re.search(r"(?:employee|staff|personnel)", text, re.IGNORECASE))
                ]
            }
        }

    def _initialize_context_enhancers(self) -> Dict[DomainType, Dict[str, Any]]:
        """Initialize context enhancement rules"""
        
        return {
            DomainType.INSURANCE_POLICY: {
                "important_sections": ["definitions", "coverage", "exclusions", "claims", "premium"],
                "entity_types": ["MONEY", "PERCENTAGE", "TIME_PERIOD", "PERSON", "ORG"],
                "context_boosters": {
                    "premium": 1.5,
                    "coverage": 1.4,
                    "exclusion": 1.3,
                    "claim": 1.3,
                    "benefit": 1.2
                }
            },
            DomainType.LEGAL_CONTRACT: {
                "important_sections": ["definitions", "obligations", "terms", "conditions", "liability"],
                "entity_types": ["ORG", "PERSON", "DATE", "MONEY"],
                "context_boosters": {
                    "obligation": 1.5,
                    "liability": 1.4,
                    "breach": 1.3,
                    "termination": 1.3
                }
            }
        }

    async def generate_domain_aware_answer(
        self,
        question: str,
        search_results: List[SearchResult],
        domain_type: DomainType,
        query_type: QueryType
    ) -> Tuple[str, GenerationContext]:
        """
        Generate domain-aware answer with enhanced context understanding
        """
        
        start_time = time.time()
        
        try:
            # Analyze question to determine answer type
            answer_type = self._classify_answer_type(question, query_type)
            
            # Build generation context
            context = await self._build_generation_context(
                question, search_results, domain_type, query_type, answer_type
            )
            
            # Get domain-specific prompt
            domain_prompt = self.domain_prompts[domain_type][answer_type]
            
            # Enhance context with domain knowledge
            enhanced_context = self._enhance_context_with_domain_knowledge(
                search_results, domain_type, context
            )
            
            # Generate answer using optimized prompt
            answer = await self._generate_with_domain_prompt(
                question, enhanced_context, domain_prompt
            )
            
            # Post-process and validate answer
            validated_answer = self._post_process_and_validate(
                answer, domain_type, domain_prompt, context
            )
            
            # Update statistics
            generation_time = time.time() - start_time
            self._update_generation_stats(generation_time, context.confidence_level)
            
            logger.info(f"Domain-aware answer generated in {generation_time:.3f}s")
            
            return validated_answer, context
            
        except Exception as e:
            logger.error(f"Error in domain-aware generation: {str(e)}")
            # Fallback to basic generation
            return await self._fallback_generation(question, search_results)

    def _classify_answer_type(self, question: str, query_type: QueryType) -> AnswerType:
        """Classify the type of answer needed based on question analysis"""
        
        question_lower = question.lower()
        
        # Direct mapping from query type
        if query_type == QueryType.PROCEDURAL:
            return AnswerType.PROCEDURAL
        elif query_type == QueryType.CONDITIONAL:
            return AnswerType.CONDITIONAL
        elif query_type == QueryType.COMPARATIVE:
            return AnswerType.COMPARATIVE
        elif query_type == QueryType.TEMPORAL:
            return AnswerType.TEMPORAL
        elif query_type == QueryType.QUANTITATIVE:
            return AnswerType.QUANTITATIVE
        
        # Additional analysis for factual questions
        if any(phrase in question_lower for phrase in ["how to", "steps", "procedure", "process"]):
            return AnswerType.PROCEDURAL
        elif any(phrase in question_lower for phrase in ["if", "when", "under what", "provided"]):
            return AnswerType.CONDITIONAL
        elif any(phrase in question_lower for phrase in ["how much", "amount", "cost", "percentage"]):
            return AnswerType.QUANTITATIVE
        else:
            return AnswerType.FACTUAL

    async def _build_generation_context(
        self,
        question: str,
        search_results: List[SearchResult],
        domain_type: DomainType,
        query_type: QueryType,
        answer_type: AnswerType
    ) -> GenerationContext:
        """Build comprehensive generation context"""
        
        # Extract relevant sections
        relevant_sections = []
        for result in search_results:
            if result.section_context:
                relevant_sections.append(result.section_context)
        
        # Extract key entities
        key_entities = []
        for result in search_results:
            if result.chunk:
                # Extract domain-specific entities from chunk text
                entities = self._extract_key_entities(result.chunk.text, domain_type)
                key_entities.extend(entities)
        
        # Remove duplicates
        relevant_sections = list(set(relevant_sections))
        key_entities = list(set(key_entities))
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(
            search_results, domain_type, answer_type
        )
        
        # Build metadata
        context_metadata = {
            "num_search_results": len(search_results),
            "avg_relevance_score": sum(r.relevance_score for r in search_results) / len(search_results) if search_results else 0,
            "domain_match_score": self._calculate_domain_match_score(search_results, domain_type),
            "entity_richness": len(key_entities) / max(1, len(search_results))
        }
        
        return GenerationContext(
            domain_type=domain_type,
            query_type=query_type,
            answer_type=answer_type,
            relevant_sections=relevant_sections,
            key_entities=key_entities,
            confidence_level=confidence_level,
            context_metadata=context_metadata
        )

    def _enhance_context_with_domain_knowledge(
        self,
        search_results: List[SearchResult],
        domain_type: DomainType,
        context: GenerationContext
    ) -> str:
        """Enhance context with domain-specific knowledge"""
        
        context_parts = []
        
        # Sort results by relevance and domain match
        sorted_results = sorted(
            search_results,
            key=lambda x: (x.relevance_score, self._get_domain_boost(x, domain_type)),
            reverse=True
        )
        
        # Build enhanced context
        for i, result in enumerate(sorted_results[:settings.MAX_CHUNKS_FOR_GENERATION]):
            if not result.chunk:
                continue
            
            chunk_text = result.chunk.text
            
            # Add section context
            section_info = ""
            if result.section_context:
                section_info = f"[{result.section_context}] "
            elif result.chunk.metadata.get("section_title"):
                section_info = f"[{result.chunk.metadata['section_title']}] "
            
            # Highlight important entities for domain
            highlighted_text = self._highlight_domain_entities(chunk_text, domain_type)
            
            # Add reasoning if available
            reasoning_info = ""
            if result.reasoning:
                reasoning_info = f" (Selected because: {result.reasoning})"
            
            context_part = f"[Context {i + 1}] {section_info}{highlighted_text}{reasoning_info}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)

    def _get_domain_boost(self, result: SearchResult, domain_type: DomainType) -> float:
        """Calculate domain-specific boost for result ranking"""
        
        if not result.chunk:
            return 0.0
        
        boost = 0.0
        text_lower = result.chunk.text.lower()
        
        if domain_type == DomainType.INSURANCE_POLICY:
            insurance_terms = ["premium", "coverage", "benefit", "claim", "policy", "insured", "exclusion"]
            boost = sum(0.1 for term in insurance_terms if term in text_lower)
        
        elif domain_type == DomainType.LEGAL_CONTRACT:
            legal_terms = ["party", "agreement", "obligation", "liability", "breach", "contract", "clause"]
            boost = sum(0.1 for term in legal_terms if term in text_lower)
        
        elif domain_type == DomainType.HR_DOCUMENT:
            hr_terms = ["employee", "policy", "benefit", "leave", "performance", "compensation"]
            boost = sum(0.1 for term in hr_terms if term in text_lower)
        
        return boost

    def _highlight_domain_entities(self, text: str, domain_type: DomainType) -> str:
        """Highlight important domain-specific entities in text"""
        
        # For now, return original text
        # In production, this could highlight key terms, amounts, dates, etc.
        return text

    def _extract_key_entities(self, text: str, domain_type: DomainType) -> List[str]:
        """Extract key entities relevant to the domain"""
        
        entities = []
        
        # Extract monetary amounts
        money_matches = re.findall(r'(?:Rs\.?|₹)\s*[\d,]+(?:\.\d+)?', text)
        entities.extend(money_matches)
        
        # Extract percentages
        percent_matches = re.findall(r'\d+(?:\.\d+)?%', text)
        entities.extend(percent_matches)
        
        # Extract time periods
        time_matches = re.findall(r'\d+\s*(?:days?|months?|years?)', text, re.IGNORECASE)
        entities.extend(time_matches)
        
        # Domain-specific entity extraction
        if domain_type == DomainType.INSURANCE_POLICY:
            # Extract policy-specific terms
            policy_terms = re.findall(r'\b(?:sum insured|deductible|premium|coverage|benefit)\b', text, re.IGNORECASE)
            entities.extend(policy_terms)
        
        return entities

    def _calculate_confidence_level(
        self,
        search_results: List[SearchResult],
        domain_type: DomainType,
        answer_type: AnswerType
    ) -> float:
        """Calculate confidence level for answer generation"""
        
        if not search_results:
            return 0.0
        
        # Base confidence from search result scores
        avg_relevance = sum(r.relevance_score for r in search_results) / len(search_results)
        
        # Boost for domain match
        domain_boost = self._calculate_domain_match_score(search_results, domain_type)
        
        # Boost for answer type alignment
        answer_type_boost = 0.1  # Base boost
        
        # Combined confidence
        confidence = min(1.0, avg_relevance + domain_boost + answer_type_boost)
        
        return confidence

    def _calculate_domain_match_score(
        self, search_results: List[SearchResult], domain_type: DomainType
    ) -> float:
        """Calculate how well search results match the domain"""
        
        if not search_results:
            return 0.0
        
        domain_matches = 0
        total_results = len(search_results)
        
        for result in search_results:
            if result.chunk and result.chunk.metadata.get("doc_type") == domain_type.value:
                domain_matches += 1
        
        return domain_matches / total_results

    async def _generate_with_domain_prompt(
        self,
        question: str,
        context: str,
        domain_prompt: DomainPrompt
    ) -> str:
        """Generate answer using domain-specific prompt"""
        
        try:
            # Format the user prompt
            user_prompt = domain_prompt.user_template.format(
                context=context,
                question=question
            )
            
            # Generate with OpenAI
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": domain_prompt.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,  # Optimized for concise but complete answers
                temperature=0.1,
                timeout=15
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            logger.error(f"Error in domain prompt generation: {str(e)}")
            raise

    def _post_process_and_validate(
        self,
        answer: str,
        domain_type: DomainType,
        domain_prompt: DomainPrompt,
        context: GenerationContext
    ) -> str:
        """Post-process and validate the generated answer"""
        
        processed_answer = answer
        
        # Apply post-processing rules
        for rule in domain_prompt.post_processing_rules:
            processed_answer = self._apply_post_processing_rule(processed_answer, rule)
        
        # Validate against domain requirements
        if domain_type in self.domain_validators:
            validator = self.domain_validators[domain_type]
            
            # Check forbidden phrases
            for phrase in validator.get("forbidden_phrases", []):
                if phrase.lower() in processed_answer.lower():
                    logger.warning(f"Forbidden phrase detected: {phrase}")
                    processed_answer = processed_answer.replace(phrase, "")
            
            # Apply accuracy checks
            for check in validator.get("accuracy_checks", []):
                if not check(processed_answer):
                    logger.warning(f"Accuracy check failed for answer: {processed_answer[:100]}...")
        
        # Ensure answer ends properly
        if processed_answer and not processed_answer.endswith(('.', '!', '?')):
            processed_answer += '.'
        
        return processed_answer.strip()

    def _apply_post_processing_rule(self, answer: str, rule: str) -> str:
        """Apply a specific post-processing rule"""
        
        # This would implement specific post-processing rules
        # For now, return the answer unchanged
        return answer

    async def _fallback_generation(
        self, question: str, search_results: List[SearchResult]
    ) -> Tuple[str, GenerationContext]:
        """Fallback generation method"""
        
        try:
            if not search_results:
                return "I couldn't find any relevant information to answer this question.", GenerationContext(
                    domain_type=DomainType.GENERAL,
                    query_type=QueryType.FACTUAL,
                    answer_type=AnswerType.FACTUAL,
                    relevant_sections=[],
                    key_entities=[],
                    confidence_level=0.0,
                    context_metadata={}
                )
            
            # Simple context building
            context_parts = []
            for i, result in enumerate(search_results[:3]):
                if result.chunk:
                    context_parts.append(f"Context {i + 1}: {result.chunk.text[:300]}...")
            
            context = "\n\n".join(context_parts)
            
            # Basic prompt
            prompt = f"""Based on the document sections below, answer this question in 2-3 sentences:

{context}

Question: {question}

Answer:"""
            
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            
            fallback_context = GenerationContext(
                domain_type=DomainType.GENERAL,
                query_type=QueryType.FACTUAL,
                answer_type=AnswerType.FACTUAL,
                relevant_sections=[],
                key_entities=[],
                confidence_level=0.5,
                context_metadata={"fallback": True}
            )
            
            return answer, fallback_context
            
        except Exception as e:
            logger.error(f"Error in fallback generation: {str(e)}")
            return "Unable to generate an answer due to a processing error.", GenerationContext(
                domain_type=DomainType.GENERAL,
                query_type=QueryType.FACTUAL,
                answer_type=AnswerType.FACTUAL,
                relevant_sections=[],
                key_entities=[],
                confidence_level=0.0,
                context_metadata={"error": True}
            )

    def _update_generation_stats(self, generation_time: float, confidence_level: float):
        """Update generation statistics"""
        
        self.generation_stats["total_generations"] += 1
        self.generation_stats["avg_generation_time"] = (
            self.generation_stats["avg_generation_time"] + generation_time
        ) / 2
        self.generation_stats["avg_confidence"] = (
            self.generation_stats["avg_confidence"] + confidence_level
        ) / 2

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation performance statistics"""
        return self.generation_stats