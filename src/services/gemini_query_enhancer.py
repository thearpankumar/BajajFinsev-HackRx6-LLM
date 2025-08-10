"""
Gemini Query Enhancement Service
Uses Gemini LLM to enhance user queries with domain-specific keywords for better retrieval
"""

import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from src.core.config import config
from src.core.llm_config import llm_config_manager
from src.services.gemini_service import GeminiService, QueryContext

logger = logging.getLogger(__name__)


@dataclass
class EnhancedQuery:
    """Enhanced query result"""
    original_query: str
    enhanced_query: str
    domain: str
    added_keywords: List[str]
    confidence: float
    processing_time: float
    enhancement_method: str = "gemini"


class GeminiQueryEnhancer:
    """
    Uses Gemini LLM to enhance queries with domain-specific keywords
    Specifically designed to improve retrieval accuracy in insurance, legal, HR, and compliance domains
    """

    def __init__(self):
        # Initialize Gemini service for query processing
        self.gemini_service = GeminiService()
        self.is_initialized = False
        
        # Domain-specific keyword maps for context
        self.domain_keywords = {
            "insurance": {
                "policy": ["coverage", "benefits", "premium", "deductible", "claim", "policyholder"],
                "medical": ["healthcare", "hospital", "treatment", "medical expenses", "cashless", "reimbursement"],
                "claim": ["settlement", "documentation", "approval", "processing", "investigation"],
                "coverage": ["scope", "exclusions", "limitations", "terms", "conditions"]
            },
            "legal": {
                "constitution": ["article", "amendment", "fundamental rights", "directive principles", "constitutional law"],
                "rights": ["fundamental rights", "civil liberties", "human rights", "legal protection", "constitutional guarantee"],
                "law": ["statute", "legislation", "legal provision", "judicial interpretation", "court ruling"],
                "legal procedure": ["due process", "legal proceedings", "court procedure", "judicial process"]
            },
            "hr": {
                "employment": ["job", "workplace", "employee rights", "labor law", "employment contract"],
                "leave": ["vacation", "sick leave", "maternity leave", "paid time off", "absence policy"],
                "performance": ["appraisal", "evaluation", "review", "improvement", "development"],
                "workplace": ["office policy", "code of conduct", "workplace harassment", "employee handbook"]
            },
            "compliance": {
                "regulation": ["regulatory compliance", "audit", "governance", "risk management", "compliance framework"],
                "audit": ["internal audit", "external audit", "compliance check", "audit trail", "audit report"],
                "governance": ["corporate governance", "board oversight", "risk committee", "compliance officer"],
                "violation": ["non-compliance", "breach", "penalty", "corrective action", "remediation"]
            }
        }
        
        # Performance tracking
        self.total_enhancements = 0
        self.total_processing_time = 0.0
        self.domain_distribution = {"insurance": 0, "legal": 0, "hr": 0, "compliance": 0, "general": 0}
        
        logger.info("GeminiQueryEnhancer initialized for domain-specific query enhancement")

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the query enhancement service"""
        try:
            logger.info("🔄 Initializing Gemini Query Enhancement Service...")
            start_time = time.time()
            
            # Initialize Gemini service
            gemini_result = await self.gemini_service.initialize()
            
            if gemini_result["status"] != "success":
                return {
                    "status": "error", 
                    "error": f"Gemini service initialization failed: {gemini_result.get('error', 'Unknown error')}"
                }
            
            self.is_initialized = True
            initialization_time = time.time() - start_time
            
            result = {
                "status": "success",
                "message": f"Gemini Query Enhancement Service initialized in {initialization_time:.2f}s",
                "gemini_service": gemini_result,
                "supported_domains": list(self.domain_keywords.keys()),
                "enhancement_features": [
                    "Domain-specific keyword addition",
                    "Query expansion and clarification", 
                    "Context-aware enhancement",
                    "Multi-domain support"
                ],
                "initialization_time": initialization_time
            }
            
            logger.info(f"✅ Gemini Query Enhancement Service ready")
            return result
            
        except Exception as e:
            error_msg = f"Query enhancement service initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {"status": "error", "error": error_msg}

    async def enhance_query(
        self, 
        query: str, 
        detected_domain: Optional[str] = None,
        context_info: Optional[Dict[str, Any]] = None
    ) -> EnhancedQuery:
        """
        Enhance user query with domain-specific keywords using Gemini
        
        Args:
            query: Original user query
            detected_domain: Pre-detected domain (insurance, legal, hr, compliance)
            context_info: Additional context information
            
        Returns:
            EnhancedQuery with domain keywords and expansions
        """
        logger.info(f"🔍 Enhancing query: '{query[:50]}...' (domain: {detected_domain})")
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Create enhancement prompt for Gemini
            enhancement_prompt = self._create_enhancement_prompt(query, detected_domain, context_info)
            
            # Get enhanced query from Gemini
            context = QueryContext(
                user_id=None,
                session_id=None,
                query_type="enhancement",
                domain_context=detected_domain,
                conversation_history=[],
                retrieved_documents=[],
                language="en"
            )
            
            gemini_response = await self.gemini_service.generate_response(
                enhancement_prompt,
                context=context,
                response_type="analytical",
                use_cache=True
            )
            
            # Parse enhanced query from Gemini response
            enhanced_query_result = self._parse_gemini_enhancement(
                query, 
                gemini_response.response_text, 
                detected_domain or "general"
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            enhanced_query_result.processing_time = processing_time
            self.total_enhancements += 1
            self.total_processing_time += processing_time
            self.domain_distribution[detected_domain or "general"] += 1
            
            logger.info(f"✅ Query enhanced in {processing_time:.2f}s: '{enhanced_query_result.enhanced_query[:50]}...'")
            return enhanced_query_result
            
        except Exception as e:
            error_msg = f"Query enhancement failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            # Return original query as fallback
            return EnhancedQuery(
                original_query=query,
                enhanced_query=query,
                domain=detected_domain or "general", 
                added_keywords=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                enhancement_method="fallback"
            )

    def _create_enhancement_prompt(
        self, 
        query: str, 
        domain: Optional[str], 
        context_info: Optional[Dict[str, Any]]
    ) -> str:
        """Create domain-specific enhancement prompt for Gemini"""
        
        # Always use "legal" domain since domain detection is unreliable
        fixed_domain = "legal"
        domain_context = self._get_domain_context(fixed_domain)
        
        prompt = f"""You are a query enhancement expert specializing in {fixed_domain} domain queries. Your task is to enhance the user's query by adding relevant domain-specific keywords and expanding concepts for better document retrieval.

DOMAIN: {fixed_domain}

DOMAIN-SPECIFIC KEYWORDS AND CONCEPTS:
{domain_context}

ORIGINAL QUERY: "{query}"

ENHANCEMENT INSTRUCTIONS:
1. Analyze the query and identify the core intent and concepts
2. Add relevant domain-specific keywords that would help find related documents
3. Expand abbreviations and add synonyms for better matching
4. Include related legal/regulatory terms if applicable
5. Maintain the original meaning while making it more comprehensive for search

IMPORTANT GUIDELINES:
- Keep the enhanced query natural and readable
- Focus on keywords that would appear in relevant documents
- Add terms that subject matter experts would use
- Include both common terms and technical terminology
- Don't change the core question or intent

Please provide the enhanced query in this exact format:

ENHANCED_QUERY: [your enhanced version here]
ADDED_KEYWORDS: [list the specific keywords/terms you added]
CONFIDENCE: [rate 0.0-1.0 how confident you are in the enhancement]
EXPLANATION: [brief explanation of your enhancement strategy]"""

        return prompt

    def _get_domain_context(self, domain: Optional[str]) -> str:
        """Get domain-specific context for the prompt"""
        if not domain or domain not in self.domain_keywords:
            return "General domain - add relevant technical terms and synonyms"
        
        domain_info = self.domain_keywords[domain]
        context_parts = []
        
        for category, keywords in domain_info.items():
            context_parts.append(f"- {category.title()}: {', '.join(keywords)}")
        
        return "\n".join(context_parts)

    def _parse_gemini_enhancement(
        self, 
        original_query: str, 
        gemini_response: str, 
        domain: str
    ) -> EnhancedQuery:
        """Parse Gemini's enhancement response"""
        try:
            # Initialize default values
            enhanced_query = original_query
            added_keywords = []
            confidence = 0.5
            
            # Parse structured response
            lines = gemini_response.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith("ENHANCED_QUERY:"):
                    enhanced_query = line.replace("ENHANCED_QUERY:", "").strip()
                elif line.startswith("ADDED_KEYWORDS:"):
                    keywords_text = line.replace("ADDED_KEYWORDS:", "").strip()
                    if keywords_text and keywords_text != "[]":
                        # Parse keywords (handle different formats)
                        keywords_text = keywords_text.strip("[]")
                        added_keywords = [k.strip().strip('"\'') for k in keywords_text.split(',') if k.strip()]
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.replace("CONFIDENCE:", "").strip())
                    except ValueError:
                        confidence = 0.5
            
            # Fallback parsing if structured format not found
            if enhanced_query == original_query:
                # Try to extract enhanced query from free text
                enhanced_query = self._extract_query_from_text(gemini_response, original_query)
            
            return EnhancedQuery(
                original_query=original_query,
                enhanced_query=enhanced_query,
                domain=domain,
                added_keywords=added_keywords,
                confidence=confidence,
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse Gemini enhancement: {str(e)}")
            return EnhancedQuery(
                original_query=original_query,
                enhanced_query=original_query,
                domain=domain,
                added_keywords=[],
                confidence=0.0,
                processing_time=0.0
            )

    def _extract_query_from_text(self, text: str, original_query: str) -> str:
        """Extract enhanced query from free text response"""
        try:
            # Look for common patterns in response
            text_lower = text.lower()
            
            # Try to find enhanced query patterns
            patterns = [
                "enhanced query:",
                "improved query:",
                "better query:",
                "suggested query:",
                "refined query:"
            ]
            
            for pattern in patterns:
                if pattern in text_lower:
                    start_idx = text_lower.find(pattern) + len(pattern)
                    remaining_text = text[start_idx:].strip()
                    
                    # Extract first sentence or line
                    end_markers = ['\n', '.', '?', '!']
                    end_idx = len(remaining_text)
                    
                    for marker in end_markers:
                        marker_idx = remaining_text.find(marker)
                        if marker_idx != -1 and marker_idx < end_idx:
                            end_idx = marker_idx + (1 if marker in '.?!' else 0)
                    
                    enhanced = remaining_text[:end_idx].strip().strip('"\'')
                    if enhanced and enhanced != original_query:
                        return enhanced
            
            # If no pattern found, return original
            return original_query
            
        except Exception:
            return original_query

    def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get query enhancement statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_enhancements
            if self.total_enhancements > 0 else 0.0
        )
        
        return {
            "service_status": "active" if self.is_initialized else "inactive",
            "total_enhancements": self.total_enhancements,
            "total_processing_time": round(self.total_processing_time, 2),
            "average_processing_time": round(avg_processing_time, 3),
            "domain_distribution": self.domain_distribution,
            "supported_domains": list(self.domain_keywords.keys()),
            "enhancement_features": [
                "Domain-specific keyword addition",
                "Query expansion and clarification",
                "Context-aware enhancement", 
                "Multi-domain support"
            ],
            "gemini_service_stats": self.gemini_service.get_service_stats() if self.is_initialized else {}
        }


# Global instance
gemini_query_enhancer = GeminiQueryEnhancer()