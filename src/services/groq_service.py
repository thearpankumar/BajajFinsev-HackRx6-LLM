"""
Groq LLM Service
Integration with Groq's OpenAI GPT-OSS-120B model for enhanced answer generation
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from src.core.config import config
from src.core.llm_config import llm_config_manager

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("Groq library not available. Install with: pip install groq")


class GroqService:
    """
    Service for generating human-like answers using Groq LLM from config
    """
    
    def __init__(self):
        # Use centralized config manager for consistent config access
        provider, model_name, api_key = llm_config_manager.get_response_llm_config()
        
        if provider != "groq":
            logger.warning(f"Expected Groq provider but got {provider} for response LLM")
        
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self.total_requests = 0
        self.total_tokens_used = 0
        self.total_response_time = 0.0
        
        if not self.api_key:
            logger.warning("⚠️ No Groq API key in config. Set GROQ_API_KEY environment variable.")
            return
        
        if not HAS_GROQ:
            logger.error("❌ Groq library not installed")
            return
        
        try:
            self.client = Groq(api_key=self.api_key)
            logger.info(f"✅ Groq service initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq client: {str(e)}")
    
    
    def is_available(self) -> bool:
        """Check if Groq service is available"""
        return self.client is not None and HAS_GROQ
    
    async def generate_answer(
        self, 
        question: str, 
        context_chunks: List[Dict[str, Any]], 
        domain: str = "general"
    ) -> str:
        """
        Generate human-like answer using Groq GPT-OSS-120B
        
        Args:
            question: User's question
            context_chunks: Retrieved document chunks with text and metadata
            domain: Detected domain (legal, insurance, hr, compliance)
            
        Returns:
            Human-like answer string
        """
        if not self.is_available():
            logger.warning("⚠️ Groq service not available, falling back to basic generation")
            return self._fallback_answer_generation(question, context_chunks)
        
        try:
            start_time = time.time()
            
            # Build context from chunks
            context_text = self._build_context_text(context_chunks)
            
            if not context_text.strip():
                return f"I couldn't find specific information about this in the available documents."
            
            # Create domain-specific prompt
            prompt = self._create_domain_prompt(question, context_text, domain)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt(domain)
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=500,
                temperature=0.3,  # Lower temperature for more focused answers
                top_p=0.9,
                stream=False
            )
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Track metrics
            response_time = time.time() - start_time
            self.total_requests += 1
            self.total_response_time += response_time
            if hasattr(response, 'usage'):
                self.total_tokens_used += response.usage.total_tokens
            
            logger.info(f"✅ Generated answer via Groq in {response_time:.2f}s")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Groq answer generation failed: {str(e)}")
            return self._fallback_answer_generation(question, context_chunks)
    
    def _build_context_text(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Build clean context text from chunks"""
        if not context_chunks:
            return ""
        
        # Extract and clean text from chunks
        context_parts = []
        for chunk in context_chunks[:3]:  # Use top 3 most relevant chunks
            text = chunk.get("text", "").strip()
            if text and len(text) > 50:  # Only include substantial chunks
                # Basic cleaning
                text = text.replace("=== PAGE", "").replace("===", "")
                text = " ".join(text.split())  # Normalize whitespace
                context_parts.append(text)
        
        return "\n\n".join(context_parts)
    
    def _get_system_prompt(self, domain: str) -> str:
        """Get domain-specific system prompt"""
        base_prompt = """You are an expert assistant specializing in providing clear, accurate, and helpful answers based on document content. 

Your role is to:
1. Provide direct, conversational answers based on the given context
2. Be precise and factual while remaining human-friendly
3. Acknowledge when information is not available in the context
4. Use appropriate domain terminology correctly
5. Keep responses concise but complete"""

        domain_specific = {
            "legal": "\n\nYou specialize in constitutional law, legal rights, and legal procedures. Explain legal concepts clearly for non-lawyers while maintaining accuracy.",
            
            "insurance": "\n\nYou specialize in insurance policies, coverage, claims, and benefits. Help users understand complex insurance terms in simple language.",
            
            "hr": "\n\nYou specialize in human resources policies, employment law, workplace procedures, and employee benefits. Provide practical HR guidance.",
            
            "compliance": "\n\nYou specialize in regulatory compliance, audit procedures, governance, and risk management. Explain compliance requirements clearly."
        }
        
        return base_prompt + domain_specific.get(domain, "")
    
    def _create_domain_prompt(self, question: str, context: str, domain: str) -> str:
        """Create domain-optimized prompt"""
        
        domain_instructions = {
            "legal": "Based on the legal document provided, answer the question about constitutional or legal matters clearly and accurately.",
            "insurance": "Based on the insurance policy document, explain the coverage, benefits, or procedures clearly.",
            "hr": "Based on the HR policy document, explain the employment-related information clearly and practically.", 
            "compliance": "Based on the compliance document, explain the regulatory requirements or procedures clearly."
        }
        
        instruction = domain_instructions.get(domain, "Based on the document provided, answer the question clearly and accurately.")
        
        return f"""{instruction}

CONTEXT FROM DOCUMENT:
{context}

QUESTION:
{question}

Please provide a clear, direct answer based on the context above. If the specific information is not in the context, say so clearly."""

    def _fallback_answer_generation(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Fallback answer generation when Groq is not available"""
        if not context_chunks:
            return "I couldn't find specific information about this in the available documents."
        
        # Simple extraction from first relevant chunk
        first_chunk = context_chunks[0]
        text = first_chunk.get("text", "").strip()
        
        if text:
            # Basic cleaning
            text = text.replace("=== PAGE", "").replace("===", "")
            text = " ".join(text.split())  # Normalize whitespace
            
            # Truncate if too long
            if len(text) > 300:
                text = text[:300] + "..."
            
            return f"Based on the document: {text}"
        else:
            return "I couldn't find specific information about this in the available documents."
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        avg_response_time = (
            self.total_response_time / self.total_requests 
            if self.total_requests > 0 else 0.0
        )
        
        return {
            "service": "GroqService",
            "model": self.model_name,
            "available": self.is_available(),
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "total_response_time": round(self.total_response_time, 2),
            "average_response_time": round(avg_response_time, 3),
            "supported_domains": ["legal", "insurance", "hr", "compliance", "general"],
            "features": [
                "Domain-specific prompting",
                "Context-aware generation", 
                "Fallback support",
                "Performance tracking"
            ]
        }