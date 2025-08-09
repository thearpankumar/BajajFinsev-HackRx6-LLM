"""
Human-like Answer Generator
Converts raw document chunks into natural, conversational answers
Enhanced with Groq LLM integration
"""

import logging
import re
import time
from typing import Any, List, Dict, Optional

from src.services.groq_service import GroqService

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Service to generate human-like answers from retrieved document chunks
    Enhanced with Groq LLM when available
    """
    
    def __init__(self):
        # Initialize Groq service using config settings
        self.groq_service = GroqService()
        
        self.conversation_starters = [
            "Based on the document",
            "According to the information provided",
            "From what I can find in the document",
            "The document indicates that",
            "As stated in the document"
        ]
        
        self.clarification_phrases = [
            "specifically",
            "in particular", 
            "more precisely",
            "to clarify"
        ]
        
        logger.info(f"AnswerGenerator initialized - Groq available: {self.groq_service.is_available()}")
    
    async def generate_answer(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]], 
        domain: str = "general"
    ) -> str:
        """
        Generate a human-like answer from document chunks
        
        Args:
            question: The user's question
            chunks: List of relevant document chunks with text and metadata
            domain: Detected domain (legal, insurance, hr, compliance)
            
        Returns:
            Human-like answer string
        """
        try:
            if not chunks:
                return self._generate_no_info_response(question)
            
            # Try Groq first if available
            if self.groq_service.is_available():
                try:
                    groq_answer = await self.groq_service.generate_answer(question, chunks, domain)
                    if groq_answer and not groq_answer.startswith("I couldn't find"):
                        logger.info("✅ Using Groq-generated answer")
                        return groq_answer
                except Exception as e:
                    logger.warning(f"Groq generation failed, falling back: {str(e)}")
            
            # Fallback to rule-based generation
            logger.info("📝 Using rule-based answer generation")
            
            # Clean and prepare the content
            cleaned_chunks = self._clean_chunks(chunks)
            
            if not cleaned_chunks:
                return self._generate_no_info_response(question)
            
            # Analyze question type for appropriate response style
            question_type = self._analyze_question_type(question)
            
            # Generate answer based on question type
            answer = self._generate_contextual_answer(question, cleaned_chunks, question_type)
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "I encountered an issue while processing the information. Please try again."
    
    def _clean_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Clean and prepare chunks for answer generation"""
        cleaned = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            if not text:
                continue
            
            # Remove technical markers and formatting
            text = self._clean_text(text)
            
            # Only include chunks with substantial content
            if len(text.strip()) > 20:
                cleaned.append(text.strip())
        
        return cleaned
    
    def _clean_text(self, text: str) -> str:
        """Clean text of technical formatting and markers"""
        # Remove page markers
        text = re.sub(r'=== PAGE \d+ ===', '', text)
        
        # Remove constitutional markers and technical references
        text = re.sub(r'\d+\.\s*Subs\.\s+by.*?(?=\d+\.|$)', '', text, flags=re.MULTILINE)
        text = re.sub(r'\d+\.\s*Ins\.\s+by.*?(?=\d+\.|$)', '', text, flags=re.MULTILINE)
        text = re.sub(r'_+', '', text)
        
        # Remove excessive technical references
        text = re.sub(r'\(w\.e\.f\.\s+\d+-\d+-\d+\)', '', text)
        text = re.sub(r'UIN:\s+\w+', '', text)
        
        # Clean up multiple spaces and line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Remove very technical legal references at start
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^[\d\w]+\.\s*[A-Z][a-z].*by.*Act.*\d+.*for', line):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze question type to determine response style"""
        q_lower = question.lower()
        
        if any(word in q_lower for word in ['what is', 'what are', 'define', 'definition']):
            return 'definition'
        elif any(word in q_lower for word in ['how', 'how to', 'process']):
            return 'process'
        elif any(word in q_lower for word in ['why', 'reason', 'purpose']):
            return 'explanation'
        elif any(word in q_lower for word in ['when', 'time', 'period', 'duration']):
            return 'temporal'
        elif any(word in q_lower for word in ['where', 'location', 'place']):
            return 'location'
        elif any(word in q_lower for word in ['who', 'person', 'people']):
            return 'person'
        elif any(word in q_lower for word in ['can i', 'am i', 'is it', 'legal', 'allowed']):
            return 'permission'
        elif any(word in q_lower for word in ['list', 'types', 'kinds', 'categories']):
            return 'enumeration'
        else:
            return 'general'
    
    def _generate_contextual_answer(self, question: str, chunks: List[str], question_type: str) -> str:
        """Generate answer based on question type and context"""
        
        # Extract key information from chunks
        relevant_info = self._extract_key_information(chunks, question_type)
        
        if not relevant_info:
            return self._generate_no_info_response(question)
        
        # Generate answer based on question type
        if question_type == 'definition':
            return self._generate_definition_answer(relevant_info)
        elif question_type == 'process':
            return self._generate_process_answer(relevant_info)
        elif question_type == 'enumeration':
            return self._generate_enumeration_answer(relevant_info)
        elif question_type == 'permission':
            return self._generate_permission_answer(relevant_info, question)
        elif question_type == 'temporal':
            return self._generate_temporal_answer(relevant_info)
        else:
            return self._generate_general_answer(relevant_info)
    
    def _extract_key_information(self, chunks: List[str], question_type: str) -> str:
        """Extract the most relevant information from chunks"""
        
        # Combine chunks and find most relevant sentences
        combined_text = ' '.join(chunks)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', combined_text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return ""
        
        # Take most relevant sentences (first 2-3)
        key_sentences = sentences[:3]
        
        # Clean and format
        result = '. '.join(key_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def _generate_definition_answer(self, info: str) -> str:
        """Generate definition-style answer"""
        return f"According to the document, {info}"
    
    def _generate_process_answer(self, info: str) -> str:
        """Generate process-style answer"""
        return f"Based on the document, the process involves: {info}"
    
    def _generate_enumeration_answer(self, info: str) -> str:
        """Generate enumeration-style answer"""
        return f"The document identifies the following: {info}"
    
    def _generate_permission_answer(self, info: str, question: str) -> str:
        """Generate permission/legal-style answer"""
        if 'prohibited' in info.lower() or 'not allowed' in info.lower():
            return f"According to the document, this is not permitted. {info}"
        elif 'allowed' in info.lower() or 'permitted' in info.lower():
            return f"Yes, this is permitted according to the document. {info}"
        else:
            return f"Regarding your question, the document states: {info}"
    
    def _generate_temporal_answer(self, info: str) -> str:
        """Generate time-related answer"""
        return f"According to the document, {info}"
    
    def _generate_general_answer(self, info: str) -> str:
        """Generate general answer"""
        return f"Based on the document, {info}"
    
    def _generate_no_info_response(self, question: str) -> str:
        """Generate appropriate response when no information is found"""
        responses = [
            "I couldn't find specific information about this in the document.",
            "The document doesn't contain clear information about this topic.",
            "I don't see specific details about this in the available document.",
            "This information doesn't appear to be covered in the document."
        ]
        
        # Choose response based on question type for variety
        import hashlib
        question_hash = int(hashlib.md5(question.encode()).hexdigest()[:8], 16)
        return responses[question_hash % len(responses)]
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get answer generation statistics"""
        return {
            "service": "AnswerGenerator",
            "supported_question_types": [
                "definition", "process", "enumeration", 
                "permission", "temporal", "general"
            ],
            "features": [
                "Human-like responses",
                "Context-aware formatting", 
                "Question type analysis",
                "Text cleaning and formatting"
            ]
        }