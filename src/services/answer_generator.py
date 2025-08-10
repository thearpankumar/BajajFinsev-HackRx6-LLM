"""
Human-like Answer Generator
Converts raw document chunks into natural, conversational answers
Enhanced with OpenAI GPT-4o mini integration
"""

import logging
import re
import time
from typing import Any, List, Dict, Optional

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from src.core.config import config
from src.services.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Service to generate human-like answers from retrieved document chunks
    Enhanced with OpenAI GPT-4o mini when available
    """
    
    def __init__(self):
        # Initialize OpenAI client
        self.openai_client = None
        if HAS_OPENAI and config.openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        
        # Initialize language detector for language-aware responses
        self.language_detector = LanguageDetector()
        
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
        
        logger.info(f"AnswerGenerator initialized - OpenAI available: {self.openai_client is not None}")
    
    async def generate_answer(
        self, 
        question: str, 
        chunks: List[Dict[str, Any]], 
        domain: str = "general",
        target_language: Optional[str] = None
    ) -> str:
        """
        Generate a human-like answer from document chunks
        
        Args:
            question: The user's question
            chunks: List of relevant document chunks with text and metadata
            domain: Detected domain (legal, insurance, hr, compliance)
            target_language: Target language for the response (auto-detected if None)
            
        Returns:
            Human-like answer string
        """
        try:
            if not chunks:
                return self._generate_no_info_response(question, target_language)
            
            # Auto-detect query language if not specified
            if not target_language:
                detected = self.language_detector.detect_language(question)
                target_language = detected.get("detected_language", "en")
                logger.info(f"🔍 Detected query language: {target_language}")
            
            # Try OpenAI first if available
            if self.openai_client:
                try:
                    openai_answer = await self._generate_with_openai(question, chunks, domain, target_language)
                    if openai_answer and not openai_answer.startswith("I couldn't find"):
                        logger.info(f"✅ Using OpenAI-generated answer in {target_language}")
                        return openai_answer
                except Exception as e:
                    logger.warning(f"OpenAI generation failed, falling back: {str(e)}")
            
            # Fallback to rule-based generation
            logger.info("📝 Using rule-based answer generation")
            
            # Clean and prepare the content
            cleaned_chunks = self._clean_chunks(chunks)
            
            if not cleaned_chunks:
                return self._generate_no_info_response(question, target_language)
            
            # Analyze question type for appropriate response style
            question_type = self._analyze_question_type(question)
            
            # Generate answer based on question type
            answer = self._generate_contextual_answer(question, cleaned_chunks, question_type, target_language)
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return "I encountered an issue while processing the information. Please try again."
    
    def _clean_chunks(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Clean and prepare chunks for answer generation"""
        cleaned = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            if not text or not isinstance(text, str):
                continue
            
            # Remove technical markers and formatting
            text = self._clean_text(text)
            
            # Only include chunks with substantial content
            if len(text.strip()) > 20:
                cleaned.append(text.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_cleaned = []
        for text in cleaned:
            # Use first 100 characters for duplicate detection
            text_key = text[:100].lower().strip()
            if text_key not in seen:
                seen.add(text_key)
                unique_cleaned.append(text)
        
        return unique_cleaned
    
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
    
    def _generate_contextual_answer(self, question: str, chunks: List[str], question_type: str, target_language: str = "en") -> str:
        """Generate answer based on question type and context"""
        
        # Extract key information from chunks
        relevant_info = self._extract_key_information(chunks, question_type)
        
        if not relevant_info:
            return self._generate_no_info_response(question, target_language)
        
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
    
    def _generate_no_info_response(self, question: str, target_language: str = "en") -> str:
        """Generate appropriate response when no information is found"""
        if target_language == "ml":
            responses = [
                "ഈ വിഷയത്തെക്കുറിച്ച് ഡോക്യുമെന്റിൽ വ്യക്തമായ വിവരങ്ങൾ കണ്ടെത്താനായില്ല.",
                "ഡോക്യുമെന്റിൽ ഈ വിഷയത്തെക്കുറിച്ചുള്ള വിശദാംശങ്ങൾ ഇല്ല.",
                "ലഭ്യമായ ഡോക്യുമെന്റിൽ ഈ വിവരങ്ങൾ കാണാൻ കഴിയുന്നില്ല.",
                "ഈ വിവരം ഡോക്യുമെന്റിൽ ഉൾപ്പെടുത്തിയിട്ടില്ലെന്ന് തോന്നുന്നു."
            ]
        else:
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
    
    async def _generate_with_openai(self, question: str, chunks: List[Dict[str, Any]], domain: str, target_language: str = "en") -> str:
        """Generate answer using OpenAI GPT-4o mini"""
        try:
            # Prepare context from chunks
            context_text = "\n\n".join([
                f"Document excerpt {i+1}:\n{chunk['text'][:500]}..." 
                for i, chunk in enumerate(chunks[:3])
            ])
            
            # Create domain-specific prompt
            domain_context = {
                "legal": "You are a legal assistant providing accurate information about constitutional and legal matters.",
                "insurance": "You are an insurance expert explaining policy terms and coverage details.",
                "hr": "You are an HR specialist providing information about workplace policies and employee rights.",
                "compliance": "You are a compliance expert explaining regulatory requirements and procedures.",
                "general": "You are a helpful assistant providing clear and accurate information."
            }
            
            # Language-specific system prompts
            language_instructions = {
                "ml": " Respond in Malayalam language. Provide natural, fluent Malayalam responses.",
                "en": " Respond in English language."
            }
            
            system_prompt = domain_context.get(domain, domain_context["general"]) + language_instructions.get(target_language, language_instructions["en"])
            
            # Language-specific user prompt
            if target_language == "ml":
                user_prompt = f"""താഴെ കൊടുത്തിരിക്കുന്ന ഡോക്യുമെന്റ് ഭാഗങ്ങളുടെ അടിസ്ഥാനത്തിൽ, ചോദ്യത്തിന് വ്യക്തവും കൃത്യവുമായ ഉത്തരം നൽകുക.

ചോദ്യം: {question}

ഡോക്യുമെന്റ് ഭാഗങ്ങൾ:
{context_text}

നിർദ്ദേശങ്ങൾ:
- ഡോക്യുമെന്റിലെ ഉള്ളടക്കത്തെ അടിസ്ഥാനമാക്കി നേരിട്ടുള്ള, സഹായകരമായ ഉത്തരം നൽകുക
- വിവരങ്ങൾ ഡോക്യുമെന്റിൽ ഇല്ലെങ്കിൽ അത് വ്യക്തമാക്കുക
- സ്വാഭാവികവും കൃത്യവുമായ മലയാളത്തിൽ ഉത്തരം നൽകുക
- നിയമപരമായ ചോദ്യങ്ങൾക്ക് ആർട്ടിക്കിൾ നമ്പറുകളും നിയമപദങ്ങളും കൃത്യമാക്കുക"""
            else:
                user_prompt = f"""Based on the following document excerpts, please provide a clear, concise, and accurate answer to the question.

QUESTION: {question}

DOCUMENT EXCERPTS:
{context_text}

Instructions:
- Provide a direct, helpful answer based on the document content
- If the information is not in the documents, clearly state that
- Keep the response conversational but accurate
- For legal questions, be precise about article numbers and legal terms
- Cite specific sections when relevant"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            return None
    
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