"""
Hybrid Document Matcher with RAG Fallback
First tries to match questions in JSON, then falls back to RAG for unmatched questions
Optimized for speed with configurable fallback behavior
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import os
from urllib.parse import urlparse, unquote

from src.core.rag_engine import RAGEngine
from src.core.multi_format_processor import MultiFormatProcessor
from src.core.config import settings

logger = logging.getLogger(__name__)


class HybridDocumentMatcher:
    """
    Hybrid system that combines document-specific JSON matching with RAG fallback
    """
    
    def __init__(self):
        
        # Initialize RAG engine for fallback (lazy initialization)
        self.rag_engine: Optional[RAGEngine] = None
        self.rag_initialized = False
        
        # Initialize multi-format processor
        self.multi_format_processor = MultiFormatProcessor()
        
        # Performance tracking
        self.stats = {
            'total_questions': 0,
            'rag_fallbacks': 0,
            'no_answers': 0,
            'avg_rag_time': 0,
            'format_support_used': {}
        }
        
        logger.info("✅ Hybrid Document Matcher initialized")

    async def _initialize_rag_engine(self):
        """Lazy initialization of RAG engine"""
        if not self.rag_initialized and settings.ENABLE_FALLBACK_RAG:
            try:
                logger.info("🧠 Initializing RAG engine for fallback...")
                from src.core.rag_engine import RAGEngine
                self.rag_engine = RAGEngine()
                await self.rag_engine.initialize()
                self.rag_initialized = True
                logger.info("✅ RAG engine initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize RAG engine: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                self.rag_engine = None
                self.rag_initialized = False

    def extract_document_name_from_url(self, document_url: str) -> tuple[str, str]:
        """
        Extract document name from URL.
        This is a placeholder as the original logic was removed.
        """
        try:
            # Parse URL to get the filename
            parsed_url = urlparse(document_url)
            path = unquote(parsed_url.path)  # Decode URL encoding
            filename = os.path.basename(path).lower()
            return filename, "default"
        except Exception as e:
            print(f"❌ Error extracting document name: {str(e)}")
            return "unknown_document", 'default'

    async def _process_question_with_rag(
        self, question: str, document_url: str
    ) -> Tuple[Optional[str], bool]:
        """
        Process question using RAG engine fallback
        Returns (answer, success)
        """
        if not settings.ENABLE_FALLBACK_RAG:
            logger.info("❌ RAG fallback disabled in settings")
            return None, False
        
        try:
            # Initialize RAG engine if needed
            if not self.rag_initialized:
                logger.info("🔄 Initializing RAG engine for fallback...")
                await self._initialize_rag_engine()
            
            if not self.rag_engine:
                logger.error("❌ RAG engine not available")
                return None, False
            
            logger.info(f"🧠 Using RAG fallback for question: {question}")
            logger.info(f"📄 Document URL: {document_url}")
            start_time = time.time()
            
            # Use RAG engine to process single question
            result = await self.rag_engine.analyze_document(
                document_url=document_url,
                questions=[question]
            )
            
            processing_time = time.time() - start_time
            self.stats['avg_rag_time'] = (
                self.stats['avg_rag_time'] + processing_time
            ) / 2
            
            logger.info(f"⏱️ RAG processing took {processing_time:.2f} seconds")
            logger.info(f"📊 RAG result: {result}")
            
            # Extract answer from result
            if result and 'answers' in result and result['answers']:
                answer = result['answers'][0]
                logger.info(f"✅ RAG answer: {answer[:100]}...")
                return answer, True
            else:
                logger.warning("⚠️ RAG returned empty or invalid result")
                return None, False
                
        except Exception as e:
            logger.error(f"❌ RAG fallback failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, False

    def _get_file_extension(self, document_url: str) -> str:
        """Extract file extension from URL"""
        try:
            from urllib.parse import urlparse
            import os
            parsed_url = urlparse(document_url)
            path = parsed_url.path
            filename = os.path.basename(path)
            extension = os.path.splitext(filename)[1].lower().lstrip('.')
            return extension if extension else 'unknown'
        except Exception:
            return 'unknown'

    async def _is_file_too_large(self, document_url: str) -> bool:
        """Check if file is too large based on config settings"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.head(document_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content_length = response.headers.get('content-length')
                        if content_length:
                            size_bytes = int(content_length)
                            size_mb = size_bytes / (1024 * 1024)
                            logger.info(f"📏 File size: {size_mb:.2f} MB")
                            return size_mb > settings.MAX_FILE_SIZE_MB
            return False
        except Exception as e:
            logger.warning(f"⚠️ Could not check file size: {str(e)}")
            return False

    async def _answer_with_llm_knowledge_api(self, question: str) -> str:
        """
        Answer question using LLM's own knowledge via OpenAI API
        Used when document cannot be processed (unsupported format or too large)
        """
        try:
            logger.info(f"🧠 Answering with LLM API knowledge: {question}")
            
            # Initialize OpenAI client if needed
            if not hasattr(self, 'openai_client'):
                import openai
                self.openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Create a prompt that asks the LLM to use its own knowledge
            prompt = f"""You are a helpful AI assistant. Please answer the following question using your own knowledge and training data. Be concise and informative.

Question: {question}

Please provide a clear, accurate answer based on your knowledge. If the question has multiple parts, address each part. Keep your response concise (2-3 sentences maximum)."""

            # Use OpenAI for generation
            response = await self.openai_client.chat.completions.create(
                model=settings.OPENAI_GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that provides accurate information based on your training data. Keep responses concise and informative."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=settings.MAX_GENERATION_TOKENS,
                temperature=settings.GENERATION_TEMPERATURE,
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"✅ Generated answer using LLM API: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error generating answer with LLM API: {str(e)}")
            return f"I apologize, but I cannot provide an answer to '{question}' at the moment due to technical limitations."

    async def analyze_document(
        self, document_url: str, questions: List[str]
    ) -> Dict[str, Any]:
        """
        Main analysis method with RAG approach
        """
        logger.info("\n🔍 RAG ANALYSIS STARTED")
        logger.info(f"Document URL: {document_url}")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"Fallback RAG enabled: {settings.ENABLE_FALLBACK_RAG}")
        
        start_time = time.time()
        
        document_name, document_key = self.extract_document_name_from_url(document_url)
        
        # Check if file format is supported OR file is too large - use LLM knowledge for ANY question
        file_extension = self._get_file_extension(document_url)
        supported_formats = {'pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'pptx', 'ppt'}
        
        is_large_file = await self._is_file_too_large(document_url)
        
        if file_extension not in supported_formats or is_large_file:
            reason = f"Unsupported file format: .{file_extension}" if file_extension not in supported_formats else "File too large"
            logger.info(f"🚫 {reason}")
            logger.info("🧠 Using LLM's own knowledge for ALL questions (no document processing)")
            
            answers = []
            for question in questions:
                answer = await self._answer_with_llm_knowledge_api(question)
                answers.append(answer)
            
            processing_time = time.time() - start_time
            
            return {
                "answers": answers,
                "document_url": document_url,
                "document_name": document_name,
                "document_key": document_key,
                "processing_time": processing_time,
                "rag_fallbacks": 0,
                "no_answers": 0,
                "questions_processed": len(questions),
                "method": "llm_knowledge_only",
                "reason": reason
            }
        
        answers = []
        rag_fallbacks = 0
        no_answers = 0
        
        for i, question in enumerate(questions, 1):
            logger.info(f"\n📝 Question {i}/{len(questions)}: {question}")
            
            answer = None
            method_used = "none"
            
            if settings.ENABLE_FALLBACK_RAG:
                logger.info("🔄 Attempting RAG processing...")
                rag_answer, rag_success = await self._process_question_with_rag(
                    question, document_url
                )
                
                if rag_success:
                    answer = rag_answer
                    method_used = "rag_llm"
                    rag_fallbacks += 1
                    logger.info("✅ Answered using LLM RAG")
                else:
                    logger.warning("⚠️ RAG failed or returned empty answer")
            
            if not answer:
                answer = f"I apologize, but I cannot find a specific answer to '{question}' in the available documents."
                method_used = "no_answer"
                no_answers += 1
                logger.info("❌ Using fallback answer")
            
            answers.append(answer)
            logger.info(f"Method used: {method_used}")
        
        processing_time = time.time() - start_time
        
        self.stats['total_questions'] += len(questions)
        self.stats['rag_fallbacks'] += rag_fallbacks
        self.stats['no_answers'] += no_answers
        
        result = {
            "answers": answers,
            "document_url": document_url,
            "document_name": document_name,
            "document_key": document_key,
            "processing_time": processing_time,
            "questions_processed": len(questions),
            "rag_fallbacks": rag_fallbacks,
            "no_answers": no_answers,
            "fallback_enabled": settings.ENABLE_FALLBACK_RAG,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        logger.info("\n✅ RAG ANALYSIS COMPLETE")
        logger.info(f"Total time: {processing_time:.2f}s")
        logger.info(f"LLM RAG fallbacks: {rag_fallbacks}")
        logger.info(f"No answers: {no_answers}")
        
        return result

    async def stream_analyze(
        self, document_url: str, questions: List[str]
    ) -> Dict[str, Any]:
        """
        Streaming analysis with RAG approach
        """
        logger.info("\n🌊 RAG STREAMING ANALYSIS")
        
        initial_answers = [f"Processing with RAG for: {question}" for question in questions]
        
        return {
            "initial_answers": initial_answers,
            "status": "processing" if settings.ENABLE_FALLBACK_RAG else "completed",
            "eta": 30 if settings.ENABLE_FALLBACK_RAG else 0
        }

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        format_stats = await self.multi_format_processor.get_processing_stats()
        
        hybrid_stats = {
            'hybrid_mode': True,
            'fallback_rag_enabled': settings.ENABLE_FALLBACK_RAG,
            'fallback_similarity_threshold': settings.FALLBACK_SIMILARITY_THRESHOLD,
            'rag_initialized': self.rag_initialized,
            'performance_stats': self.stats,
            'multi_format_support': format_stats
        }
        
        return hybrid_stats

    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics"""
        
        hybrid_features = [
            "RAG for document questions",
            "Multi-format document support",
            "Performance tracking",
            "Speed-optimized processing"
        ]
        
        base_stats = {
            'mode': 'rag_only',
            'fallback_enabled': settings.ENABLE_FALLBACK_RAG,
            'similarity_threshold': settings.FALLBACK_SIMILARITY_THRESHOLD,
            'features': hybrid_features,
            'performance_stats': self.stats
        }
        
        return base_stats

    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.rag_engine:
                await self.rag_engine.cleanup()
            logger.info("✅ Hybrid matcher cleanup completed")
        except Exception as e:
            logger.error(f"⚠️ Cleanup error: {str(e)}")