"""
Hybrid Response Handler for BajajFinsev RAG System
Non-invasive wrapper around DocumentSpecificMatcher that adds RAG capabilities
Preserves 100% of current functionality and response format
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from src.core.document_specific_matcher import DocumentSpecificMatcher

logger = logging.getLogger(__name__)


class HybridResponseHandler:
    """
    Non-invasive wrapper around DocumentSpecificMatcher that adds RAG capabilities
    - Preserves exact input/output format of current system
    - Only activates RAG when DocumentSpecificMatcher returns "No answer found"
    - Maintains all existing functionality unchanged
    """
    
    def __init__(self, document_matcher: DocumentSpecificMatcher):
        """
        Initialize hybrid handler with existing DocumentSpecificMatcher
        
        Args:
            document_matcher: The current DocumentSpecificMatcher instance (unchanged)
        """
        self.document_matcher = document_matcher
        self._rag_pipeline = None  # Lazy load only when needed
        self._rag_initialization_attempted = False
        self._stats = {
            'total_requests': 0,
            'json_matches': 0,
            'default_matches': 0,
            'rag_enhanced': 0,
            'rag_failures': 0,
            'avg_response_time': 0.0
        }
        
        logger.info("HybridResponseHandler initialized with DocumentSpecificMatcher wrapper")
    
    async def analyze_document(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Main method that preserves DocumentSpecificMatcher behavior exactly
        Only enhances responses that contain "No answer found"
        
        Args:
            document_url: URL to the document (same as current system)
            questions: List of questions (same as current system)
            
        Returns:
            Dict with same format as current system: {"answers": [...], ...}
        """
        start_time = time.time()
        self._stats['total_requests'] += 1
        
        logger.info(f"🔄 Hybrid analysis started for {len(questions)} questions")
        logger.info(f"Document URL: {document_url}")
        
        try:
            # STEP 1: Use current DocumentSpecificMatcher exactly as before
            logger.info("📋 Using DocumentSpecificMatcher (unchanged)")
            original_result = await self.document_matcher.analyze_document(document_url, questions)
            
            # Track original system stats
            self._stats['json_matches'] += original_result.get('json_matches', 0)
            self._stats['default_matches'] += original_result.get('default_matches', 0)
            
            # STEP 2: Analyze responses for failures
            failed_questions = self._detect_failed_answers(original_result["answers"], questions)
            
            if not failed_questions:
                # All questions answered by current system - return unchanged
                logger.info("✅ All questions answered by DocumentSpecificMatcher")
                processing_time = time.time() - start_time
                self._update_avg_response_time(processing_time)
                return original_result
            
            logger.info(f"🔍 {len(failed_questions)} questions need RAG enhancement")
            
            # STEP 3: Only activate RAG for failed questions
            enhanced_answers = await self._enhance_failed_answers(
                document_url, 
                questions, 
                original_result["answers"], 
                failed_questions
            )
            
            # STEP 4: Update result with enhanced answers (preserve format)
            enhanced_result = original_result.copy()
            enhanced_result["answers"] = enhanced_answers
            enhanced_result["rag_enhanced"] = True
            enhanced_result["rag_questions_count"] = len(failed_questions)
            enhanced_result["hybrid_processing"] = True
            
            self._stats['rag_enhanced'] += len(failed_questions)
            
            processing_time = time.time() - start_time
            self._update_avg_response_time(processing_time)
            
            logger.info(f"✅ Hybrid analysis completed in {processing_time:.2f}s")
            logger.info(f"📊 Enhanced {len(failed_questions)} answers with RAG")
            
            return enhanced_result
            
        except Exception as e:
            # If RAG fails, return original DocumentSpecificMatcher result
            logger.error(f"❌ RAG enhancement failed: {str(e)}")
            logger.info("🔄 Falling back to original DocumentSpecificMatcher result")
            
            self._stats['rag_failures'] += 1
            processing_time = time.time() - start_time
            self._update_avg_response_time(processing_time)
            
            # Return original result even if RAG fails
            return original_result
    
    async def stream_analyze(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Streaming analysis - preserves current behavior exactly
        Uses DocumentSpecificMatcher stream_analyze unchanged
        """
        logger.info("🌊 Hybrid streaming analysis")
        
        try:
            # Use original streaming functionality unchanged
            stream_result = await self.document_matcher.stream_analyze(document_url, questions)
            
            # For now, streaming just uses original system
            # Future enhancement could add RAG streaming
            logger.info("📡 Streaming using DocumentSpecificMatcher (unchanged)")
            
            return stream_result
            
        except Exception as e:
            logger.error(f"❌ Streaming analysis failed: {str(e)}")
            raise
    
    def _detect_failed_answers(self, answers: List[str], questions: List[str]) -> List[Tuple[int, str]]:
        """
        Detect which answers indicate "No answer found" and need RAG enhancement
        
        Returns:
            List of (index, question) tuples for failed answers
        """
        failed_questions = []
        
        # Patterns that indicate failed answers from DocumentSpecificMatcher
        failure_patterns = [
            "No answer found for the question",
            "Document not found in knowledge base", 
            "not found in JSON database",
            "The question may not be covered",
            "Document not found in knowledge base",
            "not covered in this specific document"
        ]
        
        for i, answer in enumerate(answers):
            # Check if answer indicates no match was found
            is_failed = any(pattern in answer for pattern in failure_patterns)
            
            if is_failed:
                failed_questions.append((i, questions[i]))
                logger.debug(f"❌ Question {i+1} failed: {questions[i][:50]}...")
        
        return failed_questions
    
    async def _enhance_failed_answers(
        self, 
        document_url: str, 
        original_questions: List[str],
        original_answers: List[str], 
        failed_questions: List[Tuple[int, str]]
    ) -> List[str]:
        """
        Enhance only the failed answers using RAG pipeline
        Preserves successful answers from DocumentSpecificMatcher
        """
        if not failed_questions:
            return original_answers
        
        logger.info(f"🚀 Enhancing {len(failed_questions)} failed answers with RAG")
        
        try:
            # Initialize RAG pipeline if needed (lazy loading)
            if not self._rag_pipeline and not self._rag_initialization_attempted:
                self._rag_pipeline = await self._initialize_rag_pipeline()
                self._rag_initialization_attempted = True
            
            if not self._rag_pipeline:
                logger.warning("⚠️ RAG pipeline not available, keeping original answers")
                return original_answers
            
            # Extract just the failed questions for RAG processing
            questions_to_enhance = [question for _, question in failed_questions]
            
            # Process with RAG pipeline
            rag_answers = await self._rag_pipeline.process_questions(
                document_url, 
                questions_to_enhance
            )
            
            # Replace only the failed answers with RAG-enhanced ones
            enhanced_answers = original_answers.copy()
            for i, (original_index, _) in enumerate(failed_questions):
                if i < len(rag_answers):
                    enhanced_answers[original_index] = rag_answers[i]
                    logger.debug(f"✅ Enhanced answer {original_index+1} with RAG")
            
            return enhanced_answers
            
        except Exception as e:
            logger.error(f"❌ RAG enhancement failed: {str(e)}")
            # Return original answers if RAG fails
            return original_answers
    
    async def _initialize_rag_pipeline(self):
        """
        Lazy initialization of RAG pipeline
        Only loads when first needed to minimize startup time
        """
        try:
            logger.info("🔄 Initializing RAG pipeline (lazy loading)...")
            
            # Import here to avoid circular imports and startup delays
            from src.core.basic_rag_pipeline import BasicRAGPipeline
            
            # Initialize with basic configuration
            rag_pipeline = BasicRAGPipeline()
            await rag_pipeline.initialize()
            
            logger.info("✅ RAG pipeline initialized successfully")
            return rag_pipeline
            
        except ImportError as e:
            logger.warning(f"⚠️ RAG pipeline not available (import failed): {str(e)}")
            logger.info("📋 System will operate in DocumentSpecificMatcher-only mode")
            return None
            
        except Exception as e:
            logger.error(f"❌ RAG pipeline initialization failed: {str(e)}")
            return None
    
    def _update_avg_response_time(self, processing_time: float):
        """Update running average of response times"""
        total_requests = self._stats['total_requests']
        current_avg = self._stats['avg_response_time']
        
        # Running average calculation
        self._stats['avg_response_time'] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics that include both DocumentSpecificMatcher and RAG stats
        Preserves all original stats and adds hybrid system metrics
        """
        # Get original DocumentSpecificMatcher stats
        original_stats = self.document_matcher.get_stats()
        
        # Add hybrid system stats
        hybrid_stats = {
            **original_stats,  # Preserve all original stats
            "hybrid_mode": True,
            "hybrid_stats": self._stats.copy(),
            "rag_pipeline_available": self._rag_pipeline is not None,
            "rag_initialization_attempted": self._rag_initialization_attempted,
            "features": original_stats.get("features", []) + [
                "Hybrid RAG enhancement",
                "Non-invasive response wrapper", 
                "Lazy RAG pipeline loading",
                "Failed answer detection",
                "Format preservation"
            ]
        }
        
        return hybrid_stats
    
    def load_questions(self):
        """Delegate to DocumentSpecificMatcher (preserve existing functionality)"""
        return self.document_matcher.load_questions()
    
    def extract_document_name_from_url(self, document_url: str) -> tuple[str, str]:
        """Delegate to DocumentSpecificMatcher (preserve existing functionality)"""
        return self.document_matcher.extract_document_name_from_url(document_url)
    
    def find_best_match_in_document(self, question: str, document_key: str):
        """Delegate to DocumentSpecificMatcher (preserve existing functionality)"""
        return self.document_matcher.find_best_match_in_document(question, document_key)
    
    def find_best_match_in_default(self, question: str):
        """Delegate to DocumentSpecificMatcher (preserve existing functionality)"""
        return self.document_matcher.find_best_match_in_default(question)
    
    def get_no_answer_response(self, question: str, document_name: str, document_key: str) -> str:
        """Delegate to DocumentSpecificMatcher (preserve existing functionality)"""
        return self.document_matcher.get_no_answer_response(question, document_name, document_key)
    
    # Properties to maintain compatibility
    @property
    def qa_data(self):
        """Access to original qa_data"""
        return self.document_matcher.qa_data
    
    @property 
    def json_file_path(self):
        """Access to original json_file_path"""
        return self.document_matcher.json_file_path
    
    @property
    def stats(self):
        """Access to original stats (for backward compatibility)"""
        return self.document_matcher.stats


class ResponseFailureDetector:
    """
    Utility class for detecting and analyzing response failures
    Helps identify when DocumentSpecificMatcher couldn't find answers
    """
    
    # Standard failure patterns from DocumentSpecificMatcher
    FAILURE_PATTERNS = [
        "No answer found for the question",
        "Document not found in knowledge base",
        "not found in JSON database", 
        "The question may not be covered",
        "not covered in this specific document",
        "Document not found in knowledge base"
    ]
    
    @classmethod
    def is_failed_response(cls, answer: str) -> bool:
        """
        Check if an answer indicates failure to find information
        
        Args:
            answer: The answer string to check
            
        Returns:
            True if answer indicates failure, False otherwise
        """
        return any(pattern in answer for pattern in cls.FAILURE_PATTERNS)
    
    @classmethod
    def analyze_responses(cls, answers: List[str], questions: List[str]) -> Dict[str, Any]:
        """
        Analyze a set of responses to identify failures and success patterns
        
        Returns:
            Analysis dict with failure statistics and recommendations
        """
        total_questions = len(questions)
        failed_indices = []
        successful_indices = []
        
        for i, answer in enumerate(answers):
            if cls.is_failed_response(answer):
                failed_indices.append(i)
            else:
                successful_indices.append(i)
        
        analysis = {
            "total_questions": total_questions,
            "successful_answers": len(successful_indices),
            "failed_answers": len(failed_indices),
            "success_rate": len(successful_indices) / total_questions * 100 if total_questions > 0 else 0,
            "failure_rate": len(failed_indices) / total_questions * 100 if total_questions > 0 else 0,
            "failed_indices": failed_indices,
            "successful_indices": successful_indices,
            "needs_rag_enhancement": len(failed_indices) > 0,
            "rag_enhancement_potential": len(failed_indices)
        }
        
        return analysis
    
    @classmethod
    def get_enhancement_recommendation(cls, analysis: Dict[str, Any]) -> str:
        """
        Get recommendation for response enhancement based on analysis
        """
        failure_rate = analysis["failure_rate"]
        
        if failure_rate == 0:
            return "All questions answered successfully. No RAG enhancement needed."
        elif failure_rate < 25:
            return f"Low failure rate ({failure_rate:.1f}%). Minimal RAG enhancement recommended."
        elif failure_rate < 50:
            return f"Moderate failure rate ({failure_rate:.1f}%). RAG enhancement recommended."
        elif failure_rate < 75:
            return f"High failure rate ({failure_rate:.1f}%). Strong RAG enhancement recommended."
        else:
            return f"Very high failure rate ({failure_rate:.1f}%). Document may need better coverage or RAG is essential."