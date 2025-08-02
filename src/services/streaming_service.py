import logging
import asyncio
from typing import AsyncGenerator, Dict, List, Any
from dataclasses import dataclass

from src.services.rag_workflow import RAGWorkflowService
from src.core.config import settings

logger = logging.getLogger(__name__)

@dataclass
class StreamingResponse:
    """Represents a streaming response chunk"""
    phase: str  # 'quick_scan', 'detailed_analysis', 'complete'
    question_index: int
    question: str
    answer: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class StreamingService:
    """
    Service for streaming responses to improve user experience with large documents.
    Provides quick initial answers while detailed processing continues in background.
    """
    
    def __init__(self):
        self.rag_service = RAGWorkflowService()
        self.logger = logger
    
    async def stream_document_analysis(
        self, 
        document_text: str, 
        questions: List[str]
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Stream document analysis responses in phases:
        1. Quick scan (30-60 seconds) - Initial answers using basic chunking
        2. Detailed analysis (background) - Full hierarchical processing
        """
        self.logger.info(f"Starting streaming analysis for {len(questions)} questions")
        
        # Phase 1: Quick Scan
        self.logger.info("Phase 1: Quick scan starting...")
        async for response in self._quick_scan_phase(document_text, questions):
            yield response
        
        # Phase 2: Detailed Analysis
        if settings.ENABLE_HIERARCHICAL_PROCESSING:
            self.logger.info("Phase 2: Detailed analysis starting...")
            async for response in self._detailed_analysis_phase(document_text, questions):
                yield response
    
    async def _quick_scan_phase(
        self, 
        document_text: str, 
        questions: List[str]
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Quick scan phase: Use basic chunking with smaller chunk set for fast initial answers
        """
        phase_start_time = asyncio.get_event_loop().time()
        
        # Use smaller chunks and fewer retrieval for speed
        quick_chunks = await self._create_quick_chunks(document_text, max_chunks=50)
        
        for i, question in enumerate(questions):
            question_start_time = asyncio.get_event_loop().time()
            
            try:
                # Get quick answer using basic retrieval
                clarified_query = await self.rag_service.clarify_query(question)
                
                # Use basic embedding search with limited chunks
                similar_chunks = await self.rag_service.embedding_service.embed_and_search(
                    query=clarified_query,
                    document_url=document_url,
                    top_k=min(5, len(quick_chunks))  # Limit to 5 chunks for speed
                )
                
                # Generate quick answer
                answer = await self.rag_service.generate_answer_from_chunks(
                    question, clarified_query, similar_chunks
                )
                
                processing_time = asyncio.get_event_loop().time() - question_start_time
                
                # Create streaming response
                response = StreamingResponse(
                    phase="quick_scan",
                    question_index=i,
                    question=question,
                    answer=answer,
                    confidence=0.7,  # Lower confidence for quick answers
                    processing_time=processing_time,
                    metadata={
                        'chunks_used': len(similar_chunks),
                        'total_chunks_available': len(quick_chunks),
                        'method': 'basic_chunking'
                    }
                )
                
                yield response
                
            except Exception as e:
                self.logger.error(f"Error in quick scan for question {i}: {e}")
                
                error_response = StreamingResponse(
                    phase="quick_scan",
                    question_index=i,
                    question=question,
                    answer=f"Quick scan unavailable: {str(e)}",
                    confidence=0.0,
                    processing_time=0.0,
                    metadata={'error': str(e)}
                )
                
                yield error_response
        
        phase_time = asyncio.get_event_loop().time() - phase_start_time
        self.logger.info(f"Quick scan phase completed in {phase_time:.2f}s")
    
    async def _detailed_analysis_phase(
        self, 
        document_text: str, 
        questions: List[str]
    ) -> AsyncGenerator[StreamingResponse, None]:
        """
        Detailed analysis phase: Use full hierarchical processing for comprehensive answers
        """
        phase_start_time = asyncio.get_event_loop().time()
        
        try:
            # Run full hierarchical workflow
            detailed_answers, metrics = await self.rag_service.run_hierarchical_workflow(
                questions, document_text
            )
            
            # Stream detailed responses
            for i, (question, answer) in enumerate(zip(questions, detailed_answers)):
                question_metrics = metrics['question_metrics'][i] if i < len(metrics['question_metrics']) else {}
                
                response = StreamingResponse(
                    phase="detailed_analysis",
                    question_index=i,
                    question=question,
                    answer=answer,
                    confidence=0.95,  # Higher confidence for detailed analysis
                    processing_time=question_metrics.get('total_processing_time', 0),
                    metadata={
                        'hierarchical_used': question_metrics.get('hierarchical_used', False),
                        'chunks_processed': question_metrics.get('processed_chunks', 0),
                        'reduction_percentage': question_metrics.get('reduction_percentage', 0),
                        'sections_analyzed': question_metrics.get('relevant_sections', 0),
                        'method': 'hierarchical_processing'
                    }
                )
                
                yield response
            
        except Exception as e:
            self.logger.error(f"Error in detailed analysis phase: {e}")
            
            # Send error responses for all questions
            for i, question in enumerate(questions):
                error_response = StreamingResponse(
                    phase="detailed_analysis",
                    question_index=i,
                    question=question,
                    answer=f"Detailed analysis failed: {str(e)}",
                    confidence=0.0,
                    processing_time=0.0,
                    metadata={'error': str(e), 'fallback_to_quick_scan': True}
                )
                
                yield error_response
        
        phase_time = asyncio.get_event_loop().time() - phase_start_time
        self.logger.info(f"Detailed analysis phase completed in {phase_time:.2f}s")
        
        # Final completion signal
        yield StreamingResponse(
            phase="complete",
            question_index=-1,
            question="",
            answer="",
            confidence=1.0,
            processing_time=phase_time,
            metadata={
                'total_questions': len(questions),
                'document_size': len(document_text),
                'total_processing_time': phase_time
            }
        )
    
    async def _create_quick_chunks(self, document_text: str, max_chunks: int = 50) -> List[str]:
        """
        Create a limited set of chunks for quick scanning.
        Takes representative samples from different parts of the document.
        """
        # Use larger chunk size for quick scanning
        quick_chunk_size = min(len(document_text) // max_chunks, 5000)
        
        if quick_chunk_size < 1000:
            # Document is small, use standard chunking
            return await self.rag_service._standard_chunking(document_text)
        
        chunks = []
        
        # Take samples from beginning, middle, and end
        sections = [
            (0, len(document_text) // 3),  # Beginning
            (len(document_text) // 3, 2 * len(document_text) // 3),  # Middle
            (2 * len(document_text) // 3, len(document_text))  # End
        ]
        
        chunks_per_section = max_chunks // 3
        
        for start_section, end_section in sections:
            section_text = document_text[start_section:end_section]
            section_chunks = await self._chunk_section_sampling(
                section_text, chunks_per_section, quick_chunk_size
            )
            chunks.extend(section_chunks)
        
        self.logger.info(f"Created {len(chunks)} quick chunks from document")
        return chunks[:max_chunks]  # Ensure we don't exceed max_chunks
    
    async def _chunk_section_sampling(
        self, 
        section_text: str, 
        target_chunks: int, 
        chunk_size: int
    ) -> List[str]:
        """Sample chunks evenly from a section"""
        if len(section_text) <= chunk_size:
            return [section_text]
        
        chunks = []
        step_size = max(len(section_text) // target_chunks, chunk_size)
        
        for i in range(0, len(section_text), step_size):
            chunk = section_text[i:i + chunk_size]
            
            # Try to end at sentence boundary
            if i + chunk_size < len(section_text):
                last_period = chunk.rfind('.')
                if last_period > len(chunk) * 0.7:
                    chunk = chunk[:last_period + 1]
            
            chunks.append(chunk)
            
            if len(chunks) >= target_chunks:
                break
        
        return chunks
    
    def format_streaming_response(self, response: StreamingResponse) -> Dict[str, Any]:
        """
        Format streaming response for API output
        """
        return {
            "phase": response.phase,
            "question_index": response.question_index,
            "question": response.question,
            "answer": response.answer,
            "confidence": response.confidence,
            "processing_time": round(response.processing_time, 2),
            "metadata": response.metadata
        }

# Global instance
streaming_service = StreamingService()