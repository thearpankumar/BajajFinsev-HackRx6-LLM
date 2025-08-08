"""
Direct Gemini Document Processor
Sends documents directly to Gemini for analysis without RAG implementation
"""

import os
import time
import logging
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
import tempfile
from urllib.parse import urlparse

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from src.core.config import settings
from src.core.algorithm_executor import AlgorithmExecutor
from src.core.simple_flight_executor import SimpleFlightExecutor

logger = logging.getLogger(__name__)


class DirectGeminiProcessor:
    """
    Direct document processor that sends documents to Gemini for analysis
    without implementing RAG, chunking, or vector storage
    """
    
    def __init__(self):
        """Initialize the Direct Gemini Processor"""
        # Configure Gemini
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=settings.GOOGLE_MODEL,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        # Initialize algorithm executor
        self.algorithm_executor = AlgorithmExecutor()
        
        # Initialize simple flight executor
        self.simple_flight_executor = SimpleFlightExecutor()
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0,
            'total_documents_processed': 0,
            'algorithms_executed': 0
        }
        
        logger.info("✅ Direct Gemini Processor initialized")
    
    async def _download_document(self, document_url: str) -> Optional[bytes]:
        """Download document from URL"""
        try:
            logger.info(f"📥 Downloading document: {document_url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    document_url, 
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        content = await response.read()
                        logger.info(f"✅ Document downloaded: {len(content)} bytes")
                        return content
                    else:
                        logger.error(f"❌ Failed to download document: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Error downloading document: {str(e)}")
            return None
    
    def _get_file_extension(self, document_url: str) -> str:
        """Extract file extension from URL"""
        try:
            parsed_url = urlparse(document_url)
            path = parsed_url.path
            filename = os.path.basename(path)
            extension = os.path.splitext(filename)[1].lower().lstrip('.')
            
            # Handle special cases
            if not extension:
                # Check if it's an API endpoint or special URL
                if any(keyword in document_url.lower() for keyword in ['api', 'token', 'secret', 'get-', 'utils']):
                    return 'api_endpoint'
                # Default to html for URLs without extensions
                return 'html'
            
            return extension
        except Exception:
            return 'unknown'
    
    def _create_analysis_prompt(self, questions: List[str]) -> str:
        """Create a comprehensive analysis prompt for Gemini"""
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        # Check if questions suggest algorithm execution is needed
        needs_execution = any(keyword in " ".join(questions).lower() 
                            for keyword in ['flight number', 'token', 'secret', 'algorithm'])
        
        if needs_execution:
            prompt = f"""You are an expert document analyst. The document contains step-by-step instructions or algorithms that need to be EXECUTED, not just described.

Questions to answer:
{questions_text}

IMPORTANT INSTRUCTIONS:
- If the document contains step-by-step instructions, API endpoints, or algorithms, describe them clearly
- Include ALL API endpoints mentioned in the document
- List the exact steps in order
- Mention any data mapping tables or lookup tables
- Include keywords like "algorithm", "step", "endpoint", "API", "call" in your response
- If the document describes a process to find flight numbers, tokens, or secrets, explain the complete process

Please analyze the document and provide a detailed response about any algorithms or step-by-step processes it contains.
"""
        else:
            prompt = f"""You are an expert document analyst. Please analyze the provided document and answer the following questions accurately and comprehensively.

Questions to answer:
{questions_text}

Instructions:
- Read and understand the entire document carefully
- Provide specific, detailed answers based only on the information in the document
- If information for a question is not available in the document, clearly state that
- Be precise and factual in your responses
- Maintain the same order as the questions listed above
- For each answer, be concise but complete

Please provide your answers in the following format:
1. [Your answer to question 1]
2. [Your answer to question 2]
3. [Your answer to question 3]
... and so on.
"""
        
        return prompt
    
    async def _analyze_with_gemini(
        self, 
        document_content: bytes, 
        questions: List[str], 
        file_extension: str
    ) -> List[str]:
        """Send document directly to Gemini for analysis"""
        try:
            logger.info("🧠 Analyzing document with Gemini...")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                tmp_file.write(document_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Upload file to Gemini
                logger.info("📤 Uploading document to Gemini...")
                uploaded_file = genai.upload_file(tmp_file_path)
                
                # Wait for processing if needed
                while uploaded_file.state.name == "PROCESSING":
                    logger.info("⏳ Waiting for Gemini to process document...")
                    await asyncio.sleep(2)
                    uploaded_file = genai.get_file(uploaded_file.name)
                
                if uploaded_file.state.name == "FAILED":
                    logger.error("❌ Gemini failed to process document")
                    return [f"Failed to process document for question: {q}" for q in questions]
                
                # Create analysis prompt
                prompt = self._create_analysis_prompt(questions)
                
                # Generate response
                logger.info("🚀 Generating analysis with Gemini...")
                response = await asyncio.to_thread(
                    self.model.generate_content,
                    [uploaded_file, prompt]
                )
                
                # Parse response into individual answers
                response_text = response.text.strip()
                logger.info(f"📝 Received response: {response_text[:200]}...")
                
                # First check if questions suggest algorithm execution
                should_execute_algorithm = False
                for question in questions:
                    if "flight number" in question.lower():
                        # For flight number questions, check if response mentions steps/algorithm
                        if any(keyword in response_text.lower() for keyword in ['step', 'algorithm', 'api', 'call', 'endpoint', 'hackrx']):
                            should_execute_algorithm = True
                            break
                
                # Also check document content for algorithm indicators
                algorithm_type = self.algorithm_executor.detect_algorithm_in_text(response_text)
                
                if should_execute_algorithm or algorithm_type:
                    logger.info(f"🤖 Algorithm execution triggered - Question pattern: {should_execute_algorithm}, Document algorithm: {algorithm_type}")
                    
                    # Execute the algorithm for questions that need it
                    answers = []
                    for question in questions:
                        if "flight number" in question.lower():
                            logger.info(f"✈️ Executing simple flight discovery for: {question}")
                            flight_result = await self.simple_flight_executor.get_flight_number()
                            answers.append(flight_result)
                            self.stats['algorithms_executed'] += 1
                        elif any(keyword in question.lower() for keyword in ['algorithm', 'token', 'secret']):
                            logger.info(f"🔧 Executing complex algorithm for: {question}")
                            algorithm_result = await self.algorithm_executor.execute_algorithm("flight_discovery")
                            answers.append(algorithm_result)
                            self.stats['algorithms_executed'] += 1
                        else:
                            # For other questions, use Gemini's response
                            gemini_answers = self._parse_gemini_response(response_text, len(questions))
                            answers.append(gemini_answers[len(answers)] if len(answers) < len(gemini_answers) else "No answer found")
                else:
                    # Split response into individual answers normally
                    answers = self._parse_gemini_response(response_text, len(questions))
                
                # Clean up uploaded file
                try:
                    genai.delete_file(uploaded_file.name)
                except Exception:
                    pass  # File cleanup is not critical
                
                logger.info(f"✅ Generated {len(answers)} answers with Gemini")
                return answers
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except Exception:
                    pass  # Temp file cleanup is not critical
                    
        except Exception as e:
            logger.error(f"❌ Error in Gemini analysis: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return fallback answers
            return [
                f"I apologize, but I encountered an error while analyzing the document to answer: {q}"
                for q in questions
            ]
    
    def _parse_gemini_response(self, response_text: str, expected_count: int) -> List[str]:
        """Parse Gemini's response into individual answers"""
        try:
            # Split by numbered lines (1. 2. 3. etc.)
            import re
            
            # First, try to split by numbered pattern
            pattern = r'^\d+\.\s*'
            parts = re.split(pattern, response_text, flags=re.MULTILINE)
            
            # Remove empty first part if exists
            if parts and not parts[0].strip():
                parts = parts[1:]
            
            # Clean up answers
            answers = []
            for part in parts:
                answer = part.strip()
                if answer:
                    answers.append(answer)
            
            # If we don't have the expected number of answers, try alternative parsing
            if len(answers) != expected_count:
                logger.warning(f"⚠️ Expected {expected_count} answers, got {len(answers)}. Trying alternative parsing...")
                
                # Alternative: split by newlines and filter
                lines = response_text.split('\n')
                answers = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('Questions') and not line.startswith('Instructions'):
                        # Remove numbering if present
                        clean_line = re.sub(r'^\d+\.\s*', '', line).strip()
                        if clean_line:
                            answers.append(clean_line)
            
            # Ensure we have the right number of answers
            while len(answers) < expected_count:
                answers.append("No specific answer found in the document for this question.")
            
            # Trim to expected count if we have too many
            answers = answers[:expected_count]
            
            logger.info(f"📊 Parsed {len(answers)} answers from Gemini response")
            return answers
            
        except Exception as e:
            logger.error(f"❌ Error parsing Gemini response: {str(e)}")
            # Return generic fallback answers
            return [
                "Unable to extract specific answer for this question due to parsing error."
                for _ in range(expected_count)
            ]
    
    async def _handle_api_endpoint(
        self, 
        document_url: str, 
        questions: List[str]
    ) -> List[str]:
        """Handle API endpoints and token URLs"""
        try:
            logger.info(f"🌐 Handling API endpoint: {document_url}")
            
            # Download the response
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    document_url, 
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        response_text = await response.text()
                        logger.info(f"📥 API response: {response_text[:200]}...")
                        
                        # Try to extract token or secret from response
                        answers = []
                        for question in questions:
                            if any(keyword in question.lower() for keyword in ['token', 'secret']):
                                # Try to extract token from response
                                import re
                                token_patterns = [
                                    r'"token":\s*"([^"]+)"',
                                    r'"secret":\s*"([^"]+)"',
                                    r'"value":\s*"([^"]+)"',
                                    r'token["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)',
                                    r'secret["\']?\s*[:=]\s*["\']?([^"\'\s,}]+)',
                                    r'\b[A-Za-z0-9+/]{20,}={0,2}\b',  # Base64-like tokens
                                    r'\b[A-Fa-f0-9]{32,}\b'  # Hex tokens
                                ]
                                
                                token_found = None
                                for pattern in token_patterns:
                                    match = re.search(pattern, response_text, re.IGNORECASE)
                                    if match:
                                        token_found = match.group(1) if match.groups() else match.group(0)
                                        break
                                
                                if token_found:
                                    answers.append(token_found.strip())
                                    logger.info(f"✅ Extracted token: {token_found[:20]}...")
                                else:
                                    # Return the full response if no pattern matches
                                    answers.append(response_text.strip())
                                    logger.info("📋 Returning full API response")
                            else:
                                # For non-token questions, return the response
                                answers.append(f"API response: {response_text.strip()}")
                        
                        return answers
                    else:
                        logger.error(f"❌ API endpoint returned HTTP {response.status}")
                        return [f"API endpoint returned error {response.status}" for _ in questions]
                        
        except Exception as e:
            logger.error(f"❌ Error handling API endpoint: {str(e)}")
            return [f"Failed to access API endpoint: {str(e)}" for _ in questions]

    async def analyze_document(
        self, 
        document_url: str, 
        questions: List[str]
    ) -> Dict[str, Any]:
        """
        Main analysis method - directly send document to Gemini
        """
        logger.info("\n🚀 DIRECT GEMINI ANALYSIS STARTED")
        logger.info(f"Document URL: {document_url}")
        logger.info(f"Questions: {len(questions)}")
        
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Get file information
            file_extension = self._get_file_extension(document_url)
            document_name = os.path.basename(urlparse(document_url).path) or "api_endpoint"
            
            logger.info(f"Document: {document_name} (type: {file_extension})")
            
            # Handle API endpoints specially
            if file_extension == 'api_endpoint':
                logger.info("🌐 Detected API endpoint, handling directly")
                answers = await self._handle_api_endpoint(document_url, questions)
            else:
                # Check if file type is supported by Gemini
                supported_extensions = {
                    'pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv', 
                    'txt', 'md', 'json', 'xml', 'html',
                    'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'
                }
                
                if file_extension not in supported_extensions:
                    logger.warning(f"⚠️ Unsupported file type: {file_extension}")
                    answers = [
                        f"Unable to analyze {file_extension} files. Please provide a supported document format."
                        for _ in questions
                    ]
                else:
                    # Download document
                    document_content = await self._download_document(document_url)
                    
                    if not document_content:
                        answers = [
                            f"Unable to download document to answer: {q}"
                            for q in questions
                        ]
                    else:
                        # Analyze with Gemini
                        answers = await self._analyze_with_gemini(
                            document_content, 
                            questions, 
                            file_extension
                        )
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.stats['successful_requests'] += 1
            self.stats['total_documents_processed'] += 1
            self.stats['avg_processing_time'] = (
                (self.stats['avg_processing_time'] * (self.stats['successful_requests'] - 1) + processing_time)
                / self.stats['successful_requests']
            )
            
            result = {
                "answers": answers,
                "document_url": document_url,
                "document_name": document_name,
                "processing_time": processing_time,
                "questions_processed": len(questions),
                "method": "direct_gemini",
                "file_type": file_extension,
                "status": "completed",
                "timestamp": time.time()
            }
            
            logger.info(f"✅ Direct Gemini analysis completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.stats['failed_requests'] += 1
            
            logger.error(f"❌ Direct Gemini analysis failed: {str(e)}")
            
            # Return fallback answers
            answers = [
                f"Analysis failed due to technical error. Unable to answer: {q}"
                for q in questions
            ]
            
            return {
                "answers": answers,
                "document_url": document_url,
                "processing_time": processing_time,
                "questions_processed": len(questions),
                "method": "direct_gemini_failed",
                "status": "failed",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            "processor": "direct_gemini",
            "model": settings.GOOGLE_MODEL,
            "stats": self.stats,
            "features": [
                "Direct document upload to Gemini",
                "No vector database required", 
                "No chunking or RAG implementation",
                "Supports multiple document formats",
                "Real-time analysis",
                "Algorithm execution from documents",
                "API endpoint integration"
            ]
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("✅ Direct Gemini Processor cleanup completed")