"""
Intelligent Challenge Handler
Integrates with RAG pipeline to automatically detect and solve HackRx challenges
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChallengeContext:
    """Context information for challenge detection"""
    query: str
    document_content: str
    detected_patterns: List[str]
    confidence_score: float
    challenge_type: Optional[str] = None


class IntelligentChallengeHandler:
    """
    Intelligent handler that detects challenge scenarios and automatically
    uses the appropriate MCP server tools to solve them
    """
    
    def __init__(self, mcp_server_url: str = None):
        # Use environment variable for Docker, fallback to localhost for development
        if mcp_server_url is None:
            import os
            mcp_server_url = os.getenv("CHALLENGE_SOLVER_URL", "http://localhost:8004")
        self.mcp_server_url = mcp_server_url
        self.challenge_patterns = {
            "hackrx_parallel_world": [
                r"sachin.*parallel.*world",
                r"flight.*number.*real.*world", 
                r"gateway.*india.*delhi",
                r"landmark.*city.*swapped",
                r"favorite.*city.*flight",
                r"parallel.*world.*discovery",
                r"strange.*new.*world",
                r"hackrx.*core.*team",
                r"flights.*live.*codes.*scrambled"
            ],
            "secret_token": [
                r"secret.*token",
                r"get.*token",
                r"extract.*token",
                r"find.*token",
                r"token.*length"
            ],
            "flight_booking": [
                r"book.*flight",
                r"flight.*reservation", 
                r"travel.*booking"
            ]
        }
        self.challenge_keywords = {
            "hackrx_parallel_world": [
                "hackrx", "sachin", "parallel world", "flight number", "landmark",
                "gateway of india", "taj mahal", "eiffel tower", "big ben",
                "delhi", "mumbai", "chennai", "favorite city", "real world"
            ],
            "secret_token": [
                "secret", "token", "extract", "find", "length"
            ]
        }
    
    async def analyze_query_context(self, query: str, document_content: str = "") -> ChallengeContext:
        """
        Analyze the query and document content to detect challenge scenarios
        
        Args:
            query: User's question/query
            document_content: Content from processed documents
            
        Returns:
            ChallengeContext with detection results
        """
        detected_patterns = []
        confidence_scores = {}
        
        # Combine query and document content for analysis
        full_text = f"{query} {document_content}".lower()
        
        # Check for challenge patterns
        for challenge_type, patterns in self.challenge_patterns.items():
            score = 0
            type_patterns = []
            
            # Pattern matching
            for pattern in patterns:
                if re.search(pattern, full_text, re.IGNORECASE):
                    type_patterns.append(pattern)
                    score += 1
            
            # Keyword matching
            keywords_found = 0
            for keyword in self.challenge_keywords.get(challenge_type, []):
                if keyword.lower() in full_text:
                    keywords_found += 1
            
            # Calculate confidence score
            pattern_score = min(score / len(patterns), 1.0)
            keyword_score = min(keywords_found / len(self.challenge_keywords.get(challenge_type, [1])), 1.0)
            final_score = (pattern_score * 0.6) + (keyword_score * 0.4)
            
            confidence_scores[challenge_type] = final_score
            detected_patterns.extend(type_patterns)
        
        # Determine best matching challenge type
        best_challenge = max(confidence_scores.items(), key=lambda x: x[1]) if confidence_scores else (None, 0.0)
        challenge_type = best_challenge[0] if best_challenge[1] > 0.3 else None
        
        return ChallengeContext(
            query=query,
            document_content=document_content,
            detected_patterns=detected_patterns,
            confidence_score=best_challenge[1],
            challenge_type=challenge_type
        )
    
    async def call_mcp_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # FastMCP endpoint format
                url = f"{self.mcp_server_url}/call/{tool_name}"
                
                response = await client.post(url, json=kwargs)
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"MCP tool '{tool_name}' called successfully")
                return result
                
        except httpx.TimeoutException:
            logger.error(f"MCP tool '{tool_name}' timed out")
            return {"status": "error", "message": "Tool call timed out"}
            
        except httpx.HTTPStatusError as e:
            logger.error(f"MCP tool '{tool_name}' HTTP error: {e.response.status_code}")
            return {"status": "error", "message": f"HTTP {e.response.status_code} error"}
            
        except Exception as e:
            logger.error(f"MCP tool '{tool_name}' failed: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def solve_hackrx_parallel_world_challenge(self) -> Dict[str, Any]:
        """
        Solve the HackRx Parallel World challenge using MCP server
        
        Returns:
            Challenge solution result
        """
        logger.info("Solving HackRx Parallel World challenge...")
        
        try:
            # Use the complete challenge solver
            result = await self.call_mcp_tool("solve_complete_challenge")
            
            if result.get("status") == "success":
                solution = result.get("solution", {})
                successful_flights = solution.get("successful_flights", [])
                
                # Extract flight information
                if successful_flights:
                    if len(successful_flights) == 1:
                        flight_info = successful_flights[0]
                        flight_number = flight_info["flight_number"]
                        landmark = flight_info["landmark"]
                        answer = f"Sachin can return to the real world using flight number: {flight_number}"
                        explanation = (
                            f"I analyzed the parallel world scenario where landmarks are in wrong cities. "
                            f"The favorite city was {solution.get('favorite_city')}, which has the "
                            f"{landmark} landmark. Based on the challenge rules, "
                            f"this leads to flight number {flight_number}."
                        )
                    else:
                        flight_list = [f"{f['flight_number']} (via {f['landmark']})" for f in successful_flights]
                        answer = f"Sachin has {len(successful_flights)} flight options: {', '.join(flight_list)}"
                        explanation = (
                            f"I analyzed the parallel world scenario. The favorite city "
                            f"{solution.get('favorite_city')} has multiple landmarks: "
                            f"{', '.join([f['landmark'] for f in successful_flights])}. "
                            f"Each provides a different flight option to return to the real world."
                        )
                else:
                    answer = "No flight numbers found for the parallel world challenge."
                    explanation = "Unable to retrieve flight information from the challenge endpoints."
                
                logger.info(f"Challenge solved! Answer: {answer}")
                
                return {
                    "status": "success",
                    "answer": answer,
                    "solution_details": solution,
                    "explanation": explanation
                }
            else:
                return {
                    "status": "error",
                    "message": f"Challenge solving failed: {result.get('message', 'Unknown error')}"
                }
                
        except Exception as e:
            logger.error(f"Challenge solving error: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to solve challenge: {str(e)}"
            }
    
    async def extract_secret_token(self, document_url: str, query: str) -> Dict[str, Any]:
        """
        Extract secret token from document (fallback to existing web processor)
        
        Args:
            document_url: URL of the document
            query: User's query about the token
            
        Returns:
            Token extraction result
        """
        logger.info("Extracting secret token...")
        
        try:
            # Use existing web page processor for token extraction
            from src.services.web_page_processor import web_page_processor
            
            processing_result = await web_page_processor.process_url(
                url=document_url,
                question=query
            )
            
            if processing_result["status"] == "success":
                answer = processing_result.get("answer", "")
                if answer:
                    # Check if the question asks for the length
                    if "length" in query.lower():
                        # Extract the token and calculate its length
                        token_match = re.search(r'([a-fA-F0-9]{64})', answer)
                        if token_match:
                            token = token_match.group(1)
                            token_length = len(token)
                            response_text = f"Your secret token is {token} and its length is {token_length}."
                        else:
                            # Fallback if the token format is not found
                            token = answer.strip()
                            token_length = len(token)
                            response_text = f"I found a token: '{token}', and its length is {token_length}."
                    else:
                        token = answer.strip()
                        response_text = f"Your secret token is {token}."
                    
                    return {
                        "status": "success",
                        "answer": response_text
                    }
            
            return {
                "status": "error",
                "message": "Could not extract secret token from document"
            }
            
        except Exception as e:
            logger.error(f"Token extraction error: {str(e)}")
            return {
                "status": "error",
                "message": f"Token extraction failed: {str(e)}"
            }
    
    async def handle_intelligent_query(self, query: str, document_content: str = "", document_url: str = "") -> Tuple[bool, Dict[str, Any]]:
        """
        Main method to intelligently handle queries and detect challenges
        
        Args:
            query: User's question
            document_content: Content extracted from documents  
            document_url: URL of the document being analyzed
            
        Returns:
            Tuple of (challenge_handled: bool, result: Dict)
        """
        logger.info(f"Analyzing query for intelligent challenge detection: {query[:100]}...")
        
        try:
            # Analyze the context
            context = await self.analyze_query_context(query, document_content)
            
            logger.info(f"Challenge analysis - Type: {context.challenge_type}, Confidence: {context.confidence_score:.2f}")
            
            # If confidence is low, don't handle as challenge
            if context.confidence_score < 0.3:
                logger.info("No challenge detected, falling back to normal RAG processing")
                return False, {"message": "No challenge pattern detected"}
            
            # Handle specific challenge types
            if context.challenge_type == "hackrx_parallel_world":
                logger.info("HackRx Parallel World challenge detected!")
                result = await self.solve_hackrx_parallel_world_challenge()
                return True, result
            
            elif context.challenge_type == "secret_token" and document_url:
                logger.info("Secret token extraction challenge detected!")
                result = await self.extract_secret_token(document_url, query)
                return True, result
            
            else:
                logger.info(f"Challenge type '{context.challenge_type}' detected but no handler available")
                return False, {"message": f"Challenge detected but no handler for type: {context.challenge_type}"}
        
        except Exception as e:
            logger.error(f"Intelligent query handling failed: {str(e)}")
            return False, {"message": f"Challenge handling error: {str(e)}"}
    
    async def get_mcp_server_status(self) -> Dict[str, Any]:
        """
        Check the status of the MCP server
        
        Returns:
            Server status information
        """
        try:
            result = await self.call_mcp_tool("get_challenge_status")
            return {
                "mcp_server_available": True,
                "server_url": self.mcp_server_url,
                "status": result
            }
        except Exception as e:
            return {
                "mcp_server_available": False,
                "server_url": self.mcp_server_url,
                "error": str(e)
            }


# Global instance for use in main application
intelligent_challenge_handler = IntelligentChallengeHandler()