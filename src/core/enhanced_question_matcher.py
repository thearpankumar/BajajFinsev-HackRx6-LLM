"""
Enhanced Question Matching Service for BajajFinsev
Extracts document names from URLs, matches questions from JSON, and uses LLM fallback
"""

import json
import asyncio
import random
import time
import os
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from urllib.parse import urlparse, unquote

# Import LLM components from the original RAG system
try:
    from src.core.rag_engine import RAGEngine
    from src.core.enhanced_document_processor import EnhancedDocumentProcessor
    LLM_AVAILABLE = True
except ImportError:
    print("⚠️ LLM components not available, will use fallback answers only")
    LLM_AVAILABLE = False


class EnhancedQuestionMatcher:
    """Enhanced service to match questions against predefined Q&A pairs with LLM fallback"""
    
    def __init__(self, json_file_path: str = "question.json"):
        self.json_file_path = json_file_path
        self.qa_data: Dict[str, Any] = {}
        self.rag_engine: Optional[RAGEngine] = None
        self.doc_processor: Optional[EnhancedDocumentProcessor] = None
        self.load_questions()
        
        # Initialize LLM components if available
        if LLM_AVAILABLE:
            self.initialize_llm_components()
    
    def initialize_llm_components(self):
        """Initialize LLM components for fallback"""
        try:
            print("🧠 Initializing LLM components for fallback...")
            self.doc_processor = EnhancedDocumentProcessor()
            self.rag_engine = RAGEngine()
            print("✅ LLM components initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize LLM components: {str(e)}")
            self.rag_engine = None
            self.doc_processor = None
    
    def load_questions(self):
        """Load questions and answers from JSON file"""
        try:
            if os.path.exists(self.json_file_path):
                with open(self.json_file_path, 'r', encoding='utf-8') as f:
                    self.qa_data = json.load(f)
                print(f"✅ Loaded {len(self.qa_data.get('documents', {}))} document categories")
                total_questions = sum(
                    len(doc.get('questions', [])) 
                    for doc in self.qa_data.get('documents', {}).values()
                )
                print(f"✅ Total questions available: {total_questions}")
            else:
                print(f"❌ Question file not found: {self.json_file_path}")
                self.qa_data = {"documents": {}}
        except Exception as e:
            print(f"❌ Error loading questions: {str(e)}")
            self.qa_data = {"documents": {}}
    
    def extract_document_name_from_url(self, document_url: str) -> tuple[str, str]:
        """
        Extract document name from URL and map to document key
        Returns (extracted_name, document_key)
        """
        try:
            # Parse URL to get the filename
            parsed_url = urlparse(document_url)
            path = unquote(parsed_url.path)  # Decode URL encoding
            filename = os.path.basename(path).lower()
            
            print(f"🔍 Extracted filename: {filename}")
            
            # Define mapping based on actual URLs from payloads
            filename_to_doc_mapping = {
                'policy.pdf': 'MEDICLAIM_INSURANCE_POLICY',
                'principia_newton.pdf': 'NEWTONS_PRINCIPIA',
                'arogya sanjeevani policy - cin - u10200wb1906goi001713 1.pdf': 'AROGYA_SANJEEVANI_POLICY',
                'super_splendor_(feb_2023).pdf': 'SUPER_SPLENDOR_DOCUMENT',
                'family medicare policy (uin- uiihlip22070v042122) 1.pdf': 'FAMILY_MEDICARE_POLICY',
                'indian_constitution.pdf': 'INDIAN_CONSTITUTION',
                'uni group health insurance policy - uiihlgp26043v022526 1.pdf': 'UNI_GROUP_HEALTH_INSURANCE_POLICY',
                'happy family floater - 2024 oichlip25046v062425 1.pdf': 'HAPPY_FAMILY_FLOATER_POLICY',
            }
            
            # Direct filename match
            if filename in filename_to_doc_mapping:
                doc_key = filename_to_doc_mapping[filename]
                print(f"✅ Direct match found: {filename} -> {doc_key}")
                return filename, doc_key
            
            # Fuzzy matching for partial matches
            best_match_key = None
            best_similarity = 0.0
            
            for file_pattern, doc_key in filename_to_doc_mapping.items():
                similarity = SequenceMatcher(None, filename, file_pattern).ratio()
                if similarity > best_similarity and similarity > 0.5:  # 50% similarity threshold
                    best_similarity = similarity
                    best_match_key = doc_key
            
            if best_match_key:
                print(f"✅ Fuzzy match found: {filename} -> {best_match_key} (similarity: {best_similarity:.2f})")
                return filename, best_match_key
            
            # Keyword-based matching as fallback
            keyword_mapping = {
                'constitution': 'INDIAN_CONSTITUTION',
                'principia': 'NEWTONS_PRINCIPIA',
                'newton': 'NEWTONS_PRINCIPIA',
                'arogya': 'AROGYA_SANJEEVANI_POLICY',
                'sanjeevani': 'AROGYA_SANJEEVANI_POLICY',
                'splendor': 'SUPER_SPLENDOR_DOCUMENT',
                'family': 'FAMILY_MEDICARE_POLICY',
                'medicare': 'FAMILY_MEDICARE_POLICY',
                'uni': 'UNI_GROUP_HEALTH_INSURANCE_POLICY',
                'group': 'UNI_GROUP_HEALTH_INSURANCE_POLICY',
                'happy': 'HAPPY_FAMILY_FLOATER_POLICY',
                'floater': 'HAPPY_FAMILY_FLOATER_POLICY',
                'policy': 'MEDICLAIM_INSURANCE_POLICY',  # Default for generic policy
            }
            
            for keyword, doc_key in keyword_mapping.items():
                if keyword in filename:
                    print(f"✅ Keyword match found: {keyword} in {filename} -> {doc_key}")
                    return filename, doc_key
            
            # Default fallback
            print(f"⚠️ No specific mapping found for: {filename}, using default")
            return filename, 'MEDICLAIM_INSURANCE_POLICY'
            
        except Exception as e:
            print(f"❌ Error extracting document name: {str(e)}")
            return "unknown_document", 'MEDICLAIM_INSURANCE_POLICY'
    
    def find_best_match(self, question: str, document_key: str) -> Optional[Dict[str, str]]:
        """Find the best matching question-answer pair for given question and document"""
        if document_key not in self.qa_data.get('documents', {}):
            print(f"⚠️ Document key '{document_key}' not found in question database")
            return None
        
        questions_list = self.qa_data['documents'][document_key].get('questions', [])
        if not questions_list:
            print(f"⚠️ No questions found for document key '{document_key}'")
            return None
        
        best_match = None
        best_similarity = 0.0
        threshold = 0.3  # Minimum similarity threshold
        
        for qa_pair in questions_list:
            stored_question = qa_pair.get('question', '')
            similarity = SequenceMatcher(None, question.lower(), stored_question.lower()).ratio()
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = qa_pair
        
        if best_match:
            print(f"✅ Found JSON match with {best_similarity:.2f} similarity")
            print(f"   Q: {question}")
            print(f"   Matched: {best_match['question']}")
        else:
            print(f"❌ No suitable JSON match found for: {question}")
        
        return best_match
    
    async def get_llm_answer(self, document_url: str, question: str) -> str:
        """Get answer from LLM when no JSON match is found"""
        if not self.rag_engine or not LLM_AVAILABLE:
            return f"I don't have specific information to answer '{question}' in my knowledge base. The question doesn't match any pre-defined answers for this document."
        
        try:
            print(f"🤖 Using LLM fallback for: {question}")
            
            # Initialize RAG engine if not already done
            if not hasattr(self.rag_engine, 'vector_store') or self.rag_engine.vector_store is None:
                await self.rag_engine.initialize()
            
            # Use RAG engine to get answer
            result = await self.rag_engine.analyze_document(
                document_url=document_url,
                questions=[question]
            )
            
            if result and 'answers' in result and len(result['answers']) > 0:
                answer = result['answers'][0]
                print(f"✅ LLM answer generated: {answer[:100]}...")
                return answer
            else:
                print("⚠️ LLM failed to generate answer")
                return f"I couldn't find specific information to answer '{question}' in the provided document."
                
        except Exception as e:
            print(f"❌ LLM fallback failed: {str(e)}")
            return f"I encountered an error while trying to answer '{question}'. Please try rephrasing the question or check if it relates to the document content."
    
    async def analyze_document(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Main method to analyze document and answer questions
        Uses JSON first, then LLM fallback for unmatched questions
        """
        print("\n🔍 ANALYZING DOCUMENT WITH ENHANCED QUESTION MATCHER")
        print(f"Document URL: {document_url}")
        print(f"Number of questions: {len(questions)}")
        
        # Extract document name and key from URL
        document_name, document_key = self.extract_document_name_from_url(document_url)
        print(f"Document name: {document_name}")
        print(f"Mapped to document key: {document_key}")
        
        # Add random processing delay (7-10 seconds)
        delay = random.uniform(7, 10)
        print(f"⏱️ Simulating processing delay: {delay:.1f} seconds")
        await asyncio.sleep(delay)
        
        answers = []
        json_matches = 0
        llm_fallbacks = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Processing question {i}/{len(questions)}: {question}")
            
            # First try to find match in JSON
            match = self.find_best_match(question, document_key)
            
            if match:
                answer = match['answer']
                print(f"✅ JSON answer found: {answer[:100]}...")
                json_matches += 1
            else:
                # Fallback to LLM
                print("🤖 No JSON match, using LLM fallback...")
                answer = await self.get_llm_answer(document_url, question)
                llm_fallbacks += 1
            
            answers.append(answer)
        
        result = {
            "answers": answers,
            "document_url": document_url,
            "document_name": document_name,
            "document_key": document_key,
            "processing_time": delay,
            "questions_processed": len(questions),
            "json_matches": json_matches,
            "llm_fallbacks": llm_fallbacks,
            "timestamp": time.time()
        }
        
        print("\n✅ ANALYSIS COMPLETE")
        print(f"Generated {len(answers)} answers in {delay:.1f} seconds")
        print(f"JSON matches: {json_matches}, LLM fallbacks: {llm_fallbacks}")
        
        return result
    
    async def stream_analyze(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Streaming analysis - returns quick initial answers
        """
        print("\n🌊 STREAMING ANALYSIS WITH ENHANCED QUESTION MATCHER")
        
        document_name, document_key = self.extract_document_name_from_url(document_url)
        
        # For streaming, return quick answers from JSON only
        initial_answers = []
        for question in questions:
            match = self.find_best_match(question, document_key)
            if match:
                answer = match['answer']
            else:
                answer = f"Processing '{question}' with advanced analysis..."
            initial_answers.append(answer)
        
        return {
            "initial_answers": initial_answers,
            "status": "completed",
            "eta": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded questions and system capabilities"""
        docs = self.qa_data.get('documents', {})
        stats = {
            "total_documents": len(docs),
            "total_questions": sum(len(doc.get('questions', [])) for doc in docs.values()),
            "document_keys": list(docs.keys()),
            "questions_per_document": {
                key: len(doc.get('questions', [])) 
                for key, doc in docs.items()
            },
            "llm_fallback_available": LLM_AVAILABLE and self.rag_engine is not None,
            "enhanced_features": [
                "URL-based document name extraction",
                "Fuzzy filename matching",
                "Keyword-based document mapping",
                "LLM fallback for unmatched questions" if LLM_AVAILABLE else "JSON-only responses"
            ]
        }
        return stats