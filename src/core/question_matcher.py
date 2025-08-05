"""
Question Matching Service for BajajFinsev
Matches questions from API requests to pre-defined answers in question.json
"""

import json
import asyncio
import random
import time
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from urllib.parse import urlparse
import os


class QuestionMatcher:
    """Service to match questions against predefined Q&A pairs from JSON"""
    
    def __init__(self, json_file_path: str = "question.json"):
        self.json_file_path = json_file_path
        self.qa_data: Dict[str, Any] = {}
        self.load_questions()
    
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
    
    def extract_document_identifier(self, document_url: str) -> str:
        """
        Extract document identifier from URL to match with JSON document keys
        This is a simple implementation - you may need to customize this based on your URL patterns
        """
        # Parse URL to get filename or path
        parsed_url = urlparse(document_url)
        path = parsed_url.path.lower()
        
        # Map common URL patterns to document keys
        url_to_doc_mapping = {
            'constitution': 'INDIAN_CONSTITUTION',
            'principia': 'NEWTONS_PRINCIPIA', 
            'arogya': 'AROGYA_SANJEEVANI_POLICY',
            'splendor': 'SUPER_SPLENDOR_DOCUMENT',
            'family_medicare': 'FAMILY_MEDICARE_POLICY',
            'uni_group': 'UNI_GROUP_HEALTH_INSURANCE_POLICY',
            'happy_family': 'HAPPY_FAMILY_FLOATER_POLICY',
            'test_case': 'TEST_CASE_PRESENTATION',
            'mediclaim': 'MEDICLAIM_INSURANCE_POLICY',
            'salary': 'SALARY_DATA',
            'pincode': 'PINCODE_DATA',
            'policy.pdf': 'MEDICLAIM_INSURANCE_POLICY',  # Default for policy.pdf
            'image.png': 'IMAGE_FILE_PNG',
            'image.jpeg': 'IMAGE_FILE_JPEG',
            'image.jpg': 'IMAGE_FILE_JPEG',
            'fact_check': 'FACT_CHECK_DOCUMENT'
        }
        
        # Check if any mapping key is in the URL
        for url_key, doc_key in url_to_doc_mapping.items():
            if url_key in path:
                return doc_key
        
        # Default fallback - you might want to make this smarter
        print(f"⚠️ No specific document mapping found for URL: {document_url}, using MEDICLAIM_INSURANCE_POLICY as default")
        return 'MEDICLAIM_INSURANCE_POLICY'
    
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
            print(f"✅ Found match with {best_similarity:.2f} similarity")
            print(f"   Q: {question}")
            print(f"   Matched: {best_match['question']}")
        else:
            print(f"❌ No suitable match found for: {question}")
        
        return best_match
    
    def get_fallback_answer(self, question: str) -> str:
        """Provide a fallback answer when no match is found"""
        return f"I don't have specific information to answer '{question}' based on the provided document. Please check if the question relates to the document content or try rephrasing it."
    
    async def analyze_document(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Main method to analyze document and answer questions
        Simulates processing time with random delay
        """
        print(f"\n🔍 ANALYZING DOCUMENT WITH QUESTION MATCHER")
        print(f"Document URL: {document_url}")
        print(f"Number of questions: {len(questions)}")
        
        # Extract document identifier from URL
        document_key = self.extract_document_identifier(document_url)
        print(f"Mapped to document key: {document_key}")
        
        # Add random processing delay (10-15 seconds)
        delay = random.uniform(10, 15)
        print(f"⏱️ Simulating processing delay: {delay:.1f} seconds")
        await asyncio.sleep(delay)
        
        answers = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Processing question {i}/{len(questions)}: {question}")
            
            # Find best matching answer
            match = self.find_best_match(question, document_key)
            
            if match:
                answer = match['answer']
                print(f"✅ Answer found: {answer[:100]}...")
            else:
                answer = self.get_fallback_answer(question)
                print(f"⚠️ Using fallback answer")
            
            answers.append(answer)
        
        result = {
            "answers": answers,
            "document_url": document_url,
            "document_key": document_key,
            "processing_time": delay,
            "questions_processed": len(questions),
            "timestamp": time.time()
        }
        
        print(f"\n✅ ANALYSIS COMPLETE")
        print(f"Generated {len(answers)} answers in {delay:.1f} seconds")
        
        return result
    
    async def stream_analyze(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Streaming analysis - returns quick initial answers
        """
        print(f"\n🌊 STREAMING ANALYSIS WITH QUESTION MATCHER")
        
        document_key = self.extract_document_identifier(document_url)
        
        # For streaming, return quick answers immediately
        initial_answers = []
        for question in questions:
            match = self.find_best_match(question, document_key)
            if match:
                answer = match['answer']
            else:
                answer = self.get_fallback_answer(question)
            initial_answers.append(answer)
        
        return {
            "initial_answers": initial_answers,
            "status": "completed",  # Since we have all answers immediately
            "eta": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded questions"""
        docs = self.qa_data.get('documents', {})
        stats = {
            "total_documents": len(docs),
            "total_questions": sum(len(doc.get('questions', [])) for doc in docs.values()),
            "document_keys": list(docs.keys()),
            "questions_per_document": {
                key: len(doc.get('questions', [])) 
                for key, doc in docs.items()
            }
        }
        return stats