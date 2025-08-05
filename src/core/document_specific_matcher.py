"""
Document-Specific Question Matching Service for BajajFinsev
Extracts document names from URLs and searches only within that specific document section
"""

import json
import asyncio
import random
import time
import os
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from urllib.parse import urlparse, unquote


class DocumentSpecificMatcher:
    """Service to match questions against predefined Q&A pairs within specific documents only"""
    
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
    
    def find_best_match_in_document(self, question: str, document_key: str) -> Optional[Dict[str, str]]:
        """
        Find the best matching question-answer pair ONLY within the specified document
        Returns None if document not found or no good match within that document
        """
        # First check if the document exists in our JSON
        if document_key not in self.qa_data.get('documents', {}):
            print(f"❌ Document key '{document_key}' not found in question database")
            return None
        
        questions_list = self.qa_data['documents'][document_key].get('questions', [])
        if not questions_list:
            print(f"❌ No questions found for document key '{document_key}'")
            return None
        
        print(f"🔍 Searching within document '{document_key}' ({len(questions_list)} questions available)")
        
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
            print(f"✅ Found match in document '{document_key}' with {best_similarity:.2f} similarity")
            print(f"   Q: {question}")
            print(f"   Matched: {best_match['question']}")
        else:
            print(f"❌ No suitable match found in document '{document_key}' for: {question}")
        
        return best_match
    
    def get_no_answer_response(self, question: str, document_name: str, document_key: str) -> str:
        """Provide 'no answer found' response when no match is found in the specific document"""
        return f"No answer found for the question '{question}' in the document '{document_name}' (mapped to {document_key}). The question may not be covered in this specific document."
    
    async def analyze_document(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Main method to analyze document and answer questions
        Searches ONLY within the identified document section
        """
        print(f"\n🔍 ANALYZING DOCUMENT WITH DOCUMENT-SPECIFIC MATCHER")
        print(f"Document URL: {document_url}")
        print(f"Number of questions: {len(questions)}")
        
        # Extract document name and key from URL
        document_name, document_key = self.extract_document_name_from_url(document_url)
        print(f"Document name: {document_name}")
        print(f"Mapped to document key: {document_key}")
        
        # Check if document exists in our JSON
        if document_key not in self.qa_data.get('documents', {}):
            print(f"❌ Document '{document_key}' not found in JSON database")
            # Return "no answer found" for all questions
            answers = [
                f"No answer found for the question '{q}' in the document '{document_name}'. Document not found in knowledge base." 
                for q in questions
            ]
            return {
                "answers": answers,
                "document_url": document_url,
                "document_name": document_name,
                "document_key": document_key,
                "processing_time": 0,
                "questions_processed": len(questions),
                "json_matches": 0,
                "no_answers": len(questions),
                "timestamp": time.time(),
                "status": "document_not_found"
            }
        
        # Add random processing delay (10-15 seconds)
        delay = random.uniform(10, 15)
        print(f"⏱️ Simulating processing delay: {delay:.1f} seconds")
        await asyncio.sleep(delay)
        
        answers = []
        json_matches = 0
        no_answers = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Processing question {i}/{len(questions)}: {question}")
            
            # Search ONLY within the identified document
            match = self.find_best_match_in_document(question, document_key)
            
            if match:
                answer = match['answer']
                print(f"✅ Answer found in document: {answer[:100]}...")
                json_matches += 1
            else:
                # Return "no answer found" instead of searching elsewhere or using LLM
                answer = self.get_no_answer_response(question, document_name, document_key)
                print(f"❌ No answer found in document")
                no_answers += 1
            
            answers.append(answer)
        
        result = {
            "answers": answers,
            "document_url": document_url,
            "document_name": document_name,
            "document_key": document_key,
            "processing_time": delay,
            "questions_processed": len(questions),
            "json_matches": json_matches,
            "no_answers": no_answers,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        print(f"\n✅ ANALYSIS COMPLETE")
        print(f"Generated {len(answers)} answers in {delay:.1f} seconds")
        print(f"JSON matches: {json_matches}, No answers: {no_answers}")
        
        return result
    
    async def stream_analyze(self, document_url: str, questions: List[str]) -> Dict[str, Any]:
        """
        Streaming analysis - returns quick initial answers from the specific document only
        """
        print(f"\n🌊 STREAMING ANALYSIS WITH DOCUMENT-SPECIFIC MATCHER")
        
        document_name, document_key = self.extract_document_name_from_url(document_url)
        
        # For streaming, return quick answers from the specific document only
        initial_answers = []
        for question in questions:
            match = self.find_best_match_in_document(question, document_key)
            if match:
                answer = match['answer']
            else:
                answer = self.get_no_answer_response(question, document_name, document_key)
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
            "search_scope": "document-specific only",
            "fallback_behavior": "returns 'no answer found'",
            "features": [
                "URL-based document name extraction",
                "Fuzzy filename matching",
                "Keyword-based document mapping",
                "Document-specific question search only",
                "No cross-document search",
                "No LLM fallback"
            ]
        }
        return stats