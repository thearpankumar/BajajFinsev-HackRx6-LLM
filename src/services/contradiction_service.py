import logging
from typing import List, Dict, Any, Tuple

# Try to import required libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ContradictionResolutionService:
    """
    Service for detecting and resolving contradictions in retrieved information.
    """
    
    def __init__(self):
        self.logger = logger
        self.model = None
        
        # Initialize sentence transformer model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                self.logger.warning(f"Failed to initialize sentence transformer: {e}")
    
    def detect_contradictions(self, answers: List[str]) -> List[Tuple[int, int, float]]:
        """
        Detect contradictions between answers using semantic similarity.
        
        Args:
            answers: List of answer strings
            
        Returns:
            List of tuples (index1, index2, contradiction_score) where contradiction_score > threshold
        """
        if not self.model or len(answers) < 2:
            return []
        
        try:
            # Generate embeddings for all answers
            embeddings = self.model.encode(answers)
            
            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings)
            
            # Find contradictory pairs (low similarity)
            contradictions = []
            contradiction_threshold = 0.3  # Adjust based on needs
            
            for i in range(len(answers)):
                for j in range(i + 1, len(answers)):
                    similarity = similarities[i][j]
                    if similarity < contradiction_threshold:
                        contradictions.append((i, j, similarity))
            
            return contradictions
            
        except Exception as e:
            self.logger.error(f"Error detecting contradictions: {e}")
            return []
    
    def resolve_contradictions(self, answers: List[str], 
                              contexts: List[str] = None) -> Dict[str, Any]:
        """
        Resolve contradictions by analyzing context and selecting most reliable answer.
        
        Args:
            answers: List of potentially contradictory answers
            contexts: Optional list of contexts for each answer
            
        Returns:
            Dictionary with resolved answer and explanation
        """
        if not answers:
            return {
                "resolved_answer": "No answers provided",
                "confidence": 0.0,
                "explanation": "No answers were provided for resolution"
            }
        
        # If only one answer, return it
        if len(answers) == 1:
            return {
                "resolved_answer": answers[0],
                "confidence": 0.9,
                "explanation": "Single answer provided, no contradictions to resolve"
            }
        
        # Detect contradictions
        contradictions = self.detect_contradictions(answers)
        
        if not contradictions:
            # No contradictions found, return first answer with high confidence
            return {
                "resolved_answer": answers[0],
                "confidence": 0.95,
                "explanation": "Multiple answers provided with no detected contradictions"
            }
        
        # Resolve contradictions by analyzing context reliability
        if contexts and len(contexts) == len(answers):
            return self._resolve_with_context(answers, contexts, contradictions)
        else:
            # Resolve by selecting most detailed answer
            return self._resolve_by_detail(answers, contradictions)
    
    def _resolve_with_context(self, answers: List[str], contexts: List[str],
                              contradictions: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """
        Resolve contradictions using context reliability.
        
        Args:
            answers: List of answers
            contexts: List of contexts
            contradictions: List of detected contradictions
            
        Returns:
            Resolution result
        """
        # Calculate context reliability scores
        context_scores = self._calculate_context_reliability(contexts)
        
        # Select answer with highest context reliability
        best_index = max(range(len(context_scores)), key=lambda i: context_scores[i])
        
        return {
            "resolved_answer": answers[best_index],
            "confidence": 0.8,
            "explanation": f"Resolved contradiction by selecting answer with most reliable context. Contradictions detected between answers: {len(contradictions)}"
        }
    
    def _resolve_by_detail(self, answers: List[str],
                          contradictions: List[Tuple[int, int, float]]) -> Dict[str, Any]:
        """
        Resolve contradictions by selecting the most detailed answer.
        
        Args:
            answers: List of answers
            contradictions: List of detected contradictions
            
        Returns:
            Resolution result
        """
        # Score answers by detail level (word count, specific terms, etc.)
        detail_scores = [self._calculate_detail_score(answer) for answer in answers]
        
        # Select answer with highest detail score
        best_index = max(range(len(detail_scores)), key=lambda i: detail_scores[i])
        
        return {
            "resolved_answer": answers[best_index],
            "confidence": 0.7,
            "explanation": f"Resolved contradiction by selecting most detailed answer. Contradictions detected between answers: {len(contradictions)}"
        }
    
    def _calculate_context_reliability(self, contexts: List[str]) -> List[float]:
        """
        Calculate reliability scores for contexts.
        
        Args:
            contexts: List of context strings
            
        Returns:
            List of reliability scores
        """
        scores = []
        for context in contexts:
            if not context:
                scores.append(0.0)
                continue
            
            # Simple reliability scoring based on length and specific terms
            score = len(context) / 1000.0  # Normalize by length
            
            # Boost score for authoritative terms
            authoritative_terms = ['policy', 'contract', 'agreement', 'regulation', 'law']
            for term in authoritative_terms:
                if term in context.lower():
                    score += 0.2
            
            # Cap score at 1.0
            scores.append(min(score, 1.0))
        
        return scores
    
    def _calculate_detail_score(self, answer: str) -> float:
        """
        Calculate detail score for an answer.
        
        Args:
            answer: Answer string
            
        Returns:
            Detail score
        """
        if not answer:
            return 0.0
        
        # Score based on word count and specific terms
        words = answer.split()
        word_score = len(words) / 50.0  # Normalize by typical answer length
        
        # Boost score for specific terms (numbers, dates, etc.)
        import re
        number_count = len(re.findall(r'\d+', answer))
        number_score = number_count * 0.1
        
        # Boost score for conditional terms
        conditional_terms = ['if', 'when', 'provided', 'subject to', 'except']
        conditional_score = sum(0.1 for term in conditional_terms if term in answer.lower())
        
        # Total score capped at 1.0
        total_score = word_score + number_score + conditional_score
        return min(total_score, 1.0)
    
    def generate_uncertainty_response(self, question: str, 
                                     confidence: float = 0.0) -> str:
        """
        Generate a response expressing uncertainty or lack of information.
        
        Args:
            question: Original question
            confidence: Confidence level in the answer (0.0 to 1.0)
            
        Returns:
            Uncertainty response
        """
        if confidence > 0.7:
            return "I'm confident in this answer based on the provided information."
        elif confidence > 0.4:
            return "This answer is based on the available information, but there may be some uncertainty."
        elif confidence > 0.2:
            return "There is significant uncertainty in this answer. Please verify with additional sources."
        else:
            return f"I cannot provide a definitive answer to '{question}' based on the available information. The document does not contain sufficient details to address this question."

# Global instance
contradiction_service = ContradictionResolutionService()

def create_contradiction_aware_prompt(question: str, answers: List[str],
                                     contexts: List[str] = None) -> str:
    """
    Create a prompt that's aware of contradictions and asks the model to resolve them.
    
    Args:
        question: Original question
        answers: List of potentially contradictory answers
        contexts: Optional list of contexts
        
    Returns:
        Contradiction-aware prompt
    """
    prompt_parts = [f"Question: {question}"]
    
    if contexts:
        prompt_parts.append("Contexts:")
        for i, context in enumerate(contexts):
            prompt_parts.append(f"  Context {i+1}: {context}")
    
    prompt_parts.append("Potential Answers:")
    for i, answer in enumerate(answers):
        prompt_parts.append(f"  Answer {i+1}: {answer}")
    
    prompt_parts.append(
        "Instructions: The above answers may contain contradictions. "
        "Please analyze the contradictions and provide a single, coherent answer. "
        "If contradictions cannot be resolved, explain why and provide the most "
        "reliable answer based on the contexts."
    )
    
    return "\n".join(prompt_parts)