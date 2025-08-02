import logging
import re
from typing import List, Dict, Any

# NLTK imports for metadata extraction
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    from nltk.chunk import ne_chunk
    from nltk.data import find
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# SpaCy imports for metadata extraction
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class MetadataExtractionService:
    """
    Service for extracting metadata from document chunks.
    Separated from RAGWorkflowService to avoid circular imports.
    """
    
    def __init__(self):
        self.logger = logger
        self._spacy_nlp = None
    
    def extract_metadata_from_chunk(self, chunk_text: str) -> Dict[str, Any]:
        """
        Extract advanced metadata (entities, concepts, categories) from a document chunk.
        """
        metadata = {
            "entities": [],
            "concepts": [],
            "categories": [],
            "keywords": []
        }
        
        # Extract named entities using NLTK or SpaCy
        entities = self._extract_entities(chunk_text)
        metadata["entities"] = entities
        
        # Extract key concepts and keywords
        concepts = self._extract_concepts(chunk_text)
        metadata["concepts"] = concepts
        
        # Extract document categories
        categories = self._extract_categories(chunk_text)
        metadata["categories"] = categories
        
        # Extract important keywords
        keywords = self._extract_keywords(chunk_text)
        metadata["keywords"] = keywords
        
        return metadata
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities (people, organizations, locations, etc.) from text.
        """
        entities = []
        
        if NLTK_AVAILABLE:
            try:
                # Download required NLTK data if not present
                try:
                    find('tokenizers/punkt')
                    find('taggers/averaged_perceptron_tagger')
                    find('chunkers/maxent_ne_chunker')
                    find('corpora/words')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                    nltk.download('maxent_ne_chunker', quiet=True)
                    nltk.download('words', quiet=True)
                
                # Tokenize and tag parts of speech
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                
                # Extract named entities
                tree = ne_chunk(pos_tags)
                
                for subtree in tree:
                    if hasattr(subtree, 'label'):
                        entity_name = ' '.join([token for token, pos in subtree.leaves()])
                        entity_type = subtree.label()
                        entities.append({
                            "name": entity_name,
                            "type": entity_type
                        })
            except Exception as e:
                self.logger.warning(f"NLTK entity extraction failed: {e}")
        
        if SPACY_AVAILABLE and not entities:
            try:
                # Load SpaCy model if not already loaded
                if not self._spacy_nlp:
                    try:
                        self._spacy_nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        self.logger.warning("SpaCy model not found")
                        return entities
                
                doc = self._spacy_nlp(text)
                for ent in doc.ents:
                    entities.append({
                        "name": ent.text,
                        "type": ent.label_
                    })
            except Exception as e:
                self.logger.warning(f"SpaCy entity extraction failed: {e}")
        
        return entities
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text using keyword extraction.
        """
        concepts = []
        
        # Simple approach: extract noun phrases and important terms
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text.lower())
                pos_tags = pos_tag(tokens)
                
                # Extract noun phrases (NN* combinations)
                current_phrase = []
                for word, pos in pos_tags:
                    if pos.startswith('NN'):
                        current_phrase.append(word)
                    else:
                        if current_phrase:
                            concepts.append(' '.join(current_phrase))
                            current_phrase = []
                
                # Add any remaining phrase
                if current_phrase:
                    concepts.append(' '.join(current_phrase))
            except Exception as e:
                self.logger.warning(f"NLTK concept extraction failed: {e}")
        
        # Fallback: comprehensive keyword extraction for business domains
        if not concepts:
            # Insurance domain terms
            insurance_terms = [
                'policy', 'coverage', 'exclusion', 'benefit', 'premium', 'deductible',
                'claim', 'settlement', 'liability', 'sum insured', 'grace period',
                'waiting period', 'copayment', 'coinsurance', 'sub-limit', 'co-pay',
                'pre-existing', 'maternity', 'hospitalization', 'icu', 'room rent',
                'ayush', 'organ donor', 'health check-up', 'no claim discount', 'ncd',
                'cataract', 'ectopic pregnancy', 'preferred provider network', 'ppn',
                'reimbursement', 'cashless', 'network hospital', 'day care', 'ambulance',
                'domiciliary', 'restoration benefit', 'cumulative bonus', 'portability',
                'free look period', 'moratorium', 'underwriting', 'endorsement'
            ]
            
            # Legal domain terms
            legal_terms = [
                'contract', 'agreement', 'liability', 'indemnity', 'warranty',
                'representation', 'covenant', 'breach', 'damages', 'remedy',
                'arbitration', 'jurisdiction', 'governing law', 'force majeure',
                'confidentiality', 'non-disclosure', 'intellectual property',
                'termination', 'severability', 'assignment', 'waiver', 'notice',
                'dispute resolution', 'mediation', 'litigation', 'precedent',
                'statute of limitations', 'due diligence', 'fiduciary duty'
            ]
            
            # HR domain terms
            hr_terms = [
                'employment', 'employee', 'employer', 'compensation', 'salary',
                'benefits', 'leave', 'vacation', 'sick leave', 'pto', 'fmla',
                'performance', 'review', 'evaluation', 'promotion', 'termination',
                'resignation', 'retirement', 'pension', '401k', 'health insurance',
                'disability', 'workers compensation', 'harassment', 'discrimination',
                'equal opportunity', 'diversity', 'inclusion', 'onboarding',
                'offboarding', 'training', 'development', 'succession planning'
            ]
            
            # Compliance domain terms
            compliance_terms = [
                'compliance', 'regulation', 'regulatory', 'audit', 'risk',
                'governance', 'policy', 'procedure', 'control', 'monitoring',
                'reporting', 'documentation', 'record keeping', 'retention',
                'privacy', 'data protection', 'gdpr', 'hipaa', 'sox', 'fcpa',
                'anti-money laundering', 'aml', 'kyc', 'due diligence',
                'whistleblower', 'ethics', 'code of conduct', 'conflict of interest',
                'internal controls', 'risk assessment', 'remediation'
            ]
            
            # Combine all domain terms
            all_terms = insurance_terms + legal_terms + hr_terms + compliance_terms
            
            text_lower = text.lower()
            for term in all_terms:
                if term in text_lower:
                    concepts.append(term)
        
        return list(set(concepts))  # Remove duplicates
    
    def _extract_categories(self, text: str) -> List[str]:
        """
        Extract document categories based on comprehensive content analysis for real-world domains.
        """
        categories = []
        text_lower = text.lower()
        
        # Insurance-related categories with sub-categories
        insurance_indicators = {
            'health_insurance': ['health insurance', 'medical insurance', 'hospitalization', 'health cover', 'mediclaim'],
            'life_insurance': ['life insurance', 'term insurance', 'whole life', 'endowment', 'ulip'],
            'general_insurance': ['motor insurance', 'home insurance', 'travel insurance', 'property insurance'],
            'insurance': ['insurance', 'policy', 'coverage', 'premium', 'benefit', 'claim', 'underwriting']
        }
        
        for category, terms in insurance_indicators.items():
            if any(term in text_lower for term in terms):
                categories.append(category)
                if category != 'insurance' and 'insurance' not in categories:
                    categories.append('insurance')
        
        # Legal-related categories
        legal_indicators = {
            'contract_law': ['contract', 'agreement', 'terms and conditions', 'breach', 'performance'],
            'corporate_law': ['incorporation', 'shareholders', 'board of directors', 'corporate governance'],
            'employment_law': ['employment contract', 'labor law', 'wrongful termination', 'workplace rights'],
            'intellectual_property': ['patent', 'trademark', 'copyright', 'trade secret', 'ip rights'],
            'legal': ['legal', 'law', 'litigation', 'arbitration', 'jurisdiction', 'liability']
        }
        
        for category, terms in legal_indicators.items():
            if any(term in text_lower for term in terms):
                categories.append(category)
                if category != 'legal' and 'legal' not in categories:
                    categories.append('legal')
        
        # HR-related categories
        hr_indicators = {
            'recruitment': ['recruitment', 'hiring', 'onboarding', 'job description', 'interview'],
            'compensation_benefits': ['salary', 'compensation', 'benefits', 'bonus', 'incentive'],
            'performance_management': ['performance review', 'appraisal', 'kpi', 'goals', 'objectives'],
            'employee_relations': ['employee relations', 'grievance', 'disciplinary', 'termination'],
            'hr': ['hr', 'human resources', 'employee', 'employment', 'workforce', 'talent']
        }
        
        for category, terms in hr_indicators.items():
            if any(term in text_lower for term in terms):
                categories.append(category)
                if category != 'hr' and 'hr' not in categories:
                    categories.append('hr')
        
        # Compliance-related categories
        compliance_indicators = {
            'regulatory_compliance': ['regulatory', 'regulation', 'regulator', 'compliance requirement'],
            'data_privacy': ['data privacy', 'gdpr', 'ccpa', 'personal data', 'data protection'],
            'financial_compliance': ['sox', 'sarbanes-oxley', 'financial reporting', 'internal controls'],
            'healthcare_compliance': ['hipaa', 'phi', 'protected health information', 'medical records'],
            'compliance': ['compliance', 'audit', 'risk management', 'governance', 'ethics']
        }
        
        for category, terms in compliance_indicators.items():
            if any(term in text_lower for term in terms):
                categories.append(category)
                if category != 'compliance' and 'compliance' not in categories:
                    categories.append('compliance')
        
        # Financial categories
        if any(term in text_lower for term in ['financial', 'payment', 'billing', 'account', 'transaction',
                                                'revenue', 'expense', 'budget', 'investment']):
            categories.append('financial')
        
        # Medical/Health categories
        if any(term in text_lower for term in ['medical', 'health', 'hospital', 'treatment', 'doctor',
                                                'patient', 'diagnosis', 'prescription', 'surgery']):
            categories.append('medical')
        
        return list(set(categories))  # Remove duplicates
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text for real-world business domains.
        """
        keywords = []
        
        # Extract capitalized words and acronyms
        capitalized_words = re.findall(r'\b[A-Z][A-Z]+\b', text)
        keywords.extend(capitalized_words)
        
        # Extract numbers with context (e.g., "30 days", "5%", "$1000")
        number_patterns = re.findall(r'\b\d+(?:\.\d+)?%?\b|\$\d+(?:,\d{3})*(?:\.\d+)?', text)
        keywords.extend(number_patterns)
        
        # Insurance domain keywords
        insurance_keywords = [
            'policy', 'coverage', 'exclusion', 'benefit', 'premium', 'deductible',
            'claim', 'settlement', 'sum insured', 'sub-limit', 'co-pay', 'copayment',
            'coinsurance', 'waiting period', 'grace period', 'cooling period',
            'pre-existing', 'pre-existing condition', 'maternity', 'hospitalization',
            'icu', 'room rent', 'ayush', 'organ donor', 'health check-up',
            'no claim discount', 'ncd', 'cataract', 'ectopic pregnancy',
            'preferred provider network', 'ppn', 'reimbursement', 'cashless',
            'network hospital', 'day care', 'ambulance', 'domiciliary',
            'restoration benefit', 'cumulative bonus', 'portability',
            'free look period', 'moratorium', 'underwriting', 'endorsement',
            'rider', 'add-on', 'floater', 'individual', 'family floater'
        ]
        
        # Legal domain keywords
        legal_keywords = [
            'contract', 'agreement', 'liability', 'indemnity', 'warranty',
            'representation', 'covenant', 'breach', 'damages', 'remedy',
            'arbitration', 'jurisdiction', 'governing law', 'force majeure',
            'confidentiality', 'non-disclosure', 'intellectual property',
            'termination', 'severability', 'assignment', 'waiver', 'notice',
            'dispute resolution', 'mediation', 'litigation', 'precedent',
            'statute of limitations', 'due diligence', 'fiduciary duty',
            'negligence', 'tort', 'plaintiff', 'defendant', 'injunction'
        ]
        
        # HR domain keywords
        hr_keywords = [
            'employment', 'employee', 'employer', 'compensation', 'salary',
            'benefits', 'leave', 'vacation', 'sick leave', 'pto', 'fmla',
            'performance', 'review', 'evaluation', 'promotion', 'termination',
            'resignation', 'retirement', 'pension', '401k', 'health insurance',
            'disability', 'workers compensation', 'harassment', 'discrimination',
            'equal opportunity', 'diversity', 'inclusion', 'onboarding',
            'offboarding', 'training', 'development', 'succession planning',
            'probation', 'notice period', 'non-compete', 'confidentiality agreement'
        ]
        
        # Compliance domain keywords
        compliance_keywords = [
            'compliance', 'regulation', 'regulatory', 'audit', 'risk',
            'governance', 'policy', 'procedure', 'control', 'monitoring',
            'reporting', 'documentation', 'record keeping', 'retention',
            'privacy', 'data protection', 'gdpr', 'hipaa', 'sox', 'fcpa',
            'anti-money laundering', 'aml', 'kyc', 'due diligence',
            'whistleblower', 'ethics', 'code of conduct', 'conflict of interest',
            'internal controls', 'risk assessment', 'remediation',
            'sanctions', 'embargo', 'export control', 'data breach'
        ]
        
        # Combine all domain keywords
        all_keywords = (insurance_keywords + legal_keywords +
                       hr_keywords + compliance_keywords)
        
        text_lower = text.lower()
        for keyword in all_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        # Extract dates in various formats
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
            r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b'      # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            dates = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(dates)
        
        return list(set(keywords))  # Remove duplicates

# Global instance
metadata_extraction_service = MetadataExtractionService()