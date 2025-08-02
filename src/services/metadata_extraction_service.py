import logging
import re
from typing import List, Dict, Any

# SpaCy imports for metadata extraction (optional)
try:
    import spacy
    # Try to load the model to check if it's available
    try:
        spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        SPACY_MODEL_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = True
        SPACY_MODEL_AVAILABLE = False
        logging.getLogger(__name__).warning("SpaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)

class MetadataExtractionService:
    """
    Service for extracting metadata from document chunks.
    Separated from RAGWorkflowService to avoid circular imports.
    """
    
    def __init__(self):
        self.logger = logger
        self._spacy_nlp = None
        
        # Initialize SpaCy model if available
        if SPACY_AVAILABLE and SPACY_MODEL_AVAILABLE:
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
                self.logger.info("SpaCy model loaded successfully for metadata extraction")
            except Exception as e:
                self.logger.warning(f"Failed to load SpaCy model: {e}")
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
        Uses SpaCy if available, otherwise falls back to simple regex patterns.
        """
        entities = []
        
        # Try SpaCy first if available
        if SPACY_AVAILABLE and SPACY_MODEL_AVAILABLE and self._spacy_nlp:
            try:
                doc = self._spacy_nlp(text)
                for ent in doc.ents:
                    entities.append({
                        "name": ent.text.strip(),
                        "type": ent.label_
                    })
                return entities
            except Exception as e:
                self.logger.warning(f"SpaCy entity extraction failed: {e}")
        
        # Fallback to simple regex-based entity extraction
        try:
            # Extract potential organization names (capitalized words)
            org_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Company|Co)\.?)\b'
            orgs = re.findall(org_pattern, text)
            for org in orgs:
                entities.append({"name": org.strip(), "type": "ORG"})
            
            # Extract potential person names (Title + Name pattern)
            person_pattern = r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
            persons = re.findall(person_pattern, text)
            for person in persons:
                entities.append({"name": person.strip(), "type": "PERSON"})
            
            # Extract dates
            date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b'
            dates = re.findall(date_pattern, text, re.IGNORECASE)
            for date in dates:
                entities.append({"name": date.strip(), "type": "DATE"})
            
            # Extract monetary amounts
            money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|INR|rupees?)\b'
            amounts = re.findall(money_pattern, text, re.IGNORECASE)
            for amount in amounts:
                entities.append({"name": amount.strip(), "type": "MONEY"})
                
        except Exception as e:
            self.logger.warning(f"Regex entity extraction failed: {e}")
        
        return entities
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text using simple keyword extraction and regex-based sentence splitting.
        """
        concepts = []
        
        try:
            # Use simple regex for sentence tokenization (fast and effective for business docs)
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            
            # Extract important terms and phrases
            for sentence in sentences:
                if not sentence.strip():
                    continue
                    
                # Extract capitalized phrases (potential concepts)
                cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
                concepts.extend(cap_phrases)
                
                # Extract technical terms (words with numbers or special patterns)
                tech_terms = re.findall(r'\b[a-zA-Z]+\d+[a-zA-Z]*\b|\b[A-Z]{2,}\b', sentence)
                concepts.extend(tech_terms)
            
            # Remove duplicates and filter short concepts
            concepts = list(set([c.strip() for c in concepts if len(c.strip()) > 2]))
            
            # Limit to top 20 concepts
            return concepts[:20]
            
        except Exception as e:
            self.logger.warning(f"Concept extraction failed: {e}")
            return []
        
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