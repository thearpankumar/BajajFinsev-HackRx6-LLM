"""
Domain-Aware Query Processor  
Specialized query processing for insurance, legal, HR, and compliance content
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DomainQueryProcessor:
    """
    Domain-aware query processor for insurance, legal, HR, and compliance queries
    Maps common language questions to domain-specific terminology and concepts
    """
    
    def __init__(self):
        # Legal/Constitutional article mappings
        self.legal_mappings = {
            "official name": "Article 1",
            "name of india": "Article 1", 
            "bharat": "Article 1",
            "union of states": "Article 1",
            
            "right to life": "Article 21",
            "life and liberty": "Article 21",
            "personal liberty": "Article 21",
            "due process": "Article 21",
            
            "equality": "Article 14",
            "equal protection": "Article 14",
            "equality before law": "Article 14",
            
            "discrimination": "Article 15",
            "caste discrimination": "Article 15",
            "religion discrimination": "Article 15",
            "gender discrimination": "Article 15",
            
            "untouchability": "Article 17",
            "abolition of untouchability": "Article 17",
            
            "freedom of speech": "Article 19",
            "expression": "Article 19",
            "assembly": "Article 19",
            "association": "Article 19",
            "movement": "Article 19",
            "profession": "Article 19",
            
            "child labor": "Article 24",
            "child labour": "Article 24",
            "hazardous work": "Article 24",
            "factory work": "Article 24",
            
            "religion": "Article 25",
            "religious freedom": "Article 25",
            "conscience": "Article 25",
            
            "religious instruction": "Article 28",
            "educational institution": "Article 28",
            
            "citizenship": "Article 11",
            "regulate citizenship": "Article 11",
            
            "state boundaries": "Article 3",
            "alter boundaries": "Article 3", 
            "new states": "Article 3"
        }
        
        # Insurance domain mappings
        self.insurance_mappings = {
            "hospitalization": ["inpatient care", "hospital admission", "medical treatment"],
            "domiciliary": ["home treatment", "home care", "outpatient"],
            "ambulance": ["emergency transport", "medical transport"],
            "telemedicine": ["teleconsultation", "remote consultation", "digital health"],
            "maternity": ["pregnancy", "childbirth", "maternal care", "delivery"],
            "waiting period": ["cooling period", "exclusion period"],
            "pre-existing": ["prior condition", "existing illness", "medical history"],
            "cashless": ["direct billing", "third party administrator", "TPA", "network hospital"],
            "room rent": ["accommodation charges", "hospital room", "ICU charges"],
            "policy": ["insurance contract", "coverage", "benefits", "terms"],
            "premium": ["insurance payment", "policy cost"],
            "claim": ["reimbursement", "settlement", "payout"],
            "exclusion": ["not covered", "limitations", "restrictions"],
            "deductible": ["excess", "copayment", "out of pocket"],
            "network": ["preferred providers", "panel hospitals"]
        }
        
        # HR domain mappings  
        self.hr_mappings = {
            "employment": ["job", "work", "position", "role"],
            "termination": ["firing", "dismissal", "layoff", "separation"],
            "probation": ["trial period", "initial period"],
            "salary": ["compensation", "wages", "remuneration", "pay"],
            "benefits": ["perks", "allowances", "medical insurance", "PF"],
            "leave": ["vacation", "sick leave", "casual leave", "earned leave"],
            "performance": ["appraisal", "review", "evaluation", "rating"],
            "harassment": ["misconduct", "inappropriate behavior", "workplace violence"],
            "discrimination": ["bias", "unfair treatment", "prejudice"],
            "grievance": ["complaint", "issue", "dispute", "concern"],
            "policy": ["company rules", "code of conduct", "guidelines"],
            "training": ["development", "skill building", "learning"],
            "promotion": ["career advancement", "growth", "elevation"],
            "resignation": ["quit", "leaving", "notice period"],
            "overtime": ["extra hours", "additional work", "extended shifts"]
        }
        
        # Compliance domain mappings
        self.compliance_mappings = {
            "regulation": ["rule", "law", "statute", "guideline", "mandate"],
            "audit": ["inspection", "review", "examination", "assessment"],
            "violation": ["breach", "non-compliance", "infringement"],
            "penalty": ["fine", "sanction", "punishment", "fee"],
            "documentation": ["records", "paperwork", "filing", "reporting"],
            "certification": ["license", "permit", "approval", "authorization"],
            "risk": ["exposure", "liability", "threat", "danger"],
            "monitoring": ["surveillance", "tracking", "oversight"],
            "reporting": ["disclosure", "notification", "filing"],
            "due diligence": ["verification", "investigation", "checking"],
            "framework": ["structure", "system", "process", "methodology"],
            "standard": ["benchmark", "criteria", "requirement", "specification"],
            "governance": ["management", "oversight", "control", "supervision"],
            "ethics": ["conduct", "integrity", "moral standards"],
            "whistleblower": ["informant", "reporter", "disclosure"]
        }
        
        # Combined concept expansions for all domains
        self.concept_expansions = {
            "constitutional": ["constitution", "fundamental rights", "directive principles"],
            "discrimination": ["equal protection", "equality", "bias", "prejudice"],
            "freedom": ["liberty", "rights", "fundamental rights"],
            "arrest": ["detention", "custody", "habeas corpus", "personal liberty"],
            "speech": ["expression", "protest", "assembly", "demonstration"],
            "religion": ["faith", "worship", "conscience", "belief"],
            "caste": ["community", "scheduled caste", "backward class"],
            "job": ["employment", "profession", "occupation", "livelihood"],
            "land": ["property", "acquisition", "eminent domain"],
            "torture": ["cruel treatment", "inhuman treatment", "custodial violence"],
            "child": ["minor", "juvenile", "below 14 years"],
            "government": ["state", "authority", "public authority"],
            "court": ["judiciary", "judicial review", "high court", "supreme court"]
        }
        
        # Question type patterns for legal content
        self.legal_question_patterns = {
            "article_reference": r"article\s+(\d+)",
            "age_reference": r"(?:below|under|age)\s+(\d+)",
            "constitutional_provision": r"according to.*constitution",
            "right_violation": r"(?:is that|that)\s+(?:legal|constitutional|allowed|permitted)",
            "prohibition": r"(?:prohibited|banned|not allowed|illegal)",
            "government_action": r"(?:government|state).*(?:can|stop|prevent|take)"
        }
        
        logger.info("DomainQueryProcessor initialized for insurance, legal, HR, and compliance queries")
    
    def detect_domain(self, query: str) -> str:
        """Detect which domain the query belongs to"""
        query_lower = query.lower()
        
        # Domain keywords
        insurance_keywords = ["policy", "claim", "premium", "hospital", "medical", "coverage", "insurance", "ambulance", "maternity", "cashless"]
        legal_keywords = ["article", "constitution", "legal", "law", "court", "judge", "rights", "freedom", "arrest"]
        hr_keywords = ["employee", "salary", "leave", "job", "employment", "termination", "performance", "harassment", "resignation"]
        compliance_keywords = ["regulation", "audit", "compliance", "violation", "penalty", "certification", "governance", "ethics"]
        
        # Count domain-specific keywords
        insurance_score = sum(1 for keyword in insurance_keywords if keyword in query_lower)
        legal_score = sum(1 for keyword in legal_keywords if keyword in query_lower)
        hr_score = sum(1 for keyword in hr_keywords if keyword in query_lower)
        compliance_score = sum(1 for keyword in compliance_keywords if keyword in query_lower)
        
        # Determine domain based on highest score
        scores = {
            "insurance": insurance_score,
            "legal": legal_score, 
            "hr": hr_score,
            "compliance": compliance_score
        }
        
        # Return domain with highest score, default to legal if tie
        detected_domain = max(scores, key=scores.get)
        if scores[detected_domain] == 0:
            return "general"  # No specific domain detected
        
        return detected_domain
    
    def enhance_domain_query(self, query: str) -> Dict[str, Any]:
        """
        Enhance a domain query with specific terminology and expansions
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query information
        """
        try:
            # Detect domain first
            detected_domain = self.detect_domain(query)
            
            enhanced_info = {
                "original_query": query,
                "enhanced_query": query,
                "detected_domain": detected_domain,
                "domain_concepts": [],
                "article_references": [],
                "expanded_terms": [],
                "question_type": "general",
                "domain_context": {}
            }
            
            query_lower = query.lower()
            
            # Domain-specific processing
            if detected_domain == "legal":
                # 1. Identify article references
                article_refs = self._identify_articles(query_lower)
                enhanced_info["article_references"] = article_refs
                
                # 2. Map to constitutional articles
                mapped_articles = self._map_to_articles(query_lower)
                enhanced_info["article_references"].extend(mapped_articles)
                
                # 3. Determine legal question type
                question_type = self._determine_legal_question_type(query_lower)
                enhanced_info["question_type"] = question_type
                
            elif detected_domain == "insurance":
                enhanced_info["domain_concepts"] = self._extract_insurance_concepts(query_lower)
                enhanced_info["question_type"] = self._determine_insurance_question_type(query_lower)
                
            elif detected_domain == "hr":
                enhanced_info["domain_concepts"] = self._extract_hr_concepts(query_lower)
                enhanced_info["question_type"] = self._determine_hr_question_type(query_lower)
                
            elif detected_domain == "compliance":
                enhanced_info["domain_concepts"] = self._extract_compliance_concepts(query_lower)
                enhanced_info["question_type"] = self._determine_compliance_question_type(query_lower)
            
            # 4. Expand with domain-specific concepts
            expanded_terms = self._expand_domain_concepts(query_lower, detected_domain)
            enhanced_info["expanded_terms"] = expanded_terms
            
            # 5. Build enhanced query
            enhanced_query = self._build_enhanced_query(query, enhanced_info)
            enhanced_info["enhanced_query"] = enhanced_query
            
            # 6. Add domain context
            enhanced_info["domain_context"] = self._build_domain_context(enhanced_info)
            
            return enhanced_info
            
        except Exception as e:
            logger.error(f"Domain query enhancement failed: {str(e)}")
            return {
                "original_query": query,
                "enhanced_query": query,
                "detected_domain": "general",
                "domain_concepts": [],
                "article_references": [],
                "expanded_terms": [],
                "question_type": "general",
                "domain_context": {}
            }
    
    def _identify_articles(self, query: str) -> List[str]:
        """Identify explicit article references"""
        articles = []
        
        # Find Article X patterns
        article_matches = re.findall(r'article\s+(\d+)', query, re.IGNORECASE)
        for match in article_matches:
            articles.append(f"Article {match}")
        
        return articles
    
    def _map_to_articles(self, query: str) -> List[str]:
        """Map query concepts to constitutional articles"""
        mapped_articles = []
        
        for concept, article in self.legal_mappings.items():
            if concept in query:
                if article not in mapped_articles:
                    mapped_articles.append(article)
        
        return mapped_articles
    
    def _extract_insurance_concepts(self, query: str) -> List[str]:
        """Extract insurance-specific concepts"""
        concepts = []
        for concept, expansions in self.insurance_mappings.items():
            if concept in query:
                concepts.append(concept)
        return concepts
    
    def _extract_hr_concepts(self, query: str) -> List[str]:
        """Extract HR-specific concepts"""
        concepts = []
        for concept, expansions in self.hr_mappings.items():
            if concept in query:
                concepts.append(concept)
        return concepts
    
    def _extract_compliance_concepts(self, query: str) -> List[str]:
        """Extract compliance-specific concepts"""
        concepts = []
        for concept, expansions in self.compliance_mappings.items():
            if concept in query:
                concepts.append(concept)
        return concepts
    
    def _determine_insurance_question_type(self, query: str) -> str:
        """Determine insurance question type"""
        if "covered" in query or "coverage" in query:
            return "coverage_inquiry"
        elif "claim" in query or "reimbursement" in query:
            return "claims_process"
        elif "waiting period" in query or "exclusion" in query:
            return "policy_limitations"
        elif "premium" in query or "cost" in query:
            return "pricing"
        elif "network" in query or "cashless" in query:
            return "network_services"
        else:
            return "general_insurance"
    
    def _determine_hr_question_type(self, query: str) -> str:
        """Determine HR question type"""
        if "termination" in query or "firing" in query:
            return "employment_termination"
        elif "salary" in query or "compensation" in query:
            return "compensation"
        elif "leave" in query or "vacation" in query:
            return "leave_policy"
        elif "harassment" in query or "discrimination" in query:
            return "workplace_conduct"
        elif "performance" in query or "appraisal" in query:
            return "performance_management"
        else:
            return "general_hr"
    
    def _determine_compliance_question_type(self, query: str) -> str:
        """Determine compliance question type"""
        if "audit" in query or "inspection" in query:
            return "audit_compliance"
        elif "violation" in query or "breach" in query:
            return "compliance_violation"
        elif "regulation" in query or "law" in query:
            return "regulatory_requirement"
        elif "penalty" in query or "fine" in query:
            return "enforcement_action"
        elif "reporting" in query or "documentation" in query:
            return "reporting_requirement"
        else:
            return "general_compliance"
    
    def _expand_domain_concepts(self, query: str, domain: str) -> List[str]:
        """Expand query with domain-specific terms"""
        expanded_terms = []
        
        # Generic concept expansions
        for concept, expansions in self.concept_expansions.items():
            if concept in query:
                expanded_terms.extend(expansions)
        
        # Domain-specific expansions
        if domain == "insurance":
            for concept, expansions in self.insurance_mappings.items():
                if concept in query:
                    expanded_terms.extend(expansions)
        elif domain == "hr":
            for concept, expansions in self.hr_mappings.items():
                if concept in query:
                    expanded_terms.extend(expansions)
        elif domain == "compliance":
            for concept, expansions in self.compliance_mappings.items():
                if concept in query:
                    expanded_terms.extend(expansions)
        
        return list(set(expanded_terms))  # Remove duplicates
    
    def _determine_legal_question_type(self, query: str) -> str:
        """Determine the type of legal question"""
        
        if re.search(self.legal_question_patterns["article_reference"], query, re.IGNORECASE):
            return "article_specific"
        elif re.search(self.legal_question_patterns["constitutional_provision"], query, re.IGNORECASE):
            return "constitutional_interpretation"
        elif re.search(self.legal_question_patterns["right_violation"], query, re.IGNORECASE):
            return "rights_violation"
        elif re.search(self.legal_question_patterns["prohibition"], query, re.IGNORECASE):
            return "prohibition_inquiry"
        elif re.search(self.legal_question_patterns["government_action"], query, re.IGNORECASE):
            return "government_power"
        elif "preamble" in query:
            return "preamble"
        elif any(word in query for word in ["right", "freedom", "liberty"]):
            return "fundamental_rights"
        else:
            return "general_legal"
    
    def _build_enhanced_query(self, original_query: str, enhanced_info: Dict[str, Any]) -> str:
        """Build enhanced query with legal terms"""
        
        enhanced_parts = [original_query]
        
        # Add article references
        if enhanced_info["article_references"]:
            enhanced_parts.extend(enhanced_info["article_references"])
        
        # Add key expanded terms (limit to avoid too long queries)
        key_expansions = enhanced_info["expanded_terms"][:5]  # Top 5 most relevant
        enhanced_parts.extend(key_expansions)
        
        # Join and clean up
        enhanced_query = " ".join(enhanced_parts)
        
        # Remove duplicates while preserving order
        words = enhanced_query.split()
        seen = set()
        unique_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower not in seen:
                unique_words.append(word)
                seen.add(word_lower)
        
        return " ".join(unique_words)
    
    def _build_domain_context(self, enhanced_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build domain-specific context for better retrieval"""
        
        domain = enhanced_info["detected_domain"]
        question_type = enhanced_info["question_type"]
        
        context = {
            "domain": domain,
            "focus_areas": [],
            "search_priority": "high",
            "specific_context": {}
        }
        
        if domain == "legal":
            context["focus_areas"] = self._build_legal_focus_areas(question_type, enhanced_info)
        elif domain == "insurance":
            context["focus_areas"] = self._build_insurance_focus_areas(question_type)
        elif domain == "hr":
            context["focus_areas"] = self._build_hr_focus_areas(question_type)
        elif domain == "compliance":
            context["focus_areas"] = self._build_compliance_focus_areas(question_type)
        
        return context
    
    def _build_legal_focus_areas(self, question_type: str, enhanced_info: Dict[str, Any]) -> List[str]:
        """Build legal-specific focus areas"""
        if question_type == "fundamental_rights":
            return ["Part III", "Fundamental Rights", "Articles 12-35"]
        elif question_type == "article_specific":
            return enhanced_info["article_references"]
        elif question_type == "preamble":
            return ["Preamble", "objectives", "ideals"]
        elif question_type == "government_power":
            return ["State powers", "government authority", "limitations"]
        else:
            return ["constitutional provisions", "legal rights"]
    
    def _build_insurance_focus_areas(self, question_type: str) -> List[str]:
        """Build insurance-specific focus areas"""
        if question_type == "coverage_inquiry":
            return ["policy coverage", "benefits", "inclusions"]
        elif question_type == "claims_process":
            return ["claim settlement", "reimbursement", "documentation"]
        elif question_type == "policy_limitations":
            return ["exclusions", "waiting periods", "limitations"]
        elif question_type == "network_services":
            return ["cashless treatment", "network hospitals", "TPA"]
        else:
            return ["insurance policy", "terms and conditions"]
    
    def _build_hr_focus_areas(self, question_type: str) -> List[str]:
        """Build HR-specific focus areas"""
        if question_type == "employment_termination":
            return ["termination policy", "severance", "notice period"]
        elif question_type == "compensation":
            return ["salary structure", "benefits", "allowances"]
        elif question_type == "workplace_conduct":
            return ["code of conduct", "harassment policy", "discipline"]
        elif question_type == "leave_policy":
            return ["leave entitlement", "vacation policy", "sick leave"]
        else:
            return ["employee policies", "HR procedures"]
    
    def _build_compliance_focus_areas(self, question_type: str) -> List[str]:
        """Build compliance-specific focus areas"""
        if question_type == "audit_compliance":
            return ["audit procedures", "compliance review", "inspection"]
        elif question_type == "compliance_violation":
            return ["violations", "non-compliance", "corrective actions"]
        elif question_type == "regulatory_requirement":
            return ["regulations", "compliance requirements", "standards"]
        elif question_type == "reporting_requirement":
            return ["reporting obligations", "documentation", "filing"]
        else:
            return ["compliance framework", "governance"]
    
    def get_domain_processor_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "processor_type": "DomainQueryProcessor", 
            "supported_domains": ["legal", "insurance", "hr", "compliance"],
            "legal_mappings_count": len(self.legal_mappings),
            "insurance_mappings_count": len(self.insurance_mappings),
            "hr_mappings_count": len(self.hr_mappings),
            "compliance_mappings_count": len(self.compliance_mappings),
            "concept_expansions_count": len(self.concept_expansions),
            "supported_question_types": {
                "legal": ["article_specific", "constitutional_interpretation", "rights_violation", "prohibition_inquiry", "government_power", "preamble", "fundamental_rights"],
                "insurance": ["coverage_inquiry", "claims_process", "policy_limitations", "pricing", "network_services"],
                "hr": ["employment_termination", "compensation", "leave_policy", "workplace_conduct", "performance_management"],
                "compliance": ["audit_compliance", "compliance_violation", "regulatory_requirement", "enforcement_action", "reporting_requirement"]
            },
            "features": [
                "Multi-domain support", "Automatic domain detection",
                "Domain-specific concept mapping", "Query expansion",
                "Context building", "Question type classification"
            ]
        }