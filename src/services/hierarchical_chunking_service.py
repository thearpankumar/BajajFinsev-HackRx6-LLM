import logging
import asyncio
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from src.services.llm_clients import GEMINI_FLASH_MODEL
from src.core.config import settings

# NLTK imports for semantic chunking
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    from nltk.data import find
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    sent_tokenize = None

# SpaCy imports for semantic chunking
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section of a large document"""
    content: str
    summary: str
    start_position: int
    end_position: int
    section_type: str  # 'header', 'content', 'table', 'conclusion'
    relevance_score: float = 0.0

@dataclass 
class ProcessingMetrics:
    """Track processing performance metrics"""
    total_chunks: int
    processed_chunks: int
    processing_time: float
    sections_identified: int
    relevant_sections: int

class HierarchicalChunkingService:
    """
    Advanced chunking service optimized for large documents (600K+ tokens).
    Uses hierarchical processing to reduce processing time by 80-90%.
    """
    
    def __init__(self):
        self.logger = logger
        self.document_sections_cache = {}  # Cache processed sections by document hash
        
    def split_into_sections(self, document: str, max_section_size: int = 50000) -> List[DocumentSection]:
        """
        Split large document into logical sections based on structure.
        Optimized for insurance documents and policies.
        """
        self.logger.info(f"Splitting document ({len(document)} chars) into sections")
        
        sections = []
        
        # Patterns for identifying section boundaries in multiple document types
        section_patterns = [
            # Document structure patterns
            r'\n\s*(?:SECTION|Section|PART|Part|CHAPTER|Chapter|ARTICLE|Article)\s+[IVXLC\d]+',  # Roman/numeric sections
            r'\n\s*\d+\.\s+[A-Z][A-Za-z\s]{10,}',  # Numbered sections with titles
            r'\n\s*[A-Z\s]{5,}:\s*\n',  # ALL CAPS headers with colons
            r'\n\s*(?:TABLE|SCHEDULE|APPENDIX|ANNEXURE|EXHIBIT|ATTACHMENT)',  # Document structure keywords
            
            # Scientific/Mathematical document patterns (for texts like Newton's Principia)
            r'\n\s*(?:PROPOSITION|Proposition)\s+[IVXLC\d]+',  # Mathematical propositions
            r'\n\s*(?:THEOREM|Theorem)\s+[IVXLC\d]+',  # Mathematical theorems
            r'\n\s*(?:COROLLARY|Corollary)\s+[IVXLC\d]+',  # Mathematical corollaries
            r'\n\s*(?:LEMMA|Lemma)\s+[IVXLC\d]+',  # Mathematical lemmas
            r'\n\s*(?:SCHOLIUM|Scholium)',  # Mathematical scholiums
            r'\n\s*(?:DEFINITION|Definition)\s+[IVXLC\d]+',  # Mathematical definitions
            r'\n\s*(?:AXIOM|Axiom)\s+[IVXLC\d]+',  # Mathematical axioms
            r'\n\s*(?:PROBLEM|Problem)\s+[IVXLC\d]+',  # Mathematical problems
            r'\n\s*(?:BOOK|Book)\s+[IVXLC\d]+',  # Book divisions
            
            # Insurance domain patterns (more comprehensive)
            r'\n\s*(?:DEFINITIONS?|INTERPRETATION|MEANING|TERMINOLOGY)',
            r'\n\s*(?:COVERAGE|COVERED|BENEFITS?|ELIGIBILITY|ENTITLEMENT)',
            r'\n\s*(?:EXCLUSIONS?|NOT COVERED|LIMITATIONS?|RESTRICTIONS?)',
            r'\n\s*(?:CONDITIONS?|REQUIREMENTS?|OBLIGATIONS?|RESPONSIBILITIES?)',
            r'\n\s*(?:CLAIMS? PROCEDURE|CLAIMS? PROCESS|HOW TO CLAIM)',
            r'\n\s*(?:PREMIUM|DEDUCTIBLE|COPAYMENT|COINSURANCE|SUM INSURED)',
            r'\n\s*(?:WAITING PERIOD|COOLING PERIOD|GRACE PERIOD)',
            r'\n\s*(?:MATERNITY|PREGNANCY|DELIVERY|CHILDBIRTH)',
            r'\n\s*(?:PRE[-]?EXISTING|EXISTING CONDITION)',
            r'\n\s*(?:HOSPITALIZATION|ICU|ROOM RENT|AMBULANCE)',
            r'\n\s*(?:AYUSH|ALTERNATIVE MEDICINE|TRADITIONAL MEDICINE)',
            r'\n\s*(?:ORGAN DONOR|TRANSPLANT|SURGERY)',
            r'\n\s*(?:HEALTH CHECK[-]?UP|MEDICAL EXAMINATION)',
            r'\n\s*(?:NO CLAIM DISCOUNT|NCD|LOYALTY DISCOUNT)',
            r'\n\s*(?:CATARACT|EYE SURGERY|VISION CARE)',
            r'\n\s*(?:ECTOPIC PREGNANCY|PREGNANCY COMPLICATION)',
            r'\n\s*(?:TERMINATION|CANCELLATION|SURRENDER)',
            r'\n\s*(?:RENEWAL|REINSTATEMENT|CONTINUITY)',
            
            # Legal domain patterns  
            r'\n\s*(?:TERMS AND CONDITIONS|AGREEMENT|CONTRACT|LIABILITY|WARRANTY|INDEMNIFICATION)',
            r'\n\s*(?:GOVERNING LAW|JURISDICTION|ARBITRATION|TERMINATION|BREACH|REMEDY)',
            
            # HR domain patterns
            r'\n\s*(?:EMPLOYMENT|COMPENSATION|BENEFITS|LEAVE|VACATION|SICK LEAVE|TERMINATION)',
            r'\n\s*(?:PERFORMANCE|DISCIPLINE|GRIEVANCE|HARASSMENT|EQUAL OPPORTUNITY)',
            
            # Compliance domain patterns
            r'\n\s*(?:COMPLIANCE|REGULATORY|AUDIT|REPORTING|DOCUMENTATION|RECORD KEEPING)',
            r'\n\s*(?:PRIVACY|CONFIDENTIAL|DATA PROTECTION|SECURITY|DISCLOSURE)',
            
            # Common business patterns
            r'\n\s*(?:PROCEDURES|PROCESS|REQUIREMENTS|OBLIGATIONS|RESPONSIBILITIES|DUTIES)',
            r'\n\s*(?:EFFECTIVE DATE|AMENDMENT|MODIFICATION|NOTICE|COMMUNICATION)',
        ]
        
        # Find all potential section breaks
        section_breaks = [0]  # Start of document
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, document, re.IGNORECASE | re.MULTILINE)
            section_breaks.extend([match.start() for match in matches])
        
        # Sort and deduplicate section breaks
        section_breaks = sorted(list(set(section_breaks)))
        section_breaks.append(len(document))  # End of document
        
        # Create sections from breaks
        for i in range(len(section_breaks) - 1):
            start_pos = section_breaks[i]
            end_pos = section_breaks[i + 1]
            
            # Skip very small sections (less than 500 chars)
            if end_pos - start_pos < 500:
                continue
            
            section_content = document[start_pos:end_pos].strip()
            
            # Further split if section is too large
            if len(section_content) > max_section_size:
                sub_sections = self._split_large_section(section_content, start_pos, max_section_size)
                sections.extend(sub_sections)
            else:
                section_type = self._identify_section_type(section_content)
                sections.append(DocumentSection(
                    content=section_content,
                    summary="",  # Will be generated later
                    start_position=start_pos,
                    end_position=end_pos,
                    section_type=section_type
                ))
        
        self.logger.info(f"Created {len(sections)} sections from document")
        return sections
    
    def _split_large_section(self, content: str, start_offset: int, max_size: int) -> List[DocumentSection]:
        """Split a large section into smaller manageable pieces"""
        sections = []
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        current_section = ""
        current_start = start_offset
        
        for para in paragraphs:
            if len(current_section + para) > max_size and current_section:
                # Save current section
                sections.append(DocumentSection(
                    content=current_section.strip(),
                    summary="",
                    start_position=current_start,
                    end_position=current_start + len(current_section),
                    section_type=self._identify_section_type(current_section)
                ))
                
                # Start new section
                current_start += len(current_section)
                current_section = para
            else:
                current_section += "\n\n" + para if current_section else para
        
        # Add final section
        if current_section.strip():
            sections.append(DocumentSection(
                content=current_section.strip(),
                summary="",
                start_position=current_start,
                end_position=current_start + len(current_section),
                section_type=self._identify_section_type(current_section)
            ))
        
        return sections
    
    def _identify_section_type(self, content: str) -> str:
        """Identify the type of document section based on comprehensive real-world patterns"""
        content_lower = content.lower()
        
        # Insurance-specific sections (comprehensive for real-world scenarios)
        if any(word in content_lower for word in ['definition', 'meaning', 'term', 'means', 'shall mean', 'defined as', 'interpretation', 'glossary']):
            return 'definitions'
        elif any(word in content_lower for word in ['scope of coverage', 'what is covered', 'coverage', 'benefit', 'cover', 'insured', 'policy limit', 'sum insured', 'benefit amount']):
            return 'coverage'
        elif any(word in content_lower for word in ['exclusion', 'not covered', 'except', 'limitation', 'excluded', 'not payable', 'what is not covered', 'circumstances not covered']):
            return 'exclusions'
        elif any(word in content_lower for word in ['claim', 'procedure', 'process', 'filing', 'settlement', 'how to claim', 'claim intimation', 'claim documentation']):
            return 'claims'
        elif any(word in content_lower for word in ['premium', 'deductible', 'copay', 'coinsurance', 'payment', 'payable', 'premium calculation', 'payment terms']):
            return 'financial_terms'
        elif any(word in content_lower for word in ['condition', 'requirement', 'must', 'obligation', 'shall', 'warrant', 'general conditions', 'special conditions']):
            return 'conditions'
        elif any(word in content_lower for word in ['waiting period', 'cooling period', 'grace period', 'moratorium period', 'probationary period']):
            return 'waiting_periods'
        elif any(word in content_lower for word in ['renewal', 'renew', 'continuation', 'portability', 'migration']):
            return 'renewal'
        elif any(word in content_lower for word in ['cancellation', 'termination', 'discontinuation', 'surrender', 'lapse']):
            return 'cancellation'
        elif any(word in content_lower for word in ['grievance', 'complaint', 'ombudsman', 'dispute resolution', 'customer service']):
            return 'grievance'
            
        # Specific insurance benefit sections
        elif any(word in content_lower for word in ['maternity', 'pregnancy', 'delivery', 'childbirth', 'newborn', 'prenatal', 'postnatal']):
            return 'maternity'
        elif any(word in content_lower for word in ['pre-existing', 'existing condition', 'ped', 'pre-existing disease']):
            return 'pre_existing'
        elif any(word in content_lower for word in ['hospitalization', 'inpatient', 'icu', 'room rent', 'ambulance', 'hospital expenses']):
            return 'hospitalization'
        elif any(word in content_lower for word in ['day care', 'daycare', 'outpatient', 'opd', 'day surgery']):
            return 'daycare'
        elif any(word in content_lower for word in ['ayush', 'alternative medicine', 'traditional medicine', 'ayurveda', 'homeopathy', 'unani']):
            return 'ayush'
        elif any(word in content_lower for word in ['organ donor', 'transplant', 'surgery', 'surgical procedure']):
            return 'organ_donor'
        elif any(word in content_lower for word in ['health check-up', 'medical examination', 'preventive care', 'wellness']):
            return 'health_checkup'
        elif any(word in content_lower for word in ['no claim discount', 'ncd', 'loyalty discount', 'bonus', 'cumulative bonus']):
            return 'no_claim_discount'
            
        # Legal document sections
        elif any(word in content_lower for word in ['whereas', 'witnesseth', 'recital', 'preamble', 'background']):
            return 'preamble'
        elif any(word in content_lower for word in ['terms and conditions', 'agreement', 'contract terms', 'general terms']):
            return 'legal_terms'
        elif any(word in content_lower for word in ['representation', 'warranty', 'covenant', 'undertaking']):
            return 'representations_warranties'
        elif any(word in content_lower for word in ['indemnification', 'indemnity', 'hold harmless', 'defend']):
            return 'indemnification'
        elif any(word in content_lower for word in ['confidentiality', 'non-disclosure', 'proprietary information', 'trade secret']):
            return 'confidentiality'
        elif any(word in content_lower for word in ['intellectual property', 'copyright', 'patent', 'trademark', 'ip rights']):
            return 'intellectual_property'
        elif any(word in content_lower for word in ['termination', 'breach', 'default', 'remedy', 'cure period']):
            return 'termination'
        elif any(word in content_lower for word in ['dispute resolution', 'arbitration', 'mediation', 'jurisdiction', 'governing law']):
            return 'dispute_resolution'
        elif any(word in content_lower for word in ['force majeure', 'act of god', 'unforeseeable circumstances']):
            return 'force_majeure'
            
        # HR document sections
        elif any(word in content_lower for word in ['job description', 'position', 'role', 'responsibilities', 'duties']):
            return 'job_description'
        elif any(word in content_lower for word in ['compensation', 'salary', 'wages', 'pay', 'remuneration']):
            return 'compensation'
        elif any(word in content_lower for word in ['benefits', 'perks', 'allowance', 'reimbursement']):
            return 'benefits'
        elif any(word in content_lower for word in ['leave policy', 'vacation', 'sick leave', 'pto', 'time off', 'absence']):
            return 'leave_policy'
        elif any(word in content_lower for word in ['performance', 'review', 'evaluation', 'appraisal', 'kpi']):
            return 'performance'
        elif any(word in content_lower for word in ['code of conduct', 'ethics', 'behavior', 'discipline', 'misconduct']):
            return 'code_of_conduct'
        elif any(word in content_lower for word in ['training', 'development', 'learning', 'skill', 'orientation']):
            return 'training'
        elif any(word in content_lower for word in ['separation', 'resignation', 'retirement', 'exit', 'notice period']):
            return 'separation'
            
        # Compliance document sections
        elif any(word in content_lower for word in ['compliance requirement', 'regulatory requirement', 'mandatory', 'statutory']):
            return 'compliance_requirements'
        elif any(word in content_lower for word in ['risk assessment', 'risk management', 'risk mitigation', 'control']):
            return 'risk_management'
        elif any(word in content_lower for word in ['audit', 'review', 'inspection', 'examination', 'assessment']):
            return 'audit'
        elif any(word in content_lower for word in ['reporting', 'disclosure', 'filing', 'submission', 'notification']):
            return 'reporting'
        elif any(word in content_lower for word in ['data privacy', 'data protection', 'gdpr', 'ccpa', 'personal information']):
            return 'data_privacy'
        elif any(word in content_lower for word in ['anti-money laundering', 'aml', 'kyc', 'customer due diligence']):
            return 'aml_kyc'
        elif any(word in content_lower for word in ['sanctions', 'embargo', 'export control', 'restricted parties']):
            return 'sanctions'
        elif any(word in content_lower for word in ['whistleblower', 'reporting mechanism', 'ethics hotline']):
            return 'whistleblower'
            
        # Document structure
        elif any(word in content_lower for word in ['table', 'schedule', 'annexure', 'appendix', 'exhibit', 'attachment']):
            return 'table'
        elif any(word in content_lower for word in ['example', 'illustration', 'scenario', 'case study']):
            return 'example'
        else:
            return 'content'
    
    async def create_section_summaries(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """
        Create concise summaries for each section using Gemini Flash.
        Summaries help in quickly identifying relevant sections.
        """
        self.logger.info(f"Creating summaries for {len(sections)} sections")
        
        # Process summaries in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(2)  # Limit concurrent summarization to avoid gRPC issues
        
        async def summarize_section(section: DocumentSection) -> DocumentSection:
            async with semaphore:
                try:
                    prompt = f"""Create a concise summary of this document section for search purposes.

Section Type: {section.section_type}
Content Length: {len(section.content)} characters

Content:
{section.content[:2000]}{'...' if len(section.content) > 2000 else ''}

Provide a 2-3 sentence summary that captures:
1. Main topic/purpose of this section
2. Key information or concepts covered  
3. Specific details (amounts, percentages, timeframes, mathematical concepts, scientific principles if present)
4. For scientific texts: key theorems, propositions, laws, or mathematical relationships
5. For insurance documents: key terms, conditions, exclusions, benefits, and limitations

Focus on extracting searchable terms and concepts that would help match user queries about the content.

Summary:"""
                    
                    response = await GEMINI_FLASH_MODEL.generate_content_async(prompt)
                    section.summary = response.text.strip()
                    self.logger.debug(f"Generated summary for {section.section_type} section")
                    
                except Exception as e:
                    self.logger.warning(f"Error generating summary for section: {e}")
                    # Fallback: use first 200 characters
                    section.summary = section.content[:200] + "..." if len(section.content) > 200 else section.content
                
                return section
        
        # Process all sections in parallel
        updated_sections = await asyncio.gather(*[summarize_section(section) for section in sections])
        
        self.logger.info("Section summaries generated successfully")
        return updated_sections
    
    async def find_relevant_sections(self, query: str, sections: List[DocumentSection], top_k: int = 3) -> List[int]:
        """
        Find the most relevant sections for a query using AI-based relevance scoring.
        Returns indices of most relevant sections.
        """
        self.logger.info(f"Finding relevant sections for query: '{query[:50]}...'")
        
        # Create a prompt to evaluate relevance of each section
        section_summaries = []
        for i, section in enumerate(sections):
            section_summaries.append(f"Section {i}: [{section.section_type}] {section.summary}")
        
        prompt = f"""You are analyzing document sections to find the most relevant ones for a user query. You must be thorough in finding relevant content, especially for scientific and mathematical queries.

Query: "{query}"

Available Sections:
{chr(10).join(section_summaries)}

Task: Identify the {top_k * 2} most relevant sections that would likely contain information to answer the query.

Consider:
1. Direct keyword matches (exact terms from the query)
2. Conceptual relevance (related concepts, synonyms, scientific principles)
3. Section type appropriateness for the domain
4. Historical/biographical information (if query asks about people)
5. For scientific queries: look for propositions, theorems, laws, principles that relate to the topic
6. For insurance queries: look for policy terms, conditions, exclusions, benefits, and limitations

Be generous in identifying relevance - err on the side of including sections that might contain useful information rather than being too restrictive.

Respond with ONLY the section numbers (comma-separated, e.g., "2,5,8") of the {top_k * 2} most relevant sections.
If fewer than {top_k * 2} sections are relevant, list only the relevant ones."""
        
        try:
            response = await GEMINI_FLASH_MODEL.generate_content_async(prompt)
            result = response.text.strip()
            
            # Parse section indices
            relevant_indices = []
            for num_str in result.split(','):
                try:
                    idx = int(num_str.strip())
                    if 0 <= idx < len(sections):
                        relevant_indices.append(idx)
                except ValueError:
                    continue
            
            if not relevant_indices:
                # Fallback: return first few sections
                relevant_indices = list(range(min(top_k * 2, len(sections))))
                
            self.logger.info(f"Identified {len(relevant_indices)} relevant sections: {relevant_indices}")
            return relevant_indices
            
        except Exception as e:
            self.logger.error(f"Error finding relevant sections: {e}")
            # Fallback: return first few sections
            return list(range(min(top_k * 2, len(sections))))
    
    def chunk_sections(self, sections: List[DocumentSection], chunk_size: int = None) -> List[Dict[str, Any]]:
        """
        Convert relevant sections into chunks for embedding processing.
        Uses NLTK or SpaCy for sentence tokenization when available.
        """
        if chunk_size is None:
            chunk_size = getattr(settings, 'CHUNK_SIZE', 2000)
        
        chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 400)
        
        all_chunks = []
        
        # Import metadata extraction service
        try:
            from src.services.metadata_extraction_service import metadata_extraction_service
            metadata_extractor = metadata_extraction_service
        except ImportError:
            metadata_extractor = None
            self.logger.warning("Metadata extraction service not available")
        
        for section in sections:
            content = section.content
            
            # Add section context to chunks
            section_header = f"[{section.section_type.upper()} SECTION]\n"
            
            # Use semantic chunking with sentence boundaries
            sentences = self._tokenize_sentences(content)
            current_chunk = section_header
            current_chunk_size = len(section_header)
            
            for sentence in sentences:
                sentence_with_space = " " + sentence
                if current_chunk_size + len(sentence_with_space) <= chunk_size:
                    current_chunk += sentence_with_space
                    current_chunk_size += len(sentence_with_space)
                else:
                    # Add current chunk to results
                    if len(current_chunk.strip()) > len(section_header):
                        chunk_text = current_chunk.strip()
                        # Add metadata to chunk
                        chunk_metadata = {
                            "section_type": section.section_type,
                            "section_summary": section.summary[:200] if section.summary else ""
                        }
                        
                        # Add advanced metadata if extractor is available
                        if metadata_extractor:
                            try:
                                advanced_metadata = metadata_extractor.extract_metadata_from_chunk(chunk_text)
                                chunk_metadata.update(advanced_metadata)
                            except Exception as e:
                                self.logger.warning(f"Metadata extraction failed for chunk: {e}")
                        
                        all_chunks.append({
                            "text": chunk_text,
                            "metadata": chunk_metadata
                        })
                    
                    # Start new chunk with overlap
                    # Include some content from previous chunk for context
                    overlap_content = self._get_overlap_content(current_chunk, chunk_overlap)
                    current_chunk = section_header + overlap_content + sentence_with_space
                    current_chunk_size = len(section_header) + len(overlap_content) + len(sentence_with_space)
            
            # Add the final chunk
            if len(current_chunk.strip()) > len(section_header):
                chunk_text = current_chunk.strip()
                # Add metadata to chunk
                chunk_metadata = {
                    "section_type": section.section_type,
                    "section_summary": section.summary[:200] if section.summary else ""
                }
                
                # Add advanced metadata if extractor is available
                if metadata_extractor:
                    try:
                        advanced_metadata = metadata_extractor.extract_metadata_from_chunk(chunk_text)
                        chunk_metadata.update(advanced_metadata)
                    except Exception as e:
                        self.logger.warning(f"Metadata extraction failed for chunk: {e}")
                
                all_chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata
                })
        
        self.logger.info(f"Generated {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks
    
    async def process_large_document(
        self, 
        document: str, 
        query: str, 
        max_sections: int = 5
    ) -> Tuple[List[str], ProcessingMetrics]:
        """
        Main hierarchical processing pipeline for large documents with caching.
        Returns relevant chunks and processing metrics.
        """
        start_time = asyncio.get_event_loop().time()
        
        # Generate document hash for caching
        import hashlib
        doc_hash = hashlib.sha256(document.encode()).hexdigest()
        
        self.logger.info(f"Starting hierarchical processing for document ({len(document)} chars)")
        
        # Check if document sections are already cached
        if doc_hash in self.document_sections_cache:
            self.logger.info("Using cached document sections")
            sections_with_summaries = self.document_sections_cache[doc_hash]
        else:
            # Phase 1: Split into sections
            sections = self.split_into_sections(document)
            
            # Phase 2: Create section summaries
            sections_with_summaries = await self.create_section_summaries(sections)
            
            # Cache the processed sections
            self.document_sections_cache[doc_hash] = sections_with_summaries
            self.logger.info(f"Cached {len(sections_with_summaries)} document sections")
        
        # Phase 3: Find relevant sections (this is query-specific, so can't be cached)
        relevant_section_indices = await self.find_relevant_sections(
            query, 
            sections_with_summaries, 
            top_k=max_sections * 2  # Increase sections for better coverage
        )
        
        # Phase 4: Extract only relevant sections
        relevant_sections = [sections_with_summaries[i] for i in relevant_section_indices]
        
        # Phase 5: Chunk relevant sections
        relevant_chunks = self.chunk_sections(relevant_sections)
        
        # Calculate metrics
        processing_time = asyncio.get_event_loop().time() - start_time
        total_possible_chunks = len(document) // getattr(settings, 'CHUNK_SIZE', 2000)
        
        metrics = ProcessingMetrics(
            total_chunks=total_possible_chunks,
            processed_chunks=len(relevant_chunks),
            processing_time=processing_time,
            sections_identified=len(sections_with_summaries),
            relevant_sections=len(relevant_sections)
        )
        
        reduction_percentage = (total_possible_chunks - len(relevant_chunks)) / total_possible_chunks * 100
        
        self.logger.info(f"Hierarchical processing completed in {processing_time:.2f}s")
        self.logger.info(f"Reduced processing by {reduction_percentage:.1f}% ({total_possible_chunks} -> {len(relevant_chunks)} chunks)")
        
        return relevant_chunks, metrics

    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences using NLTK or SpaCy if available,
        falling back to regex-based splitting.
        """
        if NLTK_AVAILABLE:
            try:
                # Download punkt tokenizer if not already present
                try:
                    find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                return sent_tokenize(text)
            except Exception as e:
                self.logger.warning(f"NLTK sentence tokenization failed: {e}")
        
        if SPACY_AVAILABLE:
            try:
                # Load a small English model if not already loaded
                if not hasattr(self, '_spacy_nlp'):
                    # Try to load a model, fallback to downloading if needed
                    try:
                        self._spacy_nlp = spacy.load("en_core_web_sm")
                    except OSError:
                        self.logger.warning("SpaCy model not found, trying to load default model")
                        try:
                            self._spacy_nlp = spacy.load("en")
                        except OSError:
                            self.logger.warning("No SpaCy model found")
                            raise
                doc = self._spacy_nlp(text)
                return [sent.text.strip() for sent in doc.sents]
            except Exception as e:
                self.logger.warning(f"SpaCy sentence tokenization failed: {e}")
        
        # Fallback to regex-based sentence splitting
        return re.split(r'(?<=[.!?])\s+', text)
    
    def _get_overlap_content(self, chunk: str, overlap_size: int) -> str:
        """
        Extract overlap content from the end of a chunk for context preservation.
        """
        if overlap_size <= 0:
            return ""
        
        # Use semantic chunking to get appropriate overlap
        sentences = self._tokenize_sentences(chunk)
        
        # Build overlap from the end, working backwards
        overlap_content = ""
        current_size = 0
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            sentence_with_space = sentence + " "
            if current_size + len(sentence_with_space) <= overlap_size:
                overlap_content = sentence_with_space + overlap_content
                current_size += len(sentence_with_space)
            else:
                break
        
        return overlap_content

# Global instance
hierarchical_chunking_service = HierarchicalChunkingService()
