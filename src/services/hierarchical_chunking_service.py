import logging
import asyncio
import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from src.services.llm_clients import GEMINI_FLASH_MODEL
from src.core.config import settings

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
        
        # Patterns for identifying section boundaries in business documents (insurance, legal, HR, compliance)
        section_patterns = [
            # Document structure patterns
            r'\n\s*(?:SECTION|Section|PART|Part|CHAPTER|Chapter|ARTICLE|Article)\s+[IVXLC\d]+',  # Roman/numeric sections
            r'\n\s*\d+\.\s+[A-Z][A-Za-z\s]{10,}',  # Numbered sections with titles
            r'\n\s*[A-Z\s]{5,}:\s*\n',  # ALL CAPS headers with colons
            r'\n\s*(?:TABLE|SCHEDULE|APPENDIX|ANNEXURE|EXHIBIT|ATTACHMENT)',  # Document structure keywords
            
            # Insurance domain patterns
            r'\n\s*(?:DEFINITIONS|COVERAGE|EXCLUSIONS|CONDITIONS|CLAIMS|BENEFITS|DEDUCTIBLE|PREMIUM|POLICY)',
            r'\n\s*(?:WAITING PERIOD|PRE-EXISTING|COPAYMENT|COINSURANCE|OUT-OF-POCKET|NETWORK)',
            
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
        """Identify the type of business document section based on content patterns"""
        content_lower = content.lower()
        
        # Insurance-specific sections
        if any(word in content_lower for word in ['definition', 'meaning', 'term', 'means', 'shall mean', 'defined as']):
            return 'definitions'
        elif any(word in content_lower for word in ['coverage', 'benefit', 'cover', 'insured', 'policy limit']):
            return 'coverage'
        elif any(word in content_lower for word in ['exclusion', 'not covered', 'except', 'limitation', 'excluded']):
            return 'exclusions'
        elif any(word in content_lower for word in ['claim', 'procedure', 'process', 'filing', 'settlement']):
            return 'claims'
        elif any(word in content_lower for word in ['premium', 'deductible', 'copay', 'coinsurance', 'payment']):
            return 'financial_terms'
        elif any(word in content_lower for word in ['condition', 'requirement', 'must', 'obligation']):
            return 'conditions'
            
        # Legal sections
        elif any(word in content_lower for word in ['terms and conditions', 'agreement', 'contract', 'liability']):
            return 'legal_terms'
        elif any(word in content_lower for word in ['termination', 'breach', 'default', 'remedy', 'arbitration']):
            return 'legal_procedures'
            
        # HR sections
        elif any(word in content_lower for word in ['employment', 'employee', 'compensation', 'salary', 'wages']):
            return 'employment'
        elif any(word in content_lower for word in ['vacation', 'leave', 'sick', 'pto', 'time off']):
            return 'benefits'
        elif any(word in content_lower for word in ['performance', 'review', 'evaluation', 'discipline']):
            return 'performance'
            
        # Compliance sections
        elif any(word in content_lower for word in ['compliance', 'regulatory', 'regulation', 'audit']):
            return 'compliance'
        elif any(word in content_lower for word in ['privacy', 'confidential', 'data protection', 'security']):
            return 'privacy'
            
        # Document structure
        elif any(word in content_lower for word in ['table', 'schedule', 'rate', 'appendix', 'exhibit']):
            return 'table'
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
3. Specific details (amounts, percentages, timeframes if present)

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
        
        prompt = f"""You are analyzing document sections to find the most relevant ones for a user query.

Query: "{query}"

Available Sections:
{chr(10).join(section_summaries)}

Task: Identify the {top_k} most relevant sections that would likely contain information to answer the query.

Consider:
1. Direct keyword matches
2. Conceptual relevance
3. Section type appropriateness

Respond with ONLY the section numbers (comma-separated, e.g., "2,5,8") of the {top_k} most relevant sections.
If fewer than {top_k} sections are relevant, list only the relevant ones."""
        
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
                relevant_indices = list(range(min(top_k, len(sections))))
                
            self.logger.info(f"Identified {len(relevant_indices)} relevant sections: {relevant_indices}")
            return relevant_indices
            
        except Exception as e:
            self.logger.error(f"Error finding relevant sections: {e}")
            # Fallback: return first few sections
            return list(range(min(top_k, len(sections))))
    
    def chunk_sections(self, sections: List[DocumentSection], chunk_size: int = None) -> List[str]:
        """
        Convert relevant sections into chunks for embedding processing.
        """
        if chunk_size is None:
            chunk_size = getattr(settings, 'CHUNK_SIZE', 2000)
        
        chunk_overlap = getattr(settings, 'CHUNK_OVERLAP', 400)
        
        all_chunks = []
        
        for section in sections:
            content = section.content
            
            # Add section context to chunks
            section_header = f"[{section.section_type.upper()} SECTION]\n"
            
            if len(content) <= chunk_size:
                # Section fits in one chunk
                all_chunks.append(section_header + content)
            else:
                # Split section into overlapping chunks
                start = 0
                while start < len(content):
                    end = start + chunk_size
                    chunk_content = content[start:end]
                    
                    # Try to end at a sentence boundary
                    if end < len(content):
                        last_period = chunk_content.rfind('.')
                        if last_period > len(chunk_content) * 0.7:  # If period is in last 30%
                            chunk_content = chunk_content[:last_period + 1]
                            end = start + len(chunk_content)
                    
                    all_chunks.append(section_header + chunk_content)
                    
                    # Move start position with overlap
                    start = end - chunk_overlap
                    if start >= len(content):
                        break
        
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
            top_k=max_sections
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

# Global instance
hierarchical_chunking_service = HierarchicalChunkingService()