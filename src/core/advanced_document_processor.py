"""
Advanced Document Processor optimized for 600+ page documents
Implements hierarchical chunking, parallel processing, and domain-specific structure understanding
"""

import asyncio
import aiohttp
import fitz  # PyMuPDF
import docx
import re
import nltk
from typing import List, Dict, Any, Tuple, Optional
from io import BytesIO
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from src.core.config import settings
from src.core.document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    INSURANCE_POLICY = "insurance_policy"
    LEGAL_CONTRACT = "legal_contract"
    HR_DOCUMENT = "hr_document"
    COMPLIANCE_DOCUMENT = "compliance_document"
    GENERAL = "general"


@dataclass
class DocumentSection:
    """Represents a logical section of a document"""
    title: str
    content: str
    level: int  # 1=main section, 2=subsection, etc.
    page_start: int
    page_end: int
    section_type: str  # table_of_contents, clause, definition, etc.
    parent_section: Optional[str] = None
    metadata: Dict[str, Any] = None


class AdvancedDocumentProcessor:
    """
    High-performance document processor optimized for 600+ page documents
    Features: Parallel processing, hierarchical chunking, domain-specific structure understanding
    """

    def __init__(self):
        self.max_chunk_size = settings.MAX_CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.max_doc_size_mb = settings.MAX_DOCUMENT_SIZE_MB
        
        # Performance optimizations
        self.max_workers = min(8, (settings.MAX_PARALLEL_QUESTIONS // 4))  # Conservative CPU usage
        self.embedding_cache = {}  # Cache for document structure patterns
        
        # Domain-specific patterns
        self.domain_patterns = self._initialize_domain_patterns()

    def _initialize_domain_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize domain-specific patterns for structure recognition"""
        return {
            "insurance_policy": {
                "section_headers": [
                    r"SECTION\s+\d+",
                    r"CLAUSE\s+\d+",
                    r"ARTICLE\s+\d+",
                    r"COVERAGE\s+[A-Z]",
                    r"BENEFIT\s+\d+",
                    r"EXCLUSIONS?",
                    r"DEFINITIONS?",
                    r"GENERAL CONDITIONS?",
                    r"CLAIMS? PROCEDURE",
                    r"PREMIUM PAYMENT",
                    r"POLICY SCHEDULE",
                    r"WAITING PERIOD",
                    r"PRE-EXISTING CONDITIONS?",
                ],
                "key_terms": [
                    r"sum insured", r"premium", r"deductible", r"co-pay", 
                    r"waiting period", r"grace period", r"coverage", r"benefit",
                    r"exclusion", r"claim", r"policyholder", r"insured person"
                ],
                "numeric_patterns": [
                    r"\d+\s*(?:days?|months?|years?)",  # Time periods
                    r"(?:Rs\.?|₹)\s*[\d,]+",  # Indian currency
                    r"\d+(?:\.\d+)?%",  # Percentages
                    r"\d+(?:\.\d+)?\s*(?:lakh|crore)",  # Indian number system
                ]
            },
            "legal_contract": {
                "section_headers": [
                    r"ARTICLE\s+[IVX\d]+",
                    r"SECTION\s+\d+",
                    r"CLAUSE\s+\d+",
                    r"WHEREAS",
                    r"NOW,?\s+THEREFORE",
                    r"TERMS AND CONDITIONS",
                    r"REPRESENTATIONS AND WARRANTIES",
                    r"INDEMNIFICATION",
                    r"TERMINATION",
                    r"GOVERNING LAW",
                    r"DISPUTE RESOLUTION",
                ],
                "key_terms": [
                    r"party", r"agreement", r"contract", r"obligation", 
                    r"liability", r"indemnify", r"breach", r"terminate",
                    r"force majeure", r"confidential", r"intellectual property"
                ],
                "numeric_patterns": [
                    r"\d+\s*(?:days?|months?|years?)",
                    r"(?:USD?|₹|Rs\.?)\s*[\d,]+",
                    r"\d+(?:\.\d+)?%",
                ]
            },
            "hr_document": {
                "section_headers": [
                    r"EMPLOYEE\s+(?:HANDBOOK|MANUAL|POLICY)",
                    r"CODE\s+OF\s+CONDUCT",
                    r"COMPENSATION",
                    r"BENEFITS?",
                    r"LEAVE\s+POLICY",
                    r"PERFORMANCE\s+MANAGEMENT",
                    r"DISCIPLINARY\s+ACTION",
                    r"GRIEVANCE\s+PROCEDURE",
                    r"EQUAL\s+OPPORTUNITY",
                    r"HEALTH\s+AND\s+SAFETY",
                ],
                "key_terms": [
                    r"employee", r"employer", r"salary", r"wages", 
                    r"benefits", r"leave", r"vacation", r"sick leave",
                    r"performance", r"appraisal", r"promotion", r"termination"
                ],
                "numeric_patterns": [
                    r"\d+\s*(?:days?|months?|years?)",
                    r"(?:USD?|₹|Rs\.?)\s*[\d,]+",
                    r"\d+(?:\.\d+)?%",
                ]
            }
        }

    async def process_document(
        self, document_url: str
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Process large documents with parallel processing and hierarchical chunking
        Optimized for 600+ page documents with <30 second processing time
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Step 1: Download document (parallel with other prep work)
            download_task = asyncio.create_task(self._download_document(document_url))
            
            # Step 2: Prepare processing pipeline while downloading
            doc_hash = hashlib.md5(document_url.encode()).hexdigest()
            
            # Step 3: Download completion
            document_data, content_type = await download_task
            
            # Step 4: Detect document type and extract text (parallel processing)
            if "pdf" in content_type.lower():
                text, metadata, sections = await self._extract_pdf_with_structure(document_data)
            elif "word" in content_type.lower() or "docx" in content_type.lower():
                text, metadata, sections = await self._extract_docx_with_structure(document_data)
            else:
                text = document_data.decode("utf-8", errors="ignore")
                metadata = {"type": "text", "size": len(text)}
                sections = []

            # Step 5: Detect document domain
            doc_type = self._detect_document_type(text, metadata)
            metadata.update({
                "source_url": document_url,
                "content_type": content_type,
                "document_type": doc_type.value,
                "sections_detected": len(sections),
                "processing_time_extract": asyncio.get_event_loop().time() - start_time
            })

            # Step 6: Create hierarchical chunks (parallel processing)
            chunks = await self._create_hierarchical_chunks(text, metadata, sections, doc_type)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            metadata["total_processing_time"] = processing_time
            
            logger.info(
                f"Advanced processing completed: {len(chunks)} chunks, "
                f"{len(sections)} sections in {processing_time:.2f}s"
            )

            return chunks, metadata

        except Exception as e:
            logger.error(f"Error in advanced document processing {document_url}: {str(e)}")
            raise

    async def _download_document(self, url: str) -> Tuple[bytes, str]:
        """Optimized document download with proper timeout and chunking"""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60, connect=10)
        ) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download document: HTTP {response.status}")

                content_type = response.headers.get("content-type", "application/octet-stream")
                
                # Stream large files efficiently
                data = BytesIO()
                chunk_size = 8192
                total_size = 0
                
                async for chunk in response.content.iter_chunked(chunk_size):
                    data.write(chunk)
                    total_size += len(chunk)
                    
                    # Check size limit during download
                    if total_size > self.max_doc_size_mb * 1024 * 1024:
                        raise Exception(f"Document too large: {total_size} bytes")

                return data.getvalue(), content_type

    async def _extract_pdf_with_structure(
        self, pdf_data: bytes
    ) -> Tuple[str, Dict[str, Any], List[DocumentSection]]:
        """
        Extract PDF text with structure analysis using parallel processing
        Optimized for large PDFs with 600+ pages
        """
        def process_pdf_pages(pdf_data: bytes, start_page: int, end_page: int) -> Dict[str, Any]:
            """Process a range of PDF pages in a separate thread"""
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            
            pages_data = []
            toc_entries = []
            
            # Extract table of contents if available
            toc = doc.get_toc()
            if toc:
                for level, title, page_num in toc:
                    if start_page <= page_num - 1 < end_page:  # TOC page numbers are 1-based
                        toc_entries.append({
                            "level": level,
                            "title": title.strip(),
                            "page": page_num - 1  # Convert to 0-based
                        })
            
            for page_num in range(start_page, min(end_page, len(doc))):
                page = doc[page_num]
                
                # Extract text with layout preservation
                page_text = page.get_text()
                
                # Extract page-level structure (headers, footers, etc.)
                blocks = page.get_text("dict")["blocks"]
                headers = []
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                font_size = span["size"]
                                flags = span["flags"]
                                
                                # Detect headers (larger font, bold, etc.)
                                if font_size > 12 and (flags & 2**4 or flags & 2**5):  # Bold or large
                                    if len(text) > 5 and len(text) < 100:
                                        headers.append({
                                            "text": text,
                                            "font_size": font_size,
                                            "page": page_num
                                        })
                
                if page_text.strip():
                    cleaned_text = self._clean_text(page_text)
                    pages_data.append({
                        "page_num": page_num,
                        "text": cleaned_text,
                        "headers": headers,
                        "word_count": len(cleaned_text.split())
                    })
            
            doc.close()
            
            return {
                "pages": pages_data,
                "toc_entries": toc_entries,
                "start_page": start_page,
                "end_page": end_page
            }

        # Parallel processing for large PDFs
        doc = fitz.open(stream=pdf_data, filetype="pdf")
        total_pages = len(doc)
        doc.close()  # Close immediately to free memory
        
        metadata = {
            "type": "pdf",
            "pages": total_pages,
            "size": len(pdf_data)
        }
        
        # Process pages in parallel chunks
        chunk_size = max(20, total_pages // self.max_workers)  # Adaptive chunk size
        tasks = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()
            
            for start_page in range(0, total_pages, chunk_size):
                end_page = min(start_page + chunk_size, total_pages)
                task = loop.run_in_executor(
                    executor,
                    process_pdf_pages,
                    pdf_data,
                    start_page,
                    end_page
                )
                tasks.append(task)
            
            # Wait for all chunks to complete
            results = await asyncio.gather(*tasks)

        # Combine results
        all_pages = []
        all_toc = []
        
        for result in results:
            all_pages.extend(result["pages"])
            all_toc.extend(result["toc_entries"])
        
        # Sort by page number
        all_pages.sort(key=lambda x: x["page_num"])
        all_toc.sort(key=lambda x: x["page"])
        
        # Combine text
        text_parts = []
        for page_data in all_pages:
            text_parts.append(f"[Page {page_data['page_num'] + 1}]\n{page_data['text']}")
        
        full_text = "\n\n".join(text_parts)
        metadata["extracted_chars"] = len(full_text)
        
        # Create document sections from TOC and headers
        sections = self._create_sections_from_structure(all_pages, all_toc)
        
        return full_text, metadata, sections

    async def _extract_docx_with_structure(
        self, docx_data: bytes
    ) -> Tuple[str, Dict[str, Any], List[DocumentSection]]:
        """Extract DOCX text with structure analysis"""
        def process_docx(docx_data: bytes) -> Dict[str, Any]:
            doc = docx.Document(BytesIO(docx_data))
            
            paragraphs_data = []
            headers = []
            
            for i, para in enumerate(doc.paragraphs):
                if para.text.strip():
                    # Analyze paragraph style for structure
                    style_name = para.style.name if para.style else "Normal"
                    
                    para_data = {
                        "index": i,
                        "text": para.text.strip(),
                        "style": style_name,
                        "is_header": "Heading" in style_name
                    }
                    
                    paragraphs_data.append(para_data)
                    
                    if para_data["is_header"]:
                        headers.append({
                            "text": para.text.strip(),
                            "level": int(style_name[-1]) if style_name[-1].isdigit() else 1,
                            "index": i
                        })
            
            return {
                "paragraphs": paragraphs_data,
                "headers": headers
            }

        # Process DOCX in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(executor, process_docx, docx_data)
        
        paragraphs = result["paragraphs"]
        headers = result["headers"]
        
        # Combine text
        text_parts = [para["text"] for para in paragraphs]
        full_text = "\n\n".join(text_parts)
        
        metadata = {
            "type": "docx",
            "paragraphs": len(paragraphs),
            "size": len(docx_data),
            "extracted_chars": len(full_text)
        }
        
        # Create sections from headers
        sections = self._create_sections_from_headers(paragraphs, headers)
        
        return full_text, metadata, sections

    def _create_sections_from_structure(
        self, pages_data: List[Dict], toc_entries: List[Dict]
    ) -> List[DocumentSection]:
        """Create document sections from page structure and TOC"""
        sections = []
        
        # Use TOC as primary structure
        for i, toc_entry in enumerate(toc_entries):
            start_page = toc_entry["page"]
            end_page = toc_entries[i + 1]["page"] if i + 1 < len(toc_entries) else len(pages_data) - 1
            
            # Collect content for this section
            section_content = []
            for page_data in pages_data:
                if start_page <= page_data["page_num"] <= end_page:
                    section_content.append(page_data["text"])
            
            section = DocumentSection(
                title=toc_entry["title"],
                content="\n\n".join(section_content),
                level=toc_entry["level"],
                page_start=start_page,
                page_end=end_page,
                section_type="toc_section",
                metadata={"word_count": sum(page["word_count"] for page in pages_data 
                                         if start_page <= page["page_num"] <= end_page)}
            )
            sections.append(section)
        
        # If no TOC, create sections from headers
        if not sections:
            sections = self._create_sections_from_page_headers(pages_data)
        
        return sections

    def _create_sections_from_page_headers(
        self, pages_data: List[Dict]
    ) -> List[DocumentSection]:
        """Create sections from detected page headers"""
        sections = []
        current_section = None
        
        for page_data in pages_data:
            page_headers = page_data.get("headers", [])
            
            # Look for new section headers
            for header in page_headers:
                if self._is_section_header(header["text"]):
                    # Close previous section
                    if current_section:
                        sections.append(current_section)
                    
                    # Start new section
                    current_section = DocumentSection(
                        title=header["text"],
                        content=page_data["text"],
                        level=1,
                        page_start=page_data["page_num"],
                        page_end=page_data["page_num"],
                        section_type="header_section",
                        metadata={"font_size": header["font_size"]}
                    )
                    break
            else:
                # No new header, add to current section
                if current_section:
                    current_section.content += "\n\n" + page_data["text"]
                    current_section.page_end = page_data["page_num"]
                else:
                    # Create default section
                    current_section = DocumentSection(
                        title=f"Section starting page {page_data['page_num'] + 1}",
                        content=page_data["text"],
                        level=1,
                        page_start=page_data["page_num"],
                        page_end=page_data["page_num"],
                        section_type="default_section"
                    )
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections

    def _create_sections_from_headers(
        self, paragraphs: List[Dict], headers: List[Dict]
    ) -> List[DocumentSection]:
        """Create sections from DOCX headers"""
        sections = []
        
        for i, header in enumerate(headers):
            start_idx = header["index"]
            end_idx = headers[i + 1]["index"] if i + 1 < len(headers) else len(paragraphs)
            
            # Collect content
            section_content = []
            for para in paragraphs[start_idx:end_idx]:
                section_content.append(para["text"])
            
            section = DocumentSection(
                title=header["text"],
                content="\n\n".join(section_content),
                level=header["level"],
                page_start=0,  # DOCX doesn't have clear page boundaries
                page_end=0,
                section_type="docx_section",
                metadata={"paragraph_start": start_idx, "paragraph_end": end_idx}
            )
            sections.append(section)
        
        return sections

    def _detect_document_type(self, text: str, metadata: Dict[str, Any]) -> DocumentType:
        """Detect document type based on content patterns"""
        text_lower = text.lower()
        
        # Count domain-specific terms
        type_scores = {}
        
        for doc_type, patterns in self.domain_patterns.items():
            score = 0
            
            # Check section headers
            for pattern in patterns["section_headers"]:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches * 3  # Headers are strong indicators
            
            # Check key terms
            for pattern in patterns["key_terms"]:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches
            
            type_scores[doc_type] = score
        
        # Return the type with highest score
        if type_scores:
            best_type = max(type_scores, key=type_scores.get)
            if type_scores[best_type] > 10:  # Minimum confidence threshold
                return DocumentType(best_type)
        
        return DocumentType.GENERAL

    def _is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header"""
        text = text.strip()
        
        # Check against known patterns
        for doc_type, patterns in self.domain_patterns.items():
            for pattern in patterns["section_headers"]:
                if re.match(pattern, text, re.IGNORECASE):
                    return True
        
        # Generic header patterns
        generic_patterns = [
            r"^[A-Z\s]{5,50}$",  # ALL CAPS short text
            r"^\d+\.\s+[A-Z]",   # Numbered sections
            r"^[A-Z][a-z\s]+:$", # Title Case with colon
        ]
        
        for pattern in generic_patterns:
            if re.match(pattern, text):
                return True
        
        return False

    async def _create_hierarchical_chunks(
        self,
        text: str,
        metadata: Dict[str, Any],
        sections: List[DocumentSection],
        doc_type: DocumentType
    ) -> List[DocumentChunk]:
        """
        Create hierarchical chunks optimized for accuracy and speed
        Uses parallel processing for large documents
        """
        if not sections:
            # Fallback to basic chunking
            return await self._create_basic_chunks(text, metadata)
        
        def process_section_chunks(section: DocumentSection) -> List[Dict[str, Any]]:
            """Process a single section into chunks"""
            section_chunks = []
            
            if len(section.content) <= self.max_chunk_size:
                # Small section, create single chunk
                chunk_data = {
                    "text": section.content,
                    "page_num": section.page_start,
                    "metadata": {
                        "section_title": section.title,
                        "section_level": section.level,
                        "section_type": section.section_type,
                        "doc_type": doc_type.value,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        **section.metadata
                    }
                }
                section_chunks.append(chunk_data)
            else:
                # Large section, split semantically
                semantic_chunks = self._create_semantic_chunks_for_section(section, doc_type)
                section_chunks.extend(semantic_chunks)
            
            return section_chunks

        # Process sections in parallel
        tasks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()
            
            for section in sections:
                task = loop.run_in_executor(executor, process_section_chunks, section)
                tasks.append(task)
            
            # Wait for all sections to be processed
            results = await asyncio.gather(*tasks)

        # Combine results and create DocumentChunk objects
        chunks = []
        chunk_id = 0
        
        for section_results in results:
            for chunk_data in section_results:
                chunk = DocumentChunk(
                    text=chunk_data["text"],
                    page_num=chunk_data["page_num"],
                    chunk_id=f"chunk_{chunk_id}",
                    metadata=chunk_data["metadata"]
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks

    def _create_semantic_chunks_for_section(
        self, section: DocumentSection, doc_type: DocumentType
    ) -> List[Dict[str, Any]]:
        """Create semantic chunks within a section based on document type"""
        chunks = []
        text = section.content
        
        # Get domain-specific sentence patterns
        if doc_type in [DocumentType.INSURANCE_POLICY, DocumentType.LEGAL_CONTRACT]:
            # Use clause-based chunking for legal documents
            clauses = self._split_by_clauses(text)
            return self._create_clause_chunks(clauses, section)
        else:
            # Use paragraph-based chunking for other documents
            paragraphs = text.split("\n\n")
            return self._create_paragraph_chunks(paragraphs, section)

    def _split_by_clauses(self, text: str) -> List[str]:
        """Split legal/insurance text by clauses"""
        # Split by numbered clauses, lettered subsections, etc.
        patterns = [
            r"\n\s*\d+\.\s+",  # Numbered clauses
            r"\n\s*\([a-z]\)\s+",  # Lettered subsections
            r"\n\s*\([ivx]+\)\s+",  # Roman numeral subsections
        ]
        
        current_text = text
        for pattern in patterns:
            parts = re.split(pattern, current_text)
            if len(parts) > 1:
                return [part.strip() for part in parts if part.strip()]
        
        # Fallback to sentence splitting
        return nltk.sent_tokenize(text)

    def _create_clause_chunks(
        self, clauses: List[str], section: DocumentSection
    ) -> List[Dict[str, Any]]:
        """Create chunks from legal clauses"""
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for clause in clauses:
            clause_size = len(clause)
            
            if current_size + clause_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "page_num": section.page_start,
                    "metadata": {
                        "section_title": section.title,
                        "section_level": section.level,
                        "section_type": section.section_type,
                        "chunk_type": "clause_based",
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                    }
                })
                
                # Start new chunk with overlap
                overlap = self._get_overlap_text(current_chunk)
                current_chunk = overlap + " " + clause if overlap else clause
                current_size = len(current_chunk)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += " " + clause
                else:
                    current_chunk = clause
                current_size += clause_size
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "page_num": section.page_start,
                "metadata": {
                    "section_title": section.title,
                    "section_level": section.level,
                    "section_type": section.section_type,
                    "chunk_type": "clause_based",
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                }
            })
        
        return chunks

    def _create_paragraph_chunks(
        self, paragraphs: List[str], section: DocumentSection
    ) -> List[Dict[str, Any]]:
        """Create chunks from paragraphs"""
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            if current_size + para_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": current_chunk.strip(),
                    "page_num": section.page_start,
                    "metadata": {
                        "section_title": section.title,
                        "section_level": section.level,
                        "section_type": section.section_type,
                        "chunk_type": "paragraph_based",
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                    }
                })
                
                # Start new chunk
                current_chunk = paragraph
                current_size = para_size
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
                current_size += para_size
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "text": current_chunk.strip(),
                "page_num": section.page_start,
                "metadata": {
                    "section_title": section.title,
                    "section_level": section.level,
                    "section_type": section.section_type,
                    "chunk_type": "paragraph_based",
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                }
            })
        
        return chunks

    async def _create_basic_chunks(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Fallback basic chunking when no structure is detected"""
        chunks = []
        
        # Split by pages if available
        if "[Page " in text:
            sections = re.split(r"\[Page \d+\]", text)
            sections = [section.strip() for section in sections if section.strip()]
        else:
            sections = [text]
        
        chunk_id = 0
        for section_idx, section in enumerate(sections):
            section_chunks = self._create_semantic_chunks(section, section_idx)
            
            for chunk_text in section_chunks:
                if len(chunk_text.strip()) > 50:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        page_num=section_idx,
                        chunk_id=f"chunk_{chunk_id}",
                        metadata={
                            "section": section_idx,
                            "doc_type": metadata.get("document_type", "general"),
                            "chunk_type": "basic"
                        }
                    )
                    chunks.append(chunk)
                    chunk_id += 1
        
        return chunks

    def _create_semantic_chunks(self, text: str, section_idx: int) -> List[str]:
        """Basic semantic chunking (from original processor)"""
        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []
        sentences = nltk.sent_tokenize(text)
        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_size = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size += sentence_size

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        if len(text) <= self.chunk_overlap:
            return text

        sentences = nltk.sent_tokenize(text)
        overlap_text = ""

        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= self.chunk_overlap:
                overlap_text = sentence + " " + overlap_text
            else:
                break

        return overlap_text.strip()

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep punctuation
        text = re.sub(
            r"[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\@\#\$\%\&\*\+\=\<\>\~\`\|\\]",
            "",
            text,
        )

        # Fix common OCR errors
        text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

        return text.strip()
    
