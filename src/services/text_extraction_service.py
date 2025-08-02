import logging
import asyncio
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple, Optional
import re
import statistics
import unicodedata
import hashlib
from src.core.config import settings

# SpaCy imports for semantic chunking (optional)
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the model to check if it's available
    try:
        spacy.load("en_core_web_sm")
        SPACY_MODEL_AVAILABLE = True
    except OSError:
        SPACY_MODEL_AVAILABLE = False
        logging.getLogger(__name__).warning("SpaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    SPACY_MODEL_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)

def extract_page_batch_worker(pdf_content: bytes, page_range: List[int], extraction_stats: dict) -> List[Tuple[str, int]]:
    """
    Worker function for process-based parallel extraction.
    This function runs in a separate process.
    """
    import fitz
    
    results = []
    
    try:
        # Open PDF in this process
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Create a temporary text extraction service instance
        temp_service = TextExtractionService()
        
        for page_num in page_range:
            try:
                page = pdf_document[page_num]
                page_text = temp_service._extract_page_text_multimethod(page, page_num, extraction_stats)
                if page_text and page_text.strip():
                    results.append((page_text, page_num + 1))
            except Exception as e:
                # Log error but continue processing other pages
                print(f"Error extracting page {page_num + 1}: {e}")
        
        pdf_document.close()
        
    except Exception as e:
        print(f"Error in batch worker for pages {page_range[0]+1}-{page_range[-1]+1}: {e}")
    
    return results

class TextExtractionService:
    """
    Advanced PDF text extraction service that extracts text and metadata (page numbers).
    """
    
    def __init__(self):
        self.logger = logger
        self.extraction_methods = ["dict", "text", "html", "blocks"]
        self._spacy_nlp = None  # Lazy loading of SpaCy model
        
        # Initialize SpaCy model if available
        if SPACY_AVAILABLE and SPACY_MODEL_AVAILABLE:
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
                self.logger.info("SpaCy model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load SpaCy model: {e}")
                self._spacy_nlp = None

    async def extract_text_from_pdf_with_metadata(self, pdf_content: bytes) -> List[Tuple[str, int]]:
        """
        Extracts text from each page of a PDF and returns it with the page number.
        """
        self.logger.info("Starting advanced PDF text extraction with metadata.")
        try:
            pages_with_metadata = await asyncio.wait_for(
                asyncio.to_thread(self._extract_text_advanced, pdf_content),
                timeout=settings.PDF_EXTRACTION_TIMEOUT
            )
            self.logger.info(f"Extracted text from {len(pages_with_metadata)} pages.")
            return pages_with_metadata
        except asyncio.TimeoutError:
            self.logger.error(f"PDF text extraction timed out after {settings.PDF_EXTRACTION_TIMEOUT} seconds.")
            raise Exception(f"PDF processing timed out after {settings.PDF_EXTRACTION_TIMEOUT} seconds.")
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
            raise
    
    def _extract_text_advanced(self, pdf_content: bytes) -> List[Tuple[str, int]]:
        """
        Core logic for extracting text and page numbers with adaptive processing.
        Fast mode optimization: Skip analysis but process ALL pages.
        """
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        total_pages = len(pdf_document)
        self.logger.info(f"PDF has {total_pages} pages. Starting extraction...")
        
        # Fast mode: Process all pages but with optimizations
        if settings.ENABLE_FAST_MODE:
            if settings.FAST_MODE_MAX_PAGES > 0 and total_pages > settings.FAST_MODE_MAX_PAGES:
                self.logger.info(f"⚡ FAST MODE: Extracting only first {settings.FAST_MODE_MAX_PAGES} pages (of {total_pages}) for speed")
                effective_pages = settings.FAST_MODE_MAX_PAGES
            else:
                self.logger.info(f"⚡ FAST MODE: Processing ALL {total_pages} pages with optimizations")
                effective_pages = total_pages
        else:
            effective_pages = total_pages
        
        # Choose processing strategy based on effective page count
        if effective_pages >= settings.ULTRA_PARALLEL_PDF_THRESHOLD:
            self.logger.info(f"Using ultra-parallel processing ({effective_pages} pages)")
            return self._ultra_parallel_extract_pages_limited(pdf_document, effective_pages)
        elif effective_pages >= settings.PARALLEL_PDF_THRESHOLD:
            self.logger.info(f"Using parallel processing ({effective_pages} pages)")
            return self._parallel_extract_pages_limited(pdf_document, effective_pages)
        else:
            self.logger.info(f"Using sequential processing ({effective_pages} pages)")
            return self._sequential_extract_pages_limited(pdf_document, effective_pages)
    
    def _parallel_extract_pages(self, pdf_document: fitz.Document) -> List[Tuple[str, int]]:
        """
        Extract pages in parallel using threading for large documents.
        """
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        
        total_pages = len(pdf_document)
        self.logger.info(f"Using parallel extraction for {total_pages} pages")
        
        # For ultra-large documents (500+ pages), use process-based parallelism
        if total_pages > 500:
            return self._ultra_parallel_extract_pages(pdf_document)
        
        # Analyze document once
        extraction_stats = self._analyze_document(pdf_document)
        self.logger.info(f"Document analysis complete. Recommended method: {extraction_stats['recommended_method']}")
        
        # Get PDF content for thread-safe processing
        pdf_content = pdf_document.tobytes()
        pdf_document.close()  # Close original to free memory
        
        pages_with_metadata = []
        
        # Process pages in batches to avoid memory issues
        batch_size = min(50, max(10, total_pages // 10))  # Adaptive batch size
        max_workers = min(8, batch_size)  # Limit concurrent threads
        
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            self.logger.info(f"Processing batch {batch_start+1}-{batch_end} ({batch_end/total_pages*100:.1f}% complete)")
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create futures for each page in the batch
                futures = {}
                for page_num in range(batch_start, batch_end):
                    future = executor.submit(
                        self._extract_single_page_safe, 
                        pdf_content, 
                        page_num, 
                        extraction_stats
                    )
                    futures[future] = page_num
                
                # Collect results as they complete
                batch_results = []
                for future in concurrent.futures.as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_text = future.result(timeout=30)  # 30 second timeout per page
                        if page_text and page_text.strip():
                            batch_results.append((page_text, page_num + 1))
                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                
                # Sort batch results by page number and add to main results
                batch_results.sort(key=lambda x: x[1])
                pages_with_metadata.extend(batch_results)
        
        self.logger.info(f"Parallel extraction complete. Extracted text from {len(pages_with_metadata)} pages.")
        return pages_with_metadata
    
    def _ultra_parallel_extract_pages(self, pdf_document: fitz.Document) -> List[Tuple[str, int]]:
        """
        Ultra-fast extraction for 500+ page documents using process-based parallelism.
        """
        import concurrent.futures
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        total_pages = len(pdf_document)
        self.logger.info(f"Using ultra-parallel extraction for {total_pages} pages")
        
        # Analyze document once
        extraction_stats = self._analyze_document(pdf_document)
        self.logger.info(f"Document analysis complete. Recommended method: {extraction_stats['recommended_method']}")
        
        # Get PDF content for process-safe processing
        pdf_content = pdf_document.tobytes()
        pdf_document.close()  # Close original to free memory
        
        pages_with_metadata = []
        
        # Use all available CPU cores
        max_workers = min(multiprocessing.cpu_count(), 8)  # Limit to 8 processes max
        batch_size = max(50, total_pages // max_workers)  # Larger batches for processes
        
        self.logger.info(f"Using {max_workers} processes with batch size {batch_size}")
        
        # Process in large batches using multiple processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                page_range = list(range(batch_start, batch_end))
                
                future = executor.submit(
                    extract_page_batch_worker,
                    pdf_content,
                    page_range,
                    extraction_stats
                )
                futures[future] = (batch_start, batch_end)
            
            # Collect results as they complete
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                batch_start, batch_end = futures[future]
                try:
                    batch_results = future.result(timeout=300)  # 5 minute timeout per batch
                    all_results.extend(batch_results)
                    self.logger.info(f"Completed batch {batch_start+1}-{batch_end} ({batch_end/total_pages*100:.1f}% complete)")
                except Exception as e:
                    self.logger.error(f"Failed to process batch {batch_start+1}-{batch_end}: {e}")
        
        # Sort all results by page number
        all_results.sort(key=lambda x: x[1])
        pages_with_metadata = all_results
        
        self.logger.info(f"Ultra-parallel extraction complete. Extracted text from {len(pages_with_metadata)} pages.")
        return pages_with_metadata
    
    def _extract_single_page_safe(self, pdf_content: bytes, page_num: int, extraction_stats: dict) -> str:
        """
        Safely extract a single page in a separate thread.
        Each thread opens its own PDF document to avoid threading issues.
        """
        try:
            # Open PDF document in this thread (PyMuPDF is not thread-safe for shared objects)
            thread_pdf = fitz.open(stream=pdf_content, filetype="pdf")
            page = thread_pdf[page_num]
            
            # Extract text using the recommended method
            page_text = self._extract_page_text_multimethod(page, page_num, extraction_stats)
            
            # Clean up
            thread_pdf.close()
            
            return page_text
            
        except Exception as e:
            self.logger.warning(f"Error extracting page {page_num + 1}: {e}")
            return ""
    
    def _sequential_extract_pages(self, pdf_document: fitz.Document) -> List[Tuple[str, int]]:
        """
        Sequential page extraction for smaller documents.
        """
        total_pages = len(pdf_document)
        self.logger.info(f"Using sequential extraction for {total_pages} pages")
        
        pages_with_metadata = []
        extraction_stats = self._analyze_document(pdf_document)
        self.logger.info(f"Document analysis complete. Recommended method: {extraction_stats['recommended_method']}")

        for page_num in range(total_pages):
            if page_num % 10 == 0:  # Log progress every 10 pages
                self.logger.info(f"Processing page {page_num + 1}/{total_pages} ({(page_num/total_pages)*100:.1f}% complete)")
            
            try:
                page = pdf_document[page_num]
                page_text = self._extract_page_text_multimethod(page, page_num, extraction_stats)
                if page_text.strip():
                    pages_with_metadata.append((page_text, page_num + 1))
            except Exception as e:
                self.logger.warning(f"Error extracting page {page_num + 1}: {e}")
        
        pdf_document.close()
        self.logger.info(f"Sequential extraction complete. Extracted text from {len(pages_with_metadata)} pages.")
        return pages_with_metadata

    def chunk_text(self, pages_with_metadata: List[Tuple[str, int]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Ultra-fast text chunking optimized for very large documents (20MB+).
        Uses different strategies based on document size with safety checks.
        """
        if not pages_with_metadata:
            return []

        total_pages = len(pages_with_metadata)
        total_text_size = sum(len(page_text) for page_text, _ in pages_with_metadata)
        
        self.logger.info(f"Processing {total_pages} pages, total size: {total_text_size/1024/1024:.1f}MB")
        
        # Safety check: Estimate expected chunk count
        expected_chunks = max(1, total_text_size // (chunk_size - overlap))
        max_allowed_chunks = expected_chunks * 3  # Allow 3x expected for safety
        
        self.logger.info(f"Expected chunks: ~{expected_chunks}, Max allowed: {max_allowed_chunks}")

        # Choose strategy based on document size AND page count
        if total_text_size > 20 * 1024 * 1024 or total_pages > 500:  # 20MB+ OR 500+ pages
            result = self._ultra_large_document_chunking(pages_with_metadata, chunk_size, overlap)
        elif total_text_size > 5 * 1024 * 1024 or total_pages > 100:  # 5MB+ OR 100+ pages
            result = self._large_document_chunking(pages_with_metadata, chunk_size, overlap)
        else:
            result = self._standard_fast_chunking(pages_with_metadata, chunk_size, overlap)
        
        # Safety check: If we got way too many chunks, something went wrong
        if len(result) > max_allowed_chunks:
            self.logger.error(f"🚨 CHUNKING ERROR: Generated {len(result)} chunks (expected ~{expected_chunks})")
            self.logger.error("This suggests an infinite loop bug. Using emergency chunking.")
            
            # Emergency: Use simple fixed-size chunking
            return self._emergency_simple_chunking(pages_with_metadata, chunk_size)
        
        return result

    def _ultra_large_document_chunking(self, pages_with_metadata: List[Tuple[str, int]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        Ultra-fast chunking for very large documents (500+ pages or 20MB+) using streaming approach.
        """
        total_pages = len(pages_with_metadata)
        self.logger.info(f"Using ultra-large document chunking strategy for {total_pages} pages")
        
        # Use even larger chunks and minimal overlap for ultra-large documents
        optimized_chunk_size = max(chunk_size, 4000)  # At least 4000 chars
        optimized_overlap = min(overlap, optimized_chunk_size // 8)  # Max 12.5% overlap
        
        self.logger.info(f"Ultra-optimized settings: chunk_size={optimized_chunk_size}, overlap={optimized_overlap}")
        
        all_chunks = []
        seen_hashes = set()
        chunk_counter = 0
        
        # Disable metadata extraction for ultra-large documents
        self.logger.info("Metadata extraction disabled for ultra-large documents to maximize speed")
        
        # Process each page as a stream
        for page_idx, (page_text, page_number) in enumerate(pages_with_metadata):
            if page_idx % 100 == 0:  # Log even less frequently
                self.logger.info(f"Processing page {page_idx + 1}/{total_pages}")
            
            # Stream process the page text in fixed blocks
            page_chunks = self._stream_process_text(page_text, optimized_chunk_size, optimized_overlap)
            
            for chunk_text in page_chunks:
                if len(chunk_text.strip()) > 150:  # Even higher threshold
                    # Use faster hash for deduplication
                    chunk_hash = hash(chunk_text) % (10**12)  # Much faster than SHA256
                    
                    if chunk_hash not in seen_hashes:
                        all_chunks.append({
                            "text": chunk_text.strip(),
                            "metadata": {
                                "page_number": page_number,
                                "chunk_index": chunk_counter,
                                "entities": [],
                                "concepts": [],
                                "categories": [],
                                "keywords": []
                            }
                        })
                        seen_hashes.add(chunk_hash)
                        chunk_counter += 1
                        
                        # Yield control periodically to prevent blocking
                        if chunk_counter % 1000 == 0:
                            self.logger.info(f"Processed {chunk_counter} chunks...")

        self.logger.info(f"Ultra-fast chunking complete: {len(all_chunks)} chunks created")
        return all_chunks

    def _stream_process_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Stream process text using optimized fixed-size blocks for maximum speed.
        """
        if not text:
            return []
        
        chunks = []
        text_len = len(text)
        
        # Use larger steps for very large texts with minimal overlap
        step_size = max(chunk_size - (overlap // 2), chunk_size // 2)  # Ensure meaningful progress
        
        # Simple fixed-size chunking - fastest possible
        for start in range(0, text_len, step_size):
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks

    def _large_document_chunking(self, pages_with_metadata: List[Tuple[str, int]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        Fast chunking for large documents (100+ pages or 5MB+) with optimized settings.
        """
        total_pages = len(pages_with_metadata)
        self.logger.info(f"Using large document chunking strategy for {total_pages} pages (no metadata extraction)")
        
        # Use larger chunks and less overlap for large documents
        optimized_chunk_size = max(chunk_size, 3000)  # At least 3000 chars
        optimized_overlap = min(overlap, optimized_chunk_size // 4)  # Max 25% overlap
        
        self.logger.info(f"Optimized settings: chunk_size={optimized_chunk_size}, overlap={optimized_overlap}")
        
        all_chunks = []
        seen_hashes = set()

        for page_idx, (page_text, page_number) in enumerate(pages_with_metadata):
            if page_idx % 50 == 0:  # Log less frequently for large docs
                self.logger.info(f"Processing page {page_idx + 1}/{total_pages}")
            
            # Use simplified chunking with minimal boundary detection
            page_chunks = self._simple_boundary_chunk(page_text, optimized_chunk_size, optimized_overlap)
            
            for chunk_text in page_chunks:
                if len(chunk_text.strip()) > 100:  # Higher threshold for large docs
                    # Use faster hash for large documents
                    chunk_hash = hash(chunk_text) % (10**12)
                    
                    if chunk_hash not in seen_hashes:
                        # No metadata extraction for large documents - just empty defaults
                        chunk_metadata = {
                            "page_number": page_number,
                            "entities": [],
                            "concepts": [],
                            "categories": [],
                            "keywords": []
                        }
                        
                        all_chunks.append({
                            "text": chunk_text.strip(),
                            "metadata": chunk_metadata
                        })
                        seen_hashes.add(chunk_hash)

        self.logger.info(f"Large document chunking complete: {len(all_chunks)} chunks created")
        return all_chunks

    def _simple_boundary_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Simple chunking with minimal boundary detection for speed.
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Only look for sentence boundaries if we're not at the end
            if end < text_len:
                # Quick search for period within last 10% of chunk
                search_start = start + int(chunk_size * 0.9)
                period_pos = text.find('. ', search_start, end)
                if period_pos > search_start:
                    end = period_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks

    def _standard_fast_chunking(self, pages_with_metadata: List[Tuple[str, int]], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        """
        Standard fast chunking for smaller documents (<5MB) with minimal metadata extraction.
        """
        self.logger.info("Using standard fast chunking strategy")
        
        all_chunks = []
        seen_hashes = set()
        total_pages = len(pages_with_metadata)

        # Only extract metadata for very small documents to avoid slowdown
        extract_metadata = (
            settings.ENABLE_METADATA_EXTRACTION and 
            total_pages <= settings.METADATA_EXTRACTION_PAGE_LIMIT
        )
        
        if extract_metadata:
            self.logger.info(f"Metadata extraction enabled for small document ({total_pages} pages)")
            try:
                from src.services.metadata_extraction_service import metadata_extraction_service
                metadata_extractor = metadata_extraction_service
            except ImportError:
                extract_metadata = False
                metadata_extractor = None
        else:
            self.logger.info(f"Metadata extraction disabled for performance ({total_pages} pages)")
            metadata_extractor = None

        chunk_counter = 0
        for page_idx, (page_text, page_number) in enumerate(pages_with_metadata):
            if page_idx % 10 == 0:
                self.logger.info(f"Processing page {page_idx + 1}/{total_pages}")
            
            # Use the fast chunking method
            page_chunks = self._fast_chunk_text(page_text, chunk_size, overlap)
            
            for chunk_text in page_chunks:
                if chunk_text and len(chunk_text.strip()) > 50:
                    chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                    if chunk_hash not in seen_hashes:
                        chunk_metadata = {"page_number": page_number}
                        
                        # Only extract metadata for first few chunks of small documents
                        if (extract_metadata and metadata_extractor and 
                            chunk_counter < settings.METADATA_EXTRACTION_CHUNK_LIMIT):
                            try:
                                advanced_metadata = metadata_extractor.extract_metadata_from_chunk(chunk_text)
                                chunk_metadata.update(advanced_metadata)
                            except Exception as e:
                                self.logger.debug(f"Metadata extraction failed: {e}")
                                # Set empty defaults
                                chunk_metadata.update({
                                    "entities": [],
                                    "concepts": [],
                                    "categories": [],
                                    "keywords": []
                                })
                        else:
                            # Set empty defaults for performance
                            chunk_metadata.update({
                                "entities": [],
                                "concepts": [],
                                "categories": [],
                                "keywords": []
                            })
                        
                        all_chunks.append({
                            "text": chunk_text.strip(),
                            "metadata": chunk_metadata
                        })
                        seen_hashes.add(chunk_hash)
                        chunk_counter += 1

        self.logger.info(f"Standard chunking complete: {len(all_chunks)} chunks created")
        return all_chunks

    def _fast_chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Ultra-fast text chunking using character-based approach with smart boundaries.
        Fixed to prevent infinite loops from excessive overlap.
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        # For very large texts, use word-based chunking (even faster)
        if len(text) > 50000:  # 50KB threshold
            return self._ultra_fast_word_chunk(text, chunk_size, overlap)
        
        chunks = []
        start = 0
        text_len = len(text)
        
        # Ensure overlap is not too large (max 50% of chunk_size)
        safe_overlap = min(overlap, chunk_size // 2)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # If not at the end of text, find a good boundary
            if end < text_len:
                # Look for sentence boundaries within the last 20% of chunk
                boundary_search_start = max(start + int(chunk_size * 0.8), start + chunk_size // 2)
                
                # Try to find sentence boundary (. ! ?)
                for boundary_char in ['. ', '! ', '? ']:
                    boundary_pos = text.rfind(boundary_char, boundary_search_start, end)
                    if boundary_pos > boundary_search_start:
                        end = boundary_pos + 1
                        break
                else:
                    # If no sentence boundary, look for paragraph break
                    paragraph_pos = text.rfind('\n\n', boundary_search_start, end)
                    if paragraph_pos > boundary_search_start:
                        end = paragraph_pos + 2
                    else:
                        # If no paragraph break, look for any newline
                        newline_pos = text.rfind('\n', boundary_search_start, end)
                        if newline_pos > boundary_search_start:
                            end = newline_pos + 1
                        else:
                            # Last resort: find word boundary
                            space_pos = text.rfind(' ', boundary_search_start, end)
                            if space_pos > boundary_search_start:
                                end = space_pos + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # FIXED: Ensure meaningful progress to prevent infinite loops
            next_start = end - safe_overlap
            if next_start <= start:
                # If overlap would cause no progress, advance by at least 25% of chunk_size
                next_start = start + (chunk_size // 4)
            
            start = next_start
            
            # Safety check: if we're not making progress, break
            if start >= text_len:
                break
        
        return chunks

    def _ultra_fast_word_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Ultra-fast word-based chunking for very large documents.
        ~50x faster than sentence-based chunking.
        """
        # Convert character-based sizes to approximate word counts
        avg_word_length = 5  # Average word length including spaces
        word_chunk_size = chunk_size // avg_word_length
        word_overlap = overlap // avg_word_length
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), word_chunk_size - word_overlap):
            chunk_words = words[i:i + word_chunk_size]
            if chunk_words:
                chunk_text = ' '.join(chunk_words)
                chunks.append(chunk_text)
        
        return chunks

    async def extract_and_chunk_pdf(self, pdf_content: bytes, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Extracts text from PDF and splits it into chunks with metadata in one operation.
        """
        pages_with_metadata = await self.extract_text_from_pdf_with_metadata(pdf_content)
        chunks = self.chunk_text(pages_with_metadata, chunk_size, overlap)
        return chunks

    def _analyze_document(self, pdf_document: fitz.Document) -> Dict[str, Any]:
        """
        Analyze document structure to determine the best extraction method.
        """
        stats = {
            "total_pages": len(pdf_document), 
            "has_images": False, 
            "has_tables": False, 
            "has_columns": False, 
            "text_density": 0.0, 
            "font_diversity": 0, 
            "recommended_method": "text"
        }
        
        if len(pdf_document) == 0:
            return stats
            
        # Sample up to 5 pages for analysis
        sample_pages = min(5, len(pdf_document))
        sample_indices = [i * len(pdf_document) // sample_pages for i in range(sample_pages)]
        
        total_chars = 0
        font_sizes = []
        
        for page_idx in sample_indices:
            try:
                page = pdf_document[page_idx]
                
                # Check for images
                if page.get_images():
                    stats["has_images"] = True
                
                # Get text blocks for analysis
                try:
                    blocks = page.get_text("dict")["blocks"]
                except Exception:
                    # Fallback if dict method fails
                    blocks = []
                
                text_blocks = [b for b in blocks if "lines" in b]
                
                # Check for columns
                if len(text_blocks) > 1:
                    x_positions = []
                    for block in text_blocks:
                        for line in block.get("lines", []):
                            bbox = line.get("bbox", [0, 0, 0, 0])
                            if len(bbox) >= 4:
                                x_positions.append(bbox[0])
                    
                    if len(set(x_positions)) > 2:
                        stats["has_columns"] = True
                
                # Check for tables
                if self._detect_tables(blocks):
                    stats["has_tables"] = True
                
                # Calculate text density
                try:
                    page_text = page.get_text()
                    total_chars += len(page_text)
                except Exception:
                    pass
                
                # Collect font sizes
                for block in text_blocks:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            size = span.get("size", 12)
                            if isinstance(size, (int, float)):
                                font_sizes.append(size)
                                
            except Exception as e:
                self.logger.warning(f"Error analyzing page {page_idx}: {e}")
                continue
        
        # Calculate final stats
        stats["text_density"] = total_chars / stats["total_pages"] if stats["total_pages"] > 0 else 0
        stats["font_diversity"] = len(set(font_sizes)) if font_sizes else 0
        
        # Determine recommended method
        if stats["has_tables"]:
            stats["recommended_method"] = "dict"
        elif stats["has_columns"]:
            stats["recommended_method"] = "blocks"
        elif stats["has_images"] or stats["text_density"] < 100:
            stats["recommended_method"] = "html"
        else:
            stats["recommended_method"] = "text"
        
        return stats

    def _detect_tables(self, blocks: List[Dict]) -> bool:
        """
        Detect if the page contains table-like structures.
        """
        try:
            text_blocks = [b for b in blocks if "lines" in b]
            if len(text_blocks) < 3:
                return False
            
            y_positions = []
            for block in text_blocks:
                for line in block.get("lines", []):
                    bbox = line.get("bbox", [0, 0, 0, 0])
                    if len(bbox) >= 4:
                        y_positions.append(bbox[1])
            
            if len(y_positions) < 3:
                return False
            
            y_positions.sort()
            differences = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            
            if len(differences) < 2:
                return False
            
            avg_diff = statistics.mean(differences)
            if avg_diff == 0:
                return False
                
            similar_diffs = sum(1 for d in differences if abs(d - avg_diff) < avg_diff * 0.3)
            return similar_diffs > len(differences) * 0.4
            
        except Exception as e:
            self.logger.warning(f"Error in table detection: {e}")
            return False

    def _extract_page_text_multimethod(self, page: fitz.Page, page_num: int, stats: Dict) -> str:
        """
        Extract text from a page using multiple methods and select the best result.
        """
        primary_method = stats.get("recommended_method", "text")
        
        try:
            text = self._extract_by_method(page, primary_method)
        except Exception as e:
            self.logger.warning(f"Primary extraction method '{primary_method}' failed on page {page_num}: {e}")
            text = ""
        
        # If primary method gives poor results, try alternatives
        if len(text.strip()) < 50 or self._is_poor_extraction(text):
            best_text = text
            best_score = self._score_extraction(text)
            
            for method in self.extraction_methods:
                if method != primary_method:
                    try:
                        alt_text = self._extract_by_method(page, method)
                        alt_score = self._score_extraction(alt_text)
                        if alt_score > best_score:
                            best_text = alt_text
                            best_score = alt_score
                    except Exception as e:
                        self.logger.debug(f"Alternative method '{method}' failed on page {page_num}: {e}")
                        continue
            
            text = best_text
        
        return self._clean_and_enhance_text(text, stats)

    def _extract_by_method(self, page: fitz.Page, method: str) -> str:
        """
        Extract text using a specific method.
        """
        try:
            if method == "text":
                return page.get_text("text")
            elif method == "html":
                html_content = page.get_text("html")
                return self._html_to_text(html_content)
            elif method == "dict":
                dict_content = page.get_text("dict")
                return self._dict_to_text(dict_content)
            elif method == "blocks":
                blocks_content = page.get_text("blocks")
                return self._blocks_to_text(blocks_content)
            else:
                # Fallback to text method
                return page.get_text("text")
        except Exception as e:
            self.logger.warning(f"Extraction method '{method}' failed: {e}")
            # Fallback to basic text extraction
            try:
                return page.get_text("text")
            except Exception as fallback_error:
                self.logger.error(f"All extraction methods failed: {fallback_error}")
                return ""

    def _html_to_text(self, html: str) -> str:
        """
        Convert HTML content to plain text.
        """
        if not html:
            return ""
            
        try:
            # Replace HTML line breaks and paragraphs with newlines
            text = re.sub(r'<br[^>]*>', '\n', html)
            text = re.sub(r'</p>', '\n\n', text)
            
            # Remove all HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Decode HTML entities
            text = text.replace('&nbsp;', ' ')
            text = text.replace('&amp;', '&')
            text = text.replace('&lt;', '<')
            text = text.replace('&gt;', '>')
            text = text.replace('&quot;', '"')
            text = text.replace('&#39;', "'")
            
            return text
        except Exception as e:
            self.logger.warning(f"HTML to text conversion failed: {e}")
            return html  # Return original if conversion fails

    def _dict_to_text(self, text_dict: Dict) -> str:
        """
        Convert dictionary format text to plain text.
        """
        if not text_dict or not isinstance(text_dict, dict):
            return ""
            
        try:
            text_lines = []
            blocks = text_dict.get("blocks", [])
            
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                    
                if "lines" in block:
                    for line in block["lines"]:
                        if not isinstance(line, dict):
                            continue
                            
                        spans = line.get("spans", [])
                        line_text = ""
                        
                        for span in spans:
                            if isinstance(span, dict) and "text" in span:
                                line_text += span["text"]
                        
                        if line_text.strip():
                            text_lines.append(line_text.strip())
                    
                    # Add spacing between blocks
                    if text_lines and text_lines[-1]:
                        text_lines.append("")
            
            return "\n".join(text_lines)
            
        except Exception as e:
            self.logger.warning(f"Dict to text conversion failed: {e}")
            return ""

    def _blocks_to_text(self, blocks) -> str:
        """
        Convert blocks format text to plain text.
        """
        if not blocks:
            return ""
            
        try:
            text_blocks = []
            
            for block in blocks:
                # Blocks format: (x0, y0, x1, y1, "text", block_no, block_type)
                if isinstance(block, (list, tuple)) and len(block) >= 5:
                    # Extract text from position 4 (0-indexed)
                    text_content = block[4]
                    if isinstance(text_content, str) and text_content.strip():
                        # Extract bounding box for sorting
                        bbox = block[:4] if len(block) >= 4 else [0, 0, 0, 0]
                        text_blocks.append((bbox, text_content.strip()))
            
            # Sort by y-coordinate (top to bottom), then x-coordinate (left to right)
            text_blocks.sort(key=lambda x: (x[0][1], x[0][0]))
            
            # Extract just the text
            return "\n".join([text for _, text in text_blocks])
            
        except Exception as e:
            self.logger.warning(f"Blocks to text conversion failed: {e}")
            return ""

    def _score_extraction(self, text: str) -> float:
        if not text.strip():
            return 0.0
        score = min(len(text) / 1000, 10.0)
        words = text.split()
        score += min(len(words) / 100, 5.0)
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]{}')
        special_ratio = special_chars / len(text) if text else 0
        score -= special_ratio * 5
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 5 <= avg_sentence_length <= 30:
            score += 2.0
        return max(score, 0.0)

    def _is_poor_extraction(self, text: str) -> bool:
        if not text.strip():
            return True
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]{}')
        if special_chars / len(text) > 0.3:
            return True
        words = text.split()
        if not words:
            return True
        short_words = sum(1 for w in words if len(w) == 1 and w.isalpha())
        if short_words / len(words) > 0.5:
            return True
        return False

    def _clean_and_enhance_text(self, text: str, stats: Dict) -> str:
        """
        Clean and enhance extracted text.
        """
        if not text:
            return ""
        
        try:
            # Normalize Unicode characters
            text = unicodedata.normalize('NFKC', text)
            
            # Apply special character mapping from settings
            text = self._apply_special_char_mapping(text)
            
            # Basic text cleaning
            text = self._basic_clean_text(text)
            
            # Table-specific cleaning if tables detected
            if stats.get("has_tables", False):
                text = self._clean_table_text(text)
            
            # Remove artifacts and fix formatting
            text = self._remove_artifacts(text)
            text = self._fix_word_breaks(text)
            text = self._normalize_spacing(text)
            
            return text.strip()
            
        except Exception as e:
            self.logger.warning(f"Text cleaning failed: {e}")
            return text.strip() if text else ""

    def _basic_clean_text(self, text: str) -> str:
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n ', '\n', text)
        text = re.sub(r' \n', '\n', text)
        text = re.sub(r'\x0c', '', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text

    def _clean_table_text(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = [re.sub(r'\s{3,}', '\t', line) if re.search(r'\s{3,}', line) else line for line in lines]
        return '\n'.join(cleaned_lines)

    def _remove_artifacts(self, text: str) -> str:
        """
        Remove common PDF extraction artifacts.
        """
        if not text:
            return ""
            
        try:
            # Remove lines with only dashes, underscores, or spaces
            text = re.sub(r'^[-_\s]{3,}$', '', text, flags=re.MULTILINE)
            
            # Remove standalone page numbers
            text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
            
            # Remove lines with only special characters
            text = re.sub(r'^[^\w\s]*$', '', text, flags=re.MULTILINE)
            
            # Remove excessive repetition of characters
            text = re.sub(r'(.)\1{5,}', r'\1\1', text)
            
            return text
            
        except Exception as e:
            self.logger.warning(f"Artifact removal failed: {e}")
            return text

    def _fix_word_breaks(self, text: str) -> str:
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        text = re.sub(r'(\w+)\n([a-z]+)', r'\1\2', text)
        return text

    def _normalize_spacing(self, text: str) -> str:
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        text = re.sub(r',([A-Za-z])', r', \1', text)
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text

    def _apply_special_char_mapping(self, text: str) -> str:
        """
        Apply configurable special character mapping to text.
        """
        if not settings.SPECIAL_CHAR_MAPPING:
            return text
            
        for special_char, replacement in settings.SPECIAL_CHAR_MAPPING.items():
            text = text.replace(special_char, replacement)
        return text

    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences using SpaCy if available,
        falling back to advanced regex-based splitting optimized for business documents.
        """
        if not text or not text.strip():
            return []
            
        # Try SpaCy first if available and model is loaded
        if SPACY_AVAILABLE and SPACY_MODEL_AVAILABLE and self._spacy_nlp is not None:
            try:
                doc = self._spacy_nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                if sentences:  # Only return if we got results
                    return sentences
            except Exception as e:
                self.logger.debug(f"SpaCy sentence tokenization failed: {e}")

        # Advanced regex-based sentence splitting optimized for business documents
        try:
            sentences = self._advanced_regex_sentence_split(text)
            if sentences:  # Only return if we got results
                return sentences
        except Exception as e:
            self.logger.debug(f"Advanced regex sentence tokenization failed: {e}")

        # Simple fallback to basic regex
        try:
            # Basic regex pattern for sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            self.logger.warning(f"Basic regex sentence tokenization failed: {e}")
            # Ultimate fallback: split by periods
            return [s.strip() for s in text.split('.') if s.strip()]

    def _advanced_regex_sentence_split(self, text: str) -> List[str]:
        """
        Advanced regex-based sentence splitting optimized for business documents.
        Handles common abbreviations, numbers, and business-specific patterns.
        """
        if not text:
            return []
        
        # Step 1: Protect common abbreviations and patterns that shouldn't be split
        protected_patterns = {
            # Common business abbreviations
            r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|Inc|Corp|LLC|Ltd|Co|vs|etc|i\.e|e\.g|cf|viz|al|et)\.\s*': 'ABBREV_{}',
            # Decimal numbers and percentages
            r'\b\d+\.\d+(?:%|\s*percent)?\b': 'DECIMAL_{}',
            # Currency amounts
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b': 'CURRENCY_{}',
            # Dates (MM.DD.YYYY, DD.MM.YYYY)
            r'\b\d{1,2}\.\d{1,2}\.\d{2,4}\b': 'DATE_{}',
            # Section numbers (1.1, 2.3.4, etc.)
            r'\b\d+(?:\.\d+){1,3}\s+[A-Z]': 'SECTION_{}',
            # Legal citations (e.g., "v. Smith", "Art. 5")
            r'\b(?:v|vs|Art|Sec|Para|Ch|Chap)\.\s*\d*\s*[A-Z]': 'LEGAL_{}',
            # Time formats (12.30 PM, 3.45 AM)
            r'\b\d{1,2}\.\d{2}\s*(?:AM|PM|am|pm)\b': 'TIME_{}',
            # Initials (J.K. Smith, A.B.C. Corp)
            r'\b[A-Z]\.[A-Z]\.(?:\s*[A-Z]\.)*\s*[A-Z][a-z]+': 'INITIALS_{}',
            # URLs and emails (basic protection)
            r'\b\w+\.\w+(?:\.\w+)*(?:@|\s*\.\s*(?:com|org|net|edu|gov))\b': 'URL_{}',
        }
        
        # Apply protections
        protected_text = text
        replacements = {}
        counter = 0
        
        for pattern, placeholder_template in protected_patterns.items():
            matches = re.finditer(pattern, protected_text, re.IGNORECASE)
            for match in reversed(list(matches)):  # Reverse to maintain positions
                placeholder = placeholder_template.format(counter)
                replacements[placeholder] = match.group()
                protected_text = protected_text[:match.start()] + placeholder + protected_text[match.end():]
                counter += 1
        
        # Step 2: Apply sophisticated sentence boundary detection
        sentence_patterns = [
            # Standard sentence endings followed by capital letter or quote
            r'(?<=[.!?])\s+(?=[A-Z]["\']?[A-Z]?)',
            # Sentence ending with quote followed by capital
            r'(?<=[.!?]["\'"])\s+(?=[A-Z])',
            # Handle numbered lists and bullet points
            r'(?<=[.!?])\s+(?=\d+\.\s+[A-Z])',
            r'(?<=[.!?])\s+(?=[•·▪▫-]\s*[A-Z])',
            # Handle paragraph breaks (double newlines)
            r'\n\s*\n\s*(?=[A-Z])',
            # Handle section headers (all caps or title case after newline)
            r'(?<=\n)\s*(?=[A-Z][A-Z\s]{5,}:?\s*\n)',
            r'(?<=\n)\s*(?=[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?\s*\n)',
        ]
        
        # Apply sentence splitting
        sentences = [protected_text]
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                parts = re.split(pattern, sentence)
                new_sentences.extend(parts)
            sentences = new_sentences
        
        # Step 3: Clean up and restore protected patterns
        final_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Restore protected patterns
            restored_sentence = sentence
            for placeholder, original in replacements.items():
                restored_sentence = restored_sentence.replace(placeholder, original)
            
            # Clean up the sentence
            cleaned_sentence = restored_sentence.strip()
            
            # Filter out very short sentences (likely artifacts)
            if len(cleaned_sentence) > 10:  # Minimum sentence length
                final_sentences.append(cleaned_sentence)
        
        # Step 4: Post-processing for business document patterns
        final_sentences = self._post_process_business_sentences(final_sentences)
        
        return final_sentences
    
    def _post_process_business_sentences(self, sentences: List[str]) -> List[str]:
        """
        Post-process sentences to handle business document specific patterns.
        """
        processed_sentences = []
        
        for sentence in sentences:
            # Skip sentences that are likely headers or artifacts
            if self._is_likely_header_or_artifact(sentence):
                continue
            
            # Handle incomplete sentences that should be merged
            if processed_sentences and self._should_merge_with_previous(sentence, processed_sentences[-1]):
                processed_sentences[-1] = processed_sentences[-1] + " " + sentence
            else:
                processed_sentences.append(sentence)
        
        return processed_sentences
    
    def _is_likely_header_or_artifact(self, sentence: str) -> bool:
        """
        Check if a sentence is likely a header, page number, or other artifact.
        """
        sentence = sentence.strip()
        
        # Very short sentences
        if len(sentence) < 15:
            return True
        
        # All caps (likely headers)
        if sentence.isupper() and len(sentence) > 50:
            return True
        
        # Only numbers and basic punctuation
        if re.match(r'^[\d\s\-\.]+$', sentence):
            return True
        
        # Common header patterns
        header_patterns = [
            r'^(?:SECTION|CHAPTER|PART|ARTICLE)\s+[IVXLC\d]+',
            r'^(?:TABLE|SCHEDULE|APPENDIX|ANNEXURE)\s+[A-Z\d]',
            r'^Page\s+\d+',
            r'^\d+\s*$',  # Just a number
        ]
        
        for pattern in header_patterns:
            if re.match(pattern, sentence, re.IGNORECASE):
                return True
        
        return False
    
    def _should_merge_with_previous(self, current: str, previous: str) -> bool:
        """
        Check if current sentence should be merged with the previous one.
        """
        current = current.strip()
        previous = previous.strip()
        
        # If current sentence starts with lowercase (likely continuation)
        if current and current[0].islower():
            return True
        
        # If previous sentence doesn't end with proper punctuation
        if previous and previous[-1] not in '.!?':
            return True
        
        # If current sentence starts with coordinating conjunction
        if re.match(r'^(?:and|but|or|nor|for|so|yet)\s+', current, re.IGNORECASE):
            return True
        
        # If current sentence is very short and looks like a continuation
        if len(current) < 30 and not re.match(r'^[A-Z][a-z]+', current):
            return True
        
        return False

    def _emergency_simple_chunking(self, pages_with_metadata: List[Tuple[str, int]], chunk_size: int) -> List[Dict[str, Any]]:
        """
        Emergency simple chunking when the main algorithm fails.
        Uses fixed-size chunks with no overlap to prevent infinite loops.
        """
        self.logger.info("Using emergency simple chunking (no overlap)")
        
        all_chunks = []
        chunk_counter = 0
        
        for page_text, page_number in pages_with_metadata:
            # Simple fixed-size chunking with no overlap
            for start in range(0, len(page_text), chunk_size):
                end = min(start + chunk_size, len(page_text))
                chunk_text = page_text[start:end].strip()
                
                if chunk_text and len(chunk_text) > 50:
                    all_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "page_number": page_number,
                            "chunk_index": chunk_counter,
                            "entities": [],
                            "concepts": [],
                            "categories": [],
                            "keywords": []
                        }
                    })
                    chunk_counter += 1
        
        self.logger.info(f"Emergency chunking complete: {len(all_chunks)} chunks created")
        return all_chunks

    def _ultra_parallel_extract_pages_limited(self, pdf_document: fitz.Document, max_pages: int) -> List[Tuple[str, int]]:
        """
        Ultra-fast extraction for limited pages using simplified processing.
        """
        import concurrent.futures
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        self.logger.info(f"Using ultra-parallel extraction for first {max_pages} pages")
        
        # Skip document analysis in fast mode for speed
        if settings.ENABLE_FAST_MODE and settings.FAST_MODE_SKIP_ANALYSIS:
            extraction_stats = {"recommended_method": "text"}  # Use fastest method
            self.logger.info("Fast mode: Skipping document analysis, using text extraction")
        else:
            extraction_stats = self._analyze_document(pdf_document)
            self.logger.info(f"Document analysis complete. Recommended method: {extraction_stats['recommended_method']}")
        
        # Get PDF content for process-safe processing
        pdf_content = pdf_document.tobytes()
        pdf_document.close()  # Close original to free memory
        
        pages_with_metadata = []
        
        # Use all available CPU cores but limit batch size for speed
        max_workers = min(multiprocessing.cpu_count(), 8)
        batch_size = max(20, max_pages // max_workers)  # Smaller batches for faster startup
        
        self.logger.info(f"Using {max_workers} processes with batch size {batch_size}")
        
        # Process in batches using multiple processes
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            for batch_start in range(0, max_pages, batch_size):
                batch_end = min(batch_start + batch_size, max_pages)
                page_range = list(range(batch_start, batch_end))
                
                future = executor.submit(
                    extract_page_batch_worker,
                    pdf_content,
                    page_range,
                    extraction_stats
                )
                futures[future] = (batch_start, batch_end)
            
            # Collect results as they complete
            all_results = []
            for future in concurrent.futures.as_completed(futures):
                batch_start, batch_end = futures[future]
                try:
                    batch_results = future.result(timeout=60)  # Shorter timeout for fast mode
                    all_results.extend(batch_results)
                    self.logger.info(f"Completed batch {batch_start+1}-{batch_end} ({batch_end/max_pages*100:.1f}% complete)")
                except Exception as e:
                    self.logger.error(f"Failed to process batch {batch_start+1}-{batch_end}: {e}")
        
        # Sort all results by page number
        all_results.sort(key=lambda x: x[1])
        pages_with_metadata = all_results
        
        self.logger.info(f"Ultra-parallel extraction complete. Extracted text from {len(pages_with_metadata)} pages.")
        return pages_with_metadata
    
    def _parallel_extract_pages_limited(self, pdf_document: fitz.Document, max_pages: int) -> List[Tuple[str, int]]:
        """
        Parallel extraction for limited pages using threading.
        """
        import concurrent.futures
        from concurrent.futures import ThreadPoolExecutor
        
        self.logger.info(f"Using parallel extraction for first {max_pages} pages")
        
        # Skip analysis in fast mode
        if settings.ENABLE_FAST_MODE and settings.FAST_MODE_SKIP_ANALYSIS:
            extraction_stats = {"recommended_method": "text"}
            self.logger.info("Fast mode: Skipping document analysis, using text extraction")
        else:
            extraction_stats = self._analyze_document(pdf_document)
            self.logger.info(f"Document analysis complete. Recommended method: {extraction_stats['recommended_method']}")
        
        # Get PDF content for thread-safe processing
        pdf_content = pdf_document.tobytes()
        pdf_document.close()
        
        pages_with_metadata = []
        
        # Process pages in smaller batches for faster response
        batch_size = min(25, max(10, max_pages // 4))
        max_workers = min(8, batch_size)
        
        for batch_start in range(0, max_pages, batch_size):
            batch_end = min(batch_start + batch_size, max_pages)
            self.logger.info(f"Processing batch {batch_start+1}-{batch_end} ({batch_end/max_pages*100:.1f}% complete)")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for page_num in range(batch_start, batch_end):
                    future = executor.submit(
                        self._extract_single_page_safe, 
                        pdf_content, 
                        page_num, 
                        extraction_stats
                    )
                    futures[future] = page_num
                
                batch_results = []
                for future in concurrent.futures.as_completed(futures):
                    page_num = futures[future]
                    try:
                        page_text = future.result(timeout=15)  # Shorter timeout
                        if page_text and page_text.strip():
                            batch_results.append((page_text, page_num + 1))
                    except Exception as e:
                        self.logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                
                batch_results.sort(key=lambda x: x[1])
                pages_with_metadata.extend(batch_results)
        
        self.logger.info(f"Parallel extraction complete. Extracted text from {len(pages_with_metadata)} pages.")
        return pages_with_metadata
    
    def _sequential_extract_pages_limited(self, pdf_document: fitz.Document, max_pages: int) -> List[Tuple[str, int]]:
        """
        Sequential extraction for limited pages.
        """
        self.logger.info(f"Using sequential extraction for first {max_pages} pages")
        
        pages_with_metadata = []
        
        # Skip analysis in fast mode
        if settings.ENABLE_FAST_MODE and settings.FAST_MODE_SKIP_ANALYSIS:
            extraction_stats = {"recommended_method": "text"}
            self.logger.info("Fast mode: Skipping document analysis, using text extraction")
        else:
            extraction_stats = self._analyze_document(pdf_document)
            self.logger.info(f"Document analysis complete. Recommended method: {extraction_stats['recommended_method']}")

        for page_num in range(max_pages):
            if page_num % 20 == 0:  # Log progress every 20 pages
                self.logger.info(f"Processing page {page_num + 1}/{max_pages} ({(page_num/max_pages)*100:.1f}% complete)")
            
            try:
                page = pdf_document[page_num]
                page_text = self._extract_page_text_multimethod(page, page_num, extraction_stats)
                if page_text.strip():
                    pages_with_metadata.append((page_text, page_num + 1))
            except Exception as e:
                self.logger.warning(f"Error extracting page {page_num + 1}: {e}")
        
        pdf_document.close()
        self.logger.info(f"Sequential extraction complete. Extracted text from {len(pages_with_metadata)} pages.")
        return pages_with_metadata

# Global instance
text_extraction_service = TextExtractionService()