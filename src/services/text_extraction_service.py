import logging
import asyncio
import fitz  # PyMuPDF
from typing import List, Tuple, Optional, Dict, Any
import re
from io import BytesIO
import statistics

logger = logging.getLogger(__name__)

class TextExtractionService:
    """
    Advanced PDF text extraction service with multiple extraction strategies.
    Optimized to extract maximum text from various PDF types including:
    - Text-based PDFs
    - Mixed content (text + images)
    - Complex layouts (tables, columns)
    - Scanned documents (with OCR fallback)
    """
    
    def __init__(self):
        self.logger = logger
        # Text extraction strategies in order of preference
        self.extraction_methods = [
            "dict",    # Structured text with positions (best for tables)
            "text",    # Plain text (fastest)
            "html",    # HTML structure (good for layouts)
            "blocks",  # Text blocks (good for columns)
        ]
    
    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Advanced PDF text extraction using multiple strategies.
        Automatically selects the best extraction method based on content analysis.
        """
        self.logger.info("Starting advanced PDF text extraction")
        
        try:
            # Run PDF processing in thread pool to avoid blocking
            text = await asyncio.to_thread(self._extract_text_advanced, pdf_content)
            self.logger.info(f"Extracted {len(text)} characters from PDF using optimized extraction")
            return text
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
            raise
    
    def _extract_text_advanced(self, pdf_content: bytes) -> str:
        """
        Advanced multi-strategy text extraction with automatic method selection.
        """
        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            all_text = []
            
            # Analyze document to choose best extraction strategy
            extraction_stats = self._analyze_document(pdf_document)
            self.logger.info(f"Document analysis: {extraction_stats}")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Try multiple extraction methods and choose the best result
                page_text = self._extract_page_text_multimethod(page, page_num, extraction_stats)
                
                if page_text.strip():
                    all_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            pdf_document.close()
            
            # Join all pages
            full_text = "\n\n".join(all_text)
            return full_text
            
        except Exception as e:
            self.logger.error(f"Error in advanced PDF processing: {e}")
            raise
    
    def _analyze_document(self, pdf_document: fitz.Document) -> Dict[str, Any]:
        """
        Analyze document structure to determine optimal extraction strategy.
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
        
        # Sample a few pages for analysis (max 5 pages)
        sample_pages = min(5, len(pdf_document))
        sample_indices = [i * len(pdf_document) // sample_pages for i in range(sample_pages)]
        
        total_chars = 0
        font_sizes = []
        
        for page_idx in sample_indices:
            page = pdf_document[page_idx]
            
            # Check for images
            if page.get_images():
                stats["has_images"] = True
            
            # Analyze text blocks for layout complexity
            blocks = page.get_text("dict")["blocks"]
            text_blocks = [b for b in blocks if "lines" in b]
            
            if len(text_blocks) > 1:
                # Check for column layout
                x_positions = []
                for block in text_blocks:
                    for line in block["lines"]:
                        x_positions.append(line["bbox"][0])  # left x position
                
                if len(set(x_positions)) > 2:  # Multiple column positions
                    stats["has_columns"] = True
            
            # Check for table structures (regular spacing patterns)
            if self._detect_tables(blocks):
                stats["has_tables"] = True
            
            # Calculate text density and font diversity
            page_text = page.get_text()
            total_chars += len(page_text)
            
            # Get font information
            for block in text_blocks:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
        
        stats["text_density"] = total_chars / stats["total_pages"]
        stats["font_diversity"] = len(set(font_sizes)) if font_sizes else 0
        
        # Choose optimal extraction method
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
        Detect if the page contains table structures.
        """
        text_blocks = [b for b in blocks if "lines" in b]
        if len(text_blocks) < 3:
            return False
        
        # Look for regular vertical alignment patterns
        y_positions = []
        for block in text_blocks:
            for line in block["lines"]:
                y_positions.append(line["bbox"][1])  # top y position
        
        if len(y_positions) < 3:
            return False
        
        # Check for regular spacing (tables often have consistent row heights)
        y_positions.sort()
        differences = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        
        if len(differences) < 2:
            return False
        
        # If many differences are similar, likely a table
        avg_diff = statistics.mean(differences)
        similar_diffs = sum(1 for d in differences if abs(d - avg_diff) < avg_diff * 0.3)
        
        return similar_diffs > len(differences) * 0.4
    
    def _extract_page_text_multimethod(self, page: fitz.Page, page_num: int, stats: Dict) -> str:
        """
        Extract text from a page using the most appropriate method.
        Falls back to other methods if the primary method fails.
        """
        primary_method = stats["recommended_method"]
        
        # Try primary method first
        text = self._extract_by_method(page, primary_method)
        
        # If primary method yields poor results, try alternatives
        if len(text.strip()) < 50 or self._is_poor_extraction(text):
            self.logger.debug(f"Page {page_num + 1}: Primary method '{primary_method}' yielded poor results, trying alternatives")
            
            best_text = text
            best_score = self._score_extraction(text)
            
            # Try all other methods
            for method in self.extraction_methods:
                if method != primary_method:
                    try:
                        alt_text = self._extract_by_method(page, method)
                        alt_score = self._score_extraction(alt_text)
                        
                        if alt_score > best_score:
                            best_text = alt_text
                            best_score = alt_score
                            self.logger.debug(f"Page {page_num + 1}: Method '{method}' produced better results (score: {alt_score:.2f})")
                    except Exception as e:
                        self.logger.debug(f"Page {page_num + 1}: Method '{method}' failed: {e}")
            
            # If all methods still yield poor results, try OCR fallback on images
            if self._score_extraction(best_text) < 2.0 and stats.get("has_images"):
                self.logger.debug(f"Page {page_num + 1}: Attempting OCR fallback for image-based content")
                ocr_text = self._try_ocr_fallback(page)
                if self._score_extraction(ocr_text) > self._score_extraction(best_text):
                    best_text = ocr_text
                    self.logger.debug(f"Page {page_num + 1}: OCR fallback produced better results")
            
            text = best_text
        
        # Clean and enhance the extracted text
        cleaned_text = self._clean_and_enhance_text(text, stats)
        return cleaned_text
    
    def _extract_by_method(self, page: fitz.Page, method: str) -> str:
        """
        Extract text using a specific method.
        """
        if method == "text":
            return page.get_text("text")
        elif method == "html":
            html = page.get_text("html")
            return self._html_to_text(html)
        elif method == "dict":
            return self._dict_to_text(page.get_text("dict"))
        elif method == "blocks":
            return self._blocks_to_text(page.get_text("blocks"))
        else:
            return page.get_text("text")  # fallback
    
    def _html_to_text(self, html: str) -> str:
        """
        Convert HTML text to clean text while preserving structure.
        """
        # Remove HTML tags but preserve line breaks
        text = re.sub(r'<br[^>]*>', '\n', html)
        text = re.sub(r'</p>', '\n\n', text)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        return text
    
    def _dict_to_text(self, text_dict: Dict) -> str:
        """
        Convert dictionary format to text, preserving table structure.
        """
        text_lines = []
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
            
            block_lines = []
            for line in block["lines"]:
                line_text = ""
                for span in line["spans"]:
                    line_text += span["text"]
                if line_text.strip():
                    block_lines.append(line_text.strip())
            
            if block_lines:
                text_lines.extend(block_lines)
                text_lines.append("")  # Add spacing between blocks
        
        return "\n".join(text_lines)
    
    def _blocks_to_text(self, blocks) -> str:
        """
        Convert blocks format to text, handling column layouts.
        """
        # Sort blocks by position (top to bottom, left to right)
        text_blocks = []
        for block in blocks:
            if len(block) >= 5 and isinstance(block[4], str):  # Text block
                bbox = block[:4]
                text = block[4]
                if text.strip():
                    text_blocks.append((bbox, text.strip()))
        
        # Sort by y-position first (top to bottom), then x-position (left to right)
        text_blocks.sort(key=lambda x: (x[0][1], x[0][0]))
        
        return "\n".join([text for _, text in text_blocks])
    
    def _score_extraction(self, text: str) -> float:
        """
        Score the quality of extracted text.
        """
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # Length bonus (more text is generally better)
        score += min(len(text) / 1000, 10.0)
        
        # Word count bonus
        words = text.split()
        score += min(len(words) / 100, 5.0)
        
        # Penalize too many special characters (might indicate extraction errors)
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]{}')
        special_ratio = special_chars / len(text) if text else 0
        score -= special_ratio * 5
        
        # Bonus for proper sentence structure
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 5 <= avg_sentence_length <= 30:  # Reasonable sentence length
            score += 2.0
        
        return max(score, 0.0)
    
    def _is_poor_extraction(self, text: str) -> bool:
        """
        Determine if extracted text is of poor quality.
        """
        if not text.strip():
            return True
        
        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]{}')
        if special_chars / len(text) > 0.3:
            return True
        
        # Check for very short "words" (might be extraction artifacts)
        words = text.split()
        short_words = sum(1 for w in words if len(w) == 1 and w.isalpha())
        if len(words) > 0 and short_words / len(words) > 0.5:
            return True
        
        return False
    
    def _clean_and_enhance_text(self, text: str, stats: Dict) -> str:
        """
        Enhanced text cleaning with document-aware optimizations.
        """
        if not text:
            return ""
        
        # Advanced text cleaning
        text = self._advanced_clean_text(text, stats)
        
        return text
    
    def _advanced_clean_text(self, text: str, stats: Dict) -> str:
        """
        Advanced text cleaning with document-aware optimizations.
        """
        if not text:
            return ""
        
        # Step 1: Basic cleaning
        text = self._basic_clean_text(text)
        
        # Step 2: Document-specific cleaning based on analysis
        if stats.get("has_tables"):
            text = self._clean_table_text(text)
        
        if stats.get("has_columns"):
            text = self._clean_column_text(text)
        
        # Step 3: Advanced cleanup
        text = self._remove_artifacts(text)
        text = self._fix_word_breaks(text)
        text = self._normalize_spacing(text)
        
        return text.strip()
    
    def _basic_clean_text(self, text: str) -> str:
        """
        Basic text cleaning operations.
        """
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n ', '\n', text)  # Remove spaces after newlines
        text = re.sub(r' \n', '\n', text)  # Remove spaces before newlines
        
        # Remove common PDF artifacts
        text = re.sub(r'\x0c', '', text)  # Form feed characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)  # Control characters
        
        return text
    
    def _clean_table_text(self, text: str) -> str:
        """
        Clean text that contains table structures.
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preserve table-like structures with multiple spaces
            if re.search(r'\s{3,}', line):  # Multiple spaces might indicate columns
                # Replace multiple spaces with tab for better structure
                line = re.sub(r'\s{3,}', '\t', line)
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _clean_column_text(self, text: str) -> str:
        """
        Clean text from multi-column layouts.
        """
        # Already handled by blocks extraction method
        return text
    
    def _remove_artifacts(self, text: str) -> str:
        """
        Remove common PDF extraction artifacts.
        """
        # Remove repeated dashes or underscores (often from tables/forms)
        text = re.sub(r'^[-_\s]{3,}$', '', text, flags=re.MULTILINE)
        
        # Remove standalone numbers that might be page numbers or artifacts
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
        
        # Remove very short lines with only punctuation
        text = re.sub(r'^[^\w\s]*$', '', text, flags=re.MULTILINE)
        
        # Remove repeated characters (artifacts from scanning/conversion)
        text = re.sub(r'(.)\1{5,}', r'\1\1', text)  # Reduce long repetitions
        
        return text
    
    def _fix_word_breaks(self, text: str) -> str:
        """
        Fix hyphenated words that were broken across lines.
        """
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        
        # Fix words broken without hyphens (common in PDFs)
        text = re.sub(r'(\w+)\n([a-z]+)', r'\1\2', text)
        
        return text
    
    def _normalize_spacing(self, text: str) -> str:
        """
        Normalize spacing throughout the text.
        """
        # Ensure proper spacing after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Fix missing spaces after commas
        text = re.sub(r',([A-Za-z])', r', \1', text)
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Normalize line breaks
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """
        Legacy clean method for backward compatibility.
        """
        return self._basic_clean_text(text)
    
    def _try_ocr_fallback(self, page: fitz.Page) -> str:
        """
        Attempt OCR on page images as fallback when text extraction fails.
        Uses PyMuPDF's built-in OCR capabilities if available.
        """
        try:
            # Get page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scaling for better OCR
            
            # Try to use PyMuPDF's OCR if available (requires pymupdf[ocr] or tesseract)
            try:
                # This requires tesseract to be installed
                ocr_text = page.get_textpage_ocr().extractText()
                if ocr_text and len(ocr_text.strip()) > 20:
                    self.logger.debug("Successfully extracted text using PyMuPDF OCR")
                    return ocr_text
            except (AttributeError, Exception) as e:
                self.logger.debug(f"PyMuPDF OCR not available or failed: {e}")
            
            # Fallback: Convert to image and indicate OCR would be needed
            # For now, return a placeholder indicating image content was found
            if pix.width > 100 and pix.height > 100:  # Reasonable image size
                return "[IMAGE CONTENT DETECTED - OCR capability not fully configured]"
            
            return ""
            
        except Exception as e:
            self.logger.debug(f"OCR fallback failed: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better context preservation.
        Tries to split on sentence boundaries when possible.
        """
        if not text:
            return []
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find a good breaking point (sentence end)
            chunk_text = text[start:end]
            
            # Look for sentence boundaries near the end
            sentence_breaks = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
            best_break = -1
            
            # Search backwards from the end for a good break point
            search_start = max(0, len(chunk_text) - 200)  # Search last 200 chars
            for i in range(len(chunk_text) - 1, search_start, -1):
                for break_pattern in sentence_breaks:
                    if chunk_text[i:i+len(break_pattern)] == break_pattern:
                        best_break = i + len(break_pattern)
                        break
                if best_break != -1:
                    break
            
            if best_break != -1:
                # Found a good break point
                actual_end = start + best_break
                chunks.append(text[start:actual_end].strip())
                start = actual_end - overlap
            else:
                # No good break point found, use hard cutoff
                chunks.append(chunk_text)
                start = end - overlap
            
            # Ensure we don't go backwards
            if start < 0:
                start = 0
        
        # Remove empty chunks and very short chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip() and len(chunk.strip()) > 50]
        
        self.logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    async def extract_and_chunk_pdf(self, pdf_content: bytes, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Extract text from PDF and split into chunks in one operation.
        """
        text = await self.extract_text_from_pdf(pdf_content)
        chunks = self.chunk_text(text, chunk_size, overlap)
        return chunks

# Global instance
text_extraction_service = TextExtractionService()