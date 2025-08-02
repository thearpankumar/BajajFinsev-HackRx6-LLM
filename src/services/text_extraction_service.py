import logging
import asyncio
import fitz  # PyMuPDF
from typing import List, Dict, Any, Tuple
import re
import statistics
import unicodedata
import hashlib

logger = logging.getLogger(__name__)

class TextExtractionService:
    """
    Advanced PDF text extraction service that extracts text and metadata (page numbers).
    """
    
    def __init__(self):
        self.logger = logger
        # All private methods from the previous version (_analyze_document, _detect_tables, etc.)
        # are assumed to be here. They are omitted for brevity in this call, but are part of the class.
        self.extraction_methods = ["dict", "text", "html", "blocks"]

    async def extract_text_from_pdf_with_metadata(self, pdf_content: bytes) -> List[Tuple[str, int]]:
        """
        Extracts text from each page of a PDF and returns it with the page number.
        """
        self.logger.info("Starting advanced PDF text extraction with metadata.")
        try:
            pages_with_metadata = await asyncio.wait_for(
                asyncio.to_thread(self._extract_text_advanced, pdf_content),
                timeout=300
            )
            self.logger.info(f"Extracted text from {len(pages_with_metadata)} pages.")
            return pages_with_metadata
        except asyncio.TimeoutError:
            self.logger.error("PDF text extraction timed out.")
            raise Exception("PDF processing timed out.")
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
            raise
    
    def _extract_text_advanced(self, pdf_content: bytes) -> List[Tuple[str, int]]:
        """
        Core logic for extracting text and page numbers.
        """
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        pages_with_metadata = []
        extraction_stats = self._analyze_document(pdf_document)

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = self._extract_page_text_multimethod(page, page_num, extraction_stats)
            if page_text.strip():
                # Append a tuple of (page_text, page_number)
                pages_with_metadata.append((page_text, page_num + 1))
        
        pdf_document.close()
        return pages_with_metadata

    def chunk_text(self, pages_with_metadata: List[Tuple[str, int]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Splits text from pages into unique, semantically coherent chunks with metadata.
        """
        if not pages_with_metadata:
            return []

        all_chunks = []
        seen_hashes = set()

        for page_text, page_number in pages_with_metadata:
            sentences = re.split(r'(?<=[.!?])\s+', page_text)
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunk_text = current_chunk.strip()
                        if chunk_text and len(chunk_text) > 50:
                            chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                            if chunk_hash not in seen_hashes:
                                all_chunks.append({
                                    "text": chunk_text,
                                    "metadata": {"page_number": page_number}
                                })
                                seen_hashes.add(chunk_hash)
                    current_chunk = sentence
            
            if current_chunk:
                chunk_text = current_chunk.strip()
                if chunk_text and len(chunk_text) > 50:
                    chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
                    if chunk_hash not in seen_hashes:
                        all_chunks.append({
                            "text": chunk_text,
                            "metadata": {"page_number": page_number}
                        })
                        seen_hashes.add(chunk_hash)

        self.logger.info(f"Split text into {len(all_chunks)} unique semantic chunks with metadata.")
        return all_chunks

    async def extract_and_chunk_pdf(self, pdf_content: bytes, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Extracts text from PDF and splits it into chunks with metadata in one operation.
        """
        pages_with_metadata = await self.extract_text_from_pdf_with_metadata(pdf_content)
        chunks = self.chunk_text(pages_with_metadata, chunk_size, overlap)
        return chunks

    # NOTE: All other private helper methods (_analyze_document, _detect_tables, 
    # _extract_page_text_multimethod, etc.) are included here as they were before.
    # They are omitted from this view for brevity but are part of the written file.
    def _analyze_document(self, pdf_document: fitz.Document) -> Dict[str, Any]:
        stats = {"total_pages": len(pdf_document), "has_images": False, "has_tables": False, "has_columns": False, "text_density": 0.0, "font_diversity": 0, "recommended_method": "text"}
        sample_pages = min(5, len(pdf_document))
        sample_indices = [i * len(pdf_document) // sample_pages for i in range(sample_pages)]
        total_chars = 0
        font_sizes = []
        for page_idx in sample_indices:
            page = pdf_document[page_idx]
            if page.get_images():
                stats["has_images"] = True
            blocks = page.get_text("dict")["blocks"]
            text_blocks = [b for b in blocks if "lines" in b]
            if len(text_blocks) > 1:
                x_positions = []
                for block in text_blocks:
                    for line in block["lines"]:
                        x_positions.append(line["bbox"][0])
                if len(set(x_positions)) > 2:
                    stats["has_columns"] = True
            if self._detect_tables(blocks):
                stats["has_tables"] = True
            page_text = page.get_text()
            total_chars += len(page_text)
            for block in text_blocks:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
        stats["text_density"] = total_chars / stats["total_pages"] if stats["total_pages"] > 0 else 0
        stats["font_diversity"] = len(set(font_sizes))
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
        text_blocks = [b for b in blocks if "lines" in b]
        if len(text_blocks) < 3:
            return False
        y_positions = []
        for block in text_blocks:
            for line in block["lines"]:
                y_positions.append(line["bbox"][1])
        if len(y_positions) < 3:
            return False
        y_positions.sort()
        differences = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
        if len(differences) < 2:
            return False
        avg_diff = statistics.mean(differences)
        similar_diffs = sum(1 for d in differences if abs(d - avg_diff) < avg_diff * 0.3)
        return similar_diffs > len(differences) * 0.4

    def _extract_page_text_multimethod(self, page: fitz.Page, page_num: int, stats: Dict) -> str:
        primary_method = stats["recommended_method"]
        text = self._extract_by_method(page, primary_method)
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
                    except Exception:
                        pass
            text = best_text
        return self._clean_and_enhance_text(text, stats)

    def _extract_by_method(self, page: fitz.Page, method: str) -> str:
        if method == "text":
            return page.get_text("text")
        elif method == "html":
            return self._html_to_text(page.get_text("html"))
        elif method == "dict":
            return self._dict_to_text(page.get_text("dict"))
        elif method == "blocks":
            return self._blocks_to_text(page.get_text("blocks"))
        return page.get_text("text")

    def _html_to_text(self, html: str) -> str:
        text = re.sub(r'<br[^>]*>', '\n', html)
        text = re.sub(r'</p>', '\n\n', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
        return text

    def _dict_to_text(self, text_dict: Dict) -> str:
        text_lines = []
        for block in text_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    line_text = "".join(span["text"] for span in line["spans"])
                    if line_text.strip():
                        text_lines.append(line_text.strip())
                text_lines.append("")
        return "\n".join(text_lines)

    def _blocks_to_text(self, blocks) -> str:
        text_blocks = []
        for block in blocks:
            if len(block) >= 5 and isinstance(block[4], str):
                if block[4].strip():
                    text_blocks.append((block[:4], block[4].strip()))
        text_blocks.sort(key=lambda x: (x[0][1], x[0][0]))
        return "\n".join([text for _, text in text_blocks])

    def _score_extraction(self, text: str) -> float:
        if not text.strip(): return 0.0
        score = min(len(text) / 1000, 10.0)
        words = text.split()
        score += min(len(words) / 100, 5.0)
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]{}')
        special_ratio = special_chars / len(text) if text else 0
        score -= special_ratio * 5
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if 5 <= avg_sentence_length <= 30: score += 2.0
        return max(score, 0.0)

    def _is_poor_extraction(self, text: str) -> bool:
        if not text.strip(): return True
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' \n\t.,!?;:-()[]{}')
        if special_chars / len(text) > 0.3: return True
        words = text.split()
        if not words: return True
        short_words = sum(1 for w in words if len(w) == 1 and w.isalpha())
        if short_words / len(words) > 0.5: return True
        return False

    def _clean_and_enhance_text(self, text: str, stats: Dict) -> str:
        if not text: return ""
        text = unicodedata.normalize('NFKC', text)
        text = self._basic_clean_text(text)
        if stats.get("has_tables"):
            text = self._clean_table_text(text)
        text = self._remove_artifacts(text)
        text = self._fix_word_breaks(text)
        text = self._normalize_spacing(text)
        return text.strip()

    def _basic_clean_text(self, text: str) -> str:
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n ', '\n', text)
        text = re.sub(r' \n', '\n', text)
        text = re.sub(r'\x0c', '', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        return text

    def _clean_table_text(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = [re.sub(r'\s{3,}', '\t', line) if re.search(r'\s{3,}', line) else line for line in lines]
        return '\n'.join(cleaned_lines)

    def _remove_artifacts(self, text: str) -> str:
        text = re.sub(r'^[-\_\s]{3,}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\^\\w\s]*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(.)\1{5,}', r'\1\1', text)
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

# Global instance
text_extraction_service = TextExtractionService()