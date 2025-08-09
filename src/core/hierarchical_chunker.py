"""
Hierarchical Document Chunker with Parallel Optimization
Advanced chunking with semantic boundary detection and parallel processing
"""

import asyncio
import hashlib
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Union

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

from src.core.config import config
from src.services.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Data class for text chunks"""
    chunk_id: str
    text: str
    start_pos: int
    end_pos: int
    token_count: int
    char_count: int
    chunk_type: str  # paragraph, section, sentence
    hierarchy_level: int
    source_info: dict[str, Any]
    metadata: dict[str, Any] | None = None


@dataclass
class ChunkingResult:
    """Data class for chunking results"""
    chunks: list[TextChunk]
    total_chunks: int
    total_tokens: int
    total_characters: int
    processing_time: float
    chunking_strategy: str
    overlap_info: dict[str, Any]


class HierarchicalChunker:
    """
    Advanced hierarchical document chunker with parallel processing
    Supports semantic boundary detection and cross-lingual chunking
    """

    def __init__(self):
        # Configuration from central config
        self.chunk_size = config.chunk_size
        self.chunk_overlap = config.chunk_overlap
        self.max_workers = config.max_workers
        self.embedding_max_length = config.embedding_max_length

        # Initialize language detector for cross-lingual support
        self.language_detector = LanguageDetector()

        # Token encoder for accurate token counting
        self.token_encoder = None
        if HAS_TIKTOKEN:
            try:
                self.token_encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
            except Exception as e:
                logger.warning(f"Failed to load tiktoken encoder: {str(e)}")

        # Chunking statistics
        self.total_documents_chunked = 0
        self.total_chunks_created = 0
        self.total_processing_time = 0.0

        # Semantic boundary patterns
        self._init_boundary_patterns()

        logger.info(f"HierarchicalChunker initialized: {self.chunk_size} tokens, "
                   f"{self.chunk_overlap} overlap, {self.max_workers} workers")

    def _init_boundary_patterns(self):
        """Initialize patterns for semantic boundary detection"""
        self.section_patterns = [
            r'^#{1,6}\s+(.+)$',  # Markdown headers
            r'^(CHAPTER|Chapter|Section|SECTION|Part|PART)\s+\d+',  # Chapter/Section headers
            r'^[A-Z][A-Z\s]{10,}$',  # ALL CAPS headers
            r'^\d+\.\s+[A-Z]',  # Numbered sections
            r'^[IVX]+\.\s+[A-Z]',  # Roman numeral sections
        ]

        self.paragraph_patterns = [
            r'\n\s*\n',  # Double newlines
            r'\.\s*\n',  # Sentence endings with newlines
            r'[.!?]\s*\n\s*[A-Z]',  # Sentence boundaries
        ]

        self.sentence_patterns = [
            r'[.!?]+\s+[A-Z]',  # Sentence boundaries
            r'[.!?]+\s*\n',  # Sentences ending with newlines
            r'[।॥]\s*',  # Devanagari sentence endings (for Indic languages)
        ]

    async def chunk_document(
        self,
        text: str,
        source_info: dict[str, Any],
        chunking_strategy: str = "hierarchical"
    ) -> ChunkingResult:
        """
        Chunk document using hierarchical approach with parallel processing
        
        Args:
            text: Document text to chunk
            source_info: Information about source document
            chunking_strategy: Strategy to use (hierarchical, fixed, semantic)
            
        Returns:
            ChunkingResult with all created chunks and metadata
        """
        logger.info(f"🔪 Starting hierarchical chunking: {len(text)} chars, strategy={chunking_strategy}")
        start_time = time.time()

        try:
            # Detect language for cross-lingual optimization
            language_info = await self._detect_document_language(text)

            # Preprocess text for better chunking
            preprocessed_text = self._preprocess_text(text, language_info)

            # Route to appropriate chunking strategy
            if chunking_strategy == "hierarchical":
                chunks = await self._hierarchical_chunking(preprocessed_text, source_info, language_info)
            elif chunking_strategy == "semantic":
                chunks = await self._semantic_chunking(preprocessed_text, source_info, language_info)
            elif chunking_strategy == "fixed":
                chunks = await self._fixed_chunking(preprocessed_text, source_info, language_info)
            else:
                chunks = await self._hierarchical_chunking(preprocessed_text, source_info, language_info)

            # Post-process chunks
            final_chunks = await self._post_process_chunks(chunks, language_info)

            # Calculate overlap information
            overlap_info = self._calculate_overlap_info(final_chunks)

            processing_time = time.time() - start_time

            # Update statistics
            self.total_documents_chunked += 1
            self.total_chunks_created += len(final_chunks)
            self.total_processing_time += processing_time

            result = ChunkingResult(
                chunks=final_chunks,
                total_chunks=len(final_chunks),
                total_tokens=sum(chunk.token_count for chunk in final_chunks),
                total_characters=sum(chunk.char_count for chunk in final_chunks),
                processing_time=processing_time,
                chunking_strategy=chunking_strategy,
                overlap_info=overlap_info
            )

            logger.info(f"✅ Hierarchical chunking completed: {len(final_chunks)} chunks "
                       f"in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"❌ Hierarchical chunking failed: {str(e)}")
            raise

    async def _detect_document_language(self, text: str) -> dict[str, Any]:
        """Detect document language for optimized chunking"""
        try:
            # Use sample text for language detection
            sample_size = min(1000, len(text))
            sample_text = text[:sample_size]

            language_result = self.language_detector.detect_language(sample_text, detailed=True)

            return {
                "primary_language": language_result.get("detected_language", "en"),
                "confidence": language_result.get("confidence", 0.0),
                "is_multilingual": False,  # Could be enhanced to detect mixed content
                "detection_details": language_result
            }

        except Exception as e:
            logger.warning(f"Language detection for chunking failed: {str(e)}")
            return {
                "primary_language": "en",
                "confidence": 0.0,
                "is_multilingual": False,
                "error": str(e)
            }

    def _preprocess_text(self, text: str, language_info: dict[str, Any]) -> str:
        """Preprocess text for better chunking based on language"""
        try:
            # Normalize Unicode
            text = self.language_detector.normalize_text_encoding(text)

            # Language-specific preprocessing
            if language_info["primary_language"] == "ml":  # Malayalam
                # Malayalam-specific preprocessing
                text = self._preprocess_malayalam_text(text)
            elif language_info["primary_language"] == "en":  # English
                # English-specific preprocessing
                text = self._preprocess_english_text(text)

            # Common preprocessing
            text = self._common_text_preprocessing(text)

            return text

        except Exception as e:
            logger.warning(f"Text preprocessing failed: {str(e)}")
            return text

    def _preprocess_malayalam_text(self, text: str) -> str:
        """Malayalam-specific text preprocessing"""
        # Normalize Malayalam punctuation
        text = text.replace('।', '.')  # Devanagari danda to period
        text = text.replace('॥', '.')  # Double danda to period

        return text

    def _preprocess_english_text(self, text: str) -> str:
        """English-specific text preprocessing"""
        # Fix common English text issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double

        return text

    def _common_text_preprocessing(self, text: str) -> str:
        """Common text preprocessing for all languages"""
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading whitespace on lines
        text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing whitespace on lines

        return text.strip()

    async def _hierarchical_chunking(
        self,
        text: str,
        source_info: dict[str, Any],
        language_info: dict[str, Any]
    ) -> list[TextChunk]:
        """Perform hierarchical chunking with parallel processing"""

        # Step 1: Split into major sections
        sections = await self._split_into_sections(text, language_info)

        # Step 2: Process sections in parallel
        all_chunks = []

        if len(sections) > 1 and self.max_workers > 1:
            # Parallel processing for multiple sections
            all_chunks = await self._process_sections_parallel(sections, source_info, language_info)
        else:
            # Sequential processing for single section or limited workers
            for i, section in enumerate(sections):
                section_chunks = await self._process_single_section(
                    section, i, source_info, language_info
                )
                all_chunks.extend(section_chunks)

        return all_chunks

    async def _split_into_sections(self, text: str, language_info: dict[str, Any]) -> list[str]:
        """Split text into major sections"""
        try:
            # Try to detect section boundaries
            section_boundaries = []

            # Look for clear section markers
            for pattern in self.section_patterns:
                matches = list(re.finditer(pattern, text, re.MULTILINE))
                for match in matches:
                    section_boundaries.append(match.start())

            # If no clear sections found, split by paragraph groups
            if not section_boundaries:
                # Split by double newlines (paragraph boundaries)
                paragraphs = re.split(r'\n\s*\n', text)

                # Group paragraphs into sections
                sections = []
                current_section = []
                current_length = 0
                target_section_size = self.chunk_size * 3  # Aim for 3 chunks per section

                for paragraph in paragraphs:
                    para_length = len(paragraph)

                    if current_length + para_length > target_section_size and current_section:
                        sections.append('\n\n'.join(current_section))
                        current_section = [paragraph]
                        current_length = para_length
                    else:
                        current_section.append(paragraph)
                        current_length += para_length

                # Add remaining paragraphs
                if current_section:
                    sections.append('\n\n'.join(current_section))

                return sections

            # Split text at section boundaries
            section_boundaries.sort()
            sections = []

            for i, start in enumerate(section_boundaries):
                end = section_boundaries[i + 1] if i + 1 < len(section_boundaries) else len(text)
                section_text = text[start:end].strip()
                if section_text:
                    sections.append(section_text)

            # If no sections created, return full text as single section
            return sections if sections else [text]

        except Exception as e:
            logger.warning(f"Section splitting failed: {str(e)}")
            return [text]

    async def _process_sections_parallel(
        self,
        sections: list[str],
        source_info: dict[str, Any],
        language_info: dict[str, Any]
    ) -> list[TextChunk]:
        """Process sections in parallel using ThreadPoolExecutor"""

        async def process_section_wrapper(section_data):
            section_text, section_index = section_data
            return await self._process_single_section(
                section_text, section_index, source_info, language_info
            )

        # Prepare section data
        section_data = [(section, i) for i, section in enumerate(sections)]

        # Use ThreadPoolExecutor for CPU-bound chunking operations
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(sections))) as executor:
            loop = asyncio.get_event_loop()

            # Submit all section processing tasks
            tasks = []
            for data in section_data:
                task = loop.run_in_executor(
                    executor,
                    lambda d=data: asyncio.run(process_section_wrapper(d))
                )
                tasks.append(task)

            # Gather results
            section_results = await asyncio.gather(*tasks)

        # Flatten results
        all_chunks = []
        for section_chunks in section_results:
            all_chunks.extend(section_chunks)

        return all_chunks

    async def _process_single_section(
        self,
        section_text: str,
        section_index: int,
        source_info: dict[str, Any],
        language_info: dict[str, Any]
    ) -> list[TextChunk]:
        """Process a single section into chunks"""

        chunks = []

        # Split section into paragraphs
        paragraphs = self._split_into_paragraphs(section_text, language_info)

        current_chunk_text = ""
        current_chunk_start = 0
        chunk_counter = 0

        for para_index, paragraph in enumerate(paragraphs):
            # Check if adding this paragraph would exceed chunk size
            potential_text = current_chunk_text + ("\n\n" if current_chunk_text else "") + paragraph
            potential_token_count = self._count_tokens(potential_text)

            if potential_token_count > self.chunk_size and current_chunk_text:
                # Create chunk from current text
                chunk = self._create_chunk(
                    current_chunk_text,
                    current_chunk_start,
                    len(current_chunk_text),
                    chunk_counter,
                    section_index,
                    source_info,
                    language_info
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk_text)
                current_chunk_text = overlap_text + paragraph
                current_chunk_start = len(current_chunk_text) - len(paragraph)
                chunk_counter += 1
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + paragraph
                else:
                    current_chunk_text = paragraph

        # Create final chunk if there's remaining text
        if current_chunk_text.strip():
            chunk = self._create_chunk(
                current_chunk_text,
                current_chunk_start,
                len(current_chunk_text),
                chunk_counter,
                section_index,
                source_info,
                language_info
            )
            chunks.append(chunk)

        return chunks

    def _split_into_paragraphs(self, text: str, language_info: dict[str, Any]) -> list[str]:
        """Split text into paragraphs based on language"""
        try:
            # Language-specific paragraph splitting
            if language_info["primary_language"] == "ml":
                # Malayalam paragraph splitting
                paragraphs = re.split(r'\n\s*\n|[।॥]\s*\n', text)
            else:
                # Standard paragraph splitting
                paragraphs = re.split(r'\n\s*\n', text)

            # Clean and filter paragraphs
            clean_paragraphs = []
            for para in paragraphs:
                para = para.strip()
                if para and len(para) > 10:  # Filter very short paragraphs
                    clean_paragraphs.append(para)

            return clean_paragraphs

        except Exception as e:
            logger.warning(f"Paragraph splitting failed: {str(e)}")
            return [text]

    def _create_chunk(
        self,
        text: str,
        start_pos: int,
        end_pos: int,
        chunk_index: int,
        section_index: int,
        source_info: dict[str, Any],
        language_info: dict[str, Any]
    ) -> TextChunk:
        """Create a TextChunk object"""

        # Generate unique chunk ID
        chunk_id = self._generate_chunk_id(text, source_info, section_index, chunk_index)

        # Count tokens and characters
        token_count = self._count_tokens(text)
        char_count = len(text)

        # Determine chunk type and hierarchy level
        chunk_type, hierarchy_level = self._analyze_chunk_type(text, language_info)

        # Create metadata
        metadata = {
            "section_index": section_index,
            "chunk_index": chunk_index,
            "language": language_info["primary_language"],
            "confidence": language_info["confidence"],
            "created_at": time.time(),
            "chunking_method": "hierarchical",
            "token_count": token_count,
            "char_count": char_count
        }

        return TextChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            start_pos=start_pos,
            end_pos=end_pos,
            token_count=token_count,
            char_count=char_count,
            chunk_type=chunk_type,
            hierarchy_level=hierarchy_level,
            source_info=source_info,
            metadata=metadata
        )

    def _generate_chunk_id(
        self,
        text: str,
        source_info: dict[str, Any],
        section_index: int,
        chunk_index: int
    ) -> str:
        """Generate unique chunk ID"""
        source_id = source_info.get("document_url", source_info.get("file_path", "unknown"))
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]

        return f"chunk_{section_index}_{chunk_index}_{content_hash}"

    def _analyze_chunk_type(self, text: str, language_info: dict[str, Any]) -> tuple[str, int]:
        """Analyze chunk type and hierarchy level"""
        try:
            # Check for headers
            for i, pattern in enumerate(self.section_patterns):
                if re.search(pattern, text, re.MULTILINE):
                    return "section", i + 1

            # Check for multiple paragraphs
            if len(text.split('\n\n')) > 1:
                return "paragraph_group", 2

            # Single paragraph
            return "paragraph", 3

        except Exception:
            return "paragraph", 3

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if self.token_encoder:
            try:
                return len(self.token_encoder.encode(text))
            except Exception:
                pass

        # Fallback: rough token estimation
        return len(text.split()) + text.count(',') + text.count('.')

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from end of chunk"""
        if not text or self.chunk_overlap <= 0:
            return ""

        # Get last N tokens for overlap
        words = text.split()
        overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words

        return " ".join(overlap_words) + "\n\n"

    async def _semantic_chunking(
        self,
        text: str,
        source_info: dict[str, Any],
        language_info: dict[str, Any]
    ) -> list[TextChunk]:
        """Semantic chunking based on content structure"""
        # Implementation for semantic chunking
        # For now, fallback to hierarchical
        return await self._hierarchical_chunking(text, source_info, language_info)

    async def _fixed_chunking(
        self,
        text: str,
        source_info: dict[str, Any],
        language_info: dict[str, Any]
    ) -> list[TextChunk]:
        """Fixed-size chunking with overlap"""
        chunks = []
        words = text.split()

        chunk_counter = 0
        i = 0

        while i < len(words):
            # Get chunk words
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)

            # Create chunk
            chunk = self._create_chunk(
                chunk_text,
                i,
                i + len(chunk_words),
                chunk_counter,
                0,  # section_index
                source_info,
                language_info
            )
            chunks.append(chunk)

            # Move forward with overlap
            i += self.chunk_size - self.chunk_overlap
            chunk_counter += 1

        return chunks

    async def _post_process_chunks(
        self,
        chunks: list[TextChunk],
        language_info: dict[str, Any]
    ) -> list[TextChunk]:
        """Post-process chunks for quality and consistency"""

        processed_chunks = []

        for chunk in chunks:
            # Filter very small chunks
            if chunk.token_count < 10:
                continue

            # Ensure chunk doesn't exceed maximum length
            if chunk.token_count > self.embedding_max_length:
                # Split large chunks
                split_chunks = await self._split_large_chunk(chunk, language_info)
                processed_chunks.extend(split_chunks)
            else:
                processed_chunks.append(chunk)

        return processed_chunks

    async def _split_large_chunk(
        self,
        chunk: TextChunk,
        language_info: dict[str, Any]
    ) -> list[TextChunk]:
        """Split a chunk that's too large"""

        # Simple splitting by sentences for now
        sentences = re.split(r'[.!?]+\s+', chunk.text)

        split_chunks = []
        current_text = ""
        chunk_counter = 0

        for sentence in sentences:
            potential_text = current_text + ". " + sentence if current_text else sentence

            if self._count_tokens(potential_text) > self.embedding_max_length and current_text:
                # Create chunk
                split_chunk = TextChunk(
                    chunk_id=f"{chunk.chunk_id}_split_{chunk_counter}",
                    text=current_text,
                    start_pos=chunk.start_pos,
                    end_pos=chunk.end_pos,
                    token_count=self._count_tokens(current_text),
                    char_count=len(current_text),
                    chunk_type="split_chunk",
                    hierarchy_level=chunk.hierarchy_level + 1,
                    source_info=chunk.source_info,
                    metadata={**chunk.metadata, "split_from": chunk.chunk_id}
                )
                split_chunks.append(split_chunk)

                current_text = sentence
                chunk_counter += 1
            else:
                current_text = potential_text

        # Add final chunk
        if current_text:
            split_chunk = TextChunk(
                chunk_id=f"{chunk.chunk_id}_split_{chunk_counter}",
                text=current_text,
                start_pos=chunk.start_pos,
                end_pos=chunk.end_pos,
                token_count=self._count_tokens(current_text),
                char_count=len(current_text),
                chunk_type="split_chunk",
                hierarchy_level=chunk.hierarchy_level + 1,
                source_info=chunk.source_info,
                metadata={**chunk.metadata, "split_from": chunk.chunk_id}
            )
            split_chunks.append(split_chunk)

        return split_chunks

    def _calculate_overlap_info(self, chunks: list[TextChunk]) -> dict[str, Any]:
        """Calculate information about chunk overlaps"""
        if not chunks:
            return {}

        total_overlap_chars = 0
        overlap_count = 0

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            previous_chunk = chunks[i - 1]

            # Simple overlap detection (could be more sophisticated)
            if (current_chunk.source_info == previous_chunk.source_info and
                current_chunk.start_pos < previous_chunk.end_pos):
                overlap_chars = previous_chunk.end_pos - current_chunk.start_pos
                total_overlap_chars += overlap_chars
                overlap_count += 1

        avg_overlap = total_overlap_chars / overlap_count if overlap_count > 0 else 0

        return {
            "total_overlaps": overlap_count,
            "total_overlap_characters": total_overlap_chars,
            "average_overlap_characters": round(avg_overlap, 1),
            "overlap_ratio": round((total_overlap_chars / sum(c.char_count for c in chunks)) * 100, 2)
        }

    def get_chunking_stats(self) -> dict[str, Any]:
        """Get chunking performance statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_documents_chunked
            if self.total_documents_chunked > 0 else 0.0
        )

        avg_chunks_per_doc = (
            self.total_chunks_created / self.total_documents_chunked
            if self.total_documents_chunked > 0 else 0.0
        )

        return {
            "total_documents_chunked": self.total_documents_chunked,
            "total_chunks_created": self.total_chunks_created,
            "total_processing_time": round(self.total_processing_time, 2),
            "average_processing_time": round(avg_processing_time, 3),
            "average_chunks_per_document": round(avg_chunks_per_doc, 1),
            "configuration": {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "max_workers": self.max_workers,
                "embedding_max_length": self.embedding_max_length
            },
            "features": {
                "hierarchical_chunking": True,
                "semantic_boundary_detection": True,
                "parallel_processing": self.max_workers > 1,
                "cross_lingual_support": True,
                "token_counting": HAS_TIKTOKEN,
                "overlap_handling": True
            }
        }
