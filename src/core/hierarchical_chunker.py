"""
Hierarchical Document Chunking for BajajFinsev Hybrid RAG System
Intelligent text segmentation with format-aware chunking strategies
Supports semantic chunking, overlap management, and chunk optimization
"""

import asyncio
import logging
import re
import time
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math

# NLP libraries for advanced chunking
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

logger = logging.getLogger(__name__)


class ChunkStrategy(Enum):
    """Chunking strategy types"""
    FIXED_SIZE = "fixed_size"
    SENTENCE_AWARE = "sentence_aware"
    PARAGRAPH_AWARE = "paragraph_aware"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    FORMAT_SPECIFIC = "format_specific"
    HIERARCHICAL = "hierarchical"


@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    chunk_id: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    sentence_count: int
    
    # Hierarchy information
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    hierarchy_level: int = 0
    
    # Content type
    content_type: str = "text"  # text, table, list, code, etc.
    format_source: str = ""  # pdf_page_1, docx_paragraph_5, etc.
    
    # Quality metrics
    information_density: float = 0.0
    semantic_coherence: float = 0.0
    
    # Processing metadata
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    overlap_with_previous: int = 0
    overlap_with_next: int = 0
    
    # Additional context
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    table_header: Optional[str] = None


@dataclass
class DocumentChunk:
    """Individual document chunk with content and metadata"""
    content: str
    metadata: ChunkMetadata
    
    # Optional embeddings (computed later)
    embeddings: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    
    # Quality scores
    content_quality: float = 0.0
    chunk_coherence: float = 0.0


@dataclass
class ChunkingResult:
    """Result of document chunking operation"""
    success: bool
    document_url: str = ""
    document_hash: str = ""
    
    # Chunks
    chunks: List[DocumentChunk] = field(default_factory=list)
    
    # Chunking metadata
    total_chunks: int = 0
    chunking_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    chunking_time: float = 0.0
    
    # Statistics
    avg_chunk_size: float = 0.0
    min_chunk_size: int = 0
    max_chunk_size: int = 0
    avg_overlap: float = 0.0
    
    # Hierarchy information
    hierarchy_levels: int = 1
    chunk_tree: Optional[Dict[str, Any]] = None
    
    # Error handling
    error_message: Optional[str] = None
    processing_warnings: List[str] = field(default_factory=list)


class HierarchicalChunker:
    """
    Advanced hierarchical document chunker
    Supports multiple chunking strategies and format-aware processing
    """
    
    # Format-specific chunking configurations
    FORMAT_CONFIGS = {
        ".pdf": {
            "default_chunk_size": 512,
            "overlap_size": 64,
            "strategy": ChunkStrategy.PARAGRAPH_AWARE,
            "preserve_page_boundaries": True,
            "table_handling": "separate_chunks"
        },
        ".docx": {
            "default_chunk_size": 400,
            "overlap_size": 50,
            "strategy": ChunkStrategy.PARAGRAPH_AWARE,
            "preserve_formatting": True,
            "table_handling": "inline_with_context"
        },
        ".xlsx": {
            "default_chunk_size": 300,
            "overlap_size": 30,
            "strategy": ChunkStrategy.FORMAT_SPECIFIC,
            "row_based_chunking": True,
            "preserve_headers": True
        },
        ".txt": {
            "default_chunk_size": 500,
            "overlap_size": 75,
            "strategy": ChunkStrategy.SENTENCE_AWARE,
            "paragraph_detection": True
        },
        # Image formats (OCR text)
        ".jpg": {"default_chunk_size": 256, "overlap_size": 32, "strategy": ChunkStrategy.SENTENCE_AWARE},
        ".jpeg": {"default_chunk_size": 256, "overlap_size": 32, "strategy": ChunkStrategy.SENTENCE_AWARE},
        ".png": {"default_chunk_size": 256, "overlap_size": 32, "strategy": ChunkStrategy.SENTENCE_AWARE}
    }
    
    def __init__(self,
                 default_chunk_size: int = 512,
                 default_overlap_size: int = 64,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 1500,
                 enable_hierarchical: bool = True,
                 enable_semantic_chunking: bool = False,
                 quality_threshold: float = 0.3):
        """
        Initialize hierarchical chunker
        
        Args:
            default_chunk_size: Default chunk size in tokens/words
            default_overlap_size: Default overlap between chunks
            min_chunk_size: Minimum chunk size
            max_chunk_size: Maximum chunk size
            enable_hierarchical: Enable hierarchical chunking
            enable_semantic_chunking: Enable semantic similarity chunking
            quality_threshold: Minimum quality threshold for chunks
        """
        self.default_chunk_size = default_chunk_size
        self.default_overlap_size = default_overlap_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.enable_hierarchical = enable_hierarchical
        self.enable_semantic_chunking = enable_semantic_chunking
        self.quality_threshold = quality_threshold
        
        # Initialize NLTK if available
        self.nltk_available = NLTK_AVAILABLE
        if self.nltk_available:
            self._ensure_nltk_data()
        
        # Statistics
        self.stats = {
            'total_chunking_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_chunks_created': 0,
            'avg_chunking_time': 0.0,
            'strategy_usage': {strategy.value: 0 for strategy in ChunkStrategy},
            'format_processing': {}
        }
        
        logger.info("HierarchicalChunker initialized")
        logger.info(f"Default chunk size: {default_chunk_size}")
        logger.info(f"Hierarchical chunking: {enable_hierarchical}")
        logger.info(f"Semantic chunking: {enable_semantic_chunking}")
        logger.info(f"NLTK available: {self.nltk_available}")
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("📥 Downloading required NLTK data...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                logger.info("✅ NLTK data downloaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Failed to download NLTK data: {e}")
                self.nltk_available = False
    
    async def chunk_document(self, 
                           text: str,
                           document_url: str = "",
                           file_format: str = "",
                           structured_content: Optional[Dict[str, Any]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> ChunkingResult:
        """
        Chunk document text using hierarchical and format-aware strategies
        
        Args:
            text: Document text to chunk
            document_url: Original document URL
            file_format: File format (e.g., .pdf, .docx)
            structured_content: Additional structured content from extraction
            metadata: Document metadata
            
        Returns:
            ChunkingResult with generated chunks and metadata
        """
        start_time = time.time()
        self.stats['total_chunking_operations'] += 1
        
        document_hash = self._generate_document_hash(document_url, text)
        
        logger.info(f"🔪 Chunking document: {len(text):,} characters")
        logger.info(f"Format: {file_format}, Hierarchical: {self.enable_hierarchical}")
        
        try:
            # Initialize result
            result = ChunkingResult(
                success=False,
                document_url=document_url,
                document_hash=document_hash,
                processing_warnings=[]
            )
            
            # Validate input
            if not text or len(text.strip()) < self.min_chunk_size:
                result.error_message = f"Text too short for chunking: {len(text)} characters"
                self._update_stats("failed")
                return result
            
            # Get format-specific configuration
            format_config = self.FORMAT_CONFIGS.get(file_format.lower(), self.FORMAT_CONFIGS[".txt"])
            chunk_size = format_config["default_chunk_size"]
            overlap_size = format_config["overlap_size"]
            strategy = format_config["strategy"]
            
            logger.debug(f"Using strategy: {strategy.value}, chunk_size: {chunk_size}, overlap: {overlap_size}")
            
            # Apply chunking strategy
            if strategy == ChunkStrategy.FORMAT_SPECIFIC:
                chunks = await self._chunk_format_specific(
                    text, file_format, structured_content, chunk_size, overlap_size
                )
            elif strategy == ChunkStrategy.HIERARCHICAL and self.enable_hierarchical:
                chunks = await self._chunk_hierarchical(
                    text, file_format, structured_content, chunk_size, overlap_size
                )
            elif strategy == ChunkStrategy.SEMANTIC_SIMILARITY and self.enable_semantic_chunking:
                chunks = await self._chunk_semantic(
                    text, chunk_size, overlap_size
                )
            elif strategy == ChunkStrategy.PARAGRAPH_AWARE:
                chunks = await self._chunk_paragraph_aware(
                    text, chunk_size, overlap_size
                )
            elif strategy == ChunkStrategy.SENTENCE_AWARE:
                chunks = await self._chunk_sentence_aware(
                    text, chunk_size, overlap_size
                )
            else:
                # Fallback to fixed size
                chunks = await self._chunk_fixed_size(
                    text, chunk_size, overlap_size
                )
            
            # Post-process chunks
            chunks = await self._post_process_chunks(chunks, file_format, metadata)
            
            # Build chunk tree if hierarchical
            chunk_tree = None
            hierarchy_levels = 1
            if self.enable_hierarchical:
                chunk_tree = self._build_chunk_tree(chunks)
                hierarchy_levels = self._calculate_hierarchy_levels(chunks)
            
            # Calculate statistics
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            overlaps = [chunk.metadata.overlap_with_previous for chunk in chunks if chunk.metadata.overlap_with_previous > 0]
            
            # Finalize result
            result.success = True
            result.chunks = chunks
            result.total_chunks = len(chunks)
            result.chunking_strategy = strategy
            result.chunking_time = time.time() - start_time
            result.avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
            result.min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
            result.max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
            result.avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
            result.hierarchy_levels = hierarchy_levels
            result.chunk_tree = chunk_tree
            
            # Update statistics
            self.stats['successful_operations'] += 1
            self.stats['total_chunks_created'] += len(chunks)
            self.stats['strategy_usage'][strategy.value] += 1
            self._update_avg_chunking_time(result.chunking_time)
            self._update_format_stats(file_format)
            
            logger.info(f"✅ Chunking completed: {len(chunks)} chunks in {result.chunking_time:.2f}s")
            logger.info(f"Avg size: {result.avg_chunk_size:.0f}, Range: {result.min_chunk_size}-{result.max_chunk_size}")
            
            return result
            
        except Exception as e:
            error_msg = f"Chunking failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            result.error_message = error_msg
            result.chunking_time = time.time() - start_time
            self._update_stats("failed")
            
            return result
    
    async def _chunk_fixed_size(self, text: str, chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Simple fixed-size chunking with overlap"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap_size):
            chunk_words = words[i:i + chunk_size]
            
            if len(chunk_words) < self.min_chunk_size // 4:  # Skip very small chunks
                continue
            
            chunk_text = ' '.join(chunk_words)
            start_char = len(' '.join(words[:i]))
            end_char = start_char + len(chunk_text)
            
            # Calculate overlap
            overlap_prev = overlap_size if i > 0 else 0
            overlap_next = overlap_size if i + chunk_size < len(words) else 0
            
            chunk_metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(i, chunk_text),
                chunk_index=len(chunks),
                start_char=start_char,
                end_char=end_char,
                word_count=len(chunk_words),
                sentence_count=self._count_sentences(chunk_text),
                chunk_strategy=ChunkStrategy.FIXED_SIZE,
                overlap_with_previous=overlap_prev,
                overlap_with_next=overlap_next
            )
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_sentence_aware(self, text: str, chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Sentence-aware chunking that respects sentence boundaries"""
        chunks = []
        
        if self.nltk_available:
            sentences = sent_tokenize(text)
        else:
            # Fallback sentence splitting
            sentences = re.split(r'[.!?]+\s+', text)
        
        current_chunk = []
        current_word_count = 0
        char_position = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, finalize current chunk
            if current_word_count + sentence_words > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                chunk_metadata = ChunkMetadata(
                    chunk_id=self._generate_chunk_id(len(chunks), chunk_text),
                    chunk_index=len(chunks),
                    start_char=char_position,
                    end_char=char_position + len(chunk_text),
                    word_count=current_word_count,
                    sentence_count=len(current_chunk),
                    chunk_strategy=ChunkStrategy.SENTENCE_AWARE,
                    overlap_with_previous=0,  # Calculated later
                    overlap_with_next=0
                )
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                
                chunks.append(chunk)
                
                # Handle overlap
                if overlap_size > 0 and len(current_chunk) > 1:
                    # Keep last few sentences for overlap
                    overlap_sentences = current_chunk[-2:]  # Keep last 2 sentences
                    current_chunk = overlap_sentences
                    current_word_count = sum(len(s.split()) for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_word_count = 0
                
                char_position += len(chunk_text)
            
            # Add current sentence
            current_chunk.append(sentence)
            current_word_count += sentence_words
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            
            chunk_metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(len(chunks), chunk_text),
                chunk_index=len(chunks),
                start_char=char_position,
                end_char=char_position + len(chunk_text),
                word_count=current_word_count,
                sentence_count=len(current_chunk),
                chunk_strategy=ChunkStrategy.SENTENCE_AWARE
            )
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_paragraph_aware(self, text: str, chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Paragraph-aware chunking that respects paragraph boundaries"""
        chunks = []
        
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = []
        current_word_count = 0
        char_position = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph_words = len(paragraph.split())
            
            # If single paragraph is too large, split it
            if paragraph_words > chunk_size * 1.5:
                # Split large paragraph by sentences
                if current_chunk:
                    # Finalize current chunk first
                    chunk_text = '\n\n'.join(current_chunk)
                    chunk = self._create_chunk_from_paragraphs(
                        current_chunk, len(chunks), char_position, ChunkStrategy.PARAGRAPH_AWARE
                    )
                    chunks.append(chunk)
                    char_position += len(chunk_text)
                    current_chunk = []
                    current_word_count = 0
                
                # Split the large paragraph
                large_para_chunks = await self._chunk_sentence_aware(paragraph, chunk_size, overlap_size)
                for large_chunk in large_para_chunks:
                    large_chunk.metadata.chunk_index = len(chunks)
                    large_chunk.metadata.start_char += char_position
                    large_chunk.metadata.end_char += char_position
                    chunks.append(large_chunk)
                
                char_position += len(paragraph)
                continue
            
            # If adding this paragraph exceeds chunk size, finalize current chunk
            if current_word_count + paragraph_words > chunk_size and current_chunk:
                chunk = self._create_chunk_from_paragraphs(
                    current_chunk, len(chunks), char_position, ChunkStrategy.PARAGRAPH_AWARE
                )
                chunks.append(chunk)
                
                char_position += len('\n\n'.join(current_chunk))
                
                # Handle overlap
                if overlap_size > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]  # Keep last paragraph
                    current_word_count = len(current_chunk[0].split())
                else:
                    current_chunk = []
                    current_word_count = 0
            
            # Add current paragraph
            current_chunk.append(paragraph)
            current_word_count += paragraph_words
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk_from_paragraphs(
                current_chunk, len(chunks), char_position, ChunkStrategy.PARAGRAPH_AWARE
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_format_specific(self, text: str, file_format: str, 
                                   structured_content: Optional[Dict[str, Any]], 
                                   chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Format-specific chunking based on document structure"""
        
        if file_format == ".xlsx" and structured_content:
            return await self._chunk_excel_format(text, structured_content, chunk_size)
        elif file_format == ".pdf" and structured_content:
            return await self._chunk_pdf_format(text, structured_content, chunk_size, overlap_size)
        elif file_format == ".docx" and structured_content:
            return await self._chunk_docx_format(text, structured_content, chunk_size, overlap_size)
        else:
            # Fallback to paragraph-aware
            return await self._chunk_paragraph_aware(text, chunk_size, overlap_size)
    
    async def _chunk_excel_format(self, text: str, structured_content: Dict[str, Any], chunk_size: int) -> List[DocumentChunk]:
        """Excel-specific chunking by sheets and rows"""
        chunks = []
        sheets = structured_content.get("sheets", [])
        
        # Split text by sheets (assuming structured format)
        sheet_sections = text.split("=== Sheet:")
        
        for i, section in enumerate(sheet_sections[1:]):  # Skip first empty section
            lines = section.split('\n')
            sheet_name = lines[0].strip() if lines else f"Sheet_{i+1}"
            
            # Group rows into chunks
            current_chunk_lines = []
            current_row_count = 0
            
            for line in lines[1:]:  # Skip sheet header
                if line.strip().startswith("Row"):
                    if current_row_count >= chunk_size // 10:  # Adjust for row-based chunking
                        if current_chunk_lines:
                            chunk_text = '\n'.join(current_chunk_lines)
                            chunk_metadata = ChunkMetadata(
                                chunk_id=self._generate_chunk_id(len(chunks), chunk_text),
                                chunk_index=len(chunks),
                                start_char=0,  # Would need better tracking
                                end_char=len(chunk_text),
                                word_count=len(chunk_text.split()),
                                sentence_count=len(current_chunk_lines),
                                chunk_strategy=ChunkStrategy.FORMAT_SPECIFIC,
                                content_type="table",
                                format_source=f"excel_sheet_{sheet_name}",
                                section_title=sheet_name
                            )
                            
                            chunk = DocumentChunk(
                                content=chunk_text,
                                metadata=chunk_metadata
                            )
                            chunks.append(chunk)
                            
                            current_chunk_lines = []
                            current_row_count = 0
                    
                    current_chunk_lines.append(line)
                    current_row_count += 1
                else:
                    current_chunk_lines.append(line)
            
            # Add final chunk for this sheet
            if current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines)
                chunk_metadata = ChunkMetadata(
                    chunk_id=self._generate_chunk_id(len(chunks), chunk_text),
                    chunk_index=len(chunks),
                    start_char=0,
                    end_char=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    sentence_count=len(current_chunk_lines),
                    chunk_strategy=ChunkStrategy.FORMAT_SPECIFIC,
                    content_type="table",
                    format_source=f"excel_sheet_{sheet_name}",
                    section_title=sheet_name
                )
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
        
        return chunks
    
    async def _chunk_pdf_format(self, text: str, structured_content: Dict[str, Any], 
                              chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """PDF-specific chunking with page awareness"""
        chunks = []
        pages = structured_content.get("pages", [])
        
        if not pages:
            # Fallback to paragraph chunking
            return await self._chunk_paragraph_aware(text, chunk_size, overlap_size)
        
        for page_idx, page_content in enumerate(pages):
            if not page_content.strip():
                continue
            
            # Remove page header
            page_text = re.sub(r'^--- Page \d+ ---\s*', '', page_content.strip())
            
            if len(page_text.split()) < self.min_chunk_size // 4:
                # Small page, combine with others or create small chunk
                chunk_metadata = ChunkMetadata(
                    chunk_id=self._generate_chunk_id(len(chunks), page_text),
                    chunk_index=len(chunks),
                    start_char=0,
                    end_char=len(page_text),
                    word_count=len(page_text.split()),
                    sentence_count=self._count_sentences(page_text),
                    chunk_strategy=ChunkStrategy.FORMAT_SPECIFIC,
                    format_source=f"pdf_page_{page_idx + 1}",
                    page_number=page_idx + 1
                )
                
                chunk = DocumentChunk(
                    content=page_text,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
            
            else:
                # Large page, chunk it
                page_chunks = await self._chunk_paragraph_aware(page_text, chunk_size, overlap_size)
                
                # Update metadata for page context
                for chunk in page_chunks:
                    chunk.metadata.chunk_index = len(chunks) + len(page_chunks) - len(page_chunks) + page_chunks.index(chunk)
                    chunk.metadata.format_source = f"pdf_page_{page_idx + 1}"
                    chunk.metadata.page_number = page_idx + 1
                    chunks.append(chunk)
        
        return chunks
    
    async def _chunk_docx_format(self, text: str, structured_content: Dict[str, Any], 
                               chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """DOCX-specific chunking with paragraph and table awareness"""
        
        paragraphs = structured_content.get("paragraphs", [])
        tables = structured_content.get("tables", [])
        
        # Combine paragraphs and tables in order (simplified)
        if paragraphs or tables:
            # Use structured content for better chunking
            return await self._chunk_structured_docx(paragraphs, tables, chunk_size, overlap_size)
        else:
            # Fallback to paragraph chunking
            return await self._chunk_paragraph_aware(text, chunk_size, overlap_size)
    
    async def _chunk_structured_docx(self, paragraphs: List[str], tables: List[str], 
                                   chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Chunk DOCX using structured paragraph and table data"""
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        # Process paragraphs
        for para in paragraphs:
            para_words = len(para.split())
            
            if current_word_count + para_words > chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunk_metadata = ChunkMetadata(
                    chunk_id=self._generate_chunk_id(len(chunks), chunk_text),
                    chunk_index=len(chunks),
                    start_char=0,
                    end_char=len(chunk_text),
                    word_count=current_word_count,
                    sentence_count=len(current_chunk),
                    chunk_strategy=ChunkStrategy.FORMAT_SPECIFIC,
                    content_type="text",
                    format_source="docx_paragraphs"
                )
                
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                chunks.append(chunk)
                
                # Handle overlap
                if overlap_size > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]
                    current_word_count = len(current_chunk[0].split())
                else:
                    current_chunk = []
                    current_word_count = 0
            
            current_chunk.append(para)
            current_word_count += para_words
        
        # Add tables as separate chunks
        for i, table in enumerate(tables):
            table_metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(len(chunks), table),
                chunk_index=len(chunks),
                start_char=0,
                end_char=len(table),
                word_count=len(table.split()),
                sentence_count=len(table.split('\n')),
                chunk_strategy=ChunkStrategy.FORMAT_SPECIFIC,
                content_type="table",
                format_source=f"docx_table_{i + 1}",
                table_header=f"Table {i + 1}"
            )
            
            chunk = DocumentChunk(
                content=table,
                metadata=table_metadata
            )
            chunks.append(chunk)
        
        # Add final text chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(len(chunks), chunk_text),
                chunk_index=len(chunks),
                start_char=0,
                end_char=len(chunk_text),
                word_count=current_word_count,
                sentence_count=len(current_chunk),
                chunk_strategy=ChunkStrategy.FORMAT_SPECIFIC,
                content_type="text",
                format_source="docx_paragraphs"
            )
            
            chunk = DocumentChunk(
                content=chunk_text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_hierarchical(self, text: str, file_format: str, 
                                structured_content: Optional[Dict[str, Any]], 
                                chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Advanced hierarchical chunking with parent-child relationships"""
        
        # First, create base chunks using appropriate strategy
        base_chunks = await self._chunk_paragraph_aware(text, chunk_size, overlap_size)
        
        # Create hierarchy levels
        if len(base_chunks) <= 3:
            # Too few chunks for hierarchy
            return base_chunks
        
        # Level 1: Large sections (combine 3-5 base chunks)
        level1_chunks = []
        level1_size = chunk_size * 3
        
        current_section = []
        current_section_word_count = 0
        
        for chunk in base_chunks:
            chunk_word_count = chunk.metadata.word_count
            
            if current_section_word_count + chunk_word_count > level1_size and current_section:
                # Create level 1 chunk
                section_content = '\n\n'.join([c.content for c in current_section])
                section_metadata = ChunkMetadata(
                    chunk_id=self._generate_chunk_id(len(level1_chunks), section_content),
                    chunk_index=len(level1_chunks),
                    start_char=current_section[0].metadata.start_char,
                    end_char=current_section[-1].metadata.end_char,
                    word_count=current_section_word_count,
                    sentence_count=sum(c.metadata.sentence_count for c in current_section),
                    hierarchy_level=1,
                    child_chunk_ids=[c.metadata.chunk_id for c in current_section],
                    chunk_strategy=ChunkStrategy.HIERARCHICAL
                )
                
                level1_chunk = DocumentChunk(
                    content=section_content,
                    metadata=section_metadata
                )
                level1_chunks.append(level1_chunk)
                
                # Update child chunks with parent reference
                for child_chunk in current_section:
                    child_chunk.metadata.parent_chunk_id = section_metadata.chunk_id
                    child_chunk.metadata.hierarchy_level = 0
                
                current_section = []
                current_section_word_count = 0
            
            current_section.append(chunk)
            current_section_word_count += chunk_word_count
        
        # Add final section
        if current_section:
            section_content = '\n\n'.join([c.content for c in current_section])
            section_metadata = ChunkMetadata(
                chunk_id=self._generate_chunk_id(len(level1_chunks), section_content),
                chunk_index=len(level1_chunks),
                start_char=current_section[0].metadata.start_char,
                end_char=current_section[-1].metadata.end_char,
                word_count=current_section_word_count,
                sentence_count=sum(c.metadata.sentence_count for c in current_section),
                hierarchy_level=1,
                child_chunk_ids=[c.metadata.chunk_id for c in current_section],
                chunk_strategy=ChunkStrategy.HIERARCHICAL
            )
            
            level1_chunk = DocumentChunk(
                content=section_content,
                metadata=section_metadata
            )
            level1_chunks.append(level1_chunk)
            
            # Update child chunks
            for child_chunk in current_section:
                child_chunk.metadata.parent_chunk_id = section_metadata.chunk_id
                child_chunk.metadata.hierarchy_level = 0
        
        # Combine all chunks (level 1 + base level)
        all_chunks = level1_chunks + base_chunks
        
        # Re-index all chunks
        for i, chunk in enumerate(all_chunks):
            chunk.metadata.chunk_index = i
        
        return all_chunks
    
    async def _chunk_semantic(self, text: str, chunk_size: int, overlap_size: int) -> List[DocumentChunk]:
        """Semantic similarity-based chunking (simplified implementation)"""
        # For now, fallback to sentence-aware chunking
        # This would require embedding computation for true semantic chunking
        logger.debug("Semantic chunking not fully implemented, using sentence-aware fallback")
        return await self._chunk_sentence_aware(text, chunk_size, overlap_size)
    
    async def _post_process_chunks(self, chunks: List[DocumentChunk], 
                                 file_format: str, metadata: Optional[Dict[str, Any]]) -> List[DocumentChunk]:
        """Post-process chunks with quality assessment and optimization"""
        
        processed_chunks = []
        
        for chunk in chunks:
            # Calculate content quality
            chunk.content_quality = await self._assess_chunk_quality(chunk.content)
            
            # Calculate chunk coherence (simplified)
            chunk.chunk_coherence = self._calculate_chunk_coherence(chunk.content)
            
            # Skip very low-quality chunks
            if chunk.content_quality < self.quality_threshold:
                logger.debug(f"Skipping low-quality chunk: {chunk.content_quality:.2f}")
                continue
            
            # Calculate information density
            chunk.metadata.information_density = self._calculate_information_density(chunk.content)
            
            # Add to processed chunks
            processed_chunks.append(chunk)
        
        logger.debug(f"Post-processing: {len(chunks)} → {len(processed_chunks)} chunks")
        
        return processed_chunks
    
    async def _assess_chunk_quality(self, content: str) -> float:
        """Assess quality of chunk content"""
        if not content.strip():
            return 0.0
        
        quality_factors = []
        
        # Length factor
        length_score = min(1.0, len(content) / 200)  # Normalize to 200 chars
        quality_factors.append(length_score * 0.2)
        
        # Word diversity
        words = content.lower().split()
        unique_words = len(set(words))
        diversity_score = unique_words / len(words) if words else 0
        quality_factors.append(diversity_score * 0.3)
        
        # Sentence structure
        sentences = self._count_sentences(content)
        sentence_score = min(1.0, sentences / 3)  # Normalize to 3 sentences
        quality_factors.append(sentence_score * 0.2)
        
        # Character variety
        unique_chars = len(set(content.lower()))
        char_score = min(1.0, unique_chars / 20)  # Normalize to 20 unique chars
        quality_factors.append(char_score * 0.3)
        
        return sum(quality_factors)
    
    def _calculate_chunk_coherence(self, content: str) -> float:
        """Calculate semantic coherence of chunk (simplified)"""
        # Simplified coherence based on repetition and structure
        words = content.lower().split()
        if len(words) < 5:
            return 0.5
        
        # Check for repeated patterns
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Higher coherence if there are some repeated key terms
        repeated_words = sum(1 for count in word_freq.values() if count > 1)
        coherence_score = min(1.0, repeated_words / len(words) * 3)
        
        return coherence_score
    
    def _calculate_information_density(self, content: str) -> float:
        """Calculate information density of content"""
        words = content.split()
        if not words:
            return 0.0
        
        # Simple heuristic: longer words = higher information density
        avg_word_length = sum(len(word) for word in words) / len(words)
        density_score = min(1.0, avg_word_length / 6)  # Normalize to 6 chars avg
        
        return density_score
    
    def _create_chunk_from_paragraphs(self, paragraphs: List[str], index: int, 
                                    char_position: int, strategy: ChunkStrategy) -> DocumentChunk:
        """Create chunk from list of paragraphs"""
        chunk_text = '\n\n'.join(paragraphs)
        word_count = sum(len(p.split()) for p in paragraphs)
        
        chunk_metadata = ChunkMetadata(
            chunk_id=self._generate_chunk_id(index, chunk_text),
            chunk_index=index,
            start_char=char_position,
            end_char=char_position + len(chunk_text),
            word_count=word_count,
            sentence_count=len(paragraphs),
            chunk_strategy=strategy
        )
        
        return DocumentChunk(
            content=chunk_text,
            metadata=chunk_metadata
        )
    
    def _build_chunk_tree(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Build hierarchical tree structure from chunks"""
        tree = {"root": {"children": [], "chunks": []}}
        
        level_nodes = {}
        
        for chunk in chunks:
            level = chunk.metadata.hierarchy_level
            
            if level not in level_nodes:
                level_nodes[level] = []
            
            node = {
                "chunk_id": chunk.metadata.chunk_id,
                "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                "word_count": chunk.metadata.word_count,
                "children": [],
                "parent": chunk.metadata.parent_chunk_id
            }
            
            level_nodes[level].append(node)
        
        # Build tree structure
        for level in sorted(level_nodes.keys(), reverse=True):
            nodes = level_nodes[level]
            
            if level == 0:
                # Leaf nodes
                tree["root"]["children"].extend(nodes)
            else:
                # Parent nodes
                for node in nodes:
                    # Find children
                    child_ids = [c.metadata.chunk_id for c in chunks 
                               if c.metadata.parent_chunk_id == node["chunk_id"]]
                    
                    for child_id in child_ids:
                        # Find child node
                        for child_level_nodes in level_nodes.values():
                            for child_node in child_level_nodes:
                                if child_node["chunk_id"] == child_id:
                                    node["children"].append(child_node)
        
        return tree
    
    def _calculate_hierarchy_levels(self, chunks: List[DocumentChunk]) -> int:
        """Calculate maximum hierarchy levels in chunks"""
        if not chunks:
            return 0
        
        return max(chunk.metadata.hierarchy_level for chunk in chunks) + 1
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        if self.nltk_available:
            try:
                return len(sent_tokenize(text))
            except:
                pass
        
        # Fallback counting
        return len(re.findall(r'[.!?]+', text))
    
    def _generate_document_hash(self, document_url: str, text: str) -> str:
        """Generate unique hash for document"""
        content = f"{document_url}:{text[:1000]}"  # Use first 1000 chars
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_chunk_id(self, index: int, content: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"chunk_{index}_{content_hash}"
    
    def _update_stats(self, outcome: str):
        """Update chunking statistics"""
        if outcome == "failed":
            self.stats['failed_operations'] += 1
    
    def _update_avg_chunking_time(self, chunking_time: float):
        """Update average chunking time"""
        successful_ops = self.stats['successful_operations']
        current_avg = self.stats['avg_chunking_time']
        
        self.stats['avg_chunking_time'] = (
            (current_avg * (successful_ops - 1) + chunking_time) / successful_ops
        )
    
    def _update_format_stats(self, file_format: str):
        """Update format processing statistics"""
        if file_format not in self.stats['format_processing']:
            self.stats['format_processing'][file_format] = 0
        self.stats['format_processing'][file_format] += 1
    
    async def get_chunking_stats(self) -> Dict[str, Any]:
        """Get comprehensive chunking statistics"""
        return {
            **self.stats,
            "configuration": {
                "default_chunk_size": self.default_chunk_size,
                "default_overlap_size": self.default_overlap_size,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
                "hierarchical_enabled": self.enable_hierarchical,
                "semantic_enabled": self.enable_semantic_chunking,
                "quality_threshold": self.quality_threshold,
                "nltk_available": self.nltk_available
            },
            "supported_formats": list(self.FORMAT_CONFIGS.keys()),
            "supported_strategies": [strategy.value for strategy in ChunkStrategy],
            "format_configs": self.FORMAT_CONFIGS
        }


# Global chunker instance
hierarchical_chunker: Optional[HierarchicalChunker] = None


def get_hierarchical_chunker(**kwargs) -> HierarchicalChunker:
    """Get or create global hierarchical chunker instance"""
    global hierarchical_chunker
    
    if hierarchical_chunker is None:
        hierarchical_chunker = HierarchicalChunker(**kwargs)
    
    return hierarchical_chunker


async def initialize_hierarchical_chunker(**kwargs) -> HierarchicalChunker:
    """Initialize and return hierarchical chunker"""
    chunker = get_hierarchical_chunker(**kwargs)
    
    # Log initialization summary
    stats = await chunker.get_chunking_stats()
    logger.info("🔪 Hierarchical Chunker Summary:")
    logger.info(f"  Default chunk size: {stats['configuration']['default_chunk_size']}")
    logger.info(f"  Hierarchical: {stats['configuration']['hierarchical_enabled']}")
    logger.info(f"  Semantic: {stats['configuration']['semantic_enabled']}")
    logger.info(f"  Supported formats: {len(stats['supported_formats'])}")
    logger.info(f"  NLTK available: {stats['configuration']['nltk_available']}")
    
    return chunker