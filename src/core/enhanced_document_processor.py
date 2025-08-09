"""
Enhanced document processor using LlamaIndex readers
Provides better document processing with fallback to existing processor
"""

import logging
from typing import List, Dict, Any, Tuple
from io import BytesIO

try:
    from llama_index.readers.file import PDFReader, DocxReader
    from llama_index.core.node_parser import SentenceWindowNodeParser, SimpleNodeParser
    from llama_index.core.schema import Document as LlamaDocument
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    try:
        # Fallback to older import structure
        from llama_index.readers import PDFReader, DocxReader
        from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser
        from llama_index.schema import Document as LlamaDocument
        LLAMA_INDEX_AVAILABLE = True
    except ImportError:
        LLAMA_INDEX_AVAILABLE = False
        PDFReader = None
        DocxReader = None
        SentenceWindowNodeParser = None
        SimpleNodeParser = None
        LlamaDocument = None

from src.core.document_processor import DocumentProcessor, DocumentChunk
from src.core.config import settings

logger = logging.getLogger(__name__)


class EnhancedDocumentProcessor(DocumentProcessor):
    """
    Enhanced processor with LlamaIndex readers
    Maintains backward compatibility with existing processor
    """
    
    def __init__(self):
        super().__init__()
        
        if LLAMA_INDEX_AVAILABLE:
            # Initialize LlamaIndex readers
            self.pdf_reader = PDFReader()
            self.docx_reader = DocxReader()
            
            # Initialize node parser for better chunking
            self.node_parser = SentenceWindowNodeParser.from_defaults(
                window_size=5,  # Larger context window for large docs
                window_metadata_key="window",
                original_text_metadata_key="original_text",
                sentence_splitter=None  # Use default sentence splitter
            )
            
            # Fallback simple parser with larger chunks for large docs
            self.simple_parser = SimpleNodeParser.from_defaults(
                chunk_size=min(settings.MAX_CHUNK_SIZE * 2, 2000),  # Larger chunks
                chunk_overlap=settings.CHUNK_OVERLAP * 2  # More overlap
            )
            
            logger.info("Enhanced document processor initialized with LlamaIndex")
        else:
            logger.warning("LlamaIndex not available, using fallback processor only")

    async def process_document(
        self, document_url: str
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Enhanced processing with fallback to existing processor
        
        Args:
            document_url: URL to the document

        Returns:
            Tuple of (chunks, metadata)
        """
        if not LLAMA_INDEX_AVAILABLE:
            logger.info("Using fallback document processor")
            return await super().process_document(document_url)
        
        try:
            logger.info(f"Processing document with enhanced processor: {document_url}")
            return await self._process_with_llamaindex(document_url)
            
        except Exception as e:
            logger.warning(f"LlamaIndex processing failed, using fallback: {str(e)}")
            # Fallback to existing processor
            return await super().process_document(document_url)

    async def _process_with_llamaindex(
        self, document_url: str
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Process document using LlamaIndex readers"""
        
        # Download document
        document_data, content_type = await self._download_document(document_url)
        
        # Create temporary file-like object
        document_stream = BytesIO(document_data)
        
        # Process based on content type
        if "pdf" in content_type.lower():
            documents = await self._process_pdf_with_llamaindex(document_stream, document_url)
        elif "word" in content_type.lower() or "docx" in content_type.lower():
            documents = await self._process_docx_with_llamaindex(document_stream, document_url)
        else:
            # Fallback to text processing
            text = document_data.decode("utf-8", errors="ignore")
            documents = [LlamaDocument(text=text, metadata={"source": document_url})]
        
        # Parse documents into nodes (chunks)
        logger.info(f"Parsing {len(documents)} documents into chunks...")
        
        try:
            # Try advanced sentence window parsing first
            nodes = self.node_parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} chunks with sentence window parser")
        except Exception as e:
            logger.warning(f"Sentence window parsing failed, using simple parser: {str(e)}")
            # Fallback to simple parsing
            nodes = self.simple_parser.get_nodes_from_documents(documents)
            logger.info(f"Created {len(nodes)} chunks with simple parser")
        
        # Convert LlamaIndex nodes to DocumentChunk objects
        chunks = []
        for i, node in enumerate(nodes):
            chunk = DocumentChunk(
                text=node.text,
                page_num=node.metadata.get("page_num", 0),
                chunk_id=f"enhanced_chunk_{i}",
                metadata={
                    "source_url": document_url,
                    "node_id": node.node_id,
                    "doc_type": "enhanced",
                    **node.metadata
                }
            )
            chunks.append(chunk)
        
        # Create metadata
        metadata = {
            "type": "enhanced_processing",
            "source_url": document_url,
            "content_type": content_type,
            "size": len(document_data),
            "num_documents": len(documents),
            "num_chunks": len(chunks),
            "processor": "llamaindex"
        }
        
        logger.info(f"Enhanced processing completed: {len(chunks)} chunks created")
        return chunks, metadata

    async def _process_pdf_with_llamaindex(
        self, document_stream: BytesIO, document_url: str
    ) -> List[LlamaDocument]:
        """Process PDF using LlamaIndex PDFReader"""
        try:
            # Save stream to temporary file (LlamaIndex readers expect file paths)
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(document_stream.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Use LlamaIndex PDF reader
                documents = self.pdf_reader.load_data(tmp_file_path)
                
                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = document_url
                    doc.metadata["file_type"] = "pdf"
                
                logger.info(f"Loaded {len(documents)} PDF documents with LlamaIndex")
                return documents
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"LlamaIndex PDF processing failed: {str(e)}")
            raise

    async def _process_docx_with_llamaindex(
        self, document_stream: BytesIO, document_url: str
    ) -> List[LlamaDocument]:
        """Process DOCX using LlamaIndex DocxReader"""
        try:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(document_stream.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Use LlamaIndex DOCX reader
                documents = self.docx_reader.load_data(tmp_file_path)
                
                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = document_url
                    doc.metadata["file_type"] = "docx"
                
                logger.info(f"Loaded {len(documents)} DOCX documents with LlamaIndex")
                return documents
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"LlamaIndex DOCX processing failed: {str(e)}")
            raise

    def is_enhanced_available(self) -> bool:
        """Check if enhanced processing is available"""
        return LLAMA_INDEX_AVAILABLE

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            "enhanced_available": LLAMA_INDEX_AVAILABLE,
            "max_chunk_size": self.max_chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
        
        if LLAMA_INDEX_AVAILABLE:
            stats.update({
                "sentence_window_available": True,
                "pdf_reader_available": self.pdf_reader is not None,
                "docx_reader_available": self.docx_reader is not None,
            })
        
        return stats
