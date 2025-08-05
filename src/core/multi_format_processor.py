"""
Multi-Format Document Processor with Fast OCR Support
Supports PDF, DOCX, Excel (XLSX, XLS), Images (JPEG, PNG, JPG)
Optimized for speed with EasyOCR integration
"""

import asyncio
import logging
import io
import os
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
import aiohttp
import pandas as pd
from PIL import Image
import numpy as np

# OCR imports with fallbacks
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None

# Enhanced document processor import
from src.core.enhanced_document_processor import EnhancedDocumentProcessor, DocumentChunk
from src.core.config import settings

logger = logging.getLogger(__name__)


class MultiFormatProcessor(EnhancedDocumentProcessor):
    """
    Enhanced processor supporting multiple formats with fast OCR
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize OCR engine
        self.ocr_reader = None
        if settings.OCR_ENGINE == "easyocr" and EASYOCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(
                    settings.OCR_LANGUAGES,
                    gpu=False,  # Use CPU for better compatibility
                    verbose=False
                )
                logger.info("✅ EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.ocr_reader = None
        
        # Supported formats - only these will be processed
        self.supported_formats = {
            # Document formats
            'pdf': self._process_pdf,
            'docx': self._process_docx,
            'doc': self._process_docx,  # Try docx processor for doc files
            
            # Spreadsheet formats
            'xlsx': self._process_excel,
            'xls': self._process_excel,
            'csv': self._process_csv,
            
            # Image formats
            'jpg': self._process_image,
            'jpeg': self._process_image,
            'png': self._process_image,
            'bmp': self._process_image,
            'tiff': self._process_image,
            'tif': self._process_image,
            
            # Presentation formats (basic support)
            'pptx': self._process_pdf,  # Fallback to PDF processor
            'ppt': self._process_pdf,   # Fallback to PDF processor
        }
        
        logger.info("✅ Multi-format processor initialized")
        logger.info(f"Supported formats: {list(self.supported_formats.keys())}")
        logger.info(f"OCR engine: {settings.OCR_ENGINE}")
        logger.info(f"OCR available: {self.ocr_reader is not None}")

    def _get_file_extension(self, url: str) -> str:
        """Extract file extension from URL"""
        try:
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            extension = os.path.splitext(path)[1].lstrip('.')
            return extension
        except Exception:
            return 'pdf'  # Default fallback

    async def process_document(
        self, document_url: str
    ) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """
        Process document based on file format with extension and size validation
        Falls back to LLM knowledge for unsupported formats
        
        Args:
            document_url: URL to the document

        Returns:
            Tuple of (chunks, metadata)
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract file extension for logging
            file_extension = self._get_file_extension(document_url)
            logger.info(f"🔍 Checking document: {document_url}")
            logger.info(f"📄 Detected file extension: .{file_extension}")
            
            # Check if document is processable (extension + size + type)
            is_processable, reason = await self._is_processable_document(document_url)
            
            if not is_processable:
                logger.warning(f"🚫 Document not processable: {reason}")
                logger.info("🧠 Will use LLM knowledge instead of document processing")
                
                # Return special metadata indicating LLM should use its own knowledge
                metadata = {
                    'source': document_url,
                    'processing_time': asyncio.get_event_loop().time() - start_time,
                    'processor': 'MultiFormatProcessor',
                    'status': 'not_processable',
                    'reason': reason,
                    'file_extension': file_extension,
                    'use_llm_knowledge': True,
                    'chunks_generated': 0,
                    'supported_formats': list(self.supported_formats.keys())
                }
                
                # Return empty chunks - this will trigger LLM knowledge fallback
                return [], metadata
            
            logger.info(f"✅ Document is processable, proceeding with {file_extension.upper()} processing")
            
            # Process based on format
            processor_func = self.supported_formats[file_extension]
            chunks, metadata = await processor_func(document_url)
            
            # Add processing metadata
            processing_time = asyncio.get_event_loop().time() - start_time
            metadata.update({
                'file_format': file_extension,
                'processing_time': processing_time,
                'processor': 'MultiFormatProcessor',
                'chunks_generated': len(chunks),
                'status': 'processed_successfully'
            })
            
            logger.info(f"✅ Processed {file_extension.upper()} document in {processing_time:.2f}s")
            logger.info(f"📊 Generated {len(chunks)} chunks")
            
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"❌ Multi-format processing failed: {str(e)}")
            # Fallback to parent processor
            return await super().process_document(document_url)

    async def _process_pdf(self, document_url: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Process PDF documents using parent processor"""
        return await super().process_document(document_url)

    async def _process_docx(self, document_url: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Process DOCX documents using parent processor"""
        return await super().process_document(document_url)

    async def _process_excel(self, document_url: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Process Excel files (XLSX, XLS)"""
        try:
            logger.info(f"📊 Processing Excel file: {document_url}")
            
            # Download file
            document_data, content_type = await self._download_document(document_url)
            
            # Read Excel file
            excel_file = io.BytesIO(document_data)
            
            # Get all sheet names
            xl_file = pd.ExcelFile(excel_file)
            sheet_names = xl_file.sheet_names[:settings.EXCEL_SHEET_LIMIT]
            
            all_text_content = []
            metadata = {
                'source': document_url,
                'content_type': content_type,
                'total_sheets': len(xl_file.sheet_names),
                'processed_sheets': len(sheet_names),
                'sheet_names': sheet_names
            }
            
            # Process each sheet
            for sheet_name in sheet_names:
                try:
                    logger.info(f"📋 Processing sheet: {sheet_name}")
                    
                    # Read sheet with row limit
                    df = pd.read_excel(
                        excel_file, 
                        sheet_name=sheet_name,
                        nrows=settings.EXCEL_MAX_ROWS
                    )
                    
                    # Convert to text
                    if settings.EXCEL_TEXT_EXTRACTION_MODE == "fast":
                        # Fast mode: just concatenate all values
                        sheet_text = f"Sheet: {sheet_name}\n"
                        sheet_text += df.to_string(index=False, na_rep='')
                    else:
                        # Comprehensive mode: include headers and structure
                        sheet_text = f"Sheet: {sheet_name}\n"
                        sheet_text += f"Columns: {', '.join(df.columns.astype(str))}\n"
                        sheet_text += df.to_string(index=True, na_rep='')
                    
                    all_text_content.append(sheet_text)
                    
                except Exception as e:
                    logger.warning(f"Failed to process sheet {sheet_name}: {e}")
                    continue
            
            # Combine all sheet content
            full_text = "\n\n".join(all_text_content)
            
            # Create chunks
            chunks = await self._create_chunks_from_text(full_text, document_url)
            
            logger.info(f"✅ Processed Excel file with {len(sheet_names)} sheets")
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"❌ Excel processing failed: {str(e)}")
            raise

    async def _process_csv(self, document_url: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Process CSV files"""
        try:
            logger.info(f"📊 Processing CSV file: {document_url}")
            
            # Download file
            document_data, content_type = await self._download_document(document_url)
            
            # Read CSV
            csv_file = io.StringIO(document_data.decode('utf-8'))
            df = pd.read_csv(csv_file, nrows=settings.EXCEL_MAX_ROWS)
            
            # Convert to text
            if settings.EXCEL_TEXT_EXTRACTION_MODE == "fast":
                text_content = df.to_string(index=False, na_rep='')
            else:
                text_content = f"Columns: {', '.join(df.columns.astype(str))}\n"
                text_content += df.to_string(index=True, na_rep='')
            
            # Create chunks
            chunks = await self._create_chunks_from_text(text_content, document_url)
            
            metadata = {
                'source': document_url,
                'content_type': content_type,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            }
            
            logger.info(f"✅ Processed CSV file with {len(df)} rows")
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"❌ CSV processing failed: {str(e)}")
            raise

    async def _process_image(self, document_url: str) -> Tuple[List[DocumentChunk], Dict[str, Any]]:
        """Process image files using OCR"""
        try:
            logger.info(f"🖼️ Processing image file: {document_url}")
            
            if not self.ocr_reader:
                raise Exception("OCR engine not available")
            
            # Download image
            document_data, content_type = await self._download_document(document_url)
            
            # Check file size
            if len(document_data) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
                raise Exception(f"Image too large: {len(document_data) / (1024*1024):.1f}MB")
            
            # Load image
            image = Image.open(io.BytesIO(document_data))
            
            # Preprocess image if enabled
            if settings.ENABLE_OCR_PREPROCESSING:
                image = await self._preprocess_image(image)
            
            # Convert to numpy array for EasyOCR
            image_array = np.array(image)
            
            # Perform OCR
            logger.info("🔍 Performing OCR extraction...")
            ocr_results = self.ocr_reader.readtext(image_array)
            
            # Extract text
            extracted_text = []
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5:  # Filter low-confidence results
                    extracted_text.append(text)
            
            full_text = "\n".join(extracted_text)
            
            if not full_text.strip():
                logger.warning("No text extracted from image")
                full_text = "No readable text found in this image."
            
            # Create chunks
            chunks = await self._create_chunks_from_text(full_text, document_url)
            
            metadata = {
                'source': document_url,
                'content_type': content_type,
                'image_size': image.size,
                'ocr_engine': settings.OCR_ENGINE,
                'text_blocks_found': len(ocr_results),
                'high_confidence_blocks': len([r for r in ocr_results if r[2] > 0.5])
            }
            
            logger.info(f"✅ Extracted text from image: {len(full_text)} characters")
            return chunks, metadata
            
        except Exception as e:
            logger.error(f"❌ Image processing failed: {str(e)}")
            raise

    async def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large (for speed)
            max_size = 2000
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image

    async def _check_file_size(self, document_url: str) -> Tuple[bool, int]:
        """
        Check file size without downloading the entire file
        Returns (is_valid_size, size_in_mb)
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Use HEAD request to get file size without downloading
                async with session.head(document_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        content_length = response.headers.get('content-length')
                        if content_length:
                            size_bytes = int(content_length)
                            size_mb = size_bytes / (1024 * 1024)
                            
                            logger.info(f"📏 File size: {size_mb:.2f} MB")
                            
                            # Check against limit from settings
                            if size_mb > settings.MAX_FILE_SIZE_MB:
                                logger.warning(f"⚠️ File too large: {size_mb:.2f} MB (limit: {settings.MAX_FILE_SIZE_MB} MB)")
                                return False, int(size_mb)
                            
                            return True, int(size_mb)
                        else:
                            logger.warning("⚠️ Could not determine file size from headers")
                            return True, 0  # Allow if we can't determine size
                    else:
                        logger.warning(f"⚠️ HEAD request failed with status: {response.status}")
                        return True, 0  # Allow if HEAD request fails
                        
        except Exception as e:
            logger.warning(f"⚠️ Error checking file size: {str(e)}")
            return True, 0  # Allow if size check fails

    async def _is_processable_document(self, document_url: str) -> Tuple[bool, str]:
        """
        Check if the URL points to a processable document
        Returns (is_processable, reason)
        """
        try:
            # Check file extension FIRST (before any download)
            file_extension = self._get_file_extension(document_url)
            
            # Check if it's a supported format
            if file_extension not in self.supported_formats:
                return False, f"Unsupported file format: .{file_extension} (supported: {', '.join(self.supported_formats.keys())})"
            
            # Check if it's a known non-document extension
            non_document_extensions = {
                'bin', 'exe', 'zip', 'rar', '7z', 'tar', 'gz',
                'mp4', 'avi', 'mkv', 'mov', 'mp3', 'wav', 'flac',
                'iso', 'img', 'dmg', 'deb', 'rpm', 'msi', 'app'
            }
            
            if file_extension in non_document_extensions:
                return False, f"Non-document file type: .{file_extension}"
            
            # Check if URL looks like a speed test or download test
            url_lower = document_url.lower()
            test_indicators = ['speed', 'test', 'download', 'benchmark', 'gb.bin', 'mb.bin', '10gb', '1gb', '100mb']
            
            for indicator in test_indicators:
                if indicator in url_lower:
                    return False, f"Detected test/benchmark file: contains '{indicator}'"
            
            # Only check file size if extension is supported (to avoid unnecessary downloads)
            is_valid_size, size_mb = await self._check_file_size(document_url)
            if not is_valid_size:
                return False, f"File too large: {size_mb} MB (limit: {settings.MAX_FILE_SIZE_MB} MB)"
            
            return True, "Document appears processable"
            
        except Exception as e:
            logger.error(f"Error checking if document is processable: {str(e)}")
            return True, "Could not verify - allowing processing"

    async def _create_chunks_from_text(
        self, text: str, source_url: str
    ) -> List[DocumentChunk]:
        """Create document chunks from extracted text"""
        try:
            # Simple chunking for speed
            chunk_size = settings.MAX_CHUNK_SIZE
            overlap = settings.CHUNK_OVERLAP
            
            chunks = []
            text_length = len(text)
            
            for i in range(0, text_length, chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                
                if chunk_text.strip():
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata={
                            'source': source_url,
                            'chunk_index': len(chunks),
                            'start_char': i,
                            'end_char': min(i + chunk_size, text_length)
                        }
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Chunk creation failed: {str(e)}")
            return []

    async def _create_chunks_from_text(
        self, text: str, source_url: str
    ) -> List[DocumentChunk]:
        """Create document chunks from extracted text"""
        try:
            # Simple chunking for speed
            chunk_size = settings.MAX_CHUNK_SIZE
            overlap = settings.CHUNK_OVERLAP
            
            chunks = []
            text_length = len(text)
            
            for i in range(0, text_length, chunk_size - overlap):
                chunk_text = text[i:i + chunk_size]
                
                if chunk_text.strip():
                    chunk = DocumentChunk(
                        content=chunk_text,
                        metadata={
                            'source': source_url,
                            'chunk_index': len(chunks),
                            'start_char': i,
                            'end_char': min(i + chunk_size, text_length)
                        }
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Chunk creation failed: {str(e)}")
            return []

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing capabilities and statistics"""
        return {
            'supported_formats': list(self.supported_formats.keys()),
            'ocr_available': self.ocr_reader is not None,
            'ocr_engine': settings.OCR_ENGINE,
            'ocr_languages': settings.OCR_LANGUAGES,
            'excel_max_rows': settings.EXCEL_MAX_ROWS,
            'excel_sheet_limit': settings.EXCEL_SHEET_LIMIT,
            'max_image_size_mb': settings.MAX_IMAGE_SIZE_MB,
            'preprocessing_enabled': settings.ENABLE_OCR_PREPROCESSING,
            'processor_type': 'MultiFormatProcessor',
            'fallback_available': True
        }

    def is_format_supported(self, file_extension: str) -> bool:
        """Check if file format is supported"""
        return file_extension.lower() in self.supported_formats
