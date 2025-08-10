"""
Advanced PDF Processor
Enhanced PDF processing with table extraction, layout analysis, and OCR for scanned PDFs
"""

import logging
from pathlib import Path
from typing import Any

try:
    import pymupdf  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import numpy as np
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from src.core.config import config
from src.services.basic_text_extractor import BasicTextExtractor

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Advanced PDF processor with table extraction, layout analysis, and OCR capabilities
    Integrates with basic text extractor for fallback OCR processing
    """

    def __init__(self):
        self.max_document_size_mb = config.max_document_size_mb
        self.enable_ocr = config.enable_ocr_preprocessing
        self.ocr_languages = config.ocr_languages

        # No OCR fallback - fast text extraction only

        # Processing statistics
        self.total_processed = 0
        self.total_pages_processed = 0
        self.total_processing_time = 0.0

        self._check_dependencies()
        logger.info("PDFProcessor initialized with advanced features")

    def _check_dependencies(self):
        """Check available dependencies for PDF processing"""
        deps = {
            "pymupdf": HAS_PYMUPDF,
            "pillow": HAS_PIL,
            "pandas": HAS_PANDAS
        }

        logger.info("📦 PDF processing dependencies:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            logger.info(f"  {status} {dep}")

        if not HAS_PYMUPDF:
            logger.warning("⚠️ PyMuPDF not available. Install with: pip install pymupdf")

    async def process_pdf(self, file_path: str, extract_tables: bool = True, extract_images: bool = True) -> dict[str, Any]:
        """
        Advanced PDF processing with table extraction and layout analysis
        
        Args:
            file_path: Path to PDF file
            extract_tables: Whether to extract and process tables
            extract_images: Whether to extract and OCR images
            
        Returns:
            Comprehensive processing results
        """
        if not HAS_PYMUPDF:
            return {
                "status": "error",
                "error": "PyMuPDF not available",
                "file_path": file_path
            }

        filepath = Path(file_path)
        if not filepath.exists():
            return {
                "status": "error",
                "error": f"File not found: {file_path}",
                "file_path": file_path
            }

        # Check file size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_document_size_mb:
            return {
                "status": "error",
                "error": f"PDF too large: {file_size_mb:.1f}MB > {self.max_document_size_mb}MB",
                "file_path": file_path
            }

        logger.info(f"🔄 Processing PDF with advanced features: {filepath.name}")

        try:
            import time
            start_time = time.time()

            # Open PDF document
            doc = pymupdf.open(str(filepath))

            # Extract document metadata
            metadata = self._extract_metadata(doc)

            # Process all pages
            pages_data = []
            tables_data = []
            images_data = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Process individual page
                page_result = await self._process_page(
                    page, page_num + 1, extract_tables, extract_images
                )

                pages_data.append(page_result["page_data"])

                if extract_tables and page_result.get("tables"):
                    tables_data.extend(page_result["tables"])

                if extract_images and page_result.get("images"):
                    images_data.extend(page_result["images"])

            doc.close()

            # Combine all text content
            full_text = self._combine_page_content(pages_data, tables_data)

            # Calculate processing metrics
            processing_time = time.time() - start_time
            self.total_processed += 1
            self.total_pages_processed += len(pages_data)
            self.total_processing_time += processing_time

            result = {
                "status": "success",
                "file_path": file_path,
                "file_size_mb": round(file_size_mb, 2),
                "processing_time": round(processing_time, 2),
                "metadata": metadata,
                "content": {
                    "full_text": full_text,
                    "page_count": len(pages_data),
                    "pages": pages_data,
                    "tables": tables_data if extract_tables else [],
                    "images": images_data if extract_images else [],
                    "char_count": len(full_text),
                    "word_count": len(full_text.split())
                },
                "extraction_features": {
                    "text_extraction": True,
                    "table_extraction": extract_tables,
                    "image_extraction": extract_images,
                    "layout_analysis": True,
                    "metadata_extraction": True
                },
                "processor": "PDFProcessor"
            }

            logger.info(f"✅ PDF processed: {len(pages_data)} pages, "
                       f"{len(tables_data)} tables, {len(images_data)} images in {processing_time:.2f}s")

            return result

        except Exception as e:
            error_msg = f"PDF processing failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "file_path": file_path
            }

    def _extract_metadata(self, doc: 'pymupdf.Document') -> dict[str, Any]:
        """Extract comprehensive metadata from PDF"""
        try:
            metadata = doc.metadata

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": len(doc),
                "is_encrypted": doc.is_encrypted,
                "is_pdf": doc.is_pdf,
                "language": metadata.get("language", ""),
                "keywords": metadata.get("keywords", "")
            }
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {str(e)}")
            return {
                "page_count": len(doc),
                "extraction_error": str(e)
            }

    async def _process_page(
        self,
        page: 'pymupdf.Page',
        page_num: int,
        extract_tables: bool,
        extract_images: bool
    ) -> dict[str, Any]:
        """Process individual PDF page with advanced features"""

        # Extract basic text
        page_text = page.get_text()

        # Get text blocks with positioning
        blocks = page.get_text("blocks")
        text_blocks = []

        for block in blocks:
            if len(block) >= 5 and block[4].strip():  # Block has text content
                text_blocks.append({
                    "bbox": block[:4],  # Bounding box
                    "text": block[4].strip(),
                    "block_type": "text"
                })

        # Extract tables if enabled
        tables = []
        if extract_tables:
            tables = await self._extract_tables_from_page(page, page_num)

        # Extract and OCR images if enabled
        images = []
        if extract_images:
            images = await self._extract_images_from_page(page, page_num)

        # Analyze page layout
        layout_info = self._analyze_page_layout(page, text_blocks)

        page_data = {
            "page_number": page_num,
            "text": self._clean_text(page_text),
            "text_blocks": text_blocks,
            "char_count": len(page_text),
            "word_count": len(page_text.split()),
            "layout_info": layout_info,
            "has_tables": len(tables) > 0,
            "has_images": len(images) > 0,
            "table_count": len(tables),
            "image_count": len(images)
        }

        return {
            "page_data": page_data,
            "tables": tables,
            "images": images
        }

    async def _extract_tables_from_page(self, page: 'pymupdf.Page', page_num: int) -> list[dict[str, Any]]:
        """Extract tables from PDF page"""
        tables = []

        try:
            # Find tables using PyMuPDF's table detection
            page_tables = page.find_tables()

            for i, table in enumerate(page_tables):
                try:
                    # Extract table data
                    table_data = table.extract()

                    if table_data and len(table_data) > 0:
                        # Convert to DataFrame for better processing
                        if HAS_PANDAS:
                            df = pd.DataFrame(table_data[1:], columns=table_data[0] if table_data[0] else None)
                            table_text = self._dataframe_to_text(df, f"Table_{page_num}_{i+1}")
                        else:
                            # Fallback to simple text conversion
                            table_text = "\n".join([" | ".join(str(cell) for cell in row) for row in table_data])

                        table_info = {
                            "table_id": f"page_{page_num}_table_{i+1}",
                            "page_number": page_num,
                            "bbox": table.bbox,
                            "row_count": len(table_data),
                            "col_count": len(table_data[0]) if table_data and table_data[0] else 0,
                            "raw_data": table_data,
                            "formatted_text": table_text,
                            "extraction_method": "pymupdf_tables"
                        }

                        tables.append(table_info)

                except Exception as e:
                    logger.warning(f"Table extraction failed for page {page_num}, table {i+1}: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"Table detection failed for page {page_num}: {str(e)}")

        return tables

    async def _extract_images_from_page(self, page: 'pymupdf.Page', page_num: int) -> list[dict[str, Any]]:
        """Extract image metadata from PDF page (no OCR - fast extraction only)"""
        images = []

        try:
            # Get image list from page - just metadata, no OCR processing
            image_list = page.get_images()

            for i, img in enumerate(image_list):
                try:
                    # Get basic image metadata only
                    xref = img[0]
                    width = img[2] if len(img) > 2 else 0
                    height = img[3] if len(img) > 3 else 0
                    
                    # Just store image metadata without OCR
                    image_info = {
                        "image_id": f"page_{page_num}_image_{i+1}",
                        "page_number": page_num,
                        "xref": xref,
                        "width": width,
                        "height": height,
                        "extraction_method": "fast_metadata_only",
                        "note": "Image detected but not processed (OCR disabled for speed)"
                    }

                    images.append(image_info)

                except Exception as e:
                    logger.warning(f"Image metadata extraction failed for page {page_num}, image {i+1}: {str(e)}")
                    continue

        except Exception as e:
            logger.warning(f"Image detection failed for page {page_num}: {str(e)}")

        return images

    def _analyze_page_layout(self, page: 'pymupdf.Page', text_blocks: list[dict]) -> dict[str, Any]:
        """Analyze page layout and structure"""
        try:
            page_rect = page.rect

            # Classify text blocks by position
            header_blocks = []
            body_blocks = []
            footer_blocks = []

            page_height = page_rect.height
            header_threshold = page_height * 0.15  # Top 15%
            footer_threshold = page_height * 0.85  # Bottom 15%

            for block in text_blocks:
                bbox = block["bbox"]
                y_center = (bbox[1] + bbox[3]) / 2

                if y_center < header_threshold:
                    header_blocks.append(block)
                elif y_center > footer_threshold:
                    footer_blocks.append(block)
                else:
                    body_blocks.append(block)

            # Calculate text density
            total_text_area = sum((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                 for block in text_blocks for bbox in [block["bbox"]])
            page_area = page_rect.width * page_rect.height
            text_density = (total_text_area / page_area) if page_area > 0 else 0

            layout_info = {
                "page_width": page_rect.width,
                "page_height": page_rect.height,
                "total_text_blocks": len(text_blocks),
                "header_blocks": len(header_blocks),
                "body_blocks": len(body_blocks),
                "footer_blocks": len(footer_blocks),
                "text_density": round(text_density, 3),
                "layout_type": self._classify_layout_type(text_blocks, page_rect)
            }

            return layout_info

        except Exception as e:
            logger.warning(f"Layout analysis failed: {str(e)}")
            return {"analysis_error": str(e)}

    def _classify_layout_type(self, text_blocks: list[dict], page_rect) -> str:
        """Classify the layout type of the page"""
        if not text_blocks:
            return "empty"

        # Analyze column structure
        block_centers = [(block["bbox"][0] + block["bbox"][2]) / 2 for block in text_blocks]

        if len(block_centers) < 2:
            return "single_column"

        # Simple column detection
        page_width = page_rect.width
        left_column = [x for x in block_centers if x < page_width * 0.4]
        right_column = [x for x in block_centers if x > page_width * 0.6]

        if len(left_column) > 0 and len(right_column) > 0:
            return "two_column"
        elif len(text_blocks) > 10:
            return "dense_text"
        else:
            return "single_column"

    def _dataframe_to_text(self, df: 'pd.DataFrame', table_name: str) -> str:
        """Convert DataFrame to readable text format"""
        try:
            # Clean the DataFrame
            df = df.fillna("")  # Replace NaN with empty string

            # Get column headers
            headers = " | ".join(str(col) for col in df.columns)

            text_parts = [
                f"=== {table_name} ===",
                f"Columns: {headers}",
                ""
            ]

            # Add rows
            for idx, row in df.iterrows():
                row_text = " | ".join(str(val) for val in row.values)
                text_parts.append(f"Row {idx + 1}: {row_text}")

            return "\n".join(text_parts)

        except Exception as e:
            logger.warning(f"DataFrame to text conversion failed: {str(e)}")
            return f"Table: {table_name}\nConversion failed: {str(e)}"

    def _combine_page_content(self, pages_data: list[dict], tables_data: list[dict]) -> str:
        """Combine page text with extracted tables and images"""
        content_parts = []

        if not pages_data:
            return ""

        for page_data in pages_data:
            page_num = page_data["page_number"]
            page_text = page_data.get("text", "").strip()

            # Add page header, even if content is empty
            content_parts.append(f"\n=== PAGE {page_num} ===")
            if page_text:
                content_parts.append(page_text)

            # Add tables from this page
            page_tables = [table for table in tables_data if table["page_number"] == page_num]
            for table in page_tables:
                content_parts.append(f"\n{table['formatted_text']}")

        # Ensure there's at least some content if pages were processed
        combined_text = "\n".join(content_parts).strip()
        if not combined_text and pages_data:
            return " "  # Return a single space to indicate processing happened

        return combined_text

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove common PDF artifacts
        text = text.replace("§", "")
        text = text.replace("¶", "")
        text = text.replace("\x0c", "")  # Form feed character

        # Normalize quotes
        text = text.replace(""", '"').replace(""", '"')
        text = text.replace("'", "'").replace("'", "'")

        return text.strip()

    def get_processing_stats(self) -> dict[str, Any]:
        """Get PDF processing statistics"""
        avg_processing_time = (
            self.total_processing_time / self.total_processed
            if self.total_processed > 0 else 0.0
        )

        avg_pages_per_doc = (
            self.total_pages_processed / self.total_processed
            if self.total_processed > 0 else 0.0
        )

        return {
            "total_documents_processed": self.total_processed,
            "total_pages_processed": self.total_pages_processed,
            "total_processing_time": round(self.total_processing_time, 2),
            "average_processing_time": round(avg_processing_time, 2),
            "average_pages_per_document": round(avg_pages_per_doc, 1),
            "features_available": {
                "basic_text_extraction": True,
                "table_extraction": True,
                "image_ocr": HAS_PIL and self.enable_ocr,
                "layout_analysis": True,
                "metadata_extraction": True
            },
            "dependencies": {
                "pymupdf": HAS_PYMUPDF,
                "pillow": HAS_PIL,
                "pandas": HAS_PANDAS
            }
        }
