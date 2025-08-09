"""
Basic Text Extraction Service
Handles text extraction from multiple document formats: PDF, DOCX, XLSX, Images
"""

import logging
from pathlib import Path
from typing import Any, Union

# PDF processing
try:
    import pymupdf  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Office document processing
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Image processing and OCR
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageEnhance
    HAS_IMAGE_PROCESSING = True
except ImportError:
    HAS_IMAGE_PROCESSING = False

try:
    import easyocr
    HAS_EASYOCR = True
except (ImportError, ValueError, Exception) as e:
    # Handle both import errors and numpy compatibility issues
    HAS_EASYOCR = False
    easyocr = None

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

from src.core.config import config

logger = logging.getLogger(__name__)


class BasicTextExtractor:
    """
    Basic text extraction service for multiple document formats
    Supports PDF, DOCX, XLSX, and images with OCR
    """

    def __init__(self):
        self.supported_formats = config.supported_formats
        self.ocr_engine = config.ocr_engine
        self.ocr_languages = config.ocr_languages
        self.max_image_size_mb = config.max_image_size_mb
        self.enable_ocr_preprocessing = config.enable_ocr_preprocessing

        # Initialize OCR readers
        self.easyocr_reader = None

        # Check available dependencies
        self._check_dependencies()

        logger.info(f"BasicTextExtractor initialized with OCR engine: {self.ocr_engine}")

    def _check_dependencies(self):
        """Check and log available dependencies"""
        deps_status = {
            "pymupdf": HAS_PYMUPDF,
            "python-docx": HAS_DOCX,
            "pandas": HAS_PANDAS,
            "image_processing": HAS_IMAGE_PROCESSING,
            "easyocr": HAS_EASYOCR,
            "tesseract": HAS_TESSERACT
        }

        logger.info("📦 Text extraction dependencies:")
        for dep, available in deps_status.items():
            status = "✅" if available else "❌"
            logger.info(f"  {status} {dep}")

        # Initialize OCR engine
        if self.ocr_engine == "tesseract":
            if HAS_TESSERACT:
                logger.info("✅ Using Tesseract OCR with Malayalam + English support")
            else:
                logger.warning("❌ Tesseract not available but required")
        else:
            logger.warning(f"❌ Unsupported OCR engine: {self.ocr_engine}")
        
        # EasyOCR initialization (legacy support if available)
        if HAS_EASYOCR:
            try:
                self.easyocr_reader = easyocr.Reader(["en"], gpu=False)
                logger.info("✅ EasyOCR available as fallback (English only)")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {str(e)}")
                self.easyocr_reader = None

    async def extract_text(self, file_path: str, file_type: Union[str, None] = None) -> dict[str, Any]:
        """
        Extract text from document based on file type
        
        Args:
            file_path: Path to the document file
            file_type: Optional file type hint
            
        Returns:
            Dictionary with extraction results
        """
        filepath = Path(file_path)

        if not filepath.exists():
            return {
                "status": "error",
                "error": f"File not found: {file_path}",
                "file_path": file_path
            }

        # Detect file type if not provided
        if not file_type:
            file_type = filepath.suffix.lower().lstrip('.')

        logger.info(f"🔄 Extracting text from {file_type.upper()} file: {filepath.name}")

        try:
            # Route to appropriate extractor
            if file_type == 'pdf':
                return await self._extract_pdf_text(filepath)
            elif file_type in ['docx', 'doc']:
                return await self._extract_docx_text(filepath)
            elif file_type in ['xlsx', 'xls', 'csv']:
                return await self._extract_spreadsheet_text(filepath)
            elif file_type in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp']:
                return await self._extract_image_text(filepath)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported file type: {file_type}",
                    "file_path": file_path
                }

        except Exception as e:
            logger.error(f"❌ Text extraction failed for {filepath.name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path,
                "file_type": file_type
            }

    async def _extract_pdf_text(self, filepath: Path) -> dict[str, Any]:
        """Extract text from PDF using PyMuPDF"""
        if not HAS_PYMUPDF:
            return {
                "status": "error",
                "error": "PyMuPDF not available. Install with: pip install pymupdf"
            }

        try:
            doc = pymupdf.open(str(filepath))
            pages_text = []
            total_chars = 0

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()

                # Clean and normalize text
                page_text = self._clean_text(page_text)

                pages_text.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
                total_chars += len(page_text)

            doc.close()

            # Combine all pages
            full_text = "\n\n".join([page["text"] for page in pages_text])

            return {
                "status": "success",
                "text": full_text,
                "page_count": len(pages_text),
                "pages": pages_text,
                "char_count": total_chars,
                "word_count": len(full_text.split()),
                "extraction_method": "pymupdf",
                "metadata": {
                    "file_type": "pdf",
                    "extractor": "BasicTextExtractor"
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"PDF extraction failed: {str(e)}"
            }

    async def _extract_docx_text(self, filepath: Path) -> dict[str, Any]:
        """Extract text from DOCX files"""
        if not HAS_DOCX:
            return {
                "status": "error",
                "error": "python-docx not available. Install with: pip install python-docx"
            }

        try:
            doc = DocxDocument(str(filepath))
            paragraphs_text = []
            total_chars = 0

            for i, paragraph in enumerate(doc.paragraphs):
                para_text = paragraph.text.strip()
                if para_text:  # Skip empty paragraphs
                    cleaned_text = self._clean_text(para_text)
                    paragraphs_text.append({
                        "paragraph_number": i + 1,
                        "text": cleaned_text,
                        "char_count": len(cleaned_text)
                    })
                    total_chars += len(cleaned_text)

            # Combine all paragraphs
            full_text = "\n\n".join([para["text"] for para in paragraphs_text])

            return {
                "status": "success",
                "text": full_text,
                "paragraph_count": len(paragraphs_text),
                "paragraphs": paragraphs_text,
                "char_count": total_chars,
                "word_count": len(full_text.split()),
                "extraction_method": "python-docx",
                "metadata": {
                    "file_type": "docx",
                    "extractor": "BasicTextExtractor"
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"DOCX extraction failed: {str(e)}"
            }

    async def _extract_spreadsheet_text(self, filepath: Path) -> dict[str, Any]:
        """Extract text from spreadsheet files (XLSX, XLS, CSV)"""
        if not HAS_PANDAS:
            return {
                "status": "error",
                "error": "pandas not available. Install with: pip install pandas openpyxl"
            }

        try:
            file_ext = filepath.suffix.lower()
            sheets_data = []

            if file_ext == '.csv':
                # Handle CSV files
                df = pd.read_csv(str(filepath))
                sheet_text = self._dataframe_to_text(df, "CSV")
                sheets_data.append({
                    "sheet_name": "CSV",
                    "text": sheet_text,
                    "row_count": len(df),
                    "col_count": len(df.columns)
                })
            else:
                # Handle Excel files
                excel_file = pd.ExcelFile(str(filepath))

                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    sheet_text = self._dataframe_to_text(df, sheet_name)

                    sheets_data.append({
                        "sheet_name": sheet_name,
                        "text": sheet_text,
                        "row_count": len(df),
                        "col_count": len(df.columns)
                    })

            # Combine all sheets
            full_text = "\n\n".join([
                f"=== {sheet['sheet_name']} ===\n{sheet['text']}"
                for sheet in sheets_data
            ])

            total_chars = len(full_text)

            return {
                "status": "success",
                "text": full_text,
                "sheet_count": len(sheets_data),
                "sheets": sheets_data,
                "char_count": total_chars,
                "word_count": len(full_text.split()),
                "extraction_method": "pandas",
                "metadata": {
                    "file_type": file_ext.lstrip('.'),
                    "extractor": "BasicTextExtractor"
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"Spreadsheet extraction failed: {str(e)}"
            }

    def _dataframe_to_text(self, df: 'pd.DataFrame', sheet_name: str) -> str:
        """Convert DataFrame to readable text"""
        try:
            # Get column headers
            headers = " | ".join(str(col) for col in df.columns)
            text_parts = [f"Sheet: {sheet_name}", f"Columns: {headers}", ""]

            # Convert each row to text
            for idx, row in df.iterrows():
                row_text = " | ".join(str(val) for val in row.values)
                text_parts.append(f"Row {idx + 1}: {row_text}")

            return "\n".join(text_parts)

        except Exception as e:
            logger.warning(f"DataFrame to text conversion failed: {str(e)}")
            return f"Sheet: {sheet_name}\nData extraction failed: {str(e)}"

    async def _extract_image_text(self, filepath: Path) -> dict[str, Any]:
        """Extract text from images using OCR"""
        if not HAS_IMAGE_PROCESSING:
            return {
                "status": "error",
                "error": "Image processing libraries not available. Install with: pip install pillow opencv-python"
            }

        # Check file size
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_image_size_mb:
            return {
                "status": "error",
                "error": f"Image too large: {file_size_mb:.1f}MB > {self.max_image_size_mb}MB"
            }

        try:
            # Load and preprocess image
            image = Image.open(str(filepath))

            if self.enable_ocr_preprocessing:
                image = self._preprocess_image(image)

            # Perform OCR with Tesseract (Malayalam + English support)
            if self.ocr_engine == "tesseract" and HAS_TESSERACT:
                ocr_result = await self._ocr_with_tesseract(image)
            elif HAS_EASYOCR and self.easyocr_reader:
                # Fallback to EasyOCR if available (English only)
                logger.warning("⚠️ Tesseract not available, falling back to EasyOCR (English only)")
                ocr_result = await self._ocr_with_easyocr(image, filepath)
            else:
                return {
                    "status": "error",
                    "error": f"No OCR engine available. Please install Tesseract for Malayalam support."
                }

            return ocr_result

        except Exception as e:
            return {
                "status": "error",
                "error": f"Image OCR failed: {str(e)}"
            }

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)

            # Apply threshold to get binary image
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Convert back to PIL Image
            processed_image = Image.fromarray(binary)

            return processed_image

        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}, using original image")
            return image

    async def _ocr_with_easyocr(self, image: Image.Image, filepath: Path) -> dict[str, Any]:
        """Perform OCR using EasyOCR"""
        try:
            # Convert PIL Image to numpy array for EasyOCR
            image_array = np.array(image)

            # Perform OCR
            results = self.easyocr_reader.readtext(image_array)

            # Extract text and confidence scores
            text_parts = []
            confidence_scores = []

            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidence_scores.append(confidence)

            full_text = "\n".join(text_parts)
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            return {
                "status": "success",
                "text": self._clean_text(full_text),
                "char_count": len(full_text),
                "word_count": len(full_text.split()),
                "extraction_method": "easyocr",
                "confidence_score": round(avg_confidence, 3),
                "detected_texts_count": len(text_parts),
                "metadata": {
                    "file_type": "image",
                    "ocr_engine": "easyocr",
                    "languages": self.ocr_languages,
                    "extractor": "BasicTextExtractor"
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"EasyOCR failed: {str(e)}"
            }

    async def _ocr_hybrid(self, image: Image.Image, filepath: Path) -> dict[str, Any]:
        """Perform hybrid OCR: detect language first, then use appropriate engine"""
        try:
            # First, do a quick language detection with Tesseract
            text_sample = pytesseract.image_to_string(image, lang="eng+mal")[:200]  # Sample text
            
            # Simple Malayalam detection (look for Malayalam Unicode characters)
            malayalam_chars = sum(1 for char in text_sample if 0x0D00 <= ord(char) <= 0x0D7F)
            total_chars = len([c for c in text_sample if c.isalpha()])
            
            if total_chars > 0:
                malayalam_ratio = malayalam_chars / total_chars
                
                if malayalam_ratio > 0.1:  # More than 10% Malayalam characters
                    logger.info("🔤 Detected Malayalam text, using Tesseract")
                    return await self._ocr_with_tesseract(image, ["mal", "eng"])
                else:
                    logger.info("🔤 Detected English text, using EasyOCR")
                    return await self._ocr_with_easyocr(image, filepath)
            else:
                # Fallback to EasyOCR for English
                return await self._ocr_with_easyocr(image, filepath)
                
        except Exception as e:
            logger.warning(f"Hybrid OCR detection failed: {str(e)}, falling back to Tesseract")
            return await self._ocr_with_tesseract(image, ["mal", "eng"])

    async def _ocr_with_tesseract(self, image: Image.Image, languages: list[str] = None) -> dict[str, Any]:
        """Perform OCR using Tesseract"""
        try:
            # Configure Tesseract for specified languages or fallback to config
            if languages:
                lang_config = "+".join(languages)
            else:
                # Convert language codes: ml -> mal for Tesseract
                tesseract_langs = []
                for lang in self.ocr_languages:
                    if lang == "ml":
                        tesseract_langs.append("mal")  # Malayalam in Tesseract
                    elif lang == "en":
                        tesseract_langs.append("eng")  # English in Tesseract
                    else:
                        tesseract_langs.append(lang)
                lang_config = "+".join(tesseract_langs)

            # Perform OCR
            text = pytesseract.image_to_string(image, lang=lang_config)

            # Get confidence data
            try:
                data = pytesseract.image_to_data(image, lang=lang_config, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            except:
                avg_confidence = 0.0

            cleaned_text = self._clean_text(text)

            return {
                "status": "success",
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "extraction_method": "tesseract",
                "confidence_score": round(avg_confidence / 100, 3),  # Convert to 0-1 scale
                "metadata": {
                    "file_type": "image",
                    "ocr_engine": "tesseract",
                    "languages": self.ocr_languages,
                    "extractor": "BasicTextExtractor"
                }
            }

        except Exception as e:
            return {
                "status": "error",
                "error": f"Tesseract OCR failed: {str(e)}"
            }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""

        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove common OCR artifacts
        text = text.replace("§", "")  # Section symbol
        text = text.replace("¶", "")  # Paragraph symbol

        # Normalize line breaks
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        return text.strip()

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats"""
        available_formats = []

        if HAS_PYMUPDF:
            available_formats.append("pdf")

        if HAS_DOCX:
            available_formats.extend(["docx", "doc"])

        if HAS_PANDAS:
            available_formats.extend(["xlsx", "xls", "csv"])

        if HAS_IMAGE_PROCESSING and (HAS_EASYOCR or HAS_TESSERACT):
            available_formats.extend(["jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"])

        return available_formats

    def get_extraction_stats(self) -> dict[str, Any]:
        """Get text extraction service statistics"""
        return {
            "supported_formats": self.get_supported_formats(),
            "configured_formats": self.supported_formats,
            "ocr_engine": self.ocr_engine,
            "ocr_languages": self.ocr_languages,
            "max_image_size_mb": self.max_image_size_mb,
            "dependencies": {
                "pymupdf": HAS_PYMUPDF,
                "python_docx": HAS_DOCX,
                "pandas": HAS_PANDAS,
                "image_processing": HAS_IMAGE_PROCESSING,
                "easyocr": HAS_EASYOCR,
                "tesseract": HAS_TESSERACT
            },
            "ocr_preprocessing_enabled": self.enable_ocr_preprocessing
        }
