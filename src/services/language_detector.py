"""
Language Detection Service
Detects Malayalam/English content and handles character encoding for Indic scripts
"""

import logging
import re
from typing import Any

try:
    import langdetect
    from langdetect import DetectorFactory, detect, detect_langs
    # Set seed for consistent results
    DetectorFactory.seed = 0
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

try:
    import polyglot
    from polyglot.detect import Detector
    HAS_POLYGLOT = True
except ImportError:
    HAS_POLYGLOT = False

from src.core.config import config

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Language detection service with focus on Malayalam-English detection
    Handles character encoding and script detection for cross-lingual support
    """

    def __init__(self):
        self.target_languages = ['en', 'ml']  # English and Malayalam
        self.enable_translation = config.enable_translation
        self.confidence_threshold = config.translation_confidence_threshold

        # Malayalam Unicode range
        self.malayalam_unicode_range = (0x0D00, 0x0D7F)  # Malayalam Unicode block

        # Common English words for fallback detection
        self.common_english_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'over', 'after', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'
        }

        self._check_dependencies()
        logger.info("LanguageDetector initialized for Malayalam-English detection")

    def _check_dependencies(self):
        """Check available language detection libraries"""
        deps_status = {
            "langdetect": HAS_LANGDETECT,
            "polyglot": HAS_POLYGLOT
        }

        logger.info("🌐 Language detection dependencies:")
        for dep, available in deps_status.items():
            status = "✅" if available else "❌"
            logger.info(f"  {status} {dep}")

        if not HAS_LANGDETECT and not HAS_POLYGLOT:
            logger.warning("⚠️ No language detection libraries available. Using fallback detection.")

    def detect_language(self, text: str, detailed: bool = False) -> dict[str, Any]:
        """
        Detect language of input text with focus on Malayalam-English
        
        Args:
            text: Input text to analyze
            detailed: Whether to return detailed analysis
            
        Returns:
            Dictionary with detection results
        """
        if not text or not text.strip():
            return {
                "status": "error",
                "error": "Empty text provided",
                "detected_language": "unknown",
                "confidence": 0.0
            }

        text = text.strip()
        logger.debug(f"🔍 Detecting language for text: {text[:100]}...")

        try:
            # Multi-method detection for better accuracy
            results = {}

            # Method 1: Script-based detection (fast and reliable for Malayalam)
            script_result = self._detect_by_script(text)
            results["script_detection"] = script_result

            # Method 2: Library-based detection
            if HAS_LANGDETECT:
                langdetect_result = self._detect_with_langdetect(text)
                results["langdetect"] = langdetect_result

            if HAS_POLYGLOT:
                polyglot_result = self._detect_with_polyglot(text)
                results["polyglot"] = polyglot_result

            # Method 3: Pattern-based detection (fallback)
            pattern_result = self._detect_by_patterns(text)
            results["pattern_detection"] = pattern_result

            # Combine results for final decision
            final_result = self._combine_detection_results(results, text)

            if detailed:
                final_result["detailed_results"] = results
                final_result["text_stats"] = self._get_text_statistics(text)

            logger.info(f"✅ Detected language: {final_result['detected_language']} "
                       f"(confidence: {final_result['confidence']:.3f})")

            return final_result

        except Exception as e:
            logger.error(f"❌ Language detection failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "detected_language": "unknown",
                "confidence": 0.0
            }

    def _detect_by_script(self, text: str) -> dict[str, Any]:
        """Detect language based on character script (most reliable for Malayalam)"""
        try:
            malayalam_chars = 0
            english_chars = 0
            total_chars = 0

            for char in text:
                if char.isalpha():
                    total_chars += 1
                    char_code = ord(char)

                    if self.malayalam_unicode_range[0] <= char_code <= self.malayalam_unicode_range[1]:
                        malayalam_chars += 1
                    elif 'a' <= char.lower() <= 'z':
                        english_chars += 1

            if total_chars == 0:
                return {
                    "method": "script_detection",
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "reason": "No alphabetic characters found"
                }

            malayalam_ratio = malayalam_chars / total_chars
            english_ratio = english_chars / total_chars

            # Determine language based on script ratios
            if malayalam_ratio > 0.3:  # At least 30% Malayalam characters
                detected_lang = "ml"
                confidence = min(malayalam_ratio * 2, 1.0)  # Scale confidence
            elif english_ratio > 0.7:  # At least 70% English characters
                detected_lang = "en"
                confidence = min(english_ratio, 1.0)
            elif malayalam_ratio > english_ratio:
                detected_lang = "ml"
                confidence = malayalam_ratio
            else:
                detected_lang = "en"
                confidence = english_ratio

            return {
                "method": "script_detection",
                "detected_language": detected_lang,
                "confidence": confidence,
                "malayalam_ratio": malayalam_ratio,
                "english_ratio": english_ratio,
                "total_alphabetic_chars": total_chars
            }

        except Exception as e:
            return {
                "method": "script_detection",
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def _detect_with_langdetect(self, text: str) -> dict[str, Any]:
        """Detect language using langdetect library"""
        try:
            # Get language with confidence
            detected_lang = detect(text)

            # Get detailed probabilities
            lang_probs = detect_langs(text)

            # Find confidence for detected language
            confidence = 0.0
            for lang_prob in lang_probs:
                if lang_prob.lang == detected_lang:
                    confidence = lang_prob.prob
                    break

            # Map some common language codes
            if detected_lang in ['hi', 'bn', 'ta', 'te', 'kn']:  # Other Indic languages
                # If it's an Indic language, there's a chance it might be Malayalam
                if detected_lang in ['hi', 'ta']:  # Closer scripts to Malayalam
                    detected_lang = 'ml'
                    confidence *= 0.7  # Reduce confidence due to uncertainty

            return {
                "method": "langdetect",
                "detected_language": detected_lang,
                "confidence": confidence,
                "all_probabilities": [(lp.lang, lp.prob) for lp in lang_probs]
            }

        except Exception as e:
            return {
                "method": "langdetect",
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def _detect_with_polyglot(self, text: str) -> dict[str, Any]:
        """Detect language using polyglot library"""
        try:
            detector = Detector(text)
            detected_lang = detector.language.code
            confidence = detector.language.confidence

            return {
                "method": "polyglot",
                "detected_language": detected_lang,
                "confidence": confidence,
                "language_name": detector.language.name
            }

        except Exception as e:
            return {
                "method": "polyglot",
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def _detect_by_patterns(self, text: str) -> dict[str, Any]:
        """Fallback pattern-based detection"""
        try:
            words = re.findall(r'\b\w+\b', text.lower())

            if not words:
                return {
                    "method": "pattern_detection",
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "reason": "No words found"
                }

            # Count English words
            english_word_count = sum(1 for word in words if word in self.common_english_words)
            english_ratio = english_word_count / len(words)

            # Simple heuristics
            if english_ratio > 0.2:  # At least 20% common English words
                detected_lang = "en"
                confidence = min(english_ratio * 3, 1.0)  # Scale up confidence
            else:
                # If not clearly English, assume Malayalam (since it's our target)
                detected_lang = "ml"
                confidence = max(0.3, 1.0 - english_ratio)  # Inverse of English ratio

            return {
                "method": "pattern_detection",
                "detected_language": detected_lang,
                "confidence": confidence,
                "english_word_ratio": english_ratio,
                "total_words": len(words),
                "english_words_found": english_word_count
            }

        except Exception as e:
            return {
                "method": "pattern_detection",
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }

    def _combine_detection_results(self, results: dict[str, Any], text: str) -> dict[str, Any]:
        """Combine results from multiple detection methods"""

        # Collect valid detections
        detections = []

        for method, result in results.items():
            if result.get("detected_language") != "unknown" and result.get("confidence", 0) > 0:
                detections.append({
                    "method": method,
                    "language": result["detected_language"],
                    "confidence": result["confidence"]
                })

        if not detections:
            return {
                "status": "success",
                "detected_language": "en",  # Default to English
                "confidence": 0.1,
                "method": "default_fallback",
                "message": "No reliable detection, defaulting to English"
            }

        # Weight the methods (script detection is most reliable for Malayalam)
        method_weights = {
            "script_detection": 3.0,
            "langdetect": 1.5,
            "polyglot": 1.2,
            "pattern_detection": 1.0
        }

        # Calculate weighted scores for each language
        language_scores = {}

        for detection in detections:
            lang = detection["language"]
            confidence = detection["confidence"]
            weight = method_weights.get(detection["method"], 1.0)

            weighted_score = confidence * weight

            if lang not in language_scores:
                language_scores[lang] = {
                    "total_score": 0.0,
                    "count": 0,
                    "methods": []
                }

            language_scores[lang]["total_score"] += weighted_score
            language_scores[lang]["count"] += 1
            language_scores[lang]["methods"].append(detection["method"])

        # Find the language with highest average weighted score
        best_language = "en"  # Default
        best_score = 0.0

        for lang, scores in language_scores.items():
            avg_score = scores["total_score"] / scores["count"]
            if avg_score > best_score:
                best_score = avg_score
                best_language = lang

        # Normalize confidence to 0-1 range
        final_confidence = min(best_score / 3.0, 1.0)  # Divide by max possible weight

        # Ensure minimum confidence for Malayalam detection
        if best_language == "ml" and final_confidence < 0.5:
            # Check if we have strong script evidence
            script_result = results.get("script_detection", {})
            if script_result.get("malayalam_ratio", 0) > 0.1:
                final_confidence = max(final_confidence, 0.6)

        return {
            "status": "success",
            "detected_language": best_language,
            "confidence": round(final_confidence, 3),
            "method": "combined",
            "contributing_methods": language_scores[best_language]["methods"],
            "alternative_languages": {
                lang: round(scores["total_score"] / scores["count"] / 3.0, 3)
                for lang, scores in language_scores.items()
                if lang != best_language
            }
        }

    def _get_text_statistics(self, text: str) -> dict[str, Any]:
        """Get detailed statistics about the text"""
        try:
            # Character analysis
            malayalam_chars = sum(1 for char in text
                                 if self.malayalam_unicode_range[0] <= ord(char) <= self.malayalam_unicode_range[1])

            english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')

            numeric_chars = sum(1 for char in text if char.isdigit())

            # Word analysis
            words = re.findall(r'\b\w+\b', text)

            # Sentence analysis
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            return {
                "total_length": len(text),
                "malayalam_chars": malayalam_chars,
                "english_chars": english_chars,
                "numeric_chars": numeric_chars,
                "word_count": len(words),
                "sentence_count": len(sentences),
                "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0,
                "malayalam_char_ratio": malayalam_chars / len(text) if text else 0,
                "english_char_ratio": english_chars / len(text) if text else 0
            }

        except Exception as e:
            logger.warning(f"Text statistics calculation failed: {str(e)}")
            return {"error": str(e)}

    def detect_mixed_content(self, text: str) -> dict[str, Any]:
        """Detect if text contains mixed Malayalam-English content"""
        try:
            # Split text into segments and analyze each
            segments = re.split(r'[.!?\n]+', text)
            segments = [s.strip() for s in segments if s.strip()]

            segment_languages = []

            for segment in segments:
                if len(segment) < 10:  # Skip very short segments
                    continue

                detection = self.detect_language(segment)
                if detection["confidence"] > 0.3:  # Only count confident detections
                    segment_languages.append({
                        "text": segment[:100],  # First 100 chars for reference
                        "language": detection["detected_language"],
                        "confidence": detection["confidence"]
                    })

            # Analyze language distribution
            languages_found = {}
            for seg in segment_languages:
                lang = seg["language"]
                if lang not in languages_found:
                    languages_found[lang] = {
                        "count": 0,
                        "avg_confidence": 0.0,
                        "confidences": []
                    }

                languages_found[lang]["count"] += 1
                languages_found[lang]["confidences"].append(seg["confidence"])

            # Calculate average confidences
            for lang in languages_found:
                confidences = languages_found[lang]["confidences"]
                languages_found[lang]["avg_confidence"] = sum(confidences) / len(confidences)

            # Determine if content is mixed
            is_mixed = len(languages_found) > 1
            dominant_language = max(languages_found, key=lambda x: languages_found[x]["count"]) if languages_found else "unknown"

            return {
                "status": "success",
                "is_mixed_content": is_mixed,
                "dominant_language": dominant_language,
                "languages_detected": languages_found,
                "segments_analyzed": len(segment_languages),
                "segment_details": segment_languages
            }

        except Exception as e:
            logger.error(f"Mixed content detection failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "is_mixed_content": False
            }

    def is_malayalam_text(self, text: str, threshold: float = 0.3) -> bool:
        """Quick check if text is primarily Malayalam"""
        try:
            result = self.detect_language(text)
            return (result["detected_language"] == "ml" and
                   result["confidence"] >= threshold)
        except:
            return False

    def is_english_text(self, text: str, threshold: float = 0.5) -> bool:
        """Quick check if text is primarily English"""
        try:
            result = self.detect_language(text)
            return (result["detected_language"] == "en" and
                   result["confidence"] >= threshold)
        except:
            return False

    def normalize_text_encoding(self, text: str) -> str:
        """Normalize text encoding for better processing"""
        try:
            # Ensure proper Unicode encoding
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')

            # Normalize Unicode characters
            import unicodedata
            text = unicodedata.normalize('NFC', text)

            return text
        except Exception as e:
            logger.warning(f"Text encoding normalization failed: {str(e)}")
            return text

    def get_language_stats(self) -> dict[str, Any]:
        """Get language detection service statistics"""
        return {
            "target_languages": self.target_languages,
            "confidence_threshold": self.confidence_threshold,
            "translation_enabled": self.enable_translation,
            "supported_detection_methods": {
                "script_detection": True,  # Always available
                "langdetect": HAS_LANGDETECT,
                "polyglot": HAS_POLYGLOT,
                "pattern_detection": True  # Always available
            },
            "malayalam_unicode_range": self.malayalam_unicode_range,
            "service_status": "active"
        }
