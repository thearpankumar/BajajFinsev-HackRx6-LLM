"""
Malayalam-English Translation Service
Advanced translation service with quality assessment and bidirectional support
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Union

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer, pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from googletrans import Translator as GoogleTranslator
    HAS_GOOGLETRANS = True
except ImportError:
    HAS_GOOGLETRANS = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.core.config import config
from src.services.language_detector import LanguageDetector
from src.services.redis_cache import redis_manager

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Translation result with metadata"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    translation_method: str
    processing_time: float
    quality_metrics: dict[str, Any] | None = None


@dataclass
class QualityAssessment:
    """Translation quality assessment"""
    fluency_score: float
    adequacy_score: float
    bleu_score: float
    length_ratio: float
    character_accuracy: float
    overall_quality: str  # excellent, good, fair, poor
    issues: list[str]
    recommendations: list[str]


class TranslationService:
    """
    Advanced Malayalam-English translation service with quality assessment
    Supports multiple translation backends and quality evaluation
    """

    def __init__(self):
        # Configuration
        self.enable_cache = config.enable_embedding_cache
        self.primary_method = getattr(config, 'translation_primary_method', 'transformers')
        self.fallback_methods = getattr(config, 'translation_fallback_methods', ['google', 'rule_based'])
        self.quality_threshold = getattr(config, 'translation_quality_threshold', 0.7)

        # Model configurations
        self.model_configs = {
            'malayalam_to_english': {
                'model_name': 'Helsinki-NLP/opus-mt-ml-en',
                'fallback_model': 'Helsinki-NLP/opus-mt-mul-en'
            },
            'english_to_malayalam': {
                'model_name': 'Helsinki-NLP/opus-mt-en-ml',
                'fallback_model': 'Helsinki-NLP/opus-mt-en-mul'
            }
        }

        # Translation models
        self.models: dict[str, Any] = {}
        self.tokenizers: dict[str, Any] = {}
        self.pipelines: dict[str, Any] = {}

        # Language detector
        self.language_detector = LanguageDetector()

        # Google Translator fallback
        self.google_translator = None
        if HAS_GOOGLETRANS:
            self.google_translator = GoogleTranslator()

        # Redis cache manager
        self.redis_manager = redis_manager

        # Performance tracking
        self.total_translations = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.method_usage = {
            'transformers': 0,
            'google': 0,
            'rule_based': 0
        }

        # Quality assessment patterns
        self._init_quality_patterns()

        logger.info("TranslationService initialized with Malayalam-English support")

    def _init_quality_patterns(self):
        """Initialize quality assessment patterns"""
        # Common translation issues
        self.quality_patterns = {
            'repeated_words': r'\b(\w+)\s+\1\b',
            'excessive_punctuation': r'[.!?]{3,}',
            'mixed_scripts': r'[a-zA-Z]+.*[\u0d00-\u0d7f]+|[\u0d00-\u0d7f]+.*[a-zA-Z]+',
            'incomplete_sentence': r'^[a-z]|\w+$',
            'formatting_issues': r'\s{2,}|^\s+|\s+$'
        }

        # Language-specific patterns
        self.malayalam_patterns = {
            'valid_chars': r'^[\u0d00-\u0d7f\s\.,!?;:()"\'-]+$',
            'common_words': ['എന്ന്', 'ആണ്', 'ഉണ്ട്', 'ചെയ്യുക', 'പറയുക', 'വരുക'],
            'sentence_endings': ['।', '.', '?', '!']
        }

        self.english_patterns = {
            'valid_chars': r'^[a-zA-Z\s\.,!?;:()"\'-0-9]+$',
            'common_words': ['the', 'and', 'is', 'are', 'in', 'of', 'to', 'for'],
            'sentence_endings': ['.', '?', '!']
        }

    async def initialize(self) -> dict[str, Any]:
        """Initialize translation service"""
        try:
            logger.info("🔄 Initializing Translation Service...")
            start_time = time.time()

            initialization_results = {}

            # Check dependencies
            if not HAS_TRANSFORMERS:
                logger.warning("⚠️ Transformers not available. Install with: pip install transformers")
                initialization_results["transformers"] = "unavailable"
            else:
                # Initialize transformer models
                await self._initialize_transformer_models()
                initialization_results["transformers"] = "initialized"

            if not HAS_GOOGLETRANS:
                logger.warning("⚠️ Google Translate not available. Install with: pip install googletrans==4.0.0-rc1")
                initialization_results["google_translate"] = "unavailable"
            else:
                initialization_results["google_translate"] = "initialized"

            # Initialize cache
            if self.enable_cache and not self.redis_manager.is_connected:
                await self.redis_manager.initialize()
                initialization_results["cache"] = "initialized"

            initialization_time = time.time() - start_time

            result = {
                "status": "success",
                "message": f"Translation service initialized in {initialization_time:.2f}s",
                "primary_method": self.primary_method,
                "fallback_methods": self.fallback_methods,
                "supported_directions": ["ml->en", "en->ml", "auto-detect"],
                "quality_assessment": True,
                "caching_enabled": self.enable_cache,
                "initialization_details": initialization_results,
                "initialization_time": initialization_time
            }

            logger.info(f"✅ Translation Service ready with {self.primary_method} primary method")
            return result

        except Exception as e:
            error_msg = f"Translation service initialization failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return {
                "status": "error",
                "error": error_msg
            }

    async def _initialize_transformer_models(self):
        """Initialize transformer models for translation"""
        try:
            if not HAS_TRANSFORMERS or not HAS_TORCH:
                return

            logger.info("📦 Loading translation models...")

            # Malayalam to English
            try:
                model_name = self.model_configs['malayalam_to_english']['model_name']
                logger.info(f"Loading {model_name}...")

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                if torch.cuda.is_available():
                    model = model.to('cuda')

                self.tokenizers['ml_to_en'] = tokenizer
                self.models['ml_to_en'] = model

                # Create pipeline
                self.pipelines['ml_to_en'] = pipeline(
                    "translation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )

                logger.info("✅ Malayalam to English model loaded")

            except Exception as e:
                logger.warning(f"Malayalam to English model loading failed: {str(e)}")

            # English to Malayalam
            try:
                model_name = self.model_configs['english_to_malayalam']['model_name']
                logger.info(f"Loading {model_name}...")

                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)

                if torch.cuda.is_available():
                    model = model.to('cuda')

                self.tokenizers['en_to_ml'] = tokenizer
                self.models['en_to_ml'] = model

                # Create pipeline
                self.pipelines['en_to_ml'] = pipeline(
                    "translation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )

                logger.info("✅ English to Malayalam model loaded")

            except Exception as e:
                logger.warning(f"English to Malayalam model loading failed: {str(e)}")

        except Exception as e:
            logger.warning(f"Transformer models initialization failed: {str(e)}")

    async def translate(
        self,
        text: str,
        target_language: str = "auto",
        source_language: str = "auto",
        quality_assessment: bool = True
    ) -> TranslationResult:
        """
        Translate text with automatic language detection and quality assessment
        
        Args:
            text: Text to translate
            target_language: Target language (en, ml, auto)
            source_language: Source language (en, ml, auto)
            quality_assessment: Whether to perform quality assessment
            
        Returns:
            TranslationResult with translation and quality metrics
        """
        logger.info(f"🔤 Translating text: '{text[:50]}...' -> {target_language}")
        start_time = time.time()

        try:
            # Input validation
            if not text or not text.strip():
                return TranslationResult(
                    original_text=text,
                    translated_text="",
                    source_language="unknown",
                    target_language=target_language,
                    confidence_score=0.0,
                    translation_method="none",
                    processing_time=time.time() - start_time
                )

            # Language detection if needed
            if source_language == "auto":
                detection_result = self.language_detector.detect_language(text)
                source_language = detection_result.get("detected_language", "en")
                logger.info(f"🔍 Detected source language: {source_language}")

            # Determine target language if auto
            if target_language == "auto":
                target_language = "en" if source_language == "ml" else "ml"

            # Check if translation is needed
            if source_language == target_language:
                logger.info("ℹ️ No translation needed (same language)")
                return TranslationResult(
                    original_text=text,
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence_score=1.0,
                    translation_method="passthrough",
                    processing_time=time.time() - start_time
                )

            # Check cache
            cached_result = await self._get_cached_translation(text, source_language, target_language)
            if cached_result:
                self.cache_hits += 1
                logger.info("✅ Translation cache hit")
                return cached_result

            self.cache_misses += 1

            # Perform translation
            translation_result = await self._perform_translation(
                text, source_language, target_language
            )

            # Quality assessment if requested
            if quality_assessment and translation_result.translated_text:
                quality_metrics = await self._assess_translation_quality(
                    text, translation_result.translated_text, source_language, target_language
                )
                translation_result.quality_metrics = quality_metrics

            # Cache result
            if self.enable_cache:
                await self._cache_translation(translation_result)

            # Update statistics
            processing_time = time.time() - start_time
            translation_result.processing_time = processing_time
            self.total_translations += 1
            self.total_processing_time += processing_time

            logger.info(f"✅ Translation completed in {processing_time:.2f}s using {translation_result.translation_method}")
            return translation_result

        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")

            return TranslationResult(
                original_text=text,
                translated_text="",
                source_language=source_language,
                target_language=target_language,
                confidence_score=0.0,
                translation_method="error",
                processing_time=time.time() - start_time
            )

    async def _perform_translation(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> TranslationResult:
        """Perform actual translation using available methods"""

        # Try primary method first
        if self.primary_method == "transformers" and HAS_TRANSFORMERS:
            result = await self._translate_with_transformers(text, source_language, target_language)
            if result and result.confidence_score > 0.3:
                self.method_usage['transformers'] += 1
                return result

        # Try fallback methods
        for method in self.fallback_methods:
            try:
                if method == "google" and HAS_GOOGLETRANS and self.google_translator:
                    result = await self._translate_with_google(text, source_language, target_language)
                    if result and result.confidence_score > 0.3:
                        self.method_usage['google'] += 1
                        return result

                elif method == "rule_based":
                    result = await self._translate_with_rules(text, source_language, target_language)
                    if result and result.confidence_score > 0.1:
                        self.method_usage['rule_based'] += 1
                        return result

            except Exception as e:
                logger.warning(f"Translation method {method} failed: {str(e)}")
                continue

        # If all methods fail, return empty result
        return TranslationResult(
            original_text=text,
            translated_text="",
            source_language=source_language,
            target_language=target_language,
            confidence_score=0.0,
            translation_method="failed",
            processing_time=0.0
        )

    async def _translate_with_transformers(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Union[TranslationResult, None]:
        """Translate using transformer models"""
        try:
            direction = f"{source_language}_to_{target_language}"
            pipeline_key = f"{source_language[:2]}_to_{target_language[:2]}"

            if pipeline_key not in self.pipelines:
                return None

            # Preprocess text
            preprocessed_text = self._preprocess_for_translation(text, source_language)

            # Perform translation
            pipeline = self.pipelines[pipeline_key]
            result = pipeline(preprocessed_text, max_length=512, do_sample=False)

            translated_text = result[0]['translation_text']

            # Post-process
            translated_text = self._postprocess_translation(translated_text, target_language)

            # Calculate confidence (simplified)
            confidence = self._calculate_transformer_confidence(text, translated_text, source_language, target_language)

            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence_score=confidence,
                translation_method="transformers",
                processing_time=0.0
            )

        except Exception as e:
            logger.warning(f"Transformer translation failed: {str(e)}")
            return None

    async def _translate_with_google(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Union[TranslationResult, None]:
        """Translate using Google Translate API"""
        try:
            if not self.google_translator:
                return None

            # Map language codes
            google_source = 'ml' if source_language == 'ml' else 'en'
            google_target = 'ml' if target_language == 'ml' else 'en'

            # Perform translation
            result = self.google_translator.translate(
                text,
                src=google_source,
                dest=google_target
            )

            translated_text = result.text
            confidence = result.confidence if hasattr(result, 'confidence') and result.confidence else 0.8

            return TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                confidence_score=confidence,
                translation_method="google",
                processing_time=0.0
            )

        except Exception as e:
            logger.warning(f"Google translation failed: {str(e)}")
            return None

    async def _translate_with_rules(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Union[TranslationResult, None]:
        """Rule-based translation for simple phrases"""
        try:
            # Simple dictionary lookup for common phrases
            ml_to_en_dict = {
                'നമസ്കാരം': 'Hello',
                'നന്ദി': 'Thank you',
                'എന്താണ്': 'What is',
                'എവിടെ': 'Where',
                'എപ്പോൾ': 'When',
                'എങ്ങനെ': 'How',
                'ആര്': 'Who',
                'എന്തുകൊണ്ട്': 'Why',
                'വരൂ': 'Come',
                'പോകൂ': 'Go',
                'ഉണ്ട്': 'Yes/There is',
                'ഇല്ല': 'No/Not there'
            }

            en_to_ml_dict = {v: k for k, v in ml_to_en_dict.items()}

            # Try direct lookup
            text_lower = text.lower().strip()

            if source_language == 'ml' and target_language == 'en':
                if text in ml_to_en_dict:
                    return TranslationResult(
                        original_text=text,
                        translated_text=ml_to_en_dict[text],
                        source_language=source_language,
                        target_language=target_language,
                        confidence_score=0.9,
                        translation_method="rule_based",
                        processing_time=0.0
                    )

            elif source_language == 'en' and target_language == 'ml':
                if text_lower in en_to_ml_dict:
                    return TranslationResult(
                        original_text=text,
                        translated_text=en_to_ml_dict[text_lower],
                        source_language=source_language,
                        target_language=target_language,
                        confidence_score=0.9,
                        translation_method="rule_based",
                        processing_time=0.0
                    )

            # Pattern-based translation for simple sentences
            return await self._pattern_based_translation(text, source_language, target_language)

        except Exception as e:
            logger.warning(f"Rule-based translation failed: {str(e)}")
            return None

    async def _pattern_based_translation(
        self,
        text: str,
        source_language: str,
        target_language: str
    ) -> Union[TranslationResult, None]:
        """Pattern-based translation for simple structures"""
        try:
            # Very basic pattern matching - this could be expanded significantly
            patterns = {
                'ml_to_en': [
                    (r'(.+)\s+എന്താണ്\?', r'What is \1?'),
                    (r'(.+)\s+എവിടെ\?', r'Where is \1?'),
                    (r'എന്റെ\s+പേര്\s+(.+)', r'My name is \1'),
                ],
                'en_to_ml': [
                    (r'What is (.+)\?', r'\1 എന്താണ്?'),
                    (r'Where is (.+)\?', r'\1 എവിടെ?'),
                    (r'My name is (.+)', r'എന്റെ പേര് \1'),
                ]
            }

            pattern_key = f"{source_language}_to_{target_language}"
            if pattern_key in patterns:
                for pattern, replacement in patterns[pattern_key]:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        translated_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                        return TranslationResult(
                            original_text=text,
                            translated_text=translated_text,
                            source_language=source_language,
                            target_language=target_language,
                            confidence_score=0.6,
                            translation_method="rule_based",
                            processing_time=0.0
                        )

            return None

        except Exception as e:
            logger.warning(f"Pattern-based translation failed: {str(e)}")
            return None

    def _preprocess_for_translation(self, text: str, source_language: str) -> str:
        """Preprocess text for better translation"""
        try:
            # Remove excessive whitespace
            text = ' '.join(text.split())

            # Language-specific preprocessing
            if source_language == 'ml':
                # Malayalam preprocessing
                text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)  # Remove repeated punctuation
            else:
                # English preprocessing
                text = text.strip()

            return text

        except Exception as e:
            logger.warning(f"Text preprocessing failed: {str(e)}")
            return text

    def _postprocess_translation(self, text: str, target_language: str) -> str:
        """Post-process translated text"""
        try:
            # Remove excessive whitespace
            text = ' '.join(text.split())

            # Language-specific post-processing
            if target_language == 'ml':
                # Malayalam post-processing
                text = re.sub(r'\s+([.!?])', r'\1', text)  # Fix punctuation spacing
            else:
                # English post-processing
                text = text.capitalize() if text and not text[0].isupper() else text

            return text.strip()

        except Exception as e:
            logger.warning(f"Text post-processing failed: {str(e)}")
            return text

    def _calculate_transformer_confidence(
        self,
        original: str,
        translated: str,
        source_lang: str,
        target_lang: str
    ) -> float:
        """Calculate confidence score for transformer translation"""
        try:
            confidence = 0.8  # Base confidence for transformer models

            # Adjust based on length ratio
            length_ratio = len(translated) / max(1, len(original))
            if 0.3 <= length_ratio <= 3.0:  # Reasonable length ratio
                confidence += 0.1
            else:
                confidence -= 0.2

            # Check for obvious issues
            if not translated.strip():
                confidence = 0.0
            elif translated == original:
                confidence = 0.5  # Might be untranslated

            return max(0.0, min(1.0, confidence))

        except Exception:
            return 0.5

    async def _assess_translation_quality(
        self,
        original: str,
        translated: str,
        source_lang: str,
        target_lang: str
    ) -> dict[str, Any]:
        """Assess translation quality with multiple metrics"""
        try:
            # Initialize scores
            fluency_score = 0.8
            adequacy_score = 0.8
            length_ratio = len(translated) / max(1, len(original))
            issues = []
            recommendations = []

            # Check for common issues
            for issue_name, pattern in self.quality_patterns.items():
                if re.search(pattern, translated):
                    issues.append(issue_name.replace('_', ' ').title())
                    fluency_score -= 0.1

            # Language-specific checks
            if target_lang == 'ml':
                if not re.search(self.malayalam_patterns['valid_chars'], translated):
                    issues.append("Invalid Malayalam characters")
                    adequacy_score -= 0.2
            elif target_lang == 'en':
                if not re.search(self.english_patterns['valid_chars'], translated):
                    issues.append("Invalid English characters")
                    adequacy_score -= 0.2

            # Length ratio check
            if length_ratio < 0.3 or length_ratio > 3.0:
                issues.append("Unusual length ratio")
                adequacy_score -= 0.1

            # Character accuracy (simplified)
            char_accuracy = min(1.0, len(set(translated.lower())) / max(1, len(set(original.lower()))))

            # BLEU score (simplified approximation)
            bleu_score = self._approximate_bleu_score(original, translated)

            # Overall quality
            overall_score = (fluency_score + adequacy_score + bleu_score) / 3

            if overall_score >= 0.8:
                overall_quality = "excellent"
            elif overall_score >= 0.6:
                overall_quality = "good"
            elif overall_score >= 0.4:
                overall_quality = "fair"
            else:
                overall_quality = "poor"
                recommendations.append("Consider using alternative translation method")

            # Add recommendations based on issues
            if "repeated words" in issues:
                recommendations.append("Review for word repetition")
            if "length ratio" in issues:
                recommendations.append("Check for missing or extra content")

            return {
                "fluency_score": round(fluency_score, 2),
                "adequacy_score": round(adequacy_score, 2),
                "bleu_score": round(bleu_score, 2),
                "length_ratio": round(length_ratio, 2),
                "character_accuracy": round(char_accuracy, 2),
                "overall_quality": overall_quality,
                "overall_score": round(overall_score, 2),
                "issues": issues,
                "recommendations": recommendations
            }

        except Exception as e:
            logger.warning(f"Quality assessment failed: {str(e)}")
            return {
                "fluency_score": 0.5,
                "adequacy_score": 0.5,
                "bleu_score": 0.5,
                "overall_quality": "unknown",
                "error": str(e)
            }

    def _approximate_bleu_score(self, reference: str, candidate: str) -> float:
        """Approximate BLEU score calculation"""
        try:
            ref_words = reference.lower().split()
            cand_words = candidate.lower().split()

            if not cand_words or not ref_words:
                return 0.0

            # Simple word overlap
            common_words = set(ref_words) & set(cand_words)
            precision = len(common_words) / len(set(cand_words))
            recall = len(common_words) / len(set(ref_words))

            if precision + recall == 0:
                return 0.0

            f1_score = 2 * (precision * recall) / (precision + recall)
            return min(1.0, f1_score)

        except Exception:
            return 0.5

    async def _get_cached_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Union[TranslationResult, None]:
        """Get cached translation if available"""
        if not self.enable_cache or not self.redis_manager.is_connected:
            return None

        try:
            import hashlib
            cache_key = f"translation:{hashlib.md5(f'{text}:{source_lang}:{target_lang}'.encode()).hexdigest()}"
            cached_data = await self.redis_manager.get_json(cache_key)

            if cached_data:
                return TranslationResult(**cached_data)

        except Exception as e:
            logger.warning(f"Translation cache retrieval failed: {str(e)}")

        return None

    async def _cache_translation(self, result: TranslationResult):
        """Cache translation result"""
        if not self.enable_cache or not self.redis_manager.is_connected:
            return

        try:
            import hashlib
            cache_key = f"translation:{hashlib.md5(f'{result.original_text}:{result.source_language}:{result.target_language}'.encode()).hexdigest()}"

            cache_data = {
                "original_text": result.original_text,
                "translated_text": result.translated_text,
                "source_language": result.source_language,
                "target_language": result.target_language,
                "confidence_score": result.confidence_score,
                "translation_method": result.translation_method,
                "processing_time": result.processing_time,
                "quality_metrics": result.quality_metrics,
                "cached_at": time.time()
            }

            # Cache for 24 hours
            await self.redis_manager.set_json(cache_key, cache_data, ex=86400)

        except Exception as e:
            logger.warning(f"Translation caching failed: {str(e)}")

    def get_translation_stats(self) -> dict[str, Any]:
        """Get comprehensive translation statistics"""
        try:
            avg_processing_time = (
                self.total_processing_time / self.total_translations
                if self.total_translations > 0 else 0.0
            )

            cache_hit_rate = (
                (self.cache_hits / (self.cache_hits + self.cache_misses)) * 100
                if (self.cache_hits + self.cache_misses) > 0 else 0.0
            )

            return {
                "total_translations": self.total_translations,
                "total_processing_time": round(self.total_processing_time, 2),
                "average_processing_time": round(avg_processing_time, 3),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "method_usage": self.method_usage,
                "primary_method": self.primary_method,
                "supported_languages": ["en", "ml"],
                "supported_directions": ["ml->en", "en->ml", "auto-detect"],
                "quality_assessment_enabled": True,
                "models_loaded": {
                    "ml_to_en": "ml_to_en" in self.pipelines,
                    "en_to_ml": "en_to_ml" in self.pipelines
                }
            }

        except Exception as e:
            logger.warning(f"Translation stats collection failed: {str(e)}")
            return {"error": str(e)}
