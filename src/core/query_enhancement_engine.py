"""
Query Enhancement and Understanding Engine for BajajFinsev Hybrid RAG System
Advanced query processing with intent detection, expansion, and optimization
Integrates with Gemini-2.5-flash-lite for query understanding
"""

import asyncio
import logging
import time
import re
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

# NLP libraries
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    nltk = None

# Google AI integration
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None

# Embedding service integration
from src.services.embedding_service import get_embedding_service
from src.services.redis_cache import get_redis_cache

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification"""
    FACTUAL = "factual"                    # Direct factual questions
    ANALYTICAL = "analytical"              # Analysis or comparison questions
    PROCEDURAL = "procedural"              # How-to or process questions
    DEFINITIONAL = "definitional"          # What is, define, explain
    NUMERICAL = "numerical"                # Calculations, statistics
    TEMPORAL = "temporal"                  # Time-based questions
    COMPARATIVE = "comparative"            # Compare, contrast, difference
    CAUSAL = "causal"                     # Why, cause, reason
    HYPOTHETICAL = "hypothetical"          # What if, scenario-based
    MULTI_PART = "multi_part"             # Complex multi-part questions


class QueryComplexity(Enum):
    """Query complexity levels"""
    SIMPLE = "simple"                     # Single concept, direct
    MODERATE = "moderate"                 # Multiple concepts, some context
    COMPLEX = "complex"                   # Multi-faceted, requires reasoning
    VERY_COMPLEX = "very_complex"         # Highly analytical, multi-step


@dataclass
class QueryAnalysis:
    """Analysis result for user query"""
    original_query: str
    
    # Intent and complexity
    intent: QueryIntent
    complexity: QueryComplexity
    confidence_score: float
    
    # Enhanced query variants
    expanded_query: str
    semantic_variants: List[str] = field(default_factory=list)
    keyword_variants: List[str] = field(default_factory=list)
    
    # Query components
    key_concepts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    question_type: str = ""
    temporal_indicators: List[str] = field(default_factory=list)
    
    # Context requirements
    requires_context: bool = False
    context_window: int = 512
    expected_answer_type: str = ""
    
    # Processing metadata
    processing_time: float = 0.0
    enhancement_strategy: str = ""
    llm_enhanced: bool = False


@dataclass
class EnhancedQuery:
    """Enhanced query with multiple variants and embeddings"""
    analysis: QueryAnalysis
    
    # Query embeddings
    original_embedding: Optional[List[float]] = None
    expanded_embedding: Optional[List[float]] = None
    variant_embeddings: List[List[float]] = field(default_factory=list)
    
    # Search strategies
    retrieval_strategies: List[str] = field(default_factory=list)
    similarity_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # Optimization parameters
    top_k_suggestions: int = 10
    reranking_enabled: bool = True
    multi_stage_retrieval: bool = False


class QueryEnhancementEngine:
    """
    Advanced query enhancement engine with LLM integration
    Provides query understanding, expansion, and optimization
    """
    
    # Query patterns for intent detection
    INTENT_PATTERNS = {
        QueryIntent.FACTUAL: [
            r'\b(what is|who is|when did|where is|which)\b',
            r'\b(fact|information|details about)\b',
            r'\b(tell me about|explain)\b'
        ],
        QueryIntent.ANALYTICAL: [
            r'\b(analyze|analysis|examine|evaluate)\b',
            r'\b(impact|effect|consequence|result)\b',
            r'\b(trend|pattern|correlation)\b'
        ],
        QueryIntent.PROCEDURAL: [
            r'\b(how to|how do|how can|steps to|process)\b',
            r'\b(procedure|method|way to|approach)\b',
            r'\b(guide|tutorial|instructions)\b'
        ],
        QueryIntent.DEFINITIONAL: [
            r'\b(define|definition|meaning|what does.*mean)\b',
            r'\b(concept of|term|terminology)\b',
            r'\b(explanation|clarify|clarification)\b'
        ],
        QueryIntent.NUMERICAL: [
            r'\b(calculate|computation|number|amount|quantity)\b',
            r'\b(statistics|percentage|rate|ratio)\b',
            r'\b(how much|how many|cost|price|value)\b'
        ],
        QueryIntent.COMPARATIVE: [
            r'\b(compare|comparison|versus|vs|difference)\b',
            r'\b(better|worse|advantage|disadvantage)\b',
            r'\b(similarity|contrast|alike|different)\b'
        ],
        QueryIntent.CAUSAL: [
            r'\b(why|reason|cause|because|due to)\b',
            r'\b(factor|influence|lead to|result in)\b',
            r'\b(explanation for|basis|foundation)\b'
        ],
        QueryIntent.TEMPORAL: [
            r'\b(when|timeline|schedule|time|date)\b',
            r'\b(before|after|during|since|until)\b',
            r'\b(history|historical|chronology)\b'
        ]
    }
    
    # Complexity indicators
    COMPLEXITY_INDICATORS = {
        QueryComplexity.SIMPLE: {
            'max_concepts': 2,
            'max_words': 10,
            'question_words': ['what', 'who', 'when', 'where'],
            'complexity_score': 0.3
        },
        QueryComplexity.MODERATE: {
            'max_concepts': 4,
            'max_words': 20,
            'question_words': ['how', 'why', 'which', 'explain'],
            'complexity_score': 0.6
        },
        QueryComplexity.COMPLEX: {
            'max_concepts': 6,
            'max_words': 35,
            'question_words': ['analyze', 'compare', 'evaluate', 'assess'],
            'complexity_score': 0.8
        },
        QueryComplexity.VERY_COMPLEX: {
            'max_concepts': float('inf'),
            'max_words': float('inf'),
            'question_words': ['comprehensive', 'detailed', 'thorough'],
            'complexity_score': 1.0
        }
    }
    
    def __init__(self,
                 google_api_key: Optional[str] = None,
                 enable_llm_enhancement: bool = True,
                 enable_caching: bool = True,
                 cache_ttl: int = 3600,
                 max_query_variants: int = 5):
        """
        Initialize query enhancement engine
        
        Args:
            google_api_key: Google AI API key for Gemini integration
            enable_llm_enhancement: Enable LLM-based query enhancement
            enable_caching: Enable Redis caching
            cache_ttl: Cache TTL in seconds
            max_query_variants: Maximum query variants to generate
        """
        self.enable_llm_enhancement = enable_llm_enhancement and GOOGLE_AI_AVAILABLE
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.max_query_variants = max_query_variants
        
        # Initialize Google AI
        self.gemini_model = None
        if self.enable_llm_enhancement and google_api_key:
            try:
                genai.configure(api_key=google_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
                logger.info("✅ Gemini-2.0-flash-exp initialized for query enhancement")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Gemini: {e}")
                self.enable_llm_enhancement = False
        
        # Initialize NLTK
        self.nltk_available = NLTK_AVAILABLE
        if self.nltk_available:
            self._ensure_nltk_data()
        
        # Initialize services
        self.embedding_service = get_embedding_service()
        self.redis_cache = get_redis_cache() if enable_caching else None
        
        # Query enhancement templates
        self.enhancement_templates = self._load_enhancement_templates()
        
        # Statistics
        self.stats = {
            'total_queries_processed': 0,
            'llm_enhanced_queries': 0,
            'cached_queries': 0,
            'avg_processing_time': 0.0,
            'intent_distribution': {intent.value: 0 for intent in QueryIntent},
            'complexity_distribution': {complexity.value: 0 for complexity in QueryComplexity},
            'enhancement_success_rate': 0.0
        }
        
        logger.info("QueryEnhancementEngine initialized")
        logger.info(f"LLM enhancement: {self.enable_llm_enhancement}")
        logger.info(f"Caching: {enable_caching}")
        logger.info(f"NLTK available: {self.nltk_available}")
    
    def _ensure_nltk_data(self):
        """Ensure required NLTK data is downloaded"""
        required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for data in required_data:
            try:
                if data == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif data == 'stopwords':
                    nltk.data.find('corpora/stopwords')
                elif data == 'wordnet':
                    nltk.data.find('corpora/wordnet')
                elif data == 'averaged_perceptron_tagger':
                    nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                try:
                    nltk.download(data, quiet=True)
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download NLTK data {data}: {e}")
                    self.nltk_available = False
    
    def _load_enhancement_templates(self) -> Dict[str, str]:
        """Load query enhancement templates"""
        return {
            'expansion': """
            Enhance this user query for better document search:
            
            Original query: "{query}"
            
            Please provide:
            1. An expanded version with more context and related terms
            2. 3-5 semantic variations of the query
            3. Key concepts and entities mentioned
            4. The type of answer expected
            
            Format your response as JSON:
            {{
                "expanded_query": "enhanced version with context",
                "semantic_variants": ["variant1", "variant2", "variant3"],
                "key_concepts": ["concept1", "concept2"],
                "entities": ["entity1", "entity2"],
                "expected_answer_type": "factual/analytical/procedural/etc"
            }}
            """,
            
            'intent_analysis': """
            Analyze the intent and complexity of this query:
            
            Query: "{query}"
            
            Classify the intent as one of:
            - factual: Direct factual questions
            - analytical: Analysis or comparison questions  
            - procedural: How-to or process questions
            - definitional: What is, define, explain
            - numerical: Calculations, statistics
            - comparative: Compare, contrast, difference
            - causal: Why, cause, reason
            - temporal: Time-based questions
            
            Rate complexity as: simple, moderate, complex, very_complex
            
            Provide confidence score (0.0-1.0) and reasoning.
            
            Format as JSON:
            {{
                "intent": "intent_category",
                "complexity": "complexity_level", 
                "confidence": 0.0,
                "reasoning": "explanation"
            }}
            """,
            
            'context_analysis': """
            Determine context requirements for this query:
            
            Query: "{query}"
            
            Analyze:
            1. Does this query require additional context to answer properly?
            2. What type of documents would best answer this?
            3. How much context (tokens) would be needed?
            4. Are there temporal or domain-specific requirements?
            
            Format as JSON:
            {{
                "requires_context": true/false,
                "context_window": 512,
                "document_types": ["type1", "type2"],
                "temporal_requirements": ["requirement1"],
                "domain_specific": true/false
            }}
            """
        }
    
    async def enhance_query(self, query: str) -> EnhancedQuery:
        """
        Enhance user query with analysis, expansion, and optimization
        
        Args:
            query: Original user query
            
        Returns:
            EnhancedQuery with analysis and enhancements
        """
        start_time = time.time()
        self.stats['total_queries_processed'] += 1
        
        logger.debug(f"🔍 Enhancing query: {query}")
        
        # Check cache first
        cache_key = None
        if self.enable_caching and self.redis_cache:
            cache_key = self._generate_cache_key(query)
            cached_result = await self._get_cached_enhancement(cache_key)
            if cached_result:
                self.stats['cached_queries'] += 1
                logger.debug("✅ Using cached query enhancement")
                return cached_result
        
        try:
            # Step 1: Basic analysis
            analysis = await self._analyze_query_basic(query)
            
            # Step 2: LLM enhancement if enabled
            if self.enable_llm_enhancement and self.gemini_model:
                analysis = await self._enhance_with_llm(analysis)
                self.stats['llm_enhanced_queries'] += 1
            
            # Step 3: Generate embeddings
            enhanced_query = await self._generate_query_embeddings(analysis)
            
            # Step 4: Determine retrieval strategies
            enhanced_query = await self._determine_retrieval_strategies(enhanced_query)
            
            # Finalize processing
            analysis.processing_time = time.time() - start_time
            self._update_stats(analysis)
            
            # Cache result
            if self.enable_caching and cache_key:
                await self._cache_enhancement(cache_key, enhanced_query)
            
            logger.debug(f"✅ Query enhanced in {analysis.processing_time:.3f}s")
            logger.debug(f"Intent: {analysis.intent.value}, Complexity: {analysis.complexity.value}")
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"❌ Query enhancement failed: {str(e)}")
            
            # Return basic enhancement as fallback
            fallback_analysis = QueryAnalysis(
                original_query=query,
                intent=QueryIntent.FACTUAL,
                complexity=QueryComplexity.SIMPLE,
                confidence_score=0.1,
                expanded_query=query,
                processing_time=time.time() - start_time,
                enhancement_strategy="fallback"
            )
            
            return EnhancedQuery(analysis=fallback_analysis)
    
    async def _analyze_query_basic(self, query: str) -> QueryAnalysis:
        """Perform basic query analysis without LLM"""
        
        # Clean and normalize query
        cleaned_query = self._clean_query(query)
        
        # Detect intent using patterns
        intent = self._detect_intent(cleaned_query)
        
        # Assess complexity
        complexity = self._assess_complexity(cleaned_query)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(cleaned_query)
        
        # Extract entities (basic)
        entities = self._extract_basic_entities(cleaned_query)
        
        # Detect temporal indicators
        temporal_indicators = self._extract_temporal_indicators(cleaned_query)
        
        # Generate basic expansion
        expanded_query = self._expand_query_basic(cleaned_query, key_concepts)
        
        # Generate variants
        semantic_variants = self._generate_basic_variants(cleaned_query, key_concepts)
        
        return QueryAnalysis(
            original_query=query,
            intent=intent,
            complexity=complexity,
            confidence_score=0.7,  # Medium confidence for rule-based
            expanded_query=expanded_query,
            semantic_variants=semantic_variants,
            key_concepts=key_concepts,
            entities=entities,
            question_type=self._detect_question_type(cleaned_query),
            temporal_indicators=temporal_indicators,
            requires_context=complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX],
            context_window=self._determine_context_window(complexity),
            expected_answer_type=self._determine_answer_type(intent),
            enhancement_strategy="rule_based"
        )
    
    async def _enhance_with_llm(self, analysis: QueryAnalysis) -> QueryAnalysis:
        """Enhance query analysis using Gemini LLM"""
        
        try:
            # Query expansion
            expansion_prompt = self.enhancement_templates['expansion'].format(
                query=analysis.original_query
            )
            
            expansion_response = await self._call_gemini(expansion_prompt)
            expansion_data = self._parse_llm_response(expansion_response)
            
            if expansion_data:
                analysis.expanded_query = expansion_data.get('expanded_query', analysis.expanded_query)
                analysis.semantic_variants = expansion_data.get('semantic_variants', analysis.semantic_variants)
                analysis.key_concepts = expansion_data.get('key_concepts', analysis.key_concepts)
                analysis.entities = expansion_data.get('entities', analysis.entities)
                analysis.expected_answer_type = expansion_data.get('expected_answer_type', analysis.expected_answer_type)
            
            # Intent analysis refinement
            intent_prompt = self.enhancement_templates['intent_analysis'].format(
                query=analysis.original_query
            )
            
            intent_response = await self._call_gemini(intent_prompt)
            intent_data = self._parse_llm_response(intent_response)
            
            if intent_data:
                # Update intent if confidence is higher
                llm_confidence = intent_data.get('confidence', 0.0)
                if llm_confidence > analysis.confidence_score:
                    try:
                        analysis.intent = QueryIntent(intent_data.get('intent', analysis.intent.value))
                        analysis.complexity = QueryComplexity(intent_data.get('complexity', analysis.complexity.value))
                        analysis.confidence_score = llm_confidence
                    except ValueError:
                        # Keep original if LLM returned invalid values
                        pass
            
            # Context analysis
            context_prompt = self.enhancement_templates['context_analysis'].format(
                query=analysis.original_query
            )
            
            context_response = await self._call_gemini(context_prompt)
            context_data = self._parse_llm_response(context_response)
            
            if context_data:
                analysis.requires_context = context_data.get('requires_context', analysis.requires_context)
                analysis.context_window = context_data.get('context_window', analysis.context_window)
                # Could add more context-specific fields here
            
            analysis.llm_enhanced = True
            analysis.enhancement_strategy = "llm_enhanced"
            
            logger.debug("✅ Query enhanced with Gemini LLM")
            
        except Exception as e:
            logger.warning(f"⚠️ LLM enhancement failed, using rule-based: {str(e)}")
        
        return analysis
    
    async def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.gemini_model.generate_content, prompt
                )
                return response.text
                
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                wait_time = 2 ** attempt
                logger.debug(f"Gemini API retry {attempt + 1}/{max_retries} after {wait_time}s")
                await asyncio.sleep(wait_time)
        
        raise Exception("Gemini API calls exhausted")
    
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON-like content
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    return None
            
            return json.loads(json_str)
            
        except Exception as e:
            logger.debug(f"Failed to parse LLM response: {e}")
            return None
    
    async def _generate_query_embeddings(self, analysis: QueryAnalysis) -> EnhancedQuery:
        """Generate embeddings for query variants"""
        
        # Prepare texts for embedding
        texts_to_embed = [analysis.original_query, analysis.expanded_query]
        texts_to_embed.extend(analysis.semantic_variants[:3])  # Limit variants
        
        try:
            # Generate embeddings
            embedding_result = await self.embedding_service.generate_embeddings(texts_to_embed)
            
            if embedding_result.success:
                embeddings = embedding_result.embeddings
                
                original_embedding = embeddings[0].tolist() if len(embeddings) > 0 else None
                expanded_embedding = embeddings[1].tolist() if len(embeddings) > 1 else None
                variant_embeddings = [emb.tolist() for emb in embeddings[2:]]
                
                return EnhancedQuery(
                    analysis=analysis,
                    original_embedding=original_embedding,
                    expanded_embedding=expanded_embedding,
                    variant_embeddings=variant_embeddings
                )
            
        except Exception as e:
            logger.warning(f"⚠️ Embedding generation failed: {e}")
        
        # Return without embeddings as fallback
        return EnhancedQuery(analysis=analysis)
    
    async def _determine_retrieval_strategies(self, enhanced_query: EnhancedQuery) -> EnhancedQuery:
        """Determine optimal retrieval strategies based on query analysis"""
        
        analysis = enhanced_query.analysis
        strategies = []
        thresholds = {}
        
        # Base strategy selection
        if analysis.complexity == QueryComplexity.SIMPLE:
            strategies.append("semantic_search")
            thresholds["semantic"] = 0.7
        elif analysis.complexity == QueryComplexity.MODERATE:
            strategies.extend(["semantic_search", "keyword_search"])
            thresholds["semantic"] = 0.6
            thresholds["keyword"] = 0.5
        else:  # Complex queries
            strategies.extend(["hybrid_search", "multi_stage_retrieval"])
            thresholds["semantic"] = 0.5
            thresholds["keyword"] = 0.4
            enhanced_query.multi_stage_retrieval = True
        
        # Intent-specific adjustments
        if analysis.intent == QueryIntent.NUMERICAL:
            strategies.append("exact_match")
            thresholds["exact"] = 0.9
        elif analysis.intent == QueryIntent.TEMPORAL:
            strategies.append("temporal_search")
            thresholds["temporal"] = 0.6
        elif analysis.intent == QueryIntent.COMPARATIVE:
            strategies.append("comparative_search")
            enhanced_query.reranking_enabled = True
        
        # Set retrieval parameters
        enhanced_query.retrieval_strategies = strategies
        enhanced_query.similarity_thresholds = thresholds
        
        # Adjust top_k based on complexity
        if analysis.complexity == QueryComplexity.VERY_COMPLEX:
            enhanced_query.top_k_suggestions = 20
        elif analysis.complexity == QueryComplexity.COMPLEX:
            enhanced_query.top_k_suggestions = 15
        else:
            enhanced_query.top_k_suggestions = 10
        
        return enhanced_query
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text"""
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Remove special characters but keep important punctuation
        query = re.sub(r'[^\w\s\?\!\.\,\-\(\)]', ' ', query)
        
        # Normalize case
        query = query.lower()
        
        return query.strip()
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect query intent using pattern matching"""
        intent_scores = {}
        
        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent, default to FACTUAL
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return QueryIntent.FACTUAL
    
    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess query complexity based on heuristics"""
        words = query.split()
        word_count = len(words)
        
        # Count concepts (simplified)
        concept_indicators = ['and', 'or', 'but', 'however', 'also', 'additionally']
        concept_count = 1 + sum(1 for word in words if word in concept_indicators)
        
        # Check for complexity indicators
        complexity_words = ['analyze', 'compare', 'evaluate', 'assess', 'comprehensive', 'detailed']
        has_complexity_words = any(word in query for word in complexity_words)
        
        # Question marks indicating multiple questions
        question_count = query.count('?')
        
        # Calculate complexity score
        complexity_score = 0.0
        
        if word_count > 30:
            complexity_score += 0.3
        elif word_count > 15:
            complexity_score += 0.2
        
        if concept_count > 3:
            complexity_score += 0.3
        elif concept_count > 2:
            complexity_score += 0.2
        
        if has_complexity_words:
            complexity_score += 0.3
        
        if question_count > 1:
            complexity_score += 0.2
        
        # Map score to complexity level
        if complexity_score >= 0.8:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_score >= 0.6:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 0.3:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from query"""
        if not self.nltk_available:
            # Simple keyword extraction
            words = query.split()
            # Filter common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'why', 'who'}
            keywords = [word for word in words if word.lower() not in stop_words and len(word) > 2]
            return keywords[:5]  # Limit to top 5
        
        try:
            # Use NLTK for better extraction
            tokens = word_tokenize(query)
            stop_words = set(stopwords.words('english'))
            
            # POS tagging to identify nouns
            tagged = nltk.pos_tag(tokens)
            
            # Extract nouns and adjectives as key concepts
            key_concepts = []
            for word, pos in tagged:
                if pos.startswith('NN') or pos.startswith('JJ'):  # Nouns and adjectives
                    if word.lower() not in stop_words and len(word) > 2:
                        key_concepts.append(word.lower())
            
            return key_concepts[:5]  # Limit to top 5
            
        except Exception:
            # Fallback to simple extraction
            return self._extract_key_concepts(query)
    
    def _extract_basic_entities(self, query: str) -> List[str]:
        """Extract basic entities (simplified NER)"""
        # Simple patterns for common entities
        entities = []
        
        # Numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, query)
        entities.extend([f"NUMBER_{num}" for num in numbers])
        
        # Dates (simple patterns)
        date_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',  # Months
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities.extend([f"DATE_{match}" for match in matches])
        
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        entities.extend([f"ENTITY_{word}" for word in capitalized])
        
        return entities[:10]  # Limit entities
    
    def _extract_temporal_indicators(self, query: str) -> List[str]:
        """Extract temporal indicators from query"""
        temporal_patterns = [
            r'\b(?:before|after|during|since|until|by|from)\b',
            r'\b(?:yesterday|today|tomorrow|now|recently|currently)\b',
            r'\b(?:last|next|this|previous)\s+(?:year|month|week|day)\b',
            r'\b\d{4}\b',  # Years
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\b'  # Dates
        ]
        
        indicators = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            indicators.extend(matches)
        
        return indicators
    
    def _detect_question_type(self, query: str) -> str:
        """Detect the type of question being asked"""
        question_words = {
            'what': 'factual',
            'who': 'person',
            'when': 'temporal',
            'where': 'location',
            'why': 'causal',
            'how': 'procedural',
            'which': 'choice',
            'can': 'capability',
            'should': 'recommendation',
            'will': 'prediction'
        }
        
        words = query.split()
        for word in words:
            if word in question_words:
                return question_words[word]
        
        return 'general'
    
    def _expand_query_basic(self, query: str, key_concepts: List[str]) -> str:
        """Basic query expansion without LLM"""
        # Add synonyms and related terms (simplified)
        expanded_parts = [query]
        
        # Add key concepts if not already in query
        for concept in key_concepts:
            if concept not in query:
                expanded_parts.append(concept)
        
        # Add domain-specific terms based on context
        domain_terms = self._get_domain_terms(query)
        expanded_parts.extend(domain_terms)
        
        return ' '.join(expanded_parts)
    
    def _generate_basic_variants(self, query: str, key_concepts: List[str]) -> List[str]:
        """Generate basic query variants"""
        variants = []
        
        # Rephrase with different question words
        if query.startswith('what'):
            variants.append(query.replace('what', 'which', 1))
        elif query.startswith('how'):
            variants.append(query.replace('how', 'what way', 1))
        
        # Add concept-based variants
        if len(key_concepts) >= 2:
            # Reorder key concepts
            variant = f"{key_concepts[1]} {key_concepts[0]} " + ' '.join(key_concepts[2:])
            variants.append(variant)
        
        # Add more specific or general versions
        if len(query.split()) > 5:
            # More general
            general_variant = ' '.join(key_concepts[:3])
            variants.append(general_variant)
        else:
            # More specific
            specific_variant = query + " details information"
            variants.append(specific_variant)
        
        return variants[:self.max_query_variants]
    
    def _get_domain_terms(self, query: str) -> List[str]:
        """Get domain-specific terms based on query context"""
        domain_terms = []
        
        # Financial terms
        if any(term in query for term in ['bank', 'finance', 'loan', 'credit', 'investment']):
            domain_terms.extend(['financial', 'monetary', 'economic', 'fiscal'])
        
        # Technical terms
        if any(term in query for term in ['system', 'process', 'method', 'technology']):
            domain_terms.extend(['technical', 'operational', 'systematic'])
        
        # Business terms
        if any(term in query for term in ['company', 'business', 'organization', 'corporate']):
            domain_terms.extend(['enterprise', 'commercial', 'organizational'])
        
        return domain_terms[:3]  # Limit domain terms
    
    def _determine_context_window(self, complexity: QueryComplexity) -> int:
        """Determine optimal context window based on complexity"""
        context_windows = {
            QueryComplexity.SIMPLE: 256,
            QueryComplexity.MODERATE: 512,
            QueryComplexity.COMPLEX: 1024,
            QueryComplexity.VERY_COMPLEX: 2048
        }
        
        return context_windows.get(complexity, 512)
    
    def _determine_answer_type(self, intent: QueryIntent) -> str:
        """Determine expected answer type based on intent"""
        answer_types = {
            QueryIntent.FACTUAL: "factual_information",
            QueryIntent.ANALYTICAL: "analysis_report",
            QueryIntent.PROCEDURAL: "step_by_step_guide",
            QueryIntent.DEFINITIONAL: "definition_explanation",
            QueryIntent.NUMERICAL: "numerical_data",
            QueryIntent.COMPARATIVE: "comparison_analysis",
            QueryIntent.CAUSAL: "causal_explanation",
            QueryIntent.TEMPORAL: "temporal_information",
            QueryIntent.HYPOTHETICAL: "scenario_analysis",
            QueryIntent.MULTI_PART: "comprehensive_response"
        }
        
        return answer_types.get(intent, "general_information")
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"query_enhancement:{query_hash}"
    
    async def _get_cached_enhancement(self, cache_key: str) -> Optional[EnhancedQuery]:
        """Get cached query enhancement"""
        if not self.redis_cache:
            return None
        
        try:
            cached_data = await self.redis_cache.get(cache_key)
            if cached_data:
                # Would need proper deserialization
                logger.debug("Found cached query enhancement")
                return None  # TODO: Implement proper serialization
        except Exception as e:
            logger.debug(f"Cache get failed: {e}")
        
        return None
    
    async def _cache_enhancement(self, cache_key: str, enhanced_query: EnhancedQuery):
        """Cache query enhancement result"""
        if not self.redis_cache:
            return
        
        try:
            # Would need proper serialization
            # cache_data = serialize_enhanced_query(enhanced_query)
            # await self.redis_cache.setex(cache_key, self.cache_ttl, cache_data)
            logger.debug("Cached query enhancement")
        except Exception as e:
            logger.debug(f"Cache set failed: {e}")
    
    def _update_stats(self, analysis: QueryAnalysis):
        """Update enhancement statistics"""
        self.stats['intent_distribution'][analysis.intent.value] += 1
        self.stats['complexity_distribution'][analysis.complexity.value] += 1
        
        # Update average processing time
        total_queries = self.stats['total_queries_processed']
        current_avg = self.stats['avg_processing_time']
        
        self.stats['avg_processing_time'] = (
            (current_avg * (total_queries - 1) + analysis.processing_time) / total_queries
        )
        
        # Update enhancement success rate
        if analysis.llm_enhanced or analysis.confidence_score > 0.5:
            successful_enhancements = sum(1 for intent in self.stats['intent_distribution'].values())
            self.stats['enhancement_success_rate'] = (successful_enhancements / total_queries * 100)
    
    async def get_enhancement_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhancement statistics"""
        return {
            "processing_stats": self.stats,
            "configuration": {
                "llm_enhancement_enabled": self.enable_llm_enhancement,
                "caching_enabled": self.enable_caching,
                "cache_ttl": self.cache_ttl,
                "max_query_variants": self.max_query_variants,
                "nltk_available": self.nltk_available
            },
            "capabilities": {
                "intent_types": [intent.value for intent in QueryIntent],
                "complexity_levels": [complexity.value for complexity in QueryComplexity],
                "enhancement_strategies": ["rule_based", "llm_enhanced", "hybrid"]
            }
        }


# Global query enhancement engine instance
query_enhancement_engine: Optional[QueryEnhancementEngine] = None


def get_query_enhancement_engine(**kwargs) -> QueryEnhancementEngine:
    """Get or create global query enhancement engine instance"""
    global query_enhancement_engine
    
    if query_enhancement_engine is None:
        query_enhancement_engine = QueryEnhancementEngine(**kwargs)
    
    return query_enhancement_engine


async def initialize_query_enhancement_engine(**kwargs) -> QueryEnhancementEngine:
    """Initialize and return query enhancement engine"""
    engine = get_query_enhancement_engine(**kwargs)
    
    # Initialize embedding service
    await engine.embedding_service.initialize()
    
    # Log initialization summary
    stats = await engine.get_enhancement_stats()
    logger.info("🔍 Query Enhancement Engine Summary:")
    logger.info(f"  LLM enhancement: {stats['configuration']['llm_enhancement_enabled']}")
    logger.info(f"  Caching: {stats['configuration']['caching_enabled']}")
    logger.info(f"  NLTK available: {stats['configuration']['nltk_available']}")
    logger.info(f"  Intent types: {len(stats['capabilities']['intent_types'])}")
    logger.info(f"  Complexity levels: {len(stats['capabilities']['complexity_levels'])}")
    
    return engine