"""
Smart Indexing Engine with document structure understanding and section-aware indexing
Optimized for accurate retrieval in large insurance, legal, HR, and compliance documents
"""

import asyncio
import logging
import re
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict, Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from src.core.config import settings
from src.core.advanced_document_processor import DocumentChunk, DocumentSection
from src.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SectionType(Enum):
    TABLE_OF_CONTENTS = "table_of_contents"
    DEFINITIONS = "definitions"
    MAIN_CONTENT = "main_content"
    CLAUSES = "clauses"
    PROCEDURES = "procedures"
    EXCLUSIONS = "exclusions"
    SCHEDULES = "schedules"
    APPENDIX = "appendix"
    FOOTNOTES = "footnotes"


@dataclass
class StructuralMetadata:
    section_hierarchy: Dict[str, Any]
    cross_references: List[Tuple[str, str]]
    entity_mappings: Dict[str, List[str]]
    concept_clusters: Dict[str, List[str]]
    importance_scores: Dict[str, float]


@dataclass
class IndexedChunk:
    chunk: DocumentChunk
    structural_features: Dict[str, Any]
    semantic_features: Dict[str, Any]
    connectivity_score: float
    authority_score: float
    section_context: List[str]


class SmartIndexingEngine:
    """
    Advanced indexing engine that understands document structure and creates
    intelligent indices for accurate retrieval in large documents
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
        # NLP models
        self.nlp = None  # Will be initialized lazily
        
        # Structural analysis
        self.section_classifier = SectionClassifier()
        self.cross_reference_analyzer = CrossReferenceAnalyzer()
        self.entity_extractor = EntityExtractor()
        
        # Index structures
        self.hierarchical_index = {}  # Section hierarchy
        self.semantic_clusters = {}   # Concept-based clusters
        self.cross_ref_graph = nx.DiGraph()  # Cross-reference graph
        self.entity_index = defaultdict(list)  # Entity-based index
        self.importance_index = {}    # Chunk importance scores
        
        # Performance tracking
        self.indexing_stats = {
            "total_chunks_indexed": 0,
            "sections_identified": 0,
            "cross_references_found": 0,
            "entities_extracted": 0,
            "clusters_created": 0
        }

    async def initialize_nlp(self):
        """Initialize NLP models (lazy loading for performance)"""
        if self.nlp is None:
            try:
                # Try to load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy NLP model loaded successfully")
            except OSError:
                logger.warning("SpaCy model not found, using basic processing")
                self.nlp = None

    async def create_smart_index(
        self, 
        chunks: List[DocumentChunk], 
        sections: List[DocumentSection],
        document_metadata: Dict[str, Any]
    ) -> StructuralMetadata:
        """
        Create comprehensive smart index with structural understanding
        """
        logger.info(f"Creating smart index for {len(chunks)} chunks and {len(sections)} sections")
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize NLP if needed
            await self.initialize_nlp()
            
            # Phase 1: Structural Analysis
            logger.info("Phase 1: Analyzing document structure")
            await self._analyze_document_structure(sections, document_metadata)
            
            # Phase 2: Section Classification and Hierarchy
            logger.info("Phase 2: Building section hierarchy")
            section_hierarchy = await self._build_section_hierarchy(sections, chunks)
            
            # Phase 3: Cross-reference Analysis
            logger.info("Phase 3: Analyzing cross-references")
            cross_references = await self._analyze_cross_references(chunks)
            
            # Phase 4: Entity Extraction and Mapping
            logger.info("Phase 4: Extracting and mapping entities")
            entity_mappings = await self._extract_and_map_entities(chunks)
            
            # Phase 5: Semantic Clustering
            logger.info("Phase 5: Creating semantic clusters")
            concept_clusters = await self._create_semantic_clusters(chunks)
            
            # Phase 6: Importance Scoring
            logger.info("Phase 6: Calculating importance scores")
            importance_scores = await self._calculate_importance_scores(chunks, cross_references)
            
            # Phase 7: Create Enhanced Index
            logger.info("Phase 7: Building enhanced indices")
            await self._build_enhanced_indices(chunks, section_hierarchy, cross_references, entity_mappings)
            
            indexing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Smart indexing completed in {indexing_time:.2f}s")
            
            # Update stats
            self.indexing_stats["total_chunks_indexed"] = len(chunks)
            self.indexing_stats["sections_identified"] = len(sections)
            self.indexing_stats["cross_references_found"] = len(cross_references)
            self.indexing_stats["entities_extracted"] = sum(len(entities) for entities in entity_mappings.values())
            self.indexing_stats["clusters_created"] = len(concept_clusters)
            
            return StructuralMetadata(
                section_hierarchy=section_hierarchy,
                cross_references=cross_references,
                entity_mappings=entity_mappings,
                concept_clusters=concept_clusters,
                importance_scores=importance_scores
            )
            
        except Exception as e:
            logger.error(f"Error creating smart index: {str(e)}")
            raise

    async def _analyze_document_structure(
        self, sections: List[DocumentSection], document_metadata: Dict[str, Any]
    ):
        """Analyze overall document structure and patterns"""
        
        doc_type = document_metadata.get("document_type", "general")
        
        # Analyze section patterns
        section_patterns = {}
        for section in sections:
            section_type = self.section_classifier.classify_section(section, doc_type)
            if section_type not in section_patterns:
                section_patterns[section_type] = []
            section_patterns[section_type].append(section)
        
        logger.info(f"Identified section patterns: {list(section_patterns.keys())}")

    async def _build_section_hierarchy(
        self, sections: List[DocumentSection], chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """Build hierarchical section structure"""
        
        hierarchy = {
            "root": {
                "level": 0,
                "children": [],
                "chunks": [],
                "metadata": {}
            }
        }
        
        # Sort sections by level and position
        sorted_sections = sorted(sections, key=lambda x: (x.level, x.page_start))
        
        # Build hierarchy tree
        section_stack = [hierarchy["root"]]
        
        for section in sorted_sections:
            section_node = {
                "title": section.title,
                "level": section.level,
                "section_type": section.section_type,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "children": [],
                "chunks": [],
                "metadata": section.metadata or {}
            }
            
            # Find parent level
            while len(section_stack) > 1 and section_stack[-1]["level"] >= section.level:
                section_stack.pop()
            
            # Add to parent
            parent = section_stack[-1]
            parent["children"].append(section_node)
            section_stack.append(section_node)
            
            # Assign chunks to sections
            for chunk in chunks:
                chunk_page = chunk.metadata.get("page_start", chunk.page_num)
                if section.page_start <= chunk_page <= section.page_end:
                    section_node["chunks"].append(chunk.chunk_id)
        
        self.hierarchical_index = hierarchy
        return hierarchy

    async def _analyze_cross_references(
        self, chunks: List[DocumentChunk]
    ) -> List[Tuple[str, str]]:
        """Analyze cross-references between document sections"""
        
        cross_references = []
        
        # Pattern matching for cross-references
        cross_ref_patterns = [
            r"(?:see|refer to|as per|according to)\s+(?:section|clause|article|paragraph)\s+(\d+(?:\.\d+)*)",
            r"(?:section|clause|article|paragraph)\s+(\d+(?:\.\d+)*)\s+(?:above|below|herein)",
            r"(?:defined in|mentioned in)\s+(?:section|clause|article|paragraph)\s+(\d+(?:\.\d+)*)",
            r"(?:page|p\.)\s+(\d+)",
            r"(?:appendix|schedule|annexure)\s+([A-Z\d]+)"
        ]
        
        for chunk in chunks:
            chunk_refs = self.cross_reference_analyzer.extract_references(
                chunk.text, cross_ref_patterns
            )
            
            for ref in chunk_refs:
                cross_references.append((chunk.chunk_id, ref))
                # Add to graph
                self.cross_ref_graph.add_edge(chunk.chunk_id, ref)
        
        return cross_references

    async def _extract_and_map_entities(
        self, chunks: List[DocumentChunk]
    ) -> Dict[str, List[str]]:
        """Extract and map entities across the document"""
        
        entity_mappings = defaultdict(list)
        
        for chunk in chunks:
            entities = await self.entity_extractor.extract_entities(chunk.text, self.nlp)
            
            for entity_type, entity_value in entities:
                entity_mappings[entity_type].append({
                    "value": entity_value,
                    "chunk_id": chunk.chunk_id,
                    "context": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                })
                
                # Add to entity index
                self.entity_index[entity_value].append(chunk.chunk_id)
        
        return dict(entity_mappings)

    async def _create_semantic_clusters(
        self, chunks: List[DocumentChunk]
    ) -> Dict[str, List[str]]:
        """Create semantic clusters of related content"""
        
        if len(chunks) < 10:  # Not enough chunks for meaningful clustering
            return {}
        
        try:
            # Prepare texts for clustering
            texts = [chunk.text for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            
            # TF-IDF vectorization
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Determine optimal number of clusters
            n_clusters = min(10, max(3, len(chunks) // 20))
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Organize clusters
            clusters = defaultdict(list)
            cluster_keywords = {}
            
            feature_names = vectorizer.get_feature_names_out()
            
            for i in range(n_clusters):
                cluster_name = f"cluster_{i}"
                
                # Get chunks in this cluster
                cluster_chunks = [chunk_ids[j] for j, label in enumerate(cluster_labels) if label == i]
                clusters[cluster_name] = cluster_chunks
                
                # Extract top keywords for cluster
                centroid = kmeans.cluster_centers_[i]
                top_indices = centroid.argsort()[-10:][::-1]
                keywords = [feature_names[idx] for idx in top_indices]
                cluster_keywords[cluster_name] = keywords
                
                logger.debug(f"Cluster {i}: {len(cluster_chunks)} chunks, keywords: {keywords[:5]}")
            
            # Store semantic clusters
            self.semantic_clusters = dict(clusters)
            
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Error in semantic clustering: {str(e)}")
            return {}

    async def _calculate_importance_scores(
        self, chunks: List[DocumentChunk], cross_references: List[Tuple[str, str]]
    ) -> Dict[str, float]:
        """Calculate importance scores for chunks based on multiple factors"""
        
        importance_scores = {}
        
        # Initialize base scores
        for chunk in chunks:
            importance_scores[chunk.chunk_id] = 1.0
        
        # Factor 1: Section-based importance
        for chunk in chunks:
            section_title = chunk.metadata.get("section_title", "").lower()
            
            # High importance sections
            if any(term in section_title for term in [
                "definition", "coverage", "benefit", "exclusion", "claim", "premium",
                "procedure", "general condition", "policy schedule"
            ]):
                importance_scores[chunk.chunk_id] *= 1.5
            
            # Medium importance sections
            elif any(term in section_title for term in [
                "term", "condition", "requirement", "obligation", "right"
            ]):
                importance_scores[chunk.chunk_id] *= 1.2
        
        # Factor 2: Cross-reference authority (PageRank-like)
        if cross_references:
            try:
                # Calculate PageRank on cross-reference graph
                pagerank_scores = nx.pagerank(self.cross_ref_graph, alpha=0.85, max_iter=100)
                
                for chunk_id, pr_score in pagerank_scores.items():
                    if chunk_id in importance_scores:
                        importance_scores[chunk_id] *= (1 + pr_score * 2)  # Boost based on authority
                        
            except Exception as e:
                logger.warning(f"PageRank calculation failed: {str(e)}")
        
        # Factor 3: Content density and quality
        for chunk in chunks:
            text = chunk.text
            
            # Higher importance for chunks with numbers/percentages
            numeric_content = len(re.findall(r'\d+(?:\.\d+)?[%₹]?', text))
            if numeric_content > 0:
                importance_scores[chunk.chunk_id] *= (1 + numeric_content * 0.1)
            
            # Higher importance for definition-like content
            if re.search(r'\b(?:means|refers to|defined as|shall mean)\b', text, re.IGNORECASE):
                importance_scores[chunk.chunk_id] *= 1.3
            
            # Higher importance for procedural content
            if re.search(r'\b(?:must|shall|required to|procedure|process|step)\b', text, re.IGNORECASE):
                importance_scores[chunk.chunk_id] *= 1.2
        
        # Normalize scores
        max_score = max(importance_scores.values()) if importance_scores else 1.0
        for chunk_id in importance_scores:
            importance_scores[chunk_id] /= max_score
        
        self.importance_index = importance_scores
        return importance_scores

    async def _build_enhanced_indices(
        self,
        chunks: List[DocumentChunk],
        section_hierarchy: Dict[str, Any],
        cross_references: List[Tuple[str, str]],
        entity_mappings: Dict[str, List[str]]
    ):
        """Build enhanced retrieval indices"""
        
        # Create indexed chunks with enhanced metadata
        indexed_chunks = []
        
        for chunk in chunks:
            # Structural features
            structural_features = {
                "section_path": self._get_section_path(chunk, section_hierarchy),
                "section_type": chunk.metadata.get("section_type", "unknown"),
                "page_range": f"{chunk.metadata.get('page_start', chunk.page_num)}-{chunk.metadata.get('page_end', chunk.page_num)}",
                "hierarchy_level": self._get_hierarchy_level(chunk, section_hierarchy)
            }
            
            # Semantic features
            semantic_features = {
                "cluster_membership": self._get_cluster_membership(chunk.chunk_id),
                "entity_density": self._calculate_entity_density(chunk.text, entity_mappings),
                "content_type": self._classify_content_type(chunk.text)
            }
            
            # Connectivity and authority scores
            connectivity_score = self._calculate_connectivity_score(chunk.chunk_id, cross_references)
            authority_score = self.importance_index.get(chunk.chunk_id, 1.0)
            
            # Section context
            section_context = self._get_section_context(chunk, section_hierarchy)
            
            indexed_chunk = IndexedChunk(
                chunk=chunk,
                structural_features=structural_features,
                semantic_features=semantic_features,
                connectivity_score=connectivity_score,
                authority_score=authority_score,
                section_context=section_context
            )
            
            indexed_chunks.append(indexed_chunk)
        
        logger.info(f"Built enhanced indices for {len(indexed_chunks)} chunks")

    def _get_section_path(self, chunk: DocumentChunk, section_hierarchy: Dict[str, Any]) -> str:
        """Get the hierarchical path to the chunk's section"""
        # This would traverse the hierarchy to find the chunk's path
        # Simplified implementation
        section_title = chunk.metadata.get("section_title", "")
        return section_title if section_title else "root"

    def _get_hierarchy_level(self, chunk: DocumentChunk, section_hierarchy: Dict[str, Any]) -> int:
        """Get the hierarchy level of the chunk's section"""
        return chunk.metadata.get("section_level", 1)

    def _get_cluster_membership(self, chunk_id: str) -> List[str]:
        """Get cluster memberships for a chunk"""
        memberships = []
        for cluster_name, chunk_ids in self.semantic_clusters.items():
            if chunk_id in chunk_ids:
                memberships.append(cluster_name)
        return memberships

    def _calculate_entity_density(self, text: str, entity_mappings: Dict[str, List[str]]) -> float:
        """Calculate entity density in the text"""
        entity_count = 0
        word_count = len(text.split())
        
        for entity_type, entities in entity_mappings.items():
            for entity_info in entities:
                if entity_info["value"].lower() in text.lower():
                    entity_count += 1
        
        return entity_count / word_count if word_count > 0 else 0.0

    def _classify_content_type(self, text: str) -> str:
        """Classify the type of content in the text"""
        text_lower = text.lower()
        
        if re.search(r'\b(?:definition|means|refers to|defined as)\b', text_lower):
            return "definition"
        elif re.search(r'\b(?:procedure|process|step|method)\b', text_lower):
            return "procedural"
        elif re.search(r'\b(?:exclusion|exception|not covered|limitation)\b', text_lower):
            return "exclusion"
        elif re.search(r'\b(?:benefit|coverage|protection|insurance)\b', text_lower):
            return "benefit"
        elif re.search(r'\b(?:claim|application|submission)\b', text_lower):
            return "claims"
        else:
            return "general"

    def _calculate_connectivity_score(self, chunk_id: str, cross_references: List[Tuple[str, str]]) -> float:
        """Calculate connectivity score based on cross-references"""
        incoming_refs = sum(1 for src, dst in cross_references if dst == chunk_id)
        outgoing_refs = sum(1 for src, dst in cross_references if src == chunk_id)
        
        return (incoming_refs * 2 + outgoing_refs) / max(1, len(cross_references))

    def _get_section_context(self, chunk: DocumentChunk, section_hierarchy: Dict[str, Any]) -> List[str]:
        """Get contextual information about the chunk's section"""
        context = []
        
        section_title = chunk.metadata.get("section_title", "")
        if section_title:
            context.append(f"Section: {section_title}")
        
        section_type = chunk.metadata.get("section_type", "")
        if section_type:
            context.append(f"Type: {section_type}")
        
        page_info = chunk.metadata.get("page_start")
        if page_info:
            context.append(f"Page: {page_info}")
        
        return context

    async def search_with_structure(
        self, 
        query: str, 
        k: int = 10,
        structure_weight: float = 0.3
    ) -> List[Tuple[DocumentChunk, float, Dict[str, Any]]]:
        """
        Search with structural understanding and enhanced ranking
        """
        
        try:
            # Base semantic search
            base_results = await self.vector_store.similarity_search(query, k * 2)
            
            # Enhance results with structural scoring
            enhanced_results = []
            
            for chunk, base_score in base_results:
                # Get structural enhancements
                importance_boost = self.importance_index.get(chunk.chunk_id, 1.0)
                
                # Calculate structural relevance
                structural_relevance = self._calculate_structural_relevance(query, chunk)
                
                # Combine scores
                final_score = (
                    base_score * (1 - structure_weight) + 
                    structural_relevance * structure_weight
                ) * importance_boost
                
                # Additional context
                context = {
                    "importance_score": importance_boost,
                    "structural_relevance": structural_relevance,
                    "section_context": self._get_section_context(chunk, self.hierarchical_index)
                }
                
                enhanced_results.append((chunk, final_score, context))
            
            # Sort by enhanced score
            enhanced_results.sort(key=lambda x: x[1], reverse=True)
            
            return enhanced_results[:k]
            
        except Exception as e:
            logger.error(f"Error in structural search: {str(e)}")
            # Fallback to basic search
            basic_results = await self.vector_store.similarity_search(query, k)
            return [(chunk, score, {}) for chunk, score in basic_results]

    def _calculate_structural_relevance(self, query: str, chunk: DocumentChunk) -> float:
        """Calculate structural relevance based on query and chunk context"""
        relevance = 0.5  # Base relevance
        
        query_lower = query.lower()
        chunk_section = chunk.metadata.get("section_title", "").lower()
        
        # Section type matching
        if "definition" in query_lower and "definition" in chunk_section:
            relevance += 0.3
        elif "procedure" in query_lower and any(term in chunk_section for term in ["procedure", "process", "claim"]):
            relevance += 0.3
        elif "exclusion" in query_lower and "exclusion" in chunk_section:
            relevance += 0.3
        elif "benefit" in query_lower and any(term in chunk_section for term in ["benefit", "coverage"]):
            relevance += 0.3
        
        # Content type matching
        content_type = self._classify_content_type(chunk.text)
        if content_type == "definition" and any(term in query_lower for term in ["what is", "define", "meaning"]):
            relevance += 0.2
        elif content_type == "procedural" and any(term in query_lower for term in ["how to", "process", "procedure"]):
            relevance += 0.2
        
        return min(1.0, relevance)

    def get_indexing_stats(self) -> Dict[str, Any]:
        """Get indexing statistics"""
        return self.indexing_stats


class SectionClassifier:
    """Classifier for document sections"""
    
    def classify_section(self, section: DocumentSection, doc_type: str) -> SectionType:
        """Classify a section based on its content and document type"""
        
        title_lower = section.title.lower()
        content_lower = section.content[:500].lower()  # First 500 chars
        
        # Check for specific section types
        if "table of contents" in title_lower or "contents" in title_lower:
            return SectionType.TABLE_OF_CONTENTS
        
        elif any(term in title_lower for term in ["definition", "interpretation", "meaning"]):
            return SectionType.DEFINITIONS
        
        elif any(term in title_lower for term in ["exclusion", "exception", "limitation"]):
            return SectionType.EXCLUSIONS
        
        elif any(term in title_lower for term in ["procedure", "process", "claim"]):
            return SectionType.PROCEDURES
        
        elif any(term in title_lower for term in ["schedule", "table", "rate"]):
            return SectionType.SCHEDULES
        
        elif any(term in title_lower for term in ["appendix", "annex", "attachment"]):
            return SectionType.APPENDIX
        
        elif any(term in title_lower for term in ["clause", "article", "section"]):
            return SectionType.CLAUSES
        
        else:
            return SectionType.MAIN_CONTENT


class CrossReferenceAnalyzer:
    """Analyzer for cross-references in documents"""
    
    def extract_references(self, text: str, patterns: List[str]) -> List[str]:
        """Extract cross-references using pattern matching"""
        
        references = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.groups():
                    references.append(match.group(1))
                else:
                    references.append(match.group(0))
        
        return list(set(references))  # Remove duplicates


class EntityExtractor:
    """Entity extractor for domain-specific entities"""
    
    async def extract_entities(self, text: str, nlp_model=None) -> List[Tuple[str, str]]:
        """Extract domain-specific entities from text"""
        
        entities = []
        
        # Financial entities
        money_pattern = r'(?:Rs\.?|₹)\s*([\d,]+(?:\.\d+)?)'
        money_matches = re.finditer(money_pattern, text, re.IGNORECASE)
        for match in money_matches:
            entities.append(("MONEY", match.group(0)))
        
        # Percentage entities
        percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
        percent_matches = re.finditer(percent_pattern, text)
        for match in percent_matches:
            entities.append(("PERCENTAGE", match.group(0)))
        
        # Time period entities
        time_pattern = r'(\d+)\s*(days?|months?|years?)'
        time_matches = re.finditer(time_pattern, text, re.IGNORECASE)
        for match in time_matches:
            entities.append(("TIME_PERIOD", match.group(0)))
        
        # Use spaCy if available
        if nlp_model:
            try:
                doc = nlp_model(text[:1000])  # Limit text length for performance
                for ent in doc.ents:
                    if ent.label_ in ["PERSON", "ORG", "DATE", "CARDINAL", "ORDINAL"]:
                        entities.append((ent.label_, ent.text))
            except Exception as e:
                logger.warning(f"spaCy entity extraction failed: {str(e)}")
        
        return entities