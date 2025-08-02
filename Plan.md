# Project Plan: High-Performance, High-Accuracy RAG System

This document outlines the plan to enhance the existing RAG system for superior accuracy and robustness while maintaining its high-performance, parallel architecture. The plan integrates advanced data preparation techniques, a production-grade vector database (Pinecone), and a sophisticated retrieval pipeline.

---

## Phase 1: Foundational Data Preparation & Retrieval

**Objective:** To build a high-quality, indexed data foundation and implement a state-of-the-art retrieval mechanism. This phase will provide the most significant initial boost to accuracy.

### 1.1. Enhanced Data Cleaning and Preprocessing
-   **File:** `src/services/text_extraction_service.py`
-   **Actions:**
    -   **Deduplication:** Implement a content-based hashing mechanism (e.g., SHA-256) for all text chunks. Before processing, check if a chunk's hash has already been seen. This will prevent duplicate content from being stored in the vector database, reducing noise and storage costs.
    -   **Text Standardization:** Enhance the existing `_clean_text` methods to include more robust rules:
        -   Normalize Unicode characters to a standard form (NFKC).
        -   Implement a configurable mapping for special characters and symbols.
        -   Remove remnant headers, footers, and page numbers that may have been missed during text extraction.
    -   **Stemming/Lemmatization Decision:** We will *not* implement stemming or lemmatization by default. Modern embedding models (like `text-embedding-3-small`) are sensitive to word form and can capture nuances that stemming would destroy. This can be revisited if the evaluation phase reveals issues with word-form matching.

### 1.2. Upgrade to Semantic Chunking
-   **File:** `src/services/text_extraction_service.py`
-   **Actions:**
    -   Replace the current fixed-size chunking logic with a semantic chunking strategy based on sentence boundaries. Libraries like `nltk` or `spaCy` can be used for robust sentence tokenization.
    -   The chunking logic will aim to create chunks that are as close to `CHUNK_SIZE` as possible without exceeding it, while respecting sentence boundaries.
    -   Maintain the `CHUNK_OVERLAP` functionality, but the overlap will also be semantic (i.e., overlapping by a certain number of sentences rather than characters).
    -   The configuration will allow for easy experimentation with different chunking parameters.

### 1.3. Integrate Pinecone Vector Database
-   **Files:**
    -   Create `src/services/pinecone_service.py`.
    -   Update `src/core/config.py` with Pinecone settings.
    -   Update `src/services/rag_workflow.py` to use the new service.
-   **Actions:**
    -   Add `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` to the `Settings` class in `config.py`.
    -   Create a `PineconeService` class to encapsulate all interactions with Pinecone:
        -   Initialization and connection management.
        -   A method to create a new index if it doesn't exist, configured for the embedding dimension (e.g., 1536).
        -   A `upsert` method to add chunk vectors, text content, and metadata to the index in batches.
        -   A `query` method to perform vector similarity searches.
    -   Replace the file-based caching logic in `embedding_service.py` with calls to the `PineconeService`.

### 1.4. Implement Hybrid Search & Re-ranking
-   **File:** `src/services/rag_workflow.py`
-   **Actions:**
    -   **BM25 (Sparse) Search:**
        -   I will use a library like `rank_bm25` to create an in-memory BM25 index from the text chunks. This index will be built alongside the Pinecone index.
        -   The `retrieve_relevant_chunks_optimized` method will be updated to query this BM25 index with the user's query to get keyword-based results.
    -   **Hybrid Fusion:**
        -   The results from Pinecone (dense) and BM25 (sparse) will be combined and scored using a fusion technique like Reciprocal Rank Fusion (RRF) to produce a single candidate list.
    -   **Cross-Encoder Re-ranking:**
        -   I will integrate a lightweight Cross-Encoder model (e.g., from the `sentence-transformers` library).
        -   A new `re_rank` method will take the fused candidate list and the original query, compute a precise relevance score for each candidate, and return the top N most relevant documents. This final, high-quality list will be used as context for the LLM.

---

## Phase 2: Advanced Optimization & Fine-Tuning

**Objective:** To leverage the new vector database capabilities and explore model specialization for further accuracy gains.

### 2.1. Metadata Enrichment & Pre-filtering
-   **Files:** `src/services/text_extraction_service.py`, `src/services/pinecone_service.py`, `src/services/rag_workflow.py`
-   **Actions:**
    -   During chunking, extract and attach metadata to each chunk (e.g., `document_url`, `page_number`, `section_title` from the hierarchical service).
    -   The `PineconeService`'s `upsert` method will store this metadata alongside the vectors.
    -   The `query` method will be updated to accept filter dictionaries, allowing it to perform metadata pre-filtering before the vector search.
    -   The API request schema (`schemas/analysis.py`) could be extended to optionally accept filter criteria.

### 2.2. Fine-tuning Exploration (Experimental)
-   **Objective:** To establish a workflow for potentially fine-tuning an embedding model.
-   **Actions:**
    -   Create scripts to prepare a fine-tuning dataset based on queries and relevant/irrelevant document chunks.
    -   Document the process for fine-tuning a model (e.g., using Sentence-Transformers or OpenAI's fine-tuning API).
    -   This is an experimental step focused on enabling future work rather than immediate implementation in the main pipeline.

---

## Phase 3: Evaluation and Continuous Improvement

**Objective:** To establish a robust framework for measuring improvements and ensuring the system remains state-of-the-art.

### 3.1. Build a RAG Evaluation Framework
-   **File:** Create `tests/evaluate_rag.py`.
-   **Actions:**
    -   Integrate a RAG evaluation library like **RAGAs**.
    -   Create a small, high-quality "golden dataset" of question-context-answer triplets in a separate file (e.g., `tests/evaluation_dataset.json`).
    -   The evaluation script will run the RAG pipeline on the test questions and use RAGAs to compute key metrics:
        -   **Faithfulness:** How factual is the answer based on the context?
        -   **Answer Relevancy:** How relevant is the answer to the question?
        -   **Context Precision & Recall:** Is the retrieved context relevant and sufficient?
    -   This script will serve as a benchmark to quantitatively measure the impact of any changes to the pipeline.

### 3.2. Iterative Prompt & Pipeline Optimization
-   **Objective:** To use the evaluation framework to systematically improve the system.
-   **Actions:**
    -   Create a prompt registry or template system to easily manage and version different system prompts.
    -   Run the evaluation suite against different prompt variations to find the most effective one.
    -   Develop strategies for handling cases where the re-ranked context is still of low quality or contains contradictions, and instruct the LLM on how to respond (e.g., "The provided documents do not contain a clear answer...").
