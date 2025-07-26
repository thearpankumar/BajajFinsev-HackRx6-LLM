### Task 2: Document Ingestion Pipeline

**Objective:** Implement the asynchronous pipeline for document upload, parsing, summarization (PostgreSQL), and embedding/vector storage (Pinecone).

**Development Status:** Requires database and Pinecone setup. This task builds upon Task 1.

---

#### Task 2.1: Database Setup (PostgreSQL)

*   **Objective:** Configure SQLAlchemy for PostgreSQL and define ORM models.
*   **Files:** `src/db/base.py`, `src/db/models.py`
*   **Details:**
    *   In `src/db/base.py`, set up the SQLAlchemy engine, session local, and `Base` class.
    *   In `src/db/models.py`, define ORM models for:
        *   `Document`: `id`, `url`, `summary`, `status`, `created_at`.
        *   `Clause`: `id`, `document_id` (ForeignKey), `text`, `embedding_id`, `metadata` (JSONB for page_number, clause_number, etc.).
    *   Use environment variables for database connection details.
*   **Tech Stack:** PostgreSQL, SQLAlchemy.

#### Task 2.2: Document Parsers

* **Objective:** Implement functions to extract text from various document types, with a fallback to OCR for scanned or image-based files.
* **File:** `src/utils/document_parsers.py`
* **Details:**
  * PDF Parsing: Implement `parse_pdf(file_content: bytes) -> str`.
    * The function will first attempt direct text extraction using PyMuPDF.
    * If it detects a scanned page (i.e., minimal text is extracted), it will automatically fall back to using a cloud OCR service like AWS Textract to ensure text is captured.
  * Standard Parsers:
    * Implement `parse_docx(file_content: bytes) -> str` using python-docx.
    * Implement `parse_email(file_content: bytes) -> str` using Python's email standard library.
  * Factory Function: Create a `get_parser(filename: str)` function to return the correct parser based on the file extension.
* **Tech Stack:** PyMuPDF, python-docx, Python email, AWS Textract

#### Task 2.3: Document Upload Endpoint

*   **Objective:** Create an API endpoint for users to upload documents.
*   **File:** `src/api/v1/endpoints/documents.py` (new file)
*   **Details:**
    *   Define a `POST /documents/upload` endpoint.
    *   Accept document URLs or file uploads.
    *   This endpoint should trigger the asynchronous ingestion process (e.g., by adding a task to a background queue or directly calling the ingestion service in a background task).
    *   Integrate authentication from `src/core/security.py`.
*   **Tech Stack:** FastAPI.

#### Task 2.4: Ingestion Service Core Logic

*   **Objective:** Orchestrate the document processing, summarization, and embedding.
*   **File:** `src/services/ingestion_service.py`
*   **Details:**
    *   **Asynchronous Document Download:** Implement logic to download document content if a URL is provided.
    *   **Parsing:** Call the appropriate parser from `src/utils/document_parsers.py`.
    *   **Semantic Segmentation:** Implement clause extraction using `spaCy` (as described in `Process Report.pdf`).
    *   **Summarization:**
        *   Use an LLM (e.g., GPT-4o) to generate a summary of the entire document.
        *   Save this summary to the `Document` table in PostgreSQL.
    *   **Embedding Generation:**
        *   Use `sentence-transformers` (e.g., `nlpaueb/legal-bert-base-uncased`) to generate embeddings for each extracted clause.
    *   **Pinecone Upsert:**
        *   Initialize Pinecone client using environment variables for API key and environment.
        *   Upsert the generated embeddings along with relevant metadata (e.g., `document_id`, `clause_text`, `source_document_url`, `clause_index`) into your Pinecone index.
*   **Tech Stack:** `aiohttp` (for async download), `spaCy`, `sentence-transformers`, Pinecone, LLMs.

#### Task 2.5: Integrate Ingestion Service with Main App

*   **Objective:** Connect the new document endpoint and ingestion service to the main FastAPI application.
*   **File:** `src/main.py`
*   **Details:**
    *   Include the new `APIRouter` from `src/api/v1/endpoints/documents.py` in `src/main.py`.
*   **Tech Stack:** FastAPI.
