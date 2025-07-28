# Refactoring Plan: Unified Workflow & Multi-LLM Strategy

This plan outlines the steps to refactor the application into a single, streamlined workflow. It integrates a multi-LLM strategy using Google Gemini and Groq (for Kimi), removes OpenAI, and cleans up the project structure.

---

### Phase 1: Code & Structure Cleanup

**Objective:** Simplify the project by removing obsolete components and preparing for the new unified workflow.

1.  **Remove Unnecessary Files:**
    *   Delete `src/api/v1/endpoints/documents.py`. Its functionality will be merged into the `analysis` endpoint.
    *   Delete `src/test_swagger_auth.py` as it's a standalone test script.
    *   The `tests` directory will be ignored for now, as per the instructions.

2.  **Update API Router:**
    *   Modify `src/api/v1/router.py` to remove the import and inclusion of the `documents_router`.

---

### Phase 2: Configuration & Dependency Updates

**Objective:** Reconfigure the application to support Google Gemini and Groq, and remove OpenAI.

1.  **Update Environment Variables (`.env.example`):**
    *   **Remove:** `OPENAI_API_KEY`.
    *   **Add:** `GOOGLE_API_KEY` and `GROQ_API_KEY`.

2.  **Update Settings Model (`src/core/config.py`):**
    *   Modify the `Settings` class in `src/core/config.py` to reflect the changes in the `.env.example` file. Remove `OPENAI_API_KEY` and add `GOOGLE_API_KEY` and `GROQ_API_KEY`.

3.  **Update Dependencies (`requirements.txt`):**
    *   **Remove:** `openai`.
    *   **Add:** `google-generativeai` and `groq`.

---

### Phase 3: Service Layer Refactoring (Multi-LLM Integration)

**Objective:** Abstract the LLM clients and refactor the services to use the new models for their specific tasks.

1.  **Create Centralized LLM Clients (`src/services/llm_clients.py`):**
    *   Create a new file to initialize and configure the clients for Google Gemini and Groq.
    *   This will provide a single, clean interface for the rest of the application to interact with the LLMs.

2.  **Refactor Ingestion Service (`src/services/ingestion_service.py`):**
    *   Modify the `__init__` method to remove the `AsyncOpenAI` client.
    *   Update the `generate_summary` function to use the **Kimi (`moonshot-k3-128k`)** model via the new Groq client for document summarization.

3.  **Create RAG Workflow Service (`src/services/rag_workflow.py`):**
    *   Create a new service that will contain the core logic for the entire RAG pipeline.
    *   **Step 1: Query Clarification:** Implement a function `clarify_query` that uses **Google Gemini 1.5 Flash** to process user questions, especially vague ones, and turn them into structured, actionable queries.
    *   **Step 2: Final Answer Synthesis:** Implement a function `generate_final_answer` that uses **Kimi via Groq** to synthesize the final, detailed answer from the retrieved context.

---

### Phase 4: Workflow Unification & API Endpoint Refactoring

**Objective:** Implement the new single-request workflow in the main analysis endpoint.

1.  **Refactor Analysis Endpoint (`src/api/v1/endpoints/analysis.py`):**
    *   The `run_analysis` function will become the main orchestrator.
    *   It will now perform the following steps sequentially:
        1.  Accept the request containing document URLs and questions.
        2.  **On-the-fly Ingestion:** For each document URL, it will call the `ingestion_service.process_document` method directly. This will download, parse (with AWS Textract fallback verification), summarize, and embed the document in real-time.
        3.  **RAG Execution:** Once all documents are processed and their data is stored, it will call the new `rag_workflow.py` service.
        4.  The RAG service will execute its pipeline (clarify query, search, synthesize answer).
        5.  The final, structured answer from the RAG service will be returned to the user.

2.  **Verify AWS Textract Integration:**
    *   During the refactoring of the ingestion flow, double-check that the `parse_pdf` function in `src/utils/document_parsers.py` correctly accesses the AWS credentials from the centralized `settings` object, ensuring the OCR fallback works seamlessly.

This plan ensures a logical and safe transition to the new, more powerful, and efficient architecture you have requested.
