### Task 3: RAG Workflow and Multi-LLM Orchestration

**Objective:** Implement the two-stage LLM and DB searching RAG workflow, ensuring modularity and asynchronous execution.

**Development Status:** Depends on Task 1 and Task 2. This is the core intelligence of the application.

---

#### Task 3.1: Implement RAG Workflow Service

*   **Objective:** Create the central orchestration logic for the RAG process.
*   **File:** `src/services/rag_workflow.py`
*   **Details:** This service will contain the main function that orchestrates all steps of the RAG process, called from the `/hackrx/run` endpoint.
*   **Tech Stack:** Python `asyncio`.

#### Task 3.2: Query Clarification (LLM)

*   **Objective:** Use an LLM to transform vague user queries into structured formats.
*   **File:** `src/services/rag_workflow.py`
*   **Details:**
    *   Implement a function (e.g., `structure_user_query`) that takes a raw user query.
    *   Use GPT-4o with its function calling capabilities to parse the natural language query into a structured Pydantic object (e.g., `StructuredQuery` as per `Process Report.pdf`).
    *   Handle cases where function calling fails or the query is too vague.
*   **Tech Stack:** OpenAI API, Pydantic.

#### Task 3.3: Relational DB Search for Document Summaries

*   **Objective:** Quickly identify relevant documents based on their summaries in PostgreSQL.
*   **File:** `src/services/rag_workflow.py`
*   **Details:**
    *   Implement a function to query the PostgreSQL `Document` table.
    *   Search for documents whose summaries or metadata are relevant to the structured user query.
    *   This step helps to pre-filter the document corpus before the more expensive vector search.
*   **Tech Stack:** PostgreSQL, SQLAlchemy.

#### Task 3.4: High-Precision Vector Retrieval (Pinecone)

*   **Objective:** Retrieve the most relevant clauses from Pinecone based on semantic similarity and metadata filters.
*   **File:** `src/services/rag_workflow.py`
*   **Details:**
    *   Implement a function (e.g., `retrieve_relevant_clauses`) that takes the structured query and potentially document IDs from the relational DB search.
    *   Generate embeddings for the semantic part of the query.
    *   Perform a hybrid search in Pinecone, combining vector similarity with metadata filtering (e.g., `document_id`, `policy_year`, `document_type`).
    *   Return a list of relevant clause metadata, including the full text of the clause.
*   **Tech Stack:** Pinecone, Sentence Transformers.

#### Task 3.5: Clause Interpretation (LLM)

*   **Objective:** Use a cost-effective LLM to interpret retrieved clauses in context.
*   **File:** `src/services/rag_workflow.py`
*   **Details:**
    *   Implement a function (e.g., `interpret_clause`) that takes a user query and a clause text.
    *   Use Mistral Medium (or similar) to analyze the clause and summarize its direct relevance to the query.
    *   Implement `interpret_retrieved_clauses` to run interpretation concurrently for all retrieved clauses.
*   **Tech Stack:** Mistral AI API.

#### Task 3.6: Final Synthesis and Logic Evaluation (LLM)

*   **Objective:** Generate the final answer and decision based on all gathered evidence.
*   **File:** `src/services/rag_workflow.py`
*   **Details:**
    *   Implement `construct_final_prompt` to build a highly structured prompt for GPT-4o, including the original query, structured query, retrieved clauses, and their interpretations.
    *   Implement `generate_final_response` to call GPT-4o in JSON mode.
    *   Ensure the output conforms to the `AnalysisResponse` Pydantic schema.
    *   The final response should be a validated, structured, and explainable answer.
*   **Tech Stack:** OpenAI API, Pydantic.

#### Task 3.7: Integrate RAG Workflow with API Endpoint

*   **Objective:** Connect the RAG workflow service to the `/hackrx/run` API endpoint.
*   **File:** `src/api/v1/endpoints/analysis.py`
*   **Details:**
    *   In the `run_analysis` function, call the main orchestration function from `src/services/rag_workflow.py`.
    *   Pass the `AnalysisRequest` data and receive the `AnalysisResponse`.
*   **Tech Stack:** FastAPI.