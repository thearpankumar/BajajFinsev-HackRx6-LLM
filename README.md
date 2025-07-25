# Project Sentinel: The Self-Critiquing Agentic Framework

## 1. Project Objective

To build a state-of-the-art system that uses a coalition of AI agents to process natural language queries, retrieve hyper-relevant information from large, unstructured documents (policies, contracts), and deliver accurate, verifiable, and fully justified decisions.

This project goes beyond simple Retrieval-Augmented Generation (RAG). It implements a **Self-Critiquing Agentic Framework** to ensure resilience, efficiency, and unparalleled accuracy by mimicking the reasoning and self-correction process of a human expert.

---

---
## step-by-step breakdown of how the entire "Self-Critiquing Agentic Framework" works, from the initial setup to a live user interaction.

### **Phase 1: The One-Time Setup (Preparing the Knowledge)**

This happens once, before any user asks a question.

1.  **Document Ingestion:** The system first consumes all the unstructured documents you provide—PDFs of insurance policies, Word documents with rules, emails, etc.

2.  **Intelligent Structuring:** For each document, it performs two critical tasks:
    *   **It creates summaries:** It reads large sections and writes short summaries, like chapter titles, to quickly understand what a document is about.
    *   **It creates detailed chunks:** It breaks down the entire text into small, numbered paragraphs or "chunks." For each chunk, it also extracts key entities like `{'procedure': 'knee surgery', 'waiting_period_months': 12}` and tags the chunk with this structured data.

3.  **Indexing for Search:** Every single chunk—both summaries and detailed paragraphs—is converted into a numerical "meaning signature" (a vector embedding). These signatures, along with the original text and metadata (like `Policy_A.pdf, Page 12`), are stored in a special, high-speed vector database. This creates a searchable library of all the knowledge.

---

### **Phase 2: A Live User Session (Answering a Query)**

This is what happens in real-time every time a user interacts with the system.

**Step 1: The User Asks a Question**
The user types their query into the chat interface. For example: *"46M, knee surgery, Pune, 3-month policy"*

**Step 2: The Triage Agent Intercepts and Clarifies**
The first AI agent, the "Triage Agent," examines the query. Its job is to prepare the question for research.
*   It looks at the chat history. Since this is the first question, the history is empty.
*   It standardizes the query into a clearer question: *"What is the coverage for a 46-year-old male undergoing knee surgery in Pune with a 3-month-old insurance policy?"*
*   It then passes this clean, standalone question to the next agent.

**Step 3: The Research Agent Hunts for Evidence**
The "Research Agent" now hunts for relevant clauses in the indexed library.
*   **First Pass (Broad Search):** It searches the *summaries* to instantly find the 1-2 most relevant documents. For instance, it identifies "General Health Policy 2024.pdf" and "Surgical Addendum.pdf".
*   **Second Pass (Focused Search):** It now searches the *detailed chunks*, but *only* within the two documents it just found. This is fast and highly accurate. It retrieves the top 5 most relevant clauses about surgery, waiting periods, and age limits.

**Step 4: The Validation Agent Performs a "Gut Check"**
This is the system's "self-critique" step. The "Validation Agent" looks at the 5 clauses found by the Research Agent.
*   It asks itself: "Based *only* on these 5 clauses, can I confidently answer the user's specific question?"
*   **Scenario A (High Confidence):** The clauses are clear and directly address the query. The agent approves the evidence and passes it to the final agent.
*   **Scenario B (Low Confidence):** The clauses are ambiguous. The agent rejects the evidence and triggers a **correction loop**. It tells the Research Agent: "That's not good enough. Search again, but specifically look for 'waiting periods for new policies'." The Research Agent tries again. This loop ensures only high-quality evidence is used.

**Step 5: The Decision Agent Synthesizes the Final Answer**
The "Decision Agent" receives the high-confidence, validated clauses.
*   It bundles the evidence with the user's query and gives a final instruction to the most powerful LLM (Gemini Pro).
*   The instruction is: "You are an insurance claims expert. Based *only* on the provided clauses, determine the decision, amount, and justification for this user. Respond in a structured JSON format."
*   The LLM provides the structured answer.

**Step 6: The Results are Presented to the User**
The user sees the final, clean answer in the chat window.
*   **Decision:** Approved
*   **Amount:** $4,500
*   **Justification:** "The policy covers knee surgery after a 90-day waiting period. [**View Source: Policy_A.pdf, Page 14**]"

**Step 7: The User Interacts and Probes Deeper**
*   The user clicks the **"View Source"** button. A window pops up showing Page 14 of the original PDF, with the exact clause highlighted. This builds immense trust.
*   The user then asks a follow-up question: *"what about post-op PT?"*
*   The **Triage Agent** sees this new query and the chat history. It creates a new standalone question: *"Does the policy cover post-operative physical therapy for knee surgery?"*
*   The entire process from Step 3 onwards repeats, providing a seamless conversational experience.

---


## 2. Core Features & Innovations

*   **Agentic Workflow:** The system uses a team of specialized agents (Triage, Research, Validation, Decision) that collaborate to handle complex queries, decomposing them and planning the best path to an answer.
*   **Self-Critique & Correction Loop:** The system's **Validation Agent** assesses the quality of retrieved information. If it's insufficient, it triggers a correction loop, forcing the system to re-search and find better evidence before answering. This overcomes the primary failure mode of basic RAG.
*   **Dynamic Knowledge Graph-Lite:** During ingestion, the system extracts structured entities (e.g., waiting periods, covered procedures) from text, enabling a powerful combination of fast, filtered queries and deep semantic search.
*   **Adaptive Hierarchical Retrieval:** To handle large documents, the system first searches document summaries to find the right document, then searches within that document for specific clauses, saving time and improving relevance.
*   **Full Conversational Memory:** The system maintains a complete chat history, allowing for natural follow-up questions ("What about in Mumbai?") and a seamless, context-aware user experience.
*   **Full Traceability & Interactive Source Verification:** Users can not only see the final answer but can also view the exact source page from the original PDF with the relevant clause highlighted, and can access a "debug" view to see the agent's entire reasoning process.
*   **User-Centric Design:** Features like a one-click user feedback loop and proactive query suggestions make the tool interactive, intelligent, and ready for continuous improvement.

---

## 3. System Architecture: The Self-Critiquing Agentic Framework

The system operates in two phases: an offline **Ingestion Phase** and a real-time **Query Phase**.

### 3.1. Phase 1: Offline - Intelligent Ingestion & Indexing

1.  **Load Documents:** Use the `unstructured` library to load text and metadata from various file formats (PDF, DOCX, etc.).
2.  **Hierarchical Chunking:**
    *   **Level 1 (Detailed Chunks):** Split documents into small, 1000-character chunks with overlap using `RecursiveCharacterTextSplitter`.
    *   **Level 2 (Summary Chunks):** Use a fast, local LLM (via **Ollama**) to generate a summary for each large document or section.
3.  **Knowledge Graph-Lite Extraction:**
    *   For each detailed chunk, use an LLM to extract key entities and rules as a JSON object (e.g., `{'procedure': 'knee surgery', 'waiting_period': '12 months'}`).
4.  **Embed & Index:**
    *   Use a `Sentence-Transformer` model (`all-MiniLM-L6-v2`) to create vector embeddings for all chunks (both detailed and summary).
    *   Store everything in a **ChromaDB** vector store, carefully storing the text, the source metadata (filename, page number), and the extracted JSON entities with each vector.

### 3.2. Phase 2: Real-time - The Agentic Query Workflow

1.  **User Query** is received by the **Triage Agent**.
2.  **Triage Agent:**
    *   Analyzes the query **and the conversation history**.
    *   **Action:** Decomposes complex questions, rephrases follow-ups into standalone questions, or asks the user for clarification. Passes the clear question(s) to the Research Agent.
3.  **Research Agent:**
    *   Receives a clear question.
    *   **Action:** Executes the **Adaptive Retrieval** strategy:
        1.  Performs semantic search on document *summaries* to find the most relevant documents.
        2.  Performs a second semantic search on the *detailed chunks* within only those relevant documents.
        3.  Uses a **Cross-Encoder** to re-rank the retrieved chunks for maximum relevance.
    *   Passes the top-k chunks of evidence to the Validation Agent.
4.  **Validation Agent (Self-Critique Loop):**
    *   Receives the evidence from the Research Agent.
    *   **Action:** Assesses the evidence against the original query.
        *   **If Confident:** Passes the evidence to the final Decision Agent.
        *   **If Not Confident:** Instructs the Research Agent to try again with a refined query (e.g., "search again, focusing on exclusions"). This loop can run 1-2 times.
5.  **Decision Agent:**
    *   Receives the high-confidence, validated evidence.
    *   **Action:** Constructs a detailed prompt containing the evidence and the user's structured query.
    *   Sends this to a powerful LLM (**Google Gemini Pro**) to synthesize the final, structured JSON answer (`{decision, amount, justification}`).
6.  The final response is returned to the user, including proactive suggestions for follow-up questions.

---

## 4. Technology Stack

*   **Backend Framework:** Python
*   **Orchestration & AI Logic:** LangChain
*   **Web API:** FastAPI
*   **Frontend UI:** Streamlit
*   **Document Loading:** `unstructured`
*   **Vector Database:** ChromaDB
*   **Powerful LLM (Reasoning):** Google Gemini Pro API
*   **Local LLM (Utility Tasks):** Ollama (with Llama 3 or Mistral)
*   **Embedding Models:** `sentence-transformers` library (for both embeddings and cross-encoders)
*   **Traceability/Debugging:** LangSmith
*   **PDF Handling (Frontend):** `PyMuPDF` (or `pypdf`) and `Pillow`

---

## 5. Step-by-Step Implementation Plan

### Step 0: Environment Setup

1.  Install Python 3.10+.
2.  Run `pip install langchain fastapi "uvicorn[standard]" streamlit chromadb unstructured[pdf,docx] sentence-transformers pypdf pymupdf Pillow google-generativeai python-dotenv`.
3.  Install Ollama on your local machine and pull a model: `ollama pull llama3`.
4.  Create a `.env` file and add your `GOOGLE_API_KEY`.
5.  **Crucially for debugging:** Sign up for a free **LangSmith** account and add the required `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, and `LANGCHAIN_PROJECT` environment variables to your `.env` file. This will give you the "Expert Traceability View" almost for free.

### Step 1: Build the Data Ingestion Script (`ingest.py`)

*(This section's implementation remains the same: load, chunk, summarize, extract KG-Lite metadata, embed, and store in ChromaDB.)*

### Step 2: Develop the Backend API (`main.py` with FastAPI)

This is the core engine where you'll implement the agent logic and QoL features.

1.  **Initialize FastAPI App and Models:** Set up FastAPI, load all LLM clients, embedding models, and the ChromaDB client on startup.
2.  **Implement Conversational Memory Logic:**
    *   **Data Structure:** At the top level of your `main.py`, create a simple in-memory dictionary to hold all ongoing conversations: `chat_histories = {}`.
    *   **Triage Agent Enhancement:** The `run_triage_agent` function will now be the heart of your chat logic.
        *   **Signature:** `def run_triage_agent(query: str, history: list[dict]) -> str:`
        *   **Prompting:** Inside, you'll make a call to your local Ollama model. The prompt is key:
            ```
            You are a helpful assistant. Given the following chat history and a new user question, rephrase the user question to be a standalone question that can be understood without the chat history. If the new question is already standalone, return it as is.

            Chat History:
            {history}

            New User Question: {query}

            Standalone Question:
            ```
        *   **Example:** If history is `[{'role': 'user', 'content': 'Is knee surgery covered?'}, {'role': 'ai', 'content': 'Yes, for in-network providers.'}]` and the new query is `"What about in Mumbai?"`, this agent should return `"Is knee surgery covered for in-network providers in Mumbai?"`. This becomes the input for the Research Agent.

3.  **Build the Main Query Endpoint: `POST /query`**
    *   **Request/Response Models:** Define Pydantic models for your API.
        ```python
        from pydantic import BaseModel
        from typing import Optional, List

        class QueryRequest(BaseModel):
            query: str
            conversation_id: Optional[str] = None

        # ... other models for the response
        ```
    *   **Endpoint Logic:**
        1.  Get or create a `conversation_id`. Use `uuid.uuid4()` if none is provided.
        2.  Fetch the `history` for this ID from `chat_histories.get(conversation_id, [])`.
        3.  Call `run_triage_agent(query, history)` to get the clean, standalone question.
        4.  Execute the Research -> Validation -> Decision agent workflow as previously described.
        5.  **Structure the Response:** Package everything into a single JSON response: the answer, source citations, debug trace, suggested queries, and the `conversation_id`.
        6.  **Update History:** Before returning, append the original user query and the final AI answer to the history list: `chat_histories[conversation_id].extend([{'role': 'user', 'content': query}, {'role': 'ai', 'content': ai_answer_text}])`.

4.  **Add QoL Endpoints:**
    *   **`GET /get_page_image/{doc_name}/{page_num}`:**
        *   This endpoint uses the `PyMuPDF` library (`fitz`).
        *   It opens the PDF, loads the specified page (`doc.load_page(page_num - 1)`), and renders it as a high-quality pixmap (`page.get_pixmap()`).
        *   This pixmap is then saved to a `BytesIO` buffer using `Pillow` and returned as a `StreamingResponse` with a `media_type="image/png"`.
    *   **`POST /feedback`:**
        *   Accepts a simple JSON body with `conversation_id` and `rating`.
        *   It opens a file named `feedback.log` in append mode (`'a'`) and writes a new line: `f"{datetime.now()},{conversation_id},{rating}\n"`.

### Step 3: Create the User Interface (`ui.py` with Streamlit)

1.  **Initialize Session State:** At the top of your script, initialize variables to persist across re-runs. This is the key to a stateful chat app.
    ```python
    import streamlit as st
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    ```
2.  **Render Chat History:** Loop through `st.session_state.messages` and display them using `st.chat_message`.
3.  **Handle User Input:**
    *   Use `if prompt := st.chat_input("Ask about your policy..."):`.
    *   Append the user's prompt to `st.session_state.messages`.
    *   Display the user's message immediately in the chat.
    *   With a spinner (`with st.spinner("Thinking..."):`), call your FastAPI `/query` endpoint. **Crucially, pass the `st.session_state.conversation_id` in the request.**
    *   Store the `conversation_id` returned from the API back into `st.session_state` so it can be used for the next turn.
4.  **Display the Rich Response:**
    *   When the API responds, append the AI's full message object to `st.session_state.messages`.
    *   In the main chat display loop, when rendering an AI message:
        *   Display the main text answer.
        *   Render the `justification` with a `st.button` for each source.
        *   If the source button is clicked, call the `/get_page_image` endpoint and display the returned image in an `st.expander`.
        *   Render the `suggested_queries` as clickable buttons that can trigger a new query.
    *   Add the feedback buttons and the "Agent Trace" expander below the main AI response.

### Step 4: Final Polish and Pitch Practice

1.  **Test End-to-End:** Run the ingestion script, then start the FastAPI server and the Streamlit app.
2.  **Practice the Demo Flow:**
    *   Start with a simple query.
    *   Ask a natural follow-up question to demo **conversational memory**.
    *   Click on a source link to demo the **interactive PDF viewer**.
    *   Use the "Thumbs Up" to demo the **feedback loop** (and show the `feedback.log` file).
    *   Open the "Agent Trace" expander and walk through the steps, then switch to the **LangSmith dashboard** to show the professional, detailed trace of the self-correction loop in action.
3.  **Refine Your Narrative:** Emphasize how these integrated features transform the tool from a simple demo into a trustworthy, user-friendly, and robust prototype ready for real-world application.
