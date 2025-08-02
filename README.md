# Advanced Document Analysis API for Enterprise

This project provides a high-performance, enterprise-grade API for deep analysis of business documents, specializing in **Insurance, Legal, HR, and Compliance** domains. It leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline, combining multiple AI models and advanced search techniques to deliver highly accurate and context-aware answers to user questions.

The architecture is designed for performance, scalability, and deep domain understanding, featuring a robust, asynchronous workflow from document ingestion to answer generation.

## Key Features

*   **Advanced RAG Pipeline:** Goes beyond simple vector search by implementing a multi-stage retrieval and ranking process:
    *   **Hybrid Search:** Combines keyword-based sparse search (BM25) with semantic dense search (OpenAI Embeddings + LanceDB) using Reciprocal Rank Fusion (RRF) to ensure both keyword precision and contextual relevance.
    *   **Re-ranking:** Employs a Cross-Encoder model to re-rank the fused search results for maximum relevance to the user's query.
*   **Hierarchical Processing for Large Documents:** For large files, the system first splits the document into logical sections, uses an LLM to summarize them, and then processes only the sections relevant to the user's query, drastically reducing processing time and cost.
*   **Rich Metadata Extraction:** Automatically extracts key metadata from document chunks, including **entities, concepts, categories, and keywords**. This metadata is used to enrich the context provided to the LLM, leading to more precise and informed answers.
*   **Multi-LLM Strategy:**
    *   **Answer Generation:** Uses **OpenAI's `gpt-4o-mini`** for its strong reasoning and generation capabilities.
    *   **Query Clarification:** Leverages **Google's `gemini-2.5-flash-lite`** to refine and expand user queries for better search results.
*   **Domain-Specific Intelligence:** Utilizes a prompt registry with detailed, domain-specific instructions and few-shot examples for Insurance, Legal, HR, and Compliance, ensuring the model's responses are tailored to the specific context.
*   **Performance & Monitoring:**
    *   **Asynchronous by Design:** Fully non-blocking architecture using FastAPI.
    *   **Streaming Responses:** An endpoint (`/hackrx/stream`) provides quick initial answers while detailed analysis continues in the background.
    *   **Performance Dashboard:** A dedicated endpoint (`/hackrx/performance`) provides detailed metrics on processing time, memory usage, cache hit rates, and more.

## Tech Stack

*   **Backend:** FastAPI
*   **Vector Database:** LanceDB (local, high-performance)
*   **LLMs:** OpenAI GPT-4o-mini, Google Gemini Flash
*   **Embedding Model:** OpenAI `text-embedding-3-small`
*   **Core Libraries:** `uvicorn`, `pydantic`, `openai`, `google-generativeai`, `lancedb`, `rank_bm25`, `sentence-transformers`, `fitz` (PyMuPDF), `python-docx`, `nltk`, `spacy`.
*   **Deployment:** Docker, Nginx

## Local Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites

*   Python 3.11+
*   `curl` for testing the API

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd BajajFinsev
```

### 2. Configure Environment Variables

Create a `.env` file. You can copy the structure from the settings in `src/core/config.py`.

```bash
# Create an empty .env file
touch .env
```

Now, open the `.env` file and add the following required variables:
```env
# A secret key of your choice for securing your API endpoint
API_KEY="12345678901"

# Your API keys for the AI services
GOOGLE_API_KEY="your_google_api_key"
OPENAI_API_KEY="your_openai_api_key"
```

### 3. Set Up the Python Virtual Environment

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate
```

### 4. Install Dependencies

Install all the required dependencies from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not up to date, you can generate it from `requirements.in`:
```bash
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
```

## Running the Application Locally

Start the development server using Uvicorn. The `--reload` flag will automatically restart the server when you make code changes.

```bash
uvicorn src.main:app --reload
```

Your API is now running at `http://1227.0.0.1:8000`.

## API Endpoints

The API provides several endpoints for analysis and monitoring:

*   `POST /api/v1/hackrx/run`: The primary endpoint for document analysis. It takes a document URL and a list of questions and returns the answers.
*   `POST /api/v1/hackrx/stream`: Streams analysis results in phases, providing faster initial feedback.
*   `GET /api/v1/hackrx/health`: A health check endpoint that provides the status of the service and its components.
*   `GET /api/v1/hackrx/performance`: Returns a JSON object with detailed performance metrics.
*   `POST /api/v1/hackrx/performance/reset`: Resets the performance counters.

## Testing the API

Use the following `curl` command to send a request to the main analysis endpoint. Make sure you have a payload file (e.g., `payloads/payload1.json`) available.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/hackrx/run" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 12345678901" \
-d @payloads/payload1.json
```
*(Note: This uses the default API key. If you changed it in your `.env` file, update the `Bearer` token here.)*

You will see processing logs in your Uvicorn terminal, and the final JSON response with the answers will be printed by the `curl` command.

```
