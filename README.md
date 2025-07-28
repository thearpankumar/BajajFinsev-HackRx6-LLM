# Document Analysis and Q&A with RAG

This project is a sophisticated, API-driven application that performs on-the-fly analysis of insurance documents. It uses a Retrieval-Augmented Generation (RAG) workflow with a multi-LLM strategy to answer user questions based on the content of the provided documents.

## Key Features

*   **Unified Workflow:** A single API endpoint handles document ingestion, processing, and question-answering in one call.
*   **Multi-LLM Strategy:**
    *   **Google Gemini 1.5 Flash:** Used for fast and intelligent user query clarification.
    *   **Groq Llama 4:** Used for high-quality document summarization and final answer synthesis.
*   **On-the-Fly Processing:** Documents are downloaded, parsed, chunked, and embedded in real-time.
*   **Advanced Document Parsing:** Supports PDF, DOCX, and EML files, with an automatic fallback to **AWS Textract** for OCR on scanned documents.
*   **Vector Search:** Uses Pinecone for high-speed semantic search of document clauses.
*   **Persistent Storage:** Leverages a PostgreSQL database to store document metadata and summaries.
*   **Containerized Database:** The PostgreSQL database is managed with Docker for a clean, isolated environment.
*   **Robust Dependency Management:** Uses a virtual environment and `pip-tools` to ensure reproducible and conflict-free dependency installation.

## Tech Stack

*   **Backend:** FastAPI
*   **Database:** PostgreSQL (managed with SQLAlchemy and Alembic)
*   **Vector Store:** Pinecone
*   **LLMs:** Google Gemini, Groq Llama 4
*   **Deployment:** Docker, Nginx
*   **Core Libraries:** `uvicorn`, `pydantic`, `sentence-transformers`, `spacy`, `PyMuPDF`

---

## Local Setup and Installation

Follow these steps to get the application running on your local machine for development and testing.

### Prerequisites

*   Python 3.11+
*   Docker and Docker Compose
*   `curl` for testing the API

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd BajajFinsev
```

### 2. Configure Environment Variables

Create a `.env` file by copying the example file.

```bash
cp .env.example .env
```

Now, open the `.env` file and populate it with your secret keys and API credentials. You will need to fill in:
*   `API_KEY` (a secret key of your choice)
*   `GOOGLE_API_KEY`
*   `GROQ_API_KEY`
*   `PINECONE_API_KEY`
*   `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (if you want to use the OCR fallback)

**Important:** For local development, ensure your `DATABASE_URL` is set to connect to `localhost`:
```
DATABASE_URL=postgresql://user:password@localhost:5432/bajaj_finsev_db
```

### 3. Set Up the Python Virtual Environment

This project uses a virtual environment to isolate its dependencies and prevent conflicts.

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the environment (you must do this every time you work on the project)
source .venv/bin/activate
```
You will see `(.venv)` in your terminal prompt when it's active.

### 4. Install Dependencies

First, compile the pinned requirements from the `requirements.in` file.

```bash
# This step might take a few minutes the first time
pip-compile requirements.in --output-file=requirements.txt
```

Now, install all the compiled dependencies into your virtual environment.

```bash
pip install -r requirements.txt
```

### 5. Download the spaCy Model

The application uses a spaCy model for text segmentation. Download it by running:

```bash
python -m spacy download en_core_web_sm
```

---

## Running the Application Locally

This setup runs the database in a Docker container and the FastAPI application directly on your local machine.

### 1. Start the Database

Make sure Docker is running, then start the PostgreSQL container in the background:

```bash
docker compose up -d postgres
```
Wait about 10-15 seconds for the database to initialize the first time.

### 2. Run Database Migrations

Apply the database schema to create the `documents` and `clauses` tables.

```bash
alembic upgrade head
```

### 3. Start the FastAPI Application

Finally, start the development server. The `--reload` flag will automatically restart the server when you make code changes.

```bash
uvicorn src.main:app --reload
```

Your API is now running at `http://127.0.0.1:8000`.

---

## Testing the API

### 1. Create the Payload File

Create a file named `payload.json` and add the following content. This makes sending the complex URL easy and reliable.

```json
{
    "documents": [
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ],
    "questions": [
        "What is the effective date of this policy?",
        "Summarize the key obligations of the insured party."
    ]
}
```

### 2. Send the Request

Run the following `curl` command from your terminal (make sure you are in the same directory as `payload.json`).

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/hackrx/run" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 12345678901" \
-d @payload.json
```
*(Note: This uses the default API key. If you changed it in your `.env` file, update the `Bearer` token here.)*

You will see the processing logs in your Uvicorn terminal, and the final JSON response with the answers will be printed by the `curl` command.