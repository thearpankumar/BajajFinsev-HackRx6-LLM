# Document Analysis API with Gemini 2.5 Pro

This project provides a high-performance, API-driven application that performs on-the-fly analysis of insurance documents. It uses a sophisticated, type-based workflow to leverage the full power of **Google's Gemini 2.5 Pro** model for answering user questions with high accuracy.

The application is architected for simplicity, speed, and robustness, using a minimal set of dependencies and a fully non-blocking request/response cycle.

## Key Features

*   **Direct-to-Model AI:** Uses Google Gemini 2.5 Pro as the core engine, eliminating the need for complex, intermediate RAG pipelines (Vector DBs, embeddings, etc.).
*   **Intelligent Document Handling:**
    *   **PDFs:** Uploaded directly to Gemini, leveraging its powerful native PDF parsing capabilities to understand layout, tables, and text.
    *   **DOCX & EML:** Text content is extracted and uploaded as a clean `.txt` file, ensuring maximum reliability for text-based documents.
*   **Optimized for Speed:** The entire request lifecycle is asynchronous. A "fire-and-forget" background process handles resource cleanup, ensuring the user gets a response immediately after the answer is generated, without waiting for cleanup tasks.
*   **Parallel Processing:** Submits all user questions to the Gemini API concurrently, significantly reducing the total time required to get a full set of answers.
*   **Simplified & Robust Tech Stack:** The architecture has been streamlined to be more maintainable and efficient.

## Tech Stack

*   **Backend:** FastAPI
*   **LLM:** Google Gemini 2.5 Pro
*   **Deployment:** Docker, Nginx
*   **Core Libraries:** `uvicorn`, `pydantic`, `google-generativeai`, `python-docx`

---

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

Create a `.env` file by copying the example file.

```bash
cp .env.example .env
```

Now, open the `.env` file and populate it with your secret keys:
*   `API_KEY` (a secret key of your choice for securing your API endpoint)
*   `GOOGLE_API_KEY` (your API key for the Google Gemini service)

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

---

## Running the Application Locally

### 1. Start the FastAPI Application

Start the development server. The `--reload` flag will automatically restart the server when you make code changes.

```bash
uvicorn src.main:app --reload
```

Your API is now running at `http://127.0.0.1:8000`.

---

## Testing the API

Use the following `curl` command to send a request using the `payload.json` file. Make sure you are in the same directory as the file.

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/hackrx/run" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 12345678901" \
-d @payload.json
```
*(Note: This uses the default API key. If you changed it in your `.env` file, update the `Bearer` token here.)*

You will see the processing logs in your Uvicorn terminal, and the final JSON response with the answers will be printed by the `curl` command almost immediately after the answers are generated.

```