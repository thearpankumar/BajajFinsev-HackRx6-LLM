# BajajFinsev Gemini-First Document Analysis API

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/r/arpankumar1119/hackrx-bajaj)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini_2.5_Flash-blue?logo=google)](https://ai.google.dev)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?logo=openai)](https://openai.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red?logo=qdrant)](https://qdrant.tech)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG_Framework-purple)](https://llamaindex.ai)
[![Performance](https://img.shields.io/badge/Performance-Optimized-brightgreen)](https://github.com)

A **high-accuracy, enterprise-grade API** for deep analysis of business documents using a **Gemini-first** approach. The system leverages **Google's Gemini 2.5 Flash** for primary analysis of documents, ensuring the highest possible accuracy. For unsupported formats or in case of failures, it intelligently falls back to a **high-performance RAG pipeline** powered by **Qdrant** and **LlamaIndex**.

## 🚀 Key Features

### 🧠 **Primary: Direct Gemini Analysis**
- **Maximum Accuracy**: Sends documents directly to Google's Gemini 2.5 Flash for state-of-the-art analysis.
- **Multi-Format Support**: Natively handles PDF, DOCX, Excel, images, and more through Gemini's multimodal capabilities.
- **Algorithm Execution**: Can detect and execute step-by-step algorithms described in documents, including making live API calls.
- **No RAG Overhead**: For supported documents, analysis is performed without the need for chunking, embedding, or vector storage.

### ⚡ **Fallback: Advanced RAG Pipeline**
- **High-Performance RAG**: When direct analysis is not possible, falls back to a sophisticated RAG pipeline.
- **Qdrant Vector Database**: Production-ready vector storage for speed and scalability.
- **LlamaIndex Integration**: Advanced document processing and chunking.
- **Hybrid Search**: Combines dense (vector) and sparse (BM25) retrieval for relevance.
- **Persistent Caching**: Reduces processing time for repeated queries.

### 📄 **Multi-Format Document Processing**
- **Broad Format Support**: Handles PDF, DOCX, DOC, XLSX, XLS, CSV, and various image formats (JPG, PNG, etc.).
- **Fast OCR**: Integrated with EasyOCR for high-speed text extraction from images.
- **Intelligent Fallback**: If a document format is unsupported or the file is too large, the system can answer questions using the LLM's general knowledge.

### 🎯 **Complex Question Handling**
- **Multi-Part Question Decomposition**: Breaks down complex questions into smaller, manageable parts.
- **Parallel Processing**: Answers multiple questions or sub-questions simultaneously for significant speed improvements.
- **Smart Answer Combination**: Intelligently combines answers from sub-questions into a single, coherent response.

### 🏭 **Production Ready**
- **Docker Deployment**: Comes with `docker-compose.yml` for easy setup of the API, Qdrant, and Nginx.
- **Health Monitoring**: Includes health check endpoints for all major components.
- **Robust Error Handling**: Gracefully handles failures and falls back to secondary systems.
- **Scalable Architecture**: Designed for horizontal scaling with a load-balanced setup.

## 🏗️ Architecture

```mermaid
graph TD
    A[Document URL] --> B{Processable by Gemini?};
    B -->|Yes| C[Direct Gemini Processor];
    C --> D[Analyze with Gemini 2.5 Flash];
    D --> E[Return High-Accuracy Answer];

    B -->|No| F[Fallback to RAG];
    F --> G[Multi-Format Processor<br/>(PDF, DOCX, Excel, OCR)];
    G --> H[Advanced Chunk Extraction<br/>(LlamaIndex)];
    H --> I[Qdrant Vector Store];
    H --> J[BM25 Index];

    K[Question] --> L[Query Enhancement];
    L --> M[Hybrid Retrieval<br/>(Dense + Sparse)];
    M --> I;
    M --> J;

    M --> N[Re-ranking];
    N --> O[Answer Generation<br/>(GPT-4o-mini)];
    O --> P[Return RAG-based Answer];
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key
- Google AI API Key

### 1. Clone and Setup
```bash
git clone <repository-url>
cd BajajFinsev

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Docker Deployment
```bash
# Quick start with Docker Compose (includes Qdrant)
docker-compose up -d

# Check health
curl http://localhost:8000/api/v1/hackrx/health
curl http://localhost:6333/health  # Qdrant health
```

### 3. Test the API
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -d 
{
    "documents": "https://www.bajajallianz.com/content/dam/bagic/health-insurance/my-health-care-plan-policy-wordings.pdf",
    "questions": [
      "What is the waiting period for pre-existing diseases?",
      "What are the room rent limits?",
      "What is covered under maternity benefits?"
    ]
  }
```

## 📋 API Endpoints

### Core Endpoints
- `POST /api/v1/hackrx/run` - **Main document analysis** (Gemini-first with RAG fallback).
- `POST /api/v1/hackrx/stream` - **Streaming analysis** for faster initial responses.
- `GET /api/v1/hackrx/health` - **Health check** with component status.

### Performance & Monitoring
- `GET /api/v1/hackrx/performance` - **Detailed performance metrics**.

### Cache Management
- `GET /api/v1/hackrx/cache/stats` - **Comprehensive cache statistics**.
- `POST /api/v1/hackrx/cache/clear` - **Clear all document caches and vector store**.
- `DELETE /api/v1/hackrx/cache/document` - **Remove a specific document from the cache**.

### Example Response
```json
{
  "answers": [
    "The waiting period for pre-existing diseases is thirty-six (36) months of continuous coverage after the date of inception of the first policy. This exclusion applies to expenses related to the treatment of a pre-existing disease and its direct complications.",
    "For Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
  ]
}
```

## ⚙️ Configuration

### Environment Configuration (`.env`)
The application is configured through environment variables. A complete list can be found in `.env.example`.

```bash

# API Authentication
API_KEY=your_api_key_here

# AI Service Keys
OPENAI_API_KEY=sk-proj-your_openai_key_here
GOOGLE_API_KEY=your_google_ai_key_here

# Qdrant Vector Database
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# RAG Fallback Settings
ENABLE_FALLBACK_RAG=true
MAX_FILE_SIZE_MB=100

# Multi-Format Processing
ENABLE_MULTI_FORMAT_SUPPORT=true
OCR_ENGINE=easyocr
```

## 📚 Tech Stack

### Core Technologies
- **Backend**: FastAPI (Python 3.11+)
- **Primary AI Model**: Google Gemini 2.5 Flash
- **Fallback Generation Model**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector Database**: Qdrant
- **RAG Framework**: LlamaIndex
- **Deployment**: Docker, Nginx

### Key Libraries
- `google-generativeai` - For Direct Gemini Analysis
- `openai` - For RAG answer generation and embeddings
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `qdrant-client` - Vector database client
- `llama-index` - RAG framework
- `easyocr` - for OCR support
- `pymupdf`, `python-docx`, `openpyxl` - For document parsing

## 🔍 Troubleshooting

### Common Issues

#### Analysis Fails or Returns Errors
1.  **Check API Keys**: Ensure your `OPENAI_API_KEY` and `GOOGLE_API_KEY` in the `.env` file are correct.
2.  **Check Docker Logs**: Review the logs for the `fastapi-app` container for specific error messages:
    ```bash
    docker-compose logs -f fastapi-app
    ```
3.  **Check Health Endpoint**: Use the health endpoint to see if all components are running correctly:
    ```bash
    curl http://localhost:8000/api/v1/hackrx/health
    ```

#### Slow Performance
- The primary Gemini analysis can sometimes be slow depending on the document size and complexity.
- If the system falls back to RAG, performance depends on the document size and the number of questions.
- Check the performance metrics endpoint for insights:
    ```bash
    curl -H "Authorization: Bearer your_api_key_here" http://localhost:8000/api/v1/hackrx/performance
    ```

#### Vector Database Issues (for RAG fallback)
- **Check Qdrant Health**:
    ```bash
    curl http://localhost:6333/health
    ```
- **Clear Cache**: If you suspect the cache or vector store is corrupted, you can clear it:
    ```bash
    curl -X POST -H "Authorization: Bearer your_api_key_here" http://localhost:8000/api/v1/hackrx/cache/clear
    ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest` (You may need to create tests first)
5. Run linting: `ruff check src/`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
