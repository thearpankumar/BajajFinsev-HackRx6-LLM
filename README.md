# BajajFinsev Advanced Document Analysis API

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/r/arpankumar1119/hackrx-bajaj)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?logo=openai)](https://openai.com)
[![Google](https://img.shields.io/badge/Google-Gemini--2.5--Flash-blue?logo=google)](https://ai.google.dev)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red?logo=qdrant)](https://qdrant.tech)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG_Framework-purple)](https://llamaindex.ai)
[![Performance](https://img.shields.io/badge/Performance-5.2x_Faster-brightgreen)](https://github.com)

A **high-performance, enterprise-grade API** for deep analysis of business documents, specializing in **Insurance, Legal, HR, and Compliance** domains. Features advanced **parallel processing**, **complex question decomposition**, **hybrid search**, and **optimized RAG pipeline** with **Qdrant vector database** and **LlamaIndex framework** for lightning-fast document analysis.

## 🚀 Key Features

### ⚡ **Ultra-Fast Performance**
- **5.2x faster** than sequential processing with parallel question handling
- **Under 60 seconds** for complex multi-part questions
- **Up to 40 questions processed simultaneously** in batches of 10
- **Smart question decomposition** for complex queries
- **Optimized for speed** while maintaining high accuracy

### 🧠 **Advanced RAG Pipeline with Qdrant + LlamaIndex**
- **Qdrant Vector Database**: High-performance, production-ready vector storage
- **LlamaIndex Integration**: Advanced document processing with automatic fallback
- **Enhanced Document Processor**: Better PDF/DOCX parsing with larger context windows
- **Hybrid Search**: Combines semantic (OpenAI embeddings) + keyword (BM25) search
- **Smart Re-ranking**: Optional cross-encoder re-ranking for maximum relevance
- **Dynamic Similarity Threshold**: Adaptive filtering for optimal results
- **Persistent Document Caching**: Fast repeated queries with intelligent cache management

### 🎯 **Complex Question Handling**
- **Multi-Part Question Decomposition**: Handles "while X, also Y, and Z" patterns
- **Rule-Based Fast Processing**: No AI overhead for decomposition
- **Parallel Sub-Question Processing**: Process question parts simultaneously
- **Smart Answer Combination**: Natural flow without repetitive phrases
- **Query Enhancement**: Automatic addition of relevant insurance terms
- **Context Aggregation**: Better information retrieval across question parts

### 🔧 **Domain Expertise**
- **Specialized for**: Insurance policies, Legal contracts, HR documents, Compliance materials
- **Multi-domain Query Enhancement**: Intelligent query expansion for each domain
- **Concise Responses**: 2-3 sentence answers with specific facts and figures
- **Context-Aware**: Deep understanding of domain-specific terminology
- **Google Gemini Integration**: Advanced query processing with Gemini 2.5 Flash

### 🏭 **Production Ready**
- **Docker deployment** with Qdrant and Nginx reverse proxy
- **Health monitoring** and comprehensive performance metrics
- **Configurable accuracy settings** (Fast vs Accurate modes)
- **Horizontal scaling** support with load balancing
- **Robust error handling** with automatic fallbacks
- **Persistent caching** with TTL and duplicate detection

## 📊 Performance Benchmarks

| Configuration | Questions | Processing Time | Avg/Question | Speedup |
|---------------|-----------|----------------|--------------|---------|
| **Parallel Mode** | 10 | 16.56s | 1.66s | **5.2x faster** |
| **Complex Questions** | 25 | <60s | <2.4s | **3x faster** |
| **Sequential Mode** | 10 | 86.83s | 8.68s | 1x (baseline) |
| **Single Question** | 1 | 3-6s | 3-6s | **2x faster** |

## 🏗️ Architecture

```mermaid
graph TB
    A[Document URL] --> B[Enhanced Document Processor<br/>LlamaIndex + Fallback]
    B --> C[Advanced Chunk Extraction<br/>SentenceWindow + Simple]
    C --> D[Qdrant Vector Store<br/>Production Ready]
    C --> E[BM25 Index<br/>Sparse Search]
    
    F[Complex Questions] --> G[Question Decomposition<br/>Rule-Based Fast]
    G --> H[Sub-Questions<br/>Parallel Processing]
    
    H --> I[Query Enhancement<br/>Domain-Specific Terms]
    I --> J[Hybrid Retrieval<br/>Dense + Sparse]
    J --> D
    J --> E
    
    J --> K[Enhanced RRF Algorithm<br/>Reciprocal Rank Fusion]
    K --> L{Re-ranking Enabled?}
    L -->|Yes| M[Cross-Encoder Re-ranking]
    L -->|No| N[Skip Re-ranking for Speed]
    M --> O[Answer Generation<br/>GPT-4o-mini]
    N --> O
    
    O --> P[Smart Answer Combination<br/>Natural Flow]
    P --> Q[Concise Answers<br/>2-3 sentences]
    
    R[Document Cache Manager] --> S[Persistent Cache<br/>TTL + Deduplication]
    S --> T[Vector DB Check<br/>Skip Duplicates]
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
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": [
      "What is the waiting period for pre-existing diseases?",
      "What are the room rent limits?",
      "What is covered under maternity benefits?"
    ]
  }'
```

## 📋 API Endpoints

### Core Endpoints
- `POST /api/v1/hackrx/run` - **Main document analysis** (parallel + complex question handling)
- `POST /api/v1/hackrx/stream` - **Streaming analysis** for faster initial responses
- `GET /api/v1/hackrx/health` - **Health check** with component status

### Performance & Monitoring
- `GET /api/v1/hackrx/performance` - **Detailed performance metrics**
- `GET /api/v1/hackrx/performance/mode` - **Current performance mode**
- `POST /api/v1/hackrx/performance/mode` - **Set performance mode** (fast/accurate)
- `POST /api/v1/hackrx/performance/complex-questions` - **Configure complex question processing**

### Enhanced Processing
- `GET /api/v1/hackrx/processing/stats` - **Document processing statistics**
- `GET /api/v1/hackrx/processing/test` - **Test enhanced processing capabilities**

### Cache Management
- `GET /api/v1/hackrx/cache/stats` - **Comprehensive cache statistics**
- `POST /api/v1/hackrx/cache/clear` - **Clear all document caches**
- `DELETE /api/v1/hackrx/cache/document` - **Remove specific document from cache**

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

### Performance Modes

#### 🚀 **Fast Mode (Default - Recommended)**
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/mode" \
  -H "Authorization: Bearer <token>" -d "fast"
```
- ✅ **5.2x faster processing**
- ✅ **Parallel question handling**
- ✅ **Fast complex question decomposition**
- ✅ **Optimized for production**
- ⚠️ **Slightly less accurate** (but still very good)

#### 🎯 **Accurate Mode**
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/mode" \
  -H "Authorization: Bearer <token>" -d "accurate"
```
- ✅ **Maximum accuracy** with re-ranking
- ✅ **Full context processing**
- ✅ **Enhanced query expansion**
- ⚠️ **Slower processing** (~2x)

### Complex Question Processing

#### ⚡ **Fast Complex Questions (Default)**
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/complex-questions" \
  -H "Authorization: Bearer <token>" -d "fast_mode=true"
```
- ✅ **Rule-based decomposition** (no AI overhead)
- ✅ **Parallel sub-question processing**
- ✅ **Under 60 seconds** for complex queries
- ✅ **Smart answer combination**

#### 🧠 **Comprehensive Complex Questions**
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/complex-questions" \
  -H "Authorization: Bearer <token>" -d "fast_mode=false"
```
- ✅ **AI-powered decomposition**
- ✅ **Context aggregation**
- ✅ **Maximum accuracy**
- ⚠️ **Slower processing**

### Environment Configuration (.env)

```bash
# API Authentication
API_KEY=your_api_key_here

# AI Service Keys
OPENAI_API_KEY=sk-proj-your_openai_key_here
GOOGLE_API_KEY=your_google_ai_key_here

# Performance Settings
FAST_MODE=true
ENABLE_RERANKING=false
MAX_CHUNKS_FOR_GENERATION=5

# Enhanced Question Processing
ENABLE_QUESTION_DECOMPOSITION=true
COMPLEX_QUESTION_MAX_TOKENS=250
FAST_COMPLEX_QUESTIONS=true
ENABLE_QUERY_ENHANCEMENT=true

# Parallel Processing
PARALLEL_PROCESSING=true
MAX_PARALLEL_QUESTIONS=40
QUESTION_BATCH_SIZE=10

# Document Caching
ENABLE_PERSISTENT_DOCUMENT_CACHE=true
DOCUMENT_CACHE_PATH=./document_cache
DOCUMENT_CACHE_TTL_HOURS=168
```

See [`.env.example`](.env.example) for complete configuration options.

## 📈 Monitoring & Metrics

### Performance Metrics
```bash
# Get detailed metrics
curl http://localhost:8000/api/v1/hackrx/performance

# Response includes:
{
  "total_requests": 150,
  "avg_processing_time": 1.66,
  "parallel_processing_enabled": true,
  "cache_hit_rate": 0.85,
  "memory_usage": "2.1GB",
  "questions_per_minute": 37,
  "complex_questions_processed": 45,
  "decomposition_success_rate": 0.92
}
```

### Health Monitoring
```bash
# Component health check
curl http://localhost:8000/api/v1/hackrx/health

# Response:
{
  "status": "healthy",
  "components": {
    "vector_database": "healthy",
    "embedding_model": "healthy",
    "generation_model": "healthy",
    "document_processor": "enhanced",
    "cache_manager": "healthy"
  },
  "timestamp": 1704188545.614973
}
```

### Cache Statistics
```bash
# Comprehensive cache stats
curl http://localhost:8000/api/v1/hackrx/cache/stats

# Response includes persistent cache, memory cache, and vector DB stats
{
  "persistent_cache": {
    "total_documents": 25,
    "cache_size_mb": 145.2,
    "hit_rate": 0.78
  },
  "memory_cache": {
    "memory_cache_documents": 12,
    "memory_chunk_cache_documents": 8
  },
  "vector_database": {
    "total_vectors": 15420,
    "collection_size": "2.1GB"
  }
}
```

## 🏭 Production Deployment

### System Requirements
- **Memory**: 4GB minimum, 8GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Storage**: 10GB for models and cache
- **Network**: Stable internet for AI API calls

### Docker Compose Production
```bash
# Production deployment with all services
docker-compose up -d

# Scale for high load
docker-compose up -d --scale fastapi-app=3

# SSL/HTTPS ready with Let's Encrypt
docker-compose --profile tools run --rm certbot certonly \
  --webroot --webroot-path /var/www/certbot/ -d your-domain.com
```

### Load Balancing with Nginx
- ✅ **Rate limiting**: 10 requests/second per IP
- ✅ **Request buffering** for large documents
- ✅ **Health check routing**
- ✅ **SSL termination** ready
- ✅ **Horizontal scaling** support

## 🧪 Testing

### Run Tests
```bash
# Basic functionality test
python test_server.py

# Performance test
python test_performance.py

# Complex question test
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer <token>" \
  -d @payloads/payload8.json
```

### Sample Test Payloads
```bash
# Test with complex multi-part questions
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer <token>" \
  -d @payloads/payload7.json

# Test with insurance policy questions
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer <token>" \
  -d @payloads/payload1.json
```

## 🔍 Troubleshooting

### Common Issues

#### Slow Performance
```bash
# Check current mode
curl http://localhost:8000/api/v1/hackrx/performance/mode

# Enable fast mode
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/mode" -d "fast"

# Enable fast complex questions
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/complex-questions" -d "fast_mode=true"
```

#### Complex Questions Taking Too Long
```bash
# Disable question decomposition temporarily
# Edit .env file:
ENABLE_QUESTION_DECOMPOSITION=false

# Or disable query enhancement
ENABLE_QUERY_ENHANCEMENT=false
```

#### Memory Issues
```bash
# Check resource usage
docker stats fastapi-app-container

# Reduce parallel processing
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/parallel" \
  -d '{"max_parallel": 20, "batch_size": 5}'
```

#### Vector Database Issues
```bash
# Check Qdrant health
curl http://localhost:6333/health

# Check collection status
curl http://localhost:6333/collections/bajaj_documents

# Clear cache if needed
curl -X POST "http://localhost:8000/api/v1/hackrx/cache/clear" \
  -H "Authorization: Bearer <token>"
```

#### Document Processing Issues
```bash
# Check processing capabilities
curl http://localhost:8000/api/v1/hackrx/processing/test

# Check processing stats
curl http://localhost:8000/api/v1/hackrx/processing/stats
```

## 📚 Tech Stack

### Core Technologies
- **Backend**: FastAPI (Python 3.12+)
- **Vector Database**: Qdrant (high-performance, production-ready)
- **RAG Framework**: LlamaIndex (advanced document processing)
- **AI Models**: 
  - OpenAI GPT-4o-mini (generation)
  - OpenAI text-embedding-3-small (embeddings)
  - Google Gemini 2.5 Flash (query processing)
- **Search**: Hybrid (Dense + Sparse BM25)
- **Deployment**: Docker, Nginx

### Key Libraries
- `uvicorn` - ASGI server
- `qdrant-client` - Vector database client
- `llama-index` - RAG framework with enhanced document processing
- `llama-index-readers-file` - Advanced PDF/DOCX readers
- `sentence-transformers` - Cross-encoder re-ranking
- `rank_bm25` - Sparse retrieval
- `google-generativeai` - Google Gemini integration
- `PyMuPDF` - PDF processing
- `python-docx` - Word document processing
- `nltk`, `spacy` - Text processing

### Architecture Components
- **Enhanced Document Processor**: LlamaIndex + fallback processing
- **Qdrant Vector Store**: Production-ready vector database
- **Document Cache Manager**: Persistent caching with TTL
- **Performance Monitor**: Comprehensive metrics tracking
- **Complex Question Handler**: Multi-part question decomposition
- **Query Enhancement**: Domain-specific term expansion

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python -m pytest`
5. Run linting: `ruff check src/`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation
- [Docker Deployment Guide](DOCKER_DEPLOYMENT.md)
- [API Documentation](http://localhost:8000/docs) (when running)
- [Performance Tuning Guide](docs/performance.md)

### Getting Help
1. Check the [troubleshooting section](#-troubleshooting)
2. Review logs: `docker-compose logs -f fastapi-app`
3. Test health: `curl http://localhost:8000/api/v1/hackrx/health`
4. Check Qdrant: `curl http://localhost:6333/health`
5. Check processing: `curl http://localhost:8000/api/v1/hackrx/processing/test`
6. Check performance: `curl http://localhost:8000/api/v1/hackrx/performance`

#### Quick Commands
```bash
# View service status
docker-compose ps

# Check logs
docker-compose logs -f fastapi-app
docker-compose logs -f qdrant

# Restart services
docker-compose restart

# Test complex questions
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer <token>" -d @payloads/payload8.json

# Check cache stats
curl http://localhost:8000/api/v1/hackrx/cache/stats
```

---

## 🎯 **Ready for Production**

The BajajFinsev RAG System is optimized for **high-performance document analysis** with **complex question handling**, **parallel processing**, **intelligent caching**, and **enterprise-grade reliability**. Deploy with confidence! 🚀

**Performance**: 5.2x faster | **Complex Questions**: <60s | **Accuracy**: Domain-specialized | **Scale**: Production-ready | **Database**: Qdrant Vector DB | **Processing**: LlamaIndex Enhanced
