# BajajFinsev Advanced Document Analysis API

[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://hub.docker.com/r/arpankumar1119/hackrx-bajaj)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange?logo=openai)](https://openai.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red?logo=qdrant)](https://qdrant.tech)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG_Framework-purple)](https://llamaindex.ai)
[![Performance](https://img.shields.io/badge/Performance-5.2x_Faster-brightgreen)](https://github.com)

A **high-performance, enterprise-grade API** for deep analysis of business documents, specializing in **Insurance, Legal, HR, and Compliance** domains. Features advanced **parallel processing**, **hybrid search**, and **optimized RAG pipeline** with **Qdrant vector database** and **LlamaIndex framework** for lightning-fast document analysis.

## 🚀 Key Features

### ⚡ **Ultra-Fast Performance**
- **5.2x faster** than sequential processing with parallel question handling
- **1.66 seconds average** per question in fast mode
- **Up to 40 questions processed simultaneously** in batches of 10
- **Optimized for speed** while maintaining high accuracy

### 🧠 **Advanced RAG Pipeline with Qdrant + LlamaIndex**
- **Qdrant Vector Database**: High-performance, production-ready vector storage
- **LlamaIndex Integration**: Advanced document processing and retrieval
- **Hybrid Search**: Combines semantic (OpenAI embeddings) + keyword (BM25) search
- **Smart Re-ranking**: Optional cross-encoder re-ranking for maximum relevance
- **Dynamic Similarity Threshold**: Adaptive filtering for optimal results
- **Rich Metadata Extraction**: Entities, concepts, categories, and keywords

### 🎯 **Domain Expertise**
- **Specialized for**: Insurance policies, Legal contracts, HR documents, Compliance materials
- **Multi-domain Query Enhancement**: Intelligent query expansion for each domain
- **Concise Responses**: 2-3 sentence answers with specific facts and figures
- **Context-Aware**: Deep understanding of domain-specific terminology

### 🔧 **Production Ready**
- **Docker deployment** with Qdrant and Nginx reverse proxy
- **Health monitoring** and performance metrics
- **Configurable accuracy settings** (Fast vs Accurate modes)
- **Horizontal scaling** support with load balancing
- **Robust error handling** with automatic fallbacks

## 📊 Performance Benchmarks

| Configuration | Questions | Processing Time | Avg/Question | Speedup |
|---------------|-----------|----------------|--------------|---------|
| **Parallel Mode** | 10 | 16.56s | 1.66s | **5.2x faster** |
| **Sequential Mode** | 10 | 86.83s | 8.68s | 1x (baseline) |
| **Single Question** | 1 | 3-6s | 3-6s | **2x faster** |

## 🏗️ Architecture

```mermaid
graph TB
    A[Document URL] --> B[LlamaIndex Document Processor]
    B --> C[Enhanced Chunk Extraction]
    C --> D[Qdrant Vector Store<br/>Production Ready]
    C --> E[BM25 Index]
    
    F[Questions] --> G{Parallel Processing?}
    G -->|Yes| H[Batch Processing<br/>40 questions max]
    G -->|No| I[Sequential Processing]
    
    H --> J[Hybrid Retrieval]
    I --> J
    J --> D
    J --> E
    
    J --> K[Enhanced RRF Algorithm]
    K --> L{Re-ranking Enabled?}
    L -->|Yes| M[Cross-Encoder Re-ranking]
    L -->|No| N[Skip Re-ranking]
    M --> O[Answer Generation<br/>GPT-4o-mini]
    N --> O
    
    O --> P[Concise Answers<br/>2-3 sentences]
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
- `POST /api/v1/hackrx/run` - **Main document analysis** (parallel processing)
- `POST /api/v1/hackrx/stream` - **Streaming analysis** for faster initial responses
- `GET /api/v1/hackrx/health` - **Health check** with component status

### Performance & Monitoring
- `GET /api/v1/hackrx/performance` - **Detailed performance metrics**
- `GET /api/v1/hackrx/performance/mode` - **Current performance mode**
- `POST /api/v1/hackrx/performance/mode` - **Set performance mode** (fast/accurate)
- `POST /api/v1/hackrx/performance/parallel` - **Configure parallel processing**

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
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/mode" -d "fast"
```
- ✅ **5.2x faster processing**
- ✅ **Parallel question handling**
- ✅ **Optimized for production**
- ⚠️ **Slightly less accurate** (but still very good)

#### 🎯 **Accurate Mode**
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/mode" -d "accurate"
```
- ✅ **Maximum accuracy** with re-ranking
- ✅ **Full context processing**
- ✅ **Enhanced query expansion**
- ⚠️ **Slower processing** (~2x)

### Accuracy Control Settings

#### 🎛️ **Easy Control Script**
```bash
python toggle_accuracy.py
# Choose: 1=Max Accuracy, 2=Balanced, 3=Max Speed
```

#### 🔧 **Manual Configuration (.env)**
```bash
# Accuracy Settings
ENABLE_RERANKING=true          # Enable cross-encoder re-ranking
USE_ENHANCED_QUERY=true        # Enhanced query processing
USE_ENHANCED_RRF=true          # Advanced ranking algorithm
SIMILARITY_THRESHOLD=0.1       # Vector similarity threshold
MAX_CHUNKS_FOR_GENERATION=8    # Context size for answers
MAX_GENERATION_TOKENS=180      # Answer length (2-3 sentences)
GENERATION_TEMPERATURE=0.05    # Answer consistency

# Performance Settings
PARALLEL_PROCESSING=true
MAX_PARALLEL_QUESTIONS=40
QUESTION_BATCH_SIZE=10
```

### Parallel Processing Configuration
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/performance/parallel" \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "max_parallel": 40,
    "batch_size": 10
  }'
```

## 🐳 Docker Deployment

### Development
```bash
# With auto-reload
docker-compose up --build

# View logs
docker-compose logs -f fastapi-app
```

### Production
```bash
# Production deployment with Nginx
docker-compose -f docker-compose.yml up -d

# Scale for high load
docker-compose up -d --scale fastapi-app=3

# SSL/HTTPS ready with Let's Encrypt
docker-compose --profile tools run --rm certbot certonly --webroot --webroot-path /var/www/certbot/ -d your-domain.com
```

## 🔧 Environment Variables

### Required Variables
```bash
# API Authentication
API_KEY=your_api_key_here

# AI Service Keys
OPENAI_API_KEY=sk-proj-your_openai_key_here
GOOGLE_API_KEY=your_google_ai_key_here
```

### Performance Optimization
```bash
# Fast Mode Settings
FAST_MODE=true
ENABLE_RERANKING=false
MAX_CHUNKS_FOR_GENERATION=5

# Parallel Processing
PARALLEL_PROCESSING=true
MAX_PARALLEL_QUESTIONS=40
QUESTION_BATCH_SIZE=10

# Retrieval Settings
TOP_K_RETRIEVAL=10
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
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
  "questions_per_minute": 37
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
    "generation_model": "healthy"
  },
  "timestamp": 1704188545.614973
}
```

## 🏭 Production Deployment

### System Requirements
- **Memory**: 4GB minimum, 8GB recommended
- **CPU**: 2 cores minimum, 4 cores recommended
- **Storage**: 10GB for models and cache
- **Network**: Stable internet for AI API calls

### Scaling Configuration
```yaml
# docker-compose.yml
deploy:
  resources:
    limits:
      memory: 6G
      cpus: '3.0'
    reservations:
      memory: 3G
      cpus: '2.0'
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

# Parallel processing test
python test_parallel.py
```

### Sample Test Payloads
```bash
# Test with sample payload
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer 589a89f8010526700b24d76902776ce49372734b564ea3324b495c4cec6f2b68" \
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

# Run diagnostics
python diagnose_vector_db.py
```

#### "No dense results found" Error
```bash
# Check similarity threshold
# Edit .env file:
SIMILARITY_THRESHOLD=0.1  # Lower threshold for more results

# Or disable threshold temporarily
SIMILARITY_THRESHOLD=0.0
```

#### API Key Issues
```bash
# Verify environment variables
docker exec fastapi-app-container env | grep API_KEY
```

## 📚 Tech Stack

### Core Technologies
- **Backend**: FastAPI (Python 3.12+)
- **Vector Database**: Qdrant (high-performance, production-ready)
- **RAG Framework**: LlamaIndex (advanced document processing)
- **AI Models**: 
  - OpenAI GPT-4o-mini (generation)
  - OpenAI text-embedding-3-small (embeddings)
  - Google Gemini 2.0 Flash (query processing)
- **Search**: Hybrid (Dense + Sparse BM25)
- **Deployment**: Docker, Nginx

### Key Libraries
- `uvicorn` - ASGI server
- `qdrant-client` - Vector database client
- `llama-index` - RAG framework
- `sentence-transformers` - Cross-encoder re-ranking
- `rank_bm25` - Sparse retrieval
- `PyMuPDF` - PDF processing
- `python-docx` - Word document processing
- `nltk`, `spacy` - Text processing

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
4. Check performance: `curl http://localhost:8000/api/v1/hackrx/performance`

---

## 🎯 **Ready for Production**

The BajajFinsev RAG System is optimized for **high-performance document analysis** with **parallel processing**, **intelligent caching**, and **enterprise-grade reliability**. Deploy with confidence! 🚀

**Performance**: 5.2x faster | **Accuracy**: Domain-specialized | **Scale**: Production-ready | **Database**: Qdrant Vector DB

### 🆘 Support

#### Documentation
- [Docker Deployment Guide](DOCKER_DEPLOYMENT.md)
- [API Documentation](http://localhost:8000/docs) (when running)
- [Performance Tuning Guide](docs/performance.md)

#### Getting Help
1. Check the [troubleshooting section](#-troubleshooting)
2. Review logs: `docker-compose logs -f fastapi-app`
3. Test health: `curl http://localhost:8000/api/v1/hackrx/health`
4. Check Qdrant: `curl http://localhost:6333/health`
5. Run diagnostics: `python diagnose_vector_db.py`
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

# Test accuracy settings
python toggle_accuracy.py

# Run diagnostics
python diagnose_vector_db.py
```
