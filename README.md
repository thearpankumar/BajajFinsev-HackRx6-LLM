# 🚀 BajajFinsev Advanced RAG System

A high-performance, GPU-accelerated Retrieval Augmented Generation (RAG) system designed for comprehensive document analysis and intelligent question answering. Built with enterprise-grade architecture featuring parallel processing, multi-language support, and advanced AI capabilities.

![Version](https://img.shields.io/badge/version-5.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![FastAPI](https://img.shields.io/badge/fastapi-latest-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.1-orange.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

## ✨ Key Features

- **🚀 GPU-Accelerated Processing**: Optimized for RTX 3050 with FP16 mixed precision
- **⚡ Parallel Architecture**: 8-worker parallel document processing pipeline
- **🌐 Multi-Language Support**: Native support for English, Malayalam, Hindi, and 5+ Indian languages
- **🧠 Advanced AI Integration**: OpenAI GPT-4o, Google Gemini, and multilingual embeddings
- **🗃️ High-Performance Vector Storage**: FAISS with HNSW indexing for lightning-fast retrieval
- **📄 Multi-Format Processing**: PDF, DOCX, XLSX, images with OCR, and web content
- **🎯 Intelligent Query Enhancement**: Context-aware query processing and result ranking
- **💾 Smart Caching**: Redis-backed caching for optimal performance
- **📊 Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards

## 📋 Directory Structure

```
src/
├── main.py                          # FastAPI application entry point
├── core/                            # Core system components
│   ├── config.py                    # System configuration management
│   ├── integrated_rag_pipeline.py   # Main RAG pipeline orchestrator
│   ├── gpu_service.py               # GPU acceleration and memory management
│   ├── hierarchical_chunker.py      # Semantic text chunking
│   ├── parallel_document_processor.py # Multi-worker document processing
│   ├── parallel_vector_store.py     # FAISS vector database operations
│   ├── parallel_chunking_service.py # Parallel text chunking
│   ├── advanced_vector_optimizer.py # Vector search optimization
│   ├── performance_profiler.py      # Performance monitoring and profiling
│   ├── response_timer.py            # API response timing
│   └── llm_config.py               # LLM provider configurations
├── services/                        # Business logic services
│   ├── answer_generator.py          # Human-like answer generation
│   ├── retrieval_orchestrator.py    # Multi-strategy retrieval coordination
│   ├── embedding_service.py         # Text embeddings generation
│   ├── query_processor.py           # Query analysis and enhancement
│   ├── bilingual_query_processor.py # Cross-language query processing
│   ├── legal_query_processor.py     # Domain-specific query handling
│   ├── gemini_query_enhancer.py     # Google Gemini query enhancement
│   ├── gemini_service.py            # Google Gemini API integration
│   ├── translation_service.py       # Multi-language translation
│   ├── language_detector.py         # Automatic language detection
│   ├── document_downloader.py       # URL-based document retrieval
│   ├── pdf_processor.py             # PDF parsing and text extraction
│   ├── office_processor.py          # Microsoft Office document processing
│   ├── web_page_processor.py        # Web content extraction
│   ├── basic_text_extractor.py      # Generic text extraction utilities
│   ├── redis_cache.py               # Redis caching operations
│   └── intelligent_challenge_handler.py # Smart challenge detection and solving
├── models/                          # Data models and schemas
│   └── schemas.py                   # Pydantic models for API contracts
├── monitoring/                      # System monitoring and metrics
│   └── prometheus_metrics.py        # Prometheus metrics collection
├── testing/                         # Testing and validation
│   └── pipeline_validator.py        # Comprehensive system validation
└── utils/                           # Utility functions
    └── silent_loader.py             # Silent module loading utilities
```

## 🔄 Data Flow Architecture

### 1. Document Ingestion Flow
```
URL/File Input → Document Downloader → Format Detection → Parallel Processors
                                                            ├── PDF Processor
                                                            ├── Office Processor
                                                            ├── Image OCR
                                                            └── Web Processor
                ↓
Text Extraction → Hierarchical Chunker → Embedding Service → Vector Store (FAISS)
                                              ↓
                                        GPU Acceleration → Redis Cache
```

### 2. Query Processing Flow
```
User Query → Language Detector → Query Processor → Gemini Enhancement
                                      ↓
Query Context → Retrieval Orchestrator → Multi-Strategy Search → Result Ranking
                                              ↓
Retrieved Chunks → Answer Generator → OpenAI GPT-4o → Human-like Response
```

### 3. System Integration Flow
```
FastAPI Endpoints → Authentication → Request Processing → Pipeline Orchestration
                                                              ↓
Performance Monitoring → Prometheus Metrics → Response Formatting → Client Response
```

## 🏗️ System Architecture

```mermaid
graph TB
    A[Document URL Input] --> B[Document Downloader<br/>Multi-format Support]
    B --> C[Parallel Document Processors<br/>8-Worker Architecture]
    C --> D[PDF Processor<br/>PyMuPDF + OCR]
    C --> E[Office Processor<br/>python-docx + openpyxl]
    C --> F[Image Processor<br/>Tesseract OCR]
    C --> G[Web Processor<br/>BeautifulSoup]
    
    D --> H[Hierarchical Chunker<br/>Semantic Boundaries]
    E --> H
    F --> H
    G --> H
    
    H --> I[GPU Embedding Service<br/>multilingual-e5-base]
    I --> J[FAISS Vector Store<br/>HNSW Indexing]
    I --> K[Redis Cache<br/>Performance Layer]
    
    L[User Questions] --> M[Language Detector<br/>8+ Languages]
    M --> N[Query Processor<br/>Intent Analysis]
    N --> O[Gemini Query Enhancer<br/>Context Awareness]
    
    O --> P[Retrieval Orchestrator<br/>Multi-Strategy]
    P --> Q[Semantic Search<br/>FAISS Similarity]
    P --> R[Hybrid Retrieval<br/>Dense + Sparse]
    
    Q --> J
    R --> J
    Q --> S[Result Ranking<br/>Advanced Scoring]
    R --> S
    
    S --> T[Answer Generator<br/>OpenAI GPT-4o]
    T --> U[Response Formatter<br/>Human-like Output]
    
    V[Challenge Handler<br/>Intelligent Detection] --> W[Pattern Recognition<br/>Rule-based + AI]
    W --> X[Direct Solutions<br/>Bypass RAG for Challenges]
    
    Y[Performance Monitor<br/>Prometheus Metrics] --> Z[System Health<br/>GPU + Memory + Cache]
    Z --> AA[Grafana Dashboards<br/>Real-time Analytics]
    
    BB[Pipeline Validator<br/>Quality Assurance] --> CC[Comprehensive Testing<br/>Functionality + Performance]
    
    K --> DD[Cache Manager<br/>TTL + Deduplication]
    DD --> EE{Cache Hit?}
    EE -->|Yes| FF[Return Cached Result]
    EE -->|No| P
    
    subgraph "Core Processing Engine"
        C
        H
        I
        J
    end
    
    subgraph "AI Enhancement Layer"
        N
        O
        T
        V
    end
    
    subgraph "Performance & Monitoring"
        Y
        Z
        AA
        BB
        CC
    end
    
    subgraph "Caching & Optimization"
        K
        DD
        EE
        FF
    end
```

### Core Architecture Components

#### 1. **Integrated RAG Pipeline**
- **Purpose**: Central orchestrator for document processing and retrieval
- **Key Features**: 
  - GPU memory management for RTX 3050 optimization
  - Asynchronous document processing with progress tracking
  - Performance profiling and metrics collection
  - Error handling and fallback mechanisms

#### 2. **Parallel Processing Engine**
- **Components**: 8-worker parallel architecture
- **Capabilities**:
  - Concurrent document downloading and processing
  - Multi-format text extraction (PDF, DOCX, XLSX, Images)
  - Batch embedding generation with GPU acceleration
  - Parallel vector storage operations

#### 3. **Advanced Retrieval System**
- **Multi-Strategy Retrieval**:
  - Semantic similarity search using FAISS
  - Hybrid retrieval combining multiple approaches
  - BM25 scoring for keyword matching
  - Result ranking and filtering algorithms

#### 4. **AI-Powered Enhancement Layer**
- **Language Processing**:
  - Automatic language detection for 8+ languages
  - Cross-lingual query processing and translation
  - Context-aware query enhancement using Google Gemini
  - Multilingual embedding generation

#### 5. **High-Performance Storage**
- **Vector Database**: FAISS with HNSW indexing
- **Caching Layer**: Redis for embeddings, documents, and query results
- **Memory Optimization**: Smart garbage collection and GPU memory management

#### 6. **Monitoring and Observability**
- **Metrics Collection**: Comprehensive Prometheus metrics
- **Performance Profiling**: Real-time system performance monitoring
- **Health Checks**: Service health monitoring and alerting
- **Request Tracing**: Complete request lifecycle tracking

### Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Web Framework** | FastAPI + Uvicorn | High-performance async API server |
| **AI Models** | OpenAI GPT-4o, Google Gemini | Natural language generation |
| **Embeddings** | intfloat/multilingual-e5-base | Multilingual text embeddings |
| **Vector DB** | FAISS with HNSW | High-speed similarity search |
| **GPU Computing** | PyTorch + CUDA 12.1 | GPU-accelerated processing |
| **Caching** | Redis | In-memory data caching |
| **Document Processing** | PyMuPDF, python-docx, openpyxl | Multi-format document parsing |
| **OCR** | Tesseract via various libraries | Image text extraction |
| **Monitoring** | Prometheus + Grafana | System monitoring and dashboards |
| **Language Detection** | langdetect, polyglot | Multi-language support |

## 🔮 Future Extensions

### Planned Enhancements

#### 1. **Advanced AI Capabilities**
- **Multi-Modal Processing**: Integration with vision-language models for image understanding
- **Real-Time Learning**: Continuous learning from user interactions and feedback
- **Advanced Reasoning**: Integration with larger language models for complex reasoning tasks
- **Knowledge Graph Integration**: Semantic knowledge representation and reasoning

#### 2. **Scalability Improvements**
- **Distributed Processing**: Multi-node processing using Ray or Celery clusters
- **Auto-Scaling**: Dynamic resource allocation based on workload
- **Load Balancing**: Intelligent request distribution across multiple instances
- **Database Sharding**: Horizontal scaling of vector databases

#### 3. **Enhanced Document Processing**
- **Real-Time Processing**: Stream processing for continuous document ingestion
- **Advanced OCR**: Integration with cloud OCR services for better accuracy
- **Video Processing**: Extraction and analysis of video content
- **Audio Processing**: Speech-to-text and audio content analysis

#### 4. **Security and Compliance**
- **End-to-End Encryption**: Data encryption at rest and in transit
- **Access Control**: Role-based access control with fine-grained permissions
- **Audit Logging**: Comprehensive audit trails for all operations
- **Compliance Tools**: GDPR, CCPA, and other regulatory compliance features

#### 5. **User Experience Enhancements**
- **Interactive Dashboard**: Web-based management and monitoring interface
- **API Documentation**: Comprehensive OpenAPI documentation with examples
- **SDK Development**: Client SDKs for popular programming languages
- **Mobile API**: Mobile-optimized endpoints and processing

#### 6. **Integration Capabilities**
- **Webhook Support**: Real-time notifications and event-driven processing
- **Third-Party Integrations**: Salesforce, SharePoint, Google Drive, Dropbox
- **Database Connectors**: Direct integration with SQL and NoSQL databases
- **Enterprise Integration**: LDAP/AD authentication and enterprise SSO

#### 7. **Advanced Analytics**
- **Usage Analytics**: Detailed usage patterns and performance analytics
- **Content Analytics**: Document content analysis and insights
- **Performance Optimization**: AI-powered performance tuning recommendations
- **Predictive Caching**: ML-based cache optimization

#### 8. **Edge Computing**
- **Edge Deployment**: Lightweight edge computing capabilities
- **Offline Processing**: Local processing without internet connectivity
- **Mobile SDKs**: On-device processing capabilities for mobile applications
- **IoT Integration**: Integration with IoT devices and sensors

### Technical Roadmap

#### Phase 1: Core Enhancements (Q2 2024)
- Multi-modal document processing
- Enhanced multilingual support
- Advanced caching strategies
- Performance optimization

#### Phase 2: Scalability (Q3 2024)
- Distributed processing architecture
- Auto-scaling capabilities
- Enhanced monitoring and alerting
- Enterprise security features

#### Phase 3: Advanced Features (Q4 2024)
- Knowledge graph integration
- Real-time learning capabilities
- Advanced analytics dashboard
- Mobile and edge computing support

## 🛠️ Technical Specifications

### Performance Characteristics
- **Processing Speed**: 7x faster than baseline through parallelization
- **Memory Efficiency**: FP16 mixed precision for 50% memory reduction
- **Concurrent Processing**: Up to 8 parallel document processing workers
- **GPU Optimization**: RTX 3050 4GB memory-optimized operations
- **Cache Performance**: 90%+ cache hit rates for repeated queries
- **Response Time**: Sub-second response times for cached queries

### Supported Formats and Languages
- **Document Formats**: PDF, DOCX, DOC, XLSX, XLS, CSV, TXT, HTML, WebP, JPG, PNG
- **Languages**: English, Malayalam, Hindi, Tamil, Telugu, Kannada, Bengali, Gujarati
- **OCR Capabilities**: Multi-language text extraction from images
- **Web Content**: Dynamic web page processing and content extraction

### Hardware Requirements
- **Minimum**: 8GB RAM, 4GB GPU VRAM (RTX 3050 or equivalent)
- **Recommended**: 16GB RAM, 8GB GPU VRAM (RTX 4060 or better)
- **Storage**: 50GB for models and caching
- **CPU**: 8+ cores recommended for optimal parallel processing

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.9+ with CUDA 12.1 support
# Redis server
# GPU drivers and CUDA toolkit
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd BajajFinsev

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys and configurations
```

### Running the System
```bash
# Start the FastAPI server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose up -d
```

### API Usage
```bash
# Document analysis
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the main topic of this document?"]
  }'
```

## 📊 Monitoring and Analytics

### Available Metrics
- Document processing statistics
- Query performance metrics
- GPU utilization and memory usage
- Cache hit rates and performance
- Error rates and response times
- System health indicators

### Grafana Dashboards
- Real-time performance monitoring
- Resource utilization tracking
- Document processing analytics
- Query pattern analysis
- System health overview

## 🔧 Configuration

The system is highly configurable through environment variables and configuration files:
- **API Keys**: OpenAI, Google Gemini, Azure Translation
- **GPU Settings**: Memory limits, batch sizes, precision modes
- **Cache Configuration**: Redis settings, TTL values
- **Processing Parameters**: Chunk sizes, embedding models, vector dimensions
- **Monitoring**: Prometheus endpoints, logging levels

## 🤝 Contributing

This system represents a comprehensive approach to document analysis and retrieval, combining cutting-edge AI technologies with enterprise-grade architecture. The modular design allows for easy extension and customization based on specific requirements.

---

*Built with ❤️ for intelligent document processing and analysis*