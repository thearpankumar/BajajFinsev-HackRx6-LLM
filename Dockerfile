# ==============================================================================
# Optimized Production Docker Image for BajajFinsev RAG System
# Updated with LlamaIndex + Qdrant integration
# ==============================================================================

#docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies including build tools for some packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ==============================================================================
# Production stage (single stage for smaller image)
# ==============================================================================

# Create non-root user
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid 1001 --home /home/appuser --create-home appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with minimal footprint
RUN pip install --upgrade pip --root-user-action=ignore && \
    pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Download NLTK data as root first, then copy to user directory
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)" && \
    find /root/nltk_data -name "*.zip" -delete 2>/dev/null || true

# Create necessary directories with proper permissions
RUN mkdir -p \
    /app/vector_db \
    /tmp/embedding_cache \
    /app/logs \
    /home/appuser/nltk_data && \
    cp -r /root/nltk_data/* /home/appuser/nltk_data/ 2>/dev/null || true && \
    chown -R appuser:appuser /app /tmp/embedding_cache /home/appuser

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser .env* ./
COPY --chown=appuser:appuser payloads/ ./payloads/

# Switch to non-root user
USER appuser

# Set performance environment variables (optimized defaults)
ENV FAST_MODE=true \
    ENABLE_RERANKING=false \
    MAX_CHUNKS_FOR_GENERATION=5 \
    PARALLEL_PROCESSING=true \
    MAX_PARALLEL_QUESTIONS=40 \
    QUESTION_BATCH_SIZE=10 \
    PYTHONPATH=/app \
    TOKENIZERS_PARALLELISM=false \
    NLTK_DATA=/home/appuser/nltk_data \
    VECTOR_DB_PATH=/app/vector_db \
    QDRANT_HOST=qdrant \
    QDRANT_PORT=6333 \
    QDRANT_COLLECTION_NAME=bajaj_documents

# Health check - updated to be more lenient during startup
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/api/v1/hackrx/health || exit 1

# Expose port
EXPOSE 8000

# Simple startup command (no complex script)
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ==============================================================================
# Migration Updates Applied:
# 
# 1. Added LlamaIndex and Qdrant client dependencies
# 2. Added migration utility scripts to container
# 3. Set Qdrant environment variables for Docker networking
# 4. Extended health check start period for Qdrant initialization
# 5. Maintained all existing optimizations
# 
# Expected benefits:
# - Eliminates LanceDB Docker issues
# - Better large document processing
# - Improved reliability and performance
# ==============================================================================
