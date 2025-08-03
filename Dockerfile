# ==============================================================================
# Optimized Production Docker Image for BajajFinsev RAG System
# Minimal size with essential components only
# ==============================================================================

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install only essential system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
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
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Download only essential NLTK data (minimal)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)" && \
    find /root/nltk_data -name "*.zip" -delete 2>/dev/null || true

# Create necessary directories
RUN mkdir -p \
    /app/vector_db \
    /tmp/embedding_cache \
    /app/logs && \
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
    TOKENIZERS_PARALLELISM=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/hackrx/health || exit 1

# Expose port
EXPOSE 8000

# Simple startup command (no complex script)
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ==============================================================================
# Size Optimizations Applied:
# 
# 1. Single-stage build (no multi-stage overhead)
# 2. Minimal system dependencies (only curl for health checks)
# 3. No pre-downloaded models (lazy loading)
# 4. Minimal NLTK data (only essential packages)
# 5. No build tools in final image
# 6. Cleaned package caches
# 7. Removed unnecessary files
# 8. Optimized layer structure
# 
# Expected size reduction: ~70% smaller (from 2GB+ to ~600MB)
# ==============================================================================
