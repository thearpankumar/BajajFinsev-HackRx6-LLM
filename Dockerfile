# ==============================================================================
# Production Docker Image for Embedding-based RAG System
# ==============================================================================
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Create a non-root user for security
RUN addgroup --system appuser && adduser --system --ingroup appuser --home /home/appuser appuser

# Copy requirements first to leverage Docker layer caching
COPY --chown=appuser:appuser requirements.txt .

# Install system dependencies for PyMuPDF, NumPy, scikit-learn, and OpenAI
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    swig \
    curl && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential gfortran swig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Create cache directory for embeddings with proper permissions
RUN mkdir -p /tmp/embedding_cache && \
    chown -R appuser:appuser /tmp/embedding_cache

# Copy the application source code
COPY --chown=appuser:appuser src/ ./src

# Copy environment file (if exists)
COPY --chown=appuser:appuser .env* ./

# Switch to the non-root user
USER appuser

# Health check for the application
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables for production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# The command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

