# ==============================================================================
# Production Docker Image for Business Document Analysis API
# Specialized for Insurance, Legal, HR, and Compliance domains
# ==============================================================================

# Use multi-stage build for better layer caching and smaller final image
FROM python:3.11-slim as builder

# Set build-time environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    swig \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --user --no-warn-script-location -r requirements.txt

# Download NLTK and SpaCy data
RUN python -c "import nltk; nltk.download('punkt_tab', download_dir='/tmp/nltk_data'); nltk.download('punkt', download_dir='/tmp/nltk_data'); nltk.download('stopwords', download_dir='/tmp/nltk_data'); nltk.download('wordnet', download_dir='/tmp/nltk_data'); nltk.download('averaged_perceptron_tagger_eng', download_dir='/tmp/nltk_data'); nltk.download('averaged_perceptron_tagger', download_dir='/tmp/nltk_data'); nltk.download('maxent_ne_chunker', download_dir='/tmp/nltk_data'); nltk.download('maxent_ne_chunker_tab', download_dir='/tmp/nltk_data'); nltk.download('words', download_dir='/tmp/nltk_data')"
RUN python -m spacy download en_core_web_sm

# Download sentence-transformers models to avoid download during runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ==============================================================================
# Production stage
# ==============================================================================
FROM python:3.11-slim as production

# Set runtime environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NLTK_DATA=/app/nltk_data
ENV TORCH_HOME=/tmp/torch_cache
ENV TRANSFORMERS_CACHE=/tmp/torch_cache

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libmupdf23 \
    libopenblas0 \
    liblapack3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN addgroup --system --gid 1001 appuser && \
    adduser --system --uid 1001 --gid 1001 --home /home/appuser appuser

# Set the working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy NLTK data from builder stage
COPY --from=builder /tmp/nltk_data /app/nltk_data

# Copy SpaCy models from builder stage
COPY --from=builder /root/.local/lib/python3.11/site-packages/en_core_web_sm /home/appuser/.local/lib/python3.11/site-packages/en_core_web_sm

# Create application directories with proper permissions
RUN mkdir -p /tmp/embedding_cache /tmp/lancedb /tmp/torch_cache && \
    chown -R appuser:appuser /app /tmp/embedding_cache /tmp/lancedb /tmp/torch_cache /home/appuser

# Copy the application source code
COPY --chown=appuser:appuser src/ ./src

# Copy environment file (if exists)
COPY --chown=appuser:appuser .env* ./

# Switch to the non-root user
USER appuser

# Update PATH to include user-installed packages
ENV PATH="/home/appuser/.local/bin:$PATH"

# Health check for the application
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port
EXPOSE 8000

# The command to run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

