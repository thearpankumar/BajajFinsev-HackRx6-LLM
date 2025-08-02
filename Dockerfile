# ==============================================================================
# Production Docker Image for Business Document Analysis API
# Specialized for Insurance, Legal, HR, and Compliance domains
# ==============================================================================

# Use multi-stage build for better layer caching and smaller final image
FROM python:3.12-slim AS builder

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

# Create NLTK download script
RUN echo 'import nltk\nimport os\nos.makedirs("/tmp/nltk_data", exist_ok=True)\npackages = ["punkt", "stopwords", "wordnet", "words"]\noptional_packages = ["punkt_tab", "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "maxent_ne_chunker", "maxent_ne_chunker_tab"]\nfor pkg in packages:\n    try:\n        nltk.download(pkg, download_dir="/tmp/nltk_data", quiet=True)\n        print(f"Downloaded {pkg}")\n    except Exception as e:\n        print(f"Failed to download {pkg}: {e}")\nfor pkg in optional_packages:\n    try:\n        nltk.download(pkg, download_dir="/tmp/nltk_data", quiet=True)\n        print(f"Downloaded optional {pkg}")\n    except Exception as e:\n        print(f"Optional package {pkg} not available: {e}")' > download_nltk.py && \
    python download_nltk.py && \
    rm download_nltk.py

# Download SpaCy models
RUN python -m spacy download en_core_web_sm --quiet

# Download sentence-transformers models to avoid download during runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ==============================================================================
# Production stage
# ==============================================================================
FROM python:3.12-slim AS production

# Set runtime environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NLTK_DATA=/app/nltk_data
ENV TORCH_HOME=/tmp/torch_cache
ENV HF_HOME=/tmp/torch_cache

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    mupdf \
    libopenblas0 \
    liblapack3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd --system --gid 1001 appuser && \
    useradd --system --uid 1001 --gid 1001 --home /home/appuser appuser

# Set the working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy NLTK data from builder stage
COPY --from=builder /tmp/nltk_data /app/nltk_data

# Copy SpaCy models from builder stage (find the correct path first)
RUN SPACY_MODEL_PATH=$(find /home/appuser/.local/lib/python3.12/site-packages -name "en_core_web_sm*" -type d 2>/dev/null | head -1) && \
    if [ -n "$SPACY_MODEL_PATH" ]; then \
        echo "SpaCy model found at: $SPACY_MODEL_PATH"; \
    else \
        echo "Warning: SpaCy model not found, will be downloaded at runtime"; \
    fi

# Create application directories with proper permissions
RUN mkdir -p /tmp/embedding_cache /tmp/lancedb /tmp/torch_cache && \
    chown -R appuser:appuser /app /tmp/embedding_cache /tmp/lancedb /tmp/torch_cache /home/appuser

# Copy the application source code
COPY --chown=appuser:appuser src/ ./src/

# Copy environment file (if exists) - use conditional copy
COPY --chown=appuser:appuser .env* ./ 
# Switch to the non-root user
USER appuser

# Update PATH to include user-installed packages
ENV PATH="/home/appuser/.local/bin:$PATH"

# Verify installation and download missing models if needed
RUN echo 'import sys\ntry:\n    import spacy\n    nlp = spacy.load("en_core_web_sm")\n    print("SpaCy model loaded successfully")\nexcept OSError:\n    print("SpaCy model not found, downloading...")\n    import subprocess\n    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--quiet"])\nexcept Exception as e:\n    print(f"Warning: SpaCy setup issue: {e}")' > verify_spacy.py && \
    python verify_spacy.py && \
    rm verify_spacy.py

# Health check for the application
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose the port
EXPOSE 8000

# The command to run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
