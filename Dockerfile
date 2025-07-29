# ==============================================================================
# Builder Stage
# ==============================================================================
# This stage installs all dependencies and pre-downloads models into a 
# self-contained virtual environment and cache.
FROM python:3.11-slim AS builder

# Set the virtual environment path
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv $VENV_PATH

# Copy only the requirements file to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy model into the virtual environment
RUN python -m spacy download en_core_web_sm

# --- Pre-download HuggingFace models ---
# Set a temporary cache directory for HuggingFace models
ENV HF_HOME=/opt/hf_cache
RUN mkdir -p $HF_HOME

# Run the download script as a single layer
RUN python -c "\
import warnings; \
warnings.filterwarnings('ignore'); \
from sentence_transformers import SentenceTransformer; \
print('Downloading nlpaueb/legal-bert-base-uncased model...'); \
SentenceTransformer('nlpaueb/legal-bert-base-uncased'); \
print('Model downloaded successfully!')"

# ==============================================================================
# Final Stage
# ==============================================================================
# This stage creates the small, final production image.
FROM python:3.11-slim

WORKDIR /app

# Install only the RUNTIME system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /home/appuser appuser

# Set the virtual environment path
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy the virtual environment from the builder stage
COPY --from=builder $VENV_PATH $VENV_PATH

# Copy the pre-downloaded HuggingFace models to the user's default cache location
COPY --from=builder /opt/hf_cache /home/appuser/.cache/huggingface

# Copy the application code
COPY . .

# Set ownership for the app directory and the user's home directory (for the cache)
RUN chown -R appuser:appgroup /app /home/appuser

# Switch to the non-root user
USER appuser

# The command is a fallback, as it's typically overridden by docker-compose.yml
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8000"]
