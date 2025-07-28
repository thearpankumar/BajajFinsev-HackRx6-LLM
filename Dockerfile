# ==============================================================================
# Builder Stage
# ==============================================================================
# This stage installs all dependencies into a self-contained virtual environment.
FROM python:3.11-slim as builder

# Set the virtual environment path
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv $VENV_PATH

# Copy only the requirements file
COPY requirements.txt .

# Install Python dependencies into the virtual environment
# Using the CPU-only torch index for a smaller image
RUN pip install --no-cache-dir -r requirements.txt

# Download the spaCy model into the virtual environment
RUN python -m spacy download en_core_web_sm

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

# Create a non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set the virtual environment path
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# Copy the virtual environment from the builder stage
COPY --from=builder $VENV_PATH $VENV_PATH

# Copy the application code
COPY . .

# Set ownership and user
RUN chown -R appuser:appgroup /app
USER appuser

# The command is handled in the docker-compose.yml file
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8000"]