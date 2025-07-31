# ==============================================================================
# Final Stage
# ==============================================================================
# This stage creates the small, final production image.
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Create a non-root user for security
RUN addgroup --system appuser &&     adduser --system --ingroup appuser --home /home/appuser appuser

# Copy requirements first to leverage Docker layer caching
COPY --chown=appuser:appgroup requirements.txt .

# Install system dependencies required for PyMuPDF and then Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    swig && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove build-essential swig && \
    rm -rf /var/lib/apt/lists/*

# Copy the application source code into a src directory
COPY --chown=appuser:appgroup src/ ./src

# Switch to the non-root user
USER appuser

# The command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

