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

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application source code
COPY --chown=appuser:appgroup src/ .

# Switch to the non-root user
USER appuser

# The command is a fallback, as it's typically overridden by docker-compose.yml
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

