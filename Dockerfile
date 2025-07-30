# ==============================================================================
# Final Stage
# ==============================================================================
# This stage creates the small, final production image.
FROM python:3.11-slim

WORKDIR /app

# Set the virtual environment path
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# Create a non-root user for security
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /home/appuser appuser

# Create and activate a virtual environment
RUN python -m venv $VENV_PATH
RUN . "$VENV_PATH/bin/activate"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set ownership for the app directory
RUN chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# The command is a fallback, as it's typically overridden by docker-compose.yml
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]