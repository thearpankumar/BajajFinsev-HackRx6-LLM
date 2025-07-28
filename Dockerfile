# ---- Builder Stage ----
# This stage installs pip-tools and compiles the requirements
FROM python:3.11-slim as builder

WORKDIR /app

# Install pip-tools
RUN pip install pip-tools

# Copy the requirements file and compile it
COPY requirements.in .
RUN pip-compile requirements.in --output-file=requirements.txt

# ---- Final Stage ----
# This stage builds the final application image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy the compiled requirements.txt from the builder stage
COPY --from=builder /app/requirements.txt .

# Install the pinned application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set ownership and user
RUN chown -R appuser:appgroup /app
USER appuser

# Run database migrations and then start the application
CMD ["sh", "-c", "alembic upgrade head && uvicorn src.main:app --host 0.0.0.0 --port 8000"]
