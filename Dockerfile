FROM python:3.9-slim-buster

# Create a non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

COPY requirements.txt .

# Add the user's local bin directory to PATH before installing dependencies
ENV PATH="/home/appuser/.local/bin:$PATH"

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH="/app/src"

# Install dependencies as the non-root user
USER appuser
RUN pip install --no-cache-dir -r requirements.txt

COPY src/. src/

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "80"]
