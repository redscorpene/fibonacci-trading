FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GOOGLE_APPLICATION_CREDENTIALS=/fibonacci_trading_ai/gcp_credentials.json

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory to root
WORKDIR /

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code to the root directory
COPY . .

# Verify critical files exist
RUN test -f "fibonacci_model.pkl" || (echo "ERROR: Missing fibonacci_model.pkl" && exit 1) && \
    test -f "state_scaler.pkl" || (echo "ERROR: Missing state_scaler.pkl" && exit 1) && \
    test -f "main.py" || (echo "ERROR: Missing main.py" && exit 1)

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start Gunicorn with Uvicorn workers
CMD ["gunicorn", "main:app", \
    "--workers", "4", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8080", \
    "--timeout", "120", \
    "--keep-alive", "60"]
