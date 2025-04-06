FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (including models)
COPY . .

# Verify model files exist in correct location
RUN if [ ! -f "/app/app/models/fibonacci_model.pkl" ]; then \
        echo "ERROR: Missing model file at /app/app/models/fibonacci_model.pkl"; \
        exit 1; \
    fi && \
    if [ ! -f "/app/app/models/state_scaler.pkl" ]; then \
        echo "ERROR: Missing scaler file at /app/app/models/state_scaler.pkl"; \
        exit 1; \
    fi

# Keep original port and command
EXPOSE 8080
CMD ["gunicorn", "app.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]