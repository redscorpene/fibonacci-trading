FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p logs data app/models

# Expose the port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_KEY=dev-key-for-testing

# Copy your service account key
#COPY key.json /app/key.json
#ENV GOOGLE_APPLICATION_CREDENTIALS=/app/key.json


# Command to run the service
CMD ["gunicorn", "app.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]