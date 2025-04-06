FROM python:3.10-slim

# Best practice: disable pyc files + enable immediate logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install required system packages (for packages like xgboost, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the app folder into the container
COPY app/ ./app

# Verify model files exist
RUN test -f "./app/models/fibonacci_model.pkl" || (echo "ERROR: Missing fibonacci_model.pkl" && exit 1)
RUN test -f "./app/models/state_scaler.pkl" || (echo "ERROR: Missing state_scaler.pkl" && exit 1)

# Expose the port your app runs on
EXPOSE 8080

# Start with gunicorn and uvicorn workers
CMD ["gunicorn", "app.main:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080"]
