# Use the official Python slim image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# --- Production stage ---
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080 \
    GOOGLE_APPLICATION_CREDENTIALS=/app/gcp_credentials.json

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# Copy from builder stage
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser . /app

# Switch to non-root user
USER appuser

# Verify critical files exist
RUN test -f "/app/fibonacci_model.pkl" && \
    test -f "/app/state_scaler.pkl" && \
    test -f "/app/main.py"

# Ensure Python packages are in PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE $PORT

# Start Gunicorn
CMD ["gunicorn", "main:app", \
    "--workers", "4", \
    "--worker-class", "uvicorn.workers.UvicornWorker", \
    "--bind", "0.0.0.0:8080", \
    "--timeout", "120", \
    "--keep-alive", "60", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "--log-level", "info"]