# syntax=docker/dockerfile:1
# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code (excluding chroma_db per .dockerignore)
COPY . .

# Create necessary directories for data persistence
# ChromaDB will create a fresh database on first run
RUN mkdir -p /app/data/materials /app/data/chroma_db && \
    chmod -R 777 /app/data

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8000
ENV DATA_PATH=/app/data/materials
ENV CHROMA_PATH=/app/data/chroma_db
ENV TF_ENABLE_ONEDNN_OPTS=0

# Health check with longer start period for model download (180s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Make entrypoint executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run application with entrypoint
ENTRYPOINT ["/entrypoint.sh"]
