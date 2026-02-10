# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first (smaller than CUDA version)
RUN pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies with increased timeout and retries
RUN pip install --default-timeout=1000 --retries 5 -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BACKEND_HOST=0.0.0.0
ENV BACKEND_PORT=8000

# Run application
CMD ["python", "main.py"]
