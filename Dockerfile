# GPU-enabled Python image for ML workloads
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY tests/ ./tests/
COPY resources/ ./resources/

# Create cache directories for models
RUN mkdir -p /app/hf_cache

# Set environment variables
ENV PYTHONPATH=/app
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache
ENV HF_DATASETS_OFFLINE=0
ENV TRANSFORMERS_OFFLINE=0

# Default command runs the Stage 1 tests
CMD ["python", "tests/test_stage1.py"]
