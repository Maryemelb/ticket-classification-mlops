# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable stdout logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CHROMA_PATH=/app/vector_db/chroma_storage/embedding_db

# Install system dependencies for some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project files
COPY ./src ./src

# Ensure vector_db folder exists for Chroma DB
RUN mkdir -p /app/vector_db/chroma_storage

# Default command to run training
CMD ["python3", "src/models/train.py"]
