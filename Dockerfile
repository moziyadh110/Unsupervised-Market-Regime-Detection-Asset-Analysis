# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/raw data/processed models results/plots results/analysis

# Expose Jupyter port
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
