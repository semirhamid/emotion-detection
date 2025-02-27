# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyTorch
RUN apt-get update && apt-get install -y \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Download the model during build
RUN set -e && python download_model.py

EXPOSE 8000

# Command to run the application using uvicorn with api.py
CMD ["uvicorn", "main:app", "--port", "8000"] 