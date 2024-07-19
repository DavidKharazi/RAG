# Use the official Python base image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
       libpq-dev \
       python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Установите chromadb
RUN pip install chromadb

# Copy the rest of the application code
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8222

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8222"]
