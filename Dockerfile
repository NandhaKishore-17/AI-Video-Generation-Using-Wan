# Use an official PyTorch runtime with CUDA 12.1 and cuDNN 8 as a parent image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend /app/backend

# Create essential directories
RUN mkdir -p /app/models /app/media_output

# Set backend as working directory for running the app
WORKDIR /app/backend

# Make the start script executable
RUN chmod +x start.sh

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the startup script
CMD ["./start.sh"]
