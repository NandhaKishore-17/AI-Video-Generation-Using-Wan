#!/bin/bash
set -e

# Model directory setting based on environment variable or default
MODEL_DIR="${WAN_MODEL_PATH:-/app/models/Wan2.2-TI2V-5B-Diffusers}"
MODEL_REPO="Wan-AI/Wan2.2-TI2V-5B-Diffusers"

echo "Checking if model exists at $MODEL_DIR..."

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A $MODEL_DIR)" ]; then
    echo "Model not found. Initiating download from HuggingFace..."
    # Ensure directory exists
    mkdir -p "$MODEL_DIR"
    
    # Download the model
    # If HF_TOKEN is provided in env, it will be used automatically by huggingface-cli
    huggingface-cli download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    echo "Model download complete."
else
    echo "Model already exists at $MODEL_DIR."
fi

echo "Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
