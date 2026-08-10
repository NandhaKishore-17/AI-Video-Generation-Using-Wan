import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "media_output"
TEMP_DIR = BASE_DIR / "temp"

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Story Universe Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "sqlite:///./universe_platform.db"
    
    # Qdrant Vector Memory
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    
    # Redis & Celery
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Open Source AI Model Endpoints (Supports local Ollama, vLLM, ComfyUI, diffusers, EdgeTTS, Piper)
    LLM_PROVIDER: str = "ollama"  # ollama, qwen, openai-compatible, mock
    LLM_API_BASE: str = "http://localhost:11434"
    LLM_MODEL: str = "llama3.2:3b"

    # Ollama settings
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_TIMEOUT: int = 600  # seconds per request
    
    IMAGE_PROVIDER: str = "flux"  # flux, sdxl, diffusers, mock
    IMAGE_API_BASE: str = "http://localhost:7860"
    
    VIDEO_PROVIDER: str = "wan_local"  # wan_local, cogvideox, wan, mock
    VIDEO_ENGINE: str = "wan"  # mock, wan, wan_local
    VIDEO_API_BASE: str = "http://localhost:7861"
    
    VOICE_PROVIDER: str = "f5-tts"  # f5-tts, edge-tts, elevenlabs, openai-tts, pyttsx3
    VOICE_API_BASE: str = "http://localhost:5000"
    ELEVENLABS_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    MUSIC_PROVIDER: str = "musicgen"  # musicgen, audiocraft, mock
    
    # GPU & Cloud Strategy
    GPU_EXECUTION_MODE: str = "LOCAL_GPU"  # LOCAL_GPU, CLOUD_GPU, HYBRID
    CUDA_DEVICE: str = "cuda:0"
    
    # Local Wan 2.2 TI2V-5B Model Settings
    WAN_MODEL_PATH: str = os.getenv("WAN_MODEL_PATH", str(MODEL_DIR / "Wan2.2-TI2V-5B"))
    WAN_DEVICE: str = "cuda:0"
    WAN_ENABLE_CPU_OFFLOAD: bool = True  # Enable model CPU offloading to save VRAM
    WAN_NUM_FRAMES: int = 81          # ~3.4 seconds at 24fps
    WAN_NUM_INFERENCE_STEPS: int = 30  # Diffusion steps (lower = faster)
    WAN_GUIDANCE_SCALE: float = 5.0    # Classifier-free guidance scale
    WAN_VIDEO_WIDTH: int = 1280         # Default generated video width
    WAN_VIDEO_HEIGHT: int = 720         # Default generated video height
    WAN_FPS: int = 24                  # Default generated video frames per second
    DEFAULT_WIDTH: int = 1280         # Default generated video width
    DEFAULT_HEIGHT: int = 720         # Default generated video height
    DEFAULT_FPS: int = 24             # Default generated video frames per second
    MAX_VIDEO_DURATION: float = 30.0   # Maximum video duration in seconds
    OUTPUT_DIR: str = str(OUTPUT_DIR)
    MEDIA_OUTPUT_DIR: str = str(OUTPUT_DIR)
    TEMP_DIR: str = str(TEMP_DIR)
    
    # HuggingFace Open Source Free Inference API & Cloud GPU Endpoints
    HF_TOKEN: Optional[str] = "0ef8eb48b25946bd76c6c842e2fb3ee2b95586fde7d780b6970a1648e5cfe263"
    VIDEO_API_KEY: Optional[str] = "0ef8eb48b25946bd76c6c842e2fb3ee2b95586fde7d780b6970a1648e5cfe263"
    HF_API_BASE: str = "https://api-inference.huggingface.co/models"
    HF_MODEL_QWEN: str = "Qwen/Qwen2.5-72B-Instruct"
    HF_MODEL_FLUX: str = "black-forest-labs/FLUX.1-schnell"
    HF_MODEL_WAN: str = "Wan-AI/Wan2.2-TI2V-5B:preferred"

    model_config = SettingsConfigDict(
        env_file=[
            str(Path(__file__).resolve().parents[3] / ".env"),
            str(Path(__file__).resolve().parents[2] / ".env"),
            ".env",
        ],
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()

os.makedirs(settings.MEDIA_OUTPUT_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)
