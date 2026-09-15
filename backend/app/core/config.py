import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "media_output"
TEMP_DIR = BASE_DIR / "temp"

class Settings(BaseSettings):
    BASE_DIR: str = str(Path(__file__).resolve().parents[2])
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
    LLM_MODEL: str = "R4C3R/qwen3-8b-heretic:q4_k_m"

    # Ollama settings
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "R4C3R/qwen3-8b-heretic:q4_k_m"
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
    GPU_EXECUTION_MODE: str = "LOCAL_GPU"  # LOCAL_GPU, HYBRID
    CUDA_DEVICE: str = "cuda:0"
    
    # Local Wan 2.1 T2V-1.3B Model Settings
    WAN_MODEL_PATH: str = os.getenv("WAN_MODEL_PATH", str(MODEL_DIR / "Wan2.1-T2V-1.3B"))
    WAN_DEVICE: str = "cuda:0"
    WAN_ENABLE_CPU_OFFLOAD: bool = True  # Enable model CPU offloading to save VRAM
    WAN_NUM_FRAMES: int = 81           # Native Wan 2.1 sweet spot (~3.4s at 24fps)
    WAN_NUM_INFERENCE_STEPS: int = 8   # Fast diffusion steps (~2.5x faster than 20)
    WAN_GUIDANCE_SCALE: float = 5.0    # Classifier-free guidance scale
    WAN_VIDEO_WIDTH: int = 480          # Generate at 480 — upscaler brings it to 960
    WAN_VIDEO_HEIGHT: int = 272         # Generate at 272 — upscaler brings it to 544
    WAN_FPS: int = 12                  # 12fps is the 1.3B model sweet spot
    DEFAULT_WIDTH: int = 480           # Generate at low-res; upscaler does 2x
    DEFAULT_HEIGHT: int = 272          # Generate at low-res; upscaler does 2x
    DEFAULT_FPS: int = 12             # Default frames per second
    MAX_VIDEO_DURATION: float = 30.0   # Maximum video duration in seconds

    # SadTalker Lip-Sync Settings
    SADTALKER_CHECKPOINT_DIR: str = "D:/hf_cache/sadtalker"
    SADTALKER_SIZE: int = 256           # Face crop resolution (256 for 4GB VRAM)
    SADTALKER_PREPROCESS: str = "crop"  # crop | resize | full
    SADTALKER_STILL: bool = True        # Reduce head motion for dialogue scenes

    # AnimateDiff-LCM Fast Diffusion Settings
    ANIMATEDIFF_BASE_MODEL: str = "emilianJR/epiCRealism"
    ANIMATEDIFF_LCM_ADAPTER: str = "wangfuyun/AnimateLCM"
    ANIMATEDIFF_NUM_FRAMES: int = 16    # Frames per clip (16 is SD1.5 sweet spot)
    ANIMATEDIFF_NUM_STEPS: int = 4      # LCM trained for 4-step convergence
    ANIMATEDIFF_GUIDANCE_SCALE: float = 2.0  # LCM uses low CFG
    OUTPUT_DIR: str = str(OUTPUT_DIR)
    MEDIA_OUTPUT_DIR: str = str(OUTPUT_DIR)
    TEMP_DIR: str = str(TEMP_DIR)

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

