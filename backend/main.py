import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.core.config import settings
from app.core.database import engine, Base
from app.api.endpoints import router as api_router

import logging
from contextlib import asynccontextmanager
from app.core.database import SessionLocal
from app.models.domain import SchedulerConfig
from app.modules.scheduler import scheduler_module

logger = logging.getLogger("main")

# Create database tables
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()
    try:
        config = db.query(SchedulerConfig).filter(SchedulerConfig.id == "default").first()
        if config and config.is_running:
            logger.info("Restoring Autonomous Scheduler background loop on server startup...")
            await scheduler_module.start(interval_minutes=config.interval_minutes, auto_publish=config.auto_publish)
    except Exception as e:
        logger.error(f"Failed to restore scheduler on startup: {e}")
    finally:
        db.close()
    
    yield
    
    if scheduler_module.is_running:
        await scheduler_module.stop()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous AI Story Universe Platform Backend powered by Qwen, FLUX, CogVideoX, Piper, Kokoro, Whisper & MusicGen.",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static media output directory for serving generated images, audio, video clips, and final rendered MP4s
os.makedirs(settings.MEDIA_OUTPUT_DIR, exist_ok=True)
app.mount("/media", StaticFiles(directory=settings.MEDIA_OUTPUT_DIR), name="media")

# Register API routes
app.include_prefix = settings.API_V1_STR
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(api_router)  # Also expose directly at root for convenience

@app.get("/")
def root():
    return {
        "platform": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "ONLINE",
        "docs_url": "/docs",
        "supported_models": {
            "story": "Qwen 2.5 14B / 72B",
            "image": "FLUX.1 / SDXL",
            "video": "CogVideoX / Wan",
            "voice": "Piper / Kokoro TTS",
            "speech_rec": "Whisper",
            "music": "MusicGen",
            "vector_memory": "Qdrant"
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
