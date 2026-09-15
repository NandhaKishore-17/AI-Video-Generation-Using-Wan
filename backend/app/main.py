import uvicorn
import logging
import os
import sys

# Workaround for torchcodec FFmpeg DLL loading on Windows (Python 3.8+)
if sys.platform == "win32":
    try:
        os.add_dll_directory(r"C:\Users\Nandha kishore\AppData\Local\Microsoft\WinGet\Packages\BtbN.FFmpeg.LGPL.Shared.8.0_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-n8.0.1-66-g27b8d1a017-win64-lgpl-shared-8.0\bin")
    except Exception:
        pass
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.core.database import engine, Base
from app.api.generation import router as generation_router
from app.api.video_generation import router as video_generation_router
from app.api.endpoints import router as endpoints_router
from app.engines.job_manager import job_manager
# Import all models so Base.metadata is fully populated before create_all()
from app.models import job as _job_model  # noqa: F401
from app.models import story as _story_model  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Initialize FastAPI app
app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

@app.get("/health", summary="Health check endpoint for load balancers")
async def health_check():
    return {"status": "ok"}

# Include all API routers
app.include_router(generation_router, prefix=settings.API_V1_STR)
app.include_router(video_generation_router, prefix=settings.API_V1_STR)
app.include_router(endpoints_router, prefix=settings.API_V1_STR)  # universes, episodes, analytics, etc.

# Serve generated media files (videos, images, audio) at /media
from app.core.config import settings
media_root_dir = settings.MEDIA_OUTPUT_DIR
os.makedirs(media_root_dir, exist_ok=True)
app.mount("/media", StaticFiles(directory=media_root_dir), name="media")

# Create DB tables on startup
@app.on_event("startup")
async def on_startup():
    Base.metadata.create_all(bind=engine)
    # Start background job worker
    import asyncio
    asyncio.create_task(job_manager.start_worker())
    # Initialize local WAN video engine
    from app.engines.video.wan_video_engine import initialize_video_engine
    try:
        video_eng = await initialize_video_engine()
        logging.getLogger("main").info("Video engine initialized: %s", type(video_eng).__name__)
    except Exception as exc:
        logging.getLogger("main").warning("Video engine initialization skipped: %s", exc)

from pydantic import BaseModel
from services.tts.tts_service import tts_service

class DebugTTSRequest(BaseModel):
    speaker: str
    text: str

@app.post("/api/debug/tts")
async def debug_tts(req: DebugTTSRequest):
    from uuid import uuid4
    from app.core.config import settings
    from pathlib import Path
    temp_dir = Path(settings.TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_name = f"debug_tts_{uuid4().hex}.wav"
    output_filepath = os.path.join(str(temp_dir), wav_name)
    
    info = await tts_service.generate_single_dialogue(
        character_name=req.speaker,
        dialogue_text=req.text,
        output_filepath=output_filepath
    )
    
    return {
        "wav_path": info["filepath"],
        "duration": info["duration"]
    }

@app.get("/api/debug/story")
async def debug_story(
    genre: str = "Fantasy",
    theme: str = "Chota Bheem Fantasy Adventure",
    episode: int = 1,
    duration: int = 30
):
    """
    Debug endpoint to verify Ollama story generation is working.
    Returns: active universe, episode number, generated story, dialogue, and Ollama model used.
    """
    from app.engines.story_engine import story_engine
    from app.engines.llm.ollama_client import ollama_client

    result = {
        "ollama_url": ollama_client.base_url,
        "ollama_model": ollama_client.model,
        "llm_provider": settings.LLM_PROVIDER,
        "request": {
            "genre": genre,
            "theme": theme,
            "episode": episode,
            "duration": duration
        }
    }

    try:
        story = story_engine.generate_story(
            genre=genre,
            theme=theme,
            duration=duration,
            language="English",
            episode_number=episode
        )
        result["status"] = "SUCCESS"
        result["story"] = story
        result["characters"] = [c.get("name") for c in story.get("characters", [])]
        result["scene_count"] = len(story.get("scenes", []))
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        result["error_type"] = type(e).__name__

    return result

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
