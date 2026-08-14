import uvicorn
import logging
import os
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
media_output_dir = settings.MEDIA_OUTPUT_DIR
os.makedirs(media_output_dir, exist_ok=True)
app.mount("/media", StaticFiles(directory=media_output_dir), name="media")

# Create DB tables on startup
@app.on_event("startup")
async def on_startup():
    Base.metadata.create_all(bind=engine)
    # Start background job worker
    import asyncio
    asyncio.create_task(job_manager.start_worker())

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
