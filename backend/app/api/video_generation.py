import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from uuid import uuid4

from app.schemas.schemas import VideoGenerateRequest
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger("video_generation")

video_tasks: Dict[str, Any] = {}

@router.get("/video/health", status_code=200)
async def health_check() -> Dict[str, Any]:
    """Health check for the local WAN video generation engine."""
    return {
        "provider": "wan_local",
        "engine": settings.VIDEO_ENGINE,
        "model_path": settings.WAN_MODEL_PATH,
        "device": settings.WAN_DEVICE,
    }

async def _background_generate(task_id: str, request: VideoGenerateRequest):
    try:
        video_tasks[task_id] = {"status": "processing"}
        from app.engines.video.wan_video_engine import get_video_engine
        import os

        engine = await get_video_engine()
        output_dir = settings.MEDIA_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{task_id}.mp4")

        await engine.render_video_async(
            scene_prompt=request.prompt,
            output_path=output_path,
            width=request.width or 640,
            height=request.height or 360,
            fps=request.fps or 12,
            duration=request.duration or 2.0,
            seed=42,
        )
        video_tasks[task_id] = {
            "status": "completed",
            "video_url": f"/media/{task_id}.mp4",
        }
    except Exception as e:
        logger.exception("Failed to generate video in background")
        video_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
        }

@router.post("/video/generate", status_code=202)
@router.post("/generate-video", status_code=202, deprecated=True)
async def generate_video(request: VideoGenerateRequest, background_tasks: BackgroundTasks):
    """
    Start video generation asynchronously and return a task_id.
    """
    task_id = uuid4().hex
    background_tasks.add_task(_background_generate, task_id, request)
    return {
        "success": True,
        "task_id": task_id,
        "status": "processing"
    }

@router.get("/video/status/{task_id}")
async def get_video_status(task_id: str):
    """
    Check the status of an asynchronous video generation task.
    """
    if task_id not in video_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = video_tasks[task_id]
    response = {
        "task_id": task_id,
        "status": task_data["status"]
    }
    if task_data["status"] == "completed":
        response["video_url"] = task_data["video_url"]
    elif task_data["status"] == "failed":
        response["error"] = task_data.get("error", "Unknown error")
        
    return response
