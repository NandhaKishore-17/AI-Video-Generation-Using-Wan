import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from uuid import uuid4

from app.schemas.schemas import VideoGenerateRequest
from app.services.video_service import video_service
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger("video_generation")

video_tasks: Dict[str, Any] = {}

@router.get("/video/health", status_code=200)
async def health_check() -> Dict[str, Any]:
    """Health check for the remote video generation provider."""
    return {
        "provider": "wan_remote",
        "configured": bool(settings.WAN_API_URL and settings.WAN_API_KEY),
        "model": getattr(settings, "WAN_MODEL", "Wan-AI/Wan2.2-TI2V-5B")
    }

async def _background_generate(task_id: str, request: VideoGenerateRequest):
    try:
        video_tasks[task_id] = {"status": "processing"}
        video_path = await video_service.generate_video(
            prompt=request.prompt,
            duration=request.duration or 2.0,
            width=request.width or 640,
            height=request.height or 360,
            fps=request.fps or 12
        )
        video_tasks[task_id] = {
            "status": "completed",
            "video_url": video_path
        }
    except Exception as e:
        logger.exception("Failed to generate video in background")
        video_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
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
