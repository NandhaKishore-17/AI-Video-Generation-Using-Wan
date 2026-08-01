import logging
from fastapi import APIRouter, HTTPException

from app.schemas.schemas import VideoGenerateRequest, VideoGenerateResponse
from app.services.video_service import video_service

router = APIRouter()
logger = logging.getLogger("video_generation")


@router.post("/generate-video", response_model=VideoGenerateResponse, status_code=201)
async def generate_video(request: VideoGenerateRequest):
    """
    Generate a Wan2.2 TI2V-5B video clip and return a media_output-relative path.
    """
    try:
        video_path = await video_service.generate_video(
            prompt=request.prompt,
            duration=request.duration,
            width=request.width,
            height=request.height,
            fps=request.fps,
        )

        return VideoGenerateResponse(status="success", video_path=video_path)
    except Exception:
        logger.exception("Failed to generate Wan2.2 video")
        raise HTTPException(
            status_code=500,
            detail="Video generation failed. Check server logs for details."
        )
