import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.core.config import settings
from app.engines.video_engine import video_engine

logger = logging.getLogger("video_service")


class VideoGenerationService:
    def __init__(self):
        self.output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_video(
        self,
        prompt: str,
        duration: float = 2.0,
        width: int = 640,
        height: int = 360,
        fps: int = 12,
        output_path: Optional[str] = None,
    ) -> str:
        """Return a placeholder path while WAN video generation remains disabled."""
        if not output_path:
            filename = f"stub_{uuid4().hex}.mp4"
            output_path = str(self.output_dir / filename)

        logger.info(
            "VideoService.generate_video is stubbed: prompt='%s', output_path=%s, resolution=%sx%s, fps=%s, duration=%ss",
            prompt[:80],
            output_path,
            width,
            height,
            fps,
            duration,
        )
        await video_engine.generate_scene(prompt=prompt)
        return f"media_output/{Path(output_path).name}"


video_service = VideoGenerationService()
