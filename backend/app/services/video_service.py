import logging
import traceback
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.core.config import settings
from app.engines.video.wan_video_engine import get_video_engine

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
        """
        Generate a video using the configured video engine.

        Returns the relative media path of the generated MP4.
        """
        if not output_path:
            filename = f"wan2_2_{uuid4().hex}.mp4"
            output_path = str(self.output_dir / filename)

        logger.info(
            "VideoService.generate_video starting: prompt='%s', output_path=%s, resolution=%sx%s, fps=%s, duration=%ss",
            prompt[:80],
            output_path,
            width,
            height,
            fps,
            duration,
        )

        try:
            engine = await get_video_engine()
            logger.info("VideoService: Acquired video engine '%s'", type(engine).__name__)

            video_path = await engine.render_video_async(
                scene_prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                seed=42,
            )

            relative_path = f"media_output/{Path(video_path).name}"
            logger.info("VideoService: Video generation successfully completed. relative_path=%s", relative_path)
            return relative_path

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("VideoService: Video generation failed with traceback:\n%s", tb)
            raise


video_service = VideoGenerationService()
