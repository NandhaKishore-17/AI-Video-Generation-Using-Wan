import logging
import os
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.core.config import settings

logger = logging.getLogger("video_service")


class VideoGenerationService:
    """Generate video using the local Wan 2.2 pipeline."""

    def __init__(self):
        self.output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_video(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration: float = 2.0,
        width: int = 640,
        height: int = 360,
        fps: int = 24,
        output_path: Optional[str] = None,
        aspect_ratio: str = "16:9",
    ) -> str:
        """Generate video using the local Wan 2.2 engine.

        Returns the relative media URL for the generated video (e.g. /media/xxx.mp4).
        """
        if not output_path:
            filename = f"scene_{uuid4().hex}.mp4"
            output_path = str(self.output_dir / filename)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 50)
        logger.info("STARTING LOCAL WAN 2.2 VIDEO GENERATION")
        logger.info("Prompt: %s", prompt[:500])
        logger.info("Output: %s", output_path)
        logger.info("Resolution: %dx%d  fps=%d  duration=%.1fs", width, height, fps, duration)
        logger.info("=" * 50)

        from app.engines.video.wan_video_engine import get_video_engine

        try:
            engine = await get_video_engine()
            await engine.render_video_async(
                scene_prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                seed=42,
            )

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise RuntimeError(f"Video generation produced no output at {output_path}")

            logger.info("Video saved successfully: %s", output_path)
            return f"/media/{Path(output_path).name}"

        except Exception:
            logger.exception("Local WAN video generation failed")
            raise


video_service = VideoGenerationService()
