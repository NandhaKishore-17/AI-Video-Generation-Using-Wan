import logging
from pathlib import Path
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
        duration: float,
        width: int,
        height: int,
        fps: int,
    ) -> str:
        """
        Generate a video using the configured video engine.

        Returns the relative media path of the generated MP4.
        """
        filename = f"wan2_2_{uuid4().hex}.mp4"
        output_path = str(self.output_dir / filename)

        engine = await get_video_engine()
        logger.info(
            "Using video engine %s to generate Wan2.2 clip: %s",
            type(engine).__name__,
            output_path,
        )

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
        logger.info("Video generation complete. Path=%s", relative_path)
        return relative_path


video_service = VideoGenerationService()
