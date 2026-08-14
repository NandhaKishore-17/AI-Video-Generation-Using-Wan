import os
import logging
from app.core.config import settings
from app.engines.video.wan_video_engine import get_video_engine

logger = logging.getLogger("video_engine")


class VideoGenerationEngine:
    """Compatibility wrapper that routes scene video generation through the selected engine."""

    def __init__(self):
        self.media_dir = settings.MEDIA_OUTPUT_DIR

    async def generate_scene_video(
        self,
        scene_id: str,
        image_relative_url: str,
        motion_prompt: str,
        duration_seconds: float = 5.0,
        fps: int = 24,
    ) -> str:
        filename = f"video_{scene_id}.mp4"
        filepath = os.path.join(self.media_dir, filename)

        import time
        import random
        seed = random.randint(0, 999999)
        engine = await get_video_engine()
        output_path = await engine.render_video_async(
            scene_prompt=motion_prompt,
            output_path=filepath,
            width=settings.WAN_VIDEO_WIDTH,
            height=settings.WAN_VIDEO_HEIGHT,
            fps=fps,
            duration=duration_seconds,
            seed=seed,
        )
        logger.info("Generated video via %s: %s", type(engine).__name__, output_path)
        timestamp = int(time.time())
        return f"/media/{filename}?t={timestamp}"


video_engine = VideoGenerationEngine()

