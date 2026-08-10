import logging
from typing import Any, Dict

logger = logging.getLogger("video_engine")


class VideoGenerationEngine:
    """Placeholder interface for WAN integration; video rendering stays disabled until implemented."""

    def __init__(self):
        self.media_dir = None

    async def generate_scene_video(
        self,
        scene_id: str,
        image_relative_url: str,
        motion_prompt: str,
        duration_seconds: float = 5.0,
        fps: int = 24,
    ) -> str:
        logger.info("Video generation is disabled; returning stub path for scene %s", scene_id)
        return f"/media/stub_{scene_id}.mp4"

    async def generate_scene(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "stubbed", "prompt": prompt, "output_path": None}

    async def generate_episode(self, scenes: list[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return {"status": "stubbed", "scenes": len(scenes), "output_path": None}


video_engine = VideoGenerationEngine()

