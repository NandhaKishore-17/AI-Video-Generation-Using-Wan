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
        logger.info("Triggering remote WAN video generation for scene %s", scene_id)
        from app.services.video_service import video_service
        import os
        
        output_filename = f"scene_video_{scene_id}.mp4"
        output_path = os.path.join(self.media_dir, output_filename) if self.media_dir else output_filename
        
        # We pass image_url as None since local relative URLs cannot be downloaded by remote cloud GPUs
        # If the backend is exposed publicly, a fully qualified URL could be constructed here.
        video_url = await video_service.generate_video(
            prompt=motion_prompt,
            image_url=None,
            duration=duration_seconds,
            fps=fps,
            output_path=output_path
        )
        return video_url

    async def generate_scene(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "wan", "prompt": prompt, "output_path": None}

    async def generate_episode(self, scenes: list[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return {"status": "wan", "scenes": len(scenes), "output_path": None}


video_engine = VideoGenerationEngine()

