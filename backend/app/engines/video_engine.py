import logging
from typing import Any, Dict

logger = logging.getLogger("video_engine")


class VideoGenerationEngine:
    """Thin wrapper that delegates scene video generation to the local Wan 2.2 pipeline."""

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
        logger.info("Generating video for scene %s via local WAN engine", scene_id)
        import os
        from app.engines.video.wan_video_engine import get_video_engine

        output_filename = f"scene_video_{scene_id}.mp4"
        output_path = os.path.join(self.media_dir, output_filename) if self.media_dir else output_filename

        engine = await get_video_engine()
        result = await engine.render_video_async(
            scene_prompt=motion_prompt,
            output_path=output_path,
            width=640,
            height=360,
            fps=fps,
            duration=duration_seconds,
            seed=42,
        )
        return result

    async def generate_scene(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "wan_local", "prompt": prompt, "output_path": None}

    async def generate_episode(self, scenes: list[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return {"status": "wan_local", "scenes": len(scenes), "output_path": None}


video_engine = VideoGenerationEngine()
