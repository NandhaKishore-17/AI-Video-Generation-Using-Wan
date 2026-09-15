import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("video_engine")


class VideoGenerationEngine:
    """Thin wrapper that delegates scene video generation to the configured video engine."""

    def __init__(self):
        from app.core.config import settings
        self.media_dir = settings.MEDIA_OUTPUT_DIR

    async def generate_scene_video(
        self,
        scene_id: str,
        image_relative_url: str,
        motion_prompt: str,
        duration_seconds: float = 5.0,
        fps: int = 24,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> str:
        logger.info("Generating video for scene %s via configured video engine", scene_id)
        import os
        from app.engines.video.wan_video_engine import get_video_engine

        output_filename = f"scene_video_{scene_id}.mp4"
        output_path = os.path.join(self.media_dir, output_filename) if self.media_dir else output_filename

        # Resolve image path from relative URL if not explicitly provided
        if not image_path and image_relative_url:
            img_filename = os.path.basename(image_relative_url)
            candidate = os.path.join(self.media_dir, img_filename)
            if os.path.exists(candidate):
                image_path = candidate

        engine = await get_video_engine()

        # Try calling with image_path and audio_path (SadTalker/Hybrid support)
        try:
            result = await engine.render_video_async(
                scene_prompt=motion_prompt,
                output_path=output_path,
                width=512,
                height=288,
                fps=fps,
                duration=duration_seconds,
                seed=42,
                image_path=image_path,
                audio_path=audio_path,
            )
        except TypeError:
            # Fallback for engines that don't accept image_path/audio_path (e.g., Wan, Mock)
            result = await engine.render_video_async(
                scene_prompt=motion_prompt,
                output_path=output_path,
                width=512,
                height=288,
                fps=fps,
                duration=duration_seconds,
                seed=42,
            )
        return result

    async def generate_scene(self, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "configured_engine", "prompt": prompt, "output_path": None}

    async def generate_episode(self, scenes: list[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        return {"status": "configured_engine", "scenes": len(scenes), "output_path": None}


video_engine = VideoGenerationEngine()

