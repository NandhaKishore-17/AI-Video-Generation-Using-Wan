"""
Hybrid Video Engine.

Smart routing engine that picks the best video generator per scene:
- SadTalker for dialogue-heavy scenes (portrait + audio → lip-sync)
- AnimateDiff-LCM for action/establishing shots (text prompt → motion video)

This gives the best of both worlds: natural lip-sync for character dialogue
and real AI motion for cinematic transitions.
"""

import logging
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.engines.video.video_engine import VideoEngine

logger = logging.getLogger("hybrid_engine")


class HybridVideoEngine(VideoEngine):
    """
    Routes video generation to the best engine per scene:
    - Has dialogue audio? → SadTalker (lip-sync)
    - No dialogue? → AnimateDiff-LCM (fast text-to-video)
    """

    def __init__(self):
        self.output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sadtalker = None
        self._animatediff = None

    def _get_sadtalker(self):
        if self._sadtalker is None:
            from app.engines.video.sadtalker_engine import SadTalkerVideoEngine
            self._sadtalker = SadTalkerVideoEngine()
        return self._sadtalker

    def _get_animatediff(self):
        if self._animatediff is None:
            from app.engines.video.animatediff_lcm_engine import AnimateDiffLCMVideoEngine
            self._animatediff = AnimateDiffLCMVideoEngine()
        return self._animatediff

    async def initialize(self) -> None:
        """Initialize both sub-engines lazily."""
        logger.info("HybridVideoEngine initialized (sub-engines load on first use).")

    def render_placeholder(self, job_id: str) -> str:
        return self._get_sadtalker().render_placeholder(job_id)

    def render_video(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        """Sync render — delegates to AnimateDiff-LCM (no audio context available)."""
        return self._get_animatediff().render_video(
            scene_prompt, output_path, width, height, fps, duration, seed
        )

    async def render_video_async(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> str:
        """
        Smart routing:
        - If image_path AND audio_path are provided → SadTalker (lip-sync)
        - Otherwise → AnimateDiff-LCM (text-to-video)
        """
        has_audio = audio_path and Path(audio_path).exists() and Path(audio_path).stat().st_size > 0
        has_image = image_path and Path(image_path).exists() and Path(image_path).stat().st_size > 0

        if has_audio and has_image:
            logger.info(
                "[HYBRID] Scene has image + audio → routing to SadTalker lip-sync engine."
            )
            engine = self._get_sadtalker()
            return await engine.render_video_async(
                scene_prompt=scene_prompt,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                seed=seed,
                image_path=image_path,
                audio_path=audio_path,
            )
        else:
            logger.info(
                "[HYBRID] Scene has no dialogue audio → routing to AnimateDiff-LCM engine."
            )
            engine = self._get_animatediff()
            return await engine.render_video_async(
                scene_prompt=scene_prompt,
                output_path=output_path,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                seed=seed,
            )
