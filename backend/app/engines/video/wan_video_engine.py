import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.engines.video.video_engine import VideoEngine

logger = logging.getLogger("wan_video_engine")


class WanVideoEngine(VideoEngine):
    """Real video engine backed by the local Wan 2.2 pipeline."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or settings.WAN_MODEL_PATH)
        self.output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        try:
            from app.engines.wan_pipeline import wan_pipeline

            if getattr(wan_pipeline, "model_path", None) != self.model_path:
                wan_pipeline.model_path = Path(self.model_path)
            if not getattr(wan_pipeline, "is_loaded", False):
                wan_pipeline.load_model()

            self._pipeline = wan_pipeline
            self._initialized = True
            logger.info("WanVideoEngine initialized with local Wan pipeline")
        except Exception as exc:
            self._pipeline = None
            self._initialized = False
            logger.warning("WanVideoEngine initialization failed: %s", exc)
            raise

    def render_placeholder(self, job_id: str) -> str:
        return self.render_video(
            scene_prompt=f"placeholder for {job_id}",
            output_path=str(self.output_dir / f"{job_id}.mp4"),
            width=640,
            height=360,
            fps=12,
            duration=2.0,
            seed=42,
        )

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
        output_path = str(Path(output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        return self._render_with_wan(scene_prompt, output_path, width, height, fps, duration, seed)

    def _render_with_wan(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(
                self.generate_real_video(
                    scene_prompt=scene_prompt,
                    output_path=output_path,
                    width=width,
                    height=height,
                    fps=fps,
                    duration=duration,
                    seed=seed,
                )
            )

        result: dict[str, object] = {}
        error: dict[str, BaseException] = {}

        def runner() -> None:
            try:
                result["value"] = asyncio.run(
                    self.generate_real_video(
                        scene_prompt=scene_prompt,
                        output_path=output_path,
                        width=width,
                        height=height,
                        fps=fps,
                        duration=duration,
                        seed=seed,
                    )
                )
            except BaseException as exc:  # pragma: no cover - defensive path
                error["value"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if "value" in result:
            return str(result["value"])
        if "value" in error:
            raise error["value"]
        raise RuntimeError("Wan rendering did not return a result")

    async def render_video_async(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        output_path = str(Path(output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        await self.generate_real_video(
            scene_prompt=scene_prompt,
            output_path=output_path,
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            seed=seed,
        )
        return output_path

    async def generate_real_video(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        await self.initialize()
        if self._pipeline is None:
            raise RuntimeError("Wan pipeline is unavailable")

        temp_image = self.output_dir / "wan_condition.png"
        from PIL import Image

        Image.new("RGB", (max(1, width), max(1, height)), color=(0, 0, 0)).save(temp_image)

        await self._pipeline.generate_video(
            image_path=str(temp_image),
            prompt=scene_prompt,
            output_path=output_path,
            height=height,
            width=width,
            fps=fps,
            seed=seed,
            num_frames=max(1, int(duration * fps)),
        )
        return output_path


_video_engine_instance: Optional[VideoEngine] = None


async def get_video_engine() -> VideoEngine:
    global _video_engine_instance
    if _video_engine_instance is not None:
        return _video_engine_instance

    engine_name = (settings.VIDEO_ENGINE or settings.VIDEO_PROVIDER or "mock").lower()

    if engine_name == "mock":
        from app.engines.video.mock_video_engine import MockVideoEngine

        _video_engine_instance = MockVideoEngine()
        return _video_engine_instance

    if engine_name in {"wan", "wan_local"}:
        engine = WanVideoEngine(model_path=settings.WAN_MODEL_PATH)
        await engine.initialize()
        _video_engine_instance = engine
        return _video_engine_instance

    raise RuntimeError(
        f"Unsupported VIDEO_ENGINE '{engine_name}'. Set VIDEO_ENGINE=wan or wan_local for real Wan2.2 generation."
    )


async def initialize_video_engine() -> VideoEngine:
    return await get_video_engine()
