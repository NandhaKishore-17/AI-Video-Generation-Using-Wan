import logging
import traceback
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.engines.wan_pipeline import wan_pipeline

logger = logging.getLogger("wan22")


class Wan22Wrapper:
    """
    Lightweight Wan 2.2 TI2V-5B wrapper that loads the model once and reuses it.
    """

    _instance: Optional["Wan22Wrapper"] = None

    def __init__(self):
        self.model_path = Path(settings.WAN_MODEL_PATH)
        self._initialized = False

    @classmethod
    def get_instance(cls) -> "Wan22Wrapper":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return

        logger.info("Initializing Wan22Wrapper with model_path=%s", self.model_path)
        try:
            if not self.model_path.exists() and "Wan-AI/" not in str(self.model_path):
                raise FileNotFoundError(f"Model path '{self.model_path}' does not exist.")
            wan_pipeline.model_path = self.model_path
            self._initialized = True
            logger.info("Wan22Wrapper initialized; model loading will be deferred to the generation step.")
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Wan22Wrapper initialization caught error:\n%s", tb)
            self._initialized = False
            raise

    async def generate_video(
        self,
        prompt: str,
        image_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        logger.info("Wan22Wrapper.generate_video called for prompt='%s', output='%s'", prompt[:60], output_path)
        self.initialize()

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        condition_path = Path(image_path)
        if not condition_path.exists():
            from PIL import Image

            Image.new("RGB", (max(1, width), max(1, height)), color=(0, 0, 0)).save(condition_path)

        try:
            return await wan_pipeline.generate_video(
                image_path=str(condition_path),
                prompt=prompt,
                output_path=output_path,
                height=height,
                width=width,
                fps=fps,
                seed=seed,
                num_frames=max(1, int(duration * fps)),
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Wan22Wrapper.generate_video failed:\n%s", tb)
            raise
