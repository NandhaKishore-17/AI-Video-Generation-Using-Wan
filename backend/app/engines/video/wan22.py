import logging
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

        wan_pipeline.model_path = self.model_path
        wan_pipeline.load_model()

        if not wan_pipeline.is_loaded:
            raise RuntimeError(
                f"Wan2.2 model failed to load from {self.model_path}. "
                "Verify the model directory and GPU availability."
            )

        self._initialized = True
        logger.info("Wan2.2 model loaded from %s", self.model_path)

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
        self.initialize()

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        condition_path = Path(image_path)
        if not condition_path.exists():
            from PIL import Image

            Image.new("RGB", (max(1, width), max(1, height)), color=(0, 0, 0)).save(condition_path)

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
