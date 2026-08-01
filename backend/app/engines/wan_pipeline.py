"""
Wan 2.2 TI2V-5B Local GPU Pipeline.

Loads and runs the Wan 2.2 Text/Image-to-Video model from local weights
at the path specified by settings.WAN_MODEL_PATH.

This pipeline is designed to load once and reuse the same model for
subsequent video requests.
"""

import asyncio
import gc
import logging
from pathlib import Path
from typing import Optional

import torch

from app.core.config import settings

logger = logging.getLogger("wan_pipeline")


class WanPipelineError(Exception):
    """Base exception for Wan pipeline failures."""


class WanModelLoadingError(WanPipelineError):
    """Raised when Wan model loading fails."""


class WanInferenceError(WanPipelineError):
    """Raised when Wan inference fails."""


class WanLocalPipeline:
    """
    Manages loading, running, and unloading the Wan 2.2 TI2V-5B model
    for local GPU-based image-to-video generation.
    """

    def __init__(self):
        self.pipeline = None
        self.model_path = Path(settings.WAN_MODEL_PATH)
        self.device = settings.WAN_DEVICE
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.pipeline is not None

    def load_model(self) -> None:
        if self.is_loaded:
            logger.info("Wan 2.2 model already loaded, skipping.")
            return

        self.model_path = Path(self.model_path)
        if not self.model_path.exists():
            raise WanModelLoadingError(
                f"Wan model directory not found: {self.model_path}"
            )

        if "cuda" not in self.device:
            raise WanModelLoadingError(
                "Wan 2.2 TI2V-5B requires a CUDA device. "
                "Set WAN_DEVICE to a valid CUDA device like cuda:0."
            )

        if not torch.cuda.is_available():
            raise WanModelLoadingError(
                "CUDA is not available. Ensure CUDA drivers and a compatible GPU are installed."
            )

        logger.info("Loading Wan 2.2 TI2V-5B from %s", self.model_path)
        logger.info("Target device: %s", self.device)

        try:
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from diffusers.schedulers import UniPCMultistepScheduler
            from transformers import CLIPVisionModel, AutoTokenizer, UMT5EncoderModel

            tokenizer_path = self.model_path / "google" / "umt5-xxl"
            if not tokenizer_path.exists():
                raise WanModelLoadingError(
                    f"Tokenizer path not found: {tokenizer_path}"
                )

            logger.info("Loading UMT5-XXL tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            logger.info("Loading UMT5-XXL text encoder...")
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.model_path,
                subfolder=None,
                torch_dtype=torch.bfloat16,
            )

            logger.info("Loading Wan 2.2 VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                self.model_path,
                subfolder=None,
                torch_dtype=torch.float32,
            )

            logger.info("Loading CLIP Vision image encoder...")
            image_encoder = CLIPVisionModel.from_pretrained(
                self.model_path,
                subfolder=None,
                torch_dtype=torch.float32,
            )

            logger.info("Assembling WanImageToVideoPipeline...")
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                self.model_path,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                image_encoder=image_encoder,
                torch_dtype=torch.bfloat16,
            )

            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                flow_shift=5.0,
            )

            if settings.WAN_ENABLE_CPU_OFFLOAD:
                logger.info("Enabling model CPU offloading for VRAM efficiency...")
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)

            try:
                self.pipeline.vae.enable_slicing()
                self.pipeline.vae.enable_tiling()
                logger.info("VAE slicing + tiling enabled.")
            except AttributeError:
                logger.debug("VAE slicing/tiling not supported; continuing without optimizations.")

            self._is_loaded = True
            logger.info("Wan 2.2 TI2V-5B pipeline loaded successfully.")

        except ImportError as exc:
            logger.error("Missing Wan2.2 dependency: %s", exc)
            raise WanModelLoadingError(
                "Missing dependency for Wan2.2. Install diffusers, transformers, accelerate, safetensors, sentencepiece."
            ) from exc
        except Exception as exc:
            logger.error("Failed to load Wan2.2 model: %s", exc, exc_info=True)
            self.pipeline = None
            self._is_loaded = False
            raise WanModelLoadingError(f"Failed to load Wan2.2 model: {exc}") from exc

    def unload_model(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._is_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Wan2.2 model unloaded and GPU memory freed.")

    async def generate_video(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        negative_prompt: str = "Distorted, discontinuous, ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, missing arms, missing legs, extra fingers",
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        fps: Optional[int] = None,
        seed: int = 42,
    ) -> str:
        if not prompt or not prompt.strip():
            raise WanInferenceError("Prompt must be a non-empty string.")

        if not self.is_loaded:
            self.load_model()

        if self.pipeline is None:
            raise WanInferenceError("Wan2.2 pipeline is unavailable after initialization.")

        num_frames = num_frames or settings.WAN_NUM_FRAMES
        height = height or settings.DEFAULT_HEIGHT
        width = width or settings.DEFAULT_WIDTH
        num_inference_steps = num_inference_steps or settings.WAN_NUM_INFERENCE_STEPS
        guidance_scale = guidance_scale if guidance_scale is not None else settings.WAN_GUIDANCE_SCALE
        fps = fps or settings.DEFAULT_FPS

        logger.info(
            "Starting Wan2.2 inference: prompt=%s, width=%s, height=%s, fps=%s, frames=%s, steps=%s, seed=%s",
            prompt[:120],
            width,
            height,
            fps,
            num_frames,
            num_inference_steps,
            seed,
        )

        loop = asyncio.get_running_loop()

        def _run_inference() -> None:
            from diffusers.utils import export_to_video
            from PIL import Image

            if Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
                if image.size != (width, height):
                    image = image.resize((width, height))
                    logger.info("Resized conditioning image to %s", (width, height))
            else:
                logger.info("Creating blank conditioning image at %s", image_path)
                image = Image.new("RGB", (width, height), color=(0, 0, 0))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            generator_device = self.device if self.device != "cpu" else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(seed)

            with torch.inference_mode():
                output = self.pipeline(
                    image=image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    generator=generator,
                )

            frames = output.frames[0]
            export_to_video(frames, output_path, fps=fps)
            logger.info("Wan2.2 inference finished, exported video to %s", output_path)

        try:
            await loop.run_in_executor(None, _run_inference)
        except Exception as exc:
            logger.error("Wan2.2 inference failed: %s", exc, exc_info=True)
            raise WanInferenceError(str(exc)) from exc

        if not Path(output_path).exists():
            raise WanInferenceError(f"Wan2.2 generation did not produce output at {output_path}")

        return output_path


wan_pipeline = WanLocalPipeline()
