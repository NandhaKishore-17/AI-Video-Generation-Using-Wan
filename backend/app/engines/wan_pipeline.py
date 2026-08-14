"""
Wan 2.2 TI2V-5B Local GPU/CPU Pipeline.

Loads and runs the Wan 2.2 Text/Image-to-Video model from local weights or
Hugging Face repository ('Wan-AI/Wan2.2-TI2V-5B-Diffusers').

This pipeline loads once and reuses the same model for subsequent requests.
"""

import asyncio
import gc
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    for image-to-video generation using diffusers.
    """

    def __init__(self):
        self.pipeline = None
        self.model_path = settings.WAN_MODEL_PATH
        self.device = settings.WAN_DEVICE
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.pipeline is not None

    def _resolve_model_id(self) -> str:
        """Resolve model path or HuggingFace repository ID."""
        model_str = str(self.model_path).strip()
        expected_hf_repo = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        
        # Enforce exactly the TI2V-5B model
        if "Wan-AI/" in model_str and model_str != expected_hf_repo:
             logger.error("Invalid model ID: %s. Only %s is supported.", model_str, expected_hf_repo)
             raise WanModelLoadingError(f"Strict enforcement: Expected model {expected_hf_repo}, got {model_str}")
        
        local_path = Path(model_str)
        if local_path.exists():
            logger.info("Using local model directory: %s", local_path.resolve())
            return str(local_path.resolve())
            
        logger.info("Using Hugging Face repository ID: %s", expected_hf_repo)
        return expected_hf_repo

    def load_model(self) -> None:
        logger.info("=" * 80)
        logger.info("STARTING REAL WAN 2.2 PIPELINE INITIALIZATION")
        logger.info("=" * 80)

        # ── Step 1: Loading configuration ───────────────────────────────────
        logger.info("[STEP 1/9] Loading configuration...")
        model_id = self._resolve_model_id()
        logger.info("  - Resolved Model ID/Path: %s", model_id)
        logger.info("  - Configured device: %s", self.device)
        logger.info("  - Enable CPU offload: %s", settings.WAN_ENABLE_CPU_OFFLOAD)

        if self.is_loaded:
            logger.info("Wan 2.2 model already loaded into memory, skipping.")
            return

        # ── Step 2: GPU/CPU detection ───────────────────────────────────────
        logger.info("[STEP 2/9] GPU/CPU detection...")
        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            logger.error("CUDA is not available. GPU is required for real Wan2.2 inference.")
            raise WanModelLoadingError("CUDA is not available for Wan2.2 inference.")
            
        torch_cuda_version = getattr(torch.version, 'cuda', None)
        logger.info("  - PyTorch Version : %s", torch.__version__)
        logger.info("  - CUDA Available  : %s", cuda_available)
        logger.info("  - CUDA Build Ver  : %s", torch_cuda_version or 'N/A')

        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
        logger.info("  - GPU Count: %s, Device 0: %s", device_count, device_name)
        target_device = settings.WAN_DEVICE if "cuda" in settings.WAN_DEVICE else "cuda:0"
        torch_dtype = torch.bfloat16


        self.device = target_device

        try:
            from diffusers import WanImageToVideoPipeline
            from diffusers.schedulers import UniPCMultistepScheduler

            # ── Step 3 to 6: Loading WanImageToVideoPipeline from pretrained ─
            logger.info("[STEP 3-6/9] Loading WanImageToVideoPipeline components from %s...", model_id)
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )

            # ── Step 7: Loading scheduler ───────────────────────────────────
            logger.info("[STEP 7/9] Configuring UniPCMultistepScheduler...")
            try:
                self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                    self.pipeline.scheduler.config,
                    flow_shift=5.0,
                )
            except Exception as sched_err:
                logger.warning("Scheduler custom flow_shift warning: %s. Using default config.", sched_err)
                self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                    self.pipeline.scheduler.config
                )

            # Only enable CPU offload when CUDA is present (requires accelerate + GPU)
            if settings.WAN_ENABLE_CPU_OFFLOAD and cuda_available:
                logger.info("Enabling model CPU offloading for VRAM efficiency...")
                self.pipeline.enable_model_cpu_offload()
            else:
                logger.info("Moving real pipeline to device %s...", self.device)
                self.pipeline = self.pipeline.to(self.device)

            try:
                if hasattr(self.pipeline, "vae") and self.pipeline.vae is not None:
                    self.pipeline.vae.enable_slicing()
                    self.pipeline.vae.enable_tiling()
                    logger.info("VAE slicing + tiling enabled for memory optimization.")
            except Exception as opt_err:
                logger.debug("VAE slicing/tiling optimization notice: %s", opt_err)

            # ── Log every loaded component ────────────────────────────────────
            logger.info("[COMPONENTS] Listing all loaded pipeline components:")
            for comp_name in ["tokenizer", "text_encoder", "transformer", "vae", "scheduler", "image_encoder"]:
                comp = getattr(self.pipeline, comp_name, None)
                if comp is not None:
                    logger.info("  + %-20s: %s", comp_name, type(comp).__name__)
                else:
                    logger.warning("  - %-20s: MISSING / None", comp_name)

            self._is_loaded = True
            logger.info("=" * 80)
            logger.info("REAL WAN 2.2 TI2V-5B PIPELINE LOADED SUCCESSFULLY!")
            logger.info("=" * 80)

        except ImportError as exc:
            tb = traceback.format_exc()
            logger.error("Missing Wan2.2 dependency:\n%s", tb)
            raise WanModelLoadingError(f"Missing dependency for Wan2.2: {exc}\nTraceback:\n{tb}") from exc
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Failed to load Wan2.2 model:\n%s", tb)
            self.pipeline = None
            self._is_loaded = False
            raise WanModelLoadingError(f"Failed to load Wan2.2 model from '{model_id}': {exc}\nTraceback:\n{tb}") from exc

    def unload_model(self) -> None:
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._is_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Wan2.2 model unloaded and memory freed.")

    async def generate_video(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        negative_prompt: str = "Distorted, discontinuous, ugly, blurry, low resolution, motionless, static, disfigured",
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

        num_frames = num_frames or settings.WAN_NUM_FRAMES
        height = height or settings.DEFAULT_HEIGHT
        width = width or settings.DEFAULT_WIDTH
        num_inference_steps = num_inference_steps or settings.WAN_NUM_INFERENCE_STEPS
        guidance_scale = guidance_scale if guidance_scale is not None else settings.WAN_GUIDANCE_SCALE
        fps = fps or settings.DEFAULT_FPS

        if not self.is_loaded:
            logger.info("Model not loaded yet, triggering load_model()...")
            try:
                self.load_model()
            except Exception as exc:
                logger.warning("Wan2.2 model load failed: %s", exc)
                self.pipeline = None
                self._is_loaded = False

        if self.pipeline is None:
            logger.error("Real Wan2.2 pipeline unavailable. Aborting inference.")
            raise WanInferenceError("Pipeline is not loaded. Cannot run GPU inference.")

        # ── Step 8: Real Video generation (inference) ───────────────────────
        logger.info(
            "[STEP 8/9] STARTING REAL WAN 2.2 INFERENCE: prompt='%s', width=%s, height=%s, fps=%s, frames=%s, steps=%s, seed=%s",
            prompt[:120],
            width,
            height,
            fps,
            num_frames,
            num_inference_steps,
            seed,
        )

        loop = asyncio.get_running_loop()

        def _run_real_inference() -> None:
            from diffusers.utils import export_to_video
            from PIL import Image

            if Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
                if image.size != (width, height):
                    image = image.resize((width, height))
                    logger.info("Resized conditioning image to %s", (width, height))
            else:
                logger.info("Creating conditioning image for Wan2.2 at %s", image_path)
                image = Image.new("RGB", (width, height), color=(0, 0, 0))

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if not torch.cuda.is_available() or "cuda" not in self.device:
                raise WanInferenceError(f"Fatal error: Generator requires a CUDA device, but self.device is {self.device}")
            generator_device = self.device
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            logger.info("Generator device: %s", generator_device)

            logger.info("Executing real PyTorch diffusion inference loop...")
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
            # ── Step 9: Saving real output video ────────────────────────────────
            logger.info("[STEP 9/9] Exporting REAL Wan2.2 video to %s...", output_path)
            export_to_video(frames, output_path, fps=fps)
            logger.info("REAL Wan2.2 inference completed successfully. Saved video to %s", output_path)

        try:
            await loop.run_in_executor(None, _run_real_inference)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Real Wan2.2 inference failed with traceback:\n%s", tb)
            raise WanInferenceError(f"Real GPU inference failed: {exc}") from exc

        if not Path(output_path).exists():
            raise WanInferenceError(f"Wan2.2 generation did not produce output at {output_path}")

        return output_path


wan_pipeline = WanLocalPipeline()
