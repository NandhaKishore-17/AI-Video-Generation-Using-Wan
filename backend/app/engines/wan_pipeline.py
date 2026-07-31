"""
Wan 2.2 TI2V-5B Local GPU Pipeline.

Loads and runs the Wan 2.2 Text/Image-to-Video model from local weights
at the path specified by settings.WAN_MODEL_PATH (default: D:/Wan2.2-TI2V-5B).

The model uses the raw (non-Diffusers) weight format:
  - Transformer: sharded safetensors (diffusion_pytorch_model-*.safetensors)
  - VAE: Wan2.2_VAE.pth
  - Text Encoder: models_t5_umt5-xxl-enc-bf16.pth
  - Tokenizer: google/umt5-xxl/

Pipeline assembly follows the diffusers WanImageToVideoPipeline pattern with
manual component loading to support the raw weight format.
"""

import asyncio
import gc
import logging
import os
from typing import Optional

import torch

from app.core.config import settings

logger = logging.getLogger("wan_pipeline")


class WanLocalPipeline:
    """
    Manages loading, running, and unloading the Wan 2.2 TI2V-5B model
    for local GPU-based image-to-video generation.
    """

    def __init__(self):
        self.pipeline = None
        self.model_path: str = settings.WAN_MODEL_PATH
        self.device: str = settings.CUDA_DEVICE if torch.cuda.is_available() else "cpu"
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.pipeline is not None

    def load_model(self) -> None:
        """
        Loads the Wan 2.2 TI2V-5B pipeline from local weights.
        Uses WanImageToVideoPipeline from diffusers with manual component loading.
        """
        if self.is_loaded:
            logger.info("Wan 2.2 model already loaded, skipping.")
            return

        if self.device == "cpu":
            logger.warning(
                "Wan 2.2 5B on CPU is not feasible for real generation. "
                "A GPU with >= 8GB VRAM is required."
            )
            return

        logger.info(f"Loading Wan 2.2 TI2V-5B from {self.model_path}...")
        logger.info(f"Target device: {self.device}")

        try:
            from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
            from diffusers.schedulers import UniPCMultistepScheduler
            from transformers import CLIPVisionModel, AutoTokenizer, UMT5EncoderModel

            # ── 1. Load Text Encoder (UMT5-XXL) ──────────────────────────
            tokenizer_path = os.path.join(self.model_path, "google", "umt5-xxl")
            t5_weights_path = os.path.join(self.model_path, "models_t5_umt5-xxl-enc-bf16.pth")

            logger.info("Loading UMT5-XXL tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

            logger.info("Loading UMT5-XXL text encoder...")
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.model_path,
                subfolder=None,
                torch_dtype=torch.bfloat16,
            )

            # ── 2. Load VAE ───────────────────────────────────────────────
            logger.info("Loading Wan 2.2 VAE...")
            vae = AutoencoderKLWan.from_pretrained(
                self.model_path,
                subfolder=None,
                torch_dtype=torch.float32,
            )

            # ── 3. Load Image Encoder (CLIP Vision) ──────────────────────
            logger.info("Loading CLIP Vision image encoder...")
            image_encoder = CLIPVisionModel.from_pretrained(
                self.model_path,
                subfolder=None,
                torch_dtype=torch.float32,
            )

            # ── 4. Assemble Pipeline ─────────────────────────────────────
            logger.info("Assembling WanImageToVideoPipeline...")
            self.pipeline = WanImageToVideoPipeline.from_pretrained(
                self.model_path,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                image_encoder=image_encoder,
                torch_dtype=torch.bfloat16,
            )

            # Use UniPC scheduler for faster, higher-quality inference
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                flow_shift=5.0,
            )

            # ── 5. Memory Optimizations ──────────────────────────────────
            if settings.WAN_ENABLE_CPU_OFFLOAD:
                logger.info("Enabling model CPU offloading for VRAM efficiency...")
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)

            # Enable VAE memory optimizations
            try:
                self.pipeline.vae.enable_slicing()
                self.pipeline.vae.enable_tiling()
                logger.info("VAE slicing + tiling enabled.")
            except AttributeError:
                logger.debug("VAE slicing/tiling not supported in this version.")

            self._is_loaded = True
            logger.info("✓ Wan 2.2 TI2V-5B pipeline loaded and ready for generation.")

        except ImportError as e:
            logger.error(
                f"Missing dependency for Wan 2.2: {e}. "
                "Install with: pip install diffusers>=0.33.0 transformers>=4.49.0 accelerate>=0.30.0"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Wan 2.2 model: {e}", exc_info=True)
            self.pipeline = None
            self._is_loaded = False
            raise

    def unload_model(self) -> None:
        """Unloads the model and frees GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._is_loaded = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Wan 2.2 model unloaded. GPU memory freed.")

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
        """
        Generates a video from an input image and text prompt.

        Args:
            image_path: Path to the conditioning keyframe image.
            prompt: Text prompt describing the desired motion/scene.
            output_path: Where to save the output MP4 file.
            negative_prompt: What to avoid in generation.
            num_frames: Number of frames to generate (default from settings).
            height: Output height in pixels (default from settings).
            width: Output width in pixels (default from settings).
            num_inference_steps: Diffusion denoising steps (default from settings).
            guidance_scale: CFG scale (default from settings).
            fps: Frames per second for output video (default from settings).
            seed: Random seed for reproducibility.

        Returns:
            The output_path of the generated video file.
        """
        if not self.is_loaded:
            self.load_model()

        if self.pipeline is None:
            raise RuntimeError(
                "Wan 2.2 pipeline could not be loaded. "
                "Check that CUDA is available and model weights exist at "
                f"{self.model_path}"
            )

        # Apply defaults from settings
        num_frames = num_frames or settings.WAN_NUM_FRAMES
        height = height or settings.WAN_VIDEO_HEIGHT
        width = width or settings.WAN_VIDEO_WIDTH
        num_inference_steps = num_inference_steps or settings.WAN_NUM_INFERENCE_STEPS
        guidance_scale = guidance_scale if guidance_scale is not None else settings.WAN_GUIDANCE_SCALE
        fps = fps or settings.WAN_FPS

        logger.info(
            f"Generating Wan 2.2 video: {width}x{height}, {num_frames} frames, "
            f"{num_inference_steps} steps, cfg={guidance_scale}"
        )
        logger.info(f"Input image: {image_path}")
        logger.info(f"Prompt: {prompt[:120]}...")

        loop = asyncio.get_running_loop()

        def _run_inference():
            from diffusers.utils import export_to_video
            from PIL import Image

            # Load and prepare the conditioning image
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                image = image.resize((width, height))
                logger.info(f"Loaded conditioning image: {image.size}")
            else:
                logger.warning(
                    f"Image not found at {image_path}, generating with a blank image."
                )
                image = Image.new("RGB", (width, height), color=(0, 0, 0))

            # Run the pipeline
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
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                )

            # Export frames to MP4
            frames = output.frames[0]
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            export_to_video(frames, output_path, fps=fps)
            logger.info(f"✓ Video exported to {output_path}")

        # Run the blocking GPU inference in a thread pool to avoid blocking the event loop
        await loop.run_in_executor(None, _run_inference)

        # Verify output
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            duration_s = num_frames / fps
            logger.info(
                f"✓ Generation complete: {file_size_mb:.1f} MB, "
                f"{duration_s:.1f}s @ {fps}fps"
            )
        else:
            logger.error(f"Output video was not created at {output_path}")

        return output_path


# Singleton instance — model is loaded lazily on first generation call
wan_pipeline = WanLocalPipeline()
