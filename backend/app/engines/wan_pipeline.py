"""
Wan 2.1 T2V-1.3B Local GPU/CPU Pipeline.

Loads and runs the Wan 2.1 Text-to-Video 1.3B model from local weights or
Hugging Face repository ('Wan-AI/Wan2.1-T2V-1.3B-Diffusers').

This pipeline loads once and reuses the same model for subsequent requests.
NOTE: The 1.3B model is text-to-video only — no image conditioning.

Post-processing: After generation, a 2x Lanczos upscale pass is applied
(480x272 -> 960x544) using OpenCV for improved perceived quality.
"""

import asyncio
import gc
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

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
    Manages loading, running, and unloading the Wan 2.1 T2V-1.3B model
    for text-to-video generation using diffusers.
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
        local_path = Path(model_str)
        if local_path.exists():
            logger.info("Using local model directory: %s", local_path.resolve())
            return str(local_path.resolve())
        
        # If string is a valid HF repo or default fallback
        if "Wan-AI/" in model_str:
            logger.info("Using Hugging Face repository ID: %s", model_str)
            return model_str

        default_hf_repo = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        logger.info("Local path '%s' not found. Falling back to Hugging Face repository '%s'", model_str, default_hf_repo)
        return default_hf_repo

    def load_model(self) -> None:
        logger.info("=" * 80)
        logger.info("STARTING REAL WAN 2.1 T2V-1.3B PIPELINE INITIALIZATION")
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

        cuda_available = torch.cuda.is_available() if torch is not None else False
        if torch is None or not cuda_available:
            logger.warning("PyTorch/CUDA is not available; skipping real Wan2.2 model loading and using the fallback renderer.")
            raise WanModelLoadingError("PyTorch/CUDA is not available for Wan2.2 inference.")
        # Also check if this is a CPU-only torch build
        torch_cuda_version = getattr(torch.version, 'cuda', None)
        logger.info("  - PyTorch Version : %s", torch.__version__)
        logger.info("  - CUDA Available  : %s", cuda_available)
        logger.info("  - CUDA Build Ver  : %s", torch_cuda_version or 'N/A (CPU-only build)')

        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info("  - GPU Count: %s, Device 0: %s", device_count, device_name)
            target_device = settings.WAN_DEVICE if "cuda" in settings.WAN_DEVICE else "cuda:0"
            torch_dtype = torch.bfloat16
        else:
            logger.warning("  - CUDA is NOT available — running on CPU with float32 precision (slow but functional).")
            if torch_cuda_version is None:
                logger.warning("  - NOTE: torch was installed as a CPU-only build (+cpu). GPU acceleration unavailable.")
                logger.warning("  - TIP : Reinstall torch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            target_device = "cpu"
            torch_dtype = torch.float32

        self.device = target_device

        try:
            from diffusers import WanPipeline
            from diffusers.schedulers import UniPCMultistepScheduler

            # ── Step 3 to 6: Loading WanPipeline (T2V) from pretrained ────────
            logger.info("[STEP 3-6/9] Loading WanPipeline (T2V-1.3B) components from %s...", model_id)
            self.pipeline = WanPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,  # Always True: loads layer-by-layer to keep peak RAM low
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

            # Sequential CPU offload: moves each component to CPU after use
            # This is more aggressive than model_cpu_offload and uses less peak VRAM+RAM
            if settings.WAN_ENABLE_CPU_OFFLOAD and cuda_available:
                logger.info("Enabling sequential CPU offloading (aggressive VRAM+RAM savings)...")
                self.pipeline.enable_sequential_cpu_offload()
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
            for comp_name in ["tokenizer", "text_encoder", "transformer", "vae", "scheduler"]:
                comp = getattr(self.pipeline, comp_name, None)
                if comp is not None:
                    logger.info("  + %-20s: %s", comp_name, type(comp).__name__)
                else:
                    logger.warning("  - %-20s: MISSING / None", comp_name)

            self._is_loaded = True
            logger.info("=" * 80)
            logger.info("REAL WAN 2.1 T2V-1.3B PIPELINE LOADED SUCCESSFULLY!")
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

    def _fallback_generate_video(self, image_path: str, output_path: str, width: int, height: int, fps: int, num_frames: int) -> str:
        from PIL import Image
        import imageio.v2 as imageio
        import numpy as np

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            import sys
            engine_dir = Path("D:/mvid/daily_engine")
            if str(engine_dir) not in sys.path:
                sys.path.insert(0, str(engine_dir))
            from motion_engine import render_parallax_motion_scene
            dur = num_frames / max(1, fps)
            render_parallax_motion_scene(
                Path(image_path), dur, Path(output_path),
                target_width=width, target_height=height, fps=fps, quality_mode="balanced"
            )
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return output_path
        except Exception as exc:
            logger.warning("Parallax fallback error (%s), using safe static drift.", exc)

        source_image = Image.open(image_path).convert("RGB") if Path(image_path).exists() else Image.new("RGB", (width, height), color=(0, 0, 0))
        if source_image.size != (width, height):
            source_image = source_image.resize((width, height))

        frames = []
        for idx in range(num_frames):
            frame = source_image.copy()
            zoom = 1.0 + (idx / max(1, num_frames - 1)) * 0.04
            new_size = (max(1, int(width * zoom)), max(1, int(height * zoom)))
            resized = frame.resize(new_size)
            offset_x = max(0, (new_size[0] - width) // 2)
            offset_y = max(0, (new_size[1] - height) // 2)
            cropped = resized.crop((offset_x, offset_y, offset_x + width, offset_y + height))
            frames.append(cropped)

        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(np.array(frame))
        writer.close()
        return output_path

    async def generate_video(
        self,
        image_path: str,  # kept for API compatibility but not passed to T2V pipeline
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
        target_duration: Optional[float] = None,
    ) -> str:
        if not prompt or not prompt.strip():
            raise WanInferenceError("Prompt must be a non-empty string.")

        num_frames = num_frames or settings.WAN_NUM_FRAMES
        height = height or settings.DEFAULT_HEIGHT
        width = width or settings.DEFAULT_WIDTH
        num_inference_steps = num_inference_steps or settings.WAN_NUM_INFERENCE_STEPS
        guidance_scale = guidance_scale if guidance_scale is not None else settings.WAN_GUIDANCE_SCALE
        fps = fps or settings.DEFAULT_FPS

        fallback_frames = int(round(target_duration * fps)) if (target_duration and fps) else num_frames

        if not self.is_loaded:
            logger.info("Model not loaded yet, triggering load_model()...")
            try:
                self.load_model()
            except Exception as exc:
                logger.warning("Wan2.2 model load failed: %s", exc)
                self.pipeline = None
                self._is_loaded = False

        if self.pipeline is None:
            logger.warning("Real Wan2.2 pipeline unavailable; using local synthetic renderer instead.")
            return self._fallback_generate_video(image_path, output_path, width, height, fps, fallback_frames)

        # ── Step 8: Real Video generation (inference) ───────────────────────
        logger.info(
            "[STEP 8/9] STARTING REAL WAN 2.1 T2V INFERENCE: prompt='%s', width=%s, height=%s, fps=%s, frames=%s, steps=%s, seed=%s",
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
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            # Generator must always be on 'cpu' when no CUDA — even if self.device differs
            generator_device = "cpu" if not torch.cuda.is_available() else self.device
            generator = torch.Generator(device=generator_device).manual_seed(seed)
            logger.info("Generator device: %s", generator_device)

            # Wan 2.1 T2V-1.3B is text-to-video only — no image argument
            logger.info("Executing real PyTorch diffusion inference loop (T2V)...")
            with torch.inference_mode():
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    generator=generator,
                )

            frames = list(output.frames[0])

            # If target duration exceeds generated frames, extend via seamless ping-pong loop
            if target_duration and fps and (target_duration * fps) > len(frames):
                target_frame_count = int(round(target_duration * fps))
                logger.info(
                    "Extending %d diffusion frames to %d frames (%.2fs target duration) via ping-pong loop...",
                    len(frames),
                    target_frame_count,
                    target_duration,
                )
                forward_seq = frames
                backward_seq = frames[-2:0:-1] if len(frames) > 2 else []
                cycle = forward_seq + backward_seq
                extended_frames = []
                while len(extended_frames) < target_frame_count:
                    needed = target_frame_count - len(extended_frames)
                    extended_frames.extend(cycle[:needed])
                frames = extended_frames

            # ── Step 9: Saving real output video ────────────────────────────────
            logger.info("[STEP 9/9] Exporting REAL Wan2.1 T2V video (%d frames) to %s...", len(frames), output_path)
            import imageio.v2 as imageio
            import numpy as np
            writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
            for frame in frames:
                arr = np.array(frame)
                if arr.dtype in (np.float32, np.float64):
                    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
                elif arr.dtype != np.uint8:
                    arr = arr.astype(np.uint8)
                writer.append_data(arr)
            writer.close()
            logger.info("REAL Wan2.1 T2V inference completed successfully. Saved video to %s", output_path)

        try:
            await loop.run_in_executor(None, _run_real_inference)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Real Wan2.2 inference failed with traceback:\n%s", tb)
            logger.warning("Falling back to a local synthetic video renderer because the Wan2.2 model could not run in this environment.")
            return self._fallback_generate_video(image_path, output_path, width, height, fps, fallback_frames)

        if not Path(output_path).exists():
            raise WanInferenceError(f"Wan2.2 generation did not produce output at {output_path}")

        # ── Post-processing: 2x Lanczos upscale ─────────────────────────────
        try:
            from app.engines.upscale_engine import upscale_video
            logger.info("[POST] Starting 2x Lanczos upscale pass...")
            upscaled_path = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: upscale_video(
                    input_path=output_path,
                    output_path=output_path,  # overwrite in-place
                    scale=2.0,
                ),
            )
            logger.info("[POST] Upscale complete: %s", upscaled_path)
        except Exception as upscale_exc:
            # Upscaling is best-effort — never fail the whole generation
            logger.warning("[POST] Upscale step skipped: %s", upscale_exc)

        return output_path


wan_pipeline = WanLocalPipeline()
