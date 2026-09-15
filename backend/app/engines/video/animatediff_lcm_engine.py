"""
AnimateDiff-LCM Fast Video Engine.

Generates text-to-video clips using AnimateDiff with LCM (Latent Consistency Model)
distillation for ultra-fast 4-step inference on low-VRAM GPUs (4GB RTX 2050).

Key differences from Wan 2.1:
- LCM is specifically trained to converge in 4 steps (Wan needs 14-20+)
- SD1.5-based (smaller model, better for 4GB VRAM)
- Produces 16 clean frames per clip instead of 81 noisy ones
- Extends to target duration via smooth slow-motion interpolation (not ping-pong)
"""

import asyncio
import gc
import logging
import traceback
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from app.core.config import settings
from app.engines.video.video_engine import VideoEngine

logger = logging.getLogger("animatediff_lcm_engine")


class AnimateDiffLCMVideoEngine(VideoEngine):
    """
    Fast text-to-video engine using AnimateDiff + LCM LoRA.
    Produces clean 4-step diffusion video on 4GB VRAM GPUs.
    """

    def __init__(self):
        self.output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None
        self._is_loaded = False

    async def initialize(self) -> None:
        """Deferred initialization — model loads on first generation."""
        logger.info("AnimateDiffLCMVideoEngine initialized (model loading deferred to first use).")

    def _load_pipeline(self) -> None:
        """Load AnimateDiff + LCM pipeline with aggressive memory optimizations."""
        if self._is_loaded and self.pipeline is not None:
            return

        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("PyTorch with CUDA is required for AnimateDiff-LCM.")

        logger.info("=" * 70)
        logger.info("LOADING ANIMATEDIFF-LCM PIPELINE")
        logger.info("=" * 70)

        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter
            from diffusers.schedulers import LCMScheduler

            base_model = getattr(settings, "ANIMATEDIFF_BASE_MODEL", "emilianJR/epiCRealism")
            lcm_adapter = getattr(settings, "ANIMATEDIFF_LCM_ADAPTER", "wangfuyun/AnimateLCM")

            logger.info("  Base model : %s", base_model)
            logger.info("  LCM adapter: %s", lcm_adapter)

            # Step 1: Load motion adapter
            logger.info("[1/4] Loading AnimateLCM motion adapter...")
            adapter = MotionAdapter.from_pretrained(
                lcm_adapter,
                torch_dtype=torch.float16,
            )

            # Step 2: Load base pipeline with motion adapter
            logger.info("[2/4] Loading base SD1.5 pipeline with motion adapter...")
            self.pipeline = AnimateDiffPipeline.from_pretrained(
                base_model,
                motion_adapter=adapter,
                torch_dtype=torch.float16,
            )

            # Step 3: Apply LCM scheduler
            logger.info("[3/4] Configuring LCM scheduler...")
            self.pipeline.scheduler = LCMScheduler.from_config(
                self.pipeline.scheduler.config,
                beta_schedule="linear",
            )

            # Step 4: Load LCM LoRA weights
            logger.info("[4/4] Loading LCM LoRA weights...")
            self.pipeline.load_lora_weights(
                lcm_adapter,
                weight_name="AnimateLCM_sd15_t2v_lora.safetensors",
                adapter_name="lcm-lora",
            )
            self.pipeline.set_adapters(["lcm-lora"], [0.8])

            # Memory optimizations for 4GB VRAM
            logger.info("Enabling memory optimizations for 4GB VRAM...")
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_model_cpu_offload()

            self._is_loaded = True
            logger.info("=" * 70)
            logger.info("ANIMATEDIFF-LCM PIPELINE LOADED SUCCESSFULLY!")
            logger.info("=" * 70)

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Failed to load AnimateDiff-LCM pipeline:\n%s", tb)
            self.pipeline = None
            self._is_loaded = False
            raise RuntimeError(f"AnimateDiff-LCM loading failed: {exc}") from exc

    def _unload_pipeline(self) -> None:
        """Free GPU memory after generation."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            self._is_loaded = False
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("AnimateDiff-LCM pipeline unloaded, VRAM freed.")

    def render_placeholder(self, job_id: str) -> str:
        output_path = str(self.output_dir / f"{job_id}.mp4")
        return self._generate_fallback(output_path, 512, 288, 24, 2.0)

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
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            return asyncio.run(
                self.render_video_async(scene_prompt, output_path, width, height, fps, duration, seed)
            )

        import threading
        result = {}
        error = {}

        def runner():
            try:
                result["value"] = asyncio.run(
                    self.render_video_async(scene_prompt, output_path, width, height, fps, duration, seed)
                )
            except BaseException as exc:
                error["value"] = exc

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

        if "value" in result:
            return str(result["value"])
        if "value" in error:
            raise error["value"]
        raise RuntimeError("AnimateDiff-LCM rendering did not return a result")

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

        num_frames = getattr(settings, "ANIMATEDIFF_NUM_FRAMES", 16)
        num_steps = getattr(settings, "ANIMATEDIFF_NUM_STEPS", 4)
        guidance_scale = getattr(settings, "ANIMATEDIFF_GUIDANCE_SCALE", 2.0)

        logger.info(
            "[ANIMATEDIFF-LCM] Generating video: prompt='%s', frames=%d, steps=%d, seed=%d",
            scene_prompt[:80], num_frames, num_steps, seed,
        )

        loop = asyncio.get_running_loop()

        def _run_inference():
            try:
                self._load_pipeline()
            except Exception:
                logger.warning("AnimateDiff-LCM pipeline unavailable, using fallback.")
                return self._generate_fallback(output_path, width, height, fps, duration)

            generator = torch.Generator(device="cpu").manual_seed(seed)

            # AnimateDiff works best at 512x512 (SD1.5 native)
            gen_size = 512

            logger.info("Running AnimateDiff-LCM inference (%d steps, %d frames)...", num_steps, num_frames)
            with torch.inference_mode():
                output = self.pipeline(
                    prompt=scene_prompt,
                    negative_prompt="ugly, blurry, low quality, distorted, disfigured",
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    height=gen_size,
                    width=gen_size,
                    generator=generator,
                )

            frames = list(output.frames[0])
            logger.info("Generated %d raw frames from AnimateDiff-LCM.", len(frames))

            # Extend frames to target duration via smooth slow-motion interpolation
            target_frame_count = int(round(duration * fps))
            if target_frame_count > len(frames):
                frames = self._extend_frames_slowmo(frames, target_frame_count)
                logger.info("Extended to %d frames via slow-motion interpolation.", len(frames))

            # Export video at target aspect ratio
            self._export_video(frames, output_path, width, height, fps)
            logger.info("AnimateDiff-LCM video saved: %s", output_path)

            # Free VRAM for next engine
            self._unload_pipeline()

            return output_path

        try:
            result = await loop.run_in_executor(None, _run_inference)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("AnimateDiff-LCM inference failed:\n%s", tb)
            return self._generate_fallback(output_path, width, height, fps, duration)

        return result

    def _extend_frames_slowmo(self, frames: list, target_count: int) -> list:
        """
        Extend a short clip to target frame count via smooth slow-motion.
        Instead of ping-pong boomerang, this stretches the motion smoothly
        by interpolating between frames using linear blending.
        """
        import numpy as np
        from PIL import Image

        if len(frames) < 2:
            return frames * target_count

        src_count = len(frames)
        extended = []

        for i in range(target_count):
            # Map target frame index back to source frame space
            src_pos = (i / max(1, target_count - 1)) * (src_count - 1)
            src_idx = int(src_pos)
            blend_factor = src_pos - src_idx

            if src_idx >= src_count - 1:
                extended.append(frames[-1])
            elif blend_factor < 0.01:
                extended.append(frames[src_idx])
            else:
                # Linear blend between adjacent source frames
                arr_a = np.array(frames[src_idx]).astype(np.float32)
                arr_b = np.array(frames[src_idx + 1]).astype(np.float32)
                blended = (arr_a * (1.0 - blend_factor) + arr_b * blend_factor).astype(np.uint8)
                extended.append(Image.fromarray(blended))

        return extended

    def _export_video(self, frames: list, output_path: str, target_w: int, target_h: int, fps: int) -> None:
        """Export frames to MP4, resizing/cropping to target aspect ratio."""
        import imageio.v2 as imageio
        import numpy as np

        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=8,
            output_params=["-pix_fmt", "yuv420p"],
        )

        for frame in frames:
            arr = np.array(frame)
            if arr.dtype in (np.float32, np.float64):
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            elif arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            # Resize to target dimensions
            if has_cv2 and (arr.shape[1] != target_w or arr.shape[0] != target_h):
                # Ensure even dimensions
                tw = target_w if target_w % 2 == 0 else target_w + 1
                th = target_h if target_h % 2 == 0 else target_h + 1
                arr = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_LANCZOS4)

            writer.append_data(arr)

        writer.close()

    def _generate_fallback(self, output_path: str, width: int, height: int, fps: int, duration: float) -> str:
        """Generate a simple black placeholder video as fallback."""
        import imageio.v2 as imageio
        import numpy as np

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        num_frames = int(round(duration * fps))
        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=8,
            output_params=["-pix_fmt", "yuv420p"],
        )
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(1, num_frames)):
            writer.append_data(black_frame)
        writer.close()
        return output_path
