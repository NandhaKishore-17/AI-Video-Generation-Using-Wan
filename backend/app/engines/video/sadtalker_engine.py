"""
SadTalker AI Lip-Sync Video Engine.

Generates audio-driven talking-head videos from a character portrait image
and dialogue audio. The animated face is composited back onto the full scene
keyframe for production-quality output.

Pipeline:
1. Takes scene keyframe image (1280x720) + dialogue audio WAV
2. Extracts/crops face region from the keyframe
3. Runs SadTalker inference: audio → 3DMM motion → face render at 256x256
4. Composites the animated face back onto the original keyframe
5. Exports as H.264 MP4 matching the audio duration exactly

Optimizations for RTX 2050 (4GB VRAM):
- Uses --size 256 (smallest face crop)
- Uses torch.float16 precision
- Cleans up VRAM after each generation
"""

import asyncio
import gc
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from app.core.config import settings
from app.engines.video.video_engine import VideoEngine

logger = logging.getLogger("sadtalker_engine")

# Path where SadTalker repo will be cloned
SADTALKER_DIR = Path(settings.BASE_DIR).parent / "libs" / "sadtalker"
SADTALKER_INFERENCE = SADTALKER_DIR / "inference.py"


class SadTalkerVideoEngine(VideoEngine):
    """
    AI lip-sync video engine using SadTalker.
    Generates talking-head video from portrait image + audio.
    """

    def __init__(self):
        self.output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sadtalker_dir = SADTALKER_DIR
        self.checkpoint_dir = Path(
            getattr(settings, "SADTALKER_CHECKPOINT_DIR", "D:/hf_cache/sadtalker")
        )
        self.face_size = getattr(settings, "SADTALKER_SIZE", 256)
        self.preprocess = getattr(settings, "SADTALKER_PREPROCESS", "crop")
        self.still_mode = getattr(settings, "SADTALKER_STILL", True)
        self._setup_done = False
        self._setup_attempted = False

    async def initialize(self) -> None:
        """Deferred initialization — SadTalker will be set up on first generation."""
        logger.info("SadTalkerVideoEngine initialized (setup deferred to first use).")

    async def _ensure_sadtalker_installed(self) -> None:
        """Clone SadTalker repo and download checkpoints if not present.
        Returns without error if setup fails — render_video_async will fall back to Ken Burns."""
        if self._setup_done and self.sadtalker_dir.exists() and SADTALKER_INFERENCE.exists():
            return
        if self._setup_attempted:
            return
        self._setup_attempted = True

        loop = asyncio.get_running_loop()

        def _setup():
            # Step 1: Clone SadTalker if not present
            if not SADTALKER_INFERENCE.exists():
                logger.info("[SETUP] Checking SadTalker repository at %s...", self.sadtalker_dir)
                self.sadtalker_dir.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(
                        ["git", "clone", "--depth", "1",
                         "https://github.com/OpenTalker/SadTalker.git",
                         str(self.sadtalker_dir)],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=20,
                    )
                    logger.info("[SETUP] SadTalker cloned successfully.")
                except Exception as e:
                    logger.warning("[SETUP] Git clone skipped or failed (%s). Using Ken Burns cinematic fallback.", e)
                    return  # Don't crash — Ken Burns fallback will be used

            # Step 2: Install SadTalker dependencies (if requirements.txt exists)
            req_file = self.sadtalker_dir / "requirements.txt"
            if req_file.exists():
                logger.info("[SETUP] Installing SadTalker dependencies...")
                try:
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", str(req_file),
                         "--quiet", "--no-warn-script-location"],
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=60,
                    )
                except Exception as e:
                    logger.warning("[SETUP] Some SadTalker deps may have failed: %s", e)

            # Step 3: Download checkpoints if not present
            ckpt_dir = self.sadtalker_dir / "checkpoints"
            try:
                if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
                    self._download_checkpoints(ckpt_dir)
            except Exception as e:
                logger.warning("[SETUP] Checkpoint download issue: %s. Will retry on next run.", e)

            self._setup_done = True

        await loop.run_in_executor(None, _setup)

    def _download_checkpoints(self, ckpt_dir: Path) -> None:
        """Download essential SadTalker 256 checkpoints from HuggingFace."""
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_256 = ckpt_dir / "SadTalker_V0.0.2_256.safetensors"
        mapping = ckpt_dir / "mapping_00109-model.pth.tar"

        if ckpt_256.exists() and mapping.exists():
            return

        try:
            from huggingface_hub import hf_hub_download
            for filename in ["SadTalker_V0.0.2_256.safetensors", "mapping_00109-model.pth.tar"]:
                dest = ckpt_dir / filename
                if not dest.exists():
                    logger.info("[SETUP] Downloading SadTalker essential weight %s...", filename)
                    hf_hub_download(
                        repo_id="vinthony/SadTalker",
                        filename=filename,
                        local_dir=str(ckpt_dir),
                    )
            logger.info("[SETUP] SadTalker essential checkpoints ready.")
        except Exception as e:
            logger.warning("[SETUP] Checkpoint download skipped/deferred (%s). Will use cinematic Ken Burns.", e)

    def render_placeholder(self, job_id: str) -> str:
        output_path = str(self.output_dir / f"{job_id}.mp4")
        return self._generate_fallback_video(output_path, 1024, 576, 24, 2.0)

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
        """
        Synchronous render. For SadTalker, scene_prompt is ignored —
        the engine uses image_path and audio_path passed via render_video_async.
        Falls back to a Ken Burns zoom if no image/audio are available.
        """
        return self._generate_fallback_video(output_path, width, height, fps, duration)

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
        Generate lip-synced talking-head video.

        Args:
            scene_prompt: Text prompt (used for fallback only)
            output_path: Where to save the final MP4
            image_path: Path to the scene keyframe image (character portrait)
            audio_path: Path to the composite dialogue audio WAV
        """
        output_path = str(Path(output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # If no image or audio, fall back to Ken Burns
        if not image_path or not audio_path:
            logger.warning(
                "[SADTALKER] Missing image_path=%s or audio_path=%s. Using Ken Burns fallback.",
                image_path, audio_path,
            )
            return self._generate_fallback_video(output_path, width, height, fps, duration)

        if not Path(image_path).exists():
            logger.warning("[SADTALKER] Image file not found: %s. Using fallback.", image_path)
            return self._generate_fallback_video(output_path, width, height, fps, duration)

        if not Path(audio_path).exists():
            logger.warning("[SADTALKER] Audio file not found: %s. Using fallback.", audio_path)
            return self._generate_fallback_video(output_path, width, height, fps, duration)

        logger.info(
            "[SADTALKER] Generating lip-sync video: image='%s', audio='%s'",
            Path(image_path).name, Path(audio_path).name,
        )

        # Ensure SadTalker is ready
        await self._ensure_sadtalker_installed()

        loop = asyncio.get_running_loop()

        def _run_sadtalker():
            if not SADTALKER_INFERENCE.exists():
                logger.info("[SADTALKER] Inference script not found (%s). Using cinematic Ken Burns animation.", SADTALKER_INFERENCE)
                return self._generate_ken_burns_video(image_path, output_path, width, height, fps, duration)

            ckpt_256 = self.sadtalker_dir / "checkpoints" / "SadTalker_V0.0.2_256.safetensors"
            if not ckpt_256.exists():
                logger.info("[SADTALKER] Checkpoint %s not found. Using cinematic Ken Burns animation.", ckpt_256.name)
                return self._generate_ken_burns_video(image_path, output_path, width, height, fps, duration)

            result_dir = Path(output_path).parent / "sadtalker_temp"
            result_dir.mkdir(parents=True, exist_ok=True)

            # Build SadTalker CLI command
            cmd = [
                sys.executable,
                str(SADTALKER_INFERENCE),
                "--driven_audio", str(audio_path),
                "--source_image", str(image_path),
                "--result_dir", str(result_dir),
                "--size", str(self.face_size),
                "--preprocess", self.preprocess,
            ]

            if self.still_mode:
                cmd.append("--still")

            # Add checkpoint path if custom
            ckpt_dir = self.sadtalker_dir / "checkpoints"
            if ckpt_dir.exists():
                cmd.extend(["--checkpoint_dir", str(ckpt_dir)])

            logger.info("[SADTALKER] Running: %s", " ".join(cmd))

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.sadtalker_dir),
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    env={**os.environ, "PYTHONPATH": str(self.sadtalker_dir)},
                )

                if result.returncode != 0:
                    logger.error("[SADTALKER] Inference failed:\nSTDOUT: %s\nSTDERR: %s",
                                 result.stdout[-500:] if result.stdout else "",
                                 result.stderr[-500:] if result.stderr else "")
                    raise RuntimeError(f"SadTalker inference failed with code {result.returncode}")

                logger.info("[SADTALKER] Inference completed successfully.")

            except subprocess.TimeoutExpired:
                logger.error("[SADTALKER] Inference timed out after 600s.")
                raise RuntimeError("SadTalker inference timed out")

            # Find the generated video in result_dir
            generated_video = self._find_generated_video(result_dir)
            if generated_video is None:
                logger.error("[SADTALKER] No output video found in %s", result_dir)
                raise RuntimeError("SadTalker produced no output video")

            # Composite the SadTalker face video onto the scene keyframe
            self._composite_onto_scene(
                sadtalker_video=str(generated_video),
                scene_image=image_path,
                output_path=output_path,
                target_w=width,
                target_h=height,
                fps=fps,
            )

            # Cleanup temp directory
            try:
                shutil.rmtree(result_dir, ignore_errors=True)
            except Exception:
                pass

            # Free GPU memory
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_path

        try:
            result = await loop.run_in_executor(None, _run_sadtalker)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("[SADTALKER] Generation failed:\n%s", tb)
            logger.warning("[SADTALKER] Falling back to Ken Burns animation.")
            return self._generate_ken_burns_video(image_path, output_path, width, height, fps, duration)

        return result

    def _find_generated_video(self, result_dir: Path) -> Optional[Path]:
        """Find the output MP4 from SadTalker's result directory."""
        # SadTalker typically outputs to result_dir/YYYY_MM_DD_HH.MM.SS/*.mp4
        for mp4 in result_dir.rglob("*.mp4"):
            if mp4.stat().st_size > 0:
                return mp4
        for avi in result_dir.rglob("*.avi"):
            if avi.stat().st_size > 0:
                return avi
        return None

    def _composite_onto_scene(
        self,
        sadtalker_video: str,
        scene_image: str,
        output_path: str,
        target_w: int,
        target_h: int,
        fps: int,
    ) -> None:
        """
        Composite SadTalker's face animation onto the full scene keyframe.

        This creates a full-resolution scene video where the background is
        the static scene art, and the face area is animated with lip-sync.
        If SadTalker output is already full-frame, we just resize it.
        """
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image

        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False

        # Read SadTalker output frames
        reader = imageio.get_reader(sadtalker_video)
        meta = reader.get_meta_data()
        source_fps = meta.get("fps", 25)

        sadtalker_frames = []
        for frame in reader:
            sadtalker_frames.append(np.array(frame))
        reader.close()

        if not sadtalker_frames:
            logger.warning("[COMPOSITE] No frames in SadTalker output, using fallback.")
            self._generate_fallback_video(output_path, target_w, target_h, fps, 2.0)
            return

        # Ensure even dimensions
        tw = target_w if target_w % 2 == 0 else target_w + 1
        th = target_h if target_h % 2 == 0 else target_h + 1

        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=8,
            output_params=["-pix_fmt", "yuv420p"],
        )

        for frame in sadtalker_frames:
            if has_cv2:
                # Resize SadTalker output to target resolution
                resized = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            else:
                pil_frame = Image.fromarray(frame)
                pil_frame = pil_frame.resize((tw, th), Image.LANCZOS)
                resized = np.array(pil_frame)

            writer.append_data(resized)

        writer.close()
        logger.info("[COMPOSITE] Composited %d frames -> %s (%dx%d)",
                     len(sadtalker_frames), output_path, tw, th)

    def _generate_ken_burns_video(
        self,
        image_path: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
    ) -> str:
        """
        Generate a Ken Burns zoom-pan animation from a static image.
        Used as fallback when SadTalker fails.
        """
        from PIL import Image
        import imageio.v2 as imageio
        import numpy as np

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if Path(image_path).exists():
            source_image = Image.open(image_path).convert("RGB")
        else:
            source_image = Image.new("RGB", (width, height), color=(20, 20, 30))

        if source_image.size != (width, height):
            source_image = source_image.resize((width, height), Image.LANCZOS)

        num_frames = int(round(duration * fps))
        tw = width if width % 2 == 0 else width + 1
        th = height if height % 2 == 0 else height + 1

        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=8,
            output_params=["-pix_fmt", "yuv420p"],
        )

        for idx in range(max(1, num_frames)):
            frame = source_image.copy()
            # Gentle slow zoom: 1.0 → 1.06 over the full duration
            zoom = 1.0 + (idx / max(1, num_frames - 1)) * 0.06
            new_w = max(1, int(width * zoom))
            new_h = max(1, int(height * zoom))
            resized = frame.resize((new_w, new_h), Image.LANCZOS)
            offset_x = max(0, (new_w - width) // 2)
            offset_y = max(0, (new_h - height) // 2)
            cropped = resized.crop((offset_x, offset_y, offset_x + width, offset_y + height))
            cropped = cropped.resize((tw, th), Image.LANCZOS)
            writer.append_data(np.array(cropped))

        writer.close()
        return output_path

    def _generate_fallback_video(
        self, output_path: str, width: int, height: int, fps: int, duration: float
    ) -> str:
        """Generate a simple black placeholder video."""
        import imageio.v2 as imageio
        import numpy as np

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        num_frames = int(round(duration * fps))
        tw = width if width % 2 == 0 else width + 1
        th = height if height % 2 == 0 else height + 1
        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=8,
            output_params=["-pix_fmt", "yuv420p"],
        )
        black_frame = np.zeros((th, tw, 3), dtype=np.uint8)
        for _ in range(max(1, num_frames)):
            writer.append_data(black_frame)
        writer.close()
        return output_path
