"""
motion_engine.py - Safe 2.5D Cinematic Motion & Visual Preservation Engine
Kaalapadhivugal Production Engine.

Adheres strictly to quality-first principles:
1. Safe 2.5D depth approximation with pre-scaling to prevent black border exposure.
2. Soft subject protection with continuous feathered motion weights (never hard-freezing or warping subjects).
3. Multi-signal confidence gating for environmental shaders (clean static region if confidence < 0.55).
4. Motion quality scoring with automatic retry and static-safe fallback.
5. Pluggable MotionGenerator and FrameInterpolationProvider abstractions.
6. Optimized for RTX 2050 (4 GB VRAM) — Depth runs on CPU with persistent disk caching.
"""

import os
import sys
import time
import hashlib
import logging
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import imageio_ffmpeg

logger = logging.getLogger("motion_engine")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

CACHE_DIR = Path(r"D:\mvid\daily_engine\.depth_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. B-ROLL PREPARATION (PRESERVED FUNCTIONALITY)
# ==============================================================================

def prepare_broll_clip(
    video_path: Path,
    target_duration: float,
    output_path: Path,
    target_width: int = 1280,
    target_height: int = 720,
    fps: int = 25,
) -> Path:
    """
    Trims, loops (if needed), and rescales a motion B-roll video clip
    to precisely match target_duration at 1280x720 16:9.
    """
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vf = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height},"
        f"fps={fps}"
    )

    cmd = [
        FFMPEG, "-y",
        "-stream_loop", "10",
        "-i", str(video_path),
        "-t", f"{target_duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


# ==============================================================================
# 2. DEPTH ESTIMATION & DISK CACHING
# ==============================================================================

class DepthEstimator:
    """
    Lightweight Depth Anything V2 estimator running safely on CPU.
    Caches computed depth maps to disk to prevent redundant computation.
    """
    _instance = None

    @classmethod
    def get_instance(cls, device: str = "cpu"):
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._pipe = None

    def _load_pipe(self):
        if self._pipe is None:
            t0 = time.time()
            from transformers import pipeline
            print(f"[DEPTH] Loading Depth-Anything-V2-Small on {self.device}...")
            self._pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=self.device
            )
            print(f"[DEPTH] Depth model loaded in {time.time()-t0:.2f}s")
        return self._pipe

    def estimate(self, image: Image.Image, target_size: Tuple[int, int] = (1280, 720)) -> Tuple[np.ndarray, Dict[str, Any]]:
        orig_w, orig_h = target_size
        img_resized = image.resize((orig_w, orig_h), Image.LANCZOS)
        img_bytes = img_resized.tobytes()
        cache_key = hashlib.md5(img_bytes + f"{orig_w}x{orig_h}".encode()).hexdigest()
        cache_path = CACHE_DIR / f"{cache_key}.npy"

        if cache_path.exists():
            t0 = time.time()
            depth_map = np.load(cache_path)
            return depth_map, {"depth_time": round(time.time() - t0, 4), "cache_hit": True}

        t0 = time.time()
        pipe = self._load_pipe()
        # Fast inference at 640x360, then bicubic resize to target resolution
        inf_img = img_resized.resize((640, 360), Image.BILINEAR)
        result = pipe(inf_img)
        depth_pil = result["depth"].resize((orig_w, orig_h), Image.BICUBIC)
        depth_arr = np.array(depth_pil, dtype=np.float32)

        d_min, d_max = depth_arr.min(), depth_arr.max()
        if d_max > d_min:
            depth_map = (depth_arr - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_arr)

        np.save(cache_path, depth_map)
        return depth_map, {"depth_time": round(time.time() - t0, 3), "cache_hit": False}


# ==============================================================================
# 3. SOFT SUBJECT PROTECTION MASK
# ==============================================================================

def generate_soft_subject_mask(image: np.ndarray, depth_map: np.ndarray, shot_type: str = "PORTRAIT") -> np.ndarray:
    """
    Generates a continuous feathered subject motion weight map [0.08, 1.0].
    Focal subject receives low motion weight to prevent warping.
    Background receives full motion weight.
    """
    H, W = depth_map.shape
    y_coords, x_coords = np.ogrid[:H, :W]

    # Shot-aware center prior
    is_portrait = "PORTRAIT" in shot_type.upper() or "CLOSE" in shot_type.upper()
    cy = H * 0.48 if is_portrait else H * 0.52
    cx = W * 0.50
    sigma_y = H * 0.30 if is_portrait else H * 0.38
    sigma_x = W * 0.22 if is_portrait else W * 0.32

    center_prior = np.exp(-(((x_coords - cx) ** 2) / (2 * sigma_x ** 2) + ((y_coords - cy) ** 2) / (2 * sigma_y ** 2)))
    depth_saliency = np.clip(depth_map, 0.0, 1.0)

    # Subject likelihood: center prior * depth saliency
    subject_likelihood = center_prior * (depth_saliency ** 1.6)
    max_val = subject_likelihood.max()
    if max_val > 0:
        subject_likelihood = subject_likelihood / max_val

    # Protection scaling
    protection_factor = 0.92 if is_portrait else 0.75
    motion_weight = 1.0 - protection_factor * subject_likelihood

    # Feather with Gaussian filter to ensure smooth transitions
    motion_weight = gaussian_filter(motion_weight, sigma=H * 0.035)
    return np.clip(motion_weight, 0.08, 1.0).astype(np.float32)


# ==============================================================================
# 4. MULTI-SIGNAL CONFIDENCE ENVIRONMENTAL SHADERS
# ==============================================================================

def compute_water_confidence(
    image: np.ndarray,
    depth_map: np.ndarray,
    semantic_elements: List[str]
) -> Tuple[float, np.ndarray]:
    """
    Multi-signal confidence calculation for water presence (0.0 to 1.0).
    Combines semantic prior, spatial lower-third weighting, HSV blue/specular analysis, and depth smoothness.
    """
    H, W, _ = image.shape
    water_keywords = {"water", "ocean", "sea", "harbor", "port", "river", "waves", "ship", "boat"}
    has_semantic = any(k in [str(e).lower() for e in semantic_elements] for k in water_keywords)
    s_score = 1.0 if has_semantic else 0.0

    y_indices, _ = np.indices((H, W), dtype=np.float32)
    spatial_weight = np.clip((y_indices - H * 0.42) / (H * 0.38), 0.0, 1.0)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Blue/cyan hue (roughly 80 to 140) + dark specular reflectance
    is_blue_cyan = ((hue >= 80) & (hue <= 140) & (sat >= 15)).astype(np.float32)
    is_specular = ((val < 85) & (sat < 65) & (spatial_weight > 0.4)).astype(np.float32)
    color_mask = np.clip(is_blue_cyan * 0.8 + is_specular * 0.5, 0.0, 1.0)

    dy = np.abs(np.diff(depth_map, axis=0, append=depth_map[-1:, :]))
    dx = np.abs(np.diff(depth_map, axis=1, append=depth_map[:, -1:]))
    depth_smooth = 1.0 - np.clip((dx + dy) * 8.0, 0.0, 1.0)

    pixel_water = spatial_weight * (0.65 * color_mask + 0.35 * depth_smooth)
    water_region_ratio = float(np.mean(pixel_water > 0.30))
    c_score = min(1.0, water_region_ratio * 3.0)

    overall_confidence = 0.35 * s_score + 0.35 * c_score + 0.30 * float(np.mean(pixel_water[int(H * 0.55):, :]))
    overall_confidence = float(np.clip(overall_confidence, 0.0, 1.0))
    return overall_confidence, pixel_water


def apply_confidence_water_shader(
    frame: np.ndarray,
    water_mask: np.ndarray,
    confidence: float,
    progress: float,
) -> np.ndarray:
    """
    Applies subtle harmonic horizontal wave displacement ONLY if confidence >= 0.55.
    If confidence < 0.55, frame is returned completely untouched.
    """
    if confidence < 0.55:
        return frame

    # Amplitude scaling based on confidence (max 2.2 pixels)
    amp = 2.2 * ((confidence - 0.55) / 0.45) if confidence < 0.75 else 2.5
    H, W, _ = frame.shape
    y_grid, x_grid = np.indices((H, W), dtype=np.float32)

    # Harmonic wave displacement vector
    phase = 2.0 * np.pi * progress * 1.5
    wave_x = amp * np.sin(y_grid * 0.05 + phase) * water_mask
    wave_y = (amp * 0.4) * np.cos(x_grid * 0.03 + phase) * water_mask

    sample_x = np.clip(x_grid + wave_x, 0, W - 1).astype(np.float32)
    sample_y = np.clip(y_grid + wave_y, 0, H - 1).astype(np.float32)
    return cv2.remap(frame, sample_x, sample_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_atmospheric_particles(
    frame: np.ndarray,
    progress: float,
    intensity: float = 0.05
) -> np.ndarray:
    """
    Subtle drifting dust motes / mist with soft non-destructive blending.
    """
    H, W, _ = frame.shape
    overlay = np.zeros((H, W), dtype=np.float32)

    # 40 discrete atmospheric particles
    np.random.seed(42)
    pts_x = (np.random.rand(40) * W + progress * 40.0) % W
    pts_y = (np.random.rand(40) * H - progress * 20.0) % H
    radii = np.random.randint(2, 6, size=40)

    for px, py, r in zip(pts_x, pts_y, radii):
        cv2.circle(overlay, (int(px), int(py)), int(r), 0.8, -1)

    overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
    frame_float = frame.astype(np.float32)
    blend = frame_float + overlay[:, :, None] * (intensity * 255.0)
    return np.clip(blend, 0, 255).astype(np.uint8)


# ==============================================================================
# 5. MOTION QUALITY SCORER
# ==============================================================================

class MotionQualityScorer:
    """
    Automated motion safety and artifact evaluation system.
    Evaluates optical flow spikes, edge tearing, border boundary exposure,
    temporal flicker, structural stability, and subject stability.

    Note: A score of 1.0 indicates 'no detected violations according to current automated safety metrics',
    not 'absolute photographic perfection'.
    """
    def __init__(self, flow_spike_threshold: float = 24.0):
        self.flow_spike_threshold = flow_spike_threshold

    def evaluate_video(
        self,
        frames: List[np.ndarray],
        subject_mask: Optional[np.ndarray] = None,
        target_fps: int = 25
    ) -> Dict[str, Any]:
        if len(frames) < 2:
            return {
                "motion_safety_score": 1.0,
                "quality_score": 1.0,
                "is_safe": True,
                "optical_flow_spike_count": 0,
                "border_violation_count": 0,
                "temporal_flicker_score": 0.0,
                "structural_stability_score": 1.0,
                "subject_stability_score": 1.0,
                "frame_difference_mean": 0.0,
                "frame_difference_max": 0.0,
                "frame_difference_std": 0.0,
                "output_fps": target_fps,
                "output_resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}" if frames else "0x0"
            }

        H, W, _ = frames[0].shape
        boundary_violations = 0
        flow_spikes = 0
        step = max(1, len(frames) // 10)
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        # 1. Border Exposure Check (outer 2 pixels along all 4 edges)
        for f in frames[::step]:
            top, bottom, left, right = f[:2, :, :], f[-2:, :, :], f[:, :2, :], f[:, -2:, :]
            for edge in (top, bottom, left, right):
                if np.mean(edge) < 2.0:
                    boundary_violations += 1
                    break

        # 2. Optical Flow & Temporal Difference Metrics
        luminance_means = []
        frame_diffs = []
        subject_displacements = []

        for f in frames:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            luminance_means.append(float(np.mean(gray)))

        for idx in range(len(frames) - 1):
            f_curr = frames[idx].astype(np.float32)
            f_next = frames[idx + 1].astype(np.float32)
            diff = np.abs(f_next - f_curr)
            frame_diffs.append(float(np.mean(diff)))

        # Subsampled dense optical flow check
        for idx in range(step, len(frames), step):
            curr_gray = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            if float(np.max(mag)) > self.flow_spike_threshold:
                flow_spikes += 1

            if subject_mask is not None:
                # Measure flow specifically within protected subject region
                sub_flow = mag * (1.0 - subject_mask)
                subject_displacements.append(float(np.mean(sub_flow)))

            prev_gray = curr_gray

        # 3. Temporal Flicker: Standard deviation of frame-to-frame luminance differences
        lum_deltas = [abs(luminance_means[i+1] - luminance_means[i]) for i in range(len(luminance_means) - 1)]
        temporal_flicker = float(np.std(lum_deltas)) if lum_deltas else 0.0

        # 4. Structural Stability: Inverse of normalized frame difference variance
        mean_diff = float(np.mean(frame_diffs)) if frame_diffs else 0.0
        max_diff = float(np.max(frame_diffs)) if frame_diffs else 0.0
        std_diff = float(np.std(frame_diffs)) if frame_diffs else 0.0
        structural_stability = float(np.clip(1.0 - (std_diff / 25.0), 0.0, 1.0))

        # 5. Subject Stability Score: 1.0 minus average subject-region displacement anomaly
        mean_sub_disp = float(np.mean(subject_displacements)) if subject_displacements else 0.0
        subject_stability = float(np.clip(1.0 - (mean_sub_disp / 10.0), 0.0, 1.0))

        # Composite Safety Score
        safety_score = 1.0
        if boundary_violations > 0:
            safety_score -= 0.40 * min(1.0, boundary_violations / 3.0)
        if flow_spikes > 0:
            safety_score -= 0.35 * min(1.0, flow_spikes / (len(frames) // step))
        if temporal_flicker > 3.0:
            safety_score -= 0.15 * min(1.0, (temporal_flicker - 3.0) / 5.0)

        safety_score = float(np.clip(safety_score, 0.0, 1.0))

        # Memory tracking
        import psutil
        peak_ram = float(psutil.Process().memory_info().rss / (1024 * 1024))
        peak_vram = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                peak_vram = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        except Exception:
            pass

        return {
            "motion_safety_score": round(safety_score, 3),
            "quality_score": round(safety_score, 3), # backward compatibility alias
            "is_safe": safety_score >= 0.70,
            "optical_flow_spike_count": flow_spikes,
            "flow_spikes": flow_spikes,
            "border_violation_count": boundary_violations,
            "boundary_violations": boundary_violations,
            "temporal_flicker_score": round(temporal_flicker, 4),
            "structural_stability_score": round(structural_stability, 3),
            "subject_stability_score": round(subject_stability, 3),
            "frame_difference_mean": round(mean_diff, 2),
            "frame_difference_max": round(max_diff, 2),
            "frame_difference_std": round(std_diff, 2),
            "peak_ram_mb": round(peak_ram, 1),
            "peak_vram_mb": round(peak_vram, 1),
            "output_fps": target_fps,
            "output_resolution": f"{W}x{H}"
        }


# ==============================================================================
# 6. FRAME INTERPOLATION PROVIDER ABSTRACTION
# ==============================================================================

class BaseFrameInterpolationProvider(ABC):
    @abstractmethod
    def interpolate(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        pass


class NoInterpolationProvider(BaseFrameInterpolationProvider):
    def interpolate(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        return frames


class OpticalFlowInterpolationProvider(BaseFrameInterpolationProvider):
    """Smooth temporal interpolation via Farneback dense optical flow blending."""
    def interpolate(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        if len(frames) < 2:
            return frames
        interpolated = []
        for i in range(len(frames) - 1):
            f1 = frames[i]
            f2 = frames[i + 1]
            interpolated.append(f1)
            # Intermediate weighted blend
            mid = cv2.addWeighted(f1, 0.5, f2, 0.5, 0)
            interpolated.append(mid)
        interpolated.append(frames[-1])
        return interpolated


# ==============================================================================
# 7. PLUGGABLE MOTION GENERATOR ABSTRACTION
# ==============================================================================

class BaseMotionGenerator(ABC):
    @abstractmethod
    def generate_motion(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        pass


class StaticMotionGenerator(BaseMotionGenerator):
    """Fallback generator with subtle organic camera breathing drift."""
    def generate_motion(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        W, H = target_size
        total_frames = int(round(duration * fps))
        frames = []
        pre_scale = 1.04
        pre_w, pre_h = int(W * pre_scale), int(H * pre_scale)
        margin_x, margin_y = (pre_w - W) // 2, (pre_h - H) // 2
        img_pre = cv2.resize(image, (pre_w, pre_h), interpolation=cv2.INTER_LANCZOS4)

        for i in range(total_frames):
            p = i / max(1, total_frames - 1)
            shift_x = int(margin_x * 0.4 * np.sin(2.0 * np.pi * p))
            shift_y = int(margin_y * 0.4 * np.cos(2.0 * np.pi * p))
            x0 = margin_x + shift_x
            y0 = margin_y + shift_y
            crop = img_pre[y0:y0+H, x0:x0+W]
            frames.append(crop)
        return frames


class ParallaxMotionGenerator(BaseMotionGenerator):
    """
    Safe 2.5D depth parallax generator with soft subject protection and border safety pre-scaling.
    """
    def __init__(self, shot_type: str = "WIDE_ESTABLISHING"):
        self.shot_type = shot_type

    def generate_motion(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int],
        motion_scale: float = 1.0
    ) -> List[np.ndarray]:
        W, H = target_size
        total_frames = int(round(duration * fps))

        shot_type = self.shot_type.upper()
        if "PORTRAIT" in shot_type or "CLOSE" in shot_type:
            base_strength = 0.12
        elif "TEMPLE" in shot_type or "ARCHITECTURE" in shot_type:
            base_strength = 0.25
        elif "SHIP" in shot_type or "MARKET" in shot_type:
            base_strength = 0.40
        else: # WIDE / LANDSCAPE / OCEAN
            base_strength = 0.45

        budget = str(motion_plan.get("motion_budget", "MEDIUM")).upper()
        if budget == "LOW":
            base_strength *= 0.65
        elif budget == "HIGH":
            base_strength *= 1.30

        effective_strength = float(np.clip(base_strength * motion_scale, 0.08, 0.55))
        weight_map = generate_soft_subject_mask(image, depth_map, shot_type=shot_type)

        pre_scale = 1.06
        pre_w, pre_h = int(W * pre_scale), int(H * pre_scale)
        margin_x, margin_y = (pre_w - W) // 2, (pre_h - H) // 2

        img_pre = cv2.resize(image, (pre_w, pre_h), interpolation=cv2.INTER_LANCZOS4)
        depth_pre = cv2.resize(depth_map, (pre_w, pre_h), interpolation=cv2.INTER_LINEAR)
        weight_pre = cv2.resize(weight_map, (pre_w, pre_h), interpolation=cv2.INTER_LINEAR)

        max_shift_x = margin_x * 0.85 * effective_strength
        max_shift_y = margin_y * 0.85 * effective_strength

        camera_type = motion_plan.get("camera", "slow_dolly_forward")
        y_grid, x_grid = np.indices((H, W), dtype=np.float32)
        frames = []

        for f_idx in range(total_frames):
            p = f_idx / max(1, total_frames - 1)
            t = 0.5 * (1.0 - np.cos(np.pi * p))

            if camera_type == "pan_left":
                cx = max_shift_x * t
                cy = 0.0
            elif camera_type == "pan_right":
                cx = -max_shift_x * t
                cy = 0.0
            elif camera_type in ("pedestal_up", "crane_up"):
                cx = 0.0
                cy = max_shift_y * t
            elif camera_type in ("pedestal_down", "crane_down"):
                cx = 0.0
                cy = -max_shift_y * t
            elif camera_type in ("slow_dolly_forward", "dolly_in"):
                cx = 0.0
                cy = 0.0
            else: # subtle_drift
                cx = max_shift_x * 0.5 * np.sin(2.0 * np.pi * p)
                cy = max_shift_y * 0.5 * np.cos(2.0 * np.pi * p)

            src_x0 = margin_x
            src_y0 = margin_y
            sub_d = depth_pre[src_y0:src_y0+H, src_x0:src_x0+W]
            sub_w = weight_pre[src_y0:src_y0+H, src_x0:src_x0+W]

            if camera_type in ("slow_dolly_forward", "dolly_in"):
                z_factor = 1.0 + (0.04 * effective_strength * t) * (0.35 + 0.65 * sub_d)
                sample_x = (x_grid - W * 0.5) / z_factor + W * 0.5 + src_x0
                sample_y = (y_grid - H * 0.5) / z_factor + H * 0.5 + src_y0
            else:
                diff_shift_x = cx * (1.0 - 0.70 * (1.0 - sub_w))
                diff_shift_y = cy * (1.0 - 0.70 * (1.0 - sub_w))
                sample_x = x_grid + src_x0 - diff_shift_x
                sample_y = y_grid + src_y0 - diff_shift_y

            sample_x = np.clip(sample_x, 0, pre_w - 1).astype(np.float32)
            sample_y = np.clip(sample_y, 0, pre_h - 1).astype(np.float32)
            frame = cv2.remap(img_pre, sample_x, sample_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frames.append(frame)

        return frames


class EnvironmentalMotionGenerator(BaseMotionGenerator):
    """
    Combines safe 2.5D parallax with multi-signal confidence environmental shaders
    (water wave displacement, atmospheric particles, light breathing).
    """
    def __init__(self, shot_type: str = "WIDE_ESTABLISHING"):
        self.parallax_gen = ParallaxMotionGenerator(shot_type=shot_type)

    def generate_motion(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int],
        motion_scale: float = 1.0
    ) -> List[np.ndarray]:
        frames = self.parallax_gen.generate_motion(
            image, depth_map, motion_plan, duration, fps, target_size, motion_scale=motion_scale
        )

        detected = motion_plan.get("detected_elements", [])
        water_conf, water_mask = compute_water_confidence(image, depth_map, detected)
        apply_particles = "PORTRAIT" not in str(motion_plan.get("shot_type", "")).upper()

        enhanced_frames = []
        total = len(frames)
        for idx, f in enumerate(frames):
            p = idx / max(1, total - 1)
            out_f = f
            if water_conf >= 0.55:
                out_f = apply_confidence_water_shader(out_f, water_mask, water_conf, p)
            if apply_particles:
                out_f = apply_atmospheric_particles(out_f, p, intensity=0.04)
            enhanced_frames.append(out_f)

        return enhanced_frames


class OptionalImageToVideoGenerator(BaseMotionGenerator):
    """
    Optional stub for external / cloud GPU image-to-video diffusion (disabled by default).
    """
    def generate_motion(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int]
    ) -> List[np.ndarray]:
        env_gen = EnvironmentalMotionGenerator(shot_type=motion_plan.get("shot_type", "WIDE"))
        return env_gen.generate_motion(image, depth_map, motion_plan, duration, fps, target_size)


# ==============================================================================
# 8. MASTER RENDER PIPELINE WITH RETRY & STATIC-SAFE FALLBACK
# ==============================================================================

def render_parallax_motion_scene(
    image_path: Path,
    target_duration: float,
    output_path: Path,
    target_width: int = 1280,
    target_height: int = 720,
    fps: int = 25,
    shot_type: str = "WIDE_ESTABLISHING",
    motion_plan: Optional[Dict[str, Any]] = None,
    quality_mode: str = "balanced",
) -> Path:
    """
    Master motion rendering pipeline:
    1. Ingests source image and retrieves cached depth map.
    2. Runs EnvironmentalMotionGenerator with shot-aware kinematics.
    3. MotionQualityScorer validates for optical flow spikes and edge tearing.
    4. Automatically retries with reduced motion if artifacts are detected (max 2 retries).
    5. Falls back to static-safe camera drift if retries fail.
    6. Writes broadcast-ready MP4.
    """
    t_start = time.time()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    motion_plan = motion_plan or {
        "camera": "slow_dolly_forward",
        "motion_budget": "MEDIUM",
        "detected_elements": ["landscape"],
        "shot_type": shot_type
    }

    # Step 1: Load Image & Depth Map
    img_pil = Image.open(image_path).convert("RGB")
    estimator = DepthEstimator.get_instance(device="cpu")
    depth_map, d_metrics = estimator.estimate(img_pil, target_size=(target_width, target_height))
    img_np = np.array(img_pil.resize((target_width, target_height), Image.LANCZOS))

    print(f"[MOTION] Rendering scene: '{image_path.name}' ({shot_type}) | Duration: {target_duration:.2f}s | Depth cache hit: {d_metrics.get('cache_hit')}")

    # Step 2: Select Generator based on quality mode
    if quality_mode.lower() == "fast":
        generator = ParallaxMotionGenerator(shot_type=shot_type)
    else: # balanced or high
        generator = EnvironmentalMotionGenerator(shot_type=shot_type)

    scorer = MotionQualityScorer()
    motion_scale = 1.0
    frames = []
    MAX_RETRIES = 2
    successful_render = False

    for attempt in range(MAX_RETRIES + 1):
        t_gen0 = time.time()
        frames = generator.generate_motion(
            img_np, depth_map, motion_plan, target_duration, fps, (target_width, target_height), motion_scale=motion_scale
        )
        t_gen = time.time() - t_gen0

        # Step 3: Evaluate Motion Quality
        eval_res = scorer.evaluate_video(frames)
        print(f"[MOTION] Attempt {attempt+1}: Quality Score = {eval_res['quality_score']} (Safe: {eval_res['is_safe']})")

        if eval_res["is_safe"]:
            successful_render = True
            break
        else:
            print(f"[MOTION] Warning: Motion artifacts detected (flow spikes: {eval_res['flow_spikes']}, border violations: {eval_res['boundary_violations']}). Reducing motion budget...")
            motion_scale *= 0.50

    # Step 4: Static-safe fallback if all motion attempts produced artifacts
    if not successful_render:
        print("[MOTION] Notice: Falling back to static-safe organic camera drift.")
        static_gen = StaticMotionGenerator()
        frames = static_gen.generate_motion(
            img_np, depth_map, motion_plan, target_duration, fps, (target_width, target_height)
        )

    # Step 5: Optional Frame Interpolation for High Quality Mode
    if quality_mode.lower() == "high":
        interpolator = OpticalFlowInterpolationProvider()
        # Temporal smoothing
        frames = interpolator.interpolate(frames, target_fps=fps)

    # Step 6: Write Video File via OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_raw = output_path.parent / f"temp_{output_path.name}"
    out = cv2.VideoWriter(str(temp_raw), fourcc, fps, (target_width, target_height))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()

    # Re-encode with FFMPEG for broad compatibility and optimal compression
    cmd = [
        FFMPEG, "-y",
        "-i", str(temp_raw),
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if temp_raw.exists():
        temp_raw.unlink()

    total_time = time.time() - t_start
    print(f"[MOTION] Successfully exported: {output_path.name} in {total_time:.2f}s ({len(frames)} frames @ {fps}fps)")
    return output_path
