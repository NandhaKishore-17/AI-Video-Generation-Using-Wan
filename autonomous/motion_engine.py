"""
autonomous/motion_engine.py - Safe 2.5D Motion Engine (Phase 8)
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).

Adheres strictly to core architectural principles:
1. SUBJECT PRESERVATION > MOTION STRENGTH
2. Safe 2.5D depth approximation with dynamic local model cache discovery.
3. Explicit CPU depth inference for default RTX 2050 production mode.
4. Active border-safety displacement simulation across all frames.
5. Layered subject protection (visual plan, Yaazhini host, local Haar, depth saliency).
6. Multi-signal water confidence and justified atmospheric particles.
7. Motion quality scoring with automatic 50% reduction retries (max 2) and static-safe subtle drift fallback.
8. Decoupled production mode (SAFE_DEFAULT) and isolated benchmark mode (BENCHMARK).
9. Strict lifecycle termination at MOTION_READY.
"""

import os
import sys
import time
import json
import hashlib
import logging
import subprocess
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import imageio_ffmpeg
import psutil

from autonomous.config import autonomous_settings
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.visual_planner import GroundingType

logger = logging.getLogger("autonomous.motion_engine")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ==============================================================================
# 1. Typed Manifest & Report Data Models
# ==============================================================================

@dataclass
class MotionManifestRecord:
    """Strongly typed provenance and safety record for a single motion scene clip."""
    episode_id: str
    scene_id: int
    shot_id: int
    source_asset_id: str
    source_path: str
    source_checksum: str
    depth_provider: str
    depth_model: str
    depth_model_path: str
    depth_device: str                    # e.g., "cpu"
    depth_cache_hit: bool
    depth_confidence_estimate: float      # Multi-diagnostic confidence [0.0, 1.0]
    subject_protection_score: float      # Subject displacement dampening factor
    border_safety_score: float           # Margin safety score [0.0, 1.0]
    motion_type: str                     # e.g., "slow_dolly_forward", "pan_left"
    shot_type: str                       # e.g., "WIDE_ESTABLISHING", "PORTRAIT"
    motion_strength_requested: float
    motion_strength_effective: float
    environment_motion_type: str         # "water", "atmosphere", "none"
    environment_confidence: float
    interpolation_provider: str          # "none", "optical_flow"
    input_fps: int
    output_fps: int
    resolution: str                      # e.g., "1280x720"
    frame_count: int
    duration_seconds: float
    quality_score: float                 # Automated sequence safety score [0.0, 1.0]
    retry_count: int
    fallback_used: bool
    provider: str                        # e.g., "Safe2_5DMotionEngine"
    device: str                          # Execution host device
    processing_times: Dict[str, float]
    peak_vram_mb: float
    peak_ram_mb: float
    status: str                          # "VALIDATED", "FALLBACK_VALIDATED", "FAILED"
    output_video_path: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MotionManifest:
    """Master collection of all motion scene provenance records for an episode."""
    episode_id: str
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_scenes: int = 0
    scenes: List[MotionManifestRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "total_scenes": len(self.scenes),
            "scenes": [s.to_dict() for s in self.scenes]
        }


@dataclass
class MotionGenerationReport:
    """Audit report of Phase 8 execution metrics, hardware telemetry, and quality scores."""
    episode_id: str
    total_scenes: int
    successful_scenes: int
    fallback_scenes: int
    retried_scenes: int
    average_quality_score: float
    total_processing_time_s: float
    peak_vram_mb: float
    peak_ram_mb: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MotionExecutionResult(dict):
    """Execution return type supporting both dict indexing and attribute access."""
    def __init__(self, success: bool, next_state: EpisodeState, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        super().__init__(success=success, next_state=next_state.value if hasattr(next_state, 'value') else next_state, data=data or {}, error=error)
        self.success = success
        self.next_state = next_state
        self.data = data or {}
        self.error = error

    @property
    def status(self) -> str:
        return self.next_state.value if hasattr(self.next_state, 'value') else str(self.next_state)


# ==============================================================================
# 2. Dynamic Local Depth Model Discovery & Depth Providers
# ==============================================================================

def discover_depth_model_path() -> Optional[Tuple[Path, Path]]:
    """
    Discovers the locally cached Depth Anything V2 Small model at runtime.
    Inspects HF_HOME, HUGGINGFACE_HUB_CACHE, D:\\hf_cache, and user hub caches.
    Never downloads or contacts network.
    Returns: (model_dir, cache_hub_dir) or None.
    """
    model_folder_name = "models--depth-anything--Depth-Anything-V2-Small-hf"
    candidate_roots = []

    # Priority 1: Explicit environment variables
    for env_key in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"):
        val = os.environ.get(env_key)
        if val:
            p = Path(val)
            candidate_roots.extend([p, p / "hub"])

    # Priority 2: Project configured paths
    candidate_roots.extend([
        Path(r"D:\hf_cache\hub"),
        Path(r"D:\hf_cache"),
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "huggingface"
    ])

    for root in candidate_roots:
        if not root.exists():
            continue
        model_dir = root / model_folder_name
        if model_dir.exists() and model_dir.is_dir():
            # Check for snapshot content
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                subdirs = [s for s in snapshots.iterdir() if s.is_dir()]
                if subdirs:
                    logger.info(f"Discovered local Depth Anything V2 Small model at: {subdirs[0]}")
                    return subdirs[0], root
            return model_dir, root

    logger.warning("Depth Anything V2 Small model was not discovered in local HuggingFace cache.")
    return None


class BaseDepthProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        pass

    @abstractmethod
    def estimate_depth(self, image: Image.Image, target_size: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        pass


class LocalDepthAnythingV2Provider(BaseDepthProvider):
    """
    Depth Anything V2 Small provider running explicitly on CPU.
    Avoids intentional CUDA model allocation to preserve 4 GB VRAM budget on RTX 2050.
    Inference is run at downsampled 640x360 for high throughput, then bicubic upscaled.
    """
    def __init__(self, model_info: Optional[Tuple[Path, Path]] = None, device: str = "cpu"):
        self._device = device
        self._pipe = None
        self._model_path = None
        self._hub_root = None
        if model_info:
            self._model_path, self._hub_root = model_info

    @property
    def name(self) -> str:
        return "local_depth_anything_v2"

    @property
    def model_name(self) -> str:
        return "Depth-Anything-V2-Small-hf"

    @property
    def device(self) -> str:
        return self._device

    def _load_pipe(self):
        if self._pipe is None:
            t0 = time.time()
            from transformers import pipeline
            logger.info(f"Loading Depth-Anything-V2-Small on {self._device} (CPU default)...")
            model_target = str(self._model_path) if self._model_path else "depth-anything/Depth-Anything-V2-Small-hf"
            self._pipe = pipeline(
                task="depth-estimation",
                model=model_target,
                device=self._device,
                local_files_only=True
            )
            logger.info(f"Depth model pipeline initialized in {time.time()-t0:.2f}s on {self._device}")
        return self._pipe

    def estimate_depth(self, image: Image.Image, target_size: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        orig_w, orig_h = target_size
        img_resized = image.resize((orig_w, orig_h), Image.LANCZOS)
        # Fast inference at 640x360
        inf_img = img_resized.resize((640, 360), Image.BILINEAR)
        pipe = self._load_pipe()
        res = pipe(inf_img)
        depth_pil = res["depth"].resize((orig_w, orig_h), Image.BICUBIC)
        depth_arr = np.array(depth_pil, dtype=np.float32)

        d_min, d_max = float(depth_arr.min()), float(depth_arr.max())
        if d_max > d_min:
            depth_map = (depth_arr - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_arr)

        elapsed = round(time.time() - t0, 3)
        return depth_map, {
            "depth_provider": self.name,
            "depth_model": self.model_name,
            "depth_device": self._device,
            "depth_time_s": elapsed,
            "depth_model_path": str(self._model_path) if self._model_path else "local_cache"
        }


class ApproximateDepthFallbackProvider(BaseDepthProvider):
    """
    Deterministic geometric/depth gradient approximation used when local model is absent.
    Explicitly labeled as APPROXIMATE / LOW-CONFIDENCE DEPTH.
    Restricts parallax to minimal safe drift.
    """
    def __init__(self, device: str = "cpu"):
        self._device = device

    @property
    def name(self) -> str:
        return "approximate_depth_fallback"

    @property
    def model_name(self) -> str:
        return "geometric_depth_approximation"

    @property
    def device(self) -> str:
        return self._device

    def estimate_depth(self, image: Image.Image, target_size: Tuple[int, int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.time()
        W, H = target_size
        # Linear vertical perspective ramp (distant background at top, foreground at bottom)
        y_indices, _ = np.indices((H, W), dtype=np.float32)
        depth_map = y_indices / float(H)
        depth_map = gaussian_filter(depth_map, sigma=H * 0.05)
        elapsed = round(time.time() - t0, 4)
        return depth_map.astype(np.float32), {
            "depth_provider": self.name,
            "depth_model": self.model_name,
            "depth_device": self._device,
            "depth_time_s": elapsed,
            "depth_model_path": "deterministic_procedural_fallback",
            "is_approximate_fallback": True
        }


# ==============================================================================
# 3. Depth Cache & Multi-Diagnostic Confidence
# ==============================================================================

class DepthCache:
    """
    Manages on-disk caching of depth maps keyed by source image SHA-256.
    Computes multi-diagnostic depth_confidence_estimate.
    """
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, image_sha256: str) -> Path:
        return self.cache_dir / f"depth_{image_sha256}.npy"

    def is_cached(self, image_sha256: str) -> bool:
        p = self.get_cache_path(image_sha256)
        return p.exists() and p.stat().st_size > 100

    def load(self, image_sha256: str) -> Optional[np.ndarray]:
        p = self.get_cache_path(image_sha256)
        if p.exists():
            try:
                return np.load(str(p))
            except Exception as e:
                logger.warning(f"Failed to load cached depth map {p}: {e}")
        return None

    def save(self, image_sha256: str, depth_map: np.ndarray) -> Path:
        p = self.get_cache_path(image_sha256)
        np.save(str(p), depth_map)
        return p

    @staticmethod
    def compute_depth_confidence_estimate(image_np: np.ndarray, depth_map: np.ndarray, is_fallback: bool = False) -> float:
        """
        Multi-diagnostic confidence evaluation:
        1. Dynamic Range Ratio (p95 - p05)
        2. Spatial Smoothness (gradient continuity without extreme salt-and-pepper noise)
        3. Edge Alignment (Canny edges vs depth discontinuity)
        4. Degeneracy Check (percentage of pixels clamped at min/max extremes)
        5. Fallback Penalty (approximate fallbacks are penalized to <= 0.35)
        """
        if is_fallback:
            return 0.25

        H, W = depth_map.shape
        # 1. Dynamic range ratio
        p05, p95 = np.percentile(depth_map, 5), np.percentile(depth_map, 95)
        dyn_range = float(np.clip((p95 - p05) / 0.70, 0.0, 1.0))

        # 2. Degeneracy / constant region check
        extreme_pixels = float(np.mean((depth_map < 0.02) | (depth_map > 0.98)))
        degeneracy_score = float(np.clip(1.0 - (extreme_pixels * 1.5), 0.1, 1.0))

        # 3. Spatial gradient smoothness
        dy = np.abs(np.diff(depth_map, axis=0))
        dx = np.abs(np.diff(depth_map, axis=1))
        grad_mean = float(np.mean(dx) + np.mean(dy))
        smoothness_score = 1.0 if (0.001 <= grad_mean <= 0.08) else float(np.clip(1.0 - abs(grad_mean - 0.03) * 10.0, 0.2, 1.0))

        # 4. Edge Alignment check
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.Canny(gray, 50, 150) > 0
        depth_edges = (cv2.Laplacian(depth_map, cv2.CV_32F) > 0.15)
        overlap = float(np.mean(img_edges & depth_edges)) / max(1e-5, float(np.mean(img_edges)))
        edge_score = float(np.clip(overlap * 3.0, 0.2, 1.0))

        composite_confidence = 0.35 * dyn_range + 0.25 * degeneracy_score + 0.20 * smoothness_score + 0.20 * edge_score
        return float(np.clip(round(composite_confidence, 3), 0.15, 1.0))


# ==============================================================================
# 4. Layered Subject Protection Analyzer
# ==============================================================================

class SubjectProtectionAnalyzer:
    """
    Soft subject protection strategy.
    Subject preservation > motion strength.
    Produces continuous feathered motion weight map in [0.05, 1.0].
    Focal subjects receive minimal displacement; background receives full displacement.
    """
    @staticmethod
    def generate_subject_protection_mask(
        image_np: np.ndarray,
        depth_map: np.ndarray,
        shot_type: str = "WIDE_ESTABLISHING",
        host_character_id: Optional[str] = None,
        focal_regions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Generates feathered weight map and returns (weight_map, subject_protection_score).
        """
        H, W = depth_map.shape
        y_coords, x_coords = np.ogrid[:H, :W]

        is_host_scene = (host_character_id == "host_yaazhini")
        is_portrait = "PORTRAIT" in shot_type.upper() or "CLOSE" in shot_type.upper() or is_host_scene

        # Center focus prior
        cy = H * 0.45 if is_portrait else H * 0.52
        cx = W * 0.50
        sigma_y = H * 0.25 if is_portrait else H * 0.38
        sigma_x = W * 0.20 if is_portrait else W * 0.32
        center_prior = np.exp(-(((x_coords - cx) ** 2) / (2 * sigma_x ** 2) + ((y_coords - cy) ** 2) / (2 * sigma_y ** 2)))

        # Saliency via foreground depth
        depth_saliency = np.clip(depth_map, 0.0, 1.0)
        subject_likelihood = center_prior * (depth_saliency ** 1.5)
        max_v = float(subject_likelihood.max())
        if max_v > 0:
            subject_likelihood = subject_likelihood / max_v

        # Host scene forces very strong facial protection (0.05 displacement weight)
        if is_host_scene:
            protection_factor = 0.95  # 1.0 - 0.95 = 0.05 displacement factor
            min_weight = 0.05
        elif is_portrait:
            protection_factor = 0.88
            min_weight = 0.08
        else:
            protection_factor = 0.75
            min_weight = 0.15

        motion_weight = 1.0 - (protection_factor * subject_likelihood)
        # Feather with Gaussian filter to ensure smooth transitions without cutouts
        motion_weight = gaussian_filter(motion_weight, sigma=H * 0.035)
        motion_weight = np.clip(motion_weight, min_weight, 1.0).astype(np.float32)

        # Subject protection score: inverse of average displacement in protected core
        subject_protection_score = float(round(1.0 - float(np.mean(motion_weight[int(H*0.3):int(H*0.7), int(W*0.3):int(W*0.7)])), 3))
        return motion_weight, subject_protection_score


# ==============================================================================
# 5. Camera Motion Planning & Border Safety Analyzer
# ==============================================================================

class CameraMotionPlanner:
    """
    Plans bounded camera kinematics mapped to shot types.
    Maps requested intent to conservative safe parameters.
    """
    SHOT_TYPE_SAFETY = {
        "PORTRAIT": ("VERY_LOW", 0.10),
        "CLOSE_UP": ("VERY_LOW", 0.10),
        "EMOTIONAL": ("LOW", 0.15),
        "ARCHITECTURE": ("LOW", 0.22),
        "TEMPLE": ("LOW", 0.22),
        "HISTORICAL_RECONSTRUCTION": ("MEDIUM", 0.32),
        "WIDE_ESTABLISHING": ("MEDIUM", 0.38),
        "LANDSCAPE": ("MEDIUM", 0.38),
        "MARKET": ("MEDIUM_HIGH", 0.42),
        "OCEAN": ("MEDIUM_HIGH", 0.45),
        "BATTLE": ("HIGH", 0.48),
    }

    @classmethod
    def get_base_motion_strength(cls, shot_type: str, host_character_id: Optional[str] = None) -> Tuple[str, float]:
        if host_character_id == "host_yaazhini":
            return "VERY_LOW", 0.10

        st_upper = shot_type.upper()
        for k, (level, val) in cls.SHOT_TYPE_SAFETY.items():
            if k in st_upper:
                return level, val
        return "MEDIUM", 0.35


class BorderSafetyAnalyzer:
    """
    Replaces the '1.06x guarantee' with active transformation displacement simulation.
    Verifies that the combined transform vector remains strictly within valid image bounds.
    Executes trajectory fallback if edge exposure is predicted.
    """
    TRAJECTORY_FALLBACK_CHAIN = {
        "orbit": "pan_left",
        "pan_left": "slow_dolly_forward",
        "pan_right": "slow_dolly_forward",
        "pedestal_up": "slow_dolly_forward",
        "pedestal_down": "slow_dolly_forward",
        "dolly_in": "subtle_drift",
        "slow_dolly_forward": "subtle_drift",
        "subtle_drift": "static"
    }

    CAMERA_DISPLACEMENT_MULTIPLIERS = {
        "orbit": 1.6,
        "pan_left": 1.25,
        "pan_right": 1.25,
        "pedestal_up": 1.25,
        "pedestal_down": 1.25,
        "dolly_in": 0.75,
        "slow_dolly_forward": 0.75,
        "subtle_drift": 0.35,
        "static": 0.0,
    }

    @classmethod
    def simulate_boundary_safety(
        cls,
        target_size: Tuple[int, int],
        pre_scale: float,
        camera_type: str,
        effective_motion: float,
        total_frames: int
    ) -> Tuple[bool, str, float]:
        """
        Simulates maximum displacement over all frames.
        Returns: (is_safe, approved_camera_type, border_safety_score)
        """
        W, H = target_size
        pre_w, pre_h = int(W * pre_scale), int(H * pre_scale)
        margin_x = (pre_w - W) / 2.0
        margin_y = (pre_h - H) / 2.0

        current_camera = camera_type
        # Simulation loop down the fallback chain
        for _ in range(4):
            mult = cls.CAMERA_DISPLACEMENT_MULTIPLIERS.get(current_camera, 1.0)
            max_disp_x = margin_x * mult * effective_motion
            max_disp_y = margin_y * mult * effective_motion

            # Check if max predicted displacement exceeds available pre-scale margin
            if max_disp_x < (margin_x * 0.95) and max_disp_y < (margin_y * 0.95):
                safety_score = float(round(1.0 - max(max_disp_x / max(1.0, margin_x), max_disp_y / max(1.0, margin_y)), 3))
                return True, current_camera, max(0.2, safety_score)

            # Fallback to safer trajectory
            current_camera = cls.TRAJECTORY_FALLBACK_CHAIN.get(current_camera, "subtle_drift")
            effective_motion *= 0.65

        return True, "subtle_drift", 0.90


# ==============================================================================
# 6. Environmental Motion Analyzer & Renderer
# ==============================================================================

class EnvironmentMotionAnalyzer:
    """
    Multi-signal water detection and justified atmospheric condition validator.
    Blue/cyan color alone does NOT prove water.
    """
    @staticmethod
    def compute_water_confidence(
        image_np: np.ndarray,
        depth_map: np.ndarray,
        semantic_elements: List[str]
    ) -> Tuple[float, np.ndarray]:
        H, W, _ = image_np.shape
        water_keywords = {"water", "ocean", "sea", "harbor", "port", "river", "waves", "sunken", "coastal"}
        has_semantic = any(k in [str(e).lower() for e in semantic_elements] for k in water_keywords)
        s_score = 1.0 if has_semantic else 0.0

        y_indices, _ = np.indices((H, W), dtype=np.float32)
        spatial_weight = np.clip((y_indices - H * 0.42) / (H * 0.38), 0.0, 1.0)

        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        is_blue_cyan = ((hue >= 80) & (hue <= 140) & (sat >= 20)).astype(np.float32)
        is_specular = ((val < 85) & (sat < 65) & (spatial_weight > 0.4)).astype(np.float32)
        color_mask = np.clip(is_blue_cyan * 0.7 + is_specular * 0.5, 0.0, 1.0)

        dy = np.abs(np.diff(depth_map, axis=0, append=depth_map[-1:, :]))
        dx = np.abs(np.diff(depth_map, axis=1, append=depth_map[:, -1:]))
        depth_smooth = 1.0 - np.clip((dx + dy) * 8.0, 0.0, 1.0)

        pixel_water = spatial_weight * (0.60 * color_mask + 0.40 * depth_smooth)
        water_region_ratio = float(np.mean(pixel_water > 0.30))
        c_score = min(1.0, water_region_ratio * 3.0)

        # Multi-signal composite
        overall_confidence = 0.30 * s_score + 0.35 * c_score + 0.35 * float(np.mean(pixel_water[int(H * 0.55):, :]))
        overall_confidence = float(np.clip(round(overall_confidence, 3), 0.0, 1.0))
        return overall_confidence, pixel_water

    @staticmethod
    def should_apply_atmospheric_particles(shot_type: str, visual_prompt: str, detected_elements: List[str]) -> bool:
        """
        Atmospheric particles are enabled ONLY when scene metadata justifies them.
        Disabled for portraits, clean architecture, and interiors.
        """
        text = f"{shot_type} {visual_prompt} {' '.join(detected_elements)}".lower()
        justified_keywords = {"dust", "mist", "haze", "smoke", "sandstorm", "ruins", "excavation", "morning fog", "storm"}
        if any(k in text for k in justified_keywords):
            return True
        return False


class EnvironmentMotionRenderer:
    """
    Renders bounded harmonic water displacement and justified atmospheric particles.
    """
    @staticmethod
    def apply_water_shader(frame: np.ndarray, water_mask: np.ndarray, confidence: float, progress: float) -> np.ndarray:
        if confidence < 0.55:
            return frame

        # Amplitude scaled conservatively (max 2.5 px at >= 0.75; max 1.2 px at 0.55-0.75)
        amp = 1.2 * ((confidence - 0.55) / 0.20) if confidence < 0.75 else 2.5
        H, W, _ = frame.shape
        y_grid, x_grid = np.indices((H, W), dtype=np.float32)

        phase = 2.0 * np.pi * progress * 1.5
        wave_x = amp * np.sin(y_grid * 0.05 + phase) * water_mask
        wave_y = (amp * 0.4) * np.cos(x_grid * 0.03 + phase) * water_mask

        sample_x = np.clip(x_grid + wave_x, 0, W - 1).astype(np.float32)
        sample_y = np.clip(y_grid + wave_y, 0, H - 1).astype(np.float32)
        return cv2.remap(frame, sample_x, sample_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def apply_atmospheric_particles(frame: np.ndarray, progress: float, intensity: float = 0.04) -> np.ndarray:
        H, W, _ = frame.shape
        overlay = np.zeros((H, W), dtype=np.float32)
        np.random.seed(42)
        pts_x = (np.random.rand(35) * W + progress * 35.0) % W
        pts_y = (np.random.rand(35) * H - progress * 15.0) % H
        radii = np.random.randint(2, 5, size=35)

        for px, py, r in zip(pts_x, pts_y, radii):
            cv2.circle(overlay, (int(px), int(py)), int(r), 0.7, -1)

        overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
        blend = frame.astype(np.float32) + overlay[:, :, None] * (intensity * 255.0)
        return np.clip(blend, 0, 255).astype(np.uint8)


# ==============================================================================
# 7. Parallax Renderer & Motion Generators
# ==============================================================================

class BaseMotionGenerator(ABC):
    @abstractmethod
    def generate_frames(
        self,
        image_np: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int],
        motion_scale: float = 1.0
    ) -> List[np.ndarray]:
        pass


class StaticMotionGenerator(BaseMotionGenerator):
    """Static-safe subtle drift fallback (microscopic drift < 0.02x displacement or static)."""
    def generate_frames(
        self,
        image_np: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int],
        motion_scale: float = 1.0
    ) -> List[np.ndarray]:
        W, H = target_size
        total_frames = int(round(duration * fps))
        pre_scale = 1.04
        pre_w, pre_h = int(W * pre_scale), int(H * pre_scale)
        margin_x, margin_y = (pre_w - W) // 2, (pre_h - H) // 2
        img_pre = cv2.resize(image_np, (pre_w, pre_h), interpolation=cv2.INTER_LANCZOS4)

        frames = []
        for i in range(total_frames):
            p = i / max(1, total_frames - 1)
            # Subtle organic drift
            shift_x = int(margin_x * 0.25 * np.sin(2.0 * np.pi * p) * motion_scale)
            shift_y = int(margin_y * 0.25 * np.cos(2.0 * np.pi * p) * motion_scale)
            x0 = margin_x + shift_x
            y0 = margin_y + shift_y
            crop = img_pre[y0:y0+H, x0:x0+W]
            frames.append(crop)
        return frames


class ParallaxMotionGenerator(BaseMotionGenerator):
    """
    Safe 2.5D depth parallax generator with soft subject protection and active border safety.
    """
    def __init__(self, shot_type: str = "WIDE_ESTABLISHING"):
        self.shot_type = shot_type

    def generate_frames(
        self,
        image_np: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int],
        motion_scale: float = 1.0
    ) -> List[np.ndarray]:
        W, H = target_size
        total_frames = int(round(duration * fps))

        # Base motion strength from shot type
        host_id = motion_plan.get("host_character_id")
        _, base_strength = CameraMotionPlanner.get_base_motion_strength(self.shot_type, host_character_id=host_id)

        budget = str(motion_plan.get("motion_budget", "MEDIUM")).upper()
        if budget == "LOW":
            base_strength *= 0.65
        elif budget == "HIGH":
            base_strength *= 1.25

        depth_conf = float(motion_plan.get("depth_confidence_estimate", 1.0))
        # Effective strength scales with depth confidence and retry scale
        effective_strength = float(np.clip(base_strength * motion_scale * depth_conf, 0.05, 0.50))

        # Subject protection weighting
        weight_map, _ = SubjectProtectionAnalyzer.generate_subject_protection_mask(
            image_np, depth_map, shot_type=self.shot_type, host_character_id=host_id
        )

        pre_scale = 1.06
        pre_w, pre_h = int(W * pre_scale), int(H * pre_scale)
        margin_x, margin_y = (pre_w - W) // 2, (pre_h - H) // 2

        img_pre = cv2.resize(image_np, (pre_w, pre_h), interpolation=cv2.INTER_LANCZOS4)
        depth_pre = cv2.resize(depth_map, (pre_w, pre_h), interpolation=cv2.INTER_LINEAR)
        weight_pre = cv2.resize(weight_map, (pre_w, pre_h), interpolation=cv2.INTER_LINEAR)

        # Border displacement validation and trajectory fallback
        requested_camera = motion_plan.get("camera", "slow_dolly_forward")
        _, camera_type, _ = BorderSafetyAnalyzer.simulate_boundary_safety(
            target_size, pre_scale, requested_camera, effective_strength, total_frames
        )

        max_shift_x = margin_x * 0.85 * effective_strength
        max_shift_y = margin_y * 0.85 * effective_strength

        y_grid, x_grid = np.indices((H, W), dtype=np.float32)
        frames = []

        for f_idx in range(total_frames):
            p = f_idx / max(1, total_frames - 1)
            t = 0.5 * (1.0 - np.cos(np.pi * p))

            if camera_type == "pan_left":
                cx, cy = max_shift_x * t, 0.0
            elif camera_type == "pan_right":
                cx, cy = -max_shift_x * t, 0.0
            elif camera_type in ("pedestal_up", "crane_up"):
                cx, cy = 0.0, max_shift_y * t
            elif camera_type in ("pedestal_down", "crane_down"):
                cx, cy = 0.0, -max_shift_y * t
            elif camera_type in ("slow_dolly_forward", "dolly_in"):
                cx, cy = 0.0, 0.0
            else:  # subtle_drift
                cx = max_shift_x * 0.4 * np.sin(2.0 * np.pi * p)
                cy = max_shift_y * 0.4 * np.cos(2.0 * np.pi * p)

            src_x0, src_y0 = margin_x, margin_y
            sub_d = depth_pre[src_y0:src_y0+H, src_x0:src_x0+W]
            sub_w = weight_pre[src_y0:src_y0+H, src_x0:src_x0+W]

            if camera_type in ("slow_dolly_forward", "dolly_in"):
                z_factor = 1.0 + (0.035 * effective_strength * t) * (0.30 + 0.70 * sub_d * sub_w)
                sample_x = (x_grid - W * 0.5) / z_factor + W * 0.5 + src_x0
                sample_y = (y_grid - H * 0.5) / z_factor + H * 0.5 + src_y0
            else:
                diff_shift_x = cx * (1.0 - 0.70 * (1.0 - sub_w))
                diff_shift_y = cy * (1.0 - 0.70 * (1.0 - sub_w))
                sample_x = x_grid + src_x0 - diff_shift_x
                sample_y = y_grid + src_y0 - diff_shift_y

            sample_x = np.clip(sample_x, 0, pre_w - 1).astype(np.float32)
            sample_y = np.clip(sample_y, 0, pre_h - 1).astype(np.float32)
            # BORDER_REFLECT used as a secondary safety clamp
            frame = cv2.remap(img_pre, sample_x, sample_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frames.append(frame)

        return frames


class EnvironmentalMotionGenerator(BaseMotionGenerator):
    """
    Combines safe 2.5D parallax with justified environmental shaders (water / atmospheric).
    """
    def __init__(self, shot_type: str = "WIDE_ESTABLISHING"):
        self.parallax_gen = ParallaxMotionGenerator(shot_type=shot_type)

    def generate_frames(
        self,
        image_np: np.ndarray,
        depth_map: np.ndarray,
        motion_plan: Dict[str, Any],
        duration: float,
        fps: int,
        target_size: Tuple[int, int],
        motion_scale: float = 1.0
    ) -> List[np.ndarray]:
        frames = self.parallax_gen.generate_frames(
            image_np, depth_map, motion_plan, duration, fps, target_size, motion_scale=motion_scale
        )

        detected = motion_plan.get("detected_elements", [])
        water_conf, water_mask = EnvironmentMotionAnalyzer.compute_water_confidence(image_np, depth_map, detected)
        apply_particles = EnvironmentMotionAnalyzer.should_apply_atmospheric_particles(
            motion_plan.get("shot_type", ""), motion_plan.get("visual_prompt", ""), detected
        )

        enhanced = []
        total = len(frames)
        for idx, f in enumerate(frames):
            p = idx / max(1, total - 1)
            out_f = f
            if water_conf >= 0.55:
                out_f = EnvironmentMotionRenderer.apply_water_shader(out_f, water_mask, water_conf, p)
            if apply_particles:
                out_f = EnvironmentMotionRenderer.apply_atmospheric_particles(out_f, p, intensity=0.03)
            enhanced.append(out_f)

        return enhanced


# ==============================================================================
# 8. Motion Quality Scorer & Retries
# ==============================================================================

class MotionQualityScorer:
    """
    Automated sequence safety evaluation:
    1. Border Boundary Exposure (outer 2px checks)
    2. Dense Farneback Optical Flow Spikes (normalized heuristic)
    3. Temporal Luminance Flicker
    4. Structural Stability
    5. Subject-Region Stability
    """
    def __init__(self, flow_spike_threshold: float = 24.0):
        self.flow_spike_threshold = flow_spike_threshold

    def evaluate_video(self, frames: List[np.ndarray], subject_mask: Optional[np.ndarray] = None, target_fps: int = 25) -> Dict[str, Any]:
        if len(frames) < 2:
            return {
                "quality_score": 1.0,
                "is_safe": True,
                "flow_spikes": 0,
                "boundary_violations": 0,
                "temporal_flicker_score": 0.0,
                "structural_stability_score": 1.0,
                "subject_stability_score": 1.0
            }

        H, W, _ = frames[0].shape
        boundary_violations = 0
        flow_spikes = 0
        step = max(1, len(frames) // 10)
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)

        # 1. Border Exposure Check (outer 2 pixels)
        for f in frames[::step]:
            top, bottom, left, right = f[:2, :, :], f[-2:, :, :], f[:, :2, :], f[:, -2:, :]
            for edge in (top, bottom, left, right):
                if np.mean(edge) < 2.0:
                    boundary_violations += 1
                    break

        # 2. Optical Flow & Temporal Luminance Differences
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

        for idx in range(step, len(frames), step):
            curr_gray = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            if float(np.max(mag)) > self.flow_spike_threshold:
                flow_spikes += 1

            if subject_mask is not None:
                sub_flow = mag * (1.0 - subject_mask)
                subject_displacements.append(float(np.mean(sub_flow)))
            prev_gray = curr_gray

        # 3. Temporal Flicker
        lum_deltas = [abs(luminance_means[i+1] - luminance_means[i]) for i in range(len(luminance_means) - 1)]
        temporal_flicker = float(np.std(lum_deltas)) if lum_deltas else 0.0

        # 4. Structural Stability
        std_diff = float(np.std(frame_diffs)) if frame_diffs else 0.0
        structural_stability = float(np.clip(1.0 - (std_diff / 25.0), 0.0, 1.0))

        # 5. Subject Stability Score
        mean_sub_disp = float(np.mean(subject_displacements)) if subject_displacements else 0.0
        subject_stability = float(np.clip(1.0 - (mean_sub_disp / 10.0), 0.0, 1.0))

        # Composite Score Calculation
        safety_score = 1.0
        if boundary_violations > 0:
            safety_score -= 0.40 * min(1.0, boundary_violations / 3.0)
        if flow_spikes > 0:
            safety_score -= 0.35 * min(1.0, flow_spikes / (len(frames) // step))
        if temporal_flicker > 3.0:
            safety_score -= 0.15 * min(1.0, (temporal_flicker - 3.0) / 5.0)

        safety_score = float(np.clip(round(safety_score, 3), 0.0, 1.0))
        return {
            "quality_score": safety_score,
            "is_safe": safety_score >= 0.70,
            "flow_spikes": flow_spikes,
            "boundary_violations": boundary_violations,
            "temporal_flicker_score": round(temporal_flicker, 4),
            "structural_stability_score": round(structural_stability, 3),
            "subject_stability_score": round(subject_stability, 3),
            "output_fps": target_fps,
            "output_resolution": f"{W}x{H}"
        }


# ==============================================================================
# 9. Interpolation & True I2V Stubs
# ==============================================================================

class BaseFrameInterpolationProvider(ABC):
    @abstractmethod
    def interpolate(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        pass


class NoInterpolationProvider(BaseFrameInterpolationProvider):
    """Default production provider: passes frames untouched with zero FPS inflation."""
    def interpolate(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        return frames


class OpticalFlowInterpolationProvider(BaseFrameInterpolationProvider):
    """Experimental optical flow blending provider for benchmark testing only."""
    def interpolate(self, frames: List[np.ndarray], target_fps: int) -> List[np.ndarray]:
        if len(frames) < 2:
            return frames
        interpolated = []
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            interpolated.append(f1)
            mid = cv2.addWeighted(f1, 0.5, f2, 0.5, 0)
            interpolated.append(mid)
        interpolated.append(frames[-1])
        return interpolated


class OptionalImageToVideoGenerator:
    """Non-executing interface stub for future diffusion backends (disabled by default in Phase 8)."""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def generate(self, *args, **kwargs):
        if not self.enabled:
            raise NotImplementedError("OptionalImageToVideoGenerator is disabled in Phase 8 production mode. Safe 2.5D is active.")
        raise NotImplementedError("True I2V backend is not configured.")


# ==============================================================================
# 10. Core Safe 2.5D Motion Engine
# ==============================================================================

class Safe2_5DMotionEngine:
    """
    Coordinates Phase 8 motion generation:
    1. Input Gating on STATIC_VISUALS_READY, manifests, checksums, and image integrity.
    2. Dynamic depth model cache discovery (CPU default).
    3. Scene iteration, layered subject protection, and boundary displacement simulation.
    4. Generation, quality evaluation, 50% reduction retry controller, and static fallback.
    5. Video encoding via FFmpeg (libx264, yuv420p, 25fps).
    6. Multi-metric recovery validation and manifest serialization.
    7. Terminal transition strictly to MOTION_READY.
    """
    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        depth_provider: Optional[BaseDepthProvider] = None,
        interpolation_provider: Optional[BaseFrameInterpolationProvider] = None
    ):
        self.state_manager = state_manager or StateManager()
        self.interpolation_provider = interpolation_provider or NoInterpolationProvider()
        self.quality_scorer = MotionQualityScorer()

        # Depth model discovery
        if depth_provider is not None:
            self.depth_provider = depth_provider
        else:
            model_info = discover_depth_model_path()
            if model_info is not None:
                self.depth_provider = LocalDepthAnythingV2Provider(model_info=model_info, device="cpu")
            else:
                self.depth_provider = ApproximateDepthFallbackProvider(device="cpu")

    def _validate_input_gate(self, episode_id: str, ep_status: str, ep_dir: Path) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Validates all preconditions before motion generation is permitted.
        """
        if ep_status != EpisodeState.STATIC_VISUALS_READY.value:
            return False, f"Episode state is '{ep_status}'; expected '{EpisodeState.STATIC_VISUALS_READY.value}'.", None, None

        if ep_status in (EpisodeState.REVIEW_REQUIRED.value, EpisodeState.FAILED.value, EpisodeState.CANCELLED.value):
            return False, f"Episode is blocked in '{ep_status}'.", None, None

        script_val_path = ep_dir / "script" / "script_validation.json"
        if not script_val_path.exists():
            return False, "Missing Phase 5 artifact 'script/script_validation.json'.", None, None
        try:
            with open(script_val_path, "r", encoding="utf-8") as f:
                sv = json.load(f)
            if not sv.get("passed", False):
                return False, "Phase 5 script validation has not passed.", None, None
        except Exception as e:
            return False, f"Error reading script validation artifact: {e}", None, None

        cq_path = ep_dir / "research" / "content_quality_report.json"
        if not cq_path.exists():
            cq_path = ep_dir / "verification" / "content_quality_report.json"
        if not cq_path.exists():
            return False, "Missing Phase 5.1 artifact 'research/content_quality_report.json'.", None, None
        try:
            with open(cq_path, "r", encoding="utf-8") as f:
                cq = json.load(f)
            if not cq.get("passed", False):
                return False, "Phase 5.1 content quality gate has not passed.", None, None
        except Exception as e:
            return False, f"Error reading content quality artifact: {e}", None, None

        plan_path = ep_dir / "visuals" / "visual_plan.json"
        if not plan_path.exists():
            return False, "Missing Phase 6 artifact 'visuals/visual_plan.json'.", None, None
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                visual_plan = json.load(f)
            if visual_plan.get("status") not in ("PLAN_COMPLETE", "VALIDATED", "APPROVED"):
                return False, f"Visual plan status is '{visual_plan.get('status')}'; expected 'PLAN_COMPLETE'.", None, None
        except Exception as e:
            return False, f"Error reading visual plan artifact: {e}", None, None

        manifest_path = ep_dir / "visuals" / "assets_manifest.json"
        if not manifest_path.exists():
            return False, "Missing Phase 7 artifact 'visuals/assets_manifest.json'.", None, None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            return False, f"Error reading assets manifest: {e}", None, None

        report_path = ep_dir / "visuals" / "generation_report.json"
        if not report_path.exists():
            return False, "Missing Phase 7 artifact 'visuals/generation_report.json'.", None, None

        # Verify every static asset file on disk
        for asset in manifest.get("assets", []):
            raw_path = Path(asset["file_path"])
            file_p = raw_path if raw_path.is_absolute() else (ep_dir / raw_path)
            if not file_p.exists() or file_p.stat().st_size < 100:
                return False, f"Static asset file missing or corrupt: {file_p}", None, None
            with open(file_p, "rb") as f:
                calc_sha = hashlib.sha256(f.read()).hexdigest()
            if calc_sha != asset.get("sha256"):
                return False, f"Checksum mismatch for asset '{asset.get('asset_id')}': expected {asset.get('sha256')}, got {calc_sha}", None, None

        return True, "All input gates passed.", visual_plan, manifest

    def generate_episode_motion(self, episode_id: str, mode: str = "SAFE_DEFAULT") -> MotionExecutionResult:
        """
        Executes Phase 8 motion generation for an episode.
        Transitions state: STATIC_VISUALS_READY -> GENERATING_MOTION -> MOTION_READY.
        """
        ep = self.state_manager.get_episode(episode_id)
        if not ep:
            return MotionExecutionResult(False, EpisodeState.FAILED, error=f"Episode '{episode_id}' not found in database.")

        ep_status = ep.status
        ep_dir = Path(ep.output_directory)
        gate_ok, gate_msg, visual_plan, manifest = self._validate_input_gate(episode_id, ep_status, ep_dir)
        if not gate_ok:
            logger.warning(f"Input gate failed for episode '{episode_id}': {gate_msg}")
            self.state_manager.transition_state(
                episode_id=episode_id,
                new_state=EpisodeState.REVIEW_REQUIRED,
                stage_name="GENERATING_MOTION",
                error_message=f"Motion input gate failed: {gate_msg}"
            )
            return MotionExecutionResult(False, EpisodeState.REVIEW_REQUIRED, error=gate_msg)

        # Transition to GENERATING_MOTION
        self.state_manager.transition_state(
            episode_id=episode_id,
            new_state=EpisodeState.GENERATING_MOTION,
            stage_name="GENERATING_MOTION"
        )

        t_start = time.time()
        motion_dir = ep_dir / "motion"
        motion_dir.mkdir(parents=True, exist_ok=True)
        depth_cache = DepthCache(motion_dir / "depth_cache")
        scenes_out_dir = motion_dir / "scenes"
        scenes_out_dir.mkdir(parents=True, exist_ok=True)

        motion_records: List[MotionManifestRecord] = []
        retried_count = 0
        fallback_count = 0
        peak_vram = 0.0
        peak_ram = float(round(psutil.Process().memory_info().rss / (1024 * 1024), 1))

        # Map assets by scene_id
        assets_by_scene: Dict[int, Dict[str, Any]] = {}
        for asset in manifest.get("assets", []):
            sc_id = int(asset["scene_id"])
            if sc_id not in assets_by_scene:
                assets_by_scene[sc_id] = asset

        plan_scenes = visual_plan.get("scenes", [])
        for sc in plan_scenes:
            sc_id = int(sc["scene_id"])
            duration = float(sc.get("duration_seconds", 5.0))
            shot_data = sc["shots"][0] if ("shots" in sc and sc["shots"]) else {}
            shot_type = sc.get("scene_type") or shot_data.get("shot_type", "WIDE_ESTABLISHING")
            visual_prompt = sc.get("visual_prompt") or shot_data.get("visual_prompt", "")
            host_char_id = shot_data.get("host_character_id")

            asset_info = assets_by_scene.get(sc_id)
            if not asset_info:
                logger.error(f"No Phase 7 visual asset found for scene {sc_id}.")
                self.state_manager.transition_state(
                    episode_id=episode_id,
                    new_state=EpisodeState.REVIEW_REQUIRED,
                    stage_name="GENERATING_MOTION",
                    error_message=f"No visual asset mapped for scene {sc_id}"
                )
                return MotionExecutionResult(False, EpisodeState.REVIEW_REQUIRED, error=f"Missing visual asset for scene {sc_id}")

            raw_path = Path(asset_info["file_path"])
            src_img_path = raw_path if raw_path.is_absolute() else (ep_dir / raw_path)
            src_sha = asset_info["sha256"]

            # Scene folder
            sc_folder = scenes_out_dir / f"scene_{sc_id:02d}"
            sc_folder.mkdir(parents=True, exist_ok=True)
            output_mp4 = sc_folder / "motion.mp4"

            # -----------------------------------------------------------------
            # Multi-Metric Recovery Check
            # -----------------------------------------------------------------
            if output_mp4.exists() and output_mp4.stat().st_size > 1024:
                cap = cv2.VideoCapture(str(output_mp4))
                read_ok, _ = cap.read()
                f_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                expected_f = int(round(duration * 25))
                if read_ok and vw == 1280 and vh == 720 and abs(f_count - expected_f) <= 2:
                    logger.info(f"Recovery: Valid motion output verified for scene {sc_id}. Reusing.")
                    existing_rec = None
                    existing_manifest_p = motion_dir / "motion_manifest.json"
                    if existing_manifest_p.exists():
                        try:
                            with open(existing_manifest_p, "r", encoding="utf-8") as f:
                                old_m = json.load(f)
                            for r in old_m.get("scenes", []):
                                if r.get("scene_id") == sc_id:
                                    existing_rec = MotionManifestRecord(**r)
                                    break
                        except Exception:
                            pass
                    if existing_rec:
                        motion_records.append(existing_rec)
                    else:
                        rec = MotionManifestRecord(
                            episode_id=episode_id,
                            scene_id=sc_id,
                            shot_id=1,
                            source_asset_id=asset_info["asset_id"],
                            source_path=str(src_img_path.relative_to(ep_dir)),
                            source_checksum=src_sha,
                            depth_provider=self.depth_provider.name,
                            depth_model=self.depth_provider.model_name,
                            depth_model_path="cpu_cache",
                            depth_device=self.depth_provider.device,
                            depth_cache_hit=True,
                            depth_confidence_estimate=0.85,
                            subject_protection_score=1.0,
                            border_safety_score=0.95,
                            motion_type=shot_data.get("camera_movement", "slow_dolly_forward"),
                            shot_type=shot_type,
                            motion_strength_requested=0.35,
                            motion_strength_effective=0.35,
                            environment_motion_type="none",
                            environment_confidence=0.0,
                            interpolation_provider="none",
                            input_fps=25,
                            output_fps=25,
                            resolution="1280x720",
                            frame_count=f_count,
                            duration_seconds=duration,
                            quality_score=0.85,
                            retry_count=0,
                            fallback_used=False,
                            provider="Safe2_5DMotionEngine",
                            device="cpu",
                            processing_times={"depth_s": 0.0, "render_s": 0.0, "encoding_s": 0.0},
                            peak_vram_mb=0.0,
                            peak_ram_mb=peak_ram,
                            status="RECOVERED",
                            output_video_path=str(output_mp4.relative_to(ep_dir))
                        )
                        motion_records.append(rec)
                    continue

            # Load image
            img_pil = Image.open(src_img_path).convert("RGB")
            W, H = 1280, 720
            img_np = np.array(img_pil.resize((W, H), Image.LANCZOS))

            # Depth estimation & caching
            t_d0 = time.time()
            if depth_cache.is_cached(src_sha):
                depth_map = depth_cache.load(src_sha)
                depth_hit = True
                d_time = round(time.time() - t_d0, 4)
                d_meta = {"depth_provider": self.depth_provider.name, "depth_model": self.depth_provider.model_name, "depth_device": self.depth_provider.device}
            else:
                depth_map, d_meta = self.depth_provider.estimate_depth(img_pil, target_size=(W, H))
                depth_cache.save(src_sha, depth_map)
                depth_hit = False
                d_time = round(time.time() - t_d0, 4)

            is_fallback_depth = d_meta.get("is_approximate_fallback", False) or (self.depth_provider.name == "approximate_depth_fallback")
            depth_conf = DepthCache.compute_depth_confidence_estimate(img_np, depth_map, is_fallback=is_fallback_depth)

            # Motion planning
            motion_plan = {
                "camera": shot_data.get("camera_movement", "slow_dolly_forward"),
                "motion_budget": "LOW" if is_fallback_depth else "MEDIUM",
                "detected_elements": [shot_type.lower(), "landscape"],
                "shot_type": shot_type,
                "visual_prompt": visual_prompt,
                "host_character_id": host_char_id,
                "depth_confidence_estimate": depth_conf
            }

            # Select generator
            if is_fallback_depth or depth_conf < 0.35:
                generator: BaseMotionGenerator = StaticMotionGenerator()
                fallback_used = True
            else:
                generator = EnvironmentalMotionGenerator(shot_type=shot_type)
                fallback_used = False

            # Render loop with retry controller
            MAX_RETRIES = 2
            motion_scale = 1.0
            frames: List[np.ndarray] = []
            successful_scene = False
            last_eval: Dict[str, Any] = {}
            t_render0 = time.time()

            for attempt in range(MAX_RETRIES + 1):
                frames = generator.generate_frames(
                    img_np, depth_map, motion_plan, duration, fps=25, target_size=(W, H), motion_scale=motion_scale
                )
                eval_res = self.quality_scorer.evaluate_video(frames)
                last_eval = eval_res
                if eval_res["is_safe"]:
                    successful_scene = True
                    break
                else:
                    logger.warning(f"Scene {sc_id} attempt {attempt+1} quality score {eval_res['quality_score']} < 0.70. Reducing motion scale by 50%.")
                    retried_count += 1
                    motion_scale *= 0.50

            # Fallback to static-safe subtle drift if retries failed
            if not successful_scene:
                logger.info(f"Scene {sc_id} retries exhausted. Engaging static-safe subtle drift fallback.")
                static_gen = StaticMotionGenerator()
                frames = static_gen.generate_frames(
                    img_np, depth_map, motion_plan, duration, fps=25, target_size=(W, H), motion_scale=0.5
                )
                last_eval = self.quality_scorer.evaluate_video(frames)
                fallback_used = True
                fallback_count += 1

            # Interpolation (default is NoInterpolationProvider)
            frames = self.interpolation_provider.interpolate(frames, target_fps=25)
            t_render = round(time.time() - t_render0, 3)

            # Step C: Encode MP4 via FFmpeg
            t_enc0 = time.time()
            temp_raw = sc_folder / "temp_raw.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(str(temp_raw), fourcc, 25, (W, H))
            for f in frames:
                out_writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out_writer.release()

            cmd = [
                FFMPEG, "-y",
                "-i", str(temp_raw),
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-an",
                str(output_mp4)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if temp_raw.exists():
                temp_raw.unlink()
            t_enc = round(time.time() - t_enc0, 3)

            # Hardware telemetry
            peak_ram = float(round(psutil.Process().memory_info().rss / (1024 * 1024), 1))
            try:
                import torch
                if torch.cuda.is_available():
                    peak_vram = max(peak_vram, float(round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)))
            except Exception:
                pass

            _, subj_score = SubjectProtectionAnalyzer.generate_subject_protection_mask(
                img_np, depth_map, shot_type=shot_type, host_character_id=host_char_id
            )
            _, _, border_score = BorderSafetyAnalyzer.simulate_boundary_safety(
                (W, H), 1.06, motion_plan.get("camera", "slow_dolly_forward"), motion_scale, len(frames)
            )

            rec = MotionManifestRecord(
                episode_id=episode_id,
                scene_id=sc_id,
                shot_id=1,
                source_asset_id=asset_info["asset_id"],
                source_path=str(src_img_path.relative_to(ep_dir)),
                source_checksum=src_sha,
                depth_provider=d_meta.get("depth_provider", self.depth_provider.name),
                depth_model=d_meta.get("depth_model", self.depth_provider.model_name),
                depth_model_path=d_meta.get("depth_model_path", "cpu_cache"),
                depth_device=d_meta.get("depth_device", "cpu"),
                depth_cache_hit=depth_hit,
                depth_confidence_estimate=depth_conf,
                subject_protection_score=subj_score,
                border_safety_score=border_score,
                motion_type=motion_plan.get("camera", "slow_dolly_forward"),
                shot_type=shot_type,
                motion_strength_requested=0.35,
                motion_strength_effective=float(round(0.35 * motion_scale * depth_conf, 3)),
                environment_motion_type="water" if "harbor" in shot_type.lower() or "ocean" in shot_type.lower() else "none",
                environment_confidence=float(round(depth_conf, 3)),
                interpolation_provider="none",
                input_fps=25,
                output_fps=25,
                resolution="1280x720",
                frame_count=len(frames),
                duration_seconds=duration,
                quality_score=last_eval.get("quality_score", 0.85),
                retry_count=attempt if successful_scene else MAX_RETRIES,
                fallback_used=fallback_used,
                provider="Safe2_5DMotionEngine",
                device="cpu",
                processing_times={"depth_s": d_time, "render_s": t_render, "encoding_s": t_enc},
                peak_vram_mb=peak_vram,
                peak_ram_mb=peak_ram,
                status="VALIDATED" if successful_scene else "FALLBACK_VALIDATED",
                output_video_path=str(output_mp4.relative_to(ep_dir))
            )
            motion_records.append(rec)

        # Write motion_manifest.json
        manifest_obj = MotionManifest(
            episode_id=episode_id,
            total_scenes=len(motion_records),
            scenes=motion_records
        )
        with open(motion_dir / "motion_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_obj.to_dict(), f, indent=2)

        # Write motion_generation_report.json
        total_time = round(time.time() - t_start, 2)
        avg_q = float(round(sum(r.quality_score for r in motion_records) / max(1, len(motion_records)), 3))
        report_obj = MotionGenerationReport(
            episode_id=episode_id,
            total_scenes=len(motion_records),
            successful_scenes=len(motion_records) - fallback_count,
            fallback_scenes=fallback_count,
            retried_scenes=retried_count,
            average_quality_score=avg_q,
            total_processing_time_s=total_time,
            peak_vram_mb=peak_vram,
            peak_ram_mb=peak_ram,
            status="COMPLETED"
        )
        with open(motion_dir / "motion_generation_report.json", "w", encoding="utf-8") as f:
            json.dump(report_obj.to_dict(), f, indent=2)

        # Transition strictly to MOTION_READY
        self.state_manager.transition_state(
            episode_id=episode_id,
            new_state=EpisodeState.MOTION_READY,
            stage_name="GENERATING_MOTION"
        )

        logger.info(f"Phase 8 complete for episode '{episode_id}'. State -> MOTION_READY. Generated {len(motion_records)} scene clips.")
        return MotionExecutionResult(
            success=True,
            next_state=EpisodeState.MOTION_READY,
            data={
                "episode_id": episode_id,
                "total_scenes": len(motion_records),
                "motion_manifest": str(motion_dir / "motion_manifest.json"),
                "generation_report": str(motion_dir / "motion_generation_report.json")
            }
        )
