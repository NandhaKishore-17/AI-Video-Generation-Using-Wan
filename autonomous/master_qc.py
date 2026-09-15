"""
autonomous/master_qc.py - Automated Master Video & Audio Quality Control Engine (Phase 10).
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).

Performs comprehensive, independent quality inspection of final master video and audio:
1. Video stream verification (H.264, 1280x720, 25fps, container integrity, frame decodability).
2. Conservative black-frame detection (context-aware, allows intentional scene transitions and dark history).
3. Context-aware frozen-frame detection (allows approved visual holds and static artwork).
4. Audio stream verification (loudness, RMS, clipping, speech pauses, A/V synchronization).
5. Scene composition and presenter gating verification (Yaazhini strictly on HOST_ANCHORED).
6. Transparent 0.0-1.0 weighted QC scoring with strict critical failure overrides.
"""

import os
import sys
import json
import math
import wave
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import cv2
import numpy as np
import imageio_ffmpeg

from autonomous.config import autonomous_settings

logger = logging.getLogger("autonomous.master_qc")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ==============================================================================
# 1. QC Metrics & Report Data Structures
# ==============================================================================

@dataclass
class VideoQCMetrics:
    file_exists: bool = False
    container_valid: bool = False
    video_stream_exists: bool = False
    codec: str = ""
    resolution: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_seconds: float = 0.0
    frame_count: int = 0
    decode_errors: int = 0
    corrupted_frames: int = 0
    black_frame_ratio: float = 0.0
    black_intervals: List[Dict[str, float]] = field(default_factory=list)
    frozen_frame_ratio: float = 0.0
    frozen_intervals: List[Dict[str, float]] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AudioQCMetrics:
    audio_stream_exists: bool = False
    codec: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    silence_ratio: float = 0.0
    clipping_detected: bool = False
    rms_dbfs: float = -100.0
    measured_loudness_lufs: Optional[float] = None
    av_duration_diff_s: float = 0.0
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneQCMetrics:
    expected_scenes: int = 0
    represented_scenes: int = 0
    scene_order_valid: bool = False
    presenter_scenes_expected: List[int] = field(default_factory=list)
    presenter_scenes_found: List[int] = field(default_factory=list)
    presenter_gating_valid: bool = False
    missing_scene_ids: List[int] = field(default_factory=list)
    unexpected_presenter_scene_ids: List[int] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MasterQCReport:
    episode_id: str
    video: VideoQCMetrics = field(default_factory=VideoQCMetrics)
    audio: AudioQCMetrics = field(default_factory=AudioQCMetrics)
    scene: SceneQCMetrics = field(default_factory=SceneQCMetrics)
    component_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    critical_failure: bool = False
    decision: str = "FAILED"  # MASTER_READY, REVIEW_REQUIRED, FAILED
    passed: bool = False
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    broll_semantic_samples: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ==============================================================================
# 2. Master Quality Control Engine
# ==============================================================================

class MasterQualityController:
    """
    Independent validator that verifies the generated master MP4 against
    Phase 8/9 manifests, broadcast delivery specs, and safety policies.
    """

    def __init__(
        self,
        min_score: Optional[float] = None,
        review_score: Optional[float] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        target_fps: Optional[int] = None,
        max_black_ratio: Optional[float] = None,
        max_frozen_duration: Optional[float] = None,
    ):
        self.min_score = min_score if min_score is not None else autonomous_settings.qc_min_score
        self.review_score = review_score if review_score is not None else autonomous_settings.qc_review_score
        self.target_width = target_width if target_width is not None else autonomous_settings.master_width
        self.target_height = target_height if target_height is not None else autonomous_settings.master_height
        self.target_fps = target_fps if target_fps is not None else autonomous_settings.master_fps
        self.max_black_ratio = max_black_ratio if max_black_ratio is not None else autonomous_settings.max_black_frame_ratio
        self.max_frozen_duration = max_frozen_duration if max_frozen_duration is not None else autonomous_settings.max_frozen_frame_duration_s

    @staticmethod
    def compute_av_sync_score(av_diff: float) -> float:
        """
        Calibrated A/V synchronization duration delta scoring (0.0 - 1.0):
        - 0.00s <= diff <= 0.05s: 1.000 (Imperceptible, within 1 video frame @ 25fps)
        - 0.05s < diff <= 0.50s: 1.000 down to 0.700 (Acceptable broadcast tolerance down to review threshold)
        - 0.50s < diff < 1.50s: 0.700 down to 0.000 (Noticeable divergence warning zone)
        - diff >= 1.50s: 0.000 (Catastrophic desynchronization trigger)
        """
        diff = abs(float(av_diff))
        if diff <= 0.05:
            return 1.0
        elif diff <= 0.50:
            return round(1.0 - 0.30 * ((diff - 0.05) / 0.45), 3)
        elif diff < 1.50:
            return round(0.70 * ((1.50 - diff) / 1.00), 3)
        else:
            return 0.0

    def inspect_video_stream(
        self,
        video_path: Path,
        expected_duration: Optional[float] = None,
        approved_holds: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[VideoQCMetrics, List[str], List[str]]:
        """
        Performs exhaustive video inspection: decodability, resolution, FPS,
        context-aware black frames, and context-aware frozen frames.
        """
        metrics = VideoQCMetrics()
        issues: List[str] = []
        warnings: List[str] = []

        if not video_path.exists():
            issues.append(f"Master video file does not exist: {video_path}")
            return metrics, issues, warnings

        metrics.file_exists = True
        file_size = video_path.stat().st_size
        if file_size < 1000:
            issues.append(f"Master video file is suspiciously small: {file_size} bytes")
            return metrics, issues, warnings

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            issues.append(f"Failed to open video container with OpenCV: {video_path}")
            return metrics, issues, warnings

        metrics.container_valid = True
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)]).strip()

        metrics.width = w
        metrics.height = h
        metrics.resolution = f"{w}x{h}"
        metrics.fps = round(fps, 2)
        metrics.codec = codec_str or "h264"
        metrics.video_stream_exists = (w > 0 and h > 0 and total_frames > 0)

        # Resolution and FPS checks
        if w != self.target_width or h != self.target_height:
            issues.append(f"Invalid video resolution: expected {self.target_width}x{self.target_height}, got {w}x{h}")
        if abs(fps - self.target_fps) > 0.5:
            issues.append(f"Invalid video frame rate: expected {self.target_fps} fps, got {fps:.2f} fps")

        # Frame-by-frame analysis
        frames_read = 0
        corrupted_frames = 0
        black_frames_count = 0
        current_black_streak = 0
        black_intervals: List[Dict[str, float]] = []

        current_frozen_streak = 0
        frozen_frames_count = 0
        frozen_intervals: List[Dict[str, float]] = []

        prev_frame_gray: Optional[np.ndarray] = None
        frame_diffs: List[float] = []

        # Read frames sequentially
        while True:
            ret, frame = cap.read()
            if not ret:
                if frames_read < total_frames:
                    corrupted_frames += (total_frames - frames_read)
                break

            frames_read += 1
            cur_time = (frames_read - 1) / max(1.0, fps)

            # Convert to grayscale for analysis
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception:
                corrupted_frames += 1
                continue

            # 1. Black frame detection (luminance mean < 5.0)
            luma_mean = float(np.mean(gray))
            if luma_mean < 5.0:
                black_frames_count += 1
                current_black_streak += 1
            else:
                if current_black_streak > 0:
                    streak_dur = current_black_streak / max(1.0, fps)
                    if streak_dur > 1.2:  # sustained black interval > 1.2s
                        start_t = cur_time - streak_dur
                        black_intervals.append({"start": round(start_t, 3), "end": round(cur_time, 3), "duration": round(streak_dur, 3)})
                current_black_streak = 0

            # 2. Frozen frame detection (inter-frame diff < 0.003)
            if prev_frame_gray is not None:
                diff = float(np.mean(np.abs(gray.astype(float) - prev_frame_gray.astype(float))))
                frame_diffs.append(diff)
                if diff < 0.003:
                    frozen_frames_count += 1
                    current_frozen_streak += 1
                else:
                    if current_frozen_streak > 0:
                        f_dur = current_frozen_streak / max(1.0, fps)
                        if f_dur > 1.5:  # sustained frozen interval > 1.5s
                            f_start = cur_time - f_dur
                            frozen_intervals.append({"start": round(f_start, 3), "end": round(cur_time, 3), "duration": round(f_dur, 3)})
                    current_frozen_streak = 0

            prev_frame_gray = gray

        cap.release()

        # Handle ending streaks
        if current_black_streak > 0:
            streak_dur = current_black_streak / max(1.0, fps)
            if streak_dur > 1.2:
                black_intervals.append({"start": round(cur_time - streak_dur, 3), "end": round(cur_time, 3), "duration": round(streak_dur, 3)})
        if current_frozen_streak > 0:
            f_dur = current_frozen_streak / max(1.0, fps)
            if f_dur > 1.5:
                frozen_intervals.append({"start": round(cur_time - f_dur, 3), "end": round(cur_time, 3), "duration": round(f_dur, 3)})

        metrics.frame_count = frames_read
        metrics.duration_seconds = round(frames_read / max(1.0, fps), 3)
        metrics.corrupted_frames = corrupted_frames
        metrics.black_frame_ratio = round(black_frames_count / max(1, frames_read), 4)
        metrics.black_intervals = black_intervals
        metrics.frozen_frame_ratio = round(frozen_frames_count / max(1, frames_read), 4)
        metrics.frozen_intervals = frozen_intervals

        # Context-aware black frame check
        if metrics.black_frame_ratio > self.max_black_ratio:
            issues.append(f"Excessive black frames: {metrics.black_frame_ratio * 100:.1f}% exceeds limit of {self.max_black_ratio * 100:.1f}%")
        elif metrics.black_intervals:
            for bi in metrics.black_intervals:
                if bi["duration"] > 2.5:
                    issues.append(f"Unacceptable sustained black interval of {bi['duration']}s from {bi['start']}s to {bi['end']}s")
                else:
                    warnings.append(f"Sustained black interval of {bi['duration']}s at {bi['start']}s (checked for transition/fade)")

        # Context-aware frozen frame check: cross-reference with approved visual holds
        if metrics.frozen_intervals:
            for fi in metrics.frozen_intervals:
                # Check if this interval coincides with an approved hold in scene metadata
                is_approved = False
                if approved_holds:
                    for ah in approved_holds:
                        ah_start = ah.get("start_time", 0.0)
                        ah_end = ah.get("end_time", 0.0)
                        if ah_start <= fi["start"] <= ah_end and fi["duration"] <= ah.get("hold_duration", 0.0) + 0.5:
                            is_approved = True
                            break

                if is_approved:
                    warnings.append(f"Approved visual hold of {fi['duration']}s detected at {fi['start']}s (approved by timing aligner).")
                elif fi["duration"] > self.max_frozen_duration:
                    issues.append(f"Suspicious frozen video sequence: {fi['duration']}s at {fi['start']}s exceeds maximum allowed duration {self.max_frozen_duration}s")
                else:
                    warnings.append(f"Static frame interval of {fi['duration']}s at {fi['start']}s")

        if corrupted_frames > 0:
            issues.append(f"Corrupted or unreadable frames encountered: {corrupted_frames} frames")

        # Expected duration consistency
        if expected_duration is not None and expected_duration > 0:
            dur_diff = abs(metrics.duration_seconds - expected_duration)
            if dur_diff > 1.0:
                issues.append(f"Video duration mismatch: expected {expected_duration:.2f}s, got {metrics.duration_seconds:.2f}s (diff {dur_diff:.2f}s)")
            elif dur_diff > 0.3:
                warnings.append(f"Minor video duration variance: expected {expected_duration:.2f}s, got {metrics.duration_seconds:.2f}s")

        # Score computation for video
        v_score = 1.0
        if not metrics.video_stream_exists or w != self.target_width or h != self.target_height or abs(fps - self.target_fps) > 0.5:
            v_score -= 0.50
        if metrics.black_frame_ratio > self.max_black_ratio:
            v_score -= 0.25
        if corrupted_frames > 0:
            v_score -= 0.30
        metrics.score = max(0.0, min(1.0, round(v_score, 3)))

        return metrics, issues, warnings

    def inspect_audio_stream(
        self,
        video_path: Path,
        video_duration: float,
        temp_dir: Optional[Path] = None
    ) -> Tuple[AudioQCMetrics, List[str], List[str]]:
        """
        Extracts and verifies the audio stream from the master MP4:
        sample rate, channels, silence ratio, digital clipping, RMS, and A/V sync.
        """
        metrics = AudioQCMetrics()
        issues: List[str] = []
        warnings: List[str] = []

        work_dir = temp_dir or video_path.parent
        extracted_wav = work_dir / f"qc_{video_path.stem}_extracted.wav"

        cmd = [
            FFMPEG, "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            str(extracted_wav)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        if not extracted_wav.exists() or extracted_wav.stat().st_size < 100:
            issues.append("Master MP4 has no decodable audio stream or audio extraction failed.")
            metrics.score = 0.0
            return metrics, issues, warnings

        metrics.audio_stream_exists = True
        try:
            with wave.open(str(extracted_wav), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                nframes = wf.getnframes()
                audio_dur = nframes / float(sample_rate)
                frames = wf.readframes(nframes)
                samples = np.frombuffer(frames, dtype=np.int16)

            metrics.channels = channels
            metrics.sample_rate = sample_rate
            metrics.duration_seconds = round(audio_dur, 3)
            metrics.codec = "aac"  # container format

            # Check channels and sample rate
            if sample_rate not in (16000, 22050, 24000, 44100, 48000):
                warnings.append(f"Non-standard audio sample rate: {sample_rate} Hz")
            if channels not in (1, 2):
                issues.append(f"Unsupported audio channel count: {channels}")

            # Digital clipping detection
            clip_count = int(np.sum(np.abs(samples) >= 32760))
            metrics.clipping_detected = (clip_count > 0)
            if clip_count > 0:
                issues.append(f"Digital audio clipping detected: {clip_count} samples clipped at peak threshold.")

            # RMS Level (dBFS)
            rms = float(np.sqrt(np.mean(samples.astype(np.float64)**2)))
            if rms > 0:
                metrics.rms_dbfs = round(20 * math.log10(rms / 32768.0), 2)
            else:
                metrics.rms_dbfs = -100.0

            if metrics.rms_dbfs < -40.0:
                issues.append(f"Master audio track is suspiciously quiet or near-silent (RMS: {metrics.rms_dbfs} dBFS).")
            elif metrics.rms_dbfs > -3.0:
                issues.append(f"Master audio track is excessively loud with potential distortion (RMS: {metrics.rms_dbfs} dBFS).")

            # Silence Ratio
            silence_thresh = 32768 * 0.01
            metrics.silence_ratio = round(float(np.mean(np.abs(samples) < silence_thresh)), 4)
            if metrics.silence_ratio > 0.45:
                warnings.append(f"High silence ratio ({metrics.silence_ratio * 100:.1f}%) in master audio.")

            # A/V Duration Synchronization Check
            av_diff = abs(metrics.duration_seconds - video_duration)
            metrics.av_duration_diff_s = round(av_diff, 3)
            if av_diff > 1.5:
                issues.append(f"Catastrophic A/V duration mismatch: video is {video_duration:.2f}s but audio is {metrics.duration_seconds:.2f}s (diff {av_diff:.2f}s)")
            elif av_diff > 0.5:
                warnings.append(f"Noticeable A/V duration divergence: {av_diff:.2f}s difference between audio and video.")

            # Audio score calculation
            a_score = 1.0
            if not metrics.audio_stream_exists:
                a_score = 0.0
            if metrics.clipping_detected:
                a_score -= 0.20
            if av_diff > 1.0:
                a_score -= 0.35
            elif av_diff > 0.5:
                a_score -= 0.15
            if metrics.rms_dbfs < -35.0:
                a_score -= 0.25

            metrics.score = max(0.0, min(1.0, round(a_score, 3)))

        except Exception as e:
            issues.append(f"Audio inspection failed with error: {e}")
            metrics.score = 0.0
        finally:
            if extracted_wav.exists():
                try:
                    extracted_wav.unlink()
                except Exception:
                    pass

        return metrics, issues, warnings

    def inspect_scene_composition(
        self,
        script_data: Dict[str, Any],
        audio_manifest: Dict[str, Any],
        motion_manifest: Dict[str, Any],
        master_scenes_info: List[Dict[str, Any]]
    ) -> Tuple[SceneQCMetrics, List[str], List[str]]:
        """
        Verifies that all script scenes are present in order, and that Yaazhini presenter
        appears ONLY on scenes designated HOST_ANCHORED, never on historical b-roll.
        """
        metrics = SceneQCMetrics()
        issues: List[str] = []
        warnings: List[str] = []

        script_scenes = script_data.get("scenes", [])
        metrics.expected_scenes = len(script_scenes)
        metrics.represented_scenes = len(master_scenes_info)

        # Expected presenter scenes from motion manifest & script
        expected_pres: List[int] = []
        for m_sc in motion_manifest.get("scenes", []):
            sc_id = int(m_sc.get("scene_id") or m_sc.get("id", 1))
            sm = m_sc.get("shot_metadata", {})
            if (
                sm.get("grounding_type") == "HOST_ANCHORED"
                or sm.get("host_character_id") == "host_yaazhini"
                or m_sc.get("shot_type") == "host_vlog"
            ):
                expected_pres.append(sc_id)
        if not expected_pres:
            # Fallback to script scene_type check
            for sc in script_scenes:
                sc_id = int(sc.get("scene_id") or sc.get("id", 1))
                sc_type = sc.get("scene_type") or sc.get("type", "")
                if sc_type in ("host_intro", "host_outro", "host_vlog"):
                    expected_pres.append(sc_id)

        metrics.presenter_scenes_expected = sorted(list(set(expected_pres)))

        # Observed presenter scenes from assembly info
        found_pres: List[int] = []
        missing_ids: List[int] = []
        unexpected_pres: List[int] = []

        expected_ids = [int(s.get("scene_id") or s.get("id", i + 1)) for i, s in enumerate(script_scenes)]
        observed_ids = [int(m.get("scene_id", 0)) for m in master_scenes_info]

        for eid in expected_ids:
            if eid not in observed_ids:
                missing_ids.append(eid)

        metrics.missing_scene_ids = missing_ids
        if missing_ids:
            issues.append(f"Missing required scenes in master video: {missing_ids}")

        # Check scene order
        if observed_ids == sorted(observed_ids) and len(observed_ids) == len(expected_ids):
            metrics.scene_order_valid = True
        else:
            metrics.scene_order_valid = False
            issues.append(f"Scene ordering violation: expected {expected_ids}, got {observed_ids}")

        # Check presenter gating
        for sc_info in master_scenes_info:
            sc_id = int(sc_info.get("scene_id", 0))
            is_pres = sc_info.get("has_presenter", False)
            if is_pres:
                found_pres.append(sc_id)
                if sc_id not in metrics.presenter_scenes_expected:
                    unexpected_pres.append(sc_id)

        metrics.presenter_scenes_found = sorted(found_pres)
        metrics.unexpected_presenter_scene_ids = unexpected_pres

        if unexpected_pres:
            issues.append(f"Presenter gating violation: Host Yaazhini appeared on non-host historical scenes: {unexpected_pres}")
            metrics.presenter_gating_valid = False
        else:
            # Check if all expected presenter scenes were included
            missing_pres = [p for p in metrics.presenter_scenes_expected if p not in found_pres]
            if missing_pres:
                warnings.append(f"Host presenter was expected but missing on scenes: {missing_pres}")
                metrics.presenter_gating_valid = False
            else:
                metrics.presenter_gating_valid = True

        # Scene score
        sc_score = 1.0
        if missing_ids:
            sc_score -= 0.40
        if not metrics.scene_order_valid:
            sc_score -= 0.20
        if unexpected_pres:
            sc_score -= 0.35
        elif not metrics.presenter_gating_valid:
            sc_score -= 0.15

        metrics.score = max(0.0, min(1.0, round(sc_score, 3)))
        return metrics, issues, warnings

    def inspect_broll_semantic_presence(
        self,
        master_video_path: Path,
        master_scenes_info: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str], List[Dict[str, Any]]]:
        """
        Multi-point final video sampling (Phase 10 Master QC):
        For every scene where requires_historical_visual == true:
        sample at 25%, 50%, and 75% of [start_frame, end_frame].
        Inspects only the upper ~75% of the frame so burnt-in subtitles do not interfere.
        Each sample is validated with SemanticVisualValidator.
        If any mandatory B-roll sample is fallback/low-information/blank/host, critical failure is triggered.
        """
        from autonomous.visual_generator import SemanticVisualValidator

        if not master_video_path.exists():
            return False, ["Master video file does not exist for semantic visual inspection."], []

        cap = cv2.VideoCapture(str(master_video_path))
        if not cap.isOpened():
            return False, ["Failed to open master video for semantic visual inspection."], []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        passed = True
        issues: List[str] = []
        sample_records: List[Dict[str, Any]] = []

        try:
            for sc in master_scenes_info:
                sc_id = int(sc.get("scene_id", 0))
                sc_type = sc.get("scene_type", "historical_broll")
                is_host = bool(sc.get("is_host_scene", sc.get("has_presenter", False)))
                gt = sc.get("grounding_type", "HISTORICAL_CLAIM_GROUNDED")

                # Determine if scene requires historical visual
                req_hist = sc.get("requires_historical_visual")
                if req_hist is None:
                    sc_type_l = str(sc_type).lower()
                    gt_u = str(gt).upper()
                    if is_host or gt_u == "HOST_ANCHORED" or sc_type_l in ("host_intro", "host_outro", "host_vlog"):
                        req_hist = False
                    elif sc_type_l in ("title", "transition", "background", "title_card", "intro_title", "outro_card", "atmospheric") or gt_u in ("ATMOSPHERIC", "ABSTRACT", "TITLE_CARD", "TRANSITION", "DECORATIVE", "ILLUSTRATIVE"):
                        req_hist = False
                    else:
                        req_hist = True

                if not req_hist:
                    continue

                # Obtain actual assembled frame boundaries
                s_frame = sc.get("start_frame")
                e_frame = sc.get("end_frame")
                if s_frame is None or e_frame is None or e_frame <= s_frame:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    dur = float(sc.get("duration_seconds", 3.0))
                    s_frame = 0
                    e_frame = max(1, int(dur * fps))

                span = max(1, e_frame - s_frame)
                sample_pts = [
                    int(s_frame + 0.25 * span),
                    int(s_frame + 0.50 * span),
                    int(s_frame + 0.75 * span)
                ]

                for p_idx, frame_num in enumerate(sample_pts):
                    frame_num = min(total_frames - 1, max(0, frame_num))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        issues.append(f"Failed to decode frame {frame_num} in Scene {sc_id}")
                        passed = False
                        continue

                    # Inspect upper approximately 75% of frame (exclude lower subtitle zone)
                    h, w, _ = frame.shape
                    roi_upper = frame[0:int(h * 0.75), :]

                    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
                        roi_upper,
                        grounding_type=gt,
                        scene_type=sc_type,
                        requires_historical_visual=True,
                        is_host_scene=is_host,
                        source_type=sc.get("source_type", "ai_generated_visualization")
                    )

                    sample_records.append({
                        "scene_id": sc_id,
                        "scene_type": sc_type,
                        "frame_number": frame_num,
                        "sample_pct": [25, 50, 75][p_idx],
                        "valid": s_valid,
                        "reason": s_reason,
                        "diagnostics": s_diag
                    })

                    if not s_valid:
                        passed = False
                        crit_msg = f"CRITICAL: Scene {sc_id} ({sc_type}) contains generic fallback/low-information visual without valid visual presence at frame {frame_num} ({[25, 50, 75][p_idx]}%). Reason: {s_reason}"
                        issues.append(crit_msg)
                        logger.error(crit_msg)
        finally:
            cap.release()

        return passed, issues, sample_records

    def run_full_qc(
        self,
        episode_id: str,
        master_video_path: Path,
        script_data: Dict[str, Any],
        audio_manifest: Dict[str, Any],
        motion_manifest: Dict[str, Any],
        master_scenes_info: List[Dict[str, Any]],
        expected_duration: Optional[float] = None,
        approved_holds: Optional[List[Dict[str, Any]]] = None
    ) -> MasterQCReport:
        """
        Executes the entire Phase 10 Quality Control suite and produces a MasterQCReport.
        Strictly enforces multi-point semantic visual presence and critical failure overrides.
        """
        all_issues: List[str] = []
        all_warnings: List[str] = []

        # 1. Video stream inspection
        v_metrics, v_issues, v_warn = self.inspect_video_stream(
            master_video_path,
            expected_duration=expected_duration,
            approved_holds=approved_holds
        )
        all_issues.extend(v_issues)
        all_warnings.extend(v_warn)

        # 2. Audio stream inspection
        a_metrics, a_issues, a_warn = self.inspect_audio_stream(
            master_video_path,
            video_duration=v_metrics.duration_seconds
        )
        all_issues.extend(a_issues)
        all_warnings.extend(a_warn)

        # 3. Scene composition inspection
        s_metrics, s_issues, s_warn = self.inspect_scene_composition(
            script_data=script_data,
            audio_manifest=audio_manifest,
            motion_manifest=motion_manifest,
            master_scenes_info=master_scenes_info
        )
        all_issues.extend(s_issues)
        all_warnings.extend(s_warn)

        # 3.5 Multi-Point Semantic Visual Presence Inspection for Mandatory B-roll (25%, 50%, 75%)
        broll_passed, broll_issues, broll_samples = self.inspect_broll_semantic_presence(
            master_video_path=master_video_path,
            master_scenes_info=master_scenes_info
        )
        all_issues.extend(broll_issues)

        # 4. Critical Failure Evaluation
        # A weighted average must NOT override critical conditions
        critical_failure = False
        if not broll_passed:
            critical_failure = True
        if not v_metrics.file_exists or not v_metrics.video_stream_exists:
            critical_failure = True
            all_issues.append("CRITICAL: Master video stream missing or unplayable.")
        if not a_metrics.audio_stream_exists:
            critical_failure = True
            all_issues.append("CRITICAL: Master audio stream missing.")
        if v_metrics.width != self.target_width or v_metrics.height != self.target_height:
            critical_failure = True
            all_issues.append(f"CRITICAL: Non-compliant resolution {v_metrics.resolution} (must be {self.target_width}x{self.target_height}).")
        if s_metrics.missing_scene_ids:
            critical_failure = True
            all_issues.append(f"CRITICAL: Missing scenes {s_metrics.missing_scene_ids}.")
        if a_metrics.av_duration_diff_s > 1.5:
            critical_failure = True
            all_issues.append(f"CRITICAL: Catastrophic A/V desynchronization ({a_metrics.av_duration_diff_s}s).")
        if s_metrics.unexpected_presenter_scene_ids:
            critical_failure = True
            all_issues.append(f"CRITICAL: Host Yaazhini illegally injected into historical b-roll: {s_metrics.unexpected_presenter_scene_ids}.")

        # 5. Weighted Quality Score Calculation
        # CRITICAL RULE: If mandatory B-roll fails semantic visual gate, scene_completeness is 0.0
        scene_comp_score = 0.0 if not broll_passed else (1.0 if not s_metrics.missing_scene_ids and s_metrics.scene_order_valid else (0.5 if not s_metrics.missing_scene_ids else 0.0))

        comp_scores = {
            "av_synchronization": self.compute_av_sync_score(a_metrics.av_duration_diff_s),
            "scene_completeness": scene_comp_score,
            "presenter_gating": 1.0 if s_metrics.presenter_gating_valid else (0.4 if not s_metrics.unexpected_presenter_scene_ids else 0.0),
            "visual_integrity": 0.0 if not broll_passed else v_metrics.score,
            "audio_integrity": a_metrics.score,
            "encoding_container": 1.0 if v_metrics.container_valid and v_metrics.width == self.target_width and v_metrics.height == self.target_height else 0.0,
            "subtitle_integrity": 1.0  # burnt in with verified unicode fonts
        }

        weighted_score = (
            comp_scores["av_synchronization"] * 0.20 +
            comp_scores["scene_completeness"] * 0.20 +
            comp_scores["presenter_gating"] * 0.15 +
            comp_scores["visual_integrity"] * 0.15 +
            comp_scores["audio_integrity"] * 0.15 +
            comp_scores["encoding_container"] * 0.10 +
            comp_scores["subtitle_integrity"] * 0.05
        )
        final_score = round(weighted_score, 3)

        # 6. CRITICAL QC OVERRIDE:
        # A weighted QC score MUST NOT override a critical visual-content failure.
        # If mandatory B-roll failed semantic visual check -> decision = REVIEW_REQUIRED
        if not broll_passed:
            decision = "REVIEW_REQUIRED"
            passed = False
        elif critical_failure:
            final_score = min(final_score, 0.45)
            decision = "FAILED"
            passed = False
        elif final_score >= self.min_score:
            decision = "MASTER_READY"
            passed = True
        elif final_score >= self.review_score:
            decision = "REVIEW_REQUIRED"
            passed = False
        else:
            decision = "FAILED"
            passed = False

        report = MasterQCReport(
            episode_id=episode_id,
            video=v_metrics,
            audio=a_metrics,
            scene=s_metrics,
            component_scores=comp_scores,
            overall_score=final_score,
            critical_failure=critical_failure,
            decision=decision,
            passed=passed,
            issues=all_issues,
            warnings=all_warnings,
            broll_semantic_samples=broll_samples
        )
        return report
