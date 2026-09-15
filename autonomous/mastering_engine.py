"""
autonomous/mastering_engine.py - Master Video & Audio Assembly Engine (Phase 10).
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).

Assembles broadcast-compliant, mastered 16:9 1280x720 25fps video:
1. Validates Phase 9 AUDIO_READY state and incoming Phase 8/9 manifests with SHA-256 checksums.
2. Synchronizes visual scenes with authoritative Phase 9 narration audio tracks.
3. Gates host Yaazhini presenter video strictly to HOST_ANCHORED scenes.
4. Normalizes diverse source aspect ratios (including 1024x852 Yaazhini portrait) safely to 16:9 without facial distortion.
5. Burns in canonical @kaalapadhivugal branding and bilingual typography (Tamil + English).
6. Atomically promotes master.tmp.mp4 -> master.mp4 only after passing MasterQualityController (QC >= 0.85).
7. Transitions episode strictly to MASTER_READY.
"""

import os
import sys
import time
import json
import math
import wave
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import cv2
import numpy as np
import imageio_ffmpeg
import psutil

from autonomous.config import autonomous_settings, BASE_DIR, EPISODES_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.master_qc import MasterQualityController, MasterQCReport
from autonomous.visual_planner import SceneType, normalize_scene_type

logger = logging.getLogger("autonomous.mastering_engine")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ==============================================================================
# 1. Manifest and Report Models
# ==============================================================================

@dataclass
class SceneAssemblyPlan:
    scene_id: int
    scene_type: str
    grounding_type: str
    is_host_anchored: bool
    visual_source_path: Path
    audio_source_path: Path
    narration_duration: float
    visual_duration: float
    visual_hold_seconds: float
    tamil_sub: str
    english_sub: str
    badge_text: str
    exact_frames: int = 0
    exact_duration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["visual_source_path"] = str(self.visual_source_path)
        d["audio_source_path"] = str(self.audio_source_path)
        return d


@dataclass
class MasterManifest:
    episode_id: str
    schema_version: str = "1.0.0"
    channel_identity: Dict[str, Any] = field(default_factory=dict)
    host_identity: Dict[str, Any] = field(default_factory=dict)
    source_manifests: Dict[str, str] = field(default_factory=dict)
    scenes: List[Dict[str, Any]] = field(default_factory=list)
    master_video_path: str = ""
    master_video_sha256: str = ""
    duration_seconds: float = 0.0
    resolution: str = "1280x720"
    fps: float = 25.0
    codecs: Dict[str, str] = field(default_factory=dict)
    audio_settings: Dict[str, Any] = field(default_factory=dict)
    subtitle_settings: Dict[str, Any] = field(default_factory=dict)
    qc_metrics: Dict[str, Any] = field(default_factory=dict)
    qc_score: float = 0.0
    qc_decision: str = "MASTER_READY"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MasterGenerationReport:
    episode_id: str
    total_scenes: int = 0
    total_duration_seconds: float = 0.0
    mastering_time_seconds: float = 0.0
    ffmpeg_time_seconds: float = 0.0
    qc_time_seconds: float = 0.0
    peak_ram_mb: float = 0.0
    peak_vram_mb: float = 0.0
    qc_score: float = 0.0
    status: str = "COMPLETED"
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MasterExecutionResult:
    success: bool
    next_state: EpisodeState
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==============================================================================
# 2. Autonomous Mastering Engine
# ==============================================================================

class AutonomousMasteringEngine:
    """
    Coordinates Phase 10 Video Mastering and Quality Control.
    Consumes: AUDIO_READY (Phase 9)
    Produces: MASTER_READY with master.mp4 and master_manifest.json
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        qc_controller: Optional[MasterQualityController] = None
    ):
        self.state_manager = state_manager or StateManager()
        self.qc = qc_controller or MasterQualityController()

        # Locate verified fonts
        self.font_tamil = self._resolve_font_path(
            [
                BASE_DIR / "NotoSansTamil-Bold.ttf",
                BASE_DIR / "daily_engine" / "assets" / "fonts" / "NotoSansTamil-Bold.ttf",
                Path("C:/Windows/Fonts/Nirmala.ttf"),
                Path("C:/Windows/Fonts/segoeui.ttf")
            ]
        )
        self.font_en = self._resolve_font_path(
            [
                Path("C:/Windows/Fonts/segoeui.ttf"),
                Path("C:/Windows/Fonts/arial.ttf")
            ]
        )

    def _resolve_font_path(self, candidates: List[Path]) -> str:
        for p in candidates:
            if p.exists() and p.stat().st_size > 1000:
                # Escape for FFmpeg filtergraph on Windows
                return str(p).replace("\\", "/").replace(":", r"\:")
        return "Arial"

    def _validate_input_gate(
        self,
        episode_id: str,
        ep_status: str,
        ep_dir: Path
    ) -> Tuple[bool, str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Validates all prerequisites before mastering is permitted.
        Requires AUDIO_READY.
        Requires valid Phase 5 script, Phase 8 motion_manifest, and Phase 9 audio_manifest.
        """
        permitted_states = (
            EpisodeState.AUDIO_READY.value,
            EpisodeState.MASTERING.value  # For crash recovery resumption
        )
        if ep_status not in permitted_states:
            return False, f"Episode state is '{ep_status}'; expected 'AUDIO_READY'. Rejected without execution.", None, None, None

        # 1. Phase 5 Script
        script_path = ep_dir / "script" / "script.json"
        if not script_path.exists() or script_path.stat().st_size < 10:
            val_script = ep_dir / "script" / "validated_script.json"
            if val_script.exists() and val_script.stat().st_size >= 10:
                script_path = val_script
            else:
                return False, "Phase 5 script missing: 'script/script.json' not found.", None, None, None

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                script_data = json.load(f)
            if not script_data.get("scenes"):
                return False, "Phase 5 script contains zero scenes.", None, None, None
        except Exception as e:
            return False, f"Failed to parse Phase 5 script: {e}", None, None, None

        # 2. Phase 8 Motion Manifest
        motion_manifest_path = ep_dir / "motion" / "motion_manifest.json"
        if not motion_manifest_path.exists() or motion_manifest_path.stat().st_size < 10:
            return False, "Phase 8 motion manifest missing: 'motion/motion_manifest.json' not found.", None, None, None

        try:
            with open(motion_manifest_path, "r", encoding="utf-8") as f:
                motion_data = json.load(f)
            if not motion_data.get("scenes"):
                return False, "Phase 8 motion manifest contains zero scenes.", None, None, None
        except Exception as e:
            return False, f"Failed to parse motion manifest: {e}", None, None, None

        # 3. Phase 9 Audio Manifest
        audio_manifest_path = ep_dir / "audio" / "audio_manifest.json"
        if not audio_manifest_path.exists() or audio_manifest_path.stat().st_size < 10:
            return False, "Phase 9 audio manifest missing: 'audio/audio_manifest.json' not found.", None, None, None

        try:
            with open(audio_manifest_path, "r", encoding="utf-8") as f:
                audio_data = json.load(f)
            if not audio_data.get("audio_segments"):
                return False, "Phase 9 audio manifest contains zero audio segments.", None, None, None
        except Exception as e:
            return False, f"Failed to parse audio manifest: {e}", None, None, None

        return True, "Input gate passed.", script_data, motion_data, audio_data

    def _render_scene_composite(
        self,
        plan: SceneAssemblyPlan,
        output_path: Path,
        target_width: int = 1280,
        target_height: int = 720,
        fps: int = 25
    ) -> float:
        """
        Renders a single scene: handles aspect ratio normalization, visual extension
        or holding, bilingual subtitle drawing, channel watermark, and muxes scene audio.
        Returns the exact duration of the scene in seconds.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        t_start = time.time()

        # Clean strings for FFmpeg drawtext
        b_txt = plan.badge_text.replace("'", "").replace(":", "-")
        t_txt = plan.tamil_sub.replace("'", "").replace(":", "-")
        e_txt = plan.english_sub.replace("'", "").replace(":", "-")

        # Visual duration vs Narration duration handling
        # Narration duration is authoritative!
        n_dur = plan.narration_duration
        v_dur = plan.visual_duration

        # Compute exact integer frame count and duration for genuine CFR 25 FPS output
        exact_scene_frames = plan.exact_frames if plan.exact_frames > 0 else max(1, math.ceil(n_dur * fps))
        exact_scene_duration = plan.exact_duration if plan.exact_duration > 0.0 else (exact_scene_frames / float(fps))

        # Build video filter chain
        # Safe aspect ratio scaling: preserves Yaazhini proportions without stretching
        if plan.is_host_anchored:
            # 1024x852 portrait crop: scale with aspect increase, crop vertically biased to face
            scale_filter = (
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
                f"crop={target_width}:{target_height}:(in_w-out_w)/2:min(in_h-out_h\\,max(0\\,(in_h-out_h)*0.30))"
            )
        else:
            # Standard 16:9 scaling
            scale_filter = (
                f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
                f"crop={target_width}:{target_height}"
            )

        # Handle visual hold / extension to guarantee trailing frames for exact CFR rounding
        hold_sec = max(1.0, round(exact_scene_duration - v_dur + 0.5, 3))
        ext_filter = f"{scale_filter},tpad=stop_mode=clone:stop_duration={hold_sec}"

        vf = (
            f"{ext_filter},"
            f"fps=fps={fps}:round=near,"
            # Top Badge: Gold text on semi-transparent dark box
            f"drawtext=fontfile='{self.font_tamil}':text='{b_txt}':fontsize=18:fontcolor=0xFFD700:x=(w-text_w)/2:y=32:box=1:boxcolor=black@0.70:boxborderw=8,"
            # Watermark: top-left channel handle
            f"drawtext=fontfile='{self.font_en}':text='@kaalapadhivugal  |  Kaalapadhivugal':fontsize=15:fontcolor=white@0.65:x=35:y=35,"
            # Bottom subtitle gradient box
            f"drawbox=y=ih-120:color=black@0.72:width=iw:height=120:t=fill,"
            f"drawbox=y=ih-120:color=0xD4AF37:width=iw:height=2:t=fill,"
            # Tamil subtitle
            f"drawtext=fontfile='{self.font_tamil}':text='{t_txt}':fontsize=20:fontcolor=white:x=(w-text_w)/2:y=h-88,"
            # English subtitle
            f"drawtext=fontfile='{self.font_en}':text='{e_txt}':fontsize=15:fontcolor=0xFFD700:x=(w-text_w)/2:y=h-45"
        )

        af = f"apad,atrim=0:{exact_scene_duration:.4f}"

        cmd = [
            FFMPEG, "-y",
            "-i", str(plan.visual_source_path),
            "-i", str(plan.audio_source_path),
            "-vf", vf,
            "-af", af,
            "-c:v", autonomous_settings.master_video_codec,
            "-crf", str(autonomous_settings.master_crf),
            "-preset", autonomous_settings.master_preset,
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-fps_mode", "cfr",
            "-frames:v", str(exact_scene_frames),
            "-c:a", autonomous_settings.master_audio_codec,
            "-b:a", autonomous_settings.master_audio_bitrate,
            str(output_path)
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return round(time.time() - t_start, 3)

    def master_episode(self, episode_id: str) -> MasterExecutionResult:
        """
        Executes Phase 10 Video Mastering & Quality Control for the given episode.
        Produces master/master.mp4 and master_manifest.json.
        """
        ep = self.state_manager.get_episode(episode_id)
        if not ep:
            err = f"Episode '{episode_id}' not found in database."
            logger.error(err)
            return MasterExecutionResult(False, EpisodeState.FAILED, error=err)

        ep_status = ep.status
        ep_dir = Path(ep.output_directory)

        # 1. Strict Immutable Protection: do not mutate blocked episodes
        if ep_status in (EpisodeState.REVIEW_REQUIRED.value, EpisodeState.FAILED.value, EpisodeState.CANCELLED.value):
            err_msg = f"Episode '{episode_id}' is in blocked state '{ep_status}'. Mastering rejected without mutation."
            logger.warning(err_msg)
            return MasterExecutionResult(False, EpisodeState(ep_status), error=err_msg)

        # 2. Input Gate Validation
        gate_ok, gate_msg, script_data, motion_data, audio_data = self._validate_input_gate(episode_id, ep_status, ep_dir)
        if not gate_ok:
            logger.warning(f"Phase 10 input gate failed for episode '{episode_id}': {gate_msg}")
            curr_state = EpisodeState(ep_status) if ep_status in EpisodeState._value2member_map_ else EpisodeState.FAILED
            return MasterExecutionResult(False, curr_state, error=gate_msg)

        # 3. Transition to MASTERING
        self.state_manager.transition_state(
            episode_id=episode_id,
            new_state=EpisodeState.MASTERING,
            stage_name="MASTERING"
        )

        t_start = time.time()
        master_dir = ep_dir / "master"
        scenes_dir = master_dir / "scenes"
        master_dir.mkdir(parents=True, exist_ok=True)
        scenes_dir.mkdir(parents=True, exist_ok=True)
        temp_master_path = master_dir / "master.tmp.mp4"
        final_master_path = master_dir / "master.mp4"

        ffmpeg_total_time = 0.0
        peak_ram = float(round(psutil.Process().memory_info().rss / (1024 * 1024), 1))
        peak_vram = 0.0

        try:
            # 4. Plan Scene Assembly
            script_scenes = script_data.get("scenes", [])
            motion_scenes = {int(s.get("scene_id", 1)): s for s in motion_data.get("scenes", [])}
            audio_segments = {int(a.get("scene_id", 1)): a for a in audio_data.get("audio_segments", [])}
            presenter_segments = {int(p.get("scene_id", 1)): p for p in audio_data.get("presenter_segments", [])}
            audio_timings = {int(t.get("scene_id", 1)): t for t in audio_data.get("timing", [])}

            assembly_plans: List[SceneAssemblyPlan] = []
            approved_holds: List[Dict[str, Any]] = []
            cur_time_cursor = 0.0

            for sc in script_scenes:
                sc_id = int(sc.get("scene_id") or sc.get("id", 1))
                sc_type = sc.get("scene_type") or sc.get("type", "historical_broll")
                m_info = motion_scenes.get(sc_id, {})
                shot_meta = m_info.get("shot_metadata", {})
                grounding = shot_meta.get("grounding_type", "")
                host_id = shot_meta.get("host_character_id", "")

                # Host Presenter Gating: Yaazhini only if HOST_ANCHORED or host scene
                is_host = (
                    grounding == "HOST_ANCHORED"
                    or host_id == "host_yaazhini"
                    or sc_type in ("host_intro", "host_outro", "host_vlog")
                    or sc_id in presenter_segments
                )

                # Visual Source Selection
                if is_host and sc_id in presenter_segments:
                    p_rec = presenter_segments[sc_id]
                    v_source = ep_dir / p_rec.get("output_video_path", f"audio/presenter/scene_{sc_id:02d}_presenter.mp4")
                    if not v_source.exists():
                        # Fallback to direct path in audio/presenter
                        v_source = ep_dir / "audio" / "presenter" / f"scene_{sc_id:02d}_presenter.mp4"
                else:
                    is_host = False  # B-roll or artifact scene
                    v_source_candidates = [
                        ep_dir / m_info.get("output_video_path", ""),
                        ep_dir / m_info.get("motion_video_path", ""),
                        ep_dir / "motion" / "scenes" / f"scene_{sc_id:02d}" / "motion.mp4",
                        ep_dir / "motion" / f"scene_{sc_id:02d}_motion.mp4",
                    ]
                    v_source = None
                    for cand in v_source_candidates:
                        if cand and cand.exists() and cand.is_file():
                            v_source = cand
                            break
                    if not v_source:
                        v_source = ep_dir / "motion" / f"scene_{sc_id:02d}_motion.mp4"

                # Audio Source Selection
                a_source_candidates = [
                    ep_dir / audio_segments.get(sc_id, {}).get("output_path", ""),
                    ep_dir / audio_segments.get(sc_id, {}).get("file_path", ""),
                    ep_dir / "audio" / f"scene_{sc_id:02d}_audio.wav",
                    ep_dir / "audio" / f"scene_{sc_id}_audio.wav",
                ]
                a_source = None
                for cand in a_source_candidates:
                    if cand and cand.exists() and cand.is_file():
                        a_source = cand
                        break
                if not a_source:
                    a_source = ep_dir / "audio" / f"scene_{sc_id:02d}_audio.wav"

                # Durations and exact frame count for genuine CFR 25 FPS
                t_info = audio_timings.get(sc_id, {})
                narr_dur = float(audio_segments.get(sc_id, {}).get("duration_seconds", t_info.get("narration_duration", 3.0)))
                vis_dur = float(m_info.get("duration_seconds", t_info.get("visual_duration", 3.0)))
                hold_sec = float(t_info.get("required_hold_or_extension", 0.0))

                exact_frames = max(1, math.ceil(narr_dur * autonomous_settings.master_fps))
                exact_dur = round(exact_frames / float(autonomous_settings.master_fps), 4)

                # Track approved hold interval for QC
                if hold_sec > 0:
                    approved_holds.append({
                        "scene_id": sc_id,
                        "start_time": round(cur_time_cursor, 3),
                        "end_time": round(cur_time_cursor + exact_dur, 3),
                        "hold_duration": round(hold_sec, 3)
                    })
                cur_time_cursor += exact_dur

                tamil_sub = sc.get("narration_tamil", sc.get("tamil_text", ""))
                english_sub = sc.get("narration_english", sc.get("english_sub", ""))
                badge = sc.get("badge_text", script_data.get("title", autonomous_settings.channel.channel_name_ta))

                assembly_plans.append(SceneAssemblyPlan(
                    scene_id=sc_id,
                    scene_type=sc_type,
                    grounding_type=grounding,
                    is_host_anchored=is_host,
                    visual_source_path=v_source,
                    audio_source_path=a_source,
                    narration_duration=narr_dur,
                    visual_duration=vis_dur,
                    visual_hold_seconds=hold_sec,
                    tamil_sub=tamil_sub,
                    english_sub=english_sub,
                    badge_text=badge,
                    exact_frames=exact_frames,
                    exact_duration=exact_dur
                ))

            # Load visual plan and assets manifest if present for authoritative scene mapping
            visual_plan_file = ep_dir / "visuals" / "visual_plan.json"
            assets_manifest_file = ep_dir / "visuals" / "assets_manifest.json"
            plan_scenes_by_id: Dict[int, Dict[str, Any]] = {}
            if visual_plan_file.exists():
                try:
                    with open(visual_plan_file, "r", encoding="utf-8") as f:
                        vp_data = json.load(f)
                    for vps in vp_data.get("scenes", []):
                        plan_scenes_by_id[int(vps.get("scene_id", 0))] = vps
                except Exception as e:
                    logger.warning(f"Could not load visual_plan.json in mastering: {e}")

            assets_by_scene_id: Dict[int, Dict[str, Any]] = {}
            if assets_manifest_file.exists():
                try:
                    with open(assets_manifest_file, "r", encoding="utf-8") as f:
                        am_data = json.load(f)
                    for ast in am_data.get("assets", []):
                        assets_by_scene_id[int(ast.get("scene_id", 0))] = ast
                except Exception as e:
                    logger.warning(f"Could not load assets_manifest.json in mastering: {e}")

            # 5. Render Scene Master Clips
            rendered_scene_paths: List[Path] = []
            master_scenes_info: List[Dict[str, Any]] = []
            cumulative_frames = 0

            for plan in assembly_plans:
                out_sc = scenes_dir / f"scene_{plan.scene_id:02d}_master.mp4"
                logger.info(f"Mastering scene {plan.scene_id} ({plan.scene_type}, host={plan.is_host_anchored})...")

                ffmpeg_t = self._render_scene_composite(
                    plan=plan,
                    output_path=out_sc,
                    target_width=autonomous_settings.master_width,
                    target_height=autonomous_settings.master_height,
                    fps=autonomous_settings.master_fps
                )
                ffmpeg_total_time += ffmpeg_t
                rendered_scene_paths.append(out_sc)

                start_frame = cumulative_frames
                end_frame = cumulative_frames + plan.exact_frames - 1
                cumulative_frames += plan.exact_frames

                v_info = plan_scenes_by_id.get(plan.scene_id, {})
                a_info = assets_by_scene_id.get(plan.scene_id, {})

                vis_asset_id = a_info.get("asset_id") or v_info.get("asset_requirement", {}).get("asset_id") or f"asset_ep_{episode_id}_sc{plan.scene_id:02d}"
                grounding_type = plan.grounding_type or a_info.get("grounding_type") or v_info.get("grounding_type", "HISTORICAL_CLAIM_GROUNDED")
                is_host_scene = plan.is_host_anchored or normalize_scene_type(str(plan.scene_type)) == SceneType.HOST or str(grounding_type).upper() == "HOST_ANCHORED"

                if "requires_historical_visual" in v_info:
                    requires_hist = bool(v_info["requires_historical_visual"])
                elif is_host_scene:
                    requires_hist = False
                elif str(plan.scene_type).lower() in ("title", "transition", "background", "title_card", "intro_title", "outro_card", "atmospheric") or str(grounding_type).upper() in ("ATMOSPHERIC", "ABSTRACT", "TITLE_CARD", "TRANSITION", "DECORATIVE", "ILLUSTRATIVE"):
                    requires_hist = False
                else:
                    requires_hist = True

                master_scenes_info.append({
                    "scene_id": plan.scene_id,
                    "scene_type": plan.scene_type,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "visual_asset_id": vis_asset_id,
                    "grounding_type": grounding_type,
                    "requires_historical_visual": requires_hist,
                    "is_host_scene": is_host_scene,
                    "has_presenter": plan.is_host_anchored,
                    "clip_path": str(out_sc.relative_to(ep_dir)),
                    "duration_seconds": round(plan.exact_duration, 3)
                })

                peak_ram = max(peak_ram, float(round(psutil.Process().memory_info().rss / (1024 * 1024), 1)))

            # 6. Concatenate Scene Clips into Atomic Temporary Master
            temp_master_path = master_dir / "master.tmp.mp4"
            final_master_path = master_dir / "master.mp4"
            concat_list_file = master_dir / "concat_scenes.txt"

            with open(concat_list_file, "w", encoding="utf-8") as f:
                for p in rendered_scene_paths:
                    f.write(f"file '{str(p).replace('\\', '/')}'\n")

            cmd_concat = [
                FFMPEG, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c:v", "copy",
                "-c:a", autonomous_settings.master_audio_codec,
                "-b:a", autonomous_settings.master_audio_bitrate,
                "-movflags", "+faststart",
                str(temp_master_path)
            ]
            t_cat0 = time.time()
            subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            ffmpeg_total_time += (time.time() - t_cat0)

            # 7. Execute Independent Quality Control Suite
            t_qc0 = time.time()
            expected_total_duration = sum(p.exact_duration for p in assembly_plans)

            qc_report = self.qc.run_full_qc(
                episode_id=episode_id,
                master_video_path=temp_master_path,
                script_data=script_data,
                audio_manifest=audio_data,
                motion_manifest=motion_data,
                master_scenes_info=master_scenes_info,
                expected_duration=expected_total_duration,
                approved_holds=approved_holds
            )
            qc_duration = round(time.time() - t_qc0, 3)

            # 8. Atomic Promotion or Failure Handling
            if qc_report.passed:
                # Atomic rename/replacement
                if final_master_path.exists():
                    final_master_path.unlink()
                temp_master_path.replace(final_master_path)

                # Compute final SHA-256
                with open(final_master_path, "rb") as f:
                    master_sha = hashlib.sha256(f.read()).hexdigest()

                # Source manifest hashes for provenance
                with open(ep_dir / "motion" / "motion_manifest.json", "rb") as f:
                    m_sha = hashlib.sha256(f.read()).hexdigest()
                with open(ep_dir / "audio" / "audio_manifest.json", "rb") as f:
                    a_sha = hashlib.sha256(f.read()).hexdigest()

                # Serialize Master Manifest
                manifest_obj = MasterManifest(
                    episode_id=episode_id,
                    schema_version="1.0.0",
                    channel_identity={
                        "channel_name_ta": autonomous_settings.channel.channel_name_ta,
                        "channel_name_en": autonomous_settings.channel.channel_name_en,
                        "channel_handle": autonomous_settings.channel.channel_handle,
                        "canonical_handle": autonomous_settings.channel.channel_handle,
                    },
                    host_identity={
                        "host_name_en": autonomous_settings.channel.host_name_en,
                        "host_name_ta": autonomous_settings.channel.host_name_ta,
                        "character_id": autonomous_settings.channel.host_character_id,
                        "host_consistency_token": autonomous_settings.channel.host_consistency_token
                    },
                    source_manifests={
                        "motion_manifest_sha256": m_sha,
                        "audio_manifest_sha256": a_sha
                    },
                    scenes=master_scenes_info,
                    master_video_path=str(final_master_path.relative_to(ep_dir)),
                    master_video_sha256=master_sha,
                    duration_seconds=qc_report.video.duration_seconds,
                    resolution=qc_report.video.resolution,
                    fps=qc_report.video.fps,
                    codecs={"video": qc_report.video.codec, "audio": qc_report.audio.codec},
                    audio_settings={
                        "sample_rate": qc_report.audio.sample_rate,
                        "channels": qc_report.audio.channels,
                        "rms_dbfs": qc_report.audio.rms_dbfs
                    },
                    subtitle_settings={
                        "enabled": autonomous_settings.enable_subtitles,
                        "font_tamil": self.font_tamil,
                        "font_english": self.font_en
                    },
                    qc_metrics=qc_report.to_dict(),
                    qc_score=qc_report.overall_score,
                    qc_decision=qc_report.decision
                )

                with open(master_dir / "master_manifest.json", "w", encoding="utf-8") as f:
                    json.dump(manifest_obj.to_dict(), f, ensure_ascii=False, indent=2)

                # Serialize Generation Report
                total_dur_s = round(time.time() - t_start, 2)
                report_obj = MasterGenerationReport(
                    episode_id=episode_id,
                    total_scenes=len(assembly_plans),
                    total_duration_seconds=qc_report.video.duration_seconds,
                    mastering_time_seconds=total_dur_s,
                    ffmpeg_time_seconds=round(ffmpeg_total_time, 2),
                    qc_time_seconds=qc_duration,
                    peak_ram_mb=peak_ram,
                    peak_vram_mb=peak_vram,
                    qc_score=qc_report.overall_score,
                    status="COMPLETED",
                    issues=qc_report.issues,
                    warnings=qc_report.warnings
                )

                with open(master_dir / "master_generation_report.json", "w", encoding="utf-8") as f:
                    json.dump(report_obj.to_dict(), f, ensure_ascii=False, indent=2)

                # Transition state to MASTER_READY
                self.state_manager.transition_state(
                    episode_id=episode_id,
                    new_state=EpisodeState.MASTER_READY,
                    stage_name="MASTER_READY"
                )

                # Update database record
                if hasattr(self.state_manager, "update_artifacts"):
                    self.state_manager.update_artifacts(
                        episode_id=episode_id,
                        video_path=str(final_master_path),
                        video_qc_status="PASS"
                    )
                elif hasattr(self.state_manager, "update_episode"):
                    self.state_manager.update_episode(
                        episode_id=episode_id,
                        video_path=str(final_master_path),
                        video_qc_status="PASS"
                    )

                logger.info(f"Phase 10: Episode '{episode_id}' mastered successfully (QC score: {qc_report.overall_score}). Reached MASTER_READY.")
                return MasterExecutionResult(
                    success=True,
                    next_state=EpisodeState.MASTER_READY,
                    data={
                        "master_manifest": str(master_dir / "master_manifest.json"),
                        "master_video": str(final_master_path),
                        "qc_score": qc_report.overall_score,
                        "qc_decision": qc_report.decision
                    }
                )

            elif qc_report.decision == "REVIEW_REQUIRED":
                # Preserve temporary master as review master
                review_path = master_dir / "master_review.mp4"
                if temp_master_path.exists():
                    temp_master_path.replace(review_path)

                err_msg = f"Master QC score ({qc_report.overall_score}) requires human review: {qc_report.issues}"
                logger.warning(f"Phase 10: Episode '{episode_id}' flagged for review: {err_msg}")

                self.state_manager.transition_state(
                    episode_id=episode_id,
                    new_state=EpisodeState.REVIEW_REQUIRED,
                    stage_name="MASTERING",
                    error_message=err_msg
                )
                return MasterExecutionResult(False, EpisodeState.REVIEW_REQUIRED, error=err_msg)

            else:
                # Critical Failure or Low Score (< 0.70)
                if temp_master_path.exists():
                    temp_master_path.unlink()

                err_msg = f"Master QC failed (score {qc_report.overall_score}): {qc_report.issues}"
                logger.error(f"Phase 10: Episode '{episode_id}' failed mastering: {err_msg}")

                self.state_manager.transition_state(
                    episode_id=episode_id,
                    new_state=EpisodeState.FAILED,
                    stage_name="MASTERING",
                    error_message=err_msg
                )
                return MasterExecutionResult(False, EpisodeState.FAILED, error=err_msg)

        except Exception as e:
            logger.exception(f"Unexpected error during mastering of episode '{episode_id}': {e}")
            if "temp_master_path" in locals() and temp_master_path and temp_master_path.exists():
                try:
                    temp_master_path.unlink()
                except Exception:
                    pass

            self.state_manager.transition_state(
                episode_id=episode_id,
                new_state=EpisodeState.FAILED,
                stage_name="MASTERING",
                error_message=str(e)
            )
            return MasterExecutionResult(False, EpisodeState.FAILED, error=str(e))


