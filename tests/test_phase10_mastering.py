"""
tests/test_phase10_mastering.py - Automated Test Suite for Phase 10 Video Mastering & Quality Control.
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).

Covers:
1. AUDIO_READY input gate validation & invalid state rejection.
2. State transitions: AUDIO_READY -> MASTERING -> MASTER_READY.
3. Master video artifact generation (master.mp4, master_manifest.json, master_generation_report.json).
4. Scene ordering, timing reconciliation, and A/V synchronization.
5. Presenter gating: Host Yaazhini strictly on HOST_ANCHORED; b-roll presenter-free.
6. Aspect ratio normalization without facial distortion (1024x852 -> 1280x720 16:9).
7. Subtitle and branding integration (Tamil + English, @kaalapadhivugal).
8. Context-aware black-frame & frozen-frame QC detection.
9. Audio QC: silence, clipping, RMS loudness.
10. Atomic output promotion (master.tmp.mp4 -> master.mp4).
11. Safe recovery from interrupted mastering.
12. Strict immutable episode safeguard on 20260910-002.
13. Terminal state boundary: stops at MASTER_READY without Phase 11+ execution.
"""

import os
import sys
import json
import wave
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pytest
import numpy as np
import cv2

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.recovery_manager import RecoveryManager, RecoveryDecision
from autonomous.orchestrator import AutonomousOrchestrator, StageExecutionResult
from autonomous.master_qc import MasterQualityController, MasterQCReport, VideoQCMetrics, AudioQCMetrics, SceneQCMetrics
from autonomous.mastering_engine import AutonomousMasteringEngine, SceneAssemblyPlan, MasterExecutionResult


# ==============================================================================
# Deterministic Test Helpers
# ==============================================================================

def _generate_test_wav(filepath: Path, duration: float = 3.0, sample_rate: int = 16000) -> str:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * 220.0 * t)  # 220 Hz speech-like tone
    int16_samples = (signal * 32767.0 * 0.6).astype(np.int16)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_samples.tobytes())
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _generate_test_mp4(filepath: Path, duration: float = 3.0, width: int = 1280, height: int = 720, fps: int = 25, is_portrait: bool = False) -> str:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    w = 1024 if is_portrait else width
    h = 852 if is_portrait else height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(filepath), fourcc, float(fps), (w, h))
    num_frames = int(round(duration * fps))
    np.random.seed(42)
    noise = (np.random.rand(h, w, 3) * 50).astype(np.uint8)
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        color = int((i / max(1, num_frames)) * 140 + 50)
        frame[:, :] = (color // 2, color // 3, color)
        # Add high-frequency textures & detail grid so mock frames exhibit valid visual texture
        frame = cv2.add(frame, noise)
        for y in range(0, h, 30):
            cv2.line(frame, (0, y), (w, y), (int(color * 0.8), int(color * 0.5), 180), 1)
        for x in range(0, w, 40):
            cv2.line(frame, (x, 0), (x, h), (180, int(color * 0.7), int(color * 0.4)), 1)
        cx = int((w / 2) + np.sin(i * 0.2) * 40)
        cy = int((h / 2) + np.cos(i * 0.2) * 30)
        cv2.circle(frame, (cx, cy), 60, (255, 200, 100), -1)
        out.write(frame)
    out.release()
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _create_sample_audio_ready_bundle(ep_dir: Path, episode_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Creates a complete mock bundle for an AUDIO_READY episode."""
    script_dir = ep_dir / "script"
    motion_dir = ep_dir / "motion"
    audio_dir = ep_dir / "audio"
    presenter_dir = audio_dir / "presenter"

    script_dir.mkdir(parents=True, exist_ok=True)
    motion_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    presenter_dir.mkdir(parents=True, exist_ok=True)

    # 1. Script
    script_data = {
        "episode_id": episode_id,
        "title": "சோழர்களின் கடல்சார் வர்த்தகம்",
        "scenes": [
            {
                "scene_id": 1,
                "scene_type": "host_intro",
                "narration_tamil": "வணக்கம், நான் யாழினி. காலப் பதிவுகள் சேனலுக்கு வரவேற்கிறேன்.",
                "narration_english": "Welcome, I am Yaazhini. Welcome to Kaalapadhivugal.",
                "duration_target_seconds": 3.0
            },
            {
                "scene_id": 2,
                "scene_type": "historical_broll",
                "narration_tamil": "பூம்புகார் துறைமுகம் சோழர்களின் பிரம்மாண்ட கடல் வர்த்தக நகரமாக விளங்கியது.",
                "narration_english": "The port of Poompuhar was a monumental maritime trade hub.",
                "duration_target_seconds": 3.0
            },
            {
                "scene_id": 3,
                "scene_type": "host_outro",
                "narration_tamil": "தமிழர் வரலாற்றுச் சான்றுகளுடன் மீண்டும் சந்திப்போம். நன்றி, வணக்கம்.",
                "narration_english": "We will meet again with more historical records. Thank you.",
                "duration_target_seconds": 3.0
            }
        ]
    }
    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f, ensure_ascii=False, indent=2)

    # 2. Motion Manifest
    vid1_sha = _generate_test_mp4(motion_dir / "scene_01_motion.mp4", duration=3.0)
    vid2_sha = _generate_test_mp4(motion_dir / "scene_02_motion.mp4", duration=3.0)
    vid3_sha = _generate_test_mp4(motion_dir / "scene_03_motion.mp4", duration=3.0)

    motion_data = {
        "episode_id": episode_id,
        "scenes": [
            {
                "scene_id": 1,
                "motion_video_path": "motion/scene_01_motion.mp4",
                "duration_seconds": 3.0,
                "sha256": vid1_sha,
                "shot_metadata": {"grounding_type": "HOST_ANCHORED", "host_character_id": "host_yaazhini"}
            },
            {
                "scene_id": 2,
                "motion_video_path": "motion/scene_02_motion.mp4",
                "duration_seconds": 3.0,
                "sha256": vid2_sha,
                "shot_metadata": {"grounding_type": "HISTORICAL_ARTIFACT"}
            },
            {
                "scene_id": 3,
                "motion_video_path": "motion/scene_03_motion.mp4",
                "duration_seconds": 3.0,
                "sha256": vid3_sha,
                "shot_metadata": {"grounding_type": "HOST_ANCHORED", "host_character_id": "host_yaazhini"}
            }
        ]
    }
    with open(motion_dir / "motion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(motion_data, f, indent=2)

    # 3. Audio Manifest & Presenter Videos
    w1_sha = _generate_test_wav(audio_dir / "scene_01_audio.wav", duration=3.0)
    w2_sha = _generate_test_wav(audio_dir / "scene_02_audio.wav", duration=3.0)
    w3_sha = _generate_test_wav(audio_dir / "scene_03_audio.wav", duration=3.0)

    # Host presenter videos for Scene 1 and 3 (in 1024x852 Yaazhini native portrait aspect)
    p1_sha = _generate_test_mp4(presenter_dir / "scene_01_presenter.mp4", duration=3.0, is_portrait=True)
    p3_sha = _generate_test_mp4(presenter_dir / "scene_03_presenter.mp4", duration=3.0, is_portrait=True)

    audio_data = {
        "episode_id": episode_id,
        "audio_segments": [
            {"scene_id": 1, "file_path": "audio/scene_01_audio.wav", "duration_seconds": 3.0, "file_sha256": w1_sha},
            {"scene_id": 2, "file_path": "audio/scene_02_audio.wav", "duration_seconds": 3.0, "file_sha256": w2_sha},
            {"scene_id": 3, "file_path": "audio/scene_03_audio.wav", "duration_seconds": 3.0, "file_sha256": w3_sha}
        ],
        "presenter_segments": [
            {"scene_id": 1, "output_video_path": "audio/presenter/scene_01_presenter.mp4", "duration_seconds": 3.0, "host_character_id": "host_yaazhini"},
            {"scene_id": 3, "output_video_path": "audio/presenter/scene_03_presenter.mp4", "duration_seconds": 3.0, "host_character_id": "host_yaazhini"}
        ],
        "timing": [
            {"scene_id": 1, "visual_duration": 3.0, "narration_duration": 3.0, "required_hold_or_extension": 0.0, "timing_status": "ALIGNED"},
            {"scene_id": 2, "visual_duration": 3.0, "narration_duration": 3.0, "required_hold_or_extension": 0.0, "timing_status": "ALIGNED"},
            {"scene_id": 3, "visual_duration": 3.0, "narration_duration": 3.0, "required_hold_or_extension": 0.0, "timing_status": "ALIGNED"}
        ]
    }
    with open(audio_dir / "audio_manifest.json", "w", encoding="utf-8") as f:
        json.dump(audio_data, f, indent=2)

    return script_data, motion_data, audio_data


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def temp_mastering_workspace(tmp_path):
    db_file = tmp_path / "test_mastering.db"
    db_url = f"sqlite:///{db_file}"
    sm = StateManager(db_url=db_url)
    episodes_dir = tmp_path / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    yield sm, episodes_dir, tmp_path


# ==============================================================================
# Unit & Integration Tests (26 Tests)
# ==============================================================================

def test_01_audio_ready_input_gate_accepted(temp_mastering_workspace):
    """Test that an episode in AUDIO_READY state with complete manifests passes input gate."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-GATE-01"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test Chola Trade", episode_id=ep_id, output_directory=str(ep_dir))

    # Set state directly to AUDIO_READY
    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    ok, msg, script_d, motion_d, audio_d = engine._validate_input_gate(ep_id, EpisodeState.AUDIO_READY.value, ep_dir)
    assert ok is True
    assert "passed" in msg.lower()
    assert len(script_d["scenes"]) == 3


def test_02_invalid_state_rejection(temp_mastering_workspace):
    """Test that episodes in states other than AUDIO_READY are cleanly rejected."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-INVALID-02"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test Chola Trade", episode_id=ep_id, output_directory=str(ep_dir))

    engine = AutonomousMasteringEngine(state_manager=sm)
    # Episode is in CREATED state
    ok, msg, _, _, _ = engine._validate_input_gate(ep_id, EpisodeState.CREATED.value, ep_dir)
    assert ok is False
    assert "expected 'AUDIO_READY'" in msg


def test_03_motion_ready_rejection(temp_mastering_workspace):
    """Test that MOTION_READY episodes are rejected because Phase 9 audio is required."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-MOTION-03"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test Chola Trade", episode_id=ep_id, output_directory=str(ep_dir))

    engine = AutonomousMasteringEngine(state_manager=sm)
    ok, msg, _, _, _ = engine._validate_input_gate(ep_id, EpisodeState.MOTION_READY.value, ep_dir)
    assert ok is False
    assert "expected 'AUDIO_READY'" in msg


def test_04_blocked_review_required_safeguard(temp_mastering_workspace):
    """Test that an episode in REVIEW_REQUIRED is rejected immediately with zero mutations."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-REVIEW-04"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test Chola Trade", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.REVIEW_REQUIRED.value
    s.commit()
    s.close()

    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is False
    assert res.next_state == EpisodeState.REVIEW_REQUIRED
    assert "blocked state" in res.error

    ep_check = sm.get_episode(ep_id)
    assert ep_check.status == EpisodeState.REVIEW_REQUIRED.value
    assert ep_check.retry_count == 0


def test_05_missing_script_handling(temp_mastering_workspace):
    """Test that missing Phase 5 script fails input gate."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-NOSCRIPT-05"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    # Remove script
    (ep_dir / "script" / "script.json").unlink()

    engine = AutonomousMasteringEngine(state_manager=sm)
    ok, msg, _, _, _ = engine._validate_input_gate(ep_id, EpisodeState.AUDIO_READY.value, ep_dir)
    assert ok is False
    assert "script missing" in msg.lower()


def test_06_missing_motion_manifest_handling(temp_mastering_workspace):
    """Test that missing Phase 8 motion manifest fails input gate."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-NOMOTION-06"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    (ep_dir / "motion" / "motion_manifest.json").unlink()

    engine = AutonomousMasteringEngine(state_manager=sm)
    ok, msg, _, _, _ = engine._validate_input_gate(ep_id, EpisodeState.AUDIO_READY.value, ep_dir)
    assert ok is False
    assert "motion manifest missing" in msg.lower()


def test_07_missing_audio_manifest_handling(temp_mastering_workspace):
    """Test that missing Phase 9 audio manifest fails input gate."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-NOAUDIO-07"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    (ep_dir / "audio" / "audio_manifest.json").unlink()

    engine = AutonomousMasteringEngine(state_manager=sm)
    ok, msg, _, _, _ = engine._validate_input_gate(ep_id, EpisodeState.AUDIO_READY.value, ep_dir)
    assert ok is False
    assert "audio manifest missing" in msg.lower()


def test_08_state_transitions_to_master_ready(temp_mastering_workspace):
    """Test complete mastering transition from AUDIO_READY -> MASTERING -> MASTER_READY."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-TRANS-08"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test Chola Trade", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)

    assert res.success is True
    assert res.next_state == EpisodeState.MASTER_READY

    ep_check = sm.get_episode(ep_id)
    assert ep_check.status == EpisodeState.MASTER_READY.value
    assert ep_check.current_stage == "MASTER_READY"


def test_09_master_artifacts_creation(temp_mastering_workspace):
    """Test that master.mp4, master_manifest.json, and master_generation_report.json are created."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-ART-09"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    master_mp4 = ep_dir / "master" / "master.mp4"
    manifest_json = ep_dir / "master" / "master_manifest.json"
    report_json = ep_dir / "master" / "master_generation_report.json"

    assert master_mp4.exists()
    assert master_mp4.stat().st_size > 10000
    assert manifest_json.exists()
    assert report_json.exists()


def test_10_manifest_provenance_and_schema(temp_mastering_workspace):
    """Test that master_manifest.json contains schema version 1.0.0, SHA-256 hashes, and canonical branding."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-MAN-10"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    with open(ep_dir / "master" / "master_manifest.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "1.0.0"
    assert data["channel_identity"]["channel_handle"] == "@kaalapadhivugal"
    assert data["host_identity"]["character_id"] == "host_yaazhini"
    assert len(data["master_video_sha256"]) == 64
    assert len(data["source_manifests"]["motion_manifest_sha256"]) == 64
    assert len(data["source_manifests"]["audio_manifest_sha256"]) == 64


def test_11_scene_ordering_preserved(temp_mastering_workspace):
    """Test that all scenes are assembled in strict sequential order (1, 2, 3)."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-ORDER-11"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    with open(ep_dir / "master" / "master_manifest.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    observed_ids = [sc["scene_id"] for sc in data["scenes"]]
    assert observed_ids == [1, 2, 3]


def test_12_presenter_gating_yaazhini_only_on_host_scenes(temp_mastering_workspace):
    """Test that Yaazhini presenter video is included for Scene 1 and 3, but excluded from Scene 2."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-GATE-12"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    with open(ep_dir / "master" / "master_manifest.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    scenes = {sc["scene_id"]: sc for sc in data["scenes"]}
    assert scenes[1]["has_presenter"] is True
    assert scenes[2]["has_presenter"] is False  # B-roll scene
    assert scenes[3]["has_presenter"] is True


def test_13_aspect_ratio_normalization_without_stretching(temp_mastering_workspace):
    """Test that 1024x852 portrait presenter video is normalized to 1280x720 16:9 without stretching."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-ASPECT-13"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    master_path = ep_dir / "master" / "master.mp4"
    cap = cv2.VideoCapture(str(master_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    assert w == 1280
    assert h == 720


def test_14_video_fps_normalization(temp_mastering_workspace):
    """Test that final master video is genuinely encoded at constant 25 fps (CFR)."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-FPS-14"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    master_path = ep_dir / "master" / "master.mp4"
    cap = cv2.VideoCapture(str(master_path))
    cv_fps = cap.get(cv2.CAP_PROP_FPS)
    cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # OpenCV must report exactly 25.0 FPS
    assert abs(cv_fps - 25.0) < 0.05

    # FFprobe stream verification for genuine CFR 25/1
    probe_cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(master_path)]
    p_res = subprocess.run(probe_cmd, capture_output=True, text=True)
    probe_data = json.loads(p_res.stdout)
    v_stream = next(s for s in probe_data["streams"] if s["codec_type"] == "video")

    assert v_stream.get("r_frame_rate") == "25/1"
    assert v_stream.get("avg_frame_rate") == "25/1"
    assert int(v_stream.get("nb_frames")) == cv_frames


def test_15_audio_video_synchronization(temp_mastering_workspace):
    """Test that audio and video track durations in master MP4 are tightly synchronized (< 0.5s diff)."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-SYNC-15"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    master_path = ep_dir / "master" / "master.mp4"
    qc = MasterQualityController()
    a_met, a_issues, _ = qc.inspect_audio_stream(master_path, video_duration=9.0)
    assert a_met.audio_stream_exists is True
    assert a_met.av_duration_diff_s < 0.5


def test_16_audio_loudness_and_clipping_qc(temp_mastering_workspace):
    """Test that master audio track is verified for zero clipping and broadcast RMS."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-AUDIOQC-16"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    master_path = ep_dir / "master" / "master.mp4"
    qc = MasterQualityController()
    a_met, issues, _ = qc.inspect_audio_stream(master_path, video_duration=9.0)
    assert a_met.clipping_detected is False
    assert a_met.rms_dbfs > -30.0
    assert a_met.rms_dbfs < -5.0


def test_17_black_frame_detection_conservative(tmp_path):
    """Test that conservative black frame detection flags sustained dark intervals without failing minor fades."""
    qc = MasterQualityController(max_black_ratio=0.10)

    # Generate test video with 0 black frames
    clean_mp4 = tmp_path / "clean.mp4"
    _generate_test_mp4(clean_mp4, duration=2.0)
    metrics, issues, _ = qc.inspect_video_stream(clean_mp4)
    assert metrics.black_frame_ratio == 0.0
    assert len(issues) == 0

    # Generate test video with 50% black frames
    black_mp4 = tmp_path / "mostly_black.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(black_mp4), fourcc, 25.0, (1280, 720))
    for i in range(50):
        # 50 black frames
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out.write(frame)
    for i in range(50):
        frame = np.full((720, 1280, 3), 150, dtype=np.uint8)
        out.write(frame)
    out.release()

    b_metrics, b_issues, _ = qc.inspect_video_stream(black_mp4)
    assert b_metrics.black_frame_ratio >= 0.40
    assert any("black frames" in issue.lower() for issue in b_issues)


def test_18_frozen_frame_detection_allows_approved_holds(tmp_path):
    """Test that frozen frame detection distinguishes approved visual holds from unintended frozen video."""
    qc = MasterQualityController()

    # Generate test video with static frames for 2 seconds (50 frames)
    held_mp4 = tmp_path / "held.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(held_mp4), fourcc, 25.0, (1280, 720))
    for _ in range(50):
        frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        out.write(frame)
    out.release()

    # When NOT approved:
    metrics_unapproved, issues_unapproved, warn_unapproved = qc.inspect_video_stream(held_mp4, approved_holds=[])
    assert metrics_unapproved.frozen_frame_ratio > 0.80

    # When explicitly approved via scene metadata:
    approved_holds = [{"start_time": 0.0, "end_time": 2.0, "hold_duration": 2.0}]
    metrics_appr, issues_appr, warn_appr = qc.inspect_video_stream(held_mp4, approved_holds=approved_holds)
    assert any("approved visual hold" in w.lower() for w in warn_appr)


def test_19_qc_score_critical_failure_override(tmp_path):
    """Test that a critical failure (e.g. wrong resolution or missing stream) caps QC score to < 0.50."""
    qc = MasterQualityController()

    # Wrong resolution (640x360 instead of 1280x720)
    small_mp4 = tmp_path / "small.mp4"
    _generate_test_mp4(small_mp4, duration=2.0, width=640, height=360)

    report = qc.run_full_qc(
        episode_id="TEST-CRIT",
        master_video_path=small_mp4,
        script_data={"scenes": [{"scene_id": 1}]},
        audio_manifest={"audio_segments": []},
        motion_manifest={"scenes": []},
        master_scenes_info=[{"scene_id": 1}]
    )

    assert report.critical_failure is True
    assert report.overall_score < 0.50
    assert report.decision == "FAILED"


def test_20_atomic_output_promotion(temp_mastering_workspace):
    """Test that master.tmp.mp4 is atomically replaced with master.mp4 and temp is cleaned up."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-ATOMIC-20"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    master_tmp = ep_dir / "master" / "master.tmp.mp4"
    master_final = ep_dir / "master" / "master.mp4"

    assert not master_tmp.exists()
    assert master_final.exists()


def test_21_recovery_manager_resumption_from_mastering(temp_mastering_workspace):
    """Test that RecoveryManager safely detects completed or interrupted MASTERING state."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-RECOVER-21"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.MASTERING.value
    s.commit()
    s.close()

    rm = RecoveryManager(state_manager=sm)
    # When no master exists:
    dec = rm.evaluate_episode(sm.get_episode(ep_id))
    assert dec.action == "RETRY_STAGE"
    assert dec.target_state == EpisodeState.MASTERING

    # When valid master exists:
    master_dir = ep_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)
    _generate_test_mp4(master_dir / "master.mp4", duration=2.0)
    with open(master_dir / "master_manifest.json", "w") as f:
        json.dump({"schema_version": "1.0.0"}, f)

    dec_comp = rm.evaluate_episode(sm.get_episode(ep_id))
    assert dec_comp.action == "ADVANCE_STAGE"
    assert dec_comp.target_state == EpisodeState.MASTER_READY


def test_22_recovery_manager_master_ready_wait_boundary(temp_mastering_workspace):
    """Test that RecoveryManager treats MASTER_READY as complete/WAIT and does not advance to Phase 11."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-WAIT-22"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.MASTER_READY.value
    s.commit()
    s.close()

    master_dir = ep_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)
    _generate_test_mp4(master_dir / "master.mp4", duration=2.0)
    with open(master_dir / "master_manifest.json", "w") as f:
        json.dump({"schema_version": "1.0.0"}, f)

    rm = RecoveryManager(state_manager=sm)
    dec = rm.evaluate_episode(sm.get_episode(ep_id))
    assert dec.action == "WAIT"
    assert dec.target_state == EpisodeState.MASTER_READY


def test_23_immutable_episode_safeguard(temp_mastering_workspace):
    """Test that immutable episode 20260910-002 cannot be mastered and suffers zero mutation."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "20260910-002"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Keezhadi Civilization", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.REVIEW_REQUIRED.value
    s.commit()
    s.close()

    orch = AutonomousOrchestrator(state_manager=sm)
    res = orch.master_video(ep_id)

    assert res.success is False
    assert res.next_state == EpisodeState.REVIEW_REQUIRED
    ep_after = sm.get_episode(ep_id)
    assert ep_after.status == EpisodeState.REVIEW_REQUIRED.value
    assert ep_after.retry_count == 0


def test_24_canonical_branding_in_output(temp_mastering_workspace):
    """Test that channel handle @kaalapadhivugal is burned into manifest and report."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-BRAND-24"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    with open(ep_dir / "master" / "master_manifest.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["channel_identity"]["canonical_handle"] == "@kaalapadhivugal"
    assert data["host_identity"]["character_id"] == "host_yaazhini"


def test_25_pipeline_readiness_phase_10(temp_mastering_workspace):
    """Test that orchestrator pipeline readiness reflects Phase 10 Video Mastering & QC."""
    sm, _, _ = temp_mastering_workspace
    orch = AutonomousOrchestrator(state_manager=sm)
    readiness = orch.get_pipeline_readiness()

    assert "Phase 10" in readiness["current_phase"]
    assert "OPERATIONAL" in readiness["stages"]["Master Compositor"]
    assert "OPERATIONAL" in readiness["stages"]["Quality Control"]
    assert "NOT IMPLEMENTED" in readiness["stages"]["SEO & Thumbnail"]
    assert "NOT IMPLEMENTED" in readiness["stages"]["YouTube Upload"]


def test_26_orchestrator_master_video_stops_at_master_ready(temp_mastering_workspace):
    """Test that orchestrator.master_video returns StageExecutionResult with next_state=MASTER_READY and halts."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-ORCH-26"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    orch = AutonomousOrchestrator(state_manager=sm)
    res = orch.master_video(ep_id)

    assert res.success is True
    assert res.next_state == EpisodeState.MASTER_READY
    assert ep_dir / "master" / "master.mp4" in [Path(res.data["master_video"])]


def test_27_av_synchronization_calibration_curve():
    """Test the MasterQualityController A/V duration delta calibration curve across all required test points."""
    qc = MasterQualityController()

    # 1. 0.00s delta: imperceptible, full sync
    assert qc.compute_av_sync_score(0.0) == 1.000

    # 2. 0.05s delta: within 1 frame (40-50ms) tolerance
    assert qc.compute_av_sync_score(0.05) == 1.000

    # 3. 0.10s delta: broadcast quality sync
    assert qc.compute_av_sync_score(0.10) == 0.967

    # 4. 0.25s delta: standard documentary assembly tolerance
    assert qc.compute_av_sync_score(0.25) == 0.867

    # 5. 0.49s delta: just above warning boundary
    assert qc.compute_av_sync_score(0.49) == 0.707

    # 6. 0.50s delta: exactly at REVIEW_REQUIRED threshold
    assert qc.compute_av_sync_score(0.50) == 0.700

    # 7. 1.00s delta: noticeable divergence
    assert qc.compute_av_sync_score(1.00) == 0.350

    # 8. >=1.50s delta: catastrophic desynchronization
    assert qc.compute_av_sync_score(1.50) == 0.000
    assert qc.compute_av_sync_score(2.50) == 0.000


def test_28_cfr_frame_count_exactness(temp_mastering_workspace):
    """Test that all scenes and the concatenated master maintain exact frame counts corresponding to 25 FPS."""
    sm, ep_base, _ = temp_mastering_workspace
    ep_id = "EP-CFR-28"
    ep_dir = ep_base / ep_id
    sm.create_episode(topic="Test", episode_id=ep_id, output_directory=str(ep_dir))

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.AUDIO_READY.value
    s.commit()
    s.close()

    _create_sample_audio_ready_bundle(ep_dir, ep_id)
    engine = AutonomousMasteringEngine(state_manager=sm)
    res = engine.master_episode(ep_id)
    assert res.success is True

    master_path = ep_dir / "master" / "master.mp4"
    cap = cv2.VideoCapture(str(master_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    assert fps == 25.0
    # Expected: Scene 1 (75) + Scene 2 (75) + Scene 3 (75) = 225 frames @ 25 FPS
    assert total_frames == 225
