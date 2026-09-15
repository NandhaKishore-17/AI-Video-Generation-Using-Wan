"""
tests/test_phase12_youtube.py - Comprehensive Test Suite for Phase 12.
Autonomous YouTube Publication Architecture (Offline / Mock Gate).

Covers all 48 distinct acceptance requirements:
- Input package validation (21-point gate)
- Safety locks and integration disabled by default
- Mock YouTube client operations
- Offline execution and network isolation
- Duplicate protection and idempotency
- Thumbnail failure recovery and video preservation
- Publication manifest generation and cryptographic provenance
- Recovery from interrupted states
- Secret sanitization
- Artifact immutability and 20260910-002 safeguard
- CLI execution and status reporting
"""

import os
import sys
import json
import socket
import hashlib
import tempfile
import cv2
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from autonomous.config import autonomous_settings, AutonomousSettings
from autonomous.state_manager import (
    StateManager, EpisodeState, AutonomousEpisode,
    VALID_TRANSITIONS, PHASE_CONSUMPTION_TRANSITIONS, TERMINAL_STATES
)
from autonomous.recovery_manager import RecoveryManager, RecoveryDecision
from autonomous.orchestrator import AutonomousOrchestrator, StageExecutionResult
from autonomous.youtube_client import (
    BaseYouTubeClient, MockYouTubeClient, RealYouTubeClient, get_youtube_client,
    YouTubeClientError, YouTubeIntegrationDisabledError, YouTubeUploadError,
    YouTubeDuplicateError, YouTubeVerificationError, ErrorClassification
)
from autonomous.youtube_validator import YouTubePublicationValidator, YouTubeValidationReport
from autonomous.upload_manager import YouTubeUploadManager, UploadResult
from autonomous.publication_verifier import YouTubePublicationVerifier
from autonomous.publication_manager import PublicationManager
from autonomous.publish_package import PublishPackageBuilder
from run_channel import (
    handle_youtube_dry_run, handle_mock_youtube_upload, handle_youtube_status,
    handle_validate_youtube, handle_youtube_integration_status
)


def _sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _generate_test_mp4(filepath: Path, duration: float = 1.0, width: int = 1280, height: int = 720, fps: int = 25) -> str:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(filepath), fourcc, float(fps), (width, height))
    num_frames = int(round(duration * fps))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color = int((i / max(1, num_frames)) * 180 + 40)
        frame[:, :] = (color // 2, color // 3, color)
        cv2.circle(frame, (width // 2, height // 2), 60, (255, 200, 100), -1)
        out.write(frame)
    out.release()
    return _sha256_file(filepath)


def _generate_test_image(filepath: Path, width: int = 1280, height: int = 720) -> str:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (50, 40, 70)
    cv2.circle(img, (width // 2, height // 2), 120, (200, 180, 100), -1)
    cv2.imwrite(str(filepath), img)
    return _sha256_file(filepath)


@pytest.fixture
def temp_phase12_workspace(tmp_path):
    """Sets up an isolated database and PUBLISH_PACKAGE_READY episode workspace."""
    db_file = tmp_path / "test_platform_p12.db"
    ep_base = tmp_path / "episodes"
    ep_base.mkdir(parents=True, exist_ok=True)

    sm = StateManager(db_url=f"sqlite:///{db_file}")

    ep_id = "EP-TEST-P12"
    ep_dir = ep_base / ep_id
    ep_dir.mkdir(parents=True, exist_ok=True)

    sm.create_episode(
        topic="சோழர்களின் கடல்சார் வர்த்தகம் மற்றும் துறைமுகங்கள்",
        category="Ancient Tamil history",
        episode_id=ep_id,
        output_directory=str(ep_dir)
    )

    # 1. Script
    script_dir = ep_dir / "script"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_data = {
        "episode_id": ep_id,
        "title": "சோழர்களின் கடல்சார் வர்த்தகம்",
        "scenes": [
            {
                "scene_id": 1,
                "narration_tamil": "வணக்கம், நான் யாழினி. காலப் பதிவுகள் சேனலுக்கு வரவேற்கிறேன்.",
                "narration_english": "Welcome to Kaalapadhivugal. I am Yaazhini."
            }
        ]
    }
    script_file = script_dir / "validated_script.json"
    with open(script_file, "w", encoding="utf-8") as f:
        json.dump(script_data, f, indent=2, ensure_ascii=False)
    val_script_sha = _sha256_file(script_file)

    # 2. Master
    master_dir = ep_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_mp4 = master_dir / "master.mp4"
    master_sha = _generate_test_mp4(master_mp4, duration=1.0)
    master_manifest = {
        "episode_id": ep_id,
        "master_sha256": master_sha,
        "duration_seconds": 1.0,
        "resolution": "1280x720",
        "fps": 25.0,
    }
    master_manifest_file = master_dir / "master_manifest.json"
    with open(master_manifest_file, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    # 3. Publish Directory & Phase 11 Artifacts
    publish_dir = ep_dir / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    thumb_jpg = publish_dir / "thumbnail.jpg"
    thumb_sha = _generate_test_image(thumb_jpg)

    seo_data = {
        "episode_id": ep_id,
        "selected_title": "சோழர்களின் கடல்சார் வர்த்தகம் — வரலாற்றுச் சான்றுகள்",
        "description": "காலப் பதிவுகள் (@kaalapadhivugal) வழங்கும் ஆவணம். யாழினி (Yaazhini). #Kaalapadhivugal",
        "tags": ["Kaalapadhivugal", "காலப் பதிவுகள்", "@kaalapadhivugal", "Yaazhini", "யாழினி", "சோழர்"],
        "category": "Education",
        "category_id": 27,
        "hashtags": ["#Kaalapadhivugal", "#Chola"],
    }
    with open(publish_dir / "seo.json", "w", encoding="utf-8") as f:
        json.dump(seo_data, f, indent=2, ensure_ascii=False)

    seo_report = {
        "passed": True,
        "errors": [],
        "warnings": [],
        "checks": {"canonical_identity": True, "no_obsolete_identity": True}
    }
    with open(publish_dir / "seo_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(seo_report, f, indent=2)

    thumb_manifest = {
        "episode_id": ep_id,
        "thumbnail_path": "publish/thumbnail.jpg",
        "thumbnail_sha256": thumb_sha,
        "dimensions": [1280, 720],
        "aspect_ratio": "16:9",
        "qc_passed": True,
    }
    with open(publish_dir / "thumbnail_manifest.json", "w", encoding="utf-8") as f:
        json.dump(thumb_manifest, f, indent=2)

    thumb_qc = {
        "passed": True,
        "dimensions": [1280, 720],
        "aspect_ratio": "16:9",
        "contrast_score": 42.0,
        "is_blank": False,
        "has_black_borders": False,
        "errors": []
    }
    with open(publish_dir / "thumbnail_qc_report.json", "w", encoding="utf-8") as f:
        json.dump(thumb_qc, f, indent=2)

    package_data = {
        "episode_id": ep_id,
        "schema_version": "1.0.0",
        "generator_version": "1.0.0",
        "channel_identity": {
            "channel_name_ta": "காலப் பதிவுகள்",
            "channel_name_en": "Kaalapadhivugal",
            "channel_handle": "@kaalapadhivugal",
        },
        "host_identity": {
            "host_name_ta": "யாழினி",
            "host_name_en": "Yaazhini",
            "character_id": "host_yaazhini",
        },
        "master": {
            "path": "master/master.mp4",
            "sha256": master_sha,
            "manifest_path": "master/master_manifest.json",
            "duration_seconds": 1.0,
            "file_size_bytes": master_mp4.stat().st_size,
        },
        "script": {
            "script_path": "script/validated_script.json",
            "script_sha256": val_script_sha,
            "validated_script_sha256": val_script_sha,
        },
        "seo": seo_data,
        "thumbnail": {
            "path": "publish/thumbnail.jpg",
            "sha256": thumb_sha,
            "dimensions": [1280, 720],
        },
        "provenance": {
            "generator_version": "1.0.0",
            "schema_version": "1.0.0",
            "source_master_sha256": master_sha,
            "source_script_sha256": val_script_sha,
            "source_validated_script_sha256": val_script_sha,
        }
    }
    with open(publish_dir / "publish_package.json", "w", encoding="utf-8") as f:
        json.dump(package_data, f, indent=2, ensure_ascii=False)

    session = sm._get_session()
    ep = session.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.PUBLISH_PACKAGE_READY.value
    ep.current_stage = EpisodeState.PUBLISH_PACKAGE_READY.value
    session.commit()
    session.close()

    return sm, ep_dir, ep_id


# ==============================================================================
# Tests 1-11: Input Gate & Package Validation
# ==============================================================================

def test_01_publish_package_ready_accepted(temp_phase12_workspace):
    """Test 1: Episode in PUBLISH_PACKAGE_READY with valid package is accepted."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is True
    assert len(report.errors) == 0


def test_02_review_required_rejected(temp_phase12_workspace):
    """Test 2: Episode in REVIEW_REQUIRED state is rejected by publication gate."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    # Set status directly in DB to simulate REVIEW_REQUIRED
    session = sm._get_session()
    ep = session.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.REVIEW_REQUIRED.value
    session.commit()
    session.close()

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("REVIEW_REQUIRED" in err or "Gate 2" in err for err in report.errors)


def test_03_failed_rejected(temp_phase12_workspace):
    """Test 3: Episode in FAILED state is rejected by publication gate."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    session = sm._get_session()
    ep = session.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.FAILED.value
    session.commit()
    session.close()

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("FAILED" in err or "Gate 2" in err for err in report.errors)


def test_04_missing_package_rejected(temp_phase12_workspace):
    """Test 4: Missing publish_package.json fails validation."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pkg_file = ep_dir / "publish" / "publish_package.json"
    pkg_file.unlink()

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("publish_package.json" in err for err in report.errors)


def test_05_corrupt_package_rejected(temp_phase12_workspace):
    """Test 5: Corrupt JSON in publish_package.json fails validation."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pkg_file = ep_dir / "publish" / "publish_package.json"
    pkg_file.write_text("NOT_VALID_JSON{", encoding="utf-8")

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("parse error" in err.lower() for err in report.errors)


def test_06_master_hash_mismatch_rejected(temp_phase12_workspace):
    """Test 6: Master MP4 hash mismatch with package record fails validation."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pkg_file = ep_dir / "publish" / "publish_package.json"
    with open(pkg_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["master"]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with open(pkg_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("Master MP4 actual hash" in err for err in report.errors)


def test_07_thumbnail_hash_mismatch_rejected(temp_phase12_workspace):
    """Test 7: Thumbnail JPG hash mismatch with package record fails validation."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pkg_file = ep_dir / "publish" / "publish_package.json"
    with open(pkg_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["thumbnail"]["sha256"] = "1111111111111111111111111111111111111111111111111111111111111111"
    with open(pkg_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("Thumbnail JPG actual hash" in err for err in report.errors)


def test_08_invalid_seo_rejected(temp_phase12_workspace):
    """Test 8: Failed SEO validation report fails input gate."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    rep_file = ep_dir / "publish" / "seo_validation_report.json"
    rep_file.write_text(json.dumps({"passed": False, "errors": ["SEO validation failed"]}), encoding="utf-8")

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("SEO validation report status is False" in err for err in report.errors)


def test_09_invalid_thumbnail_qc_rejected(temp_phase12_workspace):
    """Test 9: Failed thumbnail QC report fails input gate."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    qc_file = ep_dir / "publish" / "thumbnail_qc_report.json"
    qc_file.write_text(json.dumps({"passed": False, "errors": ["Contrast below threshold"]}), encoding="utf-8")

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("Thumbnail QC report status is False" in err for err in report.errors)


def test_10_obsolete_branding_rejected(temp_phase12_workspace):
    """Test 10: Obsolete branding tokens in package fail validation."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pkg_file = ep_dir / "publish" / "publish_package.json"
    with open(pkg_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["seo"]["description"] += " Host: Vennila"
    with open(pkg_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("Obsolete branding tokens" in err for err in report.errors)


def test_11_unsupported_metadata_rejected(temp_phase12_workspace):
    """Test 11: Sensational clickbait in title fails validation."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pkg_file = ep_dir / "publish" / "publish_package.json"
    with open(pkg_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["seo"]["selected_title"] = "SHOCKING 100% PROOF சோழர்கள் வரலாறு"
    with open(pkg_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    validator = YouTubePublicationValidator(sm)
    report = validator.validate_package(ep_id)
    assert report.passed is False
    assert any("Clickbait" in err or "Gate 21" in err for err in report.errors)


# ==============================================================================
# Tests 12-20: Security, Client Safety Lock & Default Modes
# ==============================================================================

def test_12_integration_disabled_by_default():
    """Test 12: YouTube integration is disabled by default in config."""
    cfg = AutonomousSettings()
    assert cfg.youtube_integration_enabled is False


def test_13_no_real_api_call_when_disabled():
    """Test 13: RealYouTubeClient raises YouTubeIntegrationDisabledError when integration is disabled."""
    cfg = AutonomousSettings()
    cfg.youtube_integration_enabled = False
    with pytest.raises(YouTubeIntegrationDisabledError) as exc_info:
        RealYouTubeClient(settings=cfg)
    assert "YOUTUBE_INTEGRATION_ENABLED=false" in str(exc_info.value)
    assert exc_info.value.classification == ErrorClassification.SAFE_CONFIGURATION_BLOCK


def test_14_dry_run_zero_network(temp_phase12_workspace):
    """Test 14: Dry-run performs 0 network requests and returns structured audit."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)
    dry_res = pm.execute_dry_run(ep_id)
    assert dry_res["validation_passed"] is True
    assert dry_res["verdict"] == "READY_FOR_FUTURE_YOUTUBE_INTEGRATION"
    assert dry_res["integration_status"]["real_api_calls"] == 0
    assert dry_res["integration_status"]["client_mode"] == "MOCK"


def test_15_mock_upload_success(temp_phase12_workspace):
    """Test 15: Mock client uploads video and generates deterministic ID."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    video_path = ep_dir / "master" / "master.mp4"
    res = client.upload_video(
        episode_id=ep_id,
        video_path=video_path,
        title="Test Title",
        description="Test Desc",
        tags=["tag1"],
        privacy_status="MOCK_PRIVATE"
    )
    assert res["id"].startswith("MOCK_")
    assert res["provider"] == "mock"
    assert res["real_api_called"] is False
    assert res["status"] == "UPLOADED"


def test_16_mock_thumbnail_success(temp_phase12_workspace):
    """Test 16: Mock client uploads thumbnail and attaches to mock video."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    video_path = ep_dir / "master" / "master.mp4"
    thumb_path = ep_dir / "publish" / "thumbnail.jpg"

    vid_res = client.upload_video(
        episode_id=ep_id,
        video_path=video_path,
        title="Test",
        description="Test",
        tags=[]
    )
    thumb_res = client.upload_thumbnail(vid_res["id"], thumb_path)
    assert thumb_res["thumbnail_status"] == "UPLOADED"
    assert thumb_res["real_api_called"] is False


def test_17_mock_verification_success(temp_phase12_workspace):
    """Test 17: Mock verification succeeds for video and thumbnail."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    video_path = ep_dir / "master" / "master.mp4"
    thumb_path = ep_dir / "publish" / "thumbnail.jpg"

    vid_res = client.upload_video(
        episode_id=ep_id,
        video_path=video_path,
        title="Exact Title",
        description="Desc",
        tags=[],
        privacy_status="MOCK_PRIVATE"
    )
    client.upload_thumbnail(vid_res["id"], thumb_path)

    v_ver = client.verify_video(vid_res["id"], expected_title="Exact Title", expected_privacy="MOCK_PRIVATE")
    assert v_ver["verification_status"] == "VERIFIED"

    t_ver = client.verify_thumbnail(vid_res["id"])
    assert t_ver["thumbnail_verified"] is True


def test_18_mock_private_mode(temp_phase12_workspace):
    """Test 18: Mock upload defaults to MOCK_PRIVATE privacy status."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)
    ok, manifest, _ = pm.execute_mock_upload(ep_id, publication_mode="MOCK_PRIVATE")
    assert ok is True
    assert manifest["privacy_status"] in ("MOCK_PRIVATE", "private")


def test_19_public_mode_blocked():
    """Test 19: auto_publish_public is disabled by default."""
    cfg = AutonomousSettings()
    assert cfg.auto_publish_public is False


def test_20_future_channel_id_not_invented():
    """Test 20: youtube_channel_id remains None (unset), never invented."""
    cfg = AutonomousSettings()
    assert cfg.youtube_channel_id is None


# ==============================================================================
# Tests 21-30: Upload Manager, Verification & Manifest Provenance
# ==============================================================================

def test_21_payload_generated_from_package(temp_phase12_workspace):
    """Test 21: Payload is assembled strictly from package without re-generation."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    um = YouTubeUploadManager(MockYouTubeClient(), sm)
    with open(ep_dir / "publish" / "publish_package.json", "r", encoding="utf-8") as f:
        pkg = json.load(f)
    payload = um.prepare_payload(ep_dir, pkg)
    assert payload["title"] == pkg["seo"]["selected_title"]
    assert payload["description"] == pkg["seo"]["description"]
    assert payload["tags"] == pkg["seo"]["tags"]


def test_22_category_correctness(temp_phase12_workspace):
    """Test 22: Upload category defaults to Education (ID 27)."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    um = YouTubeUploadManager(MockYouTubeClient(), sm)
    with open(ep_dir / "publish" / "publish_package.json", "r", encoding="utf-8") as f:
        pkg = json.load(f)
    payload = um.prepare_payload(ep_dir, pkg)
    assert payload["category_id"] == "27"


def test_23_made_for_kids_explicit_setting(temp_phase12_workspace):
    """Test 23: made_for_kids is explicitly False."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    um = YouTubeUploadManager(MockYouTubeClient(), sm)
    with open(ep_dir / "publish" / "publish_package.json", "r", encoding="utf-8") as f:
        pkg = json.load(f)
    payload = um.prepare_payload(ep_dir, pkg)
    assert payload["made_for_kids"] is False


def test_24_duplicate_local_upload_prevention(temp_phase12_workspace):
    """Test 24: Reusing existing video ID prevents duplicate upload calls."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    um = YouTubeUploadManager(client, sm)
    with open(ep_dir / "publish" / "publish_package.json", "r", encoding="utf-8") as f:
        pkg = json.load(f)

    # First upload
    res1 = um.execute_upload(ep_id, ep_dir, pkg)
    assert res1.success is True

    # Second upload with existing_video_id: must reuse resource without error
    res2 = um.execute_upload(ep_id, ep_dir, pkg, existing_video_id=res1.video_id)
    assert res2.success is True
    assert res2.video_id == res1.video_id


def test_25_duplicate_mock_resource_prevention(temp_phase12_workspace):
    """Test 25: Attempting to recreate the same mock video ID raises YouTubeDuplicateError."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    video_path = ep_dir / "master" / "master.mp4"
    client.upload_video(ep_id, video_path, "Title 1", "Desc", [])

    with pytest.raises(YouTubeDuplicateError) as exc_info:
        client.upload_video(ep_id, video_path, "Title 2", "Desc", [])
    assert "Duplicate mock upload blocked" in str(exc_info.value)


def test_26_existing_publication_verification(temp_phase12_workspace):
    """Test 26: Verifying already-uploaded resource retrieves stored metadata."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    video_path = ep_dir / "master" / "master.mp4"
    thumb_path = ep_dir / "publish" / "thumbnail.jpg"

    vid_res = client.upload_video(ep_id, video_path, "Title", "Desc", [])
    client.upload_thumbnail(vid_res["id"], thumb_path)

    stored = client.get_video(vid_res["id"])
    assert stored is not None
    assert stored["id"] == vid_res["id"]
    assert stored["thumbnail_uploaded"] is True


def test_27_thumbnail_failure_recovery(temp_phase12_workspace):
    """Test 27: Thumbnail failure preserves video resource and allows thumbnail retry."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    client.fail_next_thumbnail = True  # Inject failure on thumbnail

    um = YouTubeUploadManager(client, sm)
    with open(ep_dir / "publish" / "publish_package.json", "r", encoding="utf-8") as f:
        pkg = json.load(f)

    res = um.execute_upload(ep_id, ep_dir, pkg)
    assert res.success is False
    assert res.video_uploaded is True
    assert res.thumbnail_uploaded is False
    assert res.video_id is not None

    # Video is preserved in client store
    assert client.get_video(res.video_id) is not None

    # Retry thumbnail upload independently
    retry_thumb = client.upload_thumbnail(res.video_id, ep_dir / "publish" / "thumbnail.jpg")
    assert retry_thumb["thumbnail_status"] == "UPLOADED"


def test_28_video_preserved_after_thumbnail_failure(temp_phase12_workspace):
    """Test 28: Video is never unlinked or removed when thumbnail upload fails."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    client.fail_next_thumbnail = True

    um = YouTubeUploadManager(client, sm)
    with open(ep_dir / "publish" / "publish_package.json", "r", encoding="utf-8") as f:
        pkg = json.load(f)

    res = um.execute_upload(ep_id, ep_dir, pkg)
    # Master video file on disk remains intact
    assert (ep_dir / "master" / "master.mp4").exists()
    assert (ep_dir / "master" / "master.mp4").stat().st_size > 1000


def test_29_publication_manifest_generation(temp_phase12_workspace):
    """Test 29: Manifest youtube_publication_manifest.json is atomically created."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)
    ok, manifest, msg = pm.execute_mock_upload(ep_id)
    assert ok is True
    manifest_file = ep_dir / "publish" / "youtube_publication_manifest.json"
    assert manifest_file.exists()
    assert manifest_file.stat().st_size > 100


def test_30_manifest_provenance(temp_phase12_workspace):
    """Test 30: Manifest records accurate cryptographic provenance and mock identity."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)
    ok, manifest, _ = pm.execute_mock_upload(ep_id)
    assert ok is True
    assert manifest["provider"] == "mock"
    assert manifest["mock"] is True
    assert manifest["real_api_called"] is False
    assert manifest["channel_id"] is None
    assert "master_sha256" in manifest
    assert "thumbnail_sha256" in manifest
    assert "description_hash" in manifest
    assert "tags_hash" in manifest


# ==============================================================================
# Tests 31-40: Idempotency, Recovery, Immutability & Security
# ==============================================================================

def test_31_idempotent_second_execution(temp_phase12_workspace):
    """Test 31: Second mock publication execution is idempotent and reuses manifest."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)

    ok1, manifest1, _ = pm.execute_mock_upload(ep_id)
    assert ok1 is True

    manifest_path = ep_dir / "publish" / "youtube_publication_manifest.json"
    mtime1 = manifest_path.stat().st_mtime_ns

    # Second execution
    ok2, manifest2, msg2 = pm.execute_mock_upload(ep_id)
    assert ok2 is True
    assert "already published and verified" in msg2
    assert manifest2["youtube_video_id"] == manifest1["youtube_video_id"]
    # File was not needlessly modified
    assert manifest_path.stat().st_mtime_ns == mtime1


def test_32_interrupted_upload_recovery(temp_phase12_workspace):
    """Test 32: RecoveryManager safely inspects and resumes interrupted upload states."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    rm = RecoveryManager(state_manager=sm)

    # In YOUTUBE_UPLOADING without manifest
    sm.transition(ep_id, EpisodeState.YOUTUBE_VALIDATING)
    sm.transition(ep_id, EpisodeState.YOUTUBE_READY)
    sm.transition(ep_id, EpisodeState.YOUTUBE_UPLOADING)

    decision = rm.inspect_active_episode()
    assert decision is not None
    assert decision.action == "RETRY_STAGE"
    assert decision.target_state == EpisodeState.YOUTUBE_UPLOADING


def test_33_verification_failure(temp_phase12_workspace):
    """Test 33: Client verification failure transitions episode to PUBLICATION_REVIEW_REQUIRED."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    client = MockYouTubeClient()
    client.fail_next_verification = True

    pm = PublicationManager(state_manager=sm, client=client)
    ok, manifest, msg = pm.execute_mock_upload(ep_id)
    assert ok is False

    ep = sm.get_episode(ep_id)
    assert ep.status == EpisodeState.PUBLICATION_REVIEW_REQUIRED.value


def test_34_retryable_error_handling():
    """Test 34: YouTubeUploadError classifies retryable transient errors."""
    err = YouTubeUploadError("Temporary network glitch", retryable=True)
    assert err.classification == ErrorClassification.RETRYABLE


def test_35_non_retryable_error_handling():
    """Test 35: Permanent upload error classifies as NON_RETRYABLE."""
    err = YouTubeUploadError("Permanent payload violation", retryable=False)
    assert err.classification == ErrorClassification.NON_RETRYABLE


def test_36_state_transition_correctness(temp_phase12_workspace):
    """Test 36: Full state sequence reaches PUBLICATION_VERIFIED."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)
    ok, _, _ = pm.execute_mock_upload(ep_id)
    assert ok is True

    ep = sm.get_episode(ep_id)
    assert ep.status == EpisodeState.PUBLICATION_VERIFIED.value
    assert ep.youtube_status == "MOCK_VERIFIED"


def test_37_phase11_immutability(temp_phase12_workspace):
    """Test 37: Phase 11 artifacts remain completely unchanged after mock upload."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    seo_file = ep_dir / "publish" / "seo.json"
    thumb_file = ep_dir / "publish" / "thumbnail.jpg"
    pkg_file = ep_dir / "publish" / "publish_package.json"

    sha_seo_before = _sha256_file(seo_file)
    sha_thumb_before = _sha256_file(thumb_file)
    sha_pkg_before = _sha256_file(pkg_file)

    pm = PublicationManager(state_manager=sm)
    pm.execute_mock_upload(ep_id)

    assert _sha256_file(seo_file) == sha_seo_before
    assert _sha256_file(thumb_file) == sha_thumb_before
    assert _sha256_file(pkg_file) == sha_pkg_before


def test_38_phase10_master_immutability(temp_phase12_workspace):
    """Test 38: Phase 10 master video and manifest remain byte-identical."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    master_mp4 = ep_dir / "master" / "master.mp4"
    master_manifest = ep_dir / "master" / "master_manifest.json"

    sha_master_before = _sha256_file(master_mp4)
    sha_man_before = _sha256_file(master_manifest)

    pm = PublicationManager(state_manager=sm)
    pm.execute_mock_upload(ep_id)

    assert _sha256_file(master_mp4) == sha_master_before
    assert _sha256_file(master_manifest) == sha_man_before


def test_39_20260910_002_immutability():
    """Test 39: Episode 20260910-002 safeguard is maintained."""
    ep_dir = BASE_DIR / "outputs" / "episodes" / "20260910-002"
    if ep_dir.exists():
        manifest = ep_dir / "publish" / "youtube_publication_manifest.json"
        assert not manifest.exists()


def test_40_secret_sanitization(temp_phase12_workspace):
    """Test 40: No secrets, access tokens, or credentials are recorded in manifest."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)
    ok, manifest, _ = pm.execute_mock_upload(ep_id)
    assert ok is True

    manifest_text = json.dumps(manifest).lower()
    for forbidden in ("access_token", "refresh_token", "client_secret", "bearer", "authorization_code"):
        assert forbidden not in manifest_text


# ==============================================================================
# Tests 41-48: Network Isolation, CLI & Boundary Tests
# ==============================================================================

def test_41_network_isolation(temp_phase12_workspace):
    """Test 41: Mock publication succeeds with network sockets completely disabled."""
    sm, ep_dir, ep_id = temp_phase12_workspace

    # Block socket connection attempt
    def blocked_socket(*args, **kwargs):
        raise RuntimeError("CRITICAL ERROR: Real network access attempted during offline Phase 12 test!")

    with patch("socket.socket", side_effect=blocked_socket):
        pm = PublicationManager(state_manager=sm)
        ok, manifest, _ = pm.execute_mock_upload(ep_id)
        assert ok is True
        assert manifest["provider"] == "mock"
        assert manifest["real_api_called"] is False


def test_42_cli_dry_run(temp_phase12_workspace, capsys):
    """Test 42: CLI handle_youtube_dry_run executes cleanly without error."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    rm = RecoveryManager(state_manager=sm)
    orch = AutonomousOrchestrator(sm, rm)

    handle_youtube_dry_run(orch, episode_id=ep_id)
    captured = capsys.readouterr().out
    assert "YOUTUBE PUBLICATION DRY RUN" in captured
    assert "YouTube integration: DISABLED" in captured
    assert "Real API calls: 0" in captured


def test_43_cli_mock_upload(temp_phase12_workspace, capsys):
    """Test 43: CLI handle_mock_youtube_upload executes cleanly."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    rm = RecoveryManager(state_manager=sm)
    orch = AutonomousOrchestrator(sm, rm)

    handle_mock_youtube_upload(orch, episode_id=ep_id)
    captured = capsys.readouterr().out
    assert "AUTONOMOUS MOCK YOUTUBE PUBLICATION" in captured
    assert "Real YouTube Upload : NO" in captured
    assert "Mock YouTube Upload : YES" in captured
    assert "Publication Status  : SUCCESS" in captured


def test_44_status_reporting(temp_phase12_workspace, capsys):
    """Test 44: CLI handle_youtube_status reports required strings."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    rm = RecoveryManager(state_manager=sm)
    orch = AutonomousOrchestrator(sm, rm)

    # Upload first
    orch.mock_upload_youtube(ep_id)
    handle_youtube_status(sm, orch, episode_id=ep_id)
    captured = capsys.readouterr().out

    assert "MOCK_VERIFIED" in captured
    assert "Real YouTube Upload : NOT PERFORMED" in captured
    assert "Mock YouTube Upload : YES" in captured
    assert "YouTube Channel     : NOT CONFIGURED" in captured
    assert "Google OAuth        : NOT CONFIGURED" in captured
    assert "Real Integration    : DORMANT / DISABLED" in captured


def test_45_no_fake_channel_id(temp_phase12_workspace):
    """Test 45: Generated publication manifest explicitly records channel_id as null."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)
    ok, manifest, _ = pm.execute_mock_upload(ep_id)
    assert ok is True
    assert manifest["channel_id"] is None


def test_46_mock_vs_real_distinction(temp_phase12_workspace):
    """Test 46: Manifest explicitly distinguishes provider='mock' and real_api_called=False."""
    sm, ep_dir, ep_id = temp_phase12_workspace
    pm = PublicationManager(state_manager=sm)
    ok, manifest, _ = pm.execute_mock_upload(ep_id)
    assert ok is True
    assert manifest["provider"] == "mock"
    assert manifest["mock"] is True
    assert manifest["real_api_called"] is False


def test_47_publication_verified_terminal_state():
    """Test 47: PUBLICATION_VERIFIED is terminal with no outgoing transitions."""
    transitions = VALID_TRANSITIONS.get(EpisodeState.PUBLICATION_VERIFIED, [])
    assert len(transitions) == 0
    assert EpisodeState.PUBLICATION_VERIFIED in TERMINAL_STATES


def test_48_publish_package_ready_consumption_boundary():
    """Test 48: PUBLISH_PACKAGE_READY invariant preserved in VALID_TRANSITIONS and consumed cleanly."""
    # From Phase 11 perspective: VALID_TRANSITIONS is empty
    transitions = VALID_TRANSITIONS.get(EpisodeState.PUBLISH_PACKAGE_READY, [])
    assert len(transitions) == 0

    # From Phase 12 perspective: consumed via PHASE_CONSUMPTION_TRANSITIONS
    assert EpisodeState.YOUTUBE_VALIDATING in PHASE_CONSUMPTION_TRANSITIONS[EpisodeState.PUBLISH_PACKAGE_READY]
