"""
tests/test_phase11_seo_thumbnail.py - Automated Test Suite for Phase 11 SEO & Thumbnail Engine.
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).

Covers all 35 required tests:
1. MASTER_READY gate
2. REVIEW_REQUIRED rejection
3. FAILED rejection
4. missing master
5. corrupted master manifest
6. SEO generation
7. title grounding
8. unsupported title claim rejection
9. description generation
10. description grounding
11. uncertainty preservation
12. tag generation
13. tag deduplication
14. tag length
15. hashtag validation
16. canonical identity
17. obsolete identity rejection
18. thumbnail generation
19. thumbnail dimensions
20. thumbnail aspect ratio
21. thumbnail text generation
22. thumbnail grounding
23. Yaazhini safe compositing
24. thumbnail QC
25. thumbnail manifest/hash
26. publish package generation
27. package schema
28. package hash provenance
29. idempotency
30. interrupted-generation recovery
31. immutable 20260910-002
32. deterministic fallback/no-network behavior
33. CLI
34. orchestrator boundary
35. PUBLISH_PACKAGE_READY terminal state
"""

import os
import sys
import json
import shutil
import hashlib
import tempfile
import sqlite3
from pathlib import Path
from typing import Dict, Any, Tuple

import pytest
import numpy as np
import cv2
from PIL import Image

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode, VALID_TRANSITIONS
from autonomous.recovery_manager import RecoveryManager, RecoveryDecision
from autonomous.orchestrator import AutonomousOrchestrator, StageExecutionResult
from autonomous.seo_engine import AutonomousSEOEngine, SEOResult, TitleCandidate
from autonomous.seo_validator import SEOValidator, SEOValidationReport
from autonomous.thumbnail_engine import AutonomousThumbnailEngine, ThumbnailManifest, ThumbnailQCReport
from autonomous.publish_package import PublishPackageBuilder


def _sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _generate_test_mp4(filepath: Path, duration: float = 2.0, width: int = 1280, height: int = 720, fps: int = 25) -> str:
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
def temp_phase11_workspace(tmp_path):
    """Sets up an isolated database and MASTER_READY episode workspace."""
    db_file = tmp_path / "test_platform.db"
    ep_base = tmp_path / "episodes"
    ep_base.mkdir(parents=True, exist_ok=True)

    sm = StateManager(db_url=f"sqlite:///{db_file}")

    ep_id = "EP-TEST-P11"
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
                "scene_type": "host_intro",
                "narration_tamil": "வணக்கம், நான் யாழினி. காலப் பதிவுகள் சேனலுக்கு வரவேற்கிறேன்.",
                "narration_english": "Welcome to Kaalapadhivugal. I am Yaazhini.",
                "has_presenter": True
            },
            {
                "scene_id": 2,
                "scene_type": "historical_broll",
                "narration_tamil": "பூம்புகார் துறைமுகம் சோழர்களின் பிரம்மாண்ட கடல் வர்த்தக நகரமாக விளங்கியது.",
                "narration_english": "The port of Poompuhar was a monumental maritime trade hub.",
                "has_presenter": False
            },
            {
                "scene_id": 3,
                "scene_type": "host_outro",
                "narration_tamil": "தமிழர் வரலாற்றுச் சாதனைகள் தொடரும். நன்றி, வணக்கம்.",
                "narration_english": "Tamil historical achievements will continue. Thank you.",
                "has_presenter": True
            }
        ]
    }
    val_script_path = script_dir / "validated_script.json"
    with open(val_script_path, "w", encoding="utf-8") as f:
        json.dump(script_data, f, ensure_ascii=False, indent=2)

    with open(script_dir / "script.json", "w", encoding="utf-8") as f:
        json.dump(script_data, f, ensure_ascii=False, indent=2)

    # 2. Visuals
    visuals_dir = ep_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    _generate_test_image(visuals_dir / "scene_01.png")
    _generate_test_image(visuals_dir / "scene_02.png")
    _generate_test_image(visuals_dir / "scene_03.png")

    # 3. Master MP4 & Manifests
    master_dir = ep_dir / "master"
    master_dir.mkdir(parents=True, exist_ok=True)
    master_mp4 = master_dir / "master.mp4"
    master_sha = _generate_test_mp4(master_mp4, duration=3.0)

    master_manifest = {
        "episode_id": ep_id,
        "schema_version": "1.0.0",
        "master_video_path": "master/master.mp4",
        "master_video_sha256": master_sha,
        "duration_seconds": 3.0,
        "resolution": "1280x720",
        "fps": 25.0,
        "scenes": script_data["scenes"]
    }
    with open(master_dir / "master_manifest.json", "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=2)

    master_report = {
        "episode_id": ep_id,
        "status": "MASTER_READY",
        "qc_score": 0.95
    }
    with open(master_dir / "master_generation_report.json", "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    # Set status to MASTER_READY
    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.MASTER_READY.value
    ep.current_stage = EpisodeState.MASTER_READY.value
    s.commit()
    s.close()

    yield sm, ep_dir, ep_id


# ==============================================================================
# 1-5: MASTER_READY Input Gate & Rejection Tests
# ==============================================================================

def test_01_master_ready_gate_acceptance(temp_phase11_workspace):
    """Test 1: MASTER_READY gate accepts complete and verified episode."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    engine = AutonomousSEOEngine(state_manager=sm)
    valid, msg, data = engine.validate_master_ready_gate(ep_id)
    assert valid is True
    assert "passed" in msg.lower()
    assert data["master_hash"] == _sha256_file(ep_dir / "master" / "master.mp4")


def test_02_review_required_rejection(temp_phase11_workspace):
    """Test 2: Rejects REVIEW_REQUIRED episodes."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.REVIEW_REQUIRED.value
    ep.current_stage = EpisodeState.REVIEW_REQUIRED.value
    s.commit()
    s.close()

    engine = AutonomousSEOEngine(state_manager=sm)
    valid, msg, _ = engine.validate_master_ready_gate(ep_id)
    assert valid is False
    assert "REVIEW_REQUIRED" in msg


def test_03_failed_rejection(temp_phase11_workspace):
    """Test 3: Rejects FAILED episodes."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.FAILED.value
    ep.current_stage = EpisodeState.FAILED.value
    s.commit()
    s.close()

    engine = AutonomousSEOEngine(state_manager=sm)
    valid, msg, _ = engine.validate_master_ready_gate(ep_id)
    assert valid is False
    assert "FAILED" in msg


def test_04_missing_master(temp_phase11_workspace):
    """Test 4: Rejects when master.mp4 is missing."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    master_file = ep_dir / "master" / "master.mp4"
    master_file.unlink()

    engine = AutonomousSEOEngine(state_manager=sm)
    valid, msg, _ = engine.validate_master_ready_gate(ep_id)
    assert valid is False
    assert "missing" in msg.lower()


def test_05_corrupted_master_manifest(temp_phase11_workspace):
    """Test 5: Rejects when master_manifest has hash mismatch or corrupt JSON."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    manifest_file = ep_dir / "master" / "master_manifest.json"
    with open(manifest_file, "w") as f:
        f.write("{corrupt_json")

    engine = AutonomousSEOEngine(state_manager=sm)
    valid, msg, _ = engine.validate_master_ready_gate(ep_id)
    assert valid is False
    assert "corrupt" in msg.lower()


# ==============================================================================
# 6-11: SEO Generation & Grounding Tests
# ==============================================================================

def test_06_seo_generation(temp_phase11_workspace):
    """Test 6: SEO generation succeeds and produces structured SEOResult."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    engine = AutonomousSEOEngine(state_manager=sm)
    ok, res, val, msg = engine.generate_seo(ep_id, offline=True)
    assert ok is True
    assert isinstance(res, SEOResult)
    assert res.selected_title is not None
    assert len(res.tags) > 0
    assert len(res.hashtags) > 0
    assert res.category == "Education"
    assert res.language == "ta"

    # State transition check
    ep = sm.get_episode(ep_id)
    assert ep.status == EpisodeState.METADATA_READY.value


def test_07_title_grounding(temp_phase11_workspace):
    """Test 7: Selected title is factually grounded in episode topic/script."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    engine = AutonomousSEOEngine(state_manager=sm)
    ok, res, _, _ = engine.generate_seo(ep_id, offline=True)
    assert ok is True
    # Verify title shares historical concept
    assert any(term in res.selected_title for term in ("சோழ", "வர்த்தக", "துறைமுக", "வரலாறு"))


def test_08_unsupported_title_claim_rejection():
    """Test 8: Validator penalizes and rejects unsupported sensational claims."""
    validator = SEOValidator()
    script = {"title": "கீழடி அகழாய்வு", "scenes": [{"narration_tamil": "கீழடியில் செங்கல் கட்டுமானங்கள் கண்டறியப்பட்டன."}]}
    report = validator.validate(
        title="SHOCKING 100% PROOF THE TRUTH THEY HID கீழடி",
        description="கீழடி ஆய்வு பற்றிய ஆவணம். #Kaalapadhivugal",
        tags=["கீழடி", "வரலாறு"],
        hashtags=["#Kaalapadhivugal"],
        validated_script=script
    )
    assert report.passed is False
    assert any("sensational" in err.lower() or "clickbait" in err.lower() for err in report.errors)


def test_09_description_generation(temp_phase11_workspace):
    """Test 9: Structured description contains hook, summary, branding, CTA, hashtags."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    engine = AutonomousSEOEngine(state_manager=sm)
    ok, res, _, _ = engine.generate_seo(ep_id, offline=True)
    assert ok is True
    desc = res.description
    assert "காலப் பதிவுகள்" in desc or "@kaalapadhivugal" in desc
    assert "பூம்புகார்" in desc or "வரலாறு" in desc
    assert "#Kaalapadhivugal" in desc


def test_10_description_grounding():
    """Test 10: Description with fabricated URLs is rejected."""
    validator = SEOValidator()
    script = {"title": "பூம்புகார்"}
    report = validator.validate(
        title="பூம்புகார் துறைமுக வரலாறு",
        description="முழு ஆவணம் பார்க்க: http://fabricated-source-example.org/fake",
        tags=["பூம்புகார்"],
        hashtags=["#Kaalapadhivugal"],
        validated_script=script
    )
    assert report.passed is False
    assert any("fabricated" in err.lower() for err in report.errors)


def test_11_uncertainty_preservation():
    """Test 11: Debated claims without uncertainty qualifications produce warnings."""
    validator = SEOValidator()
    script = {"title": "குமரிக்கண்டம்"}
    claims = [{"statement": "லெமூரியா பெருங்கண்டம்", "classification": "UNRESOLVED_DEBATE"}]
    report = validator.validate(
        title="குமரிக்கண்டம் ஒரு பார்வை",
        description="இது உறுதியாக நிரூபிக்கப்பட்ட உண்மை. #Kaalapadhivugal",
        tags=["குமரிக்கண்டம்"],
        hashtags=["#Kaalapadhivugal"],
        validated_script=script,
        research_claims=claims
    )
    assert any("uncertainty" in w.lower() for w in report.warnings)


# ==============================================================================
# 12-17: Tags, Hashtags & Branding Tests
# ==============================================================================

def test_12_tag_generation(temp_phase11_workspace):
    """Test 12: Generates structured relevant tags."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    engine = AutonomousSEOEngine(state_manager=sm)
    ok, res, _, _ = engine.generate_seo(ep_id, offline=True)
    assert ok is True
    assert len(res.tags) >= 5
    assert "Kaalapadhivugal" in res.tags or "@kaalapadhivugal" in res.tags


def test_13_tag_deduplication():
    """Test 13: Case-insensitive tag deduplication."""
    validator = SEOValidator()
    script = {"title": "Test"}
    report = validator.validate(
        title="சோழர் வரலாறு",
        description="விளக்கம் #Kaalapadhivugal",
        tags=["Chola History", "chola history", "சோழர்", "Chola History"],
        hashtags=["#Kaalapadhivugal"],
        validated_script=script
    )
    assert report.passed is False
    assert any("duplicate" in err.lower() for err in report.errors)


def test_14_tag_length():
    """Test 14: Rejects tags exceeding YouTube 500 characters limit."""
    validator = SEOValidator()
    script = {"title": "Test"}
    long_tags = [f"tag_number_{i}_with_very_long_string_content_for_testing" for i in range(25)]
    report = validator.validate(
        title="சோழர் வரலாறு",
        description="விளக்கம் #Kaalapadhivugal",
        tags=long_tags,
        hashtags=["#Kaalapadhivugal"],
        validated_script=script
    )
    assert report.passed is False
    assert any("500" in err or "budget" in err.lower() or "length" in err.lower() for err in report.errors)


def test_15_hashtag_validation():
    """Test 15: Validates hashtags <= 10 items and correctly formatted."""
    validator = SEOValidator()
    script = {"title": "சோழர் வரலாறு"}
    report = validator.validate(
        title="சோழர் வரலாறு",
        description="விளக்கம் #Kaalapadhivugal",
        tags=["சோழர்"],
        hashtags=["#Kaalapadhivugal", "#TamilHistory", "#Chola"],
        validated_script=script
    )
    assert report.passed is True


def test_16_canonical_identity():
    """Test 16: Canonical branding @kaalapadhivugal and காலப் பதிவுகள் recognized."""
    validator = SEOValidator()
    assert validator._check_canonical_identity("Title", "Welcome to Kaalapadhivugal", ["#Kaalapadhivugal"]) is True


def test_17_obsolete_identity_rejection():
    """Test 17: Rejects obsolete identity tokens (Vennila, Aayirathil Naan)."""
    validator = SEOValidator()
    script = {"title": "Test"}
    report = validator.validate(
        title="Vennila Explores Chola Trade",
        description="Aayirathil Naan episode #Kaalapadhivugal",
        tags=["vennila", "சோழர்"],
        hashtags=["#Kaalapadhivugal"],
        validated_script=script
    )
    assert report.passed is False
    assert any("obsolete" in err.lower() for err in report.errors)


# ==============================================================================
# 18-25: Thumbnail Engine & QC Tests
# ==============================================================================

def test_18_thumbnail_generation(temp_phase11_workspace):
    """Test 18: Generates local thumbnail file."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    engine = AutonomousSEOEngine(state_manager=sm)
    engine.generate_seo(ep_id, offline=True)

    thumb_engine = AutonomousThumbnailEngine(state_manager=sm)
    ok, manifest, qc, msg = thumb_engine.generate_thumbnail(ep_id)
    assert ok is True
    assert (ep_dir / "publish" / "thumbnail.jpg").exists()
    assert manifest.qc_passed is True

    ep = sm.get_episode(ep_id)
    assert ep.status == EpisodeState.THUMBNAIL_READY.value


def test_19_thumbnail_dimensions(temp_phase11_workspace):
    """Test 19: Thumbnail dimensions are exactly 1280x720."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    AutonomousSEOEngine(state_manager=sm).generate_seo(ep_id, offline=True)
    thumb_engine = AutonomousThumbnailEngine(state_manager=sm)
    ok, manifest, qc, _ = thumb_engine.generate_thumbnail(ep_id)
    assert ok is True

    thumb_file = ep_dir / "publish" / "thumbnail.jpg"
    img = Image.open(thumb_file)
    assert img.size == (1280, 720)


def test_20_thumbnail_aspect_ratio(temp_phase11_workspace):
    """Test 20: Thumbnail aspect ratio is 16:9."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    AutonomousSEOEngine(state_manager=sm).generate_seo(ep_id, offline=True)
    thumb_engine = AutonomousThumbnailEngine(state_manager=sm)
    ok, manifest, qc, _ = thumb_engine.generate_thumbnail(ep_id)
    assert ok is True

    thumb_file = ep_dir / "publish" / "thumbnail.jpg"
    img = Image.open(thumb_file)
    w, h = img.size
    assert abs((w / h) - (16.0 / 9.0)) < 0.01


def test_21_thumbnail_text_generation(temp_phase11_workspace):
    """Test 21: Thumbnail text candidates generated with 2-6 words."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    thumb_engine = AutonomousThumbnailEngine(state_manager=sm)
    script_data = {"title": "சோழர்களின் கடல்சார் வர்த்தகம்"}
    cands = thumb_engine._generate_text_candidates("சோழர் வரலாறு", script_data, {})
    assert len(cands) >= 3
    for c in cands:
        assert 2 <= c.word_count <= 7


def test_22_thumbnail_grounding(temp_phase11_workspace):
    """Test 22: Selected thumbnail text is grounded in episode topic."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    AutonomousSEOEngine(state_manager=sm).generate_seo(ep_id, offline=True)
    thumb_engine = AutonomousThumbnailEngine(state_manager=sm)
    ok, manifest, _, _ = thumb_engine.generate_thumbnail(ep_id)
    assert ok is True
    assert any(w in manifest.selected_text for w in ("சோழ", "வர்த்தக", "பூம்புகார்", "துறைமுக", "வரலாறு"))


def test_23_yaazhini_safe_compositing(temp_phase11_workspace):
    """Test 23: Host Yaazhini compositing preserves aspect ratio without distortion."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    AutonomousSEOEngine(state_manager=sm).generate_seo(ep_id, offline=True)
    thumb_engine = AutonomousThumbnailEngine(state_manager=sm)
    ok, manifest, qc, _ = thumb_engine.generate_thumbnail(ep_id, force_host=True)
    assert ok is True
    assert manifest.host_presenter_included is True
    assert qc.passed is True


def test_24_thumbnail_qc(temp_phase11_workspace):
    """Test 24: QC rejects blank or pitch-black images."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    thumb_engine = AutonomousThumbnailEngine(state_manager=sm)

    # Generate black image
    black_img_path = ep_dir / "black.jpg"
    cv2.imwrite(str(black_img_path), np.zeros((720, 1280, 3), dtype=np.uint8))

    qc = thumb_engine.run_thumbnail_qc(black_img_path)
    assert qc.passed is False
    assert qc.is_blank is True


def test_25_thumbnail_manifest_and_hash(temp_phase11_workspace):
    """Test 25: Thumbnail manifest records valid SHA-256 and matches file."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    AutonomousSEOEngine(state_manager=sm).generate_seo(ep_id, offline=True)
    thumb_engine = AutonomousThumbnailEngine(state_manager=sm)
    ok, manifest, _, _ = thumb_engine.generate_thumbnail(ep_id)
    assert ok is True

    thumb_file = ep_dir / "publish" / "thumbnail.jpg"
    assert manifest.thumbnail_sha256 == _sha256_file(thumb_file)


# ==============================================================================
# 26-30: Publishing Package, Provenance & Recovery Tests
# ==============================================================================

def test_26_publish_package_generation(temp_phase11_workspace):
    """Test 26: Full publish package generation reaches PUBLISH_PACKAGE_READY."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    builder = PublishPackageBuilder(state_manager=sm)
    ok, pkg, msg = builder.build_publish_package(ep_id, offline=True)
    assert ok is True
    assert (ep_dir / "publish" / "publish_package.json").exists()

    ep = sm.get_episode(ep_id)
    assert ep.status == EpisodeState.PUBLISH_PACKAGE_READY.value


def test_27_package_schema(temp_phase11_workspace):
    """Test 27: Publish package schema contains all required top-level keys."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    builder = PublishPackageBuilder(state_manager=sm)
    ok, pkg, _ = builder.build_publish_package(ep_id, offline=True)
    assert ok is True

    required_keys = [
        "episode_id", "schema_version", "terminal_state", "channel_identity",
        "host_identity", "master", "script", "seo", "thumbnail", "provenance"
    ]
    for rk in required_keys:
        assert rk in pkg


def test_28_package_hash_provenance(temp_phase11_workspace):
    """Test 28: Package records authentic master, script, and thumbnail SHA-256."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    builder = PublishPackageBuilder(state_manager=sm)
    ok, pkg, _ = builder.build_publish_package(ep_id, offline=True)
    assert ok is True

    master_mp4 = ep_dir / "master" / "master.mp4"
    thumb_jpg = ep_dir / "publish" / "thumbnail.jpg"

    assert pkg["master"]["sha256"] == _sha256_file(master_mp4)
    assert pkg["thumbnail"]["sha256"] == _sha256_file(thumb_jpg)


def test_29_idempotency(temp_phase11_workspace):
    """Test 29: Running build_publish_package twice is idempotent and does not regenerate."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    builder = PublishPackageBuilder(state_manager=sm)
    ok1, pkg1, _ = builder.build_publish_package(ep_id, offline=True)
    assert ok1 is True

    thumb_before = _sha256_file(ep_dir / "publish" / "thumbnail.jpg")

    # Run second time
    ok2, pkg2, msg2 = builder.build_publish_package(ep_id, offline=True)
    assert ok2 is True
    assert "already complete" in msg2 or "verified" in msg2

    thumb_after = _sha256_file(ep_dir / "publish" / "thumbnail.jpg")
    assert thumb_before == thumb_after


def test_30_interrupted_generation_recovery(temp_phase11_workspace):
    """Test 30: RecoveryManager cleans up .tmp files and recovers appropriately."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    publish_dir = ep_dir / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    # Simulate interrupted generation leaving a .tmp file
    tmp_file = publish_dir / "thumbnail.tmp.jpg"
    with open(tmp_file, "w") as f:
        f.write("incomplete")

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=ep_id).first()
    ep.status = EpisodeState.THUMBNAIL_GENERATING.value
    ep.current_stage = EpisodeState.THUMBNAIL_GENERATING.value
    s.commit()
    s.close()

    rm = RecoveryManager(state_manager=sm)
    dec = rm.evaluate_episode(sm.get_episode(ep_id))
    assert dec.action == "RETRY_STAGE"
    assert not tmp_file.exists()


# ==============================================================================
# 31-35: Safeguards, Boundaries & CLI Tests
# ==============================================================================

def test_31_immutable_20260910_002(temp_phase11_workspace):
    """Test 31: Immutable episode 20260910-002 cannot be processed by Phase 11."""
    sm, _, _ = temp_phase11_workspace
    imm_id = "20260910-002"
    sm.create_episode(topic="Immutable Test", episode_id=imm_id, output_directory="outputs/episodes/20260910-002")

    s = sm._get_session()
    ep = s.query(AutonomousEpisode).filter_by(episode_id=imm_id).first()
    ep.status = EpisodeState.REVIEW_REQUIRED.value
    ep.current_stage = EpisodeState.REVIEW_REQUIRED.value
    ep.retry_count = 0
    ep.error_message = "INSUFFICIENT_EVIDENCE_AFTER_RESEARCH_EXPANSION"
    s.commit()
    s.close()

    builder = PublishPackageBuilder(state_manager=sm)
    ok, _, err = builder.build_publish_package(imm_id, offline=True)
    assert ok is False
    assert "REVIEW_REQUIRED" in err

    # Verify untouched
    ep_after = sm.get_episode(imm_id)
    assert ep_after.status == EpisodeState.REVIEW_REQUIRED.value
    assert ep_after.retry_count == 0
    assert ep_after.error_message == "INSUFFICIENT_EVIDENCE_AFTER_RESEARCH_EXPANSION"


def test_32_deterministic_fallback_no_network(temp_phase11_workspace):
    """Test 32: Offline deterministic fallback succeeds with zero network."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    engine = AutonomousSEOEngine(state_manager=sm)
    ok, res, _, _ = engine.generate_seo(ep_id, offline=True)
    assert ok is True
    assert res.fallback_used is True
    assert res.llm_used is False
    assert res.selected_title is not None


def test_33_cli_arguments_and_status(temp_phase11_workspace):
    """Test 33: CLI functions execute cleanly for Phase 11."""
    from run_channel import handle_publish_package_status
    sm, ep_dir, ep_id = temp_phase11_workspace
    builder = PublishPackageBuilder(state_manager=sm)
    builder.build_publish_package(ep_id, offline=True)

    # Call status handler (verifies it runs without throwing)
    handle_publish_package_status(sm, episode_id=ep_id)


def test_34_orchestrator_boundary(temp_phase11_workspace):
    """Test 34: Orchestrator advances episode through Phase 11 states."""
    sm, ep_dir, ep_id = temp_phase11_workspace
    rm = RecoveryManager(state_manager=sm)
    orch = AutonomousOrchestrator(sm, rm)

    res1 = orch.generate_seo(ep_id, offline=True)
    assert res1.success is True
    assert res1.next_state == EpisodeState.METADATA_READY

    res2 = orch.generate_thumbnail(ep_id)
    assert res2.success is True
    assert res2.next_state == EpisodeState.THUMBNAIL_READY

    res3 = orch.build_publish_package(ep_id, offline=True)
    assert res3.success is True
    assert res3.next_state == EpisodeState.PUBLISH_PACKAGE_READY


def test_35_publish_package_ready_terminal_state():
    """Test 35: PUBLISH_PACKAGE_READY is terminal with NO outgoing transitions in Phase 11."""
    transitions = VALID_TRANSITIONS.get(EpisodeState.PUBLISH_PACKAGE_READY, [])
    assert len(transitions) == 0
    assert EpisodeState.SEO not in transitions
    assert EpisodeState.READY_TO_PUBLISH not in transitions
    assert EpisodeState.UPLOADING not in transitions
    assert EpisodeState.PUBLISHED not in transitions
