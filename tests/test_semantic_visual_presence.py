"""
tests/test_semantic_visual_presence.py - Comprehensive Unit & Regression Tests
for Semantic Visual-Presence Gate (Phase 7) and Master QC (Phase 10).
"""

import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import cv2
from PIL import Image

from autonomous.config import BASE_DIR
from autonomous.visual_generator import (
    SemanticVisualValidator,
    DeterministicFallbackProvider,
    ImageValidator,
    ExistingLocalAssetProvider
)
from autonomous.master_qc import MasterQualityController, MasterQCReport


# -----------------------------------------------------------------------------
# Fixture Helpers
# -----------------------------------------------------------------------------

def create_synthetic_historical_image(tmp_path: Path, name: str, style: str = "photo") -> Path:
    """Create a high-detail synthetic image simulating photos, paintings, maps, or manuscripts."""
    img_path = tmp_path / name
    arr = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Base background
    if style == "dark":
        arr[:] = (20, 18, 16)  # Dark cave/relief background (mean < 30)
    elif style == "manuscript":
        arr[:] = (180, 195, 215)  # Aged palm-leaf / parchment
    elif style == "map":
        arr[:] = (190, 210, 225)  # Historical map parchment
    elif style == "painting":
        arr[:] = (45, 60, 90)  # Oil painting base
    else:
        arr[:] = (80, 95, 110)

    # Add rich high-frequency structures, contours, and line details
    rng = np.random.RandomState(42)
    # Background texture noise
    noise = rng.randint(-15, 16, (720, 1280, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw complex contours, hatching, and inscriptions
    for i in range(40):
        pt1 = (rng.randint(50, 1200), rng.randint(50, 670))
        pt2 = (pt1[0] + rng.randint(-150, 150), pt1[1] + rng.randint(-150, 150))
        color = (rng.randint(180, 255), rng.randint(180, 255), rng.randint(180, 255))
        if style == "dark":
            color = (rng.randint(70, 110), rng.randint(65, 100), rng.randint(60, 95))
        cv2.line(arr, pt1, pt2, color, thickness=rng.randint(1, 3))

    # Add text-like shapes / inscriptions
    for r in range(15):
        y = 100 + r * 35
        for c in range(25):
            x = 80 + c * 45
            cv2.putText(arr, chr(65 + (r * 25 + c) % 26), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220) if style != "dark" else (85, 80, 75), 1)

    Image.fromarray(arr).save(img_path, quality=95)
    return img_path


def create_blank_canvas(tmp_path: Path, name: str) -> Path:
    """Create a completely flat low-information image."""
    img_path = tmp_path / name
    arr = np.full((720, 1280, 3), 40, dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    return img_path


def create_smooth_gradient(tmp_path: Path, name: str) -> Path:
    """Create a featureless soft linear gradient."""
    img_path = tmp_path / name
    grad_1d = np.linspace(20, 60, 1280, dtype=np.uint8)
    arr = np.tile(grad_1d, (720, 1))
    arr_3ch = np.stack([arr, arr, arr], axis=-1)
    Image.fromarray(arr_3ch).save(img_path)
    return img_path


def reconstruct_20260913_fallback_fixture(tmp_path: Path) -> Path:
    """Reconstruct exact deterministic fallback visual from episode 20260913-001."""
    out_path = tmp_path / "fallback_20260913_reconstruction.png"
    provider = DeterministicFallbackProvider()
    provider.generate(
        prompt="Ancient Chola seaport of Poompuhar with docks and ships",
        negative_prompt="",
        width=1280,
        height=720,
        seed=12345,
        output_path=out_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED"
    )
    return out_path


# -----------------------------------------------------------------------------
# Test Cases (1 to 15)
# -----------------------------------------------------------------------------

def test_1_exact_regression_20260913_001_fallback_fails_gate(tmp_path):
    """
    1. Exact regression fixture for 20260913-001 fallback canvas:
    PROVE:
      OLD basic image validation = could pass (valid dimensions, non-corrupt, valid format)
      NEW semantic visual gate = FAIL with VISUAL_CONTENT_INVALID
    """
    fb_img = reconstruct_20260913_fallback_fixture(tmp_path)

    # Old basic technical validation passed
    valid_tech, tech_err = ImageValidator.validate_image_file(fb_img, expected_resolution=(1280, 720))
    assert valid_tech is True, f"Technical validation should pass: {tech_err}"

    # New semantic visual-presence gate MUST fail for mandatory historical B-roll
    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        fb_img,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="broll_motion",
        requires_historical_visual=True,
        source_type="fallback"
    )
    assert s_valid is False
    assert "VISUAL_CONTENT_INVALID" in s_reason
    assert s_diag["semantic_visual_presence"] == "FAIL"
    assert s_diag["visual_presence_valid"] is False
    assert s_diag["features"]["is_fallback_canvas"] is True


def test_2_generic_fallback_allowed_for_intentional_transition(tmp_path):
    """
    2. Generic fallback used as intentional transition/title/atmospheric scene:
    PASS (semantic_visual_presence = PASS)
    """
    fb_img = reconstruct_20260913_fallback_fixture(tmp_path)

    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        fb_img,
        grounding_type="ATMOSPHERIC",
        scene_type="transition",
        requires_historical_visual=False,
        source_type="fallback"
    )
    assert s_valid is True
    assert s_diag["semantic_visual_presence"] == "PASS"
    assert s_diag["rule_applied"] == "TRANSITION_ALLOWED_FALLBACK"


def test_3_detailed_historical_photograph_passes(tmp_path):
    """3. Detailed historical photograph passes."""
    photo_path = create_synthetic_historical_image(tmp_path, "hist_photo.jpg", style="photo")
    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        photo_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="source_photograph"
    )
    assert s_valid is True
    assert s_diag["semantic_visual_presence"] == "PASS"
    assert s_diag["visual_presence_valid"] is True
    assert s_diag["historical_source_authentic"] is True


def test_4_detailed_historical_painting_passes(tmp_path):
    """4. Detailed historical painting passes."""
    painting_path = create_synthetic_historical_image(tmp_path, "hist_painting.jpg", style="painting")
    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        painting_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="historical_artwork"
    )
    assert s_valid is True
    assert s_diag["semantic_visual_presence"] == "PASS"
    assert s_diag["visual_presence_valid"] is True
    assert s_diag["historical_source_authentic"] is True


def test_5_dark_detailed_historical_image_passes(tmp_path):
    """
    5. Dark detailed historical image (e.g. dimly lit cave, dark relief carving):
    Mean brightness < 35, but high Laplacian variance and edge density -> PASS.
    """
    dark_path = create_synthetic_historical_image(tmp_path, "dark_relief.jpg", style="dark")
    feats = SemanticVisualValidator.analyze_image_features(dark_path)
    assert feats["mean_brightness"] < 35.0, f"Expected dark image, got {feats['mean_brightness']}"

    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        dark_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="source_photograph"
    )
    assert s_valid is True
    assert s_diag["semantic_visual_presence"] == "PASS"
    assert s_diag["visual_presence_valid"] is True
    assert s_diag["rule_applied"] in ("DARK_DETAILED_HISTORICAL_ACCEPTED", "DETAILED_HISTORICAL_ACCEPTED")


def test_6_detailed_map_passes(tmp_path):
    """6. Detailed historical map passes."""
    map_path = create_synthetic_historical_image(tmp_path, "ancient_map.jpg", style="map")
    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        map_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="historical_artwork"
    )
    assert s_valid is True
    assert s_diag["semantic_visual_presence"] == "PASS"
    assert s_diag["visual_presence_valid"] is True


def test_7_detailed_manuscript_inscription_passes(tmp_path):
    """7. Detailed manuscript / stone inscription passes."""
    manuscript_path = create_synthetic_historical_image(tmp_path, "inscriptions.jpg", style="manuscript")
    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        manuscript_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="source_photograph"
    )
    assert s_valid is True
    assert s_diag["semantic_visual_presence"] == "PASS"
    assert s_diag["visual_presence_valid"] is True


def test_8_low_information_blank_fails(tmp_path):
    """8. Flat blank image fails with VISUAL_CONTENT_INVALID."""
    blank_path = create_blank_canvas(tmp_path, "flat_blank.png")
    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        blank_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="ai_generated_visualization"
    )
    assert s_valid is False
    assert "VISUAL_CONTENT_INVALID" in s_reason
    assert s_diag["semantic_visual_presence"] == "FAIL"
    assert s_diag["visual_presence_valid"] is False


def test_9_generic_gradient_fails(tmp_path):
    """9. Generic soft gradient fails with VISUAL_CONTENT_INVALID."""
    grad_path = create_smooth_gradient(tmp_path, "soft_gradient.png")
    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        grad_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="ai_generated_visualization"
    )
    assert s_valid is False
    assert "VISUAL_CONTENT_INVALID" in s_reason
    assert s_diag["semantic_visual_presence"] == "FAIL"


def test_10_yaazhini_presenter_passes_when_host_anchored():
    """10. Yaazhini presenter PASSES when scene is HOST_ANCHORED."""
    yaazhini_path = BASE_DIR / "assets" / "yaazhini_presenter.jpg"
    assert yaazhini_path.exists(), "Canonical Yaazhini asset must exist"

    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        yaazhini_path,
        grounding_type="HOST_ANCHORED",
        scene_type="host_vlog",
        requires_historical_visual=False,
        is_host_scene=True,
        source_type="host_asset"
    )
    assert s_valid is True
    assert s_diag["semantic_visual_presence"] == "PASS"
    assert s_diag["rule_applied"] == "HOST_PRESENTER_VALID"


def test_11_yaazhini_presenter_fails_when_mandatory_historical_broll():
    """
    11. Yaazhini presenter FAILS when mistakenly used as mandatory historical B-roll.
    Must return VISUAL_CONTENT_INVALID.
    """
    yaazhini_path = BASE_DIR / "assets" / "yaazhini_presenter.jpg"

    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        yaazhini_path,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="host_asset"
    )
    assert s_valid is False
    assert "VISUAL_CONTENT_INVALID" in s_reason
    assert "Host presenter asset cannot be substituted" in s_reason
    assert s_diag["semantic_visual_presence"] == "FAIL"


def test_12_ai_generated_historical_visualization_provenance(tmp_path):
    """
    12. AI-generated historical visualization:
    PASS visual presence (valid_presence = True)
    BUT:
    historical_source_authentic = False.
    """
    ai_img = create_synthetic_historical_image(tmp_path, "ai_viz.jpg", style="photo")
    s_valid, s_reason, s_diag = SemanticVisualValidator.validate_semantic_presence(
        ai_img,
        grounding_type="HISTORICAL_CLAIM_GROUNDED",
        scene_type="historical_broll",
        requires_historical_visual=True,
        source_type="ai_generated_visualization"
    )
    assert s_valid is True
    assert s_diag["semantic_visual_presence"] == "PASS"
    assert s_diag["visual_presence_valid"] is True
    assert s_diag["historical_source_authentic"] is False
    assert s_diag["source_type"] == "ai_generated_visualization"


def test_13_master_qc_catches_generic_fallback_in_final_broll(tmp_path):
    """
    13. Master QC catches generic fallback in final B-roll:
    - multi-point sampling (25%, 50%, 75%) triggers critical failure
    - critical_failure = True
    - scene_completeness = 0.0
    - decision = REVIEW_REQUIRED
    - weighted score cannot override critical failure
    """
    # Create test video containing fallback frames
    video_path = tmp_path / "test_fallback_master.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (1280, 720))

    # Scene 1: Host frames (0 to 74)
    host_arr = np.full((720, 1280, 3), 120, dtype=np.uint8)
    for _ in range(75):
        out.write(host_arr)

    # Scene 2: Fallback canvas frames (75 to 149)
    fb_path = reconstruct_20260913_fallback_fixture(tmp_path)
    fb_frame = cv2.imread(str(fb_path))
    for _ in range(75):
        out.write(fb_frame)

    out.release()

    scenes_info = [
        {
            "scene_id": 1,
            "scene_type": "host_vlog",
            "start_frame": 0,
            "end_frame": 74,
            "visual_asset_id": "asset_host_01",
            "grounding_type": "HOST_ANCHORED",
            "requires_historical_visual": False,
            "is_host_scene": True,
            "clip_path": "scenes/scene_01.mp4",
            "duration_seconds": 3.0
        },
        {
            "scene_id": 2,
            "scene_type": "historical_broll",
            "start_frame": 75,
            "end_frame": 149,
            "visual_asset_id": "asset_broll_02",
            "grounding_type": "HISTORICAL_CLAIM_GROUNDED",
            "requires_historical_visual": True,
            "is_host_scene": False,
            "source_type": "fallback",
            "clip_path": "scenes/scene_02.mp4",
            "duration_seconds": 3.0
        }
    ]

    qc = MasterQualityController()
    b_pass, b_issues, b_samples = qc.inspect_broll_semantic_presence(video_path, scenes_info)

    assert b_pass is False, "Fallback in mandatory B-roll must fail multi-point inspection"
    assert any("CRITICAL: Scene 2" in iss for iss in b_issues)

    # Run full QC and verify critical override
    mock_script = {"scenes": [{"id": 1, "type": "host_vlog"}, {"id": 2, "type": "historical_broll"}]}
    mock_audio = {"audio_segments": [{"scene_id": 1, "duration_seconds": 3.0}, {"scene_id": 2, "duration_seconds": 3.0}], "presenter_segments": [{"scene_id": 1}], "timing": []}
    mock_motion = {"scenes": []}

    report = qc.run_full_qc(
        episode_id="test_ep",
        master_video_path=video_path,
        script_data=mock_script,
        audio_manifest=mock_audio,
        motion_manifest=mock_motion,
        master_scenes_info=scenes_info,
        expected_duration=6.0
    )

    assert report.critical_failure is True
    assert report.component_scores["scene_completeness"] == 0.0
    assert report.decision == "REVIEW_REQUIRED"
    assert report.passed is False


def test_14_master_qc_accepts_detailed_static_historical_image(tmp_path):
    """14. Master QC multi-point sampling accepts authentic detailed historical image."""
    video_path = tmp_path / "test_authentic_master.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, 25.0, (1280, 720))

    hist_path = create_synthetic_historical_image(tmp_path, "authentic_broll.jpg", style="photo")
    hist_frame = cv2.imread(str(hist_path))

    for _ in range(100):
        out.write(hist_frame)
    out.release()

    scenes_info = [
        {
            "scene_id": 1,
            "scene_type": "historical_broll",
            "start_frame": 0,
            "end_frame": 99,
            "visual_asset_id": "asset_broll_01",
            "grounding_type": "HISTORICAL_CLAIM_GROUNDED",
            "requires_historical_visual": True,
            "is_host_scene": False,
            "source_type": "source_photograph",
            "clip_path": "scenes/scene_01.mp4",
            "duration_seconds": 4.0
        }
    ]

    qc = MasterQualityController()
    b_pass, b_issues, b_samples = qc.inspect_broll_semantic_presence(video_path, scenes_info)
    assert b_pass is True, f"Authentic historical visual should pass: {b_issues}"
    assert len(b_samples) == 3  # sampled at 25%, 50%, 75%
    for s in b_samples:
        assert s["valid"] is True


def test_15_protected_fixture_audit():
    """
    15. Protected fixture audit:
    TEST-P9-AUDIT unchanged
    20260910-002 unchanged
    20260913-001 unchanged
    Verifies 100% exact SHA-256 matches against baseline snapshot.
    """
    baseline_path = BASE_DIR / "scratch" / "protected_fixtures_baseline.json"
    assert baseline_path.exists(), "Baseline fixture hashes must exist"

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    for ep_id, files in baseline.items():
        ep_dir = BASE_DIR / "outputs" / "episodes" / ep_id
        assert ep_dir.exists(), f"Protected episode {ep_id} directory missing"

        for rel_path, meta in files.items():
            full_path = ep_dir / rel_path
            assert full_path.exists(), f"Protected file was deleted: {full_path}"
            expected_sha = meta["sha256"] if isinstance(meta, dict) else meta
            with open(full_path, "rb") as f:
                actual_sha = hashlib.sha256(f.read()).hexdigest()
            assert actual_sha == expected_sha, f"Protected artifact {ep_id}/{rel_path} was modified! Hash mismatch."
