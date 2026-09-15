"""
tests/test_phase8_motion_engine.py - Unit, Integration & Safety Tests for Phase 8 Safe 2.5D Motion Engine.
Kaalapadhivugal Production System (@kaalapadhivugal).
"""

import sys
import os
import json
import socket
import hashlib
import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.recovery_manager import RecoveryManager
from autonomous.orchestrator import AutonomousOrchestrator
from autonomous.visual_planner import VisualPlan, GroundingType
from autonomous.motion_engine import (
    Safe2_5DMotionEngine,
    discover_depth_model_path,
    BaseDepthProvider,
    LocalDepthAnythingV2Provider,
    ApproximateDepthFallbackProvider,
    DepthCache,
    SubjectProtectionAnalyzer,
    CameraMotionPlanner,
    BorderSafetyAnalyzer,
    EnvironmentMotionAnalyzer,
    EnvironmentMotionRenderer,
    BaseMotionGenerator,
    StaticMotionGenerator,
    ParallaxMotionGenerator,
    EnvironmentalMotionGenerator,
    MotionQualityScorer,
    BaseFrameInterpolationProvider,
    NoInterpolationProvider,
    OpticalFlowInterpolationProvider,
    OptionalImageToVideoGenerator,
    MotionManifestRecord,
    MotionManifest,
    MotionGenerationReport
)


class TestPhase8MotionEngine(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_phase8_")
        self.db_path = Path(self.test_dir) / "test_universe.db"
        self.db_url = f"sqlite:///{self.db_path}"
        self.state_manager = StateManager(db_url=self.db_url)
        self.recovery_manager = RecoveryManager(self.state_manager)
        self.orchestrator = AutonomousOrchestrator(self.state_manager, self.recovery_manager)

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def _create_sample_static_visual_bundle(
        self,
        episode_id: str = "TEST-EP-001",
        state: EpisodeState = EpisodeState.STATIC_VISUALS_READY,
        include_script_validation: bool = True,
        script_val_passed: bool = True,
        include_content_quality: bool = True,
        content_quality_passed: bool = True,
        include_visual_plan: bool = True,
        visual_plan_status: str = "PLAN_COMPLETE",
        include_manifest: bool = True,
        corrupt_asset: bool = False,
        scenes: list = None
    ) -> AutonomousEpisode:
        """Creates an episode bundle with Phase 5, 5.1, 6, and 7 artifacts and valid static images."""
        ep_dir = Path(self.test_dir) / "episodes" / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "script").mkdir(parents=True, exist_ok=True)
        (ep_dir / "research").mkdir(parents=True, exist_ok=True)
        (ep_dir / "visuals" / "generated" / "scene_01").mkdir(parents=True, exist_ok=True)
        (ep_dir / "visuals" / "generated" / "scene_02").mkdir(parents=True, exist_ok=True)

        session = self.state_manager._get_session()
        try:
            ep = AutonomousEpisode(
                episode_id=episode_id,
                topic="Ancient Keezhadi Urban Civilization",
                category="Ancient Tamil history",
                status=state.value,
                current_stage=state.value,
                output_directory=str(ep_dir)
            )
            session.add(ep)
            session.commit()
            session.refresh(ep)
            session.expunge(ep)
        finally:
            session.close()

        # Phase 5 script validation artifact
        if include_script_validation:
            sv_data = {"passed": script_val_passed, "episode_id": episode_id, "violations": []}
            with open(ep_dir / "script" / "script_validation.json", "w", encoding="utf-8") as f:
                json.dump(sv_data, f, indent=2)

        # Phase 5.1 content quality artifact
        if include_content_quality:
            cq_data = {"passed": content_quality_passed, "episode_id": episode_id, "quality_score": 0.95}
            with open(ep_dir / "research" / "content_quality_report.json", "w", encoding="utf-8") as f:
                json.dump(cq_data, f, indent=2)

        # Phase 6 visual plan artifact
        if scenes is None:
            scenes = [
                {
                    "scene_id": 1,
                    "scene_type": "host_intro",
                    "duration_seconds": 3.0,
                    "shots": [
                        {
                            "shot_id": 1,
                            "duration_seconds": 3.0,
                            "grounding_type": "HOST_ANCHORED",
                            "visual_prompt": "Tamil documentary presenter Yaazhini introducing Keezhadi artifacts",
                            "camera_movement": "slow_dolly_forward",
                            "host_character_id": "host_yaazhini",
                            "reference_asset": "assets/yaazhini_presenter.jpg"
                        }
                    ]
                },
                {
                    "scene_id": 2,
                    "scene_type": "historical_reconstruction",
                    "duration_seconds": 3.0,
                    "shots": [
                        {
                            "shot_id": 1,
                            "duration_seconds": 3.0,
                            "grounding_type": "VERIFIED_FACT",
                            "visual_prompt": "Ancient water channel brick construction site in Keezhadi excavation",
                            "camera_movement": "pan_left",
                            "host_character_id": None
                        }
                    ]
                }
            ]

        if include_visual_plan:
            plan_data = {
                "episode_id": episode_id,
                "status": visual_plan_status,
                "total_scenes": len(scenes),
                "scenes": scenes
            }
            with open(ep_dir / "visuals" / "visual_plan.json", "w", encoding="utf-8") as f:
                json.dump(plan_data, f, indent=2)

        # Generate actual static image files (1280x720)
        img1_path = ep_dir / "visuals" / "generated" / "scene_01" / "scene_01_visual.png"
        img2_path = ep_dir / "visuals" / "generated" / "scene_02" / "scene_02_visual.png"

        img1 = Image.new("RGB", (1280, 720), color=(140, 110, 80))
        d1 = ImageDraw.Draw(img1)
        d1.rectangle([200, 100, 1000, 600], fill=(200, 170, 140))
        img1.save(img1_path)

        img2 = Image.new("RGB", (1280, 720), color=(80, 120, 140))
        d2 = ImageDraw.Draw(img2)
        d2.rectangle([100, 300, 1180, 650], fill=(50, 90, 120))  # Water/lower region
        img2.save(img2_path)

        with open(img1_path, "rb") as f:
            sha1 = hashlib.sha256(f.read()).hexdigest()
        with open(img2_path, "rb") as f:
            sha2 = hashlib.sha256(f.read()).hexdigest()

        if corrupt_asset:
            with open(img2_path, "w") as f:
                f.write("CORRUPTED_FILE_DATA")

        if include_manifest:
            manifest_data = {
                "episode_id": episode_id,
                "total_assets": 2,
                "assets": [
                    {
                        "asset_id": f"asset_ep_{episode_id}_sc01",
                        "episode_id": episode_id,
                        "scene_id": 1,
                        "shot_id": 1,
                        "file_path": str(img1_path.relative_to(ep_dir)),
                        "provider": "existing_local_asset",
                        "sha256": sha1,
                        "width": 1280,
                        "height": 720,
                        "format": "PNG",
                        "grounding_type": "HOST_ANCHORED",
                        "host_character_id": "host_yaazhini"
                    },
                    {
                        "asset_id": f"asset_ep_{episode_id}_sc02",
                        "episode_id": episode_id,
                        "scene_id": 2,
                        "shot_id": 1,
                        "file_path": str(img2_path.relative_to(ep_dir)),
                        "provider": "fallback",
                        "sha256": sha2,
                        "width": 1280,
                        "height": 720,
                        "format": "PNG",
                        "grounding_type": "VERIFIED_FACT",
                        "host_character_id": None
                    }
                ]
            }
            with open(ep_dir / "visuals" / "assets_manifest.json", "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            report_data = {
                "episode_id": episode_id,
                "total_assets": 2,
                "provider": "test_provider",
                "final_status": "COMPLETED"
            }
            with open(ep_dir / "visuals" / "generation_report.json", "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2)

        return self.state_manager.get_episode(episode_id)

    # =========================================================================
    # 1. STRICT INPUT GATE TESTS
    # =========================================================================

    def test_input_gate_requires_static_visuals_ready(self):
        """Phase 8 rejects execution if episode is not in STATIC_VISUALS_READY."""
        ep = self._create_sample_static_visual_bundle(state=EpisodeState.SCRIPT_VALIDATED)
        engine = Safe2_5DMotionEngine(self.state_manager)
        res = engine.generate_episode_motion(ep.episode_id)
        self.assertFalse(res.success)
        self.assertEqual(res.next_state, EpisodeState.REVIEW_REQUIRED)

    def test_input_gate_missing_assets_manifest(self):
        """Phase 8 rejects execution if Phase 7 assets_manifest.json is missing."""
        ep = self._create_sample_static_visual_bundle(include_manifest=False)
        engine = Safe2_5DMotionEngine(self.state_manager)
        res = engine.generate_episode_motion(ep.episode_id)
        self.assertFalse(res.success)
        self.assertEqual(res.next_state, EpisodeState.REVIEW_REQUIRED)
        self.assertIn("assets_manifest.json", res.error)

    def test_input_gate_corrupted_or_missing_image_file(self):
        """Phase 8 rejects execution if a static image file is corrupted or truncated."""
        ep = self._create_sample_static_visual_bundle(corrupt_asset=True)
        engine = Safe2_5DMotionEngine(self.state_manager)
        res = engine.generate_episode_motion(ep.episode_id)
        self.assertFalse(res.success)
        self.assertEqual(res.next_state, EpisodeState.REVIEW_REQUIRED)

    def test_input_gate_checksum_mismatch(self):
        """Phase 8 rejects execution if actual image SHA-256 does not match manifest."""
        ep = self._create_sample_static_visual_bundle()
        # Alter manifest checksum
        m_path = Path(ep.output_directory) / "visuals" / "assets_manifest.json"
        with open(m_path, "r") as f:
            m = json.load(f)
        m["assets"][0]["sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
        with open(m_path, "w") as f:
            json.dump(m, f)

        engine = Safe2_5DMotionEngine(self.state_manager)
        res = engine.generate_episode_motion(ep.episode_id)
        self.assertFalse(res.success)
        self.assertEqual(res.next_state, EpisodeState.REVIEW_REQUIRED)
        self.assertIn("Checksum mismatch", res.error)

    def test_blocked_on_review_required_state(self):
        """Episodes in REVIEW_REQUIRED must remain blocked."""
        ep = self._create_sample_static_visual_bundle(state=EpisodeState.REVIEW_REQUIRED)
        engine = Safe2_5DMotionEngine(self.state_manager)
        res = engine.generate_episode_motion(ep.episode_id)
        self.assertFalse(res.success)
        self.assertEqual(res.next_state, EpisodeState.REVIEW_REQUIRED)

    def test_historical_episode_20260910_002_safety_immutable(self):
        """Historical episode 20260910-002 must remain in REVIEW_REQUIRED and never be mutated."""
        session = self.state_manager._get_session()
        try:
            hist_ep = AutonomousEpisode(
                episode_id="20260910-002",
                topic="Keezhadi Civilization",
                category="Sangam Civilization",
                status=EpisodeState.REVIEW_REQUIRED.value,
                current_stage="REVIEW_REQUIRED",
                output_directory=str(Path(self.test_dir) / "hist_ep")
            )
            session.add(hist_ep)
            session.commit()
        finally:
            session.close()

        engine = Safe2_5DMotionEngine(self.state_manager)
        res = engine.generate_episode_motion("20260910-002")
        self.assertFalse(res.success)
        self.assertEqual(res.next_state, EpisodeState.REVIEW_REQUIRED)

        # Verify state is completely unchanged in DB
        ep_check = self.state_manager.get_episode("20260910-002")
        self.assertEqual(ep_check.status, EpisodeState.REVIEW_REQUIRED.value)

    # =========================================================================
    # 2. DEPTH ESTIMATION & CACHING TESTS
    # =========================================================================

    def test_depth_cache_miss_then_hit(self):
        """Depth map is computed on miss and loaded from .npy cache on subsequent call."""
        cache_dir = Path(self.test_dir) / "depth_cache"
        cache = DepthCache(cache_dir)

        dummy_img = Image.new("RGB", (1280, 720), color=(100, 120, 140))
        dummy_sha = "test_sha_abcdef1234567890"

        self.assertFalse(cache.is_cached(dummy_sha))

        fallback_provider = ApproximateDepthFallbackProvider(device="cpu")
        depth_map, _ = fallback_provider.estimate_depth(dummy_img, (1280, 720))
        cache.save(dummy_sha, depth_map)

        self.assertTrue(cache.is_cached(dummy_sha))
        loaded_depth = cache.load(dummy_sha)
        self.assertIsNotNone(loaded_depth)
        self.assertEqual(loaded_depth.shape, (720, 1280))
        np.testing.assert_allclose(loaded_depth, depth_map, atol=1e-5)

    def test_approximate_depth_fallback_provider_low_confidence(self):
        """ApproximateDepthFallbackProvider operates reliably on CPU and sets low confidence."""
        provider = ApproximateDepthFallbackProvider(device="cpu")
        img = Image.new("RGB", (1280, 720), color=(120, 120, 120))
        depth, meta = provider.estimate_depth(img, (1280, 720))

        self.assertEqual(provider.device, "cpu")
        self.assertTrue(meta.get("is_approximate_fallback"))
        self.assertEqual(depth.shape, (720, 1280))

        img_np = np.array(img)
        conf = DepthCache.compute_depth_confidence_estimate(img_np, depth, is_fallback=True)
        self.assertEqual(conf, 0.25)

    def test_depth_confidence_multi_diagnostic(self):
        """Multi-diagnostic confidence estimates dynamic range, smoothness, and edge alignment."""
        img_np = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Synthetic ramp depth
        y, _ = np.indices((720, 1280), dtype=np.float32)
        good_depth = y / 720.0
        conf_good = DepthCache.compute_depth_confidence_estimate(img_np, good_depth, is_fallback=False)
        self.assertGreater(conf_good, 0.40)

        # Degenerate flat depth
        flat_depth = np.ones((720, 1280), dtype=np.float32) * 0.5
        conf_bad = DepthCache.compute_depth_confidence_estimate(img_np, flat_depth, is_fallback=False)
        self.assertLess(conf_bad, conf_good)

    # =========================================================================
    # 3. SUBJECT PROTECTION & CHARACTER CONSISTENCY TESTS
    # =========================================================================

    def test_yaazhini_host_scene_very_low_motion_and_facial_protection(self):
        """Yaazhini host scene forces VERY_LOW motion and 0.05 facial displacement weight."""
        img_np = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        depth_map = np.ones((720, 1280), dtype=np.float32) * 0.5

        weight_map, score = SubjectProtectionAnalyzer.generate_subject_protection_mask(
            img_np, depth_map, shot_type="PORTRAIT", host_character_id="host_yaazhini"
        )
        self.assertAlmostEqual(float(weight_map.min()), 0.05, delta=0.03)

        level, strength = CameraMotionPlanner.get_base_motion_strength("PORTRAIT", host_character_id="host_yaazhini")
        self.assertEqual(level, "VERY_LOW")
        self.assertLessEqual(strength, 0.12)

    def test_subject_protection_mask_weight_clamping(self):
        """Protection mask is feathered and clamped between [0.05, 1.0]."""
        img_np = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        depth_map = np.random.rand(720, 1280).astype(np.float32)

        weight_map, _ = SubjectProtectionAnalyzer.generate_subject_protection_mask(
            img_np, depth_map, shot_type="WIDE_ESTABLISHING"
        )
        self.assertGreaterEqual(float(weight_map.min()), 0.05)
        self.assertLessEqual(float(weight_map.max()), 1.0)

    # =========================================================================
    # 4. CAMERA KINEMATICS & BORDER SAFETY TESTS
    # =========================================================================

    def test_camera_motion_planner_safety_levels(self):
        """Camera motion planner maps shot types to conservative defaults."""
        level_port, str_port = CameraMotionPlanner.get_base_motion_strength("PORTRAIT")
        level_arch, str_arch = CameraMotionPlanner.get_base_motion_strength("ARCHITECTURE")
        level_wide, str_wide = CameraMotionPlanner.get_base_motion_strength("WIDE_ESTABLISHING")

        self.assertEqual(level_port, "VERY_LOW")
        self.assertEqual(level_arch, "LOW")
        self.assertEqual(level_wide, "MEDIUM")
        self.assertLess(str_port, str_arch)
        self.assertLess(str_arch, str_wide)

    def test_border_safety_analyzer_trajectory_fallback(self):
        """BorderSafetyAnalyzer predicts boundary exposure and falls back down safety chain."""
        # Unsafe large motion should trigger fallback to safer trajectory
        is_safe, approved_cam, score = BorderSafetyAnalyzer.simulate_boundary_safety(
            (1280, 720), pre_scale=1.06, camera_type="orbit", effective_motion=0.95, total_frames=75
        )
        self.assertTrue(is_safe)
        self.assertNotEqual(approved_cam, "orbit")

    # =========================================================================
    # 5. ENVIRONMENTAL MOTION TESTS
    # =========================================================================

    def test_water_confidence_threshold_gating(self):
        """Water shader is completely disabled when confidence is below 0.55."""
        img_dry = np.ones((720, 1280, 3), dtype=np.uint8) * 50  # Dark brown dry ground
        depth_dry = np.ones((720, 1280), dtype=np.float32) * 0.5
        conf_dry, _ = EnvironmentMotionAnalyzer.compute_water_confidence(img_dry, depth_dry, ["desert", "sand"])
        self.assertLess(conf_dry, 0.55)

        # Applying water shader with confidence < 0.55 returns frame untouched
        out_frame = EnvironmentMotionRenderer.apply_water_shader(img_dry, np.zeros((720, 1280)), conf_dry, 0.5)
        np.testing.assert_array_equal(out_frame, img_dry)

    def test_atmospheric_particles_justified_only(self):
        """Atmospheric particles are enabled for dusty/ruins scenes and disabled for portraits."""
        self.assertTrue(EnvironmentMotionAnalyzer.should_apply_atmospheric_particles(
            "HISTORICAL", "excavation ruins in dust", ["ruins", "archaeology"]
        ))
        self.assertFalse(EnvironmentMotionAnalyzer.should_apply_atmospheric_particles(
            "PORTRAIT", "Host speaking clearly to camera", ["interview"]
        ))

    # =========================================================================
    # 6. QUALITY SCORER & RETRY CONTROLLER TESTS
    # =========================================================================

    def test_motion_quality_scorer_clean_frames(self):
        """Clean sequence with bounded movement achieves safe score >= 0.70."""
        scorer = MotionQualityScorer()
        frames = [np.ones((720, 1280, 3), dtype=np.uint8) * 100 for _ in range(25)]
        eval_res = scorer.evaluate_video(frames)
        self.assertTrue(eval_res["is_safe"])
        self.assertGreaterEqual(eval_res["quality_score"], 0.70)
        self.assertEqual(eval_res["boundary_violations"], 0)

    def test_motion_quality_scorer_detects_border_violations(self):
        """Scorer penalizes pure black border exposure along edges."""
        scorer = MotionQualityScorer()
        frames = []
        for _ in range(10):
            f = np.ones((720, 1280, 3), dtype=np.uint8) * 100
            f[:5, :, :] = 0  # Black top border
            frames.append(f)

        eval_res = scorer.evaluate_video(frames)
        self.assertGreater(eval_res["boundary_violations"], 0)
        self.assertLess(eval_res["quality_score"], 0.80)

    # =========================================================================
    # 7. PROVIDERS & INTERFACES TESTS
    # =========================================================================

    def test_no_interpolation_provider_default(self):
        """NoInterpolationProvider leaves frames and FPS untouched."""
        provider = NoInterpolationProvider()
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(15)]
        out = provider.interpolate(frames, target_fps=25)
        self.assertEqual(len(out), 15)

    def test_optional_i2v_interface_stub_disabled_default(self):
        """OptionalImageToVideoGenerator verifies disabled status and throws NotImplementedError."""
        stub = OptionalImageToVideoGenerator(enabled=False)
        self.assertFalse(stub.enabled)
        with self.assertRaises(NotImplementedError):
            stub.generate()

    # =========================================================================
    # 8. INTEGRATION, RECOVERY & STRICT END STATE TESTS
    # =========================================================================

    def test_full_phase8_generation_and_manifest_schema(self):
        """Full execution creates valid MP4s, typed manifest (33 fields), and report."""
        ep = self._create_sample_static_visual_bundle(episode_id="EP-FULL-01")
        engine = Safe2_5DMotionEngine(
            self.state_manager,
            depth_provider=ApproximateDepthFallbackProvider(device="cpu")
        )

        res = engine.generate_episode_motion(ep.episode_id)
        self.assertTrue(res.success)
        self.assertEqual(res.next_state, EpisodeState.MOTION_READY)

        ep_dir = Path(ep.output_directory)
        manifest_path = ep_dir / "motion" / "motion_manifest.json"
        report_path = ep_dir / "motion" / "motion_generation_report.json"

        self.assertTrue(manifest_path.exists())
        self.assertTrue(report_path.exists())

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["total_scenes"], 2)
        rec = manifest["scenes"][0]
        # Verify required typed fields
        required_fields = [
            "episode_id", "scene_id", "shot_id", "source_asset_id", "source_path",
            "source_checksum", "depth_provider", "depth_model", "depth_model_path",
            "depth_device", "depth_cache_hit", "depth_confidence_estimate",
            "subject_protection_score", "border_safety_score", "motion_type",
            "shot_type", "motion_strength_requested", "motion_strength_effective",
            "environment_motion_type", "environment_confidence", "interpolation_provider",
            "input_fps", "output_fps", "resolution", "frame_count", "duration_seconds",
            "quality_score", "retry_count", "fallback_used", "provider", "device",
            "processing_times", "output_video_path"
        ]
        for field_name in required_fields:
            self.assertIn(field_name, rec, f"Missing manifest field: {field_name}")

        # Verify output video file exists and is valid
        mp4_path = ep_dir / rec["output_video_path"]
        self.assertTrue(mp4_path.exists())
        self.assertGreater(mp4_path.stat().st_size, 1024)

    def test_recovery_skips_valid_existing_motion_scenes(self):
        """Recovery recognizes complete, valid scene MP4s and skips re-encoding."""
        ep = self._create_sample_static_visual_bundle(episode_id="EP-REC-01")
        engine = Safe2_5DMotionEngine(
            self.state_manager,
            depth_provider=ApproximateDepthFallbackProvider(device="cpu")
        )

        res1 = engine.generate_episode_motion(ep.episode_id)
        self.assertTrue(res1.success)

        # Reset episode state back to STATIC_VISUALS_READY
        session = self.state_manager._get_session()
        try:
            ep_row = session.query(AutonomousEpisode).filter_by(episode_id=ep.episode_id).first()
            if ep_row:
                ep_row.status = EpisodeState.STATIC_VISUALS_READY.value
                ep_row.current_stage = EpisodeState.STATIC_VISUALS_READY.value
                session.commit()
        finally:
            session.close()

        # Re-run: recovery should succeed immediately
        res2 = engine.generate_episode_motion(ep.episode_id)
        self.assertTrue(res2.success)
        self.assertEqual(res2.next_state, EpisodeState.MOTION_READY)

    def test_orchestrator_generate_motion_contract(self):
        """AutonomousOrchestrator.generate_motion returns StageExecutionResult with MOTION_READY."""
        ep = self._create_sample_static_visual_bundle(episode_id="EP-ORCH-01")
        res = self.orchestrator.generate_motion(
            ep.episode_id,
            depth_provider=ApproximateDepthFallbackProvider(device="cpu")
        )
        self.assertTrue(res.success)
        self.assertEqual(res.next_state, EpisodeState.MOTION_READY)

    def test_offline_mode_zero_network_calls(self):
        """Verifies motion generation operates 100% offline without socket calls."""
        ep = self._create_sample_static_visual_bundle(episode_id="EP-OFFLINE-01")
        engine = Safe2_5DMotionEngine(
            self.state_manager,
            depth_provider=ApproximateDepthFallbackProvider(device="cpu")
        )

        orig_socket = socket.socket

        def blocked_socket(*args, **kwargs):
            raise RuntimeError("Network activity blocked in offline mode!")

        socket.socket = blocked_socket
        try:
            res = engine.generate_episode_motion(ep.episode_id)
            self.assertTrue(res.success)
            self.assertEqual(res.next_state, EpisodeState.MOTION_READY)
        finally:
            socket.socket = orig_socket


if __name__ == "__main__":
    unittest.main()
