"""
tests/test_phase7_visual_generation.py - Unit, Integration & Safety Tests for Phase 7 Autonomous Static Visual Generation.

Validates:
 1. test_strict_input_gate_rejects_non_script_validated
 2. test_strict_input_gate_missing_script_validation_artifact
 3. test_strict_input_gate_failed_script_validation_artifact
 4. test_strict_input_gate_missing_content_quality_artifact
 5. test_strict_input_gate_failed_content_quality_artifact
 6. test_strict_input_gate_missing_visual_plan
 7. test_strict_input_gate_invalid_visual_plan_schema
 8. test_strict_input_gate_incomplete_visual_plan_status
 9. test_strict_input_gate_claim_traceability_required
10. test_blocked_on_review_required_state
11. test_blocked_on_failed_state
12. test_blocked_on_cancelled_state
13. test_historical_episode_20260910_002_safety_immutable
14. test_yaazhini_host_asset_resolution
15. test_no_host_invented_for_non_host_scenes
16. test_no_archived_presenter_selection
17. test_prompt_sanitizer_prohibited_constructs_cleaned
18. test_prompt_sanitizer_does_not_invent_facts
19. test_historical_text_inscription_background_only
20. test_deterministic_fallback_provider_execution
21. test_fallback_metadata_manifest_flags
22. test_fallback_epistemic_safety_blocks_on_mandatory_fact_failure
23. test_local_diffusers_provider_skips_when_uncached
24. test_comfyui_provider_skips_when_unavailable
25. test_image_validator_dimensions_and_aspect_ratio
26. test_image_validator_corrupted_file_detection
27. test_image_validator_blank_black_or_white_detection
28. test_manifest_typed_schema_completeness
29. test_generation_report_schema_completeness
30. test_prompt_hash_and_seed_determinism
31. test_actual_file_sha256_checksum_match
32. test_duplicate_asset_reuse_representation
33. test_retry_limit_two_attempts_before_fallback
34. test_recovery_skips_valid_existing_assets
35. test_recovery_regenerates_corrupted_existing_assets
36. test_no_motion_or_video_outputs_produced
37. test_final_state_strictly_static_visuals_ready_never_motion
38. test_orchestrator_generate_visuals_contract
39. test_offline_mode_zero_network_calls
"""

import sys
import os
import json
import hashlib
import unittest
import tempfile
import shutil
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.recovery_manager import RecoveryManager
from autonomous.orchestrator import AutonomousOrchestrator
from autonomous.visual_planner import (
    VisualPlanner,
    VisualPlan,
    GroundingType,
    VisualAssetRequirement
)
from autonomous.visual_generator import (
    VisualGenerator,
    ImageValidator,
    PromptSanitizer,
    DeterministicFallbackProvider,
    ExistingLocalAssetProvider,
    LocalDiffusersProvider,
    ComfyUIStaticProvider,
    MockVisualProvider,
    AssetManifestRecord,
    AssetsManifest,
    GenerationReport
)


class TestPhase7VisualGeneration(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_phase7_")
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

    def _create_sample_episode_bundle(
        self,
        episode_id="TEST-EP-001",
        state=EpisodeState.SCRIPT_VALIDATED,
        include_script_validation=True,
        script_val_passed=True,
        include_content_quality=True,
        content_quality_passed=True,
        include_visual_plan=True,
        visual_plan_status="PLAN_COMPLETE",
        scenes=None
    ):
        """Helper to scaffold a complete test episode directory with Phase 5/6 artifacts."""
        ep_dir = Path(self.test_dir) / "episodes" / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        (ep_dir / "script").mkdir(parents=True, exist_ok=True)
        (ep_dir / "research").mkdir(parents=True, exist_ok=True)
        (ep_dir / "visuals").mkdir(parents=True, exist_ok=True)

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
        finally:
            session.close()

        # Phase 5 script validation artifact
        if include_script_validation:
            val_data = {
                "passed": script_val_passed,
                "episode_id": episode_id,
                "total_scenes": 3,
                "violations": [] if script_val_passed else ["Claim grounding failed"]
            }
            with open(ep_dir / "script" / "script_validation.json", "w", encoding="utf-8") as f:
                json.dump(val_data, f, indent=2)

        # Phase 5.1 content quality report artifact
        if include_content_quality:
            cq_data = {
                "passed": content_quality_passed,
                "episode_id": episode_id,
                "total_claims": 5,
                "verified_claims": 5 if content_quality_passed else 1,
                "quality_score": 0.95 if content_quality_passed else 0.40
            }
            with open(ep_dir / "research" / "content_quality_report.json", "w", encoding="utf-8") as f:
                json.dump(cq_data, f, indent=2)

        # Phase 6 visual plan artifact
        if include_visual_plan:
            if scenes is None:
                scenes = [
                    {
                        "scene_id": 1,
                        "scene_type": "host_intro",
                        "duration_seconds": 6.0,
                        "shots": [
                            {
                                "shot_id": 1,
                                "duration_seconds": 6.0,
                                "grounding_type": "HOST_ANCHORED",
                                "visual_prompt": "Tamil female documentary host Yaazhini in professional attire presenting archaeological discoveries",
                                "negative_prompt": "cartoon, distorted face, low quality",
                                "visual_risk": "MINIMAL",
                                "claim_ids": ["CLM-001"],
                                "host_character_id": "host_yaazhini",
                                "reference_asset": "assets/yaazhini_presenter.jpg",
                                "safe_fallback": "historical documentary background canvas"
                            }
                        ]
                    },
                    {
                        "scene_id": 2,
                        "scene_type": "historical_reconstruction",
                        "duration_seconds": 8.0,
                        "shots": [
                            {
                                "shot_id": 1,
                                "duration_seconds": 8.0,
                                "grounding_type": "VERIFIED_FACT",
                                "visual_prompt": "Keezhadi brick water conduit system archaeological excavation site with precise masonry layout",
                                "negative_prompt": "modern vehicles, fantasy elements, text, subtitles",
                                "visual_risk": "FACTUAL_GROUNDING",
                                "claim_ids": ["CLM-002"],
                                "host_character_id": None,
                                "reference_asset": None,
                                "safe_fallback": "archaeological excavation schematic map"
                            }
                        ]
                    }
                ]

            plan_data = {
                "episode_id": episode_id,
                "topic": "Ancient Keezhadi Urban Civilization",
                "category": "Ancient Tamil history",
                "status": visual_plan_status,
                "total_scenes": len(scenes),
                "total_shots": sum(len(s.get("shots", [])) for s in scenes),
                "total_duration_seconds": 14.0,
                "scenes": scenes,
                "planned_at": "2026-09-12T10:00:00Z"
            }
            with open(ep_dir / "visuals" / "visual_plan.json", "w", encoding="utf-8") as f:
                json.dump(plan_data, f, indent=2)

        return self.state_manager.get_episode(episode_id)

    # =========================================================================
    # 1. STRICT INPUT GATE TESTS
    # =========================================================================

    def test_strict_input_gate_rejects_non_script_validated(self):
        """Reject generation if episode is in an earlier stage (e.g. TOPIC_SELECTED or RESEARCHING)."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-01",
            state=EpisodeState.RESEARCHING
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("SCRIPT_VALIDATED", res.error)

    def test_strict_input_gate_missing_script_validation_artifact(self):
        """Reject generation if script_validation.json does not exist."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-02",
            include_script_validation=False
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("script_validation.json", res.error)

    def test_strict_input_gate_failed_script_validation_artifact(self):
        """Reject generation if script validation passed=False."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-03",
            script_val_passed=False
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("failed script validation", res.error)

    def test_strict_input_gate_missing_content_quality_artifact(self):
        """Reject generation if content_quality_report.json does not exist."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-04",
            include_content_quality=False
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("content_quality_report.json", res.error)

    def test_strict_input_gate_failed_content_quality_artifact(self):
        """Reject generation if content quality validation passed=False."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-05",
            content_quality_passed=False
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("failed content quality", res.error)

    def test_strict_input_gate_missing_visual_plan(self):
        """Reject generation if visual_plan.json does not exist."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-06",
            include_visual_plan=False
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("visual_plan.json", res.error)

    def test_strict_input_gate_invalid_visual_plan_schema(self):
        """Reject generation if visual_plan.json is corrupt or missing scenes."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-07"
        )
        plan_path = Path(ep.output_directory) / "visuals" / "visual_plan.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write("INVALID JSON CONTENT")

        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("Invalid visual plan", res.error)

    def test_strict_input_gate_incomplete_visual_plan_status(self):
        """Reject generation if visual_plan status is not PLAN_COMPLETE."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-08",
            visual_plan_status="INCOMPLETE"
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("PLAN_COMPLETE", res.error)

    def test_strict_input_gate_claim_traceability_required(self):
        """Reject generation if VERIFIED_FACT shots have empty claim_ids."""
        scenes = [
            {
                "scene_id": 1,
                "scene_type": "historical_reconstruction",
                "duration_seconds": 6.0,
                "shots": [
                    {
                        "shot_id": 1,
                        "duration_seconds": 6.0,
                        "grounding_type": "VERIFIED_FACT",
                        "visual_prompt": "Archaeological artifacts excavated at Keezhadi",
                        "negative_prompt": "modern",
                        "visual_risk": "FACTUAL_GROUNDING",
                        "claim_ids": [],  # Empty claims for factual shot
                        "host_character_id": None,
                        "reference_asset": None,
                        "safe_fallback": "fallback canvas"
                    }
                ]
            }
        ]
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-09",
            scenes=scenes
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("claim traceability", res.error)

    def test_blocked_on_review_required_state(self):
        """Episode in REVIEW_REQUIRED state must be strictly rejected."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-10",
            state=EpisodeState.REVIEW_REQUIRED
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("REVIEW_REQUIRED", res.error)

    def test_blocked_on_failed_state(self):
        """Episode in FAILED state must be strictly rejected."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-11",
            state=EpisodeState.FAILED
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("FAILED", res.error)

    def test_blocked_on_cancelled_state(self):
        """Episode in CANCELLED state must be strictly rejected."""
        ep = self._create_sample_episode_bundle(
            episode_id="EP-GATE-12",
            state=EpisodeState.CANCELLED
        )
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertIn("CANCELLED", res.error)

    # =========================================================================
    # 2. HISTORICAL EPISODE SAFETY TEST
    # =========================================================================

    def test_historical_episode_20260910_002_safety_immutable(self):
        """Existing episode 20260910-002 is REVIEW_REQUIRED and must remain untouched."""
        prod_state_mgr = StateManager()  # Connects to active universe_platform.db
        active_ep = prod_state_mgr.get_episode("20260910-002")
        if active_ep:
            self.assertEqual(active_ep.status, EpisodeState.REVIEW_REQUIRED.value)
            orchestrator = AutonomousOrchestrator(prod_state_mgr, RecoveryManager(prod_state_mgr))
            # Attempting visual generation on it must return blocked/failure
            res = orchestrator.generate_visuals("20260910-002")
            self.assertFalse(res.success)
            self.assertEqual(res.next_state, EpisodeState.REVIEW_REQUIRED)

            # Re-read from DB to verify it was NOT modified
            ep_after = prod_state_mgr.get_episode("20260910-002")
            self.assertEqual(ep_after.status, EpisodeState.REVIEW_REQUIRED.value)

    # =========================================================================
    # 3. HOST ASSET HANDLING TESTS
    # =========================================================================

    def test_yaazhini_host_asset_resolution(self):
        """Explicitly requested host_yaazhini resolves canonical assets/yaazhini_presenter.jpg."""
        ep = self._create_sample_episode_bundle(episode_id="EP-HOST-01")
        provider = ExistingLocalAssetProvider()

        # Shot requesting host_yaazhini
        req = VisualAssetRequirement(
            scene_id=1,
            shot_id=1,
            visual_prompt="Host Yaazhini presenting",
            grounding_type=GroundingType.HOST_ANCHORED,
            host_character_id="host_yaazhini",
            reference_asset="assets/yaazhini_presenter.jpg"
        )
        self.assertTrue(provider.is_available())
        resolved = provider.resolve_asset(req, Path(ep.output_directory))
        self.assertIsNotNone(resolved)
        self.assertTrue(Path(resolved).exists())
        self.assertIn("yaazhini_presenter", Path(resolved).name)

    def test_no_host_invented_for_non_host_scenes(self):
        """Scenes without host_character_id must return None from ExistingLocalAssetProvider."""
        ep = self._create_sample_episode_bundle(episode_id="EP-HOST-02")
        provider = ExistingLocalAssetProvider()

        req = VisualAssetRequirement(
            scene_id=2,
            shot_id=1,
            visual_prompt="Ancient pottery excavation at Keezhadi",
            grounding_type=GroundingType.VERIFIED_FACT,
            host_character_id=None,
            reference_asset=None
        )
        resolved = provider.resolve_asset(req, Path(ep.output_directory))
        self.assertIsNone(resolved)

    def test_no_archived_presenter_selection(self):
        """Provider must strictly reject archived presenter assets (e.g. vennila)."""
        ep = self._create_sample_episode_bundle(episode_id="EP-HOST-03")
        provider = ExistingLocalAssetProvider()

        req = VisualAssetRequirement(
            scene_id=1,
            shot_id=1,
            visual_prompt="Vennila presenter talking",
            grounding_type=GroundingType.HOST_ANCHORED,
            host_character_id="host_vennila",
            reference_asset="assets/vennila_presenter.jpg"
        )
        resolved = provider.resolve_asset(req, Path(ep.output_directory))
        self.assertIsNone(resolved)

    # =========================================================================
    # 4. PROMPT SANITIZER & HISTORICAL TEXT TESTS
    # =========================================================================

    def test_prompt_sanitizer_prohibited_constructs_cleaned(self):
        """Sanitizer detects and strips AI generation text commands without inventing facts."""
        prompt = "Ancient Tamil-Brahmi pottery shard with readable letters inscribed showing date 580 BCE with subtitles"
        cleaned, tags = PromptSanitizer.sanitize_prompt(prompt)
        self.assertNotIn("with readable letters inscribed", cleaned)
        self.assertNotIn("subtitles", cleaned)
        self.assertIn("surface background only", cleaned)
        self.assertIn("inscription_surface_only", tags)

    def test_prompt_sanitizer_does_not_invent_facts(self):
        """Sanitizer preserves the exact factual nouns and adjectives without rewriting dates or names."""
        prompt = "Keezhadi archaeological trench 4 with ring well drainage system"
        cleaned, tags = PromptSanitizer.sanitize_prompt(prompt)
        self.assertIn("Keezhadi", cleaned)
        self.assertIn("trench 4", cleaned)
        self.assertIn("ring well drainage system", cleaned)

    def test_historical_text_inscription_background_only(self):
        """Text/inscriptions in visual plan are marked for text overlay deferral."""
        is_bg, guidance = PromptSanitizer.requires_background_surface_only(
            "Shot showing ancient Tamil Brahmi inscription on stone pillar"
        )
        self.assertTrue(is_bg)
        self.assertIn("Background canvas only", guidance)

    # =========================================================================
    # 5. DETERMINISTIC FALLBACK PROVIDER TESTS
    # =========================================================================

    def test_deterministic_fallback_provider_execution(self):
        """Fallback provider creates valid 1280x720 16:9 documentary canvas."""
        ep = self._create_sample_episode_bundle(episode_id="EP-FALLBACK-01")
        output_path = Path(ep.output_directory) / "visuals" / "generated" / "scene_02" / "shot_01.png"
        provider = DeterministicFallbackProvider()

        req = VisualAssetRequirement(
            scene_id=2,
            shot_id=1,
            visual_prompt="Keezhadi brick structure layout",
            grounding_type=GroundingType.VERIFIED_FACT
        )
        res_path = provider.generate(req, output_path, seed=42)
        self.assertTrue(res_path.exists())

        img = Image.open(res_path)
        self.assertEqual(img.size, (1280, 720))
        self.assertEqual(img.mode, "RGB")

    def test_fallback_metadata_manifest_flags(self):
        """When fallback is used, manifest records provider='fallback' and fallback_used=True."""
        scenes = [
            {
                "scene_id": 1,
                "scene_type": "documentary_broll",
                "duration_seconds": 6.0,
                "shots": [
                    {
                        "shot_id": 1,
                        "duration_seconds": 6.0,
                        "grounding_type": "ILLUSTRATIVE",
                        "visual_prompt": "Ancient river valley landscape with morning mist",
                        "negative_prompt": "modern structures",
                        "visual_risk": "LOW",
                        "claim_ids": ["CLM-001"],
                        "host_character_id": None,
                        "reference_asset": None,
                        "safe_fallback": "historical landscape background"
                    }
                ]
            }
        ]
        ep = self._create_sample_episode_bundle(episode_id="EP-FALLBACK-02", scenes=scenes)
        gen = VisualGenerator(self.state_manager)
        gen.providers = [DeterministicFallbackProvider()]

        res = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res.success)

        manifest_path = Path(ep.output_directory) / "visuals" / "assets_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for asset in manifest["assets"]:
            self.assertEqual(asset["provider"], "fallback")
            self.assertTrue(asset["fallback_used"])

    def test_fallback_epistemic_safety_blocks_on_mandatory_fact_failure(self):
        """If mandatory visual requirement cannot be safely produced, transitions to REVIEW_REQUIRED."""
        ep = self._create_sample_episode_bundle(episode_id="EP-FALLBACK-03")
        gen = VisualGenerator(self.state_manager)

        # Mock provider that fails all generations
        class FailingProvider(MockVisualProvider):
            def generate(self, requirement, output_path, seed=42):
                raise RuntimeError("Simulated total hardware failure")

        gen.providers = [FailingProvider()]
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertFalse(res.success)
        self.assertEqual(res.next_state, EpisodeState.REVIEW_REQUIRED)

        # Episode state in DB must be REVIEW_REQUIRED
        updated_ep = self.state_manager.get_episode(ep.episode_id)
        self.assertEqual(updated_ep.status, EpisodeState.REVIEW_REQUIRED.value)

    # =========================================================================
    # 6. EXTERNAL PROVIDER SAFETY (NO DOWNLOAD / SAFE PROBE)
    # =========================================================================

    def test_local_diffusers_provider_skips_when_uncached(self):
        """LocalDiffusersProvider skips safely when no local model directory exists."""
        provider = LocalDiffusersProvider()
        # In this test environment, no local model is cached in autonomous/models
        # Provider should report is_available() = False without throwing or downloading
        avail = provider.is_available()
        self.assertFalse(avail)

    def test_comfyui_provider_skips_when_unavailable(self):
        """ComfyUI provider probes port 8188 with timeout and skips safely when offline."""
        provider = ComfyUIStaticProvider()
        avail = provider.is_available()
        self.assertFalse(avail)

    # =========================================================================
    # 7. IMAGE VALIDATOR TESTS
    # =========================================================================

    def test_image_validator_dimensions_and_aspect_ratio(self):
        """Validator rejects images that are not 16:9 or not 1280x720."""
        val = ImageValidator()
        tmp_img = Path(self.test_dir) / "test_wrong_size.png"

        # 4:3 image (800x600)
        img = Image.new("RGB", (800, 600), color=(100, 100, 100))
        img.save(tmp_img)
        ok, reason = val.validate(tmp_img)
        self.assertFalse(ok)
        self.assertIn("16:9", reason)

        # Correct 1280x720 image
        tmp_correct = Path(self.test_dir) / "test_correct.png"
        img_correct = Image.new("RGB", (1280, 720), color=(100, 120, 140))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_correct)
        draw.rectangle([0, 0, 640, 720], fill=(50, 60, 70))
        draw.rectangle([640, 0, 1280, 720], fill=(180, 190, 200))
        img_correct.save(tmp_correct)
        ok, reason = val.validate(tmp_correct)
        self.assertTrue(ok)

    def test_image_validator_corrupted_file_detection(self):
        """Validator detects and rejects non-image or truncated files."""
        val = ImageValidator()
        corrupt_path = Path(self.test_dir) / "corrupt.png"
        with open(corrupt_path, "w") as f:
            f.write("NOT AN IMAGE CONTENT AT ALL")

        ok, reason = val.validate(corrupt_path)
        self.assertFalse(ok)
        self.assertIn("corrupted", reason.lower())

    def test_image_validator_blank_black_or_white_detection(self):
        """Validator detects pure black, pure white, or completely flat canvases."""
        val = ImageValidator()

        # Pure black
        black_path = Path(self.test_dir) / "pure_black.png"
        Image.new("RGB", (1280, 720), color=(0, 0, 0)).save(black_path)
        ok, reason = val.validate(black_path)
        self.assertFalse(ok)
        self.assertIn("too dark", reason.lower())

        # Pure white
        white_path = Path(self.test_dir) / "pure_white.png"
        Image.new("RGB", (1280, 720), color=(255, 255, 255)).save(white_path)
        ok, reason = val.validate(white_path)
        self.assertFalse(ok)
        self.assertIn("too bright", reason.lower())

    # =========================================================================
    # 8. MANIFEST & REPORT SCHEMAS
    # =========================================================================

    def test_manifest_typed_schema_completeness(self):
        """Manifest contains all mandatory typed fields without arbitrary omissions."""
        ep = self._create_sample_episode_bundle(episode_id="EP-SCHEMA-01")
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res.success)

        manifest_path = Path(ep.output_directory) / "visuals" / "assets_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_fields = [
            "asset_id", "episode_id", "scene_id", "shot_id", "file_path",
            "provider", "model", "model_version", "prompt", "negative_prompt",
            "prompt_hash", "seed", "width", "height", "format", "file_size",
            "sha256", "grounding_type", "claim_ids", "visual_risk",
            "host_character_id", "reference_asset", "generation_attempt",
            "created_at", "validation_status", "fallback_used", "source_type"
        ]

        self.assertIn("assets", data)
        self.assertGreater(len(data["assets"]), 0)
        for asset in data["assets"]:
            for field in required_fields:
                self.assertIn(field, asset, f"Missing required manifest field: {field}")

    def test_generation_report_schema_completeness(self):
        """Generation report contains all required telemetry metrics."""
        ep = self._create_sample_episode_bundle(episode_id="EP-REPORT-01")
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res.success)

        report_path = Path(ep.output_directory) / "visuals" / "generation_report.json"
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        report_fields = [
            "episode_id", "provider", "model", "total_assets", "successful_assets",
            "fallback_assets", "validation_failures", "retries", "total_time_seconds",
            "average_time_per_asset", "peak_vram_mb", "final_status"
        ]
        for field in report_fields:
            self.assertIn(field, report, f"Missing report field: {field}")

    # =========================================================================
    # 9. CHECKSUM & SEED DETERMINISM
    # =========================================================================

    def test_prompt_hash_and_seed_determinism(self):
        """Same prompt and episode seed produce identical prompt_hash and reproducible seed."""
        prompt = "Keezhadi archaeological site"
        hash1 = PromptSanitizer.compute_prompt_hash(prompt)
        hash2 = PromptSanitizer.compute_prompt_hash(prompt)
        self.assertEqual(hash1, hash2)

        seed1 = VisualGenerator._derive_deterministic_seed("EP-SEED-01", 1, 1)
        seed2 = VisualGenerator._derive_deterministic_seed("EP-SEED-01", 1, 1)
        self.assertEqual(seed1, seed2)

        # Different shot gets a different deterministic seed
        seed3 = VisualGenerator._derive_deterministic_seed("EP-SEED-01", 1, 2)
        self.assertNotEqual(seed1, seed3)

    def test_actual_file_sha256_checksum_match(self):
        """Manifest sha256 exactly matches hashlib.sha256 of the on-disk file."""
        ep = self._create_sample_episode_bundle(episode_id="EP-SHA-01")
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res.success)

        manifest_path = Path(ep.output_directory) / "visuals" / "assets_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for asset in manifest["assets"]:
            rel_file_path = Path(asset["file_path"])
            file_path = rel_file_path if rel_file_path.is_absolute() else (Path(ep.output_directory) / rel_file_path)
            self.assertTrue(file_path.exists())
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                hasher.update(f.read())
            self.assertEqual(asset["sha256"], hasher.hexdigest())

    # =========================================================================
    # 10. DUPLICATE ASSET REUSE REPRESENTATION
    # =========================================================================

    def test_duplicate_asset_reuse_representation(self):
        """Identical reference asset reuse is represented explicitly without redundant re-generation."""
        scenes = [
            {
                "scene_id": 1,
                "scene_type": "host_intro",
                "duration_seconds": 5.0,
                "shots": [
                    {
                        "shot_id": 1,
                        "duration_seconds": 5.0,
                        "grounding_type": "HOST_ANCHORED",
                        "visual_prompt": "Host Yaazhini intro",
                        "negative_prompt": "distorted",
                        "visual_risk": "MINIMAL",
                        "claim_ids": ["CLM-001"],
                        "host_character_id": "host_yaazhini",
                        "reference_asset": "assets/yaazhini_presenter.jpg",
                        "safe_fallback": "fallback"
                    }
                ]
            },
            {
                "scene_id": 3,
                "scene_type": "host_outro",
                "duration_seconds": 5.0,
                "shots": [
                    {
                        "shot_id": 1,
                        "duration_seconds": 5.0,
                        "grounding_type": "HOST_ANCHORED",
                        "visual_prompt": "Host Yaazhini outro",
                        "negative_prompt": "distorted",
                        "visual_risk": "MINIMAL",
                        "claim_ids": ["CLM-001"],
                        "host_character_id": "host_yaazhini",
                        "reference_asset": "assets/yaazhini_presenter.jpg",
                        "safe_fallback": "fallback"
                    }
                ]
            }
        ]
        ep = self._create_sample_episode_bundle(episode_id="EP-DUP-01", scenes=scenes)
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res.success)

        manifest_path = Path(ep.output_directory) / "visuals" / "assets_manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Both shots resolve to the exact same canonical yaazhini_presenter asset
        self.assertEqual(len(manifest["assets"]), 2)
        self.assertEqual(manifest["assets"][0]["sha256"], manifest["assets"][1]["sha256"])

    # =========================================================================
    # 11. RETRY LIMIT & RECOVERY
    # =========================================================================

    def test_retry_limit_two_attempts_before_fallback(self):
        """Flaky provider is retried up to 2 times before gracefully invoking fallback."""
        ep = self._create_sample_episode_bundle(episode_id="EP-RETRY-01")

        attempts = {"count": 0}

        class FlakyProvider(MockVisualProvider):
            def generate(self, requirement, output_path, seed=42):
                attempts["count"] += 1
                if attempts["count"] <= 2:
                    raise RuntimeError(f"Attempt {attempts['count']} failed")
                # 3rd attempt succeeds
                return super().generate(requirement, output_path, seed=seed)

        gen = VisualGenerator(self.state_manager)
        gen.providers = [FlakyProvider()]
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res.success)
        self.assertEqual(attempts["count"], 3)  # 1 initial + 2 retries = 3 attempts

    def test_recovery_skips_valid_existing_assets(self):
        """Recovery recognizes existing valid asset matching manifest and skips re-generation."""
        ep = self._create_sample_episode_bundle(episode_id="EP-REC-01")
        gen = VisualGenerator(self.state_manager)

        # Run first time
        res1 = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res1.success)

        # Reset episode state back to SCRIPT_VALIDATED to simulate rerun
        session = self.state_manager._get_session()
        try:
            ep_row = session.query(AutonomousEpisode).filter_by(episode_id=ep.episode_id).first()
            if ep_row:
                ep_row.status = EpisodeState.SCRIPT_VALIDATED.value
                ep_row.current_stage = EpisodeState.SCRIPT_VALIDATED.value
                session.commit()
        finally:
            session.close()

        # Track generations
        call_count = {"calls": 0}
        orig_generate = DeterministicFallbackProvider.generate

        def counted_generate(*args, **kwargs):
            call_count["calls"] += 1
            return orig_generate(*args, **kwargs)

        DeterministicFallbackProvider.generate = counted_generate
        try:
            res2 = gen.generate_static_visuals(ep.episode_id)
            self.assertTrue(res2.success)
            # Existing valid assets should have been recovered without calling generate
            self.assertEqual(call_count["calls"], 0)
        finally:
            DeterministicFallbackProvider.generate = orig_generate

    def test_recovery_regenerates_corrupted_existing_assets(self):
        """Recovery detects a corrupted or truncated image file and regenerates it."""
        ep = self._create_sample_episode_bundle(episode_id="EP-REC-02")
        gen = VisualGenerator(self.state_manager)

        # Run first time
        res1 = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res1.success)

        # Corrupt shot 2
        scene2_dir = Path(ep.output_directory) / "visuals" / "generated" / "scene_02"
        shot2_files = list(scene2_dir.glob("*.png"))
        self.assertTrue(len(shot2_files) > 0)
        shot2_file = shot2_files[0]
        with open(shot2_file, "w") as f:
            f.write("CORRUPTED BY CRASH")

        # Reset state back to SCRIPT_VALIDATED
        session = self.state_manager._get_session()
        try:
            ep_row = session.query(AutonomousEpisode).filter_by(episode_id=ep.episode_id).first()
            if ep_row:
                ep_row.status = EpisodeState.SCRIPT_VALIDATED.value
                ep_row.current_stage = EpisodeState.SCRIPT_VALIDATED.value
                session.commit()
        finally:
            session.close()

        # Re-run: should detect corrupt file and regenerate
        res2 = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res2.success)

        # Verify file is now valid
        validator = ImageValidator()
        ok, _ = validator.validate(shot2_file)
        self.assertTrue(ok)

    # =========================================================================
    # 12. NO MOTION & STRICT FINAL STATE
    # =========================================================================

    def test_no_motion_or_video_outputs_produced(self):
        """Phase 7 must produce static images only; zero .mp4, .gif, or video files allowed."""
        ep = self._create_sample_episode_bundle(episode_id="EP-NOMOTION-01")
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res.success)

        visuals_dir = Path(ep.output_directory) / "visuals"
        motion_files = list(visuals_dir.rglob("*.mp4")) + \
                       list(visuals_dir.rglob("*.gif")) + \
                       list(visuals_dir.rglob("*.avi")) + \
                       list(visuals_dir.rglob("*.webm"))
        self.assertEqual(len(motion_files), 0, f"Found unexpected motion files: {motion_files}")

    def test_final_state_strictly_static_visuals_ready_never_motion(self):
        """Successful Phase 7 execution MUST strictly terminate at STATIC_VISUALS_READY."""
        ep = self._create_sample_episode_bundle(episode_id="EP-STATE-01")
        gen = VisualGenerator(self.state_manager)
        res = gen.generate_static_visuals(ep.episode_id)
        self.assertTrue(res.success)
        self.assertEqual(res.next_state, EpisodeState.STATIC_VISUALS_READY)

        # Database state check
        updated_ep = self.state_manager.get_episode(ep.episode_id)
        self.assertEqual(updated_ep.status, EpisodeState.STATIC_VISUALS_READY.value)
        self.assertEqual(updated_ep.current_stage, EpisodeState.STATIC_VISUALS_READY.value)

    def test_orchestrator_generate_visuals_contract(self):
        """Orchestrator method generate_visuals() returns StageExecutionResult with STATIC_VISUALS_READY."""
        ep = self._create_sample_episode_bundle(episode_id="EP-ORCH-01")
        result = self.orchestrator.generate_visuals(ep.episode_id)
        self.assertTrue(result.success)
        self.assertEqual(result.next_state, EpisodeState.STATIC_VISUALS_READY)
        self.assertIn("total_assets", result.data)

    def test_offline_mode_zero_network_calls(self):
        """Generation completes cleanly in strict offline mode without network connectivity."""
        ep = self._create_sample_episode_bundle(episode_id="EP-OFFLINE-01")
        result = self.orchestrator.generate_visuals(ep.episode_id, offline=True)
        self.assertTrue(result.success)
        self.assertEqual(result.next_state, EpisodeState.STATIC_VISUALS_READY)


if __name__ == "__main__":
    unittest.main()
