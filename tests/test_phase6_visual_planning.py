"""
tests/test_phase6_visual_planning.py - Unit & Regression Tests for Phase 6 Visual Planning.

Validates:
1. test_no_unconfigured_character_invented
2. test_configured_character_preserved
3. test_review_required_cannot_enter_visual_planning
4. test_non_validated_episode_cannot_enter_visual_planning
5. test_visual_prompt_does_not_add_unsupported_factual_detail
6. test_phase7_consumable_visual_plan_schema
7. test_grounding_type_claim_traceability
8. test_script_character_metadata_priority
9. test_channel_config_priority_over_script
10. test_valid_transition_to_visual_planning
11. test_recovery_manager_visual_planning_state
12. test_pipeline_readiness_operational
"""

import sys
import os
import json
import unittest
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
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
from autonomous.claim_models import VerifiedClaim, ClaimClassification, ClaimRelevanceClass


class TestPhase6VisualPlanning(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_phase6_")
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

    def _create_sample_validated_episode(
        self,
        episode_id: str = "ep_test_001",
        state: EpisodeState = EpisodeState.SCRIPT_VALIDATED,
        include_character_in_script: bool = False,
        scene_1_type: str = "host_vlog"
    ) -> Path:
        """Helper to create an episode and populate a standard validated script."""
        ep_dir = Path(self.test_dir) / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)

        session = self.state_manager._get_session()
        try:
            ep = AutonomousEpisode(
                episode_id=episode_id,
                topic="Ancient Keezhadi Brick Architecture",
                category="Sangam Civilization",
                status=state.value,
                current_stage=state.value,
                output_directory=str(ep_dir)
            )
            session.add(ep)
            session.commit()
        finally:
            session.close()

        # Create script
        script_dir = ep_dir / "script"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_data = {
            "title_english": "Ancient Keezhadi Brick Architecture",
            "scenes": [
                {
                    "id": 1,
                    "type": scene_1_type,
                    "shot_type": "PORTRAIT" if scene_1_type == "host_vlog" else "WIDE_ESTABLISHING",
                    "badge": "Introduction to Keezhadi",
                    "english_sub": "Welcome to the ancient archaeological site of Keezhadi.",
                    "tamil_text": "கீழடி தொல்லியல் தளத்திற்கு வரவேற்கிறோம்.",
                    "visual_description": "Cinematic shot of ancient Keezhadi setting, excavation trenches in background",
                    "visual_prompt": "Cinematic shot of ancient Keezhadi setting, excavation trenches in background",
                    "motion_plan": {"camera": "slow_dolly_forward", "motion_budget": "LOW"},
                    "claim_ids": ["clm_001"]
                },
                {
                    "id": 2,
                    "type": "broll_motion",
                    "shot_type": "ARCHITECTURE",
                    "badge": "Sangam Brick Structures",
                    "english_sub": "Excavations revealed structured brick water channels.",
                    "tamil_text": "அகழ்வாராய்ச்சியில் செங்கல் வடிகால்கள் கண்டறியப்பட்டன.",
                    "visual_description": "Archaeological excavation trench with documented ancient brick drainage channels",
                    "visual_prompt": "Archaeological excavation trench with documented ancient brick drainage channels",
                    "motion_plan": {"camera": "pan_right", "motion_budget": "MEDIUM"},
                    "claim_ids": ["clm_001"]
                },
                {
                    "id": 3,
                    "type": "broll_motion",
                    "shot_type": "MAP",
                    "badge": "Vaigai Basin Map",
                    "english_sub": "The settlement was positioned along the fertile Vaigai river valley.",
                    "tamil_text": "வைகை நதிப் படுகையில் இந்த நாகரிகம் அமைந்திருந்தது.",
                    "visual_description": "Ancient topographic trade route map of Vaigai river basin",
                    "visual_prompt": "Ancient topographic trade route map of Vaigai river basin",
                    "motion_plan": {"camera": "static_safe", "motion_budget": "LOW"},
                    "claim_ids": []
                },
                {
                    "id": 4,
                    "type": "broll_motion",
                    "shot_type": "DETAIL",
                    "badge": "Conclusion",
                    "english_sub": "These findings shed light on early urban planning.",
                    "tamil_text": "இந்த கண்டுபிடிப்புகள் பண்டைய நகர திட்டமிடலை விளக்குகின்றன.",
                    "visual_description": "Detailed close up of excavated Sangam pottery sherds",
                    "visual_prompt": "Detailed close up of excavated Sangam pottery sherds",
                    "motion_plan": {"camera": "subtle_drift", "motion_budget": "LOW"},
                    "claim_ids": ["clm_001"]
                }
            ]
        }

        if include_character_in_script:
            script_data["character"] = {
                "character_id": "script_host",
                "name": "Arun Scholar",
                "description": "Archaeology documentary presenter in field attire"
            }

        with open(script_dir / "script.json", "w", encoding="utf-8") as f:
            json.dump(script_data, f, indent=2)

        # Create verification confidence report with verified claims
        verif_dir = ep_dir / "verification"
        verif_dir.mkdir(parents=True, exist_ok=True)
        conf_data = {
            "claims_breakdown": [
                {
                    "claim_id": "clm_001",
                    "statement": "Excavations at Keezhadi revealed Sangam era brick structures and drainage systems.",
                    "normalized_statement": "Excavations at Keezhadi revealed Sangam era brick structures and drainage systems.",
                    "claim_type": "HISTORICAL_FACT",
                    "importance": "HIGH",
                    "topic_aspect": "archaeology",
                    "confidence_score": 0.95,
                    "classification": "VERIFIED_FACT",
                    "relevance_class": "CORE_TOPIC"
                }
            ]
        }
        with open(verif_dir / "confidence_report.json", "w", encoding="utf-8") as f:
            json.dump(conf_data, f, indent=2)

        return ep_dir

    def test_no_unconfigured_character_invented(self):
        """
        1. REMOVE HARDCODED HOST VENNILA ASSUMPTION:
        When no character is configured in channel/universe or script metadata,
        Phase 6 must NOT invent Host Vennila or any other recurring presenter.
        """
        ep_dir = self._create_sample_validated_episode("ep_no_char", scene_1_type="host_vlog")
        planner = VisualPlanner(state_manager=self.state_manager, channel_config={})

        plan = planner.generate_plan("ep_no_char")

        # Master plan must have no character consistency rules
        self.assertIsNone(plan.character_consistency_rules)

        # Every scene must have character_specs as None
        for sc in plan.scenes:
            self.assertIsNone(sc.character_specs, f"Scene {sc.scene_id} should not have character specs.")
            # Verify no mention of Yaazhini or unconfigured host in prompt
            self.assertNotIn("Yaazhini", sc.visual_prompt)
            self.assertNotIn("Vennila", sc.visual_prompt)
            self.assertNotIn("Host", sc.visual_prompt)
            self.assertNotIn("AI vlogger", sc.visual_prompt)
            # Verify asset filenames do not use '_host.png'
            self.assertTrue(sc.asset_requirement.filename.endswith("_broll.png"),
                            f"Asset filename should be broll, got {sc.asset_requirement.filename}")

        # Scene 1 was originally host_vlog, but without character it must become documentary_broll
        sc1 = plan.scenes[0]
        self.assertEqual(sc1.scene_type, "documentary_broll")
        self.assertEqual(sc1.shot_type, "WIDE_ESTABLISHING")

    def test_configured_character_preserved(self):
        """
        1. CONFIGURED CHARACTER PRESERVED:
        When channel/project configuration explicitly defines Host Yaazhini,
        it must be preserved as configuration rather than hardcoded.
        """
        ep_dir = self._create_sample_validated_episode("ep_char_cfg", scene_1_type="host_vlog")
        channel_cfg = {
            "character": {
                "character_id": "host_yaazhini",
                "name": "Host Yaazhini",
                "description": "South Indian female documentary presenter in olive-green kurti",
                "consistency_token": "yaazhini_host",
                "rules": ["Maintain consistent face across all host scenes."]
            }
        }
        planner = VisualPlanner(state_manager=self.state_manager, channel_config=channel_cfg)

        plan = planner.generate_plan("ep_char_cfg")

        self.assertIsNotNone(plan.character_consistency_rules)
        self.assertEqual(plan.character_consistency_rules["name"], "Host Yaazhini")

        # Scene 1 was host_vlog, so it must preserve character specs
        sc1 = plan.scenes[0]
        self.assertIsNotNone(sc1.character_specs)
        self.assertEqual(sc1.character_specs["name"], "Host Yaazhini")
        self.assertEqual(sc1.asset_requirement.filename, "scene_1_host.png")

    def test_canonical_yaazhini_channel_configuration(self):
        """
        Phase 6.1: Verify that canonical AutonomousSettings channel identity provides
        Yaazhini as configured character when enabled, with correct reference image and metadata.
        """
        from autonomous.config import autonomous_settings
        char_cfg = autonomous_settings.channel.get_character_config()
        self.assertIsNotNone(char_cfg)
        self.assertEqual(char_cfg["name"], "Yaazhini")
        self.assertEqual(char_cfg["name_ta"], "யாழினி")
        self.assertEqual(char_cfg["character_id"], "host_yaazhini")
        self.assertEqual(char_cfg["reference_image"], "assets/yaazhini_presenter.jpg")
        self.assertTrue((PROJECT_ROOT / char_cfg["reference_image"]).exists())

    def test_phase6_1_1_canonical_identity_spelling(self):
        """
        Phase 6.1.1: Verify exact canonical channel identity spelling.
        Brand name must be EXACTLY 'Kaalapadhivugal' (no spaces, no hyphens).
        Handle must be '@kaalapadhivugal'.
        English meaning must be 'Records of Time'.
        Tamil name must be 'காலப் பதிவுகள்'.
        Host must be Yaazhini (யாழினி).
        """
        from autonomous.config import autonomous_settings
        ch = autonomous_settings.channel
        self.assertEqual(ch.channel_name_en, "Kaalapadhivugal")
        self.assertEqual(ch.channel_name_ta, "காலப் பதிவுகள்")
        self.assertEqual(ch.channel_description_en, "Records of Time")
        self.assertEqual(ch.channel_handle, "@kaalapadhivugal")
        self.assertEqual(ch.host_name_en, "Yaazhini")
        self.assertEqual(ch.host_name_ta, "யாழினி")
        self.assertEqual(ch.host_character_id, "host_yaazhini")
        self.assertEqual(ch.host_consistency_token, "yaazhini_host")

    def test_phase6_1_1_active_host_asset_isolation(self):
        """
        Phase 6.1.1: Verify active host asset directory isolation and hygiene.
        1. daily_engine/assets/hosts/ contains ONLY yaazhini_presenter.jpg.
        2. assets/yaazhini_presenter.jpg exists.
        3. Obsolete presenter images (vennila_*) are safely isolated in archive/legacy_assets/hosts/.
        """
        daily_hosts_dir = PROJECT_ROOT / "daily_engine" / "assets" / "hosts"
        root_host_asset = PROJECT_ROOT / "assets" / "yaazhini_presenter.jpg"
        archive_hosts_dir = PROJECT_ROOT / "archive" / "legacy_assets" / "hosts"

        # Check canonical host assets exist
        self.assertTrue((daily_hosts_dir / "yaazhini_presenter.jpg").exists())
        self.assertTrue(root_host_asset.exists())

        # Verify ONLY yaazhini_presenter.jpg is present in active production hosts dir
        active_host_files = [f.name for f in daily_hosts_dir.iterdir() if f.is_file()]
        self.assertEqual(active_host_files, ["yaazhini_presenter.jpg"])

        # Verify obsolete presenter files are safely archived
        self.assertTrue(archive_hosts_dir.exists())
        for obs in ["vennila_ship_deck.jpg", "vennila_tanjore_vlog.jpg", "vennila_temple_vlog.jpg"]:
            self.assertTrue((archive_hosts_dir / obs).exists(), f"Expected {obs} in archive")
            self.assertFalse((daily_hosts_dir / obs).exists(), f"Obsolete {obs} still in active hosts dir!")

    def test_no_obsolete_presenter_identity_generated(self):
        """
        Phase 6.1: Verify that obsolete identities (Vennila, Aayirathil Naan)
        are never generated into visual plans.
        """
        from autonomous.config import autonomous_settings
        ep_dir = self._create_sample_validated_episode("ep_obs_test", scene_1_type="host_vlog")
        planner = VisualPlanner(
            state_manager=self.state_manager,
            channel_config={"character": autonomous_settings.channel.get_character_config()}
        )
        plan = planner.generate_plan("ep_obs_test")
        plan_dict = plan.to_dict()
        plan_str = json.dumps(plan_dict)

        # Check that no obsolete strings appear anywhere in the generated plan
        for obsolete in ["Vennila", "Aayirathil", "Ayirathil", "ஆயிரத்தில்"]:
            self.assertNotIn(obsolete.lower(), plan_str.lower())

    def test_review_required_cannot_enter_visual_planning(self):
        """
        2. STRICT STATE GATE:
        An episode in REVIEW_REQUIRED must NOT be allowed to enter VISUAL_PLANNING.
        """
        ep_dir = self._create_sample_validated_episode("ep_rev_req", state=EpisodeState.REVIEW_REQUIRED)
        planner = VisualPlanner(state_manager=self.state_manager)

        with self.assertRaises(ValueError) as ctx:
            planner.generate_plan("ep_rev_req")
        self.assertIn("Phase 6 blocked: episode is not SCRIPT_VALIDATED", str(ctx.exception))

        # Test orchestrator entry point
        res = self.orchestrator.plan_visuals("ep_rev_req")
        self.assertFalse(res.success)
        self.assertIn("Phase 6 blocked: episode is not SCRIPT_VALIDATED", str(res.error))

        # Check DB state remains REVIEW_REQUIRED
        ep = self.state_manager.get_episode("ep_rev_req")
        self.assertEqual(ep.status, EpisodeState.REVIEW_REQUIRED.value)

        # Confirm visual_plan.json was NOT created
        plan_file = ep_dir / "visuals" / "visual_plan.json"
        self.assertFalse(plan_file.exists())

    def test_non_validated_episode_cannot_enter_visual_planning(self):
        """
        2. STRICT STATE GATE:
        RESEARCH_COMPLETE, SCRIPTING, VERIFYING, and TOPIC_SELECTED must all be rejected.
        """
        non_validated_states = [
            EpisodeState.RESEARCH_COMPLETE,
            EpisodeState.SCRIPTING,
            EpisodeState.VERIFYING,
            EpisodeState.TOPIC_SELECTED
        ]

        for idx, st in enumerate(non_validated_states):
            ep_id = f"ep_non_val_{idx}"
            self._create_sample_validated_episode(ep_id, state=st)
            planner = VisualPlanner(state_manager=self.state_manager)

            with self.assertRaises(ValueError) as ctx:
                planner.generate_plan(ep_id)
            self.assertIn("Phase 6 blocked: episode is not SCRIPT_VALIDATED", str(ctx.exception))

            res = self.orchestrator.plan_visuals(ep_id)
            self.assertFalse(res.success)
            self.assertIn("Phase 6 blocked: episode is not SCRIPT_VALIDATED", str(res.error))

            # Confirm state did not change
            ep = self.state_manager.get_episode(ep_id)
            self.assertEqual(ep.status, st.value)

    def test_visual_prompt_does_not_add_unsupported_factual_detail(self):
        """
        4 & 5. VISUAL PROMPT HARDENING & NO INVENTED DETAILS:
        Sanitizer must strip unsupported architecture (e.g. three-storey, palatial),
        exact clothing, population numbers, weapons, and apply conservative language.
        """
        planner = VisualPlanner(state_manager=self.state_manager)

        # Example from prompt:
        # BAD: "Ancient Keezhadi citizens wearing precisely documented clothing inside a three-storey urban building."
        bad_prompt = "Ancient Keezhadi citizens wearing precisely documented clothing inside a three-storey urban building."
        sanitized = planner._sanitize_conservative_prompt(
            bad_prompt,
            topic="Keezhadi",
            grounding_type=GroundingType.RECONSTRUCTION,
            claims=[]
        )

        self.assertNotIn("three-storey", sanitized.lower())
        self.assertNotIn("precisely documented clothing", sanitized.lower())
        self.assertIn("historically cautious reconstruction", sanitized.lower())
        self.assertIn("broadly supported", sanitized.lower())

        # Test unsupported population and weapons
        bad_prompt_2 = "A sprawling multi-level city with 50,000 citizens wielding steel longswords in palatial structures."
        sanitized_2 = planner._sanitize_conservative_prompt(
            bad_prompt_2,
            topic="Keezhadi",
            grounding_type=GroundingType.RECONSTRUCTION,
            claims=[]
        )
        self.assertNotIn("50,000 citizens", sanitized_2.lower())
        self.assertNotIn("steel longswords", sanitized_2.lower())
        self.assertNotIn("palatial", sanitized_2.lower())

    def test_phase7_consumable_visual_plan_schema(self):
        """
        6. PHASE 7 CONTRACT:
        Ensure visual_plan.json contains all 17 required keys for Phase 7 consumption.
        """
        ep_dir = self._create_sample_validated_episode("ep_schema_test")
        planner = VisualPlanner(state_manager=self.state_manager)

        plan = planner.generate_plan("ep_schema_test")
        plan_file = ep_dir / "visuals" / "visual_plan.json"
        self.assertTrue(plan_file.exists())

        with open(plan_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Top level schema
        expected_top_keys = {
            "episode_id", "topic", "created_at", "script_version", "plan_version",
            "status", "scenes", "global_visual_style", "character_consistency_rules",
            "visual_safety_rules", "generation_constraints", "source_provenance"
        }
        for k in expected_top_keys:
            self.assertIn(k, data)

        # Phase 7 asset requirement contract keys (all 17 required)
        required_17_keys = {
            "asset_id", "scene_id", "asset_type", "filename", "aspect_ratio",
            "resolution", "generation_prompt", "negative_prompt", "grounding_type",
            "claim_ids", "historical_period", "geographic_context", "composition",
            "lighting", "style", "safety_constraints", "fallback_asset_type"
        }

        self.assertEqual(len(data["scenes"]), 4)
        for sc in data["scenes"]:
            asset_req = sc["asset_requirement"]
            for req_key in required_17_keys:
                self.assertIn(req_key, asset_req, f"Key '{req_key}' missing from asset requirement in Scene {sc['scene_id']}")
            self.assertEqual(asset_req["aspect_ratio"], "16:9")
            self.assertEqual(asset_req["resolution"], [1280, 720])
            self.assertIsInstance(asset_req["claim_ids"], list)

    def test_grounding_type_claim_traceability(self):
        """
        4. GROUNDING TRACEABILITY:
        HISTORICAL_CLAIM_GROUNDED visuals must have valid claim_ids.
        Scenes without claims or with hypothetical models must be RECONSTRUCTION/ABSTRACT/ILLUSTRATIVE.
        """
        ep_dir = self._create_sample_validated_episode("ep_grounding_trace")
        planner = VisualPlanner(state_manager=self.state_manager)
        plan = planner.generate_plan("ep_grounding_trace")

        for sc in plan.scenes:
            if sc.grounding_type == GroundingType.HISTORICAL_CLAIM_GROUNDED:
                self.assertTrue(len(sc.associated_claim_ids) > 0,
                                f"Scene {sc.scene_id} marked as HISTORICAL_CLAIM_GROUNDED but has no claim_ids")
            if sc.shot_type == "MAP":
                self.assertEqual(sc.grounding_type, GroundingType.ABSTRACT)

    def test_script_character_metadata_priority(self):
        """
        Priority 2: If channel has no character but script metadata explicitly specifies a character,
        it must be used.
        """
        ep_dir = self._create_sample_validated_episode("ep_script_char", include_character_in_script=True)
        planner = VisualPlanner(state_manager=self.state_manager, channel_config={})
        plan = planner.generate_plan("ep_script_char")

        self.assertIsNotNone(plan.character_consistency_rules)
        self.assertEqual(plan.character_consistency_rules["name"], "Arun Scholar")

    def test_channel_config_priority_over_script(self):
        """
        Priority 1: Channel config character must override script character metadata.
        """
        ep_dir = self._create_sample_validated_episode("ep_prio_test", include_character_in_script=True)
        channel_cfg = {
            "character": {
                "character_id": "channel_host",
                "name": "Channel Lead Host",
                "description": "Official channel presenter"
            }
        }
        planner = VisualPlanner(state_manager=self.state_manager, channel_config=channel_cfg)
        plan = planner.generate_plan("ep_prio_test")

        self.assertIsNotNone(plan.character_consistency_rules)
        self.assertEqual(plan.character_consistency_rules["name"], "Channel Lead Host")

    def test_valid_transition_to_visual_planning(self):
        """
        2. VALID TRANSITION:
        SCRIPT_VALIDATED -> VISUAL_PLANNING succeeds and updates database state.
        """
        ep_dir = self._create_sample_validated_episode("ep_valid_trans")
        planner = VisualPlanner(state_manager=self.state_manager)
        plan = planner.generate_plan("ep_valid_trans")

        ep = self.state_manager.get_episode("ep_valid_trans")
        self.assertEqual(ep.status, EpisodeState.VISUAL_PLANNING.value)
        self.assertEqual(ep.current_stage, "VISUAL_PLANNING")

    def test_recovery_manager_visual_planning_state(self):
        """
        Recovery manager should advance from VISUAL_PLANNING if visual_plan.json exists,
        or retry stage if it was incomplete.
        """
        ep_dir = self._create_sample_validated_episode("ep_recovery_test", state=EpisodeState.VISUAL_PLANNING)
        # Without visual_plan.json -> RETRY_STAGE
        decision = self.recovery_manager.inspect_active_episode()
        self.assertIsNotNone(decision)
        self.assertEqual(decision.action, "RETRY_STAGE")

        # With visual_plan.json (>100 bytes) -> ADVANCE_STAGE to GENERATING_VISUALS
        vis_dir = ep_dir / "visuals"
        vis_dir.mkdir(parents=True, exist_ok=True)
        plan_content = {
            "episode_id": "ep_recovery_test",
            "topic": "Keezhadi",
            "status": "PLAN_COMPLETE",
            "scenes": [{"scene_id": 1, "asset_requirement": {"filename": "scene_1_broll.png"}}]
        }
        with open(vis_dir / "visual_plan.json", "w", encoding="utf-8") as f:
            json.dump(plan_content, f, indent=2)

        decision2 = self.recovery_manager.inspect_active_episode()
        self.assertIsNotNone(decision2)
        self.assertEqual(decision2.action, "ADVANCE_STAGE")
        self.assertEqual(decision2.target_state, EpisodeState.GENERATING_VISUALS)

    def test_pipeline_readiness_operational(self):
        """Verify orchestrator reports Visual Planning as OPERATIONAL (Phase 6)."""
        readiness = self.orchestrator.get_pipeline_readiness()
        self.assertTrue(any(k in readiness["current_phase"] for k in ("Phase 6", "Phase 7", "Phase 8", "Visual", "Motion")))
        self.assertEqual(readiness["stages"]["Visual Planning"], "OPERATIONAL (Phase 6)")


if __name__ == "__main__":
    unittest.main()
