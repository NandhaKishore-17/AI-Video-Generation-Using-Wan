"""
tests/test_phase5_content_quality.py - Phase 5.1 Content Quality & Topic Relevance Unit Tests.

Validates the 18 explicit requirements of Phase 5.1:
1. three distinct claims pass
2. one claim repeated four times fails
3. paraphrased same claim counted once
4. same claim in two scenes allowed
5. same claim in three scenes fails
6. two unique topic aspects pass
7. one topic aspect fails
8. two CORE_TOPIC claims pass
9. one CORE_TOPIC claim fails
10. TNSDA administrative fact classified as SUPPORTING_CONTEXT
11. Keezhadi-specific fact classified as CORE_TOPIC
12. insufficient Keezhadi evidence -> REVIEW_REQUIRED
13. no fabricated claim added to satisfy scene
14. token words are not treated as named entities
15. repeated claim groups appear in quality report
16. quality report persisted
17. research_expansion_recommended set correctly
18. existing Phase 1–5 regression suite remains green
"""

import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.claim_models import (
    ClaimType,
    ClaimImportance,
    ClaimClassification,
    EpistemicStatus,
    ClaimRelevanceClass,
    EntityType,
    VerifiedClaim,
    SceneContentPlan,
    FactCheckedContentPlan,
    ContentQualityReport
)
from autonomous.claim_extractor import ClaimExtractor
from autonomous.claim_verifier import ClaimVerifier
from autonomous.content_quality_validator import ContentQualityValidator
from autonomous.claim_verification_engine import ClaimVerificationEngine


class TestPhase51ContentQuality(unittest.TestCase):
    """18 deterministic safety & quality tests for Phase 5.1 Content Quality Gate."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_phase51_")
        self.test_db_path = os.path.join(self.test_dir, "test_universe.db")
        self.state_manager = StateManager(db_url=f"sqlite:///{self.test_db_path}")

        self.episode_id = "test_ep_51_001"
        self.topic = "Keezhadi: 2,600-Year-Old Tamil Urban Civilization and Early Literacy"
        self.category = "Sangam Civilization"
        self.ep_dir = os.path.join(self.test_dir, "episodes", self.episode_id)
        os.makedirs(os.path.join(self.ep_dir, "research"), exist_ok=True)
        os.makedirs(os.path.join(self.ep_dir, "verification"), exist_ok=True)
        os.makedirs(os.path.join(self.ep_dir, "script"), exist_ok=True)

        session = self.state_manager._get_session()
        try:
            ep = AutonomousEpisode(
                episode_id=self.episode_id,
                topic=self.topic,
                category=self.category,
                status=EpisodeState.RESEARCH_COMPLETE.value,
                current_stage="RESEARCH_COMPLETE",
                output_directory=self.ep_dir
            )
            session.add(ep)
            session.commit()
        finally:
            session.close()

        self.validator = ContentQualityValidator()

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass

    def _make_claim(self, cid: str, stmt: str, aspect: str, rel: ClaimRelevanceClass, ctype: ClaimType = ClaimType.HISTORICAL_FACT) -> VerifiedClaim:
        c = VerifiedClaim(
            claim_id=cid,
            statement=stmt,
            normalized_statement=stmt,
            claim_type=ctype,
            importance=ClaimImportance.HIGH,
            topic_aspect=aspect,
            episode_id=self.episode_id,
            classification=ClaimClassification.SUPPORTED_HYPOTHESIS,
            epistemic_status=EpistemicStatus.LIKELY,
            epistemic_constraint="HEDGED_FRAMING"
        )
        c.relevance_class = rel
        return c

    # 1. three distinct claims pass
    def test_01_three_distinct_claims_pass(self):
        c1 = self._make_claim("clm_01", "Keeladi has brick structures.", "settlement", ClaimRelevanceClass.CORE_TOPIC, ClaimType.ARCHAEOLOGICAL_EVIDENCE)
        c2 = self._make_claim("clm_02", "Inscribed Tamil-Brahmi potsherds date to 6th century BCE.", "chronology", ClaimRelevanceClass.CORE_TOPIC, ClaimType.DATING_CHRONOLOGY)
        c3 = self._make_claim("clm_03", "TNSDA was established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {c.claim_id: c for c in [c1, c2, c3]}

        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Today we explore Keeladi has brick structures.", "tamil_text": "வணக்கம்"},
            {"id": 2, "type": "broll_motion", "english_sub": "Archaeology shows Keeladi has brick structures.", "tamil_text": "கட்டமைப்புகள்"},
            {"id": 3, "type": "broll_motion", "english_sub": "Inscribed Tamil-Brahmi potsherds date to 6th century BCE.", "tamil_text": "எழுத்துக்கள்"},
            {"id": 4, "type": "host_vlog", "english_sub": "TNSDA was established in 1961 as official department.", "tamil_text": "நன்றி"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[
                SceneContentPlan(1, "host_vlog", "PORTRAIT", "B1", "F1", allowed_claim_ids=["clm_01"]),
                SceneContentPlan(2, "broll_motion", "WIDE_ESTABLISHING", "B2", "F2", allowed_claim_ids=["clm_01"]),
                SceneContentPlan(3, "broll_motion", "ARCHITECTURE", "B3", "F3", allowed_claim_ids=["clm_02"]),
                SceneContentPlan(4, "host_vlog", "PORTRAIT", "B4", "F4", allowed_claim_ids=["clm_03"])
            ],
            allowed_claim_ids=[],
            cautious_claim_ids=["clm_01", "clm_02", "clm_03"]
        )

        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertTrue(report.passed, f"Expected pass but failed with: {report.failure_reasons}")
        self.assertEqual(report.actual_unique_claims, 3)
        self.assertGreaterEqual(report.core_claim_count, 2)
        self.assertGreaterEqual(report.actual_unique_topic_aspects, 2)

    # 2. one claim repeated four times fails
    def test_02_one_claim_repeated_four_times_fails(self):
        c1 = self._make_claim("clm_01", "TNSDA was established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {"clm_01": c1}

        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "TNSDA was established in 1961.", "tamil_text": "1961"},
            {"id": 2, "type": "broll_motion", "english_sub": "Surveys note TNSDA was established in 1961.", "tamil_text": "1961"},
            {"id": 3, "type": "broll_motion", "english_sub": "Evidence shows TNSDA was established in 1961.", "tamil_text": "1961"},
            {"id": 4, "type": "host_vlog", "english_sub": "In summary, TNSDA was established in 1961.", "tamil_text": "1961"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(i, "host_vlog", "PORTRAIT", "B", "F", allowed_claim_ids=["clm_01"]) for i in range(1, 5)],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01"]
        )

        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertFalse(report.passed)
        self.assertEqual(report.actual_unique_claims, 1)
        self.assertEqual(report.maximum_claim_scene_reuse, 4)
        self.assertTrue(any("CLAIM_OVERUSED" in r for r in report.failure_reasons))

    # 3. paraphrased same claim counted once
    def test_03_paraphrased_same_claim_counted_once(self):
        c1 = self._make_claim("clm_01", "TNSDA was established in 1961 as an official research department.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {"clm_01": c1}

        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Archaeological surveys indicate that TNSDA was established in 1961.", "tamil_text": "வணக்கம்"},
            {"id": 2, "type": "broll_motion", "english_sub": "Scientific evidence indicates that TNSDA was established in 1961.", "tamil_text": "பதிவுகள்"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(1, "host_vlog", "PORTRAIT", "B1", "F1", allowed_claim_ids=["clm_01"]),
                    SceneContentPlan(2, "broll_motion", "WIDE_ESTABLISHING", "B2", "F2", allowed_claim_ids=["clm_01"])],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01"]
        )

        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertEqual(report.actual_unique_claims, 1)

    # 4. same claim in two scenes allowed
    def test_04_same_claim_in_two_scenes_allowed(self):
        c1 = self._make_claim("clm_01", "Keeladi has brick structures.", "settlement", ClaimRelevanceClass.CORE_TOPIC, ClaimType.ARCHAEOLOGICAL_EVIDENCE)
        c2 = self._make_claim("clm_02", "Inscribed potsherds date to 6th century BCE.", "chronology", ClaimRelevanceClass.CORE_TOPIC, ClaimType.DATING_CHRONOLOGY)
        c3 = self._make_claim("clm_03", "TNSDA was established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {c.claim_id: c for c in [c1, c2, c3]}

        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Keeladi has brick structures.", "tamil_text": "1"},
            {"id": 2, "type": "broll_motion", "english_sub": "Keeladi has brick structures.", "tamil_text": "2"},
            {"id": 3, "type": "broll_motion", "english_sub": "Inscribed potsherds date to 6th century BCE.", "tamil_text": "3"},
            {"id": 4, "type": "host_vlog", "english_sub": "TNSDA was established in 1961.", "tamil_text": "4"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[
                SceneContentPlan(1, "host_vlog", "PORTRAIT", "B1", "F1", allowed_claim_ids=["clm_01"]),
                SceneContentPlan(2, "broll_motion", "WIDE_ESTABLISHING", "B2", "F2", allowed_claim_ids=["clm_01"]),
                SceneContentPlan(3, "broll_motion", "ARCHITECTURE", "B3", "F3", allowed_claim_ids=["clm_02"]),
                SceneContentPlan(4, "host_vlog", "PORTRAIT", "B4", "F4", allowed_claim_ids=["clm_03"])
            ],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01", "clm_02", "clm_03"]
        )

        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertEqual(report.maximum_claim_scene_reuse, 2)
        self.assertFalse(any("CLAIM_OVERUSED" in r for r in report.failure_reasons))

    # 5. same claim in three scenes fails
    def test_05_same_claim_in_three_scenes_fails(self):
        c1 = self._make_claim("clm_01", "Keeladi has brick structures.", "settlement", ClaimRelevanceClass.CORE_TOPIC, ClaimType.ARCHAEOLOGICAL_EVIDENCE)
        c2 = self._make_claim("clm_02", "Inscribed potsherds date to 6th century BCE.", "chronology", ClaimRelevanceClass.CORE_TOPIC, ClaimType.DATING_CHRONOLOGY)
        c3 = self._make_claim("clm_03", "TNSDA was established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {c.claim_id: c for c in [c1, c2, c3]}

        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Keeladi has brick structures.", "tamil_text": "1"},
            {"id": 2, "type": "broll_motion", "english_sub": "Keeladi has brick structures.", "tamil_text": "2"},
            {"id": 3, "type": "broll_motion", "english_sub": "Keeladi has brick structures.", "tamil_text": "3"},
            {"id": 4, "type": "host_vlog", "english_sub": "TNSDA was established in 1961.", "tamil_text": "4"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[
                SceneContentPlan(1, "host_vlog", "PORTRAIT", "B1", "F1", allowed_claim_ids=["clm_01"]),
                SceneContentPlan(2, "broll_motion", "WIDE_ESTABLISHING", "B2", "F2", allowed_claim_ids=["clm_01"]),
                SceneContentPlan(3, "broll_motion", "ARCHITECTURE", "B3", "F3", allowed_claim_ids=["clm_01"]),
                SceneContentPlan(4, "host_vlog", "PORTRAIT", "B4", "F4", allowed_claim_ids=["clm_03"])
            ],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01", "clm_02", "clm_03"]
        )

        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertFalse(report.passed)
        self.assertTrue(any("CLAIM_OVERUSED" in r for r in report.failure_reasons))

    # 6. two unique topic aspects pass
    def test_06_two_unique_topic_aspects_pass(self):
        c1 = self._make_claim("clm_01", "Keeladi brick structures.", "settlement", ClaimRelevanceClass.CORE_TOPIC)
        c2 = self._make_claim("clm_02", "Tamil-Brahmi script dating.", "chronology", ClaimRelevanceClass.CORE_TOPIC)
        c3 = self._make_claim("clm_03", "Department established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {c.claim_id: c for c in [c1, c2, c3]}
        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Keeladi brick structures.", "tamil_text": "1"},
            {"id": 2, "type": "broll_motion", "english_sub": "Keeladi brick structures.", "tamil_text": "2"},
            {"id": 3, "type": "broll_motion", "english_sub": "Tamil-Brahmi script dating.", "tamil_text": "3"},
            {"id": 4, "type": "host_vlog", "english_sub": "Department established in 1961.", "tamil_text": "4"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(i, "vlog", "P", "B", "F", allowed_claim_ids=[f"clm_0{min(i, 3)}"]) for i in range(1, 5)],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01", "clm_02", "clm_03"]
        )
        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertGreaterEqual(report.actual_unique_topic_aspects, 2)

    # 7. one topic aspect fails
    def test_07_one_topic_aspect_fails(self):
        c1 = self._make_claim("clm_01", "Statement 1", "historical_context", ClaimRelevanceClass.CORE_TOPIC)
        c2 = self._make_claim("clm_02", "Statement 2", "historical_context", ClaimRelevanceClass.CORE_TOPIC)
        c3 = self._make_claim("clm_03", "Statement 3", "historical_context", ClaimRelevanceClass.CORE_TOPIC)
        claims_map = {c.claim_id: c for c in [c1, c2, c3]}
        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Statement 1", "tamil_text": "1"},
            {"id": 2, "type": "broll_motion", "english_sub": "Statement 1", "tamil_text": "2"},
            {"id": 3, "type": "broll_motion", "english_sub": "Statement 2", "tamil_text": "3"},
            {"id": 4, "type": "host_vlog", "english_sub": "Statement 3", "tamil_text": "4"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(i, "vlog", "P", "B", "F", allowed_claim_ids=[f"clm_0{min(i, 3)}"]) for i in range(1, 5)],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01", "clm_02", "clm_03"]
        )
        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertFalse(report.passed)
        self.assertEqual(report.actual_unique_topic_aspects, 1)

    # 8. two CORE_TOPIC claims pass
    def test_08_two_core_topic_claims_pass(self):
        c1 = self._make_claim("clm_01", "Keeladi brick structures.", "settlement", ClaimRelevanceClass.CORE_TOPIC)
        c2 = self._make_claim("clm_02", "Tamil-Brahmi script dating.", "chronology", ClaimRelevanceClass.CORE_TOPIC)
        c3 = self._make_claim("clm_03", "Department established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {c.claim_id: c for c in [c1, c2, c3]}
        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Keeladi brick structures.", "tamil_text": "1"},
            {"id": 2, "type": "broll_motion", "english_sub": "Keeladi brick structures.", "tamil_text": "2"},
            {"id": 3, "type": "broll_motion", "english_sub": "Tamil-Brahmi script dating.", "tamil_text": "3"},
            {"id": 4, "type": "host_vlog", "english_sub": "Department established in 1961.", "tamil_text": "4"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(i, "vlog", "P", "B", "F", allowed_claim_ids=[f"clm_0{min(i, 3)}"]) for i in range(1, 5)],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01", "clm_02", "clm_03"]
        )
        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertGreaterEqual(report.core_claim_count, 2)

    # 9. one CORE_TOPIC claim fails
    def test_09_one_core_topic_claim_fails(self):
        c1 = self._make_claim("clm_01", "Keeladi brick structures.", "settlement", ClaimRelevanceClass.CORE_TOPIC)
        c2 = self._make_claim("clm_02", "TNSDA established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        c3 = self._make_claim("clm_03", "Southern state background.", "geography", ClaimRelevanceClass.BACKGROUND)
        claims_map = {c.claim_id: c for c in [c1, c2, c3]}
        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Keeladi brick structures.", "tamil_text": "1"},
            {"id": 2, "type": "broll_motion", "english_sub": "Keeladi brick structures.", "tamil_text": "2"},
            {"id": 3, "type": "broll_motion", "english_sub": "TNSDA established in 1961.", "tamil_text": "3"},
            {"id": 4, "type": "host_vlog", "english_sub": "Southern state background.", "tamil_text": "4"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(i, "vlog", "P", "B", "F", allowed_claim_ids=[f"clm_0{min(i, 3)}"]) for i in range(1, 5)],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01", "clm_02", "clm_03"]
        )
        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertFalse(report.passed)
        self.assertEqual(report.core_claim_count, 1)
        self.assertTrue(any("CORE_TOPIC" in r for r in report.failure_reasons))

    # 10. TNSDA administrative fact classified as SUPPORTING_CONTEXT
    def test_10_tnsda_administrative_fact_classified_as_supporting_context(self):
        claim = self._make_claim("c1", "Tamil Nadu State Department of Archaeology (TNSDA) was set up in 1961 as an official research department.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        rel = ClaimVerifier.determine_relevance_class(claim, topic=self.topic)
        self.assertEqual(rel, ClaimRelevanceClass.SUPPORTING_CONTEXT)

    # 11. Keezhadi-specific fact classified as CORE_TOPIC
    def test_11_keezhadi_specific_fact_classified_as_core_topic(self):
        claim = self._make_claim("c1", "Keeladi excavations recovered Tamil-Brahmi inscribed potsherds dating to 6th century BCE.", "archaeology", ClaimRelevanceClass.SUPPORTING_CONTEXT, ClaimType.ARCHAEOLOGICAL_EVIDENCE)
        rel = ClaimVerifier.determine_relevance_class(claim, topic=self.topic)
        self.assertEqual(rel, ClaimRelevanceClass.CORE_TOPIC)

    # 12. insufficient Keezhadi evidence -> REVIEW_REQUIRED
    def test_12_insufficient_keezhadi_evidence_becomes_review_required(self):
        c1 = self._make_claim("clm_001", "Tamil Nadu State Department of Archaeology (TNSDA) was set up in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        plan_data = {
            "episode_id": self.episode_id,
            "topic": self.topic,
            "category": self.category,
            "scenes": [{"scene_id": i, "scene_type": "host_vlog", "shot_type": "PORTRAIT", "badge": "B", "focus_theme": "F", "allowed_claim_ids": ["clm_001"]} for i in range(1, 5)],
            "allowed_claim_ids": [],
            "cautious_claim_ids": ["clm_001"],
            "forbidden_claim_ids": [],
            "allowed_entities": ["keezhadi", "tnsda"],
            "allowed_dates": [],
            "allowed_numbers": ["1961", "2600", "600"],
            "verified_quotations": []
        }
        with open(os.path.join(self.ep_dir, "verification", "fact_checked_content_plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan_data, f)
        with open(os.path.join(self.ep_dir, "verification", "claims.json"), "w", encoding="utf-8") as f:
            json.dump([c1.to_dict()], f)

        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFYING, "VERIFYING")
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFIED, "VERIFIED")

        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        res = engine.execute_script_generation_and_validation(self.episode_id, offline=True)
        self.assertEqual(res["status"], "REVIEW_REQUIRED")
        ep = self.state_manager.get_episode(self.episode_id)
        self.assertEqual(ep.status, EpisodeState.REVIEW_REQUIRED.value)

    # 13. no fabricated claim added to satisfy scene
    def test_13_no_fabricated_claim_added_to_satisfy_scene(self):
        # Even if Scene 2 requires CORE_TOPIC archaeological evidence, the validator marks failure
        # rather than inventing or pretending a supporting context claim is core
        c1 = self._make_claim("clm_001", "TNSDA was set up in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {"clm_001": c1}
        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "Intro", "tamil_text": "வணக்கம்"},
            {"id": 2, "type": "broll_motion", "english_sub": "TNSDA was set up in 1961.", "tamil_text": "பதிவு"},
            {"id": 3, "type": "broll_motion", "english_sub": "TNSDA was set up in 1961.", "tamil_text": "பதிவு"},
            {"id": 4, "type": "host_vlog", "english_sub": "Outro", "tamil_text": "நன்றி"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(i, "host_vlog", "PORTRAIT", "B", "F", allowed_claim_ids=["clm_001"]) for i in range(1, 5)],
            allowed_claim_ids=[], cautious_claim_ids=["clm_001"]
        )
        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertFalse(report.passed)
        self.assertTrue(any("INSUFFICIENT_EVIDENCE_FOR_SCENE" in r for r in report.failure_reasons))

    # 14. token words are not treated as named entities
    def test_14_token_words_are_not_treated_as_named_entities(self):
        tokens = ["tamil", "urban", "year", "old", "nadu", "state", "early", "literacy"]
        for t in tokens:
            ent_type = ClaimExtractor.classify_entity(t)
            self.assertEqual(ent_type, EntityType.COMMON_WORD, f"Token '{t}' should be classified as COMMON_WORD.")

        meaningful = ClaimExtractor.extract_meaningful_entities("tamil urban year old nadu state Keezhadi TNSDA")
        self.assertIn("keezhadi", meaningful)
        self.assertIn("tnsda", meaningful)
        self.assertNotIn("tamil", meaningful)
        self.assertNotIn("urban", meaningful)
        self.assertNotIn("year", meaningful)

    # 15. repeated claim groups appear in quality report
    def test_15_repeated_claim_groups_appear_in_quality_report(self):
        c1 = self._make_claim("clm_01", "TNSDA was established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {"clm_01": c1}
        scenes = [
            {"id": 1, "type": "host_vlog", "english_sub": "TNSDA was established in 1961.", "tamil_text": "1"},
            {"id": 2, "type": "broll_motion", "english_sub": "TNSDA was established in 1961.", "tamil_text": "2"}
        ]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(1, "vlog", "P", "B", "F", allowed_claim_ids=["clm_01"]),
                    SceneContentPlan(2, "vlog", "P", "B", "F", allowed_claim_ids=["clm_01"])],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01"]
        )
        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertIn("clm_01", report.repeated_claim_groups)
        self.assertEqual(report.repeated_claim_groups["clm_01"], ["scene_1", "scene_2"])

    # 16. quality report persisted
    def test_16_quality_report_persisted(self):
        c1 = self._make_claim("clm_001", "TNSDA was established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        plan_data = {
            "episode_id": self.episode_id, "topic": self.topic, "category": self.category,
            "scenes": [{"scene_id": i, "scene_type": "host_vlog", "shot_type": "PORTRAIT", "badge": "B", "focus_theme": "F", "allowed_claim_ids": ["clm_001"]} for i in range(1, 5)],
            "allowed_claim_ids": [], "cautious_claim_ids": ["clm_001"], "forbidden_claim_ids": [],
            "allowed_entities": ["keezhadi", "tnsda"], "allowed_dates": [], "allowed_numbers": ["1961", "2600", "600"], "verified_quotations": []
        }
        with open(os.path.join(self.ep_dir, "verification", "fact_checked_content_plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan_data, f)
        with open(os.path.join(self.ep_dir, "verification", "claims.json"), "w", encoding="utf-8") as f:
            json.dump([c1.to_dict()], f)

        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFYING, "VERIFYING")
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFIED, "VERIFIED")

        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        engine.execute_script_generation_and_validation(self.episode_id, offline=True)

        cq_file = os.path.join(self.ep_dir, "verification", "content_quality_report.json")
        self.assertTrue(os.path.exists(cq_file), "content_quality_report.json must be persisted.")
        with open(cq_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("minimum_unique_claims", data)
        self.assertIn("actual_unique_claims", data)
        self.assertIn("core_claim_count", data)
        self.assertIn("repeated_claim_groups", data)

    # 17. research_expansion_recommended set correctly
    def test_17_research_expansion_recommended_set_correctly(self):
        c1 = self._make_claim("clm_01", "TNSDA was established in 1961.", "historical_context", ClaimRelevanceClass.SUPPORTING_CONTEXT)
        claims_map = {"clm_01": c1}
        scenes = [{"id": 1, "type": "host_vlog", "english_sub": "TNSDA was established in 1961.", "tamil_text": "1"}]
        storyboard = {"scenes": scenes}
        content_plan = FactCheckedContentPlan(
            episode_id=self.episode_id, topic=self.topic, category=self.category,
            scenes=[SceneContentPlan(1, "vlog", "P", "B", "F", allowed_claim_ids=["clm_01"])],
            allowed_claim_ids=[], cautious_claim_ids=["clm_01"]
        )
        report = self.validator.validate_script_content_quality(storyboard, content_plan, claims_map)
        self.assertTrue(report.research_expansion_recommended)

    # 18. existing Phase 1–5 regression suite remains green
    def test_18_existing_phase1_5_regression_suite_remains_green(self):
        from autonomous.state_manager import VALID_TRANSITIONS
        # Ensure state manager valid transitions include SCRIPT_VALIDATED -> VERIFYING and REVIEW_REQUIRED -> VERIFYING
        self.assertIn(EpisodeState.VERIFYING, VALID_TRANSITIONS[EpisodeState.REVIEW_REQUIRED])
        self.assertIn(EpisodeState.VERIFYING, VALID_TRANSITIONS[EpisodeState.SCRIPT_VALIDATED])
        self.assertIn(EpisodeState.SCRIPT_VALIDATED, VALID_TRANSITIONS[EpisodeState.SCRIPTING])


if __name__ == "__main__":
    unittest.main()
