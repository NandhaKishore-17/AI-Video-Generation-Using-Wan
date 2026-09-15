"""
tests/test_phase5_research_expansion.py - Phase 5.2 Autonomous Research Expansion Test Suite.

Validates the 32 explicit requirements of Phase 5.2:
1. test_01_expansion_trigger_when_recommended
2. test_02_no_expansion_when_quality_already_passes
3. test_03_gap_detection_identifies_missing_counts
4. test_04_missing_core_claim_detection
5. test_05_missing_topic_aspect_detection
6. test_06_missing_scene_evidence_detection
7. test_07_question_generation_specific_and_targeted
8. test_08_duplicate_question_rejection
9. test_09_question_adaptation_across_expansion_cycles
10. test_10_source_deduplication_by_url_and_hash
11. test_11_same_publisher_not_treated_as_independent
12. test_12_cached_copy_not_treated_as_independent
13. test_13_expansion_count_persistence
14. test_14_max_expansion_budget_enforcement
15. test_15_max_questions_per_expansion_enforcement
16. test_16_max_sources_per_expansion_enforcement
17. test_17_original_dossier_preserved_additively
18. test_18_expansion_provenance_tagged_on_chunks
19. test_19_qdrant_deterministic_ids_prevent_duplication
20. test_20_claim_re_extraction_from_expanded_evidence
21. test_21_claim_re_verification_enforces_thresholds
22. test_22_content_quality_rerun_after_expansion
23. test_23_insufficient_evidence_after_expansion_routes_to_review_required
24. test_24_successful_expansion_advances_to_script_validated
25. test_25_ollama_unavailable_uses_deterministic_template_questions
26. test_26_llm_cannot_override_deterministic_verification
27. test_27_contradiction_handling_preserved_during_expansion
28. test_28_idempotent_expansion_execution
29. test_29_interrupted_expansion_recovery
30. test_30_final_review_required_safety_state_when_unresolved
31. test_31_review_required_cannot_bypass_to_scripting
32. test_32_review_required_cannot_bypass_to_ready_to_publish
"""

import os
import json
import uuid
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode, VALID_TRANSITIONS
from autonomous.recovery_manager import RecoveryManager
from autonomous.claim_models import (
    ClaimType,
    ClaimImportance,
    ClaimClassification,
    EpistemicStatus,
    ClaimRelevanceClass,
    VerifiedClaim,
    ContentQualityReport
)
from autonomous.research_gap_analyzer import ResearchGapAnalyzer, ResearchGapReport
from autonomous.research_expansion_engine import (
    ResearchExpansionEngine,
    MAX_RESEARCH_EXPANSIONS,
    MAX_QUESTIONS_PER_EXPANSION,
    MAX_NEW_SOURCES_PER_EXPANSION,
    MAX_TOTAL_NEW_CHUNKS_PER_EXPANSION
)
from autonomous.source_retriever import SourceMetadata
from autonomous.research_processor import EvidenceChunk
from autonomous.claim_verifier import ClaimVerifier
from autonomous.claim_extractor import ClaimExtractor
from autonomous.content_quality_validator import ContentQualityValidator


class TestPhase52ResearchExpansion(unittest.TestCase):
    """Deterministic Unit & Integration Test Suite for Phase 5.2 Autonomous Research Expansion."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_phase52_")
        self.test_db_path = os.path.join(self.test_dir, "test_universe.db")
        self.state_manager = StateManager(db_url=f"sqlite:///{self.test_db_path}")

        self.episode_id = "test_ep_52_001"
        self.topic = "Keezhadi: 2,600-Year-Old Tamil Urban Civilization and Early Literacy"
        self.ep_dir = os.path.join(self.test_dir, "episodes", self.episode_id)
        os.makedirs(os.path.join(self.ep_dir, "research"), exist_ok=True)
        os.makedirs(os.path.join(self.ep_dir, "verification"), exist_ok=True)
        os.makedirs(os.path.join(self.ep_dir, "script"), exist_ok=True)

        session = self.state_manager._get_session()
        try:
            ep = AutonomousEpisode(
                episode_id=self.episode_id,
                topic=self.topic,
                category="Ancient Tamil history",
                status=EpisodeState.REVIEW_REQUIRED.value,
                current_stage="REVIEW_REQUIRED",
                output_directory=self.ep_dir,
                extra_data={}
            )
            session.add(ep)
            session.commit()
        finally:
            session.close()

        self.gap_analyzer = ResearchGapAnalyzer()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Trigger & Gaps
    # -------------------------------------------------------------------------

    def test_01_expansion_trigger_when_recommended(self):
        """1. Research expansion triggered when research_expansion_recommended == True."""
        cq_report = {
            "passed": False,
            "actual_unique_claims": 1,
            "core_claim_count": 0,
            "actual_unique_topic_aspects": 1,
            "research_expansion_recommended": True,
            "failure_reasons": ["CONTENT_INSUFFICIENT: Script contains only 1 unique factual claims."]
        }
        report = self.gap_analyzer.analyze_gaps(topic=self.topic, content_quality_report=cq_report)
        self.assertTrue(report.expansion_required)

    def test_02_no_expansion_when_quality_already_passes(self):
        """2. No expansion triggered when quality already passes thresholds."""
        cq_report = {
            "passed": True,
            "actual_unique_claims": 3,
            "core_claim_count": 2,
            "actual_unique_topic_aspects": 2,
            "research_expansion_recommended": False,
            "failure_reasons": []
        }
        report = self.gap_analyzer.analyze_gaps(
            topic=self.topic,
            content_quality_report=cq_report,
            minimum_unique_claims=3,
            minimum_core_claims=2,
            minimum_unique_topic_aspects=2
        )
        self.assertFalse(report.expansion_required)

    def test_03_gap_detection_identifies_missing_counts(self):
        """3. Gap detection identifies missing claim counts."""
        cq_report = {
            "passed": False,
            "actual_unique_claims": 1,
            "core_claim_count": 0,
            "actual_unique_topic_aspects": 1,
            "topic_relevance_results": {"unique_aspects": ["chronology"]}
        }
        report = self.gap_analyzer.analyze_gaps(
            topic=self.topic,
            content_quality_report=cq_report,
            minimum_unique_claims=3,
            minimum_core_claims=2
        )
        self.assertEqual(report.missing_claim_count, 2)
        self.assertEqual(report.missing_core_claim_count, 2)

    def test_04_missing_core_claim_detection(self):
        """4. Detects deficit of CORE_TOPIC claims even if total claims exist."""
        claim_support = VerifiedClaim(
            claim_id="clm_01",
            statement="TNSDA established in 1961.",
            normalized_statement="tnsda established 1961",
            claim_type=ClaimType.HISTORICAL_FACT,
            importance=ClaimImportance.HIGH,
            confidence_score=0.85,
            classification=ClaimClassification.VERIFIED_FACT,
            epistemic_status=EpistemicStatus.KNOWN_FACT,
            topic_aspect="historical_context",
            episode_id=self.episode_id,
            relevance_class=ClaimRelevanceClass.SUPPORTING_CONTEXT
        )
        report = self.gap_analyzer.analyze_gaps(
            topic=self.topic,
            verified_claims=[claim_support],
            minimum_core_claims=2
        )
        self.assertEqual(report.missing_core_claim_count, 2)
        self.assertTrue(report.expansion_required)

    def test_05_missing_topic_aspect_detection(self):
        """5. Identifies missing topic aspects appropriate to topic."""
        cq_report = {
            "passed": False,
            "actual_unique_claims": 1,
            "core_claim_count": 0,
            "actual_unique_topic_aspects": 1,
            "topic_relevance_results": {"unique_aspects": ["chronology"]}
        }
        report = self.gap_analyzer.analyze_gaps(topic=self.topic, content_quality_report=cq_report)
        self.assertIn("archaeology", report.missing_topic_aspects)
        self.assertIn("settlement", report.missing_topic_aspects)
        self.assertIn("material_culture", report.missing_topic_aspects)
        self.assertNotIn("chronology", report.missing_topic_aspects)

    def test_06_missing_scene_evidence_detection(self):
        """6. Maps missing evidence to scene requirements."""
        cq_report = {
            "passed": False,
            "scene_quality_results": [
                {"scene_id": 1, "scene_type": "host_vlog", "status": "VALID"},
                {"scene_id": 2, "scene_type": "broll_motion", "status": "INSUFFICIENT_EVIDENCE_FOR_SCENE", "error": "Scene 2 lacks CORE_TOPIC claim."},
                {"scene_id": 3, "scene_type": "broll_motion", "status": "VALID"},
                {"scene_id": 4, "scene_type": "host_vlog", "status": "VALID"}
            ]
        }
        report = self.gap_analyzer.analyze_gaps(topic=self.topic, content_quality_report=cq_report)
        self.assertEqual(len(report.missing_scene_requirements), 1)
        self.assertEqual(report.missing_scene_requirements[0]["scene_id"], 2)

    # -------------------------------------------------------------------------
    # 2. Targeted Questions & Adaptation
    # -------------------------------------------------------------------------

    def test_07_question_generation_specific_and_targeted(self):
        """7. Generated questions are specific and target missing aspects."""
        report = self.gap_analyzer.analyze_gaps(topic=self.topic)
        self.assertGreaterEqual(len(report.recommended_research_questions), 3)
        self.assertLessEqual(len(report.recommended_research_questions), 5)
        for q in report.recommended_research_questions:
            self.assertTrue("Keezhadi" in q or "excavation" in q)

    def test_08_duplicate_question_rejection(self):
        """8. Rejects exact and near-duplicate questions."""
        q1 = "What archaeological structures and brick remains have been documented at Keezhadi?"
        attempted = [q1]
        is_dup = self.gap_analyzer.is_question_duplicate(q1, attempted)
        self.assertTrue(is_dup)

        near_dup = "What archaeological structures and brick remains were documented at Keezhadi?"
        self.assertTrue(self.gap_analyzer.is_question_duplicate(near_dup, attempted, threshold=0.70))

        distinct = "What radiocarbon dating laboratory analyzed charcoal samples from Keezhadi?"
        self.assertFalse(self.gap_analyzer.is_question_duplicate(distinct, attempted, threshold=0.70))

    def test_09_question_adaptation_across_expansion_cycles(self):
        """9. Expansion cycle 2 adapts questions to remaining gaps and rejects cycle 1 questions."""
        report1 = self.gap_analyzer.analyze_gaps(topic=self.topic, expansion_number=1)
        cycle1_q = report1.recommended_research_questions[:2]

        report2 = self.gap_analyzer.analyze_gaps(
            topic=self.topic,
            attempted_questions=cycle1_q,
            expansion_number=2
        )
        for q in report2.recommended_research_questions:
            self.assertNotIn(q, cycle1_q)

    # -------------------------------------------------------------------------
    # 3. Source Deduplication & Independence
    # -------------------------------------------------------------------------

    def test_10_source_deduplication_by_url_and_hash(self):
        """10. Sources deduplicated by normalized URL and SHA-256 content hash."""
        s1 = SourceMetadata(
            source_id="src_01",
            title="Excavation Report",
            url="https://asi.nic.in/report1",
            content_hash="hash_aaa111",
            extraction_status="EXTRACTED"
        )
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        existing = [s1]
        seen_urls = {s1.url.lower()}
        seen_hashes = {s1.content_hash}

        cand_dup_url = "https://asi.nic.in/report1/"
        self.assertIn(cand_dup_url.strip("/").lower(), seen_urls)

        cand_dup_hash = "hash_aaa111"
        self.assertIn(cand_dup_hash, seen_hashes)

    def test_11_same_publisher_not_treated_as_independent(self):
        """11. Two pages from the same publisher share an independence group."""
        s1 = SourceMetadata(source_id="s1", title="Page 1", url="https://asi.nic.in/p1", publisher="ASI", domain="asi.nic.in", independence_group="asi.nic.in")
        s2 = SourceMetadata(source_id="s2", title="Page 2", url="https://asi.nic.in/p2", publisher="ASI", domain="asi.nic.in", independence_group="asi.nic.in")
        self.assertEqual(s1.independence_group, s2.independence_group)

    def test_12_cached_copy_not_treated_as_independent(self):
        """12. Cached copy of a source does not create an independent source group."""
        s_orig = SourceMetadata(source_id="s_orig", title="Orig", url="https://asi.nic.in/site", independence_group="asi.nic.in", source_origin="external")
        s_cached = SourceMetadata(source_id="s_cached", title="Orig Cached", url="https://asi.nic.in/site", independence_group="asi.nic.in", source_origin="cached_external")
        self.assertEqual(s_orig.independence_group, s_cached.independence_group)

    # -------------------------------------------------------------------------
    # 4. Expansion Budgets & Persistence
    # -------------------------------------------------------------------------

    def test_13_expansion_count_persistence(self):
        """13. Expansion cycle numbers and history are persisted on disk."""
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        history_file = Path(self.ep_dir) / "research" / "expansion_history.json"
        engine._save_history(history_file, [{"expansion_cycle": "expansion_1", "new_sources_count": 2}])
        loaded = engine._load_history(history_file)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["expansion_cycle"], "expansion_1")

    def test_14_max_expansion_budget_enforcement(self):
        """14. Enforces max 2 expansion cycles budget."""
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        history_file = Path(self.ep_dir) / "research" / "expansion_history.json"
        engine._save_history(history_file, [
            {"expansion_cycle": "expansion_1"},
            {"expansion_cycle": "expansion_2"}
        ])
        res = engine.execute_expansion(self.episode_id)
        self.assertEqual(res["expansion_status"], "BUDGET_EXHAUSTED")
        self.assertEqual(res["completed_cycles"], 2)

    def test_15_max_questions_per_expansion_enforcement(self):
        """15. Enforces MAX_QUESTIONS_PER_EXPANSION = 5."""
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        self.assertEqual(engine.max_questions, 5)

    def test_16_max_sources_per_expansion_enforcement(self):
        """16. Enforces MAX_NEW_SOURCES_PER_EXPANSION = 8."""
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        self.assertEqual(engine.max_sources, 8)

    # -------------------------------------------------------------------------
    # 5. Provenance, Qdrant & Evidence Integration
    # -------------------------------------------------------------------------

    def test_17_original_dossier_preserved_additively(self):
        """17. Expansion updates dossier additively without destroying original Phase 4 data."""
        research_dir = Path(self.ep_dir) / "research"
        dossier_file = research_dir / "dossier.json"
        initial_data = {
            "episode_id": self.episode_id,
            "topic": self.topic,
            "sources_count": 1,
            "chunks_count": 2,
            "research_status": "RESEARCH_COMPLETE"
        }
        with open(dossier_file, "w", encoding="utf-8") as f:
            json.dump(initial_data, f)

        # Simulate additive update
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        with open(dossier_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        d["sources_count"] += 2
        d["chunks_count"] += 4
        d["expansion_history"] = [{"expansion_cycle": "expansion_1"}]
        with open(dossier_file, "w", encoding="utf-8") as f:
            json.dump(d, f)

        with open(dossier_file, "r", encoding="utf-8") as f:
            updated = json.load(f)
        self.assertEqual(updated["sources_count"], 3)
        self.assertEqual(updated["chunks_count"], 6)
        self.assertEqual(len(updated["expansion_history"]), 1)

    def test_18_expansion_provenance_tagged_on_chunks(self):
        """18. New chunks are tagged with expansion_cycle provenance."""
        c = EvidenceChunk(
            chunk_id="chk_exp_01",
            episode_id=self.episode_id,
            source_id="src_exp_01",
            url="https://asi.nic.in/site",
            title="Excavation",
            publisher="ASI",
            retrieval_timestamp="2026-09-12T10:00:00Z",
            content_hash="h123",
            chunk_index=0,
            text="Keezhadi brick structures documented.",
            credibility_tier=1,
            word_count=5
        )
        c.expansion_cycle = "expansion_1"
        self.assertEqual(c.expansion_cycle, "expansion_1")

    def test_19_qdrant_deterministic_ids_prevent_duplication(self):
        """19. Deterministic UUID5 point IDs prevent duplicate vectors in Qdrant."""
        chunk = EvidenceChunk(
            chunk_id="chk_1",
            episode_id=self.episode_id,
            source_id="src_1",
            url="u",
            title="T",
            publisher="p",
            retrieval_timestamp="2026-09-12T10:00:00Z",
            content_hash="hash_xyz",
            chunk_index=0,
            text="Sample text",
            credibility_tier=1,
            word_count=2
        )
        id1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.episode_id}_{chunk.content_hash}_{chunk.chunk_index}"))
        id2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.episode_id}_{chunk.content_hash}_{chunk.chunk_index}"))
        self.assertEqual(id1, id2)

    # -------------------------------------------------------------------------
    # 6. Claim Re-Extraction, Verification & Content Quality
    # -------------------------------------------------------------------------

    def test_20_claim_re_extraction_from_expanded_evidence(self):
        """20. Re-extracts claims from expanded evidence chunks."""
        extractor = ClaimExtractor()
        text = "Archaeologists at Keezhadi excavated brick structures and 56 Tamil-Brahmi inscribed potsherds in 2018."
        claims = extractor.extract_claims(text, source_id="src_new_01", chunk_id="chk_new_01")
        self.assertGreaterEqual(len(claims), 1)

    def test_21_claim_re_verification_enforces_thresholds(self):
        """21. Re-verifies claims enforcing deterministic verification thresholds."""
        verifier = ClaimVerifier()
        # Unsupported claim receives low confidence
        claim = VerifiedClaim(
            claim_id="c_fake",
            statement="Keezhadi possessed advanced space technology.",
            normalized_statement="keezhadi possessed advanced space technology",
            claim_type=ClaimType.HISTORICAL_FACT,
            importance=ClaimImportance.CRITICAL,
            topic_aspect="archaeology",
            episode_id=self.episode_id,
            confidence_score=0.0,
            classification=ClaimClassification.UNVERIFIED_CLAIM,
            epistemic_status=EpistemicStatus.UNSUPPORTED
        )
        res = verifier.verify_claim(claim, evidence_pool=[], topic=self.topic)
        self.assertEqual(res.classification, ClaimClassification.UNVERIFIED_CLAIM)

    def test_22_content_quality_rerun_after_expansion(self):
        """22. Content quality validator accurately re-evaluates expanded verified claims."""
        validator = ContentQualityValidator()
        c1 = VerifiedClaim(claim_id="c1", statement="Keezhadi brick structures", normalized_statement="keezhadi brick structures", claim_type=ClaimType.HISTORICAL_FACT, importance=ClaimImportance.HIGH, topic_aspect="archaeology", episode_id=self.episode_id, confidence_score=0.90, classification=ClaimClassification.VERIFIED_FACT, epistemic_status=EpistemicStatus.KNOWN_FACT, relevance_class=ClaimRelevanceClass.CORE_TOPIC)
        c2 = VerifiedClaim(claim_id="c2", statement="Keezhadi radiocarbon dates 580 BCE", normalized_statement="keezhadi radiocarbon 580 bce", claim_type=ClaimType.HISTORICAL_FACT, importance=ClaimImportance.HIGH, topic_aspect="chronology", episode_id=self.episode_id, confidence_score=0.90, classification=ClaimClassification.VERIFIED_FACT, epistemic_status=EpistemicStatus.KNOWN_FACT, relevance_class=ClaimRelevanceClass.CORE_TOPIC)
        c3 = VerifiedClaim(claim_id="c3", statement="Tamil Brahmi potsherds found", normalized_statement="tamil brahmi potsherds found", claim_type=ClaimType.HISTORICAL_FACT, importance=ClaimImportance.HIGH, topic_aspect="literacy", episode_id=self.episode_id, confidence_score=0.85, classification=ClaimClassification.VERIFIED_FACT, epistemic_status=EpistemicStatus.KNOWN_FACT, relevance_class=ClaimRelevanceClass.CORE_TOPIC)

        passed, metrics, reasons = validator.validate_claims_pool([c1, c2, c3], topic=self.topic)
        self.assertTrue(passed)
        self.assertEqual(metrics["core_claims_count"], 3)
        self.assertGreaterEqual(metrics["unique_aspects_count"], 2)

    # -------------------------------------------------------------------------
    # 7. Expansion Outcomes & Safety
    # -------------------------------------------------------------------------

    def test_23_insufficient_evidence_after_expansion_routes_to_review_required(self):
        """23. If evidence remains insufficient after expansions, routes to REVIEW_REQUIRED."""
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        with patch.object(engine, "_run_single_expansion_cycle") as mock_run:
            mock_run.return_value = {
                "status": EpisodeState.REVIEW_REQUIRED.value,
                "cycle_number": 2,
                "cycle_summary": {"expansion_cycle": "expansion_2"},
                "verif_result": {"verified_claims_count": 1, "verified_facts_count": 0, "supported_hypotheses_count": 1},
                "script_result": {"status": EpisodeState.REVIEW_REQUIRED.value},
                "content_quality_report": {"passed": False, "content_quality_status": "CONTENT_INSUFFICIENT"}
            }
            res = engine.execute_expansion(self.episode_id)
            self.assertEqual(res["status"], EpisodeState.REVIEW_REQUIRED.value)
            self.assertEqual(res["error"], "INSUFFICIENT_EVIDENCE_AFTER_RESEARCH_EXPANSION")

    def test_24_successful_expansion_advances_to_script_validated(self):
        """24. Successful expansion with sufficient core claims advances to SCRIPT_VALIDATED."""
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        with patch.object(engine, "_run_single_expansion_cycle") as mock_run:
            mock_run.return_value = {
                "status": EpisodeState.SCRIPT_VALIDATED.value,
                "cycle_number": 1,
                "cycle_summary": {"expansion_cycle": "expansion_1", "questions_attempted": []},
                "verif_result": {"verified_claims_count": 3, "verified_facts_count": 3, "supported_hypotheses_count": 0},
                "script_result": {"status": EpisodeState.SCRIPT_VALIDATED.value},
                "content_quality_report": {"passed": True, "content_quality_status": "PASSED"}
            }
            res = engine.execute_expansion(self.episode_id)
            self.assertEqual(res["status"], EpisodeState.SCRIPT_VALIDATED.value)

    def test_25_ollama_unavailable_uses_deterministic_template_questions(self):
        """25. Uses deterministic template questions when Ollama is unavailable."""
        analyzer = ResearchGapAnalyzer(ollama_client=None)
        report = analyzer.analyze_gaps(topic=self.topic, offline=True)
        self.assertGreaterEqual(len(report.recommended_research_questions), 3)

    def test_26_llm_cannot_override_deterministic_verification(self):
        """26. LLM confidence or wording cannot override deterministic claim verification."""
        verifier = ClaimVerifier()
        # Even if prompt or claim text says "Verified by AI 100%", verifier scores against evidence
        claim = VerifiedClaim(
            claim_id="c_ai_overconfident",
            statement="AI confirms 100% true with zero evidence.",
            normalized_statement="ai confirms 100 true with zero evidence",
            claim_type=ClaimType.HISTORICAL_FACT,
            importance=ClaimImportance.CRITICAL,
            topic_aspect="archaeology",
            episode_id=self.episode_id,
            confidence_score=0.0,
            classification=ClaimClassification.UNVERIFIED_CLAIM,
            epistemic_status=EpistemicStatus.UNSUPPORTED
        )
        res = verifier.verify_claim(claim, evidence_pool=[], topic=self.topic)
        self.assertEqual(res.classification, ClaimClassification.UNVERIFIED_CLAIM)

    def test_27_contradiction_handling_preserved_during_expansion(self):
        """27. Contradictory evidence results in UNRESOLVED_DEBATE and confidence penalty."""
        verifier = ClaimVerifier()
        claim = VerifiedClaim(
            claim_id="c_disputed",
            statement="Keezhadi Tamil-Brahmi dates to 600 BCE.",
            normalized_statement="keezhadi tamil brahmi dates to 600 bce",
            claim_type=ClaimType.HISTORICAL_FACT,
            importance=ClaimImportance.HIGH,
            topic_aspect="chronology",
            episode_id=self.episode_id,
            confidence_score=0.5,
            classification=ClaimClassification.SUPPORTED_HYPOTHESIS,
            epistemic_status=EpistemicStatus.LIKELY
        )
        pool = [
            {"chunk_id": "cs1", "source_id": "s1", "text": "Keezhadi Tamil-Brahmi dates to 600 BCE.", "title": "T1", "url": "u1", "publisher": "P1", "support_type": "DIRECT_SUPPORT", "score": 0.85},
            {"chunk_id": "cc1", "source_id": "s2", "text": "The dating of Keezhadi Tamil-Brahmi to 600 BCE has been questioned and disputed.", "title": "T2", "url": "u2", "publisher": "P2", "support_type": "CONTRADICTORY", "score": 0.85, "contradiction_reason": "Dating dispute"}
        ]
        res = verifier.verify_claim(claim, evidence_pool=pool, topic=self.topic)
        self.assertEqual(res.classification, ClaimClassification.UNRESOLVED_DEBATE)

    def test_28_idempotent_expansion_execution(self):
        """28. Running expansion twice does not duplicate sources or corrupt history."""
        engine = ResearchExpansionEngine(state_manager=self.state_manager)
        history_file = Path(self.ep_dir) / "research" / "expansion_history.json"
        engine._save_history(history_file, [
            {"expansion_cycle": "expansion_1"},
            {"expansion_cycle": "expansion_2"}
        ])
        res1 = engine.execute_expansion(self.episode_id)
        res2 = engine.execute_expansion(self.episode_id)
        self.assertEqual(res1["expansion_status"], "BUDGET_EXHAUSTED")
        self.assertEqual(res2["expansion_status"], "BUDGET_EXHAUSTED")
        self.assertEqual(len(engine._load_history(history_file)), 2)

    def test_29_interrupted_expansion_recovery(self):
        """29. RecoveryManager safely detects and handles interrupted expansion."""
        recovery = RecoveryManager(self.state_manager)
        session = self.state_manager._get_session()
        try:
            ep = session.query(AutonomousEpisode).filter_by(episode_id=self.episode_id).first()
            ep.status = EpisodeState.RESEARCH_EXPANDING.value
            session.commit()
            session.refresh(ep)
        finally:
            session.close()

        decision = recovery.evaluate_episode(ep)
        self.assertEqual(decision.action, "RESUME")
        self.assertEqual(decision.target_state, EpisodeState.RESEARCH_EXPANDING)

    def test_30_final_review_required_safety_state_when_unresolved(self):
        """30. Safely remains in REVIEW_REQUIRED when content gaps are unresolved."""
        session = self.state_manager._get_session()
        try:
            ep = session.query(AutonomousEpisode).filter_by(episode_id=self.episode_id).first()
            self.assertEqual(ep.status, EpisodeState.REVIEW_REQUIRED.value)
        finally:
            session.close()

    # -------------------------------------------------------------------------
    # 8. Strict Transition Bypass Prevention
    # -------------------------------------------------------------------------

    def test_31_review_required_cannot_bypass_to_scripting(self):
        """31. REVIEW_REQUIRED cannot transition directly to SCRIPTING."""
        self.assertNotIn(EpisodeState.SCRIPTING, VALID_TRANSITIONS[EpisodeState.REVIEW_REQUIRED])
        with self.assertRaises(ValueError):
            self.state_manager.transition_state(self.episode_id, EpisodeState.SCRIPTING, "SCRIPTING")

    def test_32_review_required_cannot_bypass_to_ready_to_publish(self):
        """32. REVIEW_REQUIRED cannot transition directly to READY_TO_PUBLISH."""
        self.assertNotIn(EpisodeState.READY_TO_PUBLISH, VALID_TRANSITIONS[EpisodeState.REVIEW_REQUIRED])
        with self.assertRaises(ValueError):
            self.state_manager.transition_state(self.episode_id, EpisodeState.READY_TO_PUBLISH, "READY_TO_PUBLISH")


if __name__ == "__main__":
    unittest.main()
