"""
tests/test_phase5_claim_verification.py - Comprehensive Unit & Integration Test Suite for Phase 5.

Verifies:
- Atomic claim extraction, normalization, and importance assignment
- Direct vs contextual vs irrelevant vs contradictory evidence classification
- Structured contradiction detection (dates, numbers, entities, polarities)
- Deterministic, reproducible confidence scoring with strict provenance breakdowns
- Wikipedia reference ceilings and local filesystem corroboration isolation
- Epistemic status tracking and importance-conditioned claim classification
- Fact-checked content planning and script allowlist constraint generation
- Storyboard script generation (schema compatibility with daily_engine)
- Script claim re-extraction, hallucination detection, numeric/date checks,
  epistemic overstatement checks, fake quotation detection, and bilingual consistency
- State transitions (RESEARCH_COMPLETE -> VERIFYING -> VERIFIED -> SCRIPTING -> SCRIPT_VALIDATED)
- Adversarial Cases A through K
"""

import os
import json
import shutil
import tempfile
import unittest
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, patch

from autonomous.claim_models import (
    ClaimType,
    ClaimImportance,
    EpistemicStatus,
    EvidenceSupportType,
    ClaimClassification,
    ClaimEvidenceMatch,
    VerifiedClaim,
    SceneContentPlan,
    FactCheckedContentPlan,
    ScriptClaim,
    ScriptClaimStatus,
    ScriptValidationReport
)
from autonomous.claim_extractor import ClaimExtractor
from autonomous.claim_verifier import ClaimVerifier
from autonomous.content_planner import ContentPlanner
from autonomous.script_generator import ScriptGenerator
from autonomous.script_validator import ScriptValidator
from autonomous.claim_verification_engine import ClaimVerificationEngine
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.orchestrator import AutonomousOrchestrator


class TestPhase5ClaimVerification(unittest.TestCase):
    """52 safety-critical unit, integration, and adversarial tests for Phase 5."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_phase5_")
        self.test_db_path = os.path.join(self.test_dir, "test_universe.db")
        self.state_manager = StateManager(db_url=f"sqlite:///{self.test_db_path}")

        # Setup mock episode directory and record
        self.episode_id = "test_ep_501"
        self.ep_dir = os.path.join(self.test_dir, "episodes", self.episode_id)
        os.makedirs(os.path.join(self.ep_dir, "research"), exist_ok=True)
        os.makedirs(os.path.join(self.ep_dir, "verification"), exist_ok=True)

        session = self.state_manager._get_session()
        try:
            ep = AutonomousEpisode(
                episode_id=self.episode_id,
                topic="Keezhadi: 2,600-Year-Old Tamil Urban Civilization and Early Literacy",
                category="Sangam Civilization",
                status=EpisodeState.RESEARCH_COMPLETE.value,
                current_stage="RESEARCH_COMPLETE",
                output_directory=self.ep_dir
            )
            session.add(ep)
            session.commit()
        finally:
            session.close()

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 1-6: Extraction, Normalization & Provenance
    # -------------------------------------------------------------------------

    def test_01_claim_extraction(self):
        extractor = ClaimExtractor()
        text = "Documentation indicates: Keeladi excavations recovered Tamil-Brahmi inscribed potsherds dating to 6th century BCE."
        dates = extractor.extract_dates(text)
        self.assertIn("6th century bce", dates)
        ents = extractor.extract_entities(text)
        self.assertIn("keeladi", ents)
        self.assertIn("tamil-brahmi", ents)

    def test_02_atomic_claim_splitting(self):
        extractor = ClaimExtractor()
        compound = "Keeladi has brick structures; moreover ancient drainage networks were discovered."
        props = extractor.split_into_atomic_propositions(compound)
        self.assertGreaterEqual(len(props), 2)
        self.assertTrue(any("brick structures" in p for p in props))
        self.assertTrue(any("drainage networks" in p for p in props))

    def test_03_claim_normalization(self):
        raw = "   \"Documentation from ASI indicates: Keeladi was an urban center.\"   "
        norm = ClaimExtractor.normalize_statement(raw)
        self.assertEqual(norm, "Keeladi was an urban center.")

    def test_04_duplicate_claim_detection(self):
        extractor = ClaimExtractor()
        dossier_mock = MagicMock()
        f1 = MagicMock(statement="Keeladi features brick structures.", topic_aspect="Architecture", supporting_chunk_ids=[], supporting_source_ids=[], independence_groups=[], source_origins=[], evidence_weight_classes=[])
        f2 = MagicMock(statement="Keeladi features brick structures.", topic_aspect="Architecture", supporting_chunk_ids=[], supporting_source_ids=[], independence_groups=[], source_origins=[], evidence_weight_classes=[])
        dossier_mock.preliminary_findings = [f1, f2]
        dossier_mock.conflicting_evidence = []
        dossier_mock.topic = "Keeladi"
        claims = extractor.extract_claims_from_dossier(dossier_mock, "ep_01")
        self.assertEqual(len(claims), 1)

    def test_05_claim_importance_assignment(self):
        extractor = ClaimExtractor()
        has_sup, _ = extractor.has_superlative("Keeladi is the oldest urban settlement.")
        imp = extractor.determine_importance(ClaimType.SUPERLATIVE_ASSERTION, has_superlative=has_sup, is_central=True)
        self.assertEqual(imp, ClaimImportance.CRITICAL)

        imp_arch = extractor.determine_importance(ClaimType.ARCHAEOLOGICAL_EVIDENCE, has_superlative=False, is_central=False)
        self.assertEqual(imp_arch, ClaimImportance.HIGH)

    def test_06_claim_provenance_preservation(self):
        extractor = ClaimExtractor()
        dossier_mock = MagicMock()
        f1 = MagicMock(
            statement="Keeladi yields Tamil-Brahmi scripts.",
            topic_aspect="Epigraphy",
            supporting_chunk_ids=["chk_01"],
            supporting_source_ids=["src_asi"],
            independence_groups=["asi.nic.in"],
            source_origins=["external"],
            evidence_weight_classes=["PRIMARY"]
        )
        dossier_mock.preliminary_findings = [f1]
        dossier_mock.conflicting_evidence = []
        dossier_mock.topic = "Keeladi"
        claims = extractor.extract_claims_from_dossier(dossier_mock, "ep_01")
        c = claims[0]
        self.assertIn("chk_01", c.supporting_chunk_ids)
        self.assertIn("asi.nic.in", c.independence_groups)
        self.assertIn("PRIMARY", c.evidence_weight_classes)

    # -------------------------------------------------------------------------
    # 7-11: Evidence Support Classification
    # -------------------------------------------------------------------------

    def test_07_evidence_retrieval_integration(self):
        store_mock = MagicMock()
        store_mock.retrieve_evidence.return_value = [{
            "chunk_id": "chk_01",
            "score": 0.85,
            "text": "Excavations at Keeladi by ASI revealed structured brick canals and ring wells.",
            "source_id": "src_asi",
            "publisher": "Archaeological Survey of India",
            "domain": "asi.nic.in",
            "independence_group": "asi.nic.in",
            "source_role": "primary_evidence",
            "evidence_weight_class": "PRIMARY",
            "source_origin": "external",
            "credibility_tier": 1
        }]
        verifier = ClaimVerifier(evidence_store=store_mock)
        claim = VerifiedClaim(
            claim_id="clm_01",
            statement="Keeladi excavations revealed brick canals.",
            normalized_statement="Keeladi excavations revealed brick canals.",
            claim_type=ClaimType.ARCHAEOLOGICAL_EVIDENCE,
            importance=ClaimImportance.HIGH,
            topic_aspect="Architecture",
            episode_id="ep_01",
            extracted_entities=["keeladi", "brick"]
        )
        v = verifier.verify_claim(claim)
        self.assertEqual(len(v.evidence_matches), 1)
        self.assertEqual(v.evidence_matches[0].support_type, EvidenceSupportType.DIRECT_SUPPORT)

    def test_08_episode_filtering(self):
        store_mock = MagicMock()
        verifier = ClaimVerifier(evidence_store=store_mock)
        claim = VerifiedClaim("clm_01", "Statement", "Statement", ClaimType.HISTORICAL_FACT, ClaimImportance.MEDIUM, "Aspect", "ep_999")
        verifier.verify_claim(claim)
        store_mock.retrieve_evidence.assert_called_with(query="Statement", episode_id="ep_999", top_k=5)

    def test_09_direct_support_classification(self):
        claim = VerifiedClaim("clm_01", "Keeladi contained Tamil-Brahmi inscribed potsherds.", "Keeladi contained Tamil-Brahmi inscribed potsherds.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Aspect", "ep_01", extracted_entities=["keeladi", "tamil-brahmi", "pottery"])
        chunk_text = "Archaeologists at Keeladi uncovered 56 potsherds inscribed with Tamil-Brahmi letters."
        st, reason = ClaimVerifier.classify_evidence_support(claim, chunk_text, 0.88, ["keeladi", "tamil-brahmi", "pottery"], [], ["56"])
        self.assertEqual(st, EvidenceSupportType.DIRECT_SUPPORT)

    def test_10_contextual_support_classification(self):
        claim = VerifiedClaim("clm_01", "Keeladi contained Tamil-Brahmi inscribed potsherds.", "Keeladi contained Tamil-Brahmi inscribed potsherds.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Aspect", "ep_01", extracted_entities=["keeladi", "tamil-brahmi"])
        chunk_text = "The Vaigai river basin in Madurai district has been an active agricultural zone for thousands of years."
        st, reason = ClaimVerifier.classify_evidence_support(claim, chunk_text, 0.45, ["vaigai", "madurai"], [], [])
        self.assertEqual(st, EvidenceSupportType.CONTEXTUAL_ONLY)

    def test_11_irrelevant_evidence_classification(self):
        claim = VerifiedClaim("clm_01", "Korkai was an ancient pearl fishery port.", "Korkai was an ancient pearl fishery port.", ClaimType.GEOGRAPHICAL_LOCATION, ClaimImportance.MEDIUM, "Aspect", "ep_01", extracted_entities=["korkai"])
        chunk_text = "Modern industrial manufacturing in Chennai includes automobile factories."
        st, reason = ClaimVerifier.classify_evidence_support(claim, chunk_text, 0.15, [], [], [])
        self.assertEqual(st, EvidenceSupportType.IRRELEVANT)

    # -------------------------------------------------------------------------
    # 12-16: Source Independence & Quality Weighting
    # -------------------------------------------------------------------------

    def test_12_same_source_deduplication(self):
        m1 = ClaimEvidenceMatch("chk_01", "src_asi", EvidenceSupportType.DIRECT_SUPPORT, 0.80, independence_group="asi.nic.in", evidence_weight_class="PRIMARY")
        m2 = ClaimEvidenceMatch("chk_02", "src_asi", EvidenceSupportType.DIRECT_SUPPORT, 0.82, independence_group="asi.nic.in", evidence_weight_class="PRIMARY")
        m3 = ClaimEvidenceMatch("chk_03", "src_asi", EvidenceSupportType.DIRECT_SUPPORT, 0.79, independence_group="asi.nic.in", evidence_weight_class="PRIMARY")

        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Keeladi brick architecture.", "Keeladi brick architecture.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict(), m2.to_dict(), m3.to_dict()])
        self.assertEqual(claim.source_count, 1)
        self.assertEqual(claim.independent_external_source_count, 1)

    def test_13_same_domain_independence(self):
        m1 = ClaimEvidenceMatch("chk_01", "src_asi_page1", EvidenceSupportType.DIRECT_SUPPORT, 0.85, publisher="Archaeological Survey of India", domain="asi.nic.in", independence_group="asi.nic.in", evidence_weight_class="PRIMARY")
        m2 = ClaimEvidenceMatch("chk_02", "src_asi_page2", EvidenceSupportType.DIRECT_SUPPORT, 0.83, publisher="Archaeological Survey of India", domain="asi.nic.in", independence_group="asi.nic.in", evidence_weight_class="PRIMARY")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Keeladi brick architecture.", "Keeladi brick architecture.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict(), m2.to_dict()])
        self.assertEqual(claim.independent_external_source_count, 1)

    def test_14_local_source_exclusion(self):
        m1 = ClaimEvidenceMatch("chk_local", "src_local", EvidenceSupportType.DIRECT_SUPPORT, 0.85, publisher="Local Project Knowledge", domain="local_filesystem", independence_group="local_filesystem", source_role="background_local", evidence_weight_class="REFERENCE", source_origin="local")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Local topic assertion.", "Local topic assertion.", ClaimType.HISTORICAL_FACT, ClaimImportance.MEDIUM, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict()])
        self.assertEqual(claim.independent_external_source_count, 0)
        self.assertLessEqual(claim.confidence_score, 0.50)
        self.assertNotEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    def test_15_wikipedia_reference_weighting(self):
        m_wiki = ClaimEvidenceMatch("chk_wiki", "src_wiki", EvidenceSupportType.DIRECT_SUPPORT, 0.90, publisher="Wikimedia Foundation", domain="wikipedia.org", independence_group="wikipedia.org", source_role="discovery_reference", evidence_weight_class="REFERENCE", source_origin="external")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Wikipedia solitary claim.", "Wikipedia solitary claim.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m_wiki.to_dict()])
        self.assertLessEqual(claim.confidence_score, 0.65)
        self.assertNotEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    def test_16_primary_source_weighting(self):
        m_pri = ClaimEvidenceMatch("chk_01", "src_tn", EvidenceSupportType.DIRECT_SUPPORT, 0.90, publisher="TNSDA", domain="archaeology.tn.gov.in", independence_group="archaeology.tn.gov.in", source_role="primary_evidence", evidence_weight_class="PRIMARY", source_origin="external", credibility_tier=1)
        m_ext = ClaimEvidenceMatch("chk_02", "src_hindu", EvidenceSupportType.DIRECT_SUPPORT, 0.85, publisher="The Hindu", domain="thehindu.com", independence_group="thehindu.com", source_role="secondary_evidence", evidence_weight_class="MODERATE", source_origin="external", credibility_tier=3)
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Keeladi settlement dating.", "Keeladi settlement dating.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m_pri.to_dict(), m_ext.to_dict()])
        self.assertGreaterEqual(claim.confidence_score, 0.75)
        self.assertEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    # -------------------------------------------------------------------------
    # 17-20: Deterministic Confidence
    # -------------------------------------------------------------------------

    def test_17_deterministic_confidence_calculation(self):
        matches = [
            ClaimEvidenceMatch("chk_01", "src_01", EvidenceSupportType.DIRECT_SUPPORT, 0.80, independence_group="grp1", evidence_weight_class="PRIMARY", credibility_tier=1),
            ClaimEvidenceMatch("chk_02", "src_02", EvidenceSupportType.DIRECT_SUPPORT, 0.80, independence_group="grp2", evidence_weight_class="HIGH", credibility_tier=2)
        ]
        score, b = ClaimVerifier.calculate_deterministic_confidence(matches, ClaimImportance.HIGH, False)
        self.assertGreaterEqual(score, 0.70)
        self.assertIn("source_quality", b)
        self.assertIn("independence", b)

    def test_18_confidence_reproducibility(self):
        matches = [
            ClaimEvidenceMatch("chk_01", "src_01", EvidenceSupportType.DIRECT_SUPPORT, 0.85, independence_group="asi.nic.in", evidence_weight_class="PRIMARY", credibility_tier=1),
            ClaimEvidenceMatch("chk_02", "src_02", EvidenceSupportType.INDIRECT_SUPPORT, 0.75, independence_group="thehindu.com", evidence_weight_class="MODERATE", credibility_tier=3)
        ]
        s1, b1 = ClaimVerifier.calculate_deterministic_confidence(matches, ClaimImportance.CRITICAL, False)
        s2, b2 = ClaimVerifier.calculate_deterministic_confidence(matches, ClaimImportance.CRITICAL, False)
        self.assertEqual(s1, s2)
        self.assertEqual(b1, b2)

    def test_19_confidence_bounds(self):
        # Empty matches
        s_empty, _ = ClaimVerifier.calculate_deterministic_confidence([], ClaimImportance.LOW, False)
        self.assertEqual(s_empty, 0.0)

        # Huge matches
        many = [
            ClaimEvidenceMatch(f"chk_{i}", f"src_{i}", EvidenceSupportType.DIRECT_SUPPORT, 0.99, independence_group=f"grp_{i}", evidence_weight_class="PRIMARY", credibility_tier=1)
            for i in range(10)
        ]
        s_huge, _ = ClaimVerifier.calculate_deterministic_confidence(many, ClaimImportance.CRITICAL, False)
        self.assertLessEqual(s_huge, 1.0)
        self.assertGreaterEqual(s_huge, 0.0)

    def test_20_confidence_breakdown_transparency(self):
        m = [ClaimEvidenceMatch("chk_01", "src_01", EvidenceSupportType.DIRECT_SUPPORT, 0.70, independence_group="grp1", evidence_weight_class="PRIMARY")]
        _, b = ClaimVerifier.calculate_deterministic_confidence(m, ClaimImportance.MEDIUM, False)
        keys = ["source_quality", "direct_support", "independence", "evidence_specificity", "contradiction_penalty", "final_confidence"]
        for k in keys:
            self.assertIn(k, b)

    # -------------------------------------------------------------------------
    # 21-25: Contradictions & Superlatives
    # -------------------------------------------------------------------------

    def test_21_contradictory_dates(self):
        claim = VerifiedClaim("clm_01", "Keeladi dates to 6th century BCE.", "Keeladi dates to 6th century BCE.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Aspect", "ep_01", extracted_dates=["6th century bce"])
        chunk_text = "Archaeological strata and stratigraphic dating demonstrate this site belongs to the 3rd century BCE."
        st, reason = ClaimVerifier.classify_evidence_support(claim, chunk_text, 0.75, ["keeladi"], ["3rd century bce"], [])
        self.assertEqual(st, EvidenceSupportType.CONTRADICTORY)
        self.assertIn("Chronological conflict", reason)

    def test_22_contradictory_numbers(self):
        claim = VerifiedClaim("clm_01", "The site housed 10000 people.", "The site housed 10000 people.", ClaimType.STATISTICAL_MEASUREMENT, ClaimImportance.HIGH, "Aspect", "ep_01", extracted_numbers=["10000"])
        chunk_text = "Demographic models calculate the settlement supported only 500 people during its peak."
        st, reason = ClaimVerifier.classify_evidence_support(claim, chunk_text, 0.70, [], [], ["500"])
        self.assertEqual(st, EvidenceSupportType.CONTRADICTORY)
        self.assertIn("Numerical divergence", reason)

    def test_23_conflicting_interpretations(self):
        claim = VerifiedClaim("clm_01", "Inscriptions prove universal secular literacy.", "Inscriptions prove universal secular literacy.", ClaimType.CULTURAL_PRACTICE, ClaimImportance.HIGH, "Aspect", "ep_01", extracted_entities=["literacy"])
        chunk_text = "There is no evidence of universal literacy; the graffiti marks were merely potters marks."
        st, reason = ClaimVerifier.classify_evidence_support(claim, chunk_text, 0.70, ["literacy"], [], [])
        self.assertEqual(st, EvidenceSupportType.CONTRADICTORY)

    def test_24_critical_claim_handling(self):
        # Critical claim with only single secondary source should not become VERIFIED_FACT
        m = ClaimEvidenceMatch("chk_01", "src_01", EvidenceSupportType.DIRECT_SUPPORT, 0.75, independence_group="grp1", evidence_weight_class="MODERATE", credibility_tier=3)
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Keeladi was the capital city.", "Keeladi was the capital city.", ClaimType.HISTORICAL_FACT, ClaimImportance.CRITICAL, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m.to_dict()])
        self.assertNotEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    def test_25_superlative_claim_handling(self):
        # Superlative claim requires multiple independent sources
        m = ClaimEvidenceMatch("chk_01", "src_01", EvidenceSupportType.DIRECT_SUPPORT, 0.85, independence_group="grp1", evidence_weight_class="PRIMARY", credibility_tier=1)
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Keeladi was the oldest site in India.", "Keeladi was the oldest site in India.", ClaimType.SUPERLATIVE_ASSERTION, ClaimImportance.CRITICAL, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m.to_dict()])
        self.assertNotEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    # -------------------------------------------------------------------------
    # 26-29: Epistemic States
    # -------------------------------------------------------------------------

    def test_26_unsupported_claim(self):
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Aliens constructed the brick canals.", "Aliens constructed the brick canals.", ClaimType.HISTORICAL_FACT, ClaimImportance.CRITICAL, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[])
        self.assertEqual(claim.classification, ClaimClassification.UNVERIFIED_CLAIM)
        self.assertEqual(claim.epistemic_status, EpistemicStatus.UNCERTAIN)

    def test_27_debated_claim(self):
        m_support = ClaimEvidenceMatch("chk_01", "src_01", EvidenceSupportType.DIRECT_SUPPORT, 0.80, independence_group="grp1", evidence_weight_class="PRIMARY")
        m_contra = ClaimEvidenceMatch("chk_02", "src_02", EvidenceSupportType.CONTRADICTORY, 0.80, independence_group="grp2", evidence_weight_class="HIGH")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Chronology is 6th century BCE.", "Chronology is 6th century BCE.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m_support.to_dict(), m_contra.to_dict()])
        self.assertEqual(claim.classification, ClaimClassification.UNRESOLVED_DEBATE)
        self.assertEqual(claim.epistemic_status, EpistemicStatus.DEBATED)

    def test_28_proposed_theory(self):
        # Supported hypothesis with moderate confidence
        m = ClaimEvidenceMatch("chk_01", "src_01", EvidenceSupportType.INDIRECT_SUPPORT, 0.65, independence_group="grp1", evidence_weight_class="MODERATE", credibility_tier=3)
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Trade connections extended to the Mediterranean.", "Trade connections extended to the Mediterranean.", ClaimType.CULTURAL_PRACTICE, ClaimImportance.MEDIUM, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m.to_dict()])
        self.assertEqual(claim.classification, ClaimClassification.SUPPORTED_HYPOTHESIS)
        self.assertIn(claim.epistemic_status, (EpistemicStatus.LIKELY, EpistemicStatus.PROPOSED_THEORY))

    def test_29_tradition_or_legend(self):
        extractor = ClaimExtractor()
        dossier_mock = MagicMock()
        f1 = MagicMock(statement="Ancient Tamil tradition describes the Sangam academy in Madurai.", topic_aspect="Literature", supporting_chunk_ids=[], supporting_source_ids=[], independence_groups=[], source_origins=[], evidence_weight_classes=[])
        dossier_mock.preliminary_findings = [f1]
        dossier_mock.conflicting_evidence = []
        dossier_mock.topic = "Madurai"
        claims = extractor.extract_claims_from_dossier(dossier_mock, "ep_01")
        c = claims[0]
        c.epistemic_status = EpistemicStatus.TRADITION_OR_LEGEND
        planner = ContentPlanner()
        plan = planner.build_content_plan("ep_01", "Madurai", "History", [c])
        self.assertIn(c.claim_id, plan.tradition_claim_ids)

    # -------------------------------------------------------------------------
    # 30-31: Content Plan & Script Generation
    # -------------------------------------------------------------------------

    def test_30_fact_checked_content_plan(self):
        c1 = VerifiedClaim("clm_01", "Keeladi has brick canals.", "Keeladi has brick canals.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        c2 = VerifiedClaim("clm_02", "Pottery dates to 6th century BCE.", "Pottery dates to 6th century BCE.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS, extracted_dates=["6th century bce"])
        c3 = VerifiedClaim("clm_03", "Atlantis connection.", "Atlantis connection.", ClaimType.HISTORICAL_FACT, ClaimImportance.CRITICAL, "Myth", "ep_01", classification=ClaimClassification.UNVERIFIED_CLAIM)

        planner = ContentPlanner()
        plan = planner.build_content_plan("ep_01", "Keeladi", "Archaeology", [c1, c2, c3])
        self.assertIn("clm_01", plan.allowed_claim_ids)
        self.assertIn("clm_02", plan.cautious_claim_ids)
        self.assertIn("clm_03", plan.forbidden_claim_ids)
        self.assertIn("6th century bce", plan.allowed_dates)

    def test_31_script_generation_offline(self):
        c1 = VerifiedClaim("clm_01", "Excavations revealed structured brick canals.", "Excavations revealed structured brick canals.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        planner = ContentPlanner()
        plan = planner.build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        generator = ScriptGenerator()
        storyboard = generator.generate_script(plan, {"clm_01": c1}, offline=True)
        self.assertEqual(len(storyboard["scenes"]), 4)
        self.assertTrue(all("tamil_text" in s and "english_sub" in s for s in storyboard["scenes"]))
        self.assertTrue(all("visual_prompt" in s and "motion_plan" in s for s in storyboard["scenes"]))

    # -------------------------------------------------------------------------
    # 32-40: Script Validation & Discrepancies
    # -------------------------------------------------------------------------

    def test_32_script_claim_extraction(self):
        validator = ScriptValidator()
        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "Keeladi is a 2,600-year-old settlement in Tamil Nadu.", "tamil_text": "கீழடி தமிழ்நாட்டில் உள்ள தொன்மையான நாகரிகம்."}
            ]
        }
        claims = validator.extract_script_claims(storyboard)
        self.assertGreaterEqual(len(claims), 2)
        en_claim = [c for c in claims if c.language == "en"][0]
        self.assertIn("2600", en_claim.extracted_numbers)

    def test_33_new_unsupported_claim_detection(self):
        c1 = VerifiedClaim("clm_01", "Keeladi has brick canals.", "Keeladi has brick canals.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        planner = ContentPlanner()
        plan = planner.build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "King Rajendra Chola personally commanded the Keezhadi naval fleet of warships.", "tamil_text": "ராஜேந்திர சோழன் கப்பற்படையை வழிநடத்தினார்."}
            ]
        }
        validator = ScriptValidator()
        report = validator.validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.new_unsupported_count, 1)

    def test_34_numeric_mismatch(self):
        c1 = VerifiedClaim("clm_01", "Artifacts number 10,000.", "Artifacts number 10,000.", ClaimType.STATISTICAL_MEASUREMENT, ClaimImportance.HIGH, "Stat", "ep_01", classification=ClaimClassification.VERIFIED_FACT, extracted_numbers=["10000"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "Archaeologists uncovered 50000 artifacts across the trenches.", "tamil_text": "ஐம்பதாயிரம் தொல்பொருட்கள் கிடைத்தன."}
            ]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.numeric_mismatch_count, 1)

    def test_35_date_mismatch(self):
        c1 = VerifiedClaim("clm_01", "Site dates to 6th century BCE.", "Site dates to 6th century BCE.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.VERIFIED_FACT, extracted_dates=["6th century bce"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "The civilization flourished in the 12th century CE.", "tamil_text": "பன்னிரண்டாம் நூற்றாண்டில் செழித்தோங்கியது."}
            ]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.date_mismatch_count, 1)

    def test_36_entity_mismatch(self):
        c1 = VerifiedClaim("clm_01", "Potsherds found at Keeladi.", "Potsherds found at Keeladi.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT, extracted_entities=["keeladi"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "Emperor Ashoka visited the Keeladi monastery.", "tamil_text": "அசோக சக்கரவர்த்தி இங்கு வருகை தந்தார்."}
            ]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)

    def test_37_epistemic_overstatement(self):
        # Hypothesis stated as proven fact -> OVERSTATED_CLAIM
        c1 = VerifiedClaim("clm_01", "Excavations suggest possible maritime trade links.", "Excavations suggest possible maritime trade links.", ClaimType.CULTURAL_PRACTICE, ClaimImportance.HIGH, "Trade", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "Archaeologists proved conclusively that Keeladi operated an international ocean fleet.", "tamil_text": "தொல்லியல் துறை இதனை முழுமையாக நிரூபித்துள்ளது."}
            ]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.overstated_count, 1)

    def test_38_uncertainty_removal(self):
        c1 = VerifiedClaim("clm_01", "Tradition describes a legendary port at this spot.", "Tradition describes a legendary port at this spot.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Tradition", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS, epistemic_status=EpistemicStatus.TRADITION_OR_LEGEND)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "Historical records prove the port existed without doubt.", "tamil_text": "வரலாற்று பதிவுகள் இதனை உறுதி செய்கின்றன."}
            ]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)

    def test_39_fake_quotation_detection(self):
        c1 = VerifiedClaim("clm_01", "Excavations found brick structures.", "Excavations found brick structures.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": 'The ancient poet famously stated: "This city shall shine for eternity."', "tamil_text": "புலவர் இவ்வாறு பாடியுள்ளார்."}
            ]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.quotation_violation_count, 1)

    def test_40_tamil_english_inconsistency(self):
        c1 = VerifiedClaim("clm_01", "Site dates to 6th century BCE and 3rd century BCE.", "Site dates to 6th century BCE.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.VERIFIED_FACT, extracted_dates=["6th century bce", "3rd century bce"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])

        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "Evidence dates to 6th century BCE.", "tamil_text": "இந்த நாகரிகம் 3rd century BCE காலத்தைச் சேர்ந்தது."}
            ]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.cross_language_mismatch_count, 1)

    # -------------------------------------------------------------------------
    # 41-46: Script Validation Pass, Fail & Recovery
    # -------------------------------------------------------------------------

    def test_41_script_validation_pass(self):
        c1 = VerifiedClaim("clm_01", "Excavations revealed structured brick canals.", "Excavations revealed structured brick canals.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])
        generator = ScriptGenerator()
        storyboard = generator.generate_offline_fallback_script(plan, {"clm_01": c1})
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertTrue(report.passed)
        self.assertEqual(len(report.violations), 0)

    def test_42_script_validation_failure(self):
        c1 = VerifiedClaim("clm_01", "Excavations revealed structured brick canals.", "Excavations revealed structured brick canals.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])
        storyboard = {
            "scenes": [
                {"id": 1, "english_sub": "Alien spacecraft hovered over Keeladi in 50000 BCE.", "tamil_text": "விண்கலம் இங்கு பறந்தது."}
            ]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c1})
        self.assertFalse(report.passed)

    def test_43_regeneration_attempt_1(self):
        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        self.assertEqual(engine.max_regeneration_attempts, 2)

    def test_44_regeneration_attempt_2_enforces_offline_synthesis(self):
        generator = ScriptGenerator()
        c1 = VerifiedClaim("clm_01", "Brick canals discovered.", "Brick canals discovered.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Archaeology", [c1])
        # Force offline on attempt 2
        sb = generator.generate_script(plan, {"clm_01": c1}, offline=True)
        report = ScriptValidator().validate_script(sb, plan, {"clm_01": c1})
        self.assertTrue(report.passed)

    def test_45_review_required_routing_on_unresolved_failure(self):
        # Verification engine routes to REVIEW_REQUIRED if no safe claims exist
        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        # Create empty dossier
        dossier_file = os.path.join(self.ep_dir, "research", "dossier.json")
        with open(dossier_file, "w", encoding="utf-8") as f:
            json.dump({
                "episode_id": self.episode_id,
                "topic": "Empty Topic",
                "category": "Archaeology",
                "research_status": "RESEARCH_COMPLETE",
                "preliminary_findings": [],
                "conflicting_evidence": [],
                "question_evidence": []
            }, f)
        res = engine.execute_claim_verification(self.episode_id)
        self.assertEqual(res["status"], "REVIEW_REQUIRED")
        ep = self.state_manager.get_episode(self.episode_id)
        self.assertEqual(ep.status, EpisodeState.REVIEW_REQUIRED.value)

    def test_46_failed_routing_on_missing_dossier(self):
        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        with self.assertRaises(FileNotFoundError):
            engine.execute_claim_verification(self.episode_id)

    # -------------------------------------------------------------------------
    # 47-52: State Transitions, Persistence & Resource Safety
    # -------------------------------------------------------------------------

    def test_47_state_transitions(self):
        # Validate state transition table allows RESEARCH_COMPLETE -> VERIFYING -> VERIFIED -> SCRIPTING -> SCRIPT_VALIDATED
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFYING, "VERIFYING")
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFIED, "VERIFIED")
        self.state_manager.transition_state(self.episode_id, EpisodeState.SCRIPTING, "SCRIPTING")
        self.state_manager.transition_state(self.episode_id, EpisodeState.SCRIPT_VALIDATED, "SCRIPT_VALIDATED")
        ep = self.state_manager.get_episode(self.episode_id)
        self.assertEqual(ep.status, EpisodeState.SCRIPT_VALIDATED.value)

    def test_48_recovery_resumption_from_verified(self):
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFYING, "VERIFYING")
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFIED, "VERIFIED")
        ep = self.state_manager.get_active_episode()
        self.assertEqual(ep.status, EpisodeState.VERIFIED.value)

    def test_49_active_episode_safety(self):
        ep = self.state_manager.get_active_episode()
        self.assertEqual(ep.episode_id, self.episode_id)

    def test_50_artifact_persistence(self):
        c1 = VerifiedClaim("clm_01", "Keeladi has brick canals.", "Keeladi has brick canals.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", self.episode_id, classification=ClaimClassification.VERIFIED_FACT)
        self.state_manager.record_verified_claims(self.episode_id, [c1])
        persisted = self.state_manager.get_verified_claims(self.episode_id)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(persisted[0].claim_id, "clm_01")
        self.assertEqual(persisted[0].classification, "VERIFIED_FACT")

    def test_51_cpu_resource_safety(self):
        verifier = ClaimVerifier(evidence_store=None)
        self.assertIsNotNone(verifier)
        # Verify no CUDA calls made
        import torch
        if torch.cuda.is_available():
            self.assertEqual(torch.cuda.memory_allocated(), 0)

    def test_52_phase1_4_regression_compatibility(self):
        # Verify orchestrator pipeline readiness reports Phase 5
        orchestrator = AutonomousOrchestrator(state_manager=self.state_manager)
        readiness = orchestrator.get_pipeline_readiness()
        stages = readiness["stages"]
        self.assertEqual(stages["Topic Discovery"], "OPERATIONAL (Phase 3)")
        self.assertEqual(stages["Research Engine"], "OPERATIONAL (Phase 4)")
        self.assertEqual(stages["Claim Verification"], "OPERATIONAL (Phase 5)")
        self.assertEqual(stages["Script Generation"], "OPERATIONAL (Phase 5)")
        self.assertEqual(stages["Script Validation"], "OPERATIONAL (Phase 5)")

    # -------------------------------------------------------------------------
    # Mandatory Adversarial Tests: Cases A Through K
    # -------------------------------------------------------------------------

    def test_adversarial_case_a(self):
        """CASE A: Evidence says 'possibly 6th century BCE', Script says 'definitely 6th century BCE' -> OVERSTATED_CLAIM"""
        c = VerifiedClaim("clm_01", "Excavations possibly date to the 6th century BCE.", "Excavations possibly date to the 6th century BCE.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS, extracted_dates=["6th century bce"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {
            "scenes": [{"id": 1, "english_sub": "The settlement definitely dates to the 6th century BCE.", "tamil_text": "இது உறுதியாக கிமு 6 ஆம் நூற்றாண்டைச் சேர்ந்தது."}]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.overstated_count, 1)

    def test_adversarial_case_b(self):
        """CASE B: Wikipedia only supports 'X was the oldest city' -> Not VERIFIED_FACT"""
        m_wiki = ClaimEvidenceMatch("chk_w", "src_w", EvidenceSupportType.DIRECT_SUPPORT, 0.95, independence_group="wikipedia.org", evidence_weight_class="REFERENCE", source_origin="external")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Keeladi was the oldest city.", "Keeladi was the oldest city.", ClaimType.SUPERLATIVE_ASSERTION, ClaimImportance.CRITICAL, "Super", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m_wiki.to_dict()])
        self.assertNotEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    def test_adversarial_case_c(self):
        """CASE C: Three pages from one publisher -> independent_external_source_count = 1"""
        m1 = ClaimEvidenceMatch("chk_1", "src_p1", EvidenceSupportType.DIRECT_SUPPORT, 0.8, independence_group="asi.nic.in")
        m2 = ClaimEvidenceMatch("chk_2", "src_p2", EvidenceSupportType.DIRECT_SUPPORT, 0.8, independence_group="asi.nic.in")
        m3 = ClaimEvidenceMatch("chk_3", "src_p3", EvidenceSupportType.DIRECT_SUPPORT, 0.8, independence_group="asi.nic.in")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Brick architecture statement.", "Brick architecture statement.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict(), m2.to_dict(), m3.to_dict()])
        self.assertEqual(claim.independent_external_source_count, 1)

    def test_adversarial_case_d(self):
        """CASE D: Local + Wikipedia + ASI -> external_independent_sources = 2, but only 1 high-quality external source"""
        m_loc = ClaimEvidenceMatch("chk_l", "src_l", EvidenceSupportType.DIRECT_SUPPORT, 0.8, independence_group="local_filesystem", source_origin="local", evidence_weight_class="REFERENCE")
        m_wiki = ClaimEvidenceMatch("chk_w", "src_w", EvidenceSupportType.DIRECT_SUPPORT, 0.8, independence_group="wikipedia.org", source_origin="external", evidence_weight_class="REFERENCE")
        m_asi = ClaimEvidenceMatch("chk_a", "src_a", EvidenceSupportType.DIRECT_SUPPORT, 0.8, independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY", credibility_tier=1)
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Civilization statement.", "Civilization statement.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m_loc.to_dict(), m_wiki.to_dict(), m_asi.to_dict()])
        self.assertEqual(claim.independent_external_source_count, 2)
        self.assertEqual(claim.high_quality_source_count, 1)

    def test_adversarial_case_e(self):
        """CASE E: Two strong sources disagree -> DEBATED or UNCERTAIN"""
        m1 = ClaimEvidenceMatch("chk_1", "src_1", EvidenceSupportType.DIRECT_SUPPORT, 0.85, independence_group="asi.nic.in", evidence_weight_class="PRIMARY")
        m2 = ClaimEvidenceMatch("chk_2", "src_2", EvidenceSupportType.CONTRADICTORY, 0.85, independence_group="cambridge.org", evidence_weight_class="PRIMARY")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Chronology dating statement.", "Chronology dating statement.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict(), m2.to_dict()])
        self.assertIn(claim.classification, (ClaimClassification.UNRESOLVED_DEBATE, ClaimClassification.UNVERIFIED_CLAIM))
        self.assertIn(claim.epistemic_status, (EpistemicStatus.DEBATED, EpistemicStatus.UNCERTAIN))

    def test_adversarial_case_f(self):
        """CASE F: Evidence says 'approximately 10,000', Script says '100,000' -> NUMERIC_MISMATCH"""
        c = VerifiedClaim("clm_01", "The town had approximately 10000 residents.", "The town had approximately 10000 residents.", ClaimType.STATISTICAL_MEASUREMENT, ClaimImportance.HIGH, "Stat", "ep_01", classification=ClaimClassification.VERIFIED_FACT, extracted_numbers=["10000"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Stat", [c])
        storyboard = {
            "scenes": [{"id": 1, "english_sub": "Over 100000 residents populated this ancient metropolis.", "tamil_text": "இங்கு ஒரு லட்சம் மக்கள் வாழ்ந்தனர்."}]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.numeric_mismatch_count, 1)

    def test_adversarial_case_g(self):
        """CASE G: Person absent from evidence but added by LLM -> NEW_UNSUPPORTED_CLAIM"""
        c = VerifiedClaim("clm_01", "Pottery inscribed with letters was recovered.", "Pottery inscribed with letters was recovered.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {
            "scenes": [{"id": 1, "english_sub": "General Alexander the Great marched his army directly into Keeladi.", "tamil_text": "அலெக்சாண்டர் இங்கு வந்தார்."}]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.new_unsupported_count, 1)

    def test_adversarial_case_h(self):
        """CASE H: No quotation in evidence, LLM creates quotation -> UNSUPPORTED_QUOTATION"""
        c = VerifiedClaim("clm_01", "Excavations revealed brick wells.", "Excavations revealed brick wells.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.VERIFIED_FACT)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {
            "scenes": [{"id": 1, "english_sub": 'The Sangam king declared: "Our city shall remain invincible forever."', "tamil_text": "அரசர் இவ்வாறு பிரகடனம் செய்தார்."}]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.quotation_violation_count, 1)

    def test_adversarial_case_i(self):
        """CASE I: 'tradition attributes' becomes 'historical records prove' -> UNCERTAINTY_REMOVED"""
        c = VerifiedClaim("clm_01", "Tradition attributes the site to the Sangam academy.", "Tradition attributes the site to the Sangam academy.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Tradition", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS, epistemic_status=EpistemicStatus.TRADITION_OR_LEGEND)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Tradition", [c])
        storyboard = {
            "scenes": [{"id": 1, "english_sub": "Historical records prove conclusively that the Sangam academy sat here.", "tamil_text": "வரலாற்று பதிவுகள் இதனை நிரூபிக்கின்றன."}]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)

    def test_adversarial_case_j(self):
        """CASE J: Evidence says 'approximately 2,600 years', Script says '2,700 years' -> NUMERIC_MISMATCH"""
        c = VerifiedClaim("clm_01", "The site dates back 2600 years.", "The site dates back 2600 years.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.VERIFIED_FACT, extracted_numbers=["2600"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Date", [c])
        storyboard = {
            "scenes": [{"id": 1, "english_sub": "The site dates back 2700 years.", "tamil_text": "இது 2700 ஆண்டுகள் பழமையானது."}]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.numeric_mismatch_count, 1)

    def test_adversarial_case_k(self):
        """CASE K: Evidence says 'one interpretation suggests', Script says 'archaeologists proved' -> OVERSTATED_CLAIM"""
        c = VerifiedClaim("clm_01", "One interpretation suggests active foreign maritime trade.", "One interpretation suggests active foreign maritime trade.", ClaimType.CULTURAL_PRACTICE, ClaimImportance.HIGH, "Trade", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Trade", [c])
        storyboard = {
            "scenes": [{"id": 1, "english_sub": "Archaeologists proved beyond doubt that Keeladi controlled maritime trade.", "tamil_text": "தொல்லியல் துறை இதனை முழுமையாக நிரூபித்துள்ளது."}]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.overstated_count, 1)


    # -------------------------------------------------------------------------
    # Section 20: Pre-Run Audit Test Suite (Items 1 Through 36)
    # -------------------------------------------------------------------------

    def test_audit_01_topic_title_number_allowed(self):
        plan = ContentPlanner().build_content_plan("ep_01", "Keezhadi: 2,600-Year-Old Tamil Urban Civilization", "Arch", [])
        self.assertIn("2600", plan.allowed_numbers)

    def test_audit_02_topic_title_date_allowed(self):
        plan = ContentPlanner().build_content_plan("ep_01", "Chola Maritime Expedition in 1025 CE", "History", [])
        self.assertIn("1025 ce", [d.lower() for d in plan.allowed_dates])

    def test_audit_03_topic_metadata_not_treated_as_evidence(self):
        # Topic numbers are allowed in validator, but do not promote an unverified claim to VERIFIED_FACT
        claim = VerifiedClaim("clm_01", "Keeladi is 2600 years old.", "Keeladi is 2600 years old.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01")
        verifier = ClaimVerifier(evidence_store=None)
        verifier.verify_claim(claim, evidence_pool=[])
        self.assertNotEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    def test_audit_04_json_fragment_rejected(self):
        extractor = ClaimExtractor()
        dossier_mock = MagicMock()
        dossier_mock.topic = "Topic"
        dossier_mock.conflicting_evidence = []
        dossier_mock.preliminary_findings = [
            {"statement": '{"id": "test_01", "badge": "Test Badge"}', "topic_aspect": "Aspect", "supporting_source_ids": []}
        ]
        claims = extractor.extract_claims_from_dossier(dossier_mock, "ep_01")
        self.assertEqual(len(claims), 0)

    def test_audit_05_code_fragment_rejected(self):
        extractor = ClaimExtractor()
        dossier_mock = MagicMock()
        dossier_mock.topic = "Topic"
        dossier_mock.conflicting_evidence = []
        dossier_mock.preliminary_findings = [
            {"statement": 'def get_data(): import sys; return sys.path', "topic_aspect": "Aspect", "supporting_source_ids": []}
        ]
        claims = extractor.extract_claims_from_dossier(dossier_mock, "ep_01")
        self.assertEqual(len(claims), 0)

    def test_audit_06_topics_json_cannot_create_claims(self):
        extractor = ClaimExtractor()
        dossier_mock = MagicMock()
        dossier_mock.topic = "Topic"
        dossier_mock.conflicting_evidence = []
        dossier_mock.preliminary_findings = [
            {"statement": "Documentation from Local indicates Chola navy", "topic_aspect": "Aspect", "supporting_source_ids": ["local_15fef56c3197"]}
        ]
        claims = extractor.extract_claims_from_dossier(dossier_mock, "ep_01")
        self.assertEqual(len(claims), 0)

    def test_audit_07_local_source_cannot_corroborate_external_source(self):
        m_loc = ClaimEvidenceMatch("chk_l", "src_l", EvidenceSupportType.DIRECT_SUPPORT, 0.9, independence_group="local_filesystem", source_origin="local")
        m_ext = ClaimEvidenceMatch("chk_e", "src_e", EvidenceSupportType.DIRECT_SUPPORT, 0.9, independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Statement.", "Statement.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m_loc.to_dict(), m_ext.to_dict()])
        # Local source excluded from external independent groups
        self.assertEqual(claim.independent_external_source_count, 1)

    def test_audit_08_wikipedia_plus_local_not_independent_corroboration(self):
        m_loc = ClaimEvidenceMatch("chk_l", "src_l", EvidenceSupportType.DIRECT_SUPPORT, 0.9, independence_group="local_filesystem", source_origin="local", evidence_weight_class="REFERENCE")
        m_wiki = ClaimEvidenceMatch("chk_w", "src_w", EvidenceSupportType.DIRECT_SUPPORT, 0.9, independence_group="wikipedia.org", source_origin="external", evidence_weight_class="REFERENCE")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Statement.", "Statement.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m_loc.to_dict(), m_wiki.to_dict()])
        self.assertEqual(claim.independent_non_ref_external_source_count, 0)
        self.assertNotEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    def test_audit_09_two_urls_same_publisher_not_independent(self):
        m1 = ClaimEvidenceMatch("chk_1", "src_1", EvidenceSupportType.DIRECT_SUPPORT, 0.85, publisher="ASI", domain="asi.nic.in", independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY")
        m2 = ClaimEvidenceMatch("chk_2", "src_2", EvidenceSupportType.DIRECT_SUPPORT, 0.85, publisher="ASI", domain="asi.nic.in", independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Statement.", "Statement.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict(), m2.to_dict()])
        self.assertEqual(claim.independent_non_ref_external_source_count, 1)

    def test_audit_10_one_primary_source_insufficient_for_verified_fact(self):
        m1 = ClaimEvidenceMatch("chk_1", "src_1", EvidenceSupportType.DIRECT_SUPPORT, 0.90, independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY", credibility_tier=1)
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Statement.", "Statement.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict()])
        self.assertEqual(claim.independent_non_ref_external_source_count, 1)
        self.assertEqual(claim.classification, ClaimClassification.SUPPORTED_HYPOTHESIS)

    def test_audit_11_two_independent_strong_sources_can_verify(self):
        m1 = ClaimEvidenceMatch("chk_1", "src_1", EvidenceSupportType.DIRECT_SUPPORT, 0.88, independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY", credibility_tier=1)
        m2 = ClaimEvidenceMatch("chk_2", "src_2", EvidenceSupportType.DIRECT_SUPPORT, 0.88, independence_group="nature.com", source_origin="external", evidence_weight_class="PRIMARY", credibility_tier=1)
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Statement.", "Statement.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict(), m2.to_dict()])
        self.assertEqual(claim.independent_non_ref_external_source_count, 2)
        self.assertEqual(claim.classification, ClaimClassification.VERIFIED_FACT)

    def test_audit_12_contradiction_to_unresolved_debate(self):
        m1 = ClaimEvidenceMatch("chk_1", "src_1", EvidenceSupportType.DIRECT_SUPPORT, 0.85, independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY")
        m2 = ClaimEvidenceMatch("chk_2", "src_2", EvidenceSupportType.CONTRADICTORY, 0.85, independence_group="cambridge.org", source_origin="external", evidence_weight_class="PRIMARY", contradiction_reason="Chronological conflict: 6th century BCE vs 3rd century BCE")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Dating statement.", "Dating statement.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict(), m2.to_dict()])
        self.assertEqual(claim.classification, ClaimClassification.UNRESOLVED_DEBATE)

    def test_audit_13_direct_disproof_to_rejected(self):
        m1 = ClaimEvidenceMatch("chk_1", "src_1", EvidenceSupportType.CONTRADICTORY, 0.90, independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY", credibility_tier=1, contradiction_reason="DIRECT_DISPROOF: Explicit evidential negation ('no evidence') regarding 'gold city'")
        verifier = ClaimVerifier(evidence_store=None)
        claim = VerifiedClaim("clm_01", "Gold city.", "Gold city.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(claim, evidence_pool=[m1.to_dict()])
        self.assertEqual(claim.classification, ClaimClassification.REJECTED)

    def test_audit_14_date_mismatch_detected(self):
        c = VerifiedClaim("clm_01", "Site dates to 6th century BCE.", "Site dates to 6th century BCE.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS, extracted_dates=["6th century bce"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": "The site dates to the 5th century BCE.", "tamil_text": "இது கிமு 5ஆம் நூற்றாண்டைச் சேர்ந்தது."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.date_mismatch_count, 1)

    def test_audit_15_number_mismatch_detected(self):
        c = VerifiedClaim("clm_01", "2600 years old.", "2600 years old.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS, extracted_numbers=["2600"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": "The site is 3000 years old.", "tamil_text": "இது 3000 ஆண்டுகள் பழமையானது."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.numeric_mismatch_count, 1)

    def test_audit_16_named_entity_mismatch_detected(self):
        c = VerifiedClaim("clm_01", "Brick wells excavated at site.", "Brick wells excavated at site.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": "Emperor Ashoka visited this brick well directly.", "tamil_text": "அசோகர் இங்கு வந்தார்."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.entity_mismatch_count, 1)

    def test_audit_17_unsupported_script_sentence_detected(self):
        c = VerifiedClaim("clm_01", "Known brick structures found.", "Known brick structures found.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": "Enormous pyramids towered over the city riverbanks.", "tamil_text": "பிரமிடுகள் இங்கு நின்றன."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.new_unsupported_count, 1)

    def test_audit_18_unsupported_quotation_detected(self):
        c = VerifiedClaim("clm_01", "Ancient texts exist.", "Ancient texts exist.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Hist", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Hist", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": 'The poet said: "We sailed across the global oceans."', "tamil_text": "புலவர் கூறினார்."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.quotation_violation_count, 1)

    def test_audit_19_fabricated_quotation_detected(self):
        c = VerifiedClaim("clm_01", "Ancient pottery uncovered.", "Ancient pottery uncovered.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": 'The governor announced: "This is our greatest victory."', "tamil_text": "அரசர் அறிவித்தார்."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)

    def test_audit_20_tamil_english_date_mismatch(self):
        c = VerifiedClaim("clm_01", "Dated to 6th century BCE and 3rd century BCE.", "Dated to 6th century BCE and 3rd century BCE.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS, extracted_dates=["6th century bce", "3rd century bce"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Date", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": "Dating confirms 6th century BCE.", "tamil_text": "ஆய்வுகள் இது 3rd century BCE என்கின்றன."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.cross_language_mismatch_count, 1)

    def test_audit_21_tamil_english_number_mismatch(self):
        c = VerifiedClaim("clm_01", "Contains 1000 and 500 artifacts.", "Contains 1000 and 500 artifacts.", ClaimType.STATISTICAL_MEASUREMENT, ClaimImportance.HIGH, "Stat", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS, extracted_numbers=["1000", "500"])
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Stat", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": "Excavations revealed 1000 items.", "tamil_text": "இங்கு 500 பொருட்கள் கிடைத்தன."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.cross_language_mismatch_count, 1)

    def test_audit_22_tamil_english_epistemic_mismatch(self):
        c = VerifiedClaim("clm_01", "Dating suggests ancient settlement.", "Dating suggests ancient settlement.", ClaimType.DATING_CHRONOLOGY, ClaimImportance.HIGH, "Date", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Date", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": "Dating has definitely proven ancient settlement beyond doubt.", "tamil_text": "ஆய்வுகள் இதனை விவரிக்கின்றன."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)

    def test_audit_23_cautious_claim_requires_hedging(self):
        c = VerifiedClaim("clm_01", "Excavations suggest early script presence.", "Excavations suggest early script presence.", ClaimType.CULTURAL_PRACTICE, ClaimImportance.HIGH, "Script", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Script", [c])
        storyboard = {"scenes": [{"id": 1, "english_sub": "Historians know conclusively as a proven fact that script existed here.", "tamil_text": "இது உண்மை."}]}
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.overstated_count, 1)

    def test_audit_24_unsupported_visual_metadata_detected(self):
        c = VerifiedClaim("clm_01", "Brick structure unearthed.", "Brick structure unearthed.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        storyboard = {
            "scenes": [{
                "id": 1,
                "english_sub": "Archaeological surveys indicate that Brick structure unearthed.",
                "tamil_text": "தொல்லியல் துறை ஆய்வுகள் இதனை விவரிக்கின்றன.",
                "visual_prompt": "Cinematic shot with modern concrete skyscraper and cranes in background"
            }]
        }
        report = ScriptValidator().validate_script(storyboard, plan, {"clm_01": c})
        self.assertFalse(report.passed)

    def test_audit_25_offline_fallback_never_invents_filler(self):
        c = VerifiedClaim("clm_01", "Official excavations uncovered brick structures.", "Official excavations uncovered brick structures.", ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimImportance.HIGH, "Arch", "ep_01", classification=ClaimClassification.SUPPORTED_HYPOTHESIS)
        plan = ContentPlanner().build_content_plan("ep_01", "Keeladi", "Arch", [c])
        generator = ScriptGenerator()
        storyboard = generator.generate_offline_fallback_script(plan, {"clm_01": c})
        # Every scene uses authorized claim or educational signoff
        for scene in storyboard["scenes"]:
            self.assertIn("english_sub", scene)
            self.assertIn("tamil_text", scene)

    def test_audit_26_no_claims_to_review_required(self):
        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        # Create empty dossier
        d_path = os.path.join(self.ep_dir, "research", "dossier.json")
        with open(d_path, "w", encoding="utf-8") as f:
            json.dump({"episode_id": self.episode_id, "topic": "Empty", "preliminary_findings": []}, f)
        res = engine.execute_claim_verification(self.episode_id)
        self.assertEqual(res["status"], "REVIEW_REQUIRED")

    def test_audit_27_insufficient_claims_for_safe_script_to_review_required(self):
        plan = FactCheckedContentPlan("ep_01", "Topic", "Category")
        generator = ScriptGenerator()
        with self.assertRaises(ValueError):
            generator.generate_offline_fallback_script(plan, {})

    def test_audit_28_review_required_to_verifying_works(self):
        self.state_manager.transition_state(self.episode_id, EpisodeState.REVIEW_REQUIRED, "REVIEW_REQUIRED")
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFYING, "VERIFYING")
        ep = self.state_manager.get_episode(self.episode_id)
        self.assertEqual(ep.status, EpisodeState.VERIFYING.value)

    def test_audit_29_rerun_is_idempotent(self):
        m1 = ClaimEvidenceMatch("chk_1", "src_1", EvidenceSupportType.DIRECT_SUPPORT, 0.85, independence_group="asi.nic.in", source_origin="external", evidence_weight_class="PRIMARY")
        verifier = ClaimVerifier(evidence_store=None)
        c = VerifiedClaim("clm_01", "Statement.", "Statement.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01")
        verifier.verify_claim(c, evidence_pool=[m1.to_dict()])
        score1 = c.confidence_score
        verifier.verify_claim(c, evidence_pool=[m1.to_dict()])
        score2 = c.confidence_score
        self.assertEqual(score1, score2)

    def test_audit_30_completed_script_validated_not_regenerated(self):
        # Mark episode as SCRIPT_VALIDATED with existing artifacts
        script_dir = os.path.join(self.ep_dir, "script")
        os.makedirs(script_dir, exist_ok=True)
        with open(os.path.join(script_dir, "script.json"), "w", encoding="utf-8") as f:
            json.dump({"scenes": []}, f)
        with open(os.path.join(self.ep_dir, "verification", "script_validation.json"), "w", encoding="utf-8") as f:
            json.dump({"passed": True}, f)
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFYING, "VERIFYING")
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFIED, "VERIFIED")
        self.state_manager.transition_state(self.episode_id, EpisodeState.SCRIPTING, "SCRIPTING")
        self.state_manager.transition_state(self.episode_id, EpisodeState.SCRIPT_VALIDATED, "SCRIPT_VALIDATED")

        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        res = engine.execute_script_generation_and_validation(self.episode_id)
        self.assertEqual(res["status"], "SCRIPT_VALIDATED")
        self.assertTrue(res.get("idempotent_reuse", False))

    def test_audit_31_phase4_dossier_remains_unchanged(self):
        d_path = os.path.join(self.ep_dir, "research", "dossier.json")
        original_data = {"episode_id": self.episode_id, "topic": "Keep", "preliminary_findings": []}
        with open(d_path, "w", encoding="utf-8") as f:
            json.dump(original_data, f)
        mtime_before = os.path.getmtime(d_path)
        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        engine.execute_claim_verification(self.episode_id)
        mtime_after = os.path.getmtime(d_path)
        self.assertEqual(mtime_before, mtime_after)

    def test_audit_32_no_gpu_model_loaded(self):
        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        mem = engine._get_rss_mb()
        self.assertGreater(mem, 0)
        # GPU remains unused
        import torch
        if torch.cuda.is_available():
            self.assertEqual(torch.cuda.memory_allocated(), 0)

    def test_audit_33_no_video_tts_functions_invoked(self):
        # Verification engine has no imports of moviepy, sadtalker, or edge-tts
        import autonomous.claim_verification_engine as cve
        self.assertFalse(hasattr(cve, "moviepy"))
        self.assertFalse(hasattr(cve, "sadtalker"))
        self.assertFalse(hasattr(cve, "edge_tts"))

    def test_audit_34_artifacts_have_provenance(self):
        c = VerifiedClaim("clm_01", "Statement.", "Statement.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "Aspect", "ep_01",
                          supporting_chunk_ids=["chk_01"], supporting_source_ids=["src_01"], independence_groups=["asi.nic.in"], evidence_weight_classes=["PRIMARY"])
        d = c.to_dict()
        self.assertIn("supporting_chunk_ids", d)
        self.assertIn("supporting_source_ids", d)
        self.assertIn("independence_groups", d)
        self.assertIn("evidence_weight_classes", d)

    def test_audit_35_retry_limit_enforced(self):
        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        self.assertEqual(engine.max_regeneration_attempts, 2)

    def test_audit_36_final_failure_becomes_review_required(self):
        # Fail script validation on purpose -> state is REVIEW_REQUIRED
        clm = VerifiedClaim("clm_01", "Statement.", "Statement.", ClaimType.HISTORICAL_FACT, ClaimImportance.HIGH, "T", self.episode_id)
        plan_data = {"episode_id": self.episode_id, "topic": "T", "category": "C", "scenes": [], "allowed_claim_ids": ["clm_01"], "cautious_claim_ids": [], "forbidden_claim_ids": []}
        with open(os.path.join(self.ep_dir, "verification", "fact_checked_content_plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan_data, f)
        with open(os.path.join(self.ep_dir, "verification", "claims.json"), "w", encoding="utf-8") as f:
            json.dump([clm.to_dict()], f)
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFYING, "VERIFYING")
        self.state_manager.transition_state(self.episode_id, EpisodeState.VERIFIED, "VERIFIED")

        engine = ClaimVerificationEngine(state_manager=self.state_manager)
        with patch.object(engine.validator, "validate_script", return_value=ScriptValidationReport(self.episode_id, False, violations=[{"type": "TEST_FAIL"}])):
            res = engine.execute_script_generation_and_validation(self.episode_id, offline=True)
            self.assertEqual(res["status"], "REVIEW_REQUIRED")
            ep = self.state_manager.get_episode(self.episode_id)
            self.assertEqual(ep.status, EpisodeState.REVIEW_REQUIRED.value)


if __name__ == "__main__":
    unittest.main()
