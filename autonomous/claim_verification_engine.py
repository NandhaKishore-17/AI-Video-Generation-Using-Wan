"""
autonomous/claim_verification_engine.py - Master Pipeline Engine for Phase 5.

Coordinates Claim Extraction, Normalization, Evidence Retrieval, Deterministic Scoring,
Content Planning, Script Generation, Script Claim Re-Extraction, and Rigorous Validation.
Governs state transitions:
    RESEARCH_COMPLETE -> VERIFYING -> VERIFIED -> SCRIPTING -> SCRIPT_VALIDATED
Enforces the 2-attempt regeneration bound and routes failures to REVIEW_REQUIRED.
"""

import os
import time
import json
import psutil
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path

from autonomous.config import autonomous_settings, EPISODES_DIR
from autonomous.state_manager import StateManager, EpisodeState
from autonomous.research_dossier import ResearchDossier
from autonomous.evidence_store import EvidenceStore
from autonomous.claim_models import (
    VerifiedClaim,
    FactCheckedContentPlan,
    ScriptValidationReport,
    ContentQualityReport,
    ClaimClassification,
    ClaimRelevanceClass,
    EpistemicStatus,
    ClaimType,
    ClaimImportance
)
from autonomous.claim_extractor import ClaimExtractor
from autonomous.claim_verifier import ClaimVerifier
from autonomous.content_planner import ContentPlanner
from autonomous.script_generator import ScriptGenerator
from autonomous.script_validator import ScriptValidator
from autonomous.content_quality_validator import ContentQualityValidator

logger = logging.getLogger("autonomous.claim_verification_engine")


class ClaimVerificationEngine:
    """Master orchestrator for Phase 5 claim verification and fact-checked scripting."""

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        evidence_store: Optional[EvidenceStore] = None,
        max_regeneration_attempts: int = 2
    ):
        self.state_manager = state_manager or StateManager()
        self.evidence_store = evidence_store or EvidenceStore()
        self.max_regeneration_attempts = max_regeneration_attempts

        self.extractor = ClaimExtractor()
        self.verifier = ClaimVerifier(evidence_store=self.evidence_store)
        self.planner = ContentPlanner()
        self.generator = ScriptGenerator()
        self.validator = ScriptValidator(max_retries=self.max_regeneration_attempts)
        self.quality_validator = ContentQualityValidator()

    def _get_rss_mb(self) -> float:
        """Helper to sample current process RSS in MB."""
        try:
            return round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def execute_claim_verification(
        self,
        episode_id: str,
        evidence_pool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute Claim Extraction -> Normalization -> Evidence Retrieval -> Deterministic Scoring.
        Transitions: RESEARCH_COMPLETE -> VERIFYING -> VERIFIED (or REVIEW_REQUIRED).
        """
        t0 = time.time()
        rss_before = self._get_rss_mb()
        peak_rss = rss_before

        # 1. Validate episode state
        session = self.state_manager._get_session()
        try:
            from autonomous.state_manager import AutonomousEpisode
            ep = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if not ep:
                raise ValueError(f"Episode '{episode_id}' does not exist.")
            curr_state = EpisodeState(ep.status)
            topic = ep.topic
            category = ep.category
            ep_dir = Path(ep.output_directory)
        finally:
            session.close()

        if curr_state not in (EpisodeState.RESEARCH_COMPLETE, EpisodeState.VERIFYING, EpisodeState.REVIEW_REQUIRED, EpisodeState.SCRIPT_VALIDATED):
            logger.warning(f"Episode '{episode_id}' status is {curr_state.value}, proceeding with verification.")

        # Transition to VERIFYING
        if curr_state in (EpisodeState.RESEARCH_COMPLETE, EpisodeState.REVIEW_REQUIRED, EpisodeState.SCRIPT_VALIDATED):
            self.state_manager.transition_state(
                episode_id=episode_id,
                target_state=EpisodeState.VERIFYING,
                stage_name="VERIFYING"
            )

        verif_dir = ep_dir / "verification"
        verif_dir.mkdir(parents=True, exist_ok=True)

        # 2. Load Research Dossier
        dossier_path = ep_dir / "research" / "dossier.json"
        if not dossier_path.exists():
            raise FileNotFoundError(f"Research dossier not found at '{dossier_path}'. Phase 4 research required first.")

        with open(dossier_path, "r", encoding="utf-8") as f:
            dossier_data = json.load(f)

        import dataclasses
        defaults = {
            "episode_id": episode_id,
            "topic": topic,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "research_plan": {},
            "sources": [],
            "evidence_chunks": [],
            "research_questions": [],
            "question_evidence": [],
            "preliminary_findings": [],
            "conflicting_evidence": [],
            "unresolved_questions": [],
            "source_statistics": {},
            "research_status": "RESEARCH_COMPLETE",
            "timing_and_resources": {},
            "quality_gate_passed": True,
            "quality_gate_reasons": []
        }
        for k, v in defaults.items():
            if k not in dossier_data:
                dossier_data[k] = v
        dossier_fields = {f.name for f in dataclasses.fields(ResearchDossier)}
        filtered_data = {k: v for k, v in dossier_data.items() if k in dossier_fields}
        dossier = ResearchDossier(**filtered_data)

        # 3. Extract & Normalize Atomic Claims
        t_extract_0 = time.time()
        candidate_claims = self.extractor.extract_claims_from_dossier(dossier, episode_id)
        t_extract = time.time() - t_extract_0
        peak_rss = max(peak_rss, self._get_rss_mb())

        # 4. Verify Claims & Deterministic Confidence Scoring
        t_verify_0 = time.time()
        verified_claims = self.verifier.verify_all_claims(candidate_claims, evidence_pool=evidence_pool, topic=topic)
        t_verify = time.time() - t_verify_0
        peak_rss = max(peak_rss, self._get_rss_mb())

        # 5. Build Fact-Checked Content Plan
        t_plan_0 = time.time()
        content_plan = self.planner.build_content_plan(
            episode_id=episode_id,
            topic=topic,
            category=category,
            claims=verified_claims
        )
        t_plan = time.time() - t_plan_0
        peak_rss = max(peak_rss, self._get_rss_mb())

        # Phase 5.1 Pre-flight claims pool quality assessment
        pool_passed, pool_metrics, pool_reasons = self.quality_validator.validate_claims_pool(verified_claims, topic)
        initial_cq_report = ContentQualityReport(
            episode_id=episode_id,
            topic=topic,
            passed=pool_passed,
            minimum_unique_claims=self.quality_validator.minimum_unique_claims,
            actual_unique_claims=pool_metrics["unique_claim_ids_count"],
            minimum_unique_topic_aspects=self.quality_validator.minimum_unique_topic_aspects,
            actual_unique_topic_aspects=pool_metrics["unique_aspects_count"],
            minimum_core_claims=self.quality_validator.minimum_core_claims,
            core_claim_count=pool_metrics["core_claims_count"],
            supporting_claim_count=len([c for c in verified_claims if getattr(c, "relevance_class", None) == ClaimRelevanceClass.SUPPORTING_CONTEXT]),
            background_claim_count=len([c for c in verified_claims if getattr(c, "relevance_class", None) == ClaimRelevanceClass.BACKGROUND]),
            repeated_claim_groups={},
            maximum_claim_scene_reuse=0,
            scene_quality_results=[],
            topic_relevance_results={"core_claims": [c.claim_id for c in verified_claims if getattr(c, "relevance_class", None) == ClaimRelevanceClass.CORE_TOPIC]},
            content_quality_status="PASSED" if pool_passed else "CONTENT_INSUFFICIENT",
            research_expansion_recommended=pool_metrics["research_expansion_recommended"],
            failure_reasons=pool_reasons
        )
        for target_dir in (verif_dir, ep_dir / "research"):
            target_dir.mkdir(parents=True, exist_ok=True)
            with open(target_dir / "content_quality_report.json", "w", encoding="utf-8") as f:
                json.dump(initial_cq_report.to_dict(), f, indent=2, ensure_ascii=False)

        # 6. Save Artifacts (Section 16: Separate artifact generation with audit provenance)
        claims_dict = [c.to_dict() for c in verified_claims]
        with open(verif_dir / "claims.json", "w", encoding="utf-8") as f:
            json.dump(claims_dict, f, indent=2, ensure_ascii=False)

        claims_dir = ep_dir / "claims"
        claims_dir.mkdir(parents=True, exist_ok=True)
        with open(claims_dir / "claims.json", "w", encoding="utf-8") as f:
            json.dump(claims_dict, f, indent=2, ensure_ascii=False)

        confidence_report = {
            "episode_id": episode_id,
            "topic": topic,
            "total_claims": len(verified_claims),
            "verified_facts_count": len([c for c in verified_claims if c.classification == ClaimClassification.VERIFIED_FACT]),
            "supported_hypotheses_count": len([c for c in verified_claims if c.classification == ClaimClassification.SUPPORTED_HYPOTHESIS]),
            "unresolved_debates_count": len([c for c in verified_claims if c.classification == ClaimClassification.UNRESOLVED_DEBATE]),
            "unverified_claims_count": len([c for c in verified_claims if c.classification == ClaimClassification.UNVERIFIED_CLAIM]),
            "average_confidence": round(sum(c.confidence_score for c in verified_claims) / max(1, len(verified_claims)), 4),
            "claims_breakdown": [
                {
                    "claim_id": c.claim_id,
                    "statement": c.statement,
                    "classification": c.classification.value,
                    "epistemic_status": c.epistemic_status.value,
                    "confidence_score": c.confidence_score,
                    "importance": c.importance.value,
                    "breakdown": c.confidence_breakdown,
                    "independent_external_sources": c.independent_external_source_count,
                    "high_quality_sources": c.high_quality_source_count,
                    "direct_support_count": c.direct_support_count,
                    "contradicting_sources": c.contradicting_source_count
                }
                for c in verified_claims
            ]
        }
        with open(verif_dir / "confidence_report.json", "w", encoding="utf-8") as f:
            json.dump(confidence_report, f, indent=2, ensure_ascii=False)
        with open(verif_dir / "verification_report.json", "w", encoding="utf-8") as f:
            json.dump(confidence_report, f, indent=2, ensure_ascii=False)

        with open(verif_dir / "fact_checked_content_plan.json", "w", encoding="utf-8") as f:
            json.dump(content_plan.to_dict(), f, indent=2, ensure_ascii=False)

        plan_dir = ep_dir / "planning"
        plan_dir.mkdir(parents=True, exist_ok=True)
        with open(plan_dir / "content_plan.json", "w", encoding="utf-8") as f:
            json.dump(content_plan.to_dict(), f, indent=2, ensure_ascii=False)

        # 7. Persist Claims to Database
        if hasattr(self.state_manager, "record_verified_claims"):
            self.state_manager.record_verified_claims(episode_id, verified_claims)

        # 8. Evaluate Quality Gate for Verification Stage
        verified_count = confidence_report["verified_facts_count"]
        cautious_count = confidence_report["supported_hypotheses_count"]
        has_safe_content = (verified_count + cautious_count) >= 1

        t_total = time.time() - t0
        rss_after = self._get_rss_mb()

        timing_and_mem = {
            "claim_extraction_time_s": round(t_extract, 3),
            "claim_verification_time_s": round(t_verify, 3),
            "content_planning_time_s": round(t_plan, 3),
            "total_verification_time_s": round(t_total, 3),
            "rss_before_mb": rss_before,
            "peak_rss_mb": peak_rss,
            "rss_after_mb": rss_after,
            "gpu_usage": "0% (CPU execution only)"
        }
        with open(verif_dir / "claim_verification.json", "w", encoding="utf-8") as f:
            json.dump({
                "status": "VERIFIED" if has_safe_content else "REVIEW_REQUIRED",
                "summary": confidence_report,
                "timing_and_resources": timing_and_mem
            }, f, indent=2, ensure_ascii=False)

        if has_safe_content:
            logger.info(f"Episode '{episode_id}' passed Claim Verification. Advancing to VERIFIED.")
            self.state_manager.transition_state(
                episode_id=episode_id,
                target_state=EpisodeState.VERIFIED,
                stage_name="VERIFIED"
            )
            return {
                "status": "VERIFIED",
                "episode_id": episode_id,
                "verified_claims_count": len(verified_claims),
                "verified_facts_count": verified_count,
                "supported_hypotheses_count": cautious_count,
                "content_plan_path": str(verif_dir / "fact_checked_content_plan.json"),
                "timing": timing_and_mem
            }
        else:
            reason = "No claims met VERIFIED_FACT or SUPPORTED_HYPOTHESIS standards. Insufficient evidence."
            logger.warning(f"Episode '{episode_id}' failed verification gate: {reason}. Routing to REVIEW_REQUIRED.")
            self.state_manager.transition_state(
                episode_id=episode_id,
                target_state=EpisodeState.REVIEW_REQUIRED,
                stage_name="REVIEW_REQUIRED",
                error_message=reason
            )
            return {
                "status": "REVIEW_REQUIRED",
                "episode_id": episode_id,
                "error": reason,
                "timing": timing_and_mem
            }

    def execute_script_generation_and_validation(
        self,
        episode_id: str,
        offline: bool = False
    ) -> Dict[str, Any]:
        """
        Execute Fact-Checked Script Generation -> Re-Extraction -> Validation.
        Enforces 2-attempt regeneration bound and transitions:
            VERIFIED -> SCRIPTING -> SCRIPT_VALIDATED (or REVIEW_REQUIRED).
        """
        t0 = time.time()
        rss_before = self._get_rss_mb()
        peak_rss = rss_before

        session = self.state_manager._get_session()
        try:
            from autonomous.state_manager import AutonomousEpisode
            ep = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if not ep:
                raise ValueError(f"Episode '{episode_id}' does not exist.")
            curr_state = EpisodeState(ep.status)
            ep_dir = Path(ep.output_directory)
        finally:
            session.close()

        script_dir = ep_dir / "script"
        verified_script_path = script_dir / "script.json"
        verif_dir = ep_dir / "verification"
        script_val_path = verif_dir / "script_validation.json"

        # Section 15 Idempotency: If SCRIPT_VALIDATED already completed and content quality didn't fail, do not regenerate
        cq_file = verif_dir / "content_quality_report.json"
        if curr_state == EpisodeState.SCRIPT_VALIDATED and verified_script_path.exists() and script_val_path.exists():
            cq_data = {}
            if cq_file.exists():
                try:
                    with open(cq_file, "r", encoding="utf-8") as f:
                        cq_data = json.load(f)
                except Exception:
                    pass
            # If content quality report exists and explicitly failed, do not reuse
            if not (cq_file.exists() and cq_data.get("passed") is False):
                logger.info(f"Episode '{episode_id}' is already SCRIPT_VALIDATED. Reusing existing validated script artifacts (idempotent).")
                with open(script_val_path, "r", encoding="utf-8") as f:
                    val_data = json.load(f)
                return {
                    "status": "SCRIPT_VALIDATED",
                    "episode_id": episode_id,
                    "script_path": str(verified_script_path),
                    "attempts": 1,
                    "validation_report": val_data,
                    "content_quality_report": cq_data,
                    "idempotent_reuse": True
                }

        # Load content plan and verified claims
        plan_file = verif_dir / "fact_checked_content_plan.json"
        claims_file = verif_dir / "claims.json"

        if not plan_file.exists() or not claims_file.exists():
            raise FileNotFoundError(f"Content plan or claims not found in '{verif_dir}'. Execute claim verification first.")

        with open(plan_file, "r", encoding="utf-8") as f:
            plan_data = json.load(f)
        with open(claims_file, "r", encoding="utf-8") as f:
            claims_data = json.load(f)

        from autonomous.claim_models import SceneContentPlan
        scenes = [SceneContentPlan(**s) for s in plan_data.get("scenes", [])]
        content_plan = FactCheckedContentPlan(
            episode_id=plan_data["episode_id"],
            topic=plan_data["topic"],
            category=plan_data["category"],
            scenes=scenes,
            allowed_claim_ids=plan_data.get("allowed_claim_ids", []),
            cautious_claim_ids=plan_data.get("cautious_claim_ids", []),
            debated_claim_ids=plan_data.get("debated_claim_ids", []),
            theory_claim_ids=plan_data.get("theory_claim_ids", []),
            tradition_claim_ids=plan_data.get("tradition_claim_ids", []),
            forbidden_claim_ids=plan_data.get("forbidden_claim_ids", []),
            allowed_entities=plan_data.get("allowed_entities", []),
            allowed_dates=plan_data.get("allowed_dates", []),
            allowed_numbers=plan_data.get("allowed_numbers", []),
            verified_quotations=plan_data.get("verified_quotations", [])
        )

        claims_map: Dict[str, VerifiedClaim] = {}
        for cd in claims_data:
            vc = VerifiedClaim(
                claim_id=cd["claim_id"],
                statement=cd["statement"],
                normalized_statement=cd.get("normalized_statement", cd["statement"]),
                claim_type=ClaimType(cd["claim_type"]),
                importance=ClaimImportance(cd["importance"]),
                topic_aspect=cd["topic_aspect"],
                episode_id=episode_id,
                classification=ClaimClassification(cd["classification"]),
                epistemic_status=EpistemicStatus(cd["epistemic_status"]),
                relevance_class=ClaimRelevanceClass(cd.get("relevance_class", "SUPPORTING_CONTEXT")),
                extracted_entities=cd.get("extracted_entities", []),
                extracted_dates=cd.get("extracted_dates", []),
                extracted_numbers=cd.get("extracted_numbers", [])
            )
            claims_map[vc.claim_id] = vc

        # Transition to SCRIPTING
        self.state_manager.transition_state(
            episode_id=episode_id,
            target_state=EpisodeState.SCRIPTING,
            stage_name="SCRIPTING"
        )

        script_dir = ep_dir / "script"
        script_dir.mkdir(parents=True, exist_ok=True)

        storyboard = None
        validation_report = None
        content_quality_report = None
        passed = False
        attempt_used = 0

        # Regeneration loop (bounded to 2 attempts)
        for attempt in range(1, self.max_regeneration_attempts + 1):
            attempt_used = attempt
            logger.info(f"Generating storyboard script for episode '{episode_id}' (Attempt {attempt}/{self.max_regeneration_attempts})...")

            # On attempt 2, if attempt 1 failed, we enforce deterministic synthesis
            use_offline = offline or (attempt > 1)
            try:
                storyboard = self.generator.generate_script(content_plan, claims_map, offline=use_offline)
            except Exception as e:
                logger.error(f"Script generation error on attempt {attempt}: {e}")
                validation_report = ScriptValidationReport(
                    episode_id=episode_id,
                    passed=False,
                    violations=[{"type": "SCRIPT_GENERATION_ERROR", "details": str(e)}]
                )
                continue
            peak_rss = max(peak_rss, self._get_rss_mb())

            # Save draft script
            with open(verif_dir / f"draft_script_attempt_{attempt}.json", "w", encoding="utf-8") as f:
                json.dump(storyboard, f, indent=2, ensure_ascii=False)

            # Validate generated script
            validation_report = self.validator.validate_script(storyboard, content_plan, claims_map)
            peak_rss = max(peak_rss, self._get_rss_mb())

            # Save validation report
            with open(verif_dir / "script_validation.json", "w", encoding="utf-8") as f:
                json.dump(validation_report.to_dict(), f, indent=2, ensure_ascii=False)

            if validation_report.passed:
                # Phase 5.1 Content Quality & Topic Relevance Validation
                content_quality_report = self.quality_validator.validate_script_content_quality(
                    storyboard=storyboard,
                    content_plan=content_plan,
                    claims_map=claims_map
                )

                # Persist content quality report
                for target_dir in (verif_dir, ep_dir / "research"):
                    target_dir.mkdir(parents=True, exist_ok=True)
                    with open(target_dir / "content_quality_report.json", "w", encoding="utf-8") as f:
                        json.dump(content_quality_report.to_dict(), f, indent=2, ensure_ascii=False)

                if content_quality_report.passed:
                    passed = True
                    logger.info(f"Script validation & Content Quality PASSED on attempt {attempt}.")
                    break
                else:
                    passed = False
                    quality_reason = (
                        f"Script validation passed evidence check but FAILED Content Quality on attempt {attempt}: "
                        f"{content_quality_report.content_quality_status} ({'; '.join(content_quality_report.failure_reasons)})"
                    )
                    logger.warning(quality_reason)
            else:
                logger.warning(
                    f"Script validation FAILED on attempt {attempt}: "
                    f"{len(validation_report.violations)} violations detected."
                )

        t_total = time.time() - t0
        rss_after = self._get_rss_mb()

        # Ensure content_quality_report is populated and saved on disk
        if not content_quality_report:
            cq_path = verif_dir / "content_quality_report.json"
            if cq_path.exists():
                try:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        cq_dict = json.load(f)
                    content_quality_report = ContentQualityReport(**cq_dict)
                except Exception:
                    pass
            if not content_quality_report:
                content_quality_report = ContentQualityReport(
                    episode_id=episode_id,
                    topic=content_plan.topic,
                    passed=False,
                    content_quality_status="CONTENT_INSUFFICIENT",
                    research_expansion_recommended=True,
                    failure_reasons=["Script validation failed before content quality could pass."]
                )
                for target_dir in (verif_dir, ep_dir / "research"):
                    target_dir.mkdir(parents=True, exist_ok=True)
                    with open(target_dir / "content_quality_report.json", "w", encoding="utf-8") as f:
                        json.dump(content_quality_report.to_dict(), f, indent=2, ensure_ascii=False)

        timing_and_mem = {
            "attempts": attempt_used,
            "total_scripting_time_s": round(t_total, 3),
            "rss_before_mb": rss_before,
            "peak_rss_mb": peak_rss,
            "rss_after_mb": rss_after,
            "gpu_usage": "0% (CPU execution only)"
        }

        if passed and storyboard:
            # Save verified script to official locations
            verified_script_path = script_dir / "script.json"
            with open(verified_script_path, "w", encoding="utf-8") as f:
                json.dump(storyboard, f, indent=2, ensure_ascii=False)

            with open(script_dir / "script_validation.json", "w", encoding="utf-8") as f:
                json.dump(validation_report.to_dict(), f, indent=2, ensure_ascii=False)

            with open(verif_dir / "verified_script.json", "w", encoding="utf-8") as f:
                json.dump(storyboard, f, indent=2, ensure_ascii=False)

            # Update database record
            session = self.state_manager._get_session()
            try:
                from autonomous.state_manager import AutonomousEpisode
                ep = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
                if ep:
                    ep.script_path = str(verified_script_path)
                    ep.script_validation_status = "PASSED"
                    session.commit()
            finally:
                session.close()

            # Transition to SCRIPT_VALIDATED
            self.state_manager.transition_state(
                episode_id=episode_id,
                target_state=EpisodeState.SCRIPT_VALIDATED,
                stage_name="SCRIPT_VALIDATED"
            )

            logger.info(f"Phase 5 Complete: Episode '{episode_id}' advanced to SCRIPT_VALIDATED.")
            return {
                "status": "SCRIPT_VALIDATED",
                "episode_id": episode_id,
                "script_path": str(verified_script_path),
                "attempts": attempt_used,
                "validation_report": validation_report.to_dict(),
                "content_quality_report": content_quality_report.to_dict() if content_quality_report else {},
                "timing": timing_and_mem
            }
        else:
            if validation_report and validation_report.passed and content_quality_report and not content_quality_report.passed:
                reason = f"Script failed content quality gate: {content_quality_report.content_quality_status} ({'; '.join(content_quality_report.failure_reasons)})"
            else:
                reason = f"Script validation failed after {attempt_used} attempts: {len(validation_report.violations) if validation_report else 0} violations."
            logger.warning(f"Episode '{episode_id}' routed to REVIEW_REQUIRED: {reason}")
            self.state_manager.transition_state(
                episode_id=episode_id,
                target_state=EpisodeState.REVIEW_REQUIRED,
                stage_name="REVIEW_REQUIRED",
                error_message=reason
            )
            return {
                "status": "REVIEW_REQUIRED",
                "episode_id": episode_id,
                "error": reason,
                "validation_report": validation_report.to_dict() if validation_report else {},
                "content_quality_report": content_quality_report.to_dict() if content_quality_report else {},
                "timing": timing_and_mem
            }

    def execute_full_phase5(
        self,
        episode_id: str,
        offline: bool = False,
        evidence_pool: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute full Phase 5 pipeline from RESEARCH_COMPLETE to SCRIPT_VALIDATED.
        """
        verif_res = self.execute_claim_verification(episode_id, evidence_pool=evidence_pool)
        if verif_res["status"] != "VERIFIED":
            return verif_res

        script_res = self.execute_script_generation_and_validation(episode_id, offline=offline)
        return script_res
