"""
autonomous/content_quality_validator.py - Phase 5.1 Content Quality & Topic Relevance Gate.

Evaluates whether a verified script delivers substantive, informative documentary content:
1. Minimum unique claims (default 3) without crediting paraphrases.
2. Repetition ceiling per claim (default max 2 scenes).
3. Topic relevance distribution (requires at least 2 CORE_TOPIC claims).
4. Aspect diversity (default at least 2 unique research aspects).
5. Scene purpose alignment (e.g. Scene 2 must contain CORE_TOPIC archaeological evidence).
6. Deterministic research expansion recommendation when evidence is insufficient.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from autonomous.claim_models import (
    VerifiedClaim,
    FactCheckedContentPlan,
    ContentQualityReport,
    ClaimClassification,
    ClaimRelevanceClass,
    ClaimType
)

logger = logging.getLogger("autonomous.content_quality_validator")


class ContentQualityValidator:
    """
    Deterministic Phase 5.1 Content Quality & Topic Relevance Validation Layer.
    Authoritatively prevents SCRIPT_VALIDATED if a script is semantically hollow,
    over-reliant on a single fact, or lacking core topic evidence.
    """

    def __init__(
        self,
        minimum_unique_claims: int = 3,
        minimum_unique_topic_aspects: int = 2,
        minimum_core_claims: int = 2,
        max_scenes_per_claim: int = 2,
    ):
        self.minimum_unique_claims = minimum_unique_claims
        self.minimum_unique_topic_aspects = minimum_unique_topic_aspects
        self.minimum_core_claims = minimum_core_claims
        self.max_scenes_per_claim = max_scenes_per_claim

    def validate_claims_pool(
        self,
        claims: List[VerifiedClaim],
        topic: str
    ) -> Tuple[bool, Dict[str, Any], List[str]]:
        """
        Pre-flight check on the pool of authorized claims.
        Determines if the research dossier provided enough core facts to sustain
        a 4-scene documentary before script generation begins.
        """
        authorized_claims = [
            c for c in claims
            if c.classification in (ClaimClassification.VERIFIED_FACT, ClaimClassification.SUPPORTED_HYPOTHESIS)
        ]
        unique_claim_ids = {c.claim_id for c in authorized_claims}
        core_claims = [
            c for c in authorized_claims
            if getattr(c, "relevance_class", None) == ClaimRelevanceClass.CORE_TOPIC
        ]
        unique_aspects = {
            getattr(c, "topic_aspect", "historical_context")
            for c in authorized_claims
        }

        metrics = {
            "total_authorized_claims": len(authorized_claims),
            "unique_claim_ids_count": len(unique_claim_ids),
            "core_claims_count": len(core_claims),
            "unique_aspects_count": len(unique_aspects),
            "research_expansion_recommended": False
        }

        reasons = []
        if len(core_claims) < self.minimum_core_claims:
            reasons.append(
                f"INSUFFICIENT_CORE_TOPIC_EVIDENCE: Authorized pool has {len(core_claims)} CORE_TOPIC claims "
                f"(minimum required: {self.minimum_core_claims})."
            )
            metrics["research_expansion_recommended"] = True

        if len(unique_claim_ids) < self.minimum_unique_claims:
            reasons.append(
                f"CONTENT_INSUFFICIENT: Authorized pool has {len(unique_claim_ids)} unique claims "
                f"(minimum required: {self.minimum_unique_claims})."
            )
            metrics["research_expansion_recommended"] = True

        if len(unique_aspects) < self.minimum_unique_topic_aspects:
            reasons.append(
                f"INSUFFICIENT_ASPECT_DIVERSITY: Authorized pool covers {len(unique_aspects)} topic aspects "
                f"(minimum required: {self.minimum_unique_topic_aspects})."
            )
            metrics["research_expansion_recommended"] = True

        passed = (len(reasons) == 0)
        return passed, metrics, reasons

    def validate_script_content_quality(
        self,
        storyboard: Dict[str, Any],
        content_plan: FactCheckedContentPlan,
        claims_map: Dict[str, VerifiedClaim]
    ) -> ContentQualityReport:
        """
        Deterministic audit of the generated storyboard script against quality criteria:
        1. Unique claim deduplication across scenes (paraphrases refer to same claim).
        2. Repetition ceiling (no claim in > 2 scenes).
        3. Core claim threshold (at least 2 CORE_TOPIC claims).
        4. Topic aspect diversity (at least 2 distinct aspects).
        5. Scene purpose alignment.
        """
        episode_id = content_plan.episode_id
        topic = content_plan.topic
        scenes = storyboard.get("scenes", [])

        # 1. Map scenes to underlying claims
        # Track: claim_id -> set of scene_ids where it appears
        claim_scene_usage: Dict[str, Set[int]] = defaultdict(set)
        scene_claim_map: Dict[int, List[str]] = defaultdict(list)
        total_propositions = 0

        for scene in scenes:
            scene_id = scene.get("id") or 1
            scene_plan = next((s for s in content_plan.scenes if s.scene_id == scene_id), None)
            
            # Identify claim IDs explicitly bound or entailed by narration in this scene
            scene_cids: Set[str] = set()

            # Check English narration against authorized claims
            en_sub = scene.get("english_sub", "")
            ta_text = scene.get("tamil_text", "")
            combined_text = f"{en_sub} {ta_text}".lower()

            for cid, claim in claims_map.items():
                if cid not in (content_plan.allowed_claim_ids + content_plan.cautious_claim_ids):
                    continue

                # Match if claim statement keywords or distinctive tokens appear in narration
                stmt_clean = claim.statement.lower()
                # If key entities or substantive claim segments appear
                stmt_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9]{3,}\b", stmt_clean) if t not in ("the", "and", "was", "set", "indicates", "evidence")]
                match_count = sum(1 for tok in stmt_tokens if tok in combined_text)
                
                # If bound to scene plan or high token overlap
                is_bound = (scene_plan and cid in scene_plan.allowed_claim_ids)
                if (match_count >= max(2, len(stmt_tokens) // 2)) or is_bound:
                    scene_cids.add(cid)

            # If no claim matched by text but scene plan authorized claims exist, assign scene plan claim
            if not scene_cids and scene_plan and scene_plan.allowed_claim_ids:
                scene_cids.add(scene_plan.allowed_claim_ids[0])

            for cid in scene_cids:
                claim_scene_usage[cid].add(scene_id)
                scene_claim_map[scene_id].append(cid)
                total_propositions += 1

        # Build repeated claim groups
        repeated_claim_groups: Dict[str, List[str]] = {}
        max_scene_reuse = 0
        for cid, sc_set in claim_scene_usage.items():
            sc_list = sorted([f"scene_{s}" for s in sc_set])
            repeated_claim_groups[cid] = sc_list
            if len(sc_set) > max_scene_reuse:
                max_scene_reuse = len(sc_set)

        unique_cids = sorted(list(claim_scene_usage.keys()))
        actual_unique_claims = len(unique_cids)

        # Categorize unique claims by relevance class and aspect
        core_claims = []
        supporting_claims = []
        background_claims = []
        unique_aspects: Set[str] = set()

        for cid in unique_cids:
            claim = claims_map.get(cid)
            if not claim:
                continue

            # Topic aspect
            asp = getattr(claim, "topic_aspect", "historical_context")
            unique_aspects.add(asp)

            # Relevance class
            rel = getattr(claim, "relevance_class", ClaimRelevanceClass.SUPPORTING_CONTEXT)
            if rel == ClaimRelevanceClass.CORE_TOPIC:
                core_claims.append(cid)
            elif rel == ClaimRelevanceClass.SUPPORTING_CONTEXT:
                supporting_claims.append(cid)
            else:
                background_claims.append(cid)

        core_count = len(core_claims)
        supporting_count = len(supporting_claims)
        background_count = len(background_claims)
        aspect_count = len(unique_aspects)

        # Evaluate failure conditions
        failure_reasons: List[str] = []
        primary_status = "PASSED"
        research_expansion_recommended = False

        # Check 1: Minimum Unique Claims
        if actual_unique_claims < self.minimum_unique_claims:
            reason = (
                f"CONTENT_INSUFFICIENT: Script contains only {actual_unique_claims} unique factual claims "
                f"(minimum required: {self.minimum_unique_claims})."
            )
            failure_reasons.append(reason)
            primary_status = "CONTENT_INSUFFICIENT"
            research_expansion_recommended = True

        # Check 2: Claim Repetition Limit (Max 2 scenes per claim)
        overused_claims = [cid for cid, sc_set in claim_scene_usage.items() if len(sc_set) > self.max_scenes_per_claim]
        if overused_claims:
            for cid in overused_claims:
                count = len(claim_scene_usage[cid])
                reason = (
                    f"CLAIM_OVERUSED: Claim '{cid}' appears in {count} scenes "
                    f"(maximum allowed: {self.max_scenes_per_claim}). Repeated across: {', '.join(repeated_claim_groups[cid])}."
                )
                failure_reasons.append(reason)
            if primary_status == "PASSED":
                primary_status = "CLAIM_OVERUSED"

        # Check 3: Core Topic Relevance (At least 2 CORE_TOPIC claims)
        if core_count < self.minimum_core_claims:
            reason = (
                f"CONTENT_INSUFFICIENT: Script contains only {core_count} CORE_TOPIC claims "
                f"(minimum required: {self.minimum_core_claims})."
            )
            failure_reasons.append(reason)
            primary_status = "CONTENT_INSUFFICIENT"
            research_expansion_recommended = True

        # Check 4: Topic Aspect Diversity (At least 2 distinct aspects)
        if aspect_count < self.minimum_unique_topic_aspects:
            reason = (
                f"CONTENT_INSUFFICIENT: Script covers only {aspect_count} topic aspects ({', '.join(unique_aspects)}) "
                f"(minimum required: {self.minimum_unique_topic_aspects})."
            )
            failure_reasons.append(reason)
            primary_status = "CONTENT_INSUFFICIENT"
            research_expansion_recommended = True

        # Check 5: Scene Purpose Validation
        scene_quality_results: List[Dict[str, Any]] = []
        for s in scenes:
            s_id = s.get("id") or 1
            cids_in_scene = scene_claim_map.get(s_id, [])
            scene_core = [cid for cid in cids_in_scene if cid in core_claims]
            
            s_res = {
                "scene_id": s_id,
                "scene_type": s.get("type", "unknown"),
                "claim_ids": cids_in_scene,
                "has_core_claim": len(scene_core) > 0,
                "status": "VALID"
            }

            # Scene 2 (Archaeology/Evidence) must contain at least one CORE_TOPIC claim
            if s_id == 2:
                if not scene_core:
                    s_res["status"] = "INSUFFICIENT_EVIDENCE_FOR_SCENE"
                    s_res["error"] = "Scene 2 (Archaeology/Evidence) lacks a verified CORE_TOPIC archaeological claim."
                    failure_reasons.append(
                        "INSUFFICIENT_EVIDENCE_FOR_SCENE: Scene 2 requires at least one CORE_TOPIC archaeological/evidence claim."
                    )
                    if primary_status == "PASSED":
                        primary_status = "INSUFFICIENT_EVIDENCE_FOR_SCENE"
                    research_expansion_recommended = True

            scene_quality_results.append(s_res)

        topic_relevance_results = {
            "core_claims": core_claims,
            "supporting_claims": supporting_claims,
            "background_claims": background_claims,
            "unique_aspects": sorted(list(unique_aspects))
        }

        passed = (len(failure_reasons) == 0)

        report = ContentQualityReport(
            episode_id=episode_id,
            topic=topic,
            passed=passed,
            minimum_unique_claims=self.minimum_unique_claims,
            actual_unique_claims=actual_unique_claims,
            minimum_unique_topic_aspects=self.minimum_unique_topic_aspects,
            actual_unique_topic_aspects=aspect_count,
            minimum_core_claims=self.minimum_core_claims,
            core_claim_count=core_count,
            supporting_claim_count=supporting_count,
            background_claim_count=background_count,
            repeated_claim_groups=repeated_claim_groups,
            maximum_claim_scene_reuse=max_scene_reuse,
            scene_quality_results=scene_quality_results,
            topic_relevance_results=topic_relevance_results,
            content_quality_status=primary_status if not passed else "PASSED",
            research_expansion_recommended=research_expansion_recommended,
            failure_reasons=failure_reasons
        )

        return report
