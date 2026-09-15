"""
autonomous/content_planner.py - Fact-Checked Content Planner & Storyboard Blueprint.

Constructs a strictly controlled 4-scene content plan containing ONLY verified,
cautiously framed, and debated historical claims. Forbids unverified facts and defines
the allowable factual vocabulary (dates, entities, numbers) for downstream script generation.
"""

import logging
from typing import List, Dict, Any, Set

from autonomous.claim_models import (
    ClaimType,
    ClaimClassification,
    EpistemicStatus,
    VerifiedClaim,
    SceneContentPlan,
    FactCheckedContentPlan
)
from autonomous.config import autonomous_settings

logger = logging.getLogger("autonomous.content_planner")


class ContentPlanner:
    """Synthesizes verified claims into an authorized 4-scene storyboard content plan."""

    def __init__(self):
        pass

    def build_content_plan(
        self,
        episode_id: str,
        topic: str,
        category: str,
        claims: List[VerifiedClaim]
    ) -> FactCheckedContentPlan:
        """
        Partition claims into strict authorization categories and build a 4-scene storyboard plan.
        """
        allowed_claim_ids: List[str] = []
        cautious_claim_ids: List[str] = []
        debated_claim_ids: List[str] = []
        theory_claim_ids: List[str] = []
        tradition_claim_ids: List[str] = []
        forbidden_claim_ids: List[str] = []

        allowed_entities: Set[str] = set()
        allowed_dates: Set[str] = set()
        allowed_numbers: Set[str] = set()
        verified_quotations: List[str] = []

        from autonomous.claim_extractor import ClaimExtractor
        from autonomous.claim_models import EntityType
        allowed_entities.update(ClaimExtractor.extract_meaningful_entities(topic))
        allowed_numbers.update(ClaimExtractor.extract_numbers(topic))
        allowed_dates.update(ClaimExtractor.extract_dates(topic))

        for c in claims:
            cid = c.claim_id
            stmt_ents = ClaimExtractor.extract_meaningful_entities(c.statement)
            stmt_dates = ClaimExtractor.extract_dates(c.statement)
            stmt_nums = ClaimExtractor.extract_numbers(c.statement)

            if c.classification == ClaimClassification.VERIFIED_FACT:
                allowed_claim_ids.append(cid)
                allowed_entities.update([e for e in (c.extracted_entities or stmt_ents) if ClaimExtractor.classify_entity(e) != EntityType.COMMON_WORD])
                allowed_dates.update(c.extracted_dates or stmt_dates)
                allowed_numbers.update(c.extracted_numbers or stmt_nums)
            elif c.classification == ClaimClassification.SUPPORTED_HYPOTHESIS:
                cautious_claim_ids.append(cid)
                allowed_entities.update([e for e in (c.extracted_entities or stmt_ents) if ClaimExtractor.classify_entity(e) != EntityType.COMMON_WORD])
                allowed_dates.update(c.extracted_dates or stmt_dates)
                allowed_numbers.update(c.extracted_numbers or stmt_nums)
            elif c.classification == ClaimClassification.UNRESOLVED_DEBATE:
                debated_claim_ids.append(cid)
                allowed_entities.update(c.extracted_entities or stmt_ents)
                allowed_dates.update(c.extracted_dates or stmt_dates)
                allowed_numbers.update(c.extracted_numbers or stmt_nums)
            elif c.epistemic_status == EpistemicStatus.PROPOSED_THEORY:
                theory_claim_ids.append(cid)
            elif c.epistemic_status == EpistemicStatus.TRADITION_OR_LEGEND:
                tradition_claim_ids.append(cid)
            else:
                forbidden_claim_ids.append(cid)

        # Distribute claims across the 4 standard scenes
        # Scene 1: Host Vlog (Introduction / Mystery Hook / Geographical Setting)
        s1_claims = [
            c.claim_id for c in claims
            if c.claim_id in (allowed_claim_ids + cautious_claim_ids)
            and c.claim_type in (ClaimType.GEOGRAPHICAL_LOCATION, ClaimType.HISTORICAL_FACT, ClaimType.HISTORICAL_FIGURE)
        ]
        if not s1_claims and (allowed_claim_ids + cautious_claim_ids):
            s1_claims = (allowed_claim_ids + cautious_claim_ids)[:2]

        # Scene 2: B-Roll (Physical Architecture / Material Discoveries / Artifacts)
        s2_claims = [
            c.claim_id for c in claims
            if c.claim_id in (allowed_claim_ids + cautious_claim_ids)
            and c.claim_type in (ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimType.MATERIAL_CULTURE)
        ]
        if not s2_claims and (allowed_claim_ids + cautious_claim_ids):
            s2_claims = (allowed_claim_ids + cautious_claim_ids)[1:3]

        # Scene 3: B-Roll (Chronology / Epigraphy / Scientific Dating)
        s3_claims = [
            c.claim_id for c in claims
            if c.claim_id in (allowed_claim_ids + cautious_claim_ids + debated_claim_ids)
            and c.claim_type in (ClaimType.DATING_CHRONOLOGY, ClaimType.STATISTICAL_MEASUREMENT, ClaimType.SUPERLATIVE_ASSERTION)
        ]
        if not s3_claims and (allowed_claim_ids + cautious_claim_ids):
            s3_claims = (allowed_claim_ids + cautious_claim_ids)[-2:]

        # Scene 4: Host Vlog (Cultural Significance / Civilizational Impact / Outro)
        s4_claims = [
            c.claim_id for c in claims
            if c.claim_id in (allowed_claim_ids + cautious_claim_ids)
            and c.claim_type in (ClaimType.CULTURAL_PRACTICE, ClaimType.HISTORICAL_FACT)
        ]
        if not s4_claims and (allowed_claim_ids + cautious_claim_ids):
            s4_claims = [(allowed_claim_ids + cautious_claim_ids)[0]]

        scenes: List[SceneContentPlan] = [
            SceneContentPlan(
                scene_id=1,
                scene_type="host_vlog",
                shot_type="PORTRAIT",
                badge=f"{category} • {autonomous_settings.channel.channel_name_en}",
                focus_theme="Location, mystery hook, and civilizational context",
                mandatory_claim_ids=s1_claims[:1],
                allowed_claim_ids=s1_claims,
                forbidden_assertions=[c.statement for c in claims if c.claim_id in forbidden_claim_ids],
                epistemic_framing_required=any(cid in cautious_claim_ids for cid in s1_claims),
                required_framing_phrases=["Excavations reveal...", "Archaeological surveys indicate..."]
            ),
            SceneContentPlan(
                scene_id=2,
                scene_type="broll_motion",
                shot_type="WIDE_ESTABLISHING",
                badge="Archaeological Evidence",
                focus_theme="Physical structures, material artifacts, and urban planning",
                mandatory_claim_ids=s2_claims[:1],
                allowed_claim_ids=s2_claims,
                forbidden_assertions=[c.statement for c in claims if c.claim_id in forbidden_claim_ids],
                epistemic_framing_required=any(cid in cautious_claim_ids for cid in s2_claims),
                required_framing_phrases=["Material findings confirm...", "Field excavations uncovered..."]
            ),
            SceneContentPlan(
                scene_id=3,
                scene_type="broll_motion",
                shot_type="ARCHITECTURE",
                badge="Scientific Chronology",
                focus_theme="Radiocarbon dating, epigraphy, and script evidence",
                mandatory_claim_ids=s3_claims[:1],
                allowed_claim_ids=s3_claims + debated_claim_ids,
                forbidden_assertions=[c.statement for c in claims if c.claim_id in forbidden_claim_ids],
                epistemic_framing_required=True,
                required_framing_phrases=["Scientific dating establishes...", "Scholarly debate notes..."]
            ),
            SceneContentPlan(
                scene_id=4,
                scene_type="host_vlog",
                shot_type="PORTRAIT",
                badge=f"{autonomous_settings.channel.channel_name_ta} • {autonomous_settings.channel.host_name_ta}",
                focus_theme="Synthesis of historical legacy and channel signoff",
                mandatory_claim_ids=s4_claims[:1],
                allowed_claim_ids=s4_claims,
                forbidden_assertions=[c.statement for c in claims if c.claim_id in forbidden_claim_ids],
                epistemic_framing_required=False,
                required_framing_phrases=[]
            ),
        ]

        content_plan = FactCheckedContentPlan(
            episode_id=episode_id,
            topic=topic,
            category=category,
            scenes=scenes,
            allowed_claim_ids=allowed_claim_ids,
            cautious_claim_ids=cautious_claim_ids,
            debated_claim_ids=debated_claim_ids,
            theory_claim_ids=theory_claim_ids,
            tradition_claim_ids=tradition_claim_ids,
            forbidden_claim_ids=forbidden_claim_ids,
            allowed_entities=sorted(list(allowed_entities)),
            allowed_dates=sorted(list(allowed_dates)),
            allowed_numbers=sorted(list(allowed_numbers)),
            verified_quotations=verified_quotations
        )

        logger.info(
            f"FactCheckedContentPlan constructed for episode '{episode_id}': "
            f"{len(allowed_claim_ids)} allowed, {len(cautious_claim_ids)} cautious, "
            f"{len(debated_claim_ids)} debated, {len(forbidden_claim_ids)} forbidden claims."
        )
        return content_plan
