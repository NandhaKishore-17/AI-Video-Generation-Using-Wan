"""
autonomous/research_gap_analyzer.py - Deterministic Research & Content Gap Analyzer.

Inspects verified claims, cautious claims, rejected claims, content quality reports,
and scene purposes to diagnose exact evidence deficits, missing topic aspects, and
scene requirements. Generates targeted, non-repetitive research questions for bounded
Phase 5.2 autonomous research expansion.
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set

from autonomous.claim_models import (
    ClaimRelevanceClass,
    ClaimClassification,
    VerifiedClaim,
    ContentQualityReport
)

logger = logging.getLogger("autonomous.research_gap_analyzer")


@dataclass
class ResearchGapReport:
    """Diagnostic report specifying research and content gaps for an episode."""
    topic: str
    missing_claim_count: int
    missing_core_claim_count: int
    missing_topic_aspects: List[str]
    missing_scene_requirements: List[Dict[str, Any]]
    weak_claims: List[Dict[str, Any]]
    failed_claims: List[Dict[str, Any]]
    recommended_research_questions: List[str]
    recommended_source_types: List[str]
    priority_order: List[str]
    expansion_required: bool
    attempted_questions: List[str] = field(default_factory=list)
    expansion_number: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchGapReport":
        return cls(**data)


# Candidate topic aspects appropriate for historical / archaeological topics
HISTORICAL_TOPIC_ASPECTS = [
    "archaeology",
    "settlement",
    "chronology",
    "dating",
    "literacy",
    "inscriptions",
    "material_culture",
    "trade",
    "technology",
    "cultural_significance",
    "geography",
    "historical_context"
]

# High-priority source categories
AUTHORITATIVE_SOURCE_TYPES = [
    "primary_archaeological_reports",
    "Archaeological Survey of India (ASI)",
    "Tamil Nadu State Department of Archaeology (TNSDA)",
    "peer_reviewed_archaeological_journals",
    "university_excavation_monographs",
    "academic_repositories",
    "museum_catalogues"
]


class ResearchGapAnalyzer:
    """
    Analyzes verified claims and content quality reports to determine concrete missing
    evidence dimensions and formulate targeted, non-duplicate research questions.
    """

    def __init__(self, ollama_client: Optional[Any] = None):
        self.ollama_client = ollama_client

    def _extract_site_or_subject(self, topic: str) -> str:
        """Extract main entity or site name from topic title."""
        cleaned = topic.split(":")[0].strip()
        cleaned = re.sub(r"\b(2,600-Year-Old|Ancient|Urban|Civilization|Early|Literacy)\b", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned or topic.split(":")[0].strip()

    def _calculate_jaccard_similarity(self, s1: str, s2: str) -> float:
        """Calculate token-level Jaccard similarity to reject near-duplicate questions."""
        tokens1 = set(re.findall(r"\w+", s1.lower()))
        tokens2 = set(re.findall(r"\w+", s2.lower()))
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union)

    def is_question_duplicate(self, candidate: str, attempted_questions: List[str], threshold: float = 0.70) -> bool:
        """Check if candidate question is identical or near-duplicate to previously attempted questions."""
        candidate_clean = re.sub(r"\W+", " ", candidate.strip().lower())
        for q in attempted_questions:
            q_clean = re.sub(r"\W+", " ", q.strip().lower())
            if candidate_clean == q_clean:
                return True
            sim = self._calculate_jaccard_similarity(candidate, q)
            if sim >= threshold:
                return True
        return False

    def analyze_gaps(
        self,
        topic: str,
        verified_claims: Optional[List[Any]] = None,
        cautious_claims: Optional[List[Any]] = None,
        rejected_claims: Optional[List[Any]] = None,
        content_quality_report: Optional[Dict[str, Any]] = None,
        content_plan: Optional[Any] = None,
        dossier: Optional[Any] = None,
        attempted_questions: Optional[List[str]] = None,
        expansion_number: int = 1,
        minimum_unique_claims: int = 3,
        minimum_core_claims: int = 2,
        minimum_unique_topic_aspects: int = 2,
        offline: bool = False
    ) -> ResearchGapReport:
        """
        Produce a comprehensive ResearchGapReport diagnosing missing evidence and generating
        targeted research questions tailored to remaining deficits.
        """
        all_attempted = list(attempted_questions or [])
        verified_claims = verified_claims or []
        cautious_claims = cautious_claims or []
        rejected_claims = rejected_claims or []

        # 1. Evaluate claim counts and relevance classes
        if content_quality_report:
            actual_unique_claims = content_quality_report.get("actual_unique_claims", 0)
            actual_core_claims = content_quality_report.get("core_claim_count", 0)
            covered_aspects = set(content_quality_report.get("topic_relevance_results", {}).get("unique_aspects", []))
            cq_passed = content_quality_report.get("passed", False)
            scene_results = content_quality_report.get("scene_quality_results", [])
        else:
            all_safe = verified_claims + cautious_claims
            actual_unique_claims = len(set(c.claim_id for c in all_safe))
            actual_core_claims = len([c for c in all_safe if getattr(c, "relevance_class", None) == ClaimRelevanceClass.CORE_TOPIC])
            covered_aspects = set(getattr(c, "topic_aspect", "") for c in all_safe if getattr(c, "topic_aspect", ""))
            cq_passed = actual_unique_claims >= minimum_unique_claims and actual_core_claims >= minimum_core_claims and len(covered_aspects) >= minimum_unique_topic_aspects
            scene_results = []

        missing_claim_count = max(0, minimum_unique_claims - actual_unique_claims)
        missing_core_claim_count = max(0, minimum_core_claims - actual_core_claims)

        # 2. Topic Aspect Gaps
        topic_lower = topic.lower()
        applicable_aspects = list(HISTORICAL_TOPIC_ASPECTS)
        if "keezhadi" in topic_lower or "keeladi" in topic_lower:
            applicable_aspects = [
                "archaeology",
                "settlement",
                "chronology",
                "dating",
                "literacy",
                "inscriptions",
                "material_culture",
                "cultural_significance"
            ]

        missing_topic_aspects = [a for a in applicable_aspects if a not in covered_aspects]

        # 3. Scene Requirement Gaps
        missing_scene_requirements: List[Dict[str, Any]] = []
        if content_quality_report:
            for sr in scene_results:
                if sr.get("status") == "INSUFFICIENT_EVIDENCE_FOR_SCENE":
                    missing_scene_requirements.append({
                        "scene_id": sr.get("scene_id"),
                        "scene_type": sr.get("scene_type"),
                        "required_class": "CORE_TOPIC",
                        "error": sr.get("error", "Scene lacks required verified evidence.")
                    })
        else:
            # Synthetic evaluation if no report was pre-computed
            has_archaeology_core = any(
                getattr(c, "relevance_class", None) == ClaimRelevanceClass.CORE_TOPIC and
                getattr(c, "topic_aspect", "") in ("archaeology", "settlement")
                for c in (verified_claims + cautious_claims)
            )
            if not has_archaeology_core:
                missing_scene_requirements.append({
                    "scene_id": 2,
                    "scene_type": "broll_motion",
                    "required_class": "CORE_TOPIC",
                    "error": "Scene 2 (Archaeology/Evidence) lacks a verified CORE_TOPIC archaeological claim."
                })

        # 4. Identify Weak and Failed Claims
        weak_claims: List[Dict[str, Any]] = []
        for c in cautious_claims:
            c_dict = c.to_dict() if hasattr(c, "to_dict") else dict(c)
            weak_claims.append({
                "claim_id": c_dict.get("claim_id"),
                "statement": c_dict.get("statement"),
                "confidence_score": c_dict.get("confidence_score", 0.0),
                "classification": c_dict.get("classification"),
                "relevance_class": str(getattr(c, "relevance_class", ClaimRelevanceClass.SUPPORTING_CONTEXT)),
                "reason": "Cautious claim requiring stronger independent corroboration."
            })

        failed_claims: List[Dict[str, Any]] = []
        for c in rejected_claims:
            c_dict = c.to_dict() if hasattr(c, "to_dict") else dict(c)
            failed_claims.append({
                "claim_id": c_dict.get("claim_id"),
                "statement": c_dict.get("statement"),
                "classification": c_dict.get("classification"),
                "reason": c_dict.get("failure_reason", "Failed verification thresholds or contradicted.")
            })

        # 5. Prioritize Aspect Ordering
        # If core claims are missing, archaeology & settlement take highest priority
        priority_order: List[str] = []
        if missing_core_claim_count > 0:
            for asp in ("archaeology", "settlement", "chronology", "dating", "inscriptions", "literacy", "material_culture", "cultural_significance"):
                if asp in missing_topic_aspects and asp not in priority_order:
                    priority_order.append(asp)
        else:
            for asp in missing_topic_aspects:
                if asp not in priority_order:
                    priority_order.append(asp)

        # Include remaining applicable aspects
        for asp in applicable_aspects:
            if asp not in priority_order:
                priority_order.append(asp)

        # 6. Generate Targeted, Evidence-Oriented Research Questions
        site_name = self._extract_site_or_subject(topic)
        candidate_questions_by_aspect = {
            "archaeology": [
                f"What archaeological structures, brick remains, and settlement features have been documented at {site_name} by excavation authorities?",
                f"What formal excavation reports from Archaeological Survey of India or TNSDA describe architectural layers at {site_name}?"
            ],
            "settlement": [
                f"What archaeological evidence supports the interpretation of {site_name} as an organized urban settlement or town?",
                f"What residential and civic layouts have been uncovered in the trenches at {site_name}?"
            ],
            "chronology": [
                f"What scientific radiocarbon dating and AMS dates establish the chronological occupation of {site_name}?",
                f"What chronological periods and BCE dates have been confirmed for the stratigraphy of {site_name}?"
            ],
            "dating": [
                f"What radiocarbon and scientific dating laboratories analyzed charcoal or artifact samples from {site_name}, and what dates were reported?",
                f"What is the oldest scientifically verified date obtained from excavations at {site_name}?"
            ],
            "literacy": [
                f"What evidence exists for early literacy or script usage associated with {site_name} excavations?",
                f"What Tamil-Brahmi potsherd inscriptions have been found at {site_name}, and how are they read and dated?"
            ],
            "inscriptions": [
                f"What personal names, signs, or graffiti marks were inscribed on pottery retrieved from {site_name}?",
                f"How do epigraphists date the Tamil-Brahmi inscribed potsherds recovered from {site_name}?"
            ],
            "material_culture": [
                f"What material culture, including beads, carnelian, spindle whorls, iron, and pottery types, has been catalogued at {site_name}?",
                f"What craft industries, weaving tools, or metallurgical remains were discovered in the excavations at {site_name}?"
            ],
            "cultural_significance": [
                f"What historical and cultural significance does the {site_name} excavation hold for ancient Tamil and Sangam age history?",
                f"How do archaeological findings at {site_name} relate to urbanization along the Vaigai river basin?"
            ]
        }

        # Select targeted questions avoiding duplicates
        recommended_questions: List[str] = []
        # Target top priority aspects first
        for asp in priority_order:
            if len(recommended_questions) >= 5:
                break
            for q_cand in candidate_questions_by_aspect.get(asp, []):
                if len(recommended_questions) >= 5:
                    break
                if not self.is_question_duplicate(q_cand, all_attempted + recommended_questions):
                    recommended_questions.append(q_cand)

        # Ensure at least 3 questions if priority order was short
        if len(recommended_questions) < 3:
            generic_fallback = [
                f"What major archaeological discoveries were published in official excavation reports for {site_name}?",
                f"What chronological evidence and AMS dates have been verified for {site_name}?",
                f"What material remains and inscriptions establish the historical significance of {site_name}?"
            ]
            for fb in generic_fallback:
                if len(recommended_questions) >= 5:
                    break
                if not self.is_question_duplicate(fb, all_attempted + recommended_questions):
                    recommended_questions.append(fb)

        # Expansion required condition:
        # Quality report explicitly failed OR missing core/unique claims OR missing scene requirements
        expansion_required = not cq_passed or missing_claim_count > 0 or missing_core_claim_count > 0 or len(missing_scene_requirements) > 0

        report = ResearchGapReport(
            topic=topic,
            missing_claim_count=missing_claim_count,
            missing_core_claim_count=missing_core_claim_count,
            missing_topic_aspects=missing_topic_aspects,
            missing_scene_requirements=missing_scene_requirements,
            weak_claims=weak_claims,
            failed_claims=failed_claims,
            recommended_research_questions=recommended_questions,
            recommended_source_types=AUTHORITATIVE_SOURCE_TYPES,
            priority_order=priority_order,
            expansion_required=expansion_required,
            attempted_questions=all_attempted,
            expansion_number=expansion_number
        )

        return report
