"""
autonomous/claim_models.py - Domain Models for Phase 5 Claim Verification & Script Generation.

Defines enums, structured evidence matches, claims, epistemic states, content plans,
and validation reports for autonomous, deterministic historical fact-checking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional


class ClaimType(str, Enum):
    """Categorical classification of factual claim nature."""
    HISTORICAL_FACT = "HISTORICAL_FACT"
    ARCHAEOLOGICAL_EVIDENCE = "ARCHAEOLOGICAL_EVIDENCE"
    DATING_CHRONOLOGY = "DATING_CHRONOLOGY"
    GEOGRAPHICAL_LOCATION = "GEOGRAPHICAL_LOCATION"
    HISTORICAL_FIGURE = "HISTORICAL_FIGURE"
    MATERIAL_CULTURE = "MATERIAL_CULTURE"
    CULTURAL_PRACTICE = "CULTURAL_PRACTICE"
    SUPERLATIVE_ASSERTION = "SUPERLATIVE_ASSERTION"  # "first", "oldest", "largest", "only"
    STATISTICAL_MEASUREMENT = "STATISTICAL_MEASUREMENT"


class ClaimImportance(str, Enum):
    """Sensitivity and burden of proof required for the claim."""
    CRITICAL = "CRITICAL"    # Superlatives, founding dates, central historical pillars
    HIGH = "HIGH"            # Specific archaeological findings, script presence, key rulers
    MEDIUM = "MEDIUM"        # General cultural practices, trade goods, urban layout
    LOW = "LOW"              # Incidental background context


class EpistemicStatus(str, Enum):
    """Explicit historical epistemic certainty level."""
    KNOWN_FACT = "KNOWN_FACT"                      # Broadly documented by peer-reviewed archaeological consensus
    STRONGLY_SUPPORTED = "STRONGLY_SUPPORTED"      # Solid multi-source or primary institutional report
    LIKELY = "LIKELY"                              # Plausible secondary evidence with slight gaps
    DEBATED = "DEBATED"                            # Competing academic dates or rival interpretations
    PROPOSED_THEORY = "PROPOSED_THEORY"            # Scholarly hypothesis not yet universally accepted
    TRADITION_OR_LEGEND = "TRADITION_OR_LEGEND"    # Folkloric / literary tradition without material proof
    UNCERTAIN = "UNCERTAIN"                        # Weak, ambiguous, or contested data
    UNSUPPORTED = "UNSUPPORTED"                    # Zero authoritative backing or contradicted


class EvidenceSupportType(str, Enum):
    """Relationship between a candidate claim and a specific evidence chunk."""
    DIRECT_SUPPORT = "DIRECT_SUPPORT"          # Chunk explicitly asserts or substantiates the claim predicate
    INDIRECT_SUPPORT = "INDIRECT_SUPPORT"      # Chunk strongly entails or closely corroborates the claim
    CONTEXTUAL_ONLY = "CONTEXTUAL_ONLY"        # Chunk discusses topic domain but does not prove the specific assertion
    CONTRADICTORY = "CONTRADICTORY"            # Chunk asserts diverging numbers, opposite facts, or rival chronology
    IRRELEVANT = "IRRELEVANT"                  # Low semantic similarity or unrelated proposition


class ClaimClassification(str, Enum):
    """Actionable classification for content planning and script authorization."""
    VERIFIED_FACT = "VERIFIED_FACT"                # Authorized for direct factual narration
    SUPPORTED_HYPOTHESIS = "SUPPORTED_HYPOTHESIS"  # Authorized ONLY with epistemic framing ("Excavations suggest...")
    UNRESOLVED_DEBATE = "UNRESOLVED_DEBATE"        # Authorized ONLY as an active historical debate
    UNVERIFIED_CLAIM = "UNVERIFIED_CLAIM"          # STRICTLY FORBIDDEN from script factual assertions
    REJECTED = "REJECTED"                          # Contradicted by authoritative evidence; excluded


class ClaimRelevanceClass(str, Enum):
    """Relevance and centrality of claim to the episode's subject matter."""
    CORE_TOPIC = "CORE_TOPIC"                  # Directly concerns central subject/excavation/findings
    SUPPORTING_CONTEXT = "SUPPORTING_CONTEXT"  # Administrative history, institutional foundation, methodology
    BACKGROUND = "BACKGROUND"                  # Regional geography, general era background, broad context


class EntityType(str, Enum):
    """Categorical classification of named entity tokens to distinguish meaningful historical entities."""
    PERSON = "PERSON"
    PLACE = "PLACE"
    ORGANIZATION = "ORGANIZATION"
    ARCHAEOLOGICAL_SITE = "ARCHAEOLOGICAL_SITE"
    DYNASTY = "DYNASTY"
    CULTURAL_OBJECT = "CULTURAL_OBJECT"
    HISTORICAL_PERIOD = "HISTORICAL_PERIOD"
    COMMON_WORD = "COMMON_WORD"


class ScriptClaimStatus(str, Enum):
    """Validation outcome for a re-extracted claim from the generated script."""
    MATCHED_VERIFIED_CLAIM = "MATCHED_VERIFIED_CLAIM"
    MATCHED_CAUTIOUS_CLAIM = "MATCHED_CAUTIOUS_CLAIM"
    MATCHED_DEBATED_CLAIM = "MATCHED_DEBATED_CLAIM"
    NEW_UNSUPPORTED_CLAIM = "NEW_UNSUPPORTED_CLAIM"
    CONTRADICTED_CLAIM = "CONTRADICTED_CLAIM"
    OVERSTATED_CLAIM = "OVERSTATED_CLAIM"              # Epistemic inflation ("suggests" -> "proves")
    NUMERIC_MISMATCH = "NUMERIC_MISMATCH"              # Altered count, distance, or quantity
    DATE_MISMATCH = "DATE_MISMATCH"                    # Altered century or year
    ENTITY_MISMATCH = "ENTITY_MISMATCH"                # Hallucinated person, dynasty, or location
    UNCERTAINTY_REMOVED = "UNCERTAINTY_REMOVED"        # Stated legend/theory as proven fact
    UNSUPPORTED_QUOTATION = "UNSUPPORTED_QUOTATION"    # Fabricated quote


@dataclass
class ClaimEvidenceMatch:
    """Detailed record of how a specific evidence chunk evaluates against a claim."""
    chunk_id: str
    source_id: str
    support_type: EvidenceSupportType
    similarity_score: float
    publisher: Optional[str] = None
    domain: Optional[str] = None
    independence_group: str = "unknown_group"
    source_role: str = "primary_evidence"
    evidence_weight_class: str = "PRIMARY"
    source_origin: str = "external"
    credibility_tier: int = 3
    snippet: str = ""
    matched_entities: List[str] = field(default_factory=list)
    matched_dates: List[str] = field(default_factory=list)
    matched_numbers: List[str] = field(default_factory=list)
    contradiction_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "support_type": self.support_type.value,
            "similarity_score": round(self.similarity_score, 4),
            "publisher": self.publisher,
            "domain": self.domain,
            "independence_group": self.independence_group,
            "source_role": self.source_role,
            "evidence_weight_class": self.evidence_weight_class,
            "source_origin": self.source_origin,
            "credibility_tier": self.credibility_tier,
            "snippet": self.snippet[:200] if self.snippet else "",
            "text": self.snippet,
            "matched_entities": self.matched_entities,
            "matched_dates": self.matched_dates,
            "matched_numbers": self.matched_numbers,
            "contradiction_reason": self.contradiction_reason,
        }


@dataclass
class VerifiedClaim:
    """A single atomic historical claim with deterministic confidence and provenance."""
    claim_id: str
    statement: str
    normalized_statement: str
    claim_type: ClaimType
    importance: ClaimImportance
    topic_aspect: str
    episode_id: str

    # Quantitative support metrics
    evidence_chunk_count: int = 0
    source_count: int = 0
    external_source_count: int = 0
    independent_external_source_count: int = 0
    independent_non_ref_external_source_count: int = 0
    high_quality_source_count: int = 0
    direct_support_count: int = 0
    contextual_support_count: int = 0
    contradicting_source_count: int = 0

    # Evaluated outcomes
    confidence_score: float = 0.0
    confidence_breakdown: Dict[str, Any] = field(default_factory=dict)
    classification: ClaimClassification = ClaimClassification.UNVERIFIED_CLAIM
    epistemic_status: EpistemicStatus = EpistemicStatus.UNCERTAIN
    epistemic_constraint: str = "FORBIDDEN"  # "DIRECT_FACT", "HEDGED_FRAMING", "DEBATE_ONLY", "FORBIDDEN"
    relevance_class: ClaimRelevanceClass = ClaimRelevanceClass.SUPPORTING_CONTEXT

    # Provenance tracking
    supporting_chunk_ids: List[str] = field(default_factory=list)
    supporting_source_ids: List[str] = field(default_factory=list)
    contradicting_chunk_ids: List[str] = field(default_factory=list)
    independence_groups: List[str] = field(default_factory=list)
    source_origins: List[str] = field(default_factory=list)
    evidence_weight_classes: List[str] = field(default_factory=list)
    extracted_entities: List[str] = field(default_factory=list)
    extracted_dates: List[str] = field(default_factory=list)
    extracted_numbers: List[str] = field(default_factory=list)
    evidence_matches: List[ClaimEvidenceMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "normalized_statement": self.normalized_statement,
            "claim_type": self.claim_type.value,
            "importance": self.importance.value,
            "topic_aspect": self.topic_aspect,
            "episode_id": self.episode_id,
            "evidence_chunk_count": self.evidence_chunk_count,
            "source_count": self.source_count,
            "external_source_count": self.external_source_count,
            "independent_external_source_count": self.independent_external_source_count,
            "independent_non_ref_external_source_count": self.independent_non_ref_external_source_count,
            "high_quality_source_count": self.high_quality_source_count,
            "direct_support_count": self.direct_support_count,
            "contextual_support_count": self.contextual_support_count,
            "contradicting_source_count": self.contradicting_source_count,
            "confidence_score": round(self.confidence_score, 4),
            "confidence_breakdown": self.confidence_breakdown,
            "classification": self.classification.value,
            "epistemic_status": self.epistemic_status.value,
            "epistemic_constraint": self.epistemic_constraint,
            "relevance_class": self.relevance_class.value if isinstance(self.relevance_class, ClaimRelevanceClass) else str(self.relevance_class),
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "supporting_source_ids": self.supporting_source_ids,
            "contradicting_chunk_ids": self.contradicting_chunk_ids,
            "independence_groups": self.independence_groups,
            "source_origins": self.source_origins,
            "evidence_weight_classes": self.evidence_weight_classes,
            "extracted_entities": self.extracted_entities,
            "extracted_dates": self.extracted_dates,
            "extracted_numbers": self.extracted_numbers,
            "evidence_matches": [m.to_dict() for m in self.evidence_matches],
        }


@dataclass
class SceneContentPlan:
    """Factual blueprint for an individual storyboard scene."""
    scene_id: int
    scene_type: str  # "host_vlog" or "broll_motion"
    shot_type: str   # "PORTRAIT", "WIDE_ESTABLISHING", "ARCHITECTURE", etc.
    badge: str
    focus_theme: str
    mandatory_claim_ids: List[str] = field(default_factory=list)
    allowed_claim_ids: List[str] = field(default_factory=list)
    forbidden_assertions: List[str] = field(default_factory=list)
    epistemic_framing_required: bool = False
    required_framing_phrases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "shot_type": self.shot_type,
            "badge": self.badge,
            "focus_theme": self.focus_theme,
            "mandatory_claim_ids": self.mandatory_claim_ids,
            "allowed_claim_ids": self.allowed_claim_ids,
            "forbidden_assertions": self.forbidden_assertions,
            "epistemic_framing_required": self.epistemic_framing_required,
            "required_framing_phrases": self.required_framing_phrases,
        }


@dataclass
class FactCheckedContentPlan:
    """Rigorous 4-scene content plan constraining script generation to verified facts."""
    episode_id: str
    topic: str
    category: str
    scenes: List[SceneContentPlan] = field(default_factory=list)
    allowed_claim_ids: List[str] = field(default_factory=list)
    cautious_claim_ids: List[str] = field(default_factory=list)
    debated_claim_ids: List[str] = field(default_factory=list)
    theory_claim_ids: List[str] = field(default_factory=list)
    tradition_claim_ids: List[str] = field(default_factory=list)
    forbidden_claim_ids: List[str] = field(default_factory=list)

    # Authorized vocabularies
    allowed_entities: List[str] = field(default_factory=list)
    allowed_dates: List[str] = field(default_factory=list)
    allowed_numbers: List[str] = field(default_factory=list)
    verified_quotations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "topic": self.topic,
            "category": self.category,
            "scenes": [s.to_dict() for s in self.scenes],
            "allowed_claim_ids": self.allowed_claim_ids,
            "cautious_claim_ids": self.cautious_claim_ids,
            "debated_claim_ids": self.debated_claim_ids,
            "theory_claim_ids": self.theory_claim_ids,
            "tradition_claim_ids": self.tradition_claim_ids,
            "forbidden_claim_ids": self.forbidden_claim_ids,
            "allowed_entities": self.allowed_entities,
            "allowed_dates": self.allowed_dates,
            "allowed_numbers": self.allowed_numbers,
            "verified_quotations": self.verified_quotations,
        }


@dataclass
class ScriptClaim:
    """Factual claim re-extracted from generated script narration."""
    claim_index: int
    scene_id: int
    language: str  # "en" or "ta"
    source_sentence: str
    extracted_statement: str
    extracted_entities: List[str] = field(default_factory=list)
    extracted_dates: List[str] = field(default_factory=list)
    extracted_numbers: List[str] = field(default_factory=list)
    is_superlative: bool = False
    is_quotation: bool = False
    status: ScriptClaimStatus = ScriptClaimStatus.NEW_UNSUPPORTED_CLAIM
    matched_claim_id: Optional[str] = None
    violation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_index": self.claim_index,
            "scene_id": self.scene_id,
            "language": self.language,
            "source_sentence": self.source_sentence,
            "extracted_statement": self.extracted_statement,
            "extracted_entities": self.extracted_entities,
            "extracted_dates": self.extracted_dates,
            "extracted_numbers": self.extracted_numbers,
            "is_superlative": self.is_superlative,
            "is_quotation": self.is_quotation,
            "status": self.status.value,
            "matched_claim_id": self.matched_claim_id,
            "violation_reason": self.violation_reason,
        }


@dataclass
class ScriptValidationReport:
    """Comprehensive validation outcome for generated narration."""
    episode_id: str
    passed: bool
    total_script_claims: int = 0
    matched_verified_count: int = 0
    matched_cautious_count: int = 0
    matched_debated_count: int = 0
    new_unsupported_count: int = 0
    contradicted_count: int = 0
    overstated_count: int = 0
    numeric_mismatch_count: int = 0
    date_mismatch_count: int = 0
    entity_mismatch_count: int = 0
    epistemic_violation_count: int = 0
    quotation_violation_count: int = 0
    cross_language_mismatch_count: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    claims: List[ScriptClaim] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "passed": self.passed,
            "total_script_claims": self.total_script_claims,
            "matched_verified_count": self.matched_verified_count,
            "matched_cautious_count": self.matched_cautious_count,
            "matched_debated_count": self.matched_debated_count,
            "new_unsupported_count": self.new_unsupported_count,
            "contradicted_count": self.contradicted_count,
            "overstated_count": self.overstated_count,
            "numeric_mismatch_count": self.numeric_mismatch_count,
            "date_mismatch_count": self.date_mismatch_count,
            "entity_mismatch_count": self.entity_mismatch_count,
            "epistemic_violation_count": self.epistemic_violation_count,
            "quotation_violation_count": self.quotation_violation_count,
            "cross_language_mismatch_count": self.cross_language_mismatch_count,
            "violations": self.violations,
            "claims": [c.to_dict() for c in self.claims],
        }


@dataclass
class ContentQualityReport:
    """Deterministic validation outcome of Phase 5.1 Content Quality & Topic Relevance."""
    episode_id: str
    topic: str
    passed: bool
    minimum_unique_claims: int = 3
    actual_unique_claims: int = 0
    minimum_unique_topic_aspects: int = 2
    actual_unique_topic_aspects: int = 0
    minimum_core_claims: int = 2
    core_claim_count: int = 0
    supporting_claim_count: int = 0
    background_claim_count: int = 0
    repeated_claim_groups: Dict[str, List[str]] = field(default_factory=dict)
    maximum_claim_scene_reuse: int = 0
    scene_quality_results: List[Dict[str, Any]] = field(default_factory=list)
    topic_relevance_results: Dict[str, Any] = field(default_factory=dict)
    content_quality_status: str = "PENDING"  # "PASSED", "CONTENT_INSUFFICIENT", "CLAIM_OVERUSED", "INSUFFICIENT_EVIDENCE_FOR_SCENE"
    research_expansion_recommended: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "topic": self.topic,
            "passed": self.passed,
            "minimum_unique_claims": self.minimum_unique_claims,
            "actual_unique_claims": self.actual_unique_claims,
            "minimum_unique_topic_aspects": self.minimum_unique_topic_aspects,
            "actual_unique_topic_aspects": self.actual_unique_topic_aspects,
            "minimum_core_claims": self.minimum_core_claims,
            "core_claim_count": self.core_claim_count,
            "supporting_claim_count": self.supporting_claim_count,
            "background_claim_count": self.background_claim_count,
            "repeated_claim_groups": self.repeated_claim_groups,
            "maximum_claim_scene_reuse": self.maximum_claim_scene_reuse,
            "scene_quality_results": self.scene_quality_results,
            "topic_relevance_results": self.topic_relevance_results,
            "content_quality_status": self.content_quality_status,
            "research_expansion_recommended": self.research_expansion_recommended,
            "failure_reasons": self.failure_reasons,
        }

