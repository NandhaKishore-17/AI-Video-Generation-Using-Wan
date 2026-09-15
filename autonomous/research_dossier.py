"""
autonomous/research_dossier.py - Structured Research Dossier & Conflict Detection.

Assembles retrieved evidence into a structured, provenance-preserving research dossier.
Detects potential numerical/chronological conflicts without resolving them, and formats
preliminary findings with non-committal language and mandatory chunk references.
"""

import re
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set

from autonomous.source_retriever import SourceMetadata
from autonomous.research_processor import EvidenceChunk
from autonomous.research_planner import ResearchPlan

logger = logging.getLogger("autonomous.research_dossier")


@dataclass
class PreliminaryFinding:
    """A preliminary finding synthesized from evidence. NEVER marked as VERIFIED in Phase 4."""
    finding_id: str
    statement: str  # e.g. "Evidence documented by Archaeological Survey of India indicates..."
    topic_aspect: str
    supporting_chunk_ids: List[str]
    supporting_source_ids: List[str]
    credibility_levels: List[int]
    status: str = "EVIDENCE_SUPPORTED"  # "EVIDENCE_SUPPORTED", "PROPOSED", "UNCERTAIN"
    # Correction 7: Indirect recovery of independence and provenance
    independence_groups: List[str] = field(default_factory=list)
    source_origins: List[str] = field(default_factory=list)
    evidence_weight_classes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceConflict:
    """Recorded conflict between independent sources. Phase 4 notes it; Phase 5 verifies it."""
    conflict_id: str
    aspect: str  # "chronology", "infrastructure_dimensions", "origin_dating"
    statement_a: str
    source_a_id: str
    source_a_title: str
    statement_b: str
    source_b_id: str
    source_b_title: str
    conflict_nature: str
    resolution_status: str = "UNRESOLVED_PENDING_PHASE5"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchDossier:
    """Comprehensive, source-backed research dossier produced at the end of Phase 4."""
    episode_id: str
    topic: str
    generated_at: str
    research_plan: Dict[str, Any]
    sources: List[Dict[str, Any]]
    evidence_chunks: List[Dict[str, Any]]
    research_questions: List[str]
    question_evidence: List[Dict[str, Any]]
    preliminary_findings: List[Dict[str, Any]]
    conflicting_evidence: List[Dict[str, Any]]
    unresolved_questions: List[str]
    source_statistics: Dict[str, Any]
    research_status: str  # "RESEARCH_COMPLETE", "INSUFFICIENT_EVIDENCE", "REVIEW_REQUIRED"
    timing_and_resources: Dict[str, Any] = field(default_factory=dict)
    quality_gate_passed: Optional[bool] = None
    quality_gate_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



class ResearchDossierBuilder:
    """Compiles sources, chunks, and retrieved evidence into a validated ResearchDossier."""

    @staticmethod
    def compute_source_statistics(sources: List[SourceMetadata], chunks: List[EvidenceChunk]) -> Dict[str, Any]:
        """
        Compute source diversity, unique publishers, domains, and credibility tiers.
        Correction 1, 2, 4, 5: Rigorous source independence, origin, and evidence weight tracking.
        """
        publishers = set()
        domains = set()
        independence_groups = set()
        external_groups = set()
        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        source_type_counts: Dict[str, int] = {}
        evidence_weight_counts: Dict[str, int] = {}

        external_count = 0
        local_count = 0
        cached_external_count = 0
        high_quality_external = 0
        reference_only = 0

        for s in sources:
            origin = getattr(s, "source_origin", "external")
            group = getattr(s, "independence_group", "") or s.domain or ""
            role = getattr(s, "source_role", "primary_evidence")
            weight = getattr(s, "evidence_weight_class", "PRIMARY")
            s_type = getattr(s, "source_type", "web")

            if s.publisher:
                publishers.add(s.publisher.lower().strip())
            if s.domain:
                domains.add(s.domain.lower().strip())
            if group:
                independence_groups.add(group.lower().strip())

            # Source types & weights
            source_type_counts[s_type] = source_type_counts.get(s_type, 0) + 1
            evidence_weight_counts[weight] = evidence_weight_counts.get(weight, 0) + 1

            # Origin accounting
            if origin == "local":
                local_count += 1
            elif origin == "cached_external":
                cached_external_count += 1
                external_count += 1
                if group and group not in ("local_filesystem", "local_project", "unknown_group"):
                    external_groups.add(group.lower().strip())
                if weight in ("PRIMARY", "HIGH"):
                    high_quality_external += 1
            else:  # external
                external_count += 1
                if group and group not in ("local_filesystem", "local_project", "unknown_group"):
                    external_groups.add(group.lower().strip())
                if weight in ("PRIMARY", "HIGH"):
                    high_quality_external += 1

            if role == "discovery_reference" or weight == "REFERENCE":
                reference_only += 1

            tier = s.credibility_tier if 1 <= s.credibility_tier <= 5 else 3
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        high_quality_count = tier_counts.get(1, 0) + tier_counts.get(2, 0)
        external_independent_sources = len(external_groups)

        return {
            "total_sources": len(sources),
            "external_sources": external_count,
            "local_sources": local_count,
            "cached_external_sources": cached_external_count,
            "external_independent_sources": external_independent_sources,
            "unique_publishers": len(publishers),
            "publishers_list": sorted(list(publishers)),
            "unique_domains": len(domains),
            "domains_list": sorted(list(domains)),
            "unique_independence_groups": len(independence_groups),
            "independence_groups_list": sorted(list(independence_groups)),
            "high_quality_external_sources": high_quality_external,
            "high_quality_sources_count": high_quality_count,  # backward compatibility
            "reference_only_sources": reference_only,
            "source_type_counts": source_type_counts,
            "credibility_tier_counts": tier_counts,
            "tier_distribution": tier_counts,  # backward compatibility
            "evidence_weight_counts": evidence_weight_counts,
            "total_chunks": len(chunks),
            "has_independent_sources": external_independent_sources >= 2 or len(publishers) >= 2 or len(domains) >= 2
        }

    @staticmethod
    def detect_conflicts(question_evidence: List[Dict[str, Any]], sources: List[SourceMetadata]) -> List[EvidenceConflict]:
        """
        Scan retrieved passages for numerical or chronological discrepancies across independent sources.
        Records potential conflict without picking a winner (Correction 8).
        """
        conflicts = []
        source_map = {s.source_id: s for s in sources}
        conflict_counter = 0

        # Pattern for centuries / years BCE/CE
        date_pattern = re.compile(r"(\b\d{1,4}\s*(?:BCE|BC|CE|AD)\b|\b\d{1,2}(?:st|nd|rd|th)?\s+century\s+(?:BCE|BC|CE|AD)\b)", re.IGNORECASE)

        for q_item in question_evidence:
            evidence_items = q_item.get("retrieved_evidence", [])
            if len(evidence_items) < 2:
                continue

            extracted_dates = []
            for ev in evidence_items:
                text = ev.get("text", "")
                s_id = ev.get("source_id", "")
                matches = date_pattern.findall(text)
                for m in matches:
                    extracted_dates.append((m.strip(), s_id, text[:150], ev.get("title", "")))

            # Check if different sources mention distinctly different dates for the same question
            if len(extracted_dates) >= 2:
                for i in range(len(extracted_dates)):
                    for j in range(i + 1, len(extracted_dates)):
                        d1, sid1, snip1, title1 = extracted_dates[i]
                        d2, sid2, snip2, title2 = extracted_dates[j]
                        if sid1 != sid2 and d1.lower() != d2.lower():
                            conflict_counter += 1
                            conflicts.append(EvidenceConflict(
                                conflict_id=f"conf_{conflict_counter:03d}",
                                aspect="chronological_variation",
                                statement_a=f"Reports: '{d1}' in passage context: \"{snip1}\"",
                                source_a_id=sid1,
                                source_a_title=title1,
                                statement_b=f"Reports: '{d2}' in passage context: \"{snip2}\"",
                                source_b_id=sid2,
                                source_b_title=title2,
                                conflict_nature=f"Different chronological dates cited across independent sources ({d1} vs {d2})",
                                resolution_status="UNRESOLVED_PENDING_PHASE5"
                            ))
                            break

        return conflicts

    @staticmethod
    def generate_preliminary_findings(
        question_evidence: List[Dict[str, Any]],
        topic: str
    ) -> List[PreliminaryFinding]:
        """
        Synthesize preliminary findings with mandatory chunk-level provenance.
        Findings use non-committal language ("Evidence indicates...", "Source A reports...").
        """
        findings = []
        finding_idx = 0

        for q_item in question_evidence:
            question = q_item.get("question", "")
            evidence = q_item.get("retrieved_evidence", [])
            if not evidence:
                continue

            # Pick top chunk
            top_ev = evidence[0]
            chunk_id = top_ev.get("chunk_id")
            source_id = top_ev.get("source_id")
            tier = top_ev.get("credibility_tier", 3)
            pub = top_ev.get("publisher") or top_ev.get("title") or "Source"

            # Extract an informative sentence from the chunk
            text = top_ev.get("text", "")
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]

            if sentences:
                finding_idx += 1
                lead_sentence = sentences[0]
                # Non-committal phrasing
                statement = f"Documentation from {pub} indicates: \"{lead_sentence}\""

                all_chunks = [e["chunk_id"] for e in evidence if "chunk_id" in e]
                all_sources = list({e["source_id"] for e in evidence if "source_id" in e})
                all_tiers = [e.get("credibility_tier", 3) for e in evidence]
                all_groups = list({e.get("independence_group", "") for e in evidence if e.get("independence_group")})
                all_origins = list({e.get("source_origin", "external") for e in evidence if e.get("source_origin")})
                all_weights = [e.get("evidence_weight_class", "PRIMARY") for e in evidence if e.get("evidence_weight_class")]

                findings.append(PreliminaryFinding(
                    finding_id=f"find_{finding_idx:03d}",
                    statement=statement,
                    topic_aspect=question,
                    supporting_chunk_ids=all_chunks[:3],
                    supporting_source_ids=all_sources[:3],
                    credibility_levels=all_tiers[:3],
                    status="EVIDENCE_SUPPORTED",
                    independence_groups=all_groups[:3],
                    source_origins=all_origins[:3],
                    evidence_weight_classes=all_weights[:3]
                ))

        return findings

    @classmethod
    def build_dossier(
        cls,
        episode_id: str,
        plan: ResearchPlan,
        sources: List[SourceMetadata],
        chunks: List[EvidenceChunk],
        question_evidence: List[Dict[str, Any]],
        research_status: str = "RESEARCH_COMPLETE"
    ) -> ResearchDossier:
        """Construct a complete ResearchDossier adhering to the required schema."""
        stats = cls.compute_source_statistics(sources, chunks)
        conflicts = cls.detect_conflicts(question_evidence, sources)
        findings = cls.generate_preliminary_findings(question_evidence, plan.topic)

        # Questions with zero retrieved evidence
        unresolved = [
            qe["question"] for qe in question_evidence
            if len(qe.get("retrieved_evidence", [])) == 0
        ]

        return ResearchDossier(
            episode_id=episode_id,
            topic=plan.topic,
            generated_at=datetime.now(timezone.utc).isoformat(),
            research_plan=plan.to_dict(),
            sources=[s.to_dict() for s in sources],
            evidence_chunks=[c.to_dict() for c in chunks],
            research_questions=plan.research_questions,
            question_evidence=question_evidence,
            preliminary_findings=[f.to_dict() for f in findings],
            conflicting_evidence=[c.to_dict() for c in conflicts],
            unresolved_questions=unresolved,
            source_statistics=stats,
            research_status=research_status
        )
