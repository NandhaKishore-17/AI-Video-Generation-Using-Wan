"""
autonomous/claim_verifier.py - Deterministic Claim Verification & Confidence Engine.

Evaluates candidate historical claims against Qdrant vector evidence and dossier chunks.
Strictly distinguishes direct from contextual evidence, computes conservative source independence,
detects contradictions across dates/numbers/interpretations, and calculates mathematically
reproducible confidence scores without LLM probability guesses.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from autonomous.claim_models import (
    ClaimType,
    ClaimImportance,
    EpistemicStatus,
    EvidenceSupportType,
    ClaimClassification,
    ClaimEvidenceMatch,
    ClaimRelevanceClass,
    VerifiedClaim
)
from autonomous.evidence_store import EvidenceStore

logger = logging.getLogger("autonomous.claim_verifier")


class ClaimVerifier:
    """Performs deterministic, provenance-backed claim verification."""

    def __init__(
        self,
        evidence_store: Optional[EvidenceStore] = None,
        min_retrieval_score: float = 0.35,
        max_chunks_per_claim: int = 5
    ):
        self.evidence_store = evidence_store or EvidenceStore()
        self.min_retrieval_score = min_retrieval_score
        self.max_chunks_per_claim = max_chunks_per_claim

    @staticmethod
    def classify_evidence_support(
        claim: VerifiedClaim,
        chunk_text: str,
        similarity_score: float,
        chunk_entities: List[str],
        chunk_dates: List[str],
        chunk_numbers: List[str]
    ) -> Tuple[EvidenceSupportType, Optional[str]]:
        """
        Classify the exact relationship between a claim and an evidence passage:
        - DIRECT_SUPPORT: Explicitly describes the asserted claim predicate and key entities/dates
        - INDIRECT_SUPPORT: Closely corroborates without exact full predicate match
        - CONTEXTUAL_ONLY: General background without confirming the specific assertion
        - CONTRADICTORY: Directly asserts conflicting dates, numbers, or opposite polarity
        - IRRELEVANT: Poor relevance or mismatched entities
        """
        if similarity_score < 0.30:
            return (EvidenceSupportType.IRRELEVANT, None)

        claim_text = claim.statement.lower()
        chunk_lower = chunk_text.lower()

        # Check for contradictions first:
        # 1. Date contradictions (e.g. 6th century BCE vs 3rd century BCE, 600 BCE vs 300 BCE)
        claim_dates = [d.lower() for d in claim.extracted_dates]
        if claim_dates and chunk_dates:
            for cd in claim_dates:
                for ed in chunk_dates:
                    if cd != ed:
                        # If both mention different centuries or different years for the same site
                        c_num = re.findall(r"\d+", cd)
                        e_num = re.findall(r"\d+", ed)
                        if c_num and e_num and c_num[0] != e_num[0]:
                            if any(k in chunk_lower for k in ["dating", "bce", "ce", "century", "strata"]):
                                return (EvidenceSupportType.CONTRADICTORY, f"Chronological conflict: claim asserts '{cd}' but evidence indicates '{ed}'")

        # 2. Numeric count / quantity contradictions
        claim_nums = set(claim.extracted_numbers)
        chunk_nums = set(chunk_numbers)
        if claim_nums and chunk_nums and not (claim_nums & chunk_nums):
            # If both mention quantities for the same artifact/population topic
            if any(w in claim_text for w in ["people", "artifacts", "inscriptions", "structures", "acres"]):
                if any(w in chunk_lower for w in ["people", "artifacts", "inscriptions", "structures", "acres"]):
                    return (EvidenceSupportType.CONTRADICTORY, f"Numerical divergence: claim specifies {list(claim_nums)} vs evidence {list(chunk_nums)}")

        # 3. Direct polarity negation
        negation_terms = ["no evidence", "refuted", "disproved", "contrary to", "fictional", "mythical", "no sign of"]
        for neg in negation_terms:
            if neg in chunk_lower:
                for ent in claim.extracted_entities:
                    if ent in chunk_lower:
                        return (EvidenceSupportType.CONTRADICTORY, f"DIRECT_DISPROOF: Explicit evidential negation ('{neg}') regarding '{ent}'")

        # 4. Evaluate Direct vs Contextual Support
        # Check entity and key predicate presence
        matched_ents = [e for e in claim.extracted_entities if e in chunk_lower]
        has_dates = bool(set(claim.extracted_dates) & set(chunk_dates)) if claim.extracted_dates else True
        has_nums = bool(set(claim.extracted_numbers) & set(chunk_numbers)) if claim.extracted_numbers else True

        # Extract core action/noun tokens from statement
        statement_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", claim_text))
        chunk_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", chunk_lower))
        word_overlap = len(statement_words & chunk_words) / max(1, len(statement_words))

        stemmed_statement = {w.rstrip("s") for w in statement_words if len(w) > 3}
        stemmed_chunk = {w.rstrip("s") for w in chunk_words if len(w) > 3}
        stemmed_overlap = len(stemmed_statement & stemmed_chunk) / max(1, len(stemmed_statement))
        effective_overlap = max(word_overlap, stemmed_overlap)

        # Direct support requires high overlap and matching entity/date
        if (effective_overlap >= 0.40 and (matched_ents or not claim.extracted_entities) and has_dates) or \
           (effective_overlap >= 0.25 and matched_ents and claim.extracted_dates and set(claim.extracted_dates) & set(chunk_dates)):
            return (EvidenceSupportType.DIRECT_SUPPORT, None)
        elif effective_overlap >= 0.20 or matched_ents:
            if similarity_score >= 0.50:
                return (EvidenceSupportType.INDIRECT_SUPPORT, None)
            return (EvidenceSupportType.CONTEXTUAL_ONLY, None)

        return (EvidenceSupportType.CONTEXTUAL_ONLY if similarity_score >= 0.35 else EvidenceSupportType.IRRELEVANT, None)

    @classmethod
    def calculate_deterministic_confidence(
        cls,
        matches: List[ClaimEvidenceMatch],
        importance: ClaimImportance,
        has_superlative: bool
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates reproducible evidence-support confidence score in [0.0, 1.0].
        Adheres to:
        - Separate accounting for direct vs contextual evidence
        - Conservative source independence (multiple chunks on same publisher = 1 source)
        - Wikipedia reference cap (0.65 ceiling)
        - Local project document cap (0.50 ceiling)
        - Unresolved contradiction penalty (-0.30)
        - Superlative requirement for strong direct multi-source proof
        """
        direct_matches = [m for m in matches if m.support_type == EvidenceSupportType.DIRECT_SUPPORT]
        indirect_matches = [m for m in matches if m.support_type == EvidenceSupportType.INDIRECT_SUPPORT]
        contextual_matches = [m for m in matches if m.support_type == EvidenceSupportType.CONTEXTUAL_ONLY]
        contradictory_matches = [m for m in matches if m.support_type == EvidenceSupportType.CONTRADICTORY]

        # 1. Source Quality Scoring (derived from best supporting sources)
        weight_points = {"PRIMARY": 0.40, "HIGH": 0.25, "MODERATE": 0.15, "REFERENCE": 0.05, "LOW": 0.00, "VERY_LOW": 0.00}
        tier_points = {1: 0.40, 2: 0.25, 3: 0.15, 4: 0.05, 5: 0.00}

        best_direct_quality = 0.0
        for m in direct_matches:
            q = max(weight_points.get(m.evidence_weight_class, 0.05), tier_points.get(m.credibility_tier, 0.05))
            if q > best_direct_quality:
                best_direct_quality = q

        best_indirect_quality = 0.0
        for m in indirect_matches:
            q = max(weight_points.get(m.evidence_weight_class, 0.05), tier_points.get(m.credibility_tier, 0.05)) * 0.70
            if q > best_indirect_quality:
                best_indirect_quality = q

        source_quality_score = max(best_direct_quality, best_indirect_quality)

        # 2. Direct Support Scoring
        # Direct support provides up to 0.35 base support
        direct_support_score = 0.0
        if direct_matches:
            direct_support_score = min(0.35, 0.20 + (len(direct_matches) * 0.05))
        elif indirect_matches:
            direct_support_score = min(0.20, len(indirect_matches) * 0.06)
        elif contextual_matches:
            direct_support_score = 0.05

        # 3. Source Independence
        # Unique external independence groups among supporting matches
        supporting_matches = direct_matches + indirect_matches
        external_groups = {
            m.independence_group for m in supporting_matches
            if m.source_origin in ("external", "cached_external") and m.independence_group not in ("local_filesystem", "unknown_group")
        }
        external_indep_count = len(external_groups)

        independence_score = 0.0
        if external_indep_count >= 2:
            independence_score = 0.20
        elif external_indep_count == 1:
            independence_score = 0.08

        # 4. Evidence Specificity (Entity & similarity boost)
        avg_sim = sum(m.similarity_score for m in supporting_matches) / max(1, len(supporting_matches)) if supporting_matches else 0.0
        evidence_specificity = round(min(0.15, avg_sim * 0.20), 4)

        # Base confidence calculation
        raw_score = source_quality_score + direct_support_score + independence_score + evidence_specificity

        # 5. Penalties & Caps
        contradiction_penalty = 0.0
        if contradictory_matches:
            contradiction_penalty = 0.35

        reference_only_penalty = 0.0
        all_origins = {m.source_origin for m in supporting_matches}
        all_weights = {m.evidence_weight_class for m in supporting_matches}
        all_groups = {m.independence_group for m in supporting_matches}

        # Wikipedia-only check: if only reference/wikipedia
        is_wiki_only = bool(supporting_matches and all(
            m.evidence_weight_class == "REFERENCE" or m.independence_group == "wikipedia.org"
            for m in supporting_matches
        ))

        # Local-only check: if all supporting chunks originate locally
        is_local_only = bool(supporting_matches and all(
            m.source_origin == "local" or m.independence_group == "local_filesystem"
            for m in supporting_matches
        ))

        # Single-source penalty (no external corroboration)
        single_source_penalty = 0.0
        if external_indep_count <= 1 and not (is_wiki_only or is_local_only):
            single_source_penalty = 0.08

        calculated_confidence = raw_score - contradiction_penalty - single_source_penalty

        # Apply strict ceiling caps
        ceiling = 1.0
        if is_wiki_only:
            ceiling = min(ceiling, 0.65)
            reference_only_penalty = 0.15
        if is_local_only:
            ceiling = min(ceiling, 0.50)

        # Superlative penalty if only single source or weak evidence
        if has_superlative and external_indep_count < 2:
            ceiling = min(ceiling, 0.60)

        final_confidence = max(0.0, min(ceiling, calculated_confidence))

        breakdown = {
            "source_quality": round(source_quality_score, 4),
            "direct_support": round(direct_support_score, 4),
            "independence": round(independence_score, 4),
            "evidence_specificity": round(evidence_specificity, 4),
            "external_independent_sources": external_indep_count,
            "contradiction_penalty": round(contradiction_penalty, 4),
            "single_source_penalty": round(single_source_penalty, 4),
            "reference_only_penalty": round(reference_only_penalty, 4),
            "is_wiki_only": is_wiki_only,
            "is_local_only": is_local_only,
            "ceiling_applied": round(ceiling, 2),
            "final_confidence": round(final_confidence, 4)
        }

        return (round(final_confidence, 4), breakdown)

    @classmethod
    def classify_claim(
        cls,
        claim: VerifiedClaim,
        confidence: float,
        breakdown: Dict[str, Any]
    ) -> Tuple[ClaimClassification, EpistemicStatus, str]:
        """
        Determines claim classification and epistemic status based on BOTH
        the deterministic confidence score AND importance-specific evidentiary requirements.
        """
        direct_count = claim.direct_support_count
        indep_count = claim.independent_external_source_count
        high_qual_count = claim.high_quality_source_count
        contra_count = claim.contradicting_source_count
        is_wiki_only = breakdown.get("is_wiki_only", False)
        is_local_only = breakdown.get("is_local_only", False)

        # 1. Contradiction analysis: Direct Disproof vs Unresolved Debate
        direct_disproof = False
        for m in claim.evidence_matches:
            if m.support_type == EvidenceSupportType.CONTRADICTORY:
                reason = (m.contradiction_reason or "").lower()
                if any(term in reason for term in ["direct_disproof", "negation", "disproved", "refuted", "no evidence", "fictional", "mythical"]) and (m.credibility_tier <= 2 or m.evidence_weight_class in ("PRIMARY", "HIGH")):
                    direct_disproof = True
                    break

        if direct_disproof:
            return (
                ClaimClassification.REJECTED,
                EpistemicStatus.UNSUPPORTED,
                "FORBIDDEN"
            )

        if contra_count > 0:
            return (
                ClaimClassification.UNRESOLVED_DEBATE,
                EpistemicStatus.DEBATED,
                "DEBATE_ONLY"
            )

        # 2. Local-only or Wikipedia-only cannot be VERIFIED_FACT
        if is_wiki_only or is_local_only:
            if confidence >= 0.50 and direct_count >= 1:
                return (
                    ClaimClassification.SUPPORTED_HYPOTHESIS,
                    EpistemicStatus.PROPOSED_THEORY if is_local_only else EpistemicStatus.LIKELY,
                    "HEDGED_FRAMING"
                )
            return (
                ClaimClassification.UNVERIFIED_CLAIM,
                EpistemicStatus.UNCERTAIN,
                "FORBIDDEN"
            )

        # 3. Section 2 VERIFIED_FACT standard:
        # Requires at least TWO independent NON-REFERENCE external evidence groups.
        # Single strong primary source without independent corroboration must be SUPPORTED_HYPOTHESIS.
        non_ref_indep_count = getattr(claim, "independent_non_ref_external_source_count", 0)
        if non_ref_indep_count < 2:
            if confidence >= 0.25 and (direct_count >= 1 or len(claim.supporting_chunk_ids) >= 1):
                return (
                    ClaimClassification.SUPPORTED_HYPOTHESIS,
                    EpistemicStatus.STRONGLY_SUPPORTED if confidence >= 0.70 else (EpistemicStatus.LIKELY if confidence >= 0.50 else EpistemicStatus.PROPOSED_THEORY),
                    "HEDGED_FRAMING"
                )
            return (
                ClaimClassification.UNVERIFIED_CLAIM,
                EpistemicStatus.UNCERTAIN,
                "FORBIDDEN"
            )

        # 4. Multi-source VERIFIED_FACT qualification when non_ref_indep_count >= 2:
        if claim.importance == ClaimImportance.CRITICAL:
            if (confidence >= 0.75 and direct_count >= 1 and high_qual_count >= 1):
                return (
                    ClaimClassification.VERIFIED_FACT,
                    EpistemicStatus.KNOWN_FACT,
                    "DIRECT_FACT"
                )
            elif confidence >= 0.55 and (direct_count >= 1 or claim.contextual_support_count >= 2):
                return (
                    ClaimClassification.SUPPORTED_HYPOTHESIS,
                    EpistemicStatus.STRONGLY_SUPPORTED if confidence >= 0.65 else EpistemicStatus.LIKELY,
                    "HEDGED_FRAMING"
                )
            else:
                return (
                    ClaimClassification.UNVERIFIED_CLAIM,
                    EpistemicStatus.UNCERTAIN,
                    "FORBIDDEN"
                )

        if claim.importance == ClaimImportance.HIGH:
            if (confidence >= 0.70 and direct_count >= 1):
                return (
                    ClaimClassification.VERIFIED_FACT,
                    EpistemicStatus.KNOWN_FACT if confidence >= 0.80 else EpistemicStatus.STRONGLY_SUPPORTED,
                    "DIRECT_FACT"
                )
            elif confidence >= 0.50:
                return (
                    ClaimClassification.SUPPORTED_HYPOTHESIS,
                    EpistemicStatus.LIKELY,
                    "HEDGED_FRAMING"
                )
            else:
                return (
                    ClaimClassification.UNVERIFIED_CLAIM,
                    EpistemicStatus.UNCERTAIN,
                    "FORBIDDEN"
                )

        if confidence >= 0.65 and direct_count >= 1:
            return (
                ClaimClassification.VERIFIED_FACT,
                EpistemicStatus.STRONGLY_SUPPORTED,
                "DIRECT_FACT"
            )
        elif confidence >= 0.25 and (direct_count >= 1 or len(claim.supporting_chunk_ids) >= 1):
            return (
                ClaimClassification.SUPPORTED_HYPOTHESIS,
                EpistemicStatus.LIKELY if confidence >= 0.45 else EpistemicStatus.PROPOSED_THEORY,
                "HEDGED_FRAMING"
            )

        return (
            ClaimClassification.UNVERIFIED_CLAIM,
            EpistemicStatus.UNSUPPORTED,
            "FORBIDDEN"
        )

    def verify_claim(
        self,
        claim: VerifiedClaim,
        evidence_pool: Optional[List[Dict[str, Any]]] = None,
        topic: Optional[str] = None
    ) -> VerifiedClaim:
        """
        Verify an individual claim against vector retrieval results or supplied evidence pool.
        """
        retrieved_hits: List[Dict[str, Any]] = []

        # Retrieve evidence if vector store is active
        if self.evidence_store:
            try:
                retrieved_hits = self.evidence_store.retrieve_evidence(
                    query=claim.statement,
                    episode_id=claim.episode_id,
                    top_k=self.max_chunks_per_claim
                )
            except Exception as e:
                logger.warning(f"EvidenceStore retrieval failed for claim '{claim.claim_id}': {e}")

        # Combine with supplied pool if provided
        if evidence_pool:
            retrieved_hits.extend(evidence_pool)

        # Deduplicate hits by chunk_id
        unique_hits: Dict[str, Dict[str, Any]] = {}
        for h in retrieved_hits:
            cid = h.get("chunk_id") or h.get("id") or str(hash(h.get("text", "")))
            if cid not in unique_hits:
                unique_hits[cid] = h

        # Classify each evidence match
        matches: List[ClaimEvidenceMatch] = []
        for cid, hit in unique_hits.items():
            # Section 4: topics.json must never produce source evidence
            source_url = str(hit.get("url", "")).lower()
            source_title = str(hit.get("title", "")).lower()
            source_id = str(hit.get("source_id", "")).lower()
            raw_path = str(hit.get("raw_cache_path", "")).lower()
            text = hit.get("text") or hit.get("snippet", "")
            if "topics.json" in source_url or "topics.json" in source_title or "topics.json" in source_id or "topics.json" in raw_path or source_id == "local_15fef56c3197":
                continue
            if '"episodes":' in text.lower() or '"badge":' in text.lower() or '"title_tamil":' in text.lower():
                continue

            score = float(hit.get("score") or hit.get("similarity_score", 0.50))

            if "support_type" in hit:
                st = hit["support_type"]
                if isinstance(st, str):
                    try:
                        support_type = EvidenceSupportType(st)
                    except (ValueError, KeyError):
                        support_type = EvidenceSupportType[st]
                else:
                    support_type = st
                contra_reason = hit.get("contradiction_reason")
            else:
                from autonomous.claim_extractor import ClaimExtractor
                chunk_dates = ClaimExtractor.extract_dates(text)
                chunk_nums = ClaimExtractor.extract_numbers(text)
                chunk_ents = ClaimExtractor.extract_entities(text)

                support_type, contra_reason = self.classify_evidence_support(
                    claim=claim,
                    chunk_text=text,
                    similarity_score=score,
                    chunk_entities=chunk_ents,
                    chunk_dates=chunk_dates,
                    chunk_numbers=chunk_nums
                )

            if support_type != EvidenceSupportType.IRRELEVANT:
                match = ClaimEvidenceMatch(
                    chunk_id=cid,
                    source_id=hit.get("source_id", "src_unknown"),
                    support_type=support_type,
                    similarity_score=score,
                    publisher=hit.get("publisher"),
                    domain=hit.get("domain"),
                    independence_group=hit.get("independence_group") or hit.get("domain") or "unknown_group",
                    source_role=hit.get("source_role", "primary_evidence"),
                    evidence_weight_class=hit.get("evidence_weight_class", "PRIMARY"),
                    source_origin=hit.get("source_origin", "external"),
                    credibility_tier=int(hit.get("credibility_tier", 3)),
                    snippet=text[:250],
                    matched_entities=[e for e in claim.extracted_entities if e in text.lower()],
                    matched_dates=[d for d in claim.extracted_dates if d in text.lower()],
                    matched_numbers=[n for n in claim.extracted_numbers if n in text],
                    contradiction_reason=contra_reason
                )
                matches.append(match)

        claim.evidence_matches = matches

        # Separate accounting metrics
        claim.evidence_chunk_count = len(matches)
        claim.supporting_chunk_ids = [m.chunk_id for m in matches if m.support_type in (EvidenceSupportType.DIRECT_SUPPORT, EvidenceSupportType.INDIRECT_SUPPORT)]
        claim.contradicting_chunk_ids = [m.chunk_id for m in matches if m.support_type == EvidenceSupportType.CONTRADICTORY]

        supporting_matches = [m for m in matches if m.support_type in (EvidenceSupportType.DIRECT_SUPPORT, EvidenceSupportType.INDIRECT_SUPPORT)]
        claim.direct_support_count = len([m for m in matches if m.support_type == EvidenceSupportType.DIRECT_SUPPORT])
        claim.contextual_support_count = len([m for m in matches if m.support_type == EvidenceSupportType.CONTEXTUAL_ONLY])
        claim.contradicting_source_count = len({m.source_id for m in matches if m.support_type == EvidenceSupportType.CONTRADICTORY})

        # Sources & independence
        supporting_sources = {m.source_id for m in supporting_matches}
        claim.source_count = len(supporting_sources)
        claim.supporting_source_ids = list(supporting_sources)

        external_matches = [m for m in supporting_matches if m.source_origin in ("external", "cached_external")]
        claim.external_source_count = len({m.source_id for m in external_matches})

        external_groups = {
            m.independence_group for m in external_matches
            if m.independence_group not in ("local_filesystem", "unknown_group")
        }
        claim.independent_external_source_count = len(external_groups)

        # Section 2 & 3: Independent NON-REFERENCE external groups
        non_ref_external_groups = {
            m.independence_group for m in external_matches
            if m.independence_group not in ("local_filesystem", "unknown_group", "wikipedia.org", "wikimedia.org")
            and m.evidence_weight_class != "REFERENCE"
        }
        claim.independent_non_ref_external_source_count = len(non_ref_external_groups)
        claim.independence_groups = list({m.independence_group for m in supporting_matches})
        claim.source_origins = list({m.source_origin for m in supporting_matches})
        claim.evidence_weight_classes = list({m.evidence_weight_class for m in supporting_matches})

        claim.high_quality_source_count = len({
            m.source_id for m in external_matches
            if m.credibility_tier <= 2 and m.evidence_weight_class in ("PRIMARY", "HIGH")
        })

        # Deterministic scoring
        has_sup = (claim.claim_type == ClaimType.SUPERLATIVE_ASSERTION)
        conf_score, breakdown = self.calculate_deterministic_confidence(
            matches=matches,
            importance=claim.importance,
            has_superlative=has_sup
        )
        claim.confidence_score = conf_score
        claim.confidence_breakdown = breakdown

        # Final classification
        classification, epistemic_status, constraint = self.classify_claim(
            claim=claim,
            confidence=conf_score,
            breakdown=breakdown
        )
        claim.classification = classification
        claim.epistemic_status = epistemic_status
        claim.epistemic_constraint = constraint

        # Section 4 & 6: Normalized topic aspect & relevance class
        claim.topic_aspect = self.normalize_topic_aspect(claim.topic_aspect, claim.statement)
        claim.relevance_class = self.determine_relevance_class(claim, topic)

        return claim

    @classmethod
    def determine_relevance_class(
        cls,
        claim: VerifiedClaim,
        topic: Optional[str] = None
    ) -> ClaimRelevanceClass:
        """
        Differentiate CORE_TOPIC, SUPPORTING_CONTEXT, and BACKGROUND.
        - Central archaeological subject -> CORE_TOPIC
        - Administrative/history-of-research facts (e.g. TNSDA established 1961) -> SUPPORTING_CONTEXT
        - Regional geography, general era background -> BACKGROUND
        """
        stmt_lower = claim.statement.lower()
        topic_lower = (topic or "").lower()

        # Administrative / organizational history indicators
        admin_patterns = [
            "was set up in", "established in", "founded in", "created in",
            "official research department", "administrative", "headquarters located in",
            "state department of archaeology", "formed in 19", "formed in 20",
            "heritage commission", "survey department"
        ]
        is_admin_fact = any(p in stmt_lower for p in admin_patterns)

        if is_admin_fact:
            if "department of archaeology" in topic_lower or "asi history" in topic_lower or "tnsda" in topic_lower:
                return ClaimRelevanceClass.CORE_TOPIC
            return ClaimRelevanceClass.SUPPORTING_CONTEXT

        # Extract core subject terms from topic (e.g. "keezhadi", "keeladi", "poompuhar", "tanjore")
        core_subject_keywords = []
        if topic_lower:
            clean_t = re.split(r"[:\-\–]", topic_lower)[0].strip()
            tokens = [tok for tok in clean_t.split() if len(tok) > 3 and tok not in ("ancient", "tamil", "history", "early")]
            core_subject_keywords.extend(tokens)

        # Common archaeological indicators
        arch_indicators = [
            "excavation", "excavations", "brick structures", "drainage", "ring well",
            "tamil-brahmi", "potsherds", "pottery", "carbon dating", "ams dating",
            "artifacts", "beads", "carnelian", "urban settlement", "strata", "trench"
        ]

        has_core_subject = any(kw in stmt_lower for kw in core_subject_keywords) if core_subject_keywords else False
        has_arch_findings = any(ind in stmt_lower for ind in arch_indicators)

        if has_core_subject and (has_arch_findings or claim.claim_type in (ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimType.DATING_CHRONOLOGY, ClaimType.MATERIAL_CULTURE)):
            return ClaimRelevanceClass.CORE_TOPIC

        if has_core_subject:
            return ClaimRelevanceClass.CORE_TOPIC

        if has_arch_findings and claim.claim_type in (ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimType.DATING_CHRONOLOGY):
            return ClaimRelevanceClass.CORE_TOPIC

        return ClaimRelevanceClass.BACKGROUND

    @classmethod
    def normalize_topic_aspect(cls, aspect_str: str, statement: str = "") -> str:
        """Normalize research aspect into canonical topic aspects."""
        text = (aspect_str + " " + statement).lower()
        if any(w in text for w in ["dating", "chronology", "century", "radiocarbon", "bce", "ce", "period"]):
            return "chronology"
        if any(w in text for w in ["inscription", "inscriptions", "brahmi", "literacy", "script", "graffiti"]):
            return "literacy"
        if any(w in text for w in ["brick", "structures", "drainage", "ring well", "settlement", "urban", "town"]):
            return "settlement"
        if any(w in text for w in ["pottery", "beads", "carnelian", "agate", "gold", "iron", "potsherds", "material"]):
            return "material_culture"
        if any(w in text for w in ["excavation", "excavations", "trench", "discovery", "uncovered", "finds"]):
            return "archaeology"
        if any(w in text for w in ["river", "bank", "vaigai", "district", "sivaganga", "location", "geography"]):
            return "geography"
        if any(w in text for w in ["department", "tnsda", "asi", "institution", "administration"]):
            return "historical_context"
        if any(w in text for w in ["sangam", "trade", "culture", "civilization", "significance"]):
            return "cultural_significance"
        return "historical_context"

    def verify_all_claims(
        self,
        claims: List[VerifiedClaim],
        evidence_pool: Optional[List[Dict[str, Any]]] = None,
        topic: Optional[str] = None
    ) -> List[VerifiedClaim]:
        """Verify an entire collection of candidate claims."""
        verified = []
        for c in claims:
            v = self.verify_claim(c, evidence_pool=evidence_pool, topic=topic)
            verified.append(v)
        return verified
