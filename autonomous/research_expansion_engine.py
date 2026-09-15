"""
autonomous/research_expansion_engine.py - Phase 5.2 Autonomous Research Expansion Engine.

When Content Quality validation determines an episode lacks sufficient unique, core, or
diverse claims, this engine autonomously:
1. Diagnoses research gaps via ResearchGapAnalyzer.
2. Generates targeted, non-duplicate research questions.
3. Discovers, retrieves, and deduplicates authoritative external sources.
4. Chunks and indexes new evidence into the existing Qdrant collection with deterministic IDs.
5. Preserves original research additively with full cycle provenance.
6. Re-extracts and re-verifies claims via deterministic ClaimVerifier.
7. Re-runs content quality validation within bounded expansion budgets.
8. Safely halts at REVIEW_REQUIRED if evidence remains insufficient.
"""

import os
import re
import json
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timezone

from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.research_gap_analyzer import ResearchGapAnalyzer, ResearchGapReport
from autonomous.source_retriever import SourceRetriever, SourceMetadata
from autonomous.source_discovery import SourceDiscovery, AUTHORITATIVE_SEEDS
from autonomous.research_processor import ResearchProcessor, EvidenceChunk
from autonomous.evidence_store import EvidenceStore
from autonomous.claim_extractor import ClaimExtractor
from autonomous.claim_verifier import ClaimVerifier
from autonomous.content_planner import ContentPlanner
from autonomous.content_quality_validator import ContentQualityValidator
from autonomous.claim_verification_engine import ClaimVerificationEngine
from autonomous.research_planner import ResearchPlan
from autonomous.research_dossier import ResearchDossier
from autonomous.claim_models import (
    ClaimRelevanceClass,
    ClaimClassification,
    VerifiedClaim,
    ContentQualityReport
)

logger = logging.getLogger("autonomous.research_expansion_engine")

# Strict expansion budget bounds
MAX_RESEARCH_EXPANSIONS = 2
MAX_QUESTIONS_PER_EXPANSION = 5
MAX_NEW_SOURCES_PER_EXPANSION = 8
MAX_TOTAL_NEW_CHUNKS_PER_EXPANSION = 25


class ResearchExpansionEngine:
    """
    Master engine executing bounded, provenance-preserving research expansion
    to resolve content quality deficits without weakening evidence rules.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        gap_analyzer: Optional[ResearchGapAnalyzer] = None,
        retriever: Optional[SourceRetriever] = None,
        discovery: Optional[SourceDiscovery] = None,
        processor: Optional[ResearchProcessor] = None,
        evidence_store: Optional[EvidenceStore] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
        claim_verifier: Optional[ClaimVerifier] = None,
        quality_validator: Optional[ContentQualityValidator] = None,
        content_planner: Optional[ContentPlanner] = None,
        verification_engine: Optional[ClaimVerificationEngine] = None,
        offline: bool = False
    ):
        self.state_manager = state_manager or StateManager()
        self.offline = offline or os.getenv("RESEARCH_OFFLINE", "false").lower() in ("true", "1", "yes")
        self.gap_analyzer = gap_analyzer or ResearchGapAnalyzer()
        self.retriever = retriever or SourceRetriever()
        self.discovery = discovery or SourceDiscovery(retriever=self.retriever, offline=self.offline)
        self.processor = processor or ResearchProcessor()
        self.evidence_store = evidence_store or EvidenceStore(device="cpu")
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.claim_verifier = claim_verifier or ClaimVerifier(evidence_store=self.evidence_store)
        self.quality_validator = quality_validator or ContentQualityValidator()
        self.content_planner = content_planner or ContentPlanner()
        self.verification_engine = verification_engine or ClaimVerificationEngine(
            state_manager=self.state_manager,
            evidence_store=self.evidence_store
        )

        self.max_expansions = MAX_RESEARCH_EXPANSIONS
        self.max_questions = MAX_QUESTIONS_PER_EXPANSION
        self.max_sources = MAX_NEW_SOURCES_PER_EXPANSION
        self.max_chunks = MAX_TOTAL_NEW_CHUNKS_PER_EXPANSION

    def _load_history(self, history_file: Path) -> List[Dict[str, Any]]:
        """Load persisted expansion history if present."""
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                logger.warning(f"Failed to read expansion history: {e}")
        return []

    def _save_history(self, history_file: Path, history: List[Dict[str, Any]]) -> None:
        """Persist expansion history."""
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def execute_expansion(
        self,
        episode_id: str,
        offline: bool = False
    ) -> Dict[str, Any]:
        """
        Execute bounded autonomous research expansion loop (up to MAX_RESEARCH_EXPANSIONS cycles).
        Advances state:
            REVIEW_REQUIRED -> RESEARCH_EXPANDING -> RESEARCH_COMPLETE -> VERIFYING -> VERIFIED -> SCRIPTING -> SCRIPT_VALIDATED
        Or if evidence remains insufficient after budget exhaustion:
            REVIEW_REQUIRED (with reason: INSUFFICIENT_EVIDENCE_AFTER_RESEARCH_EXPANSION)
        """
        is_offline = offline or self.offline

        # 1. Load episode record
        session = self.state_manager._get_session()
        try:
            ep = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if not ep:
                raise KeyError(f"Episode '{episode_id}' not found in database.")
            initial_state = EpisodeState(ep.status)
            topic = ep.topic
            ep_dir = Path(ep.output_directory)
        finally:
            session.close()

        research_dir = ep_dir / "research"
        verif_dir = ep_dir / "verification"
        research_dir.mkdir(parents=True, exist_ok=True)
        verif_dir.mkdir(parents=True, exist_ok=True)

        history_file = research_dir / "expansion_history.json"
        history = self._load_history(history_file)
        completed_cycles = len(history)

        # Idempotency check: if already at max cycles and currently in terminal/review state
        if completed_cycles >= self.max_expansions:
            logger.warning(f"Episode '{episode_id}' has already reached max expansion budget ({completed_cycles}/{self.max_expansions}).")
            if initial_state != EpisodeState.SCRIPT_VALIDATED:
                self.state_manager.transition_state(
                    episode_id=episode_id,
                    target_state=EpisodeState.REVIEW_REQUIRED,
                    stage_name="REVIEW_REQUIRED",
                    error_message="INSUFFICIENT_EVIDENCE_AFTER_RESEARCH_EXPANSION"
                )
            return {
                "status": initial_state.value,
                "episode_id": episode_id,
                "expansion_status": "BUDGET_EXHAUSTED",
                "completed_cycles": completed_cycles,
                "error": "INSUFFICIENT_EVIDENCE_AFTER_RESEARCH_EXPANSION"
            }

        # Expansion loop
        current_cycle = completed_cycles + 1
        logger.info(f"Starting Phase 5.2 Research Expansion Cycle {current_cycle}/{self.max_expansions} for episode '{episode_id}'")

        while current_cycle <= self.max_expansions:
            cycle_result = self._run_single_expansion_cycle(
                episode_id=episode_id,
                topic=topic,
                ep_dir=ep_dir,
                cycle_number=current_cycle,
                history=history,
                offline=is_offline
            )

            # Record in history
            history.append(cycle_result["cycle_summary"])
            self._save_history(history_file, history)

            if cycle_result["status"] == "SCRIPT_VALIDATED":
                logger.info(f"Research Expansion Cycle {current_cycle} SUCCEEDED! Advanced to SCRIPT_VALIDATED.")
                self._save_expansion_report(
                    ep_dir=ep_dir,
                    episode_id=episode_id,
                    topic=topic,
                    initial_state=initial_state.value,
                    final_state=EpisodeState.SCRIPT_VALIDATED.value,
                    expansion_status="SUCCESS",
                    total_cycles=current_cycle,
                    history=history,
                    cycle_result=cycle_result
                )
                return {
                    "status": "SCRIPT_VALIDATED",
                    "episode_id": episode_id,
                    "expansion_number": current_cycle,
                    "cycle_result": cycle_result
                }

            current_cycle += 1

        # Budget exhausted without passing content quality
        logger.warning(f"Research expansion exhausted ({self.max_expansions} cycles). Safe halt at REVIEW_REQUIRED.")
        fail_reason = "INSUFFICIENT_EVIDENCE_AFTER_RESEARCH_EXPANSION"
        self.state_manager.transition_state(
            episode_id=episode_id,
            target_state=EpisodeState.REVIEW_REQUIRED,
            stage_name="REVIEW_REQUIRED",
            error_message=fail_reason
        )

        self._save_expansion_report(
            ep_dir=ep_dir,
            episode_id=episode_id,
            topic=topic,
            initial_state=initial_state.value,
            final_state=EpisodeState.REVIEW_REQUIRED.value,
            expansion_status="INSUFFICIENT_EVIDENCE",
            total_cycles=len(history),
            history=history,
            cycle_result=cycle_result,
            error=fail_reason
        )

        return {
            "status": "REVIEW_REQUIRED",
            "episode_id": episode_id,
            "expansion_status": "INSUFFICIENT_EVIDENCE",
            "total_cycles": len(history),
            "error": fail_reason,
            "cycle_result": cycle_result
        }

    def _run_single_expansion_cycle(
        self,
        episode_id: str,
        topic: str,
        ep_dir: Path,
        cycle_number: int,
        history: List[Dict[str, Any]],
        offline: bool
    ) -> Dict[str, Any]:
        """Execute one complete expansion cycle."""
        research_dir = ep_dir / "research"
        verif_dir = ep_dir / "verification"

        # 1. Transition to RESEARCH_EXPANDING
        self.state_manager.transition_state(
            episode_id=episode_id,
            target_state=EpisodeState.RESEARCH_EXPANDING,
            stage_name="RESEARCH_EXPANDING"
        )

        # 2. Gather existing attempted questions across all cycles
        attempted_questions: List[str] = []
        for h in history:
            attempted_questions.extend(h.get("questions_attempted", []))

        # 3. Load current verification & quality reports
        cq_file = verif_dir / "content_quality_report.json"
        cq_data = {}
        if cq_file.exists():
            try:
                with open(cq_file, "r", encoding="utf-8") as f:
                    cq_data = json.load(f)
            except Exception:
                pass

        claims_file = verif_dir / "claims.json"
        existing_claims = []
        if claims_file.exists():
            try:
                with open(claims_file, "r", encoding="utf-8") as f:
                    raw_cl = json.load(f)
                    existing_claims = [VerifiedClaim(**c) for c in raw_cl]
            except Exception:
                pass

        # 4. Run Gap Analysis
        gap_report = self.gap_analyzer.analyze_gaps(
            topic=topic,
            verified_claims=existing_claims,
            content_quality_report=cq_data,
            attempted_questions=attempted_questions,
            expansion_number=cycle_number,
            offline=offline
        )

        # Save gap report
        gap_report_file = research_dir / "research_gap_report.json"
        with open(gap_report_file, "w", encoding="utf-8") as f:
            json.dump(gap_report.to_dict(), f, indent=2, ensure_ascii=False)

        # Questions to target for this cycle (bounded by max_questions)
        new_questions = gap_report.recommended_research_questions[:self.max_questions]
        logger.info(f"Expansion Cycle {cycle_number} targeted questions ({len(new_questions)}): {new_questions}")

        # 5. Load existing sources for deduplication
        sources_file = research_dir / "sources.json"
        existing_sources: List[SourceMetadata] = []
        seen_urls: Set[str] = set()
        seen_hashes: Set[str] = set()
        seen_domains: Set[str] = set()
        seen_source_ids: Set[str] = set()

        if sources_file.exists():
            try:
                with open(sources_file, "r", encoding="utf-8") as f:
                    raw_src = json.load(f)
                    for s in raw_src:
                        sm = SourceMetadata(**s)
                        existing_sources.append(sm)
                        seen_source_ids.add(sm.source_id)
                        if sm.url:
                            seen_urls.add(sm.url.strip().lower())
                        if sm.content_hash:
                            seen_hashes.add(sm.content_hash)
                        if sm.domain:
                            seen_domains.add(sm.domain)
            except Exception as e:
                logger.warning(f"Error loading existing sources: {e}")

        # 6. Discover & Retrieve New Sources
        new_sources: List[SourceMetadata] = []

        # Strategy A: Check authoritative seeds relevant to topic & questions
        topic_lower = topic.lower()
        matched_seeds = []
        for kw, seeds in AUTHORITATIVE_SEEDS.items():
            if kw in topic_lower:
                matched_seeds.extend(seeds)

        for seed in matched_seeds:
            if len(new_sources) >= self.max_sources:
                break
            seed_url = seed["url"].strip()
            if seed_url.lower() in seen_urls:
                continue

            if offline:
                continue

            try:
                logger.info(f"Retrieving authoritative seed during expansion: {seed['title']}")
                meta = self.retriever.retrieve_url(seed_url)
                if meta.extraction_status in ("EXTRACTED", "CACHED") and meta.cleaned_text:
                    if meta.content_hash not in seen_hashes:
                        meta.title = seed.get("title", meta.title)
                        meta.publisher = seed.get("publisher", meta.publisher)
                        meta.credibility_tier = seed.get("tier", 1)
                        s_role, e_weight = self.retriever.determine_evidence_weight(
                            meta.credibility_tier, meta.domain, meta.source_type, meta.source_origin
                        )
                        meta.source_role = s_role
                        meta.evidence_weight_class = e_weight
                        meta.extra_metadata["expansion_cycle"] = f"expansion_{cycle_number}"
                        new_sources.append(meta)
                        seen_urls.add(seed_url.lower())
                        seen_hashes.add(meta.content_hash)
                        seen_source_ids.add(meta.source_id)
            except Exception as e:
                logger.debug(f"Failed to retrieve seed {seed_url}: {e}")

        # Strategy B: Local Cache Fixtures (Always safe offline)
        cache_dirs = [
            Path("outputs/research_cache"),
            Path("data/knowledge"),
            Path("backend/data/knowledge")
        ]
        for cdir in cache_dirs:
            if len(new_sources) >= self.max_sources:
                break
            if not cdir.exists():
                continue
            for cp in cdir.glob("*.txt"):
                if len(new_sources) >= self.max_sources:
                    break
                try:
                    with open(cp, "r", encoding="utf-8", errors="ignore") as f:
                        header = f.readline()
                        url_line = f.readline()
                    m_url = re.search(r"URL:\s*(https?://[^\s]+)", url_line)
                    url_val = m_url.group(1).strip() if m_url else str(cp)
                    if url_val.lower() in seen_urls:
                        continue

                    # Retrieve via local file retrieval
                    meta = self.retriever.retrieve_local_document(str(cp.resolve()))
                    if meta.extraction_status == "EXTRACTED" and meta.cleaned_text and len(meta.cleaned_text) > 100:
                        if meta.content_hash not in seen_hashes:
                            # If url was recorded in header, preserve it
                            if m_url:
                                meta.url = url_val
                                meta.domain = self.retriever.extract_domain(url_val)
                                if "wikipedia.org" in url_val:
                                    meta.source_type = "wikipedia"
                                    meta.source_role = "discovery_reference"
                                    meta.evidence_weight_class = "REFERENCE"
                                    meta.independence_group = "wikipedia.org"
                            meta.extra_metadata["expansion_cycle"] = f"expansion_{cycle_number}"
                            new_sources.append(meta)
                            seen_urls.add(url_val.lower())
                            seen_hashes.add(meta.content_hash)
                            seen_source_ids.add(meta.source_id)
                except Exception as e:
                    logger.debug(f"Error inspecting cache file {cp}: {e}")

        # Strategy C: Targeted Wikipedia / Web searches (if online and quota remaining)
        if not offline and len(new_sources) < self.max_sources:
            # Look up specific article titles
            wiki_titles = []
            if "keezhadi" in topic_lower or "keeladi" in topic_lower:
                wiki_titles = ["Keezhadi_excavation_site", "Keeladi"]

            for w_title in wiki_titles:
                if len(new_sources) >= self.max_sources:
                    break
                # Only if not already collected
                w_url = f"https://en.wikipedia.org/wiki/{w_title}".lower()
                if w_url in seen_urls:
                    continue
                try:
                    plan_temp = ResearchPlan(topic=w_title)
                    w_srcs = self.discovery._discover_wikipedia_sources(plan_temp)
                    for ws in w_srcs:
                        if ws.url.lower() not in seen_urls and ws.content_hash not in seen_hashes:
                            ws.extra_metadata["expansion_cycle"] = f"expansion_{cycle_number}"
                            new_sources.append(ws)
                            seen_urls.add(ws.url.lower())
                            seen_hashes.add(ws.content_hash)
                            seen_source_ids.add(ws.source_id)
                            break
                except Exception as e:
                    logger.debug(f"Targeted Wikipedia search failed for {w_title}: {e}")

        logger.info(f"Expansion Cycle {cycle_number} discovered {len(new_sources)} new sources.")

        # Record new sources in database
        for s in new_sources:
            self.state_manager.record_research_source(
                episode_id=episode_id,
                source_data=s.to_dict()
            )

        # 7. Process & Chunk New Sources
        new_chunks: List[EvidenceChunk] = []
        for src in new_sources:
            if len(new_chunks) >= self.max_chunks:
                break
            chunks = self.processor.chunk_source(src, episode_id=episode_id)
            for c in chunks:
                if len(new_chunks) >= self.max_chunks:
                    break
                # Tag chunk with expansion cycle
                c.expansion_cycle = f"expansion_{cycle_number}"
                new_chunks.append(c)

        # Release in-memory raw content from new sources to preserve memory
        for s in new_sources:
            s.raw_content = None
            if s.cleaned_text and len(s.cleaned_text) > 500:
                s.cleaned_text = s.cleaned_text[:500] + "... [TRUNCATED - Full text in evidence.json]"

        logger.info(f"Expansion Cycle {cycle_number} created {len(new_chunks)} new evidence chunks.")

        # 8. Index New Chunks into Qdrant Vector Store
        index_stats = {}
        if new_chunks:
            index_stats = self.evidence_store.index_chunks(
                chunks=new_chunks,
                topic=topic,
                episode_id=episode_id
            )

        # 9. Additive Evidence & Dossier Integration
        # Load existing evidence chunks
        evidence_file = research_dir / "evidence.json"
        all_chunks_dict: List[Dict[str, Any]] = []
        existing_chunk_ids: Set[str] = set()
        if evidence_file.exists():
            try:
                with open(evidence_file, "r", encoding="utf-8") as f:
                    all_chunks_dict = json.load(f)
                    existing_chunk_ids = set(c.get("chunk_id") for c in all_chunks_dict)
            except Exception:
                pass

        for c in new_chunks:
            if c.chunk_id not in existing_chunk_ids:
                all_chunks_dict.append(c.to_dict())
                existing_chunk_ids.add(c.chunk_id)

        with open(evidence_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks_dict, f, indent=2, ensure_ascii=False)

        # Update sources.json
        all_sources = existing_sources + new_sources
        with open(sources_file, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in all_sources], f, indent=2, ensure_ascii=False)

        # Update dossier.json additively
        dossier_file = research_dir / "dossier.json"
        if dossier_file.exists():
            try:
                with open(dossier_file, "r", encoding="utf-8") as f:
                    dossier_data = json.load(f)
                dossier_data["chunks_count"] = len(all_chunks_dict)
                dossier_data["sources_count"] = len(all_sources)
                # Synthesize preliminary findings from new chunks so ClaimExtractor can extract claims
                existing_findings = dossier_data.get("preliminary_findings", [])
                existing_stmts = {f.get("statement", "") for f in existing_findings if isinstance(f, dict)}
                finding_counter = len(existing_findings) + 1

                for chunk in new_chunks:
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.text) if len(s.strip()) > 30 and len(s.strip().split()) >= 6]
                    for s in sentences[:2]:
                        pub = chunk.publisher or chunk.title or "Source"
                        stmt = f"Documentation from {pub} indicates: \"{s}\""
                        if stmt not in existing_stmts:
                            existing_stmts.add(stmt)
                            s_lower = s.lower()
                            aspect = "archaeology"
                            if any(w in s_lower for w in ["dating", "bce", "ce", "century", "radiocarbon", "ams"]):
                                aspect = "chronology"
                            elif any(w in s_lower for w in ["brahmi", "script", "inscri", "potsherd", "literacy"]):
                                aspect = "literacy"
                            elif any(w in s_lower for w in ["brick", "structure", "ring well", "drainage", "settlement"]):
                                aspect = "settlement"
                            elif any(w in s_lower for w in ["bead", "pottery", "carnelian", "quartz", "iron", "gold"]):
                                aspect = "material_culture"
                            elif any(w in s_lower for w in ["trade", "port", "roman", "coin"]):
                                aspect = "trade"

                            src_meta = next((src for src in new_sources if src.source_id == chunk.source_id), None)
                            indep_grp = src_meta.independence_group if src_meta else "external"
                            weight_cls = src_meta.evidence_weight_class if src_meta else "PRIMARY"

                            new_finding_dict = {
                                "finding_id": f"find_exp_{finding_counter:03d}",
                                "statement": stmt,
                                "topic_aspect": aspect,
                                "supporting_chunk_ids": [chunk.chunk_id],
                                "supporting_source_ids": [chunk.source_id],
                                "credibility_levels": [chunk.credibility_tier],
                                "status": "EVIDENCE_SUPPORTED",
                                "independence_groups": [indep_grp],
                                "source_origins": ["external"],
                                "evidence_weight_classes": [weight_cls]
                            }
                            existing_findings.append(new_finding_dict)
                            finding_counter += 1

                dossier_data["preliminary_findings"] = existing_findings

                if "expansion_history" not in dossier_data:
                    dossier_data["expansion_history"] = []
                dossier_data["expansion_history"].append({
                    "expansion_cycle": f"expansion_{cycle_number}",
                    "new_sources": [s.source_id for s in new_sources],
                    "new_chunks_count": len(new_chunks),
                    "questions_attempted": new_questions
                })
                with open(dossier_file, "w", encoding="utf-8") as f:
                    json.dump(dossier_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Failed to update dossier.json additively: {e}")

        # 10. Advance to RESEARCH_COMPLETE
        self.state_manager.transition_state(
            episode_id=episode_id,
            target_state=EpisodeState.RESEARCH_COMPLETE,
            stage_name="RESEARCH_COMPLETE"
        )

        # 11. Run Verification Phase (Claim Extraction -> Normalization -> Verification)
        verif_result = self.verification_engine.execute_claim_verification(episode_id=episode_id)

        # Compute independent source groups count
        new_indep_groups = list(set(s.independence_group for s in new_sources if s.independence_group))
        cycle_summary = {
            "expansion_cycle": f"expansion_{cycle_number}",
            "cycle_number": cycle_number,
            "questions_attempted": new_questions,
            "new_sources_count": len(new_sources),
            "new_sources": [s.source_id for s in new_sources],
            "new_independent_groups_count": len(new_indep_groups),
            "new_independent_groups": new_indep_groups,
            "new_chunks_count": len(new_chunks),
            "new_chunks": [c.chunk_id for c in new_chunks],
            "verified_claims_after_cycle": verif_result.get("verified_claims_count", 0),
            "verified_facts_after_cycle": verif_result.get("verified_facts_count", 0),
            "supported_hypotheses_after_cycle": verif_result.get("supported_hypotheses_count", 0),
            "core_claims_after_cycle": verif_result.get("core_claims_count", 0),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # If verification gate failed, do NOT proceed to script generation
        if verif_result.get("status") != "VERIFIED":
            logger.warning(f"Verification gate not passed after expansion cycle {cycle_number}: {verif_result.get('error')}")
            return {
                "status": EpisodeState.REVIEW_REQUIRED.value,
                "cycle_number": cycle_number,
                "cycle_summary": cycle_summary,
                "verif_result": verif_result,
                "script_result": {"status": EpisodeState.REVIEW_REQUIRED.value, "error": verif_result.get("error")},
                "content_quality_report": {"passed": False, "content_quality_status": "VERIFICATION_NOT_PASSED", "research_expansion_recommended": True}
            }

        # 12. Run Script Generation & Content Quality Gate
        script_result = self.verification_engine.execute_script_generation_and_validation(
            episode_id=episode_id,
            offline=offline
        )

        # Load post-expansion content quality report
        post_cq_data = {}
        if cq_file.exists():
            try:
                with open(cq_file, "r", encoding="utf-8") as f:
                    post_cq_data = json.load(f)
            except Exception:
                pass

        cq_passed = post_cq_data.get("passed", False)
        final_state = script_result.get("status", EpisodeState.REVIEW_REQUIRED.value)

        # Compute independent source groups count
        new_indep_groups = list(set(s.independence_group for s in new_sources if s.independence_group))

        cycle_summary = {
            "expansion_cycle": f"expansion_{cycle_number}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "questions_attempted": new_questions,
            "new_sources_count": len(new_sources),
            "new_sources": [s.source_id for s in new_sources],
            "new_independent_groups": new_indep_groups,
            "new_chunks_count": len(new_chunks),
            "new_claims_count": verif_result.get("verified_claims_count", 0),
            "verified_facts_count": verif_result.get("verified_facts_count", 0),
            "supported_hypotheses_count": verif_result.get("supported_hypotheses_count", 0),
            "content_quality_passed": cq_passed,
            "quality_status": post_cq_data.get("content_quality_status", "CONTENT_INSUFFICIENT"),
            "unique_claims": post_cq_data.get("actual_unique_claims", 0),
            "core_claims": post_cq_data.get("core_claim_count", 0),
            "unique_aspects": post_cq_data.get("actual_unique_topic_aspects", 0),
            "failure_reasons": post_cq_data.get("failure_reasons", [])
        }

        return {
            "status": final_state,
            "cycle_number": cycle_number,
            "cycle_summary": cycle_summary,
            "verif_result": verif_result,
            "script_result": script_result,
            "content_quality_report": post_cq_data
        }

    def _save_expansion_report(
        self,
        ep_dir: Path,
        episode_id: str,
        topic: str,
        initial_state: str,
        final_state: str,
        expansion_status: str,
        total_cycles: int,
        history: List[Dict[str, Any]],
        cycle_result: Dict[str, Any],
        error: Optional[str] = None
    ) -> None:
        """Persist official research/research_expansion_report.json."""
        research_dir = ep_dir / "research"
        report_path = research_dir / "research_expansion_report.json"

        all_questions = []
        all_sources = []
        all_indep_groups = set()
        total_new_chunks = 0

        for h in history:
            all_questions.extend(h.get("questions_attempted", []))
            all_sources.extend(h.get("new_sources", []))
            for g in h.get("new_independent_groups", []):
                all_indep_groups.add(g)
            total_new_chunks += h.get("new_chunks_count", 0)

        cq = cycle_result.get("content_quality_report", {})
        vr = cycle_result.get("verif_result", {})

        report_data = {
            "episode_id": episode_id,
            "topic": topic,
            "initial_state": initial_state,
            "final_state": final_state,
            "expansion_status": expansion_status,
            "total_expansions_executed": total_cycles,
            "max_expansions": self.max_expansions,
            "questions_attempted": all_questions,
            "sources_discovered": all_sources,
            "new_independent_groups": sorted(list(all_indep_groups)),
            "new_chunks_indexed": total_new_chunks,
            "final_verification_breakdown": {
                "verified_claims_count": vr.get("verified_claims_count", 0),
                "verified_facts": vr.get("verified_facts_count", 0),
                "supported_hypotheses": vr.get("supported_hypotheses_count", 0),
            },
            "content_quality_result": {
                "passed": cq.get("passed", False),
                "content_quality_status": cq.get("content_quality_status", "UNKNOWN"),
                "unique_claims": cq.get("actual_unique_claims", 0),
                "core_claims": cq.get("core_claim_count", 0),
                "unique_topic_aspects": cq.get("actual_unique_topic_aspects", 0),
                "maximum_claim_scene_reuse": cq.get("maximum_claim_scene_reuse", 0),
                "failure_reasons": cq.get("failure_reasons", [])
            },
            "research_expansion_recommended": cq.get("research_expansion_recommended", False),
            "final_reason": error or ("Research expansion succeeded and verified." if final_state == "SCRIPT_VALIDATED" else "Content quality remains unsatisfied.")
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
