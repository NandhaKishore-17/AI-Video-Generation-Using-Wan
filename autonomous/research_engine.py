"""
autonomous/research_engine.py - Master Autonomous Research Orchestration Engine.

Coordinates Phase 4 research lifecycle:
TOPIC_SELECTED -> RESEARCHING -> RESEARCH_COMPLETE / REVIEW_REQUIRED / FAILED

Executes planning, source discovery, bounded retrieval, cleaning, chunking,
CPU-only Qdrant embedding, evidence retrieval, conflict detection, dossier assembly,
and strict Research Quality Gate validation.
"""

import os
import json
import time
import psutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.research_planner import ResearchPlanner, ResearchPlan
from autonomous.source_retriever import SourceRetriever, SourceMetadata
from autonomous.source_discovery import SourceDiscovery
from autonomous.research_processor import ResearchProcessor, EvidenceChunk
from autonomous.evidence_store import EvidenceStore, RESEARCH_COLLECTION
from autonomous.research_dossier import ResearchDossier, ResearchDossierBuilder

logger = logging.getLogger("autonomous.research_engine")


class ResearchQualityGate:
    """Validates minimum research completeness requirements before advancing."""

    def __init__(
        self,
        min_sources: int = 3,
        min_high_quality_sources: int = 1,
        min_evidence_chunks: int = 5,
        min_questions_answered: int = 3,
        minimum_external_independent_sources: Optional[int] = 2,
        minimum_high_quality_external_sources: Optional[int] = 1
    ):
        self.min_sources = min_sources
        self.min_high_quality_sources = min_high_quality_sources
        self.min_evidence_chunks = min_evidence_chunks
        self.min_questions_answered = min_questions_answered
        self.minimum_external_independent_sources = minimum_external_independent_sources
        self.minimum_high_quality_external_sources = minimum_high_quality_external_sources

    def evaluate(self, dossier: ResearchDossier) -> Tuple[bool, List[str]]:
        """
        Evaluate research dossier against the completeness gate.
        NOTE: Passing this gate means research is complete enough for verification,
        NOT that the claims are verified as historically true (Correction 14).
        """
        reasons = []
        stats = dossier.source_statistics

        total_sources = stats.get("total_sources", 0)
        if total_sources < self.min_sources:
            reasons.append(f"Insufficient source count: {total_sources} < {self.min_sources} required.")

        high_quality = stats.get("high_quality_sources_count", 0)
        if high_quality < self.min_high_quality_sources:
            reasons.append(f"Insufficient Tier 1/2 high-quality sources: {high_quality} < {self.min_high_quality_sources} required.")

        # Correction 4: Separate external independent sources gate
        if self.minimum_external_independent_sources is not None:
            ext_ind = stats.get("external_independent_sources", 0)
            if ext_ind < self.minimum_external_independent_sources:
                reasons.append(
                    f"Insufficient external independent sources: {ext_ind} < {self.minimum_external_independent_sources} required. "
                    f"(Local documents and same-domain links do not count as independent external corroboration)."
                )

        # Correction 4: High-quality external sources gate
        if self.minimum_high_quality_external_sources is not None:
            high_qual_ext = stats.get("high_quality_external_sources", 0)
            if high_qual_ext < self.minimum_high_quality_external_sources:
                reasons.append(
                    f"Insufficient high-quality external sources: {high_qual_ext} < {self.minimum_high_quality_external_sources} required. "
                    f"(Wikipedia is classified as discovery/reference and cannot satisfy high-quality external proof alone)."
                )

        total_chunks = stats.get("total_chunks", 0)
        if total_chunks < self.min_evidence_chunks:
            reasons.append(f"Insufficient evidence chunks: {total_chunks} < {self.min_evidence_chunks} required.")

        # Count answered research questions
        answered_questions = sum(
            1 for qe in dossier.question_evidence
            if len(qe.get("retrieved_evidence", [])) > 0
        )
        if answered_questions < self.min_questions_answered:
            reasons.append(f"Too few research questions supported by evidence: {answered_questions} < {self.min_questions_answered} required.")

        passed = len(reasons) == 0
        return passed, reasons


class ResearchEngine:
    """Master research engine for autonomous episode production."""

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        planner: Optional[ResearchPlanner] = None,
        retriever: Optional[SourceRetriever] = None,
        discovery: Optional[SourceDiscovery] = None,
        processor: Optional[ResearchProcessor] = None,
        evidence_store: Optional[EvidenceStore] = None,
        quality_gate: Optional[ResearchQualityGate] = None,
        offline: bool = False
    ):
        self.state_manager = state_manager or StateManager()
        self.offline = offline or os.getenv("RESEARCH_OFFLINE", "false").lower() in ("true", "1", "yes")
        self.planner = planner or ResearchPlanner()
        self.retriever = retriever or SourceRetriever()
        self.discovery = discovery or SourceDiscovery(retriever=self.retriever, offline=self.offline)
        self.processor = processor or ResearchProcessor()
        self.evidence_store = evidence_store or EvidenceStore(device="cpu")
        self.quality_gate = quality_gate or ResearchQualityGate()

    def execute_research(self, episode_id: str) -> Dict[str, Any]:
        """
        Execute full Phase 4 research lifecycle for an episode.
        Advances state from TOPIC_SELECTED -> RESEARCHING -> RESEARCH_COMPLETE / REVIEW_REQUIRED.
        """
        t_start = time.time()
        logger.info(f"Initiating Phase 4 Research for episode '{episode_id}' (offline={self.offline})")

        # 1. Load episode record
        session = self.state_manager._get_session()
        try:
            episode = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if not episode:
                raise KeyError(f"Episode '{episode_id}' not found in database.")
            ep_dict = episode.to_dict()
        finally:
            session.close()

        ep_dir = Path(ep_dict["output_directory"])
        research_dir = ep_dir / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        raw_cache_dir = research_dir / "raw"
        raw_cache_dir.mkdir(parents=True, exist_ok=True)

        # 2. Crash Recovery Check: If dossier already exists and passes quality gate, reuse it
        dossier_path = research_dir / "dossier.json"
        if dossier_path.exists() and dossier_path.stat().st_size > 100:
            try:
                with open(dossier_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                dossier = ResearchDossier(**existing_data)
                gate_passed, reasons = self.quality_gate.evaluate(dossier)
                if gate_passed and dossier.research_status == "RESEARCH_COMPLETE":
                    logger.info(f"Recovered valid existing dossier for episode '{episode_id}'. Advancing state.")
                    curr_st = EpisodeState(ep_dict["status"])
                    if curr_st == EpisodeState.TOPIC_SELECTED:
                        self.state_manager.transition_state(
                            episode_id=episode_id,
                            target_state=EpisodeState.RESEARCHING,
                            stage_name="RESEARCHING"
                        )
                    self.state_manager.transition_state(
                        episode_id=episode_id,
                        target_state=EpisodeState.RESEARCH_COMPLETE,
                        stage_name="RESEARCH_COMPLETE"
                    )
                    return {
                        "status": "RESEARCH_COMPLETE",
                        "recovered": True,
                        "dossier": dossier.to_dict()
                    }
            except Exception as e:
                logger.warning(f"Failed to recover existing dossier: {e}, re-running research.")

        # 3. Transition to RESEARCHING
        current_state = EpisodeState(ep_dict["status"])
        if current_state == EpisodeState.TOPIC_SELECTED:
            self.state_manager.transition_state(
                episode_id=episode_id,
                target_state=EpisodeState.RESEARCHING,
                stage_name="RESEARCHING"
            )

        # Helper to get current process RSS in MB
        proc = psutil.Process() if hasattr(psutil, "Process") else None
        def get_rss() -> float:
            if proc:
                try:
                    return proc.memory_info().rss / (1024 * 1024)
                except Exception:
                    pass
            return 0.0

        rss_before = get_rss()

        # 4. Generate Research Plan
        plan_t0 = time.time()
        plan_file = research_dir / "research_plan.json"
        if plan_file.exists():
            with open(plan_file, "r", encoding="utf-8") as f:
                plan = ResearchPlan.from_dict(json.load(f))
        else:
            plan = self.planner.create_plan(
                topic=ep_dict["topic"],
                category=ep_dict.get("category"),
                metadata=ep_dict.get("extra_data", {})
            )
            with open(plan_file, "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, indent=2)
        plan_time = time.time() - plan_t0

        # 5. Discover and Retrieve Sources
        src_t0 = time.time()
        sources = self.discovery.discover_sources(plan=plan, max_sources=8)

        # Record sources in database
        for s in sources:
            self.state_manager.record_research_source(
                episode_id=episode_id,
                source_data=s.to_dict()
            )

        # Save sources.json
        sources_file = research_dir / "sources.json"
        with open(sources_file, "w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in sources], f, indent=2)
        src_time = time.time() - src_t0
        rss_after_ret = get_rss()

        # 6. Process and Chunk Documents
        proc_t0 = time.time()
        all_chunks: List[EvidenceChunk] = []
        for src in sources:
            chunks = self.processor.chunk_source(src, episode_id=episode_id)
            all_chunks.extend(chunks)

        # Correction 8: Release in-memory raw content and large cleaned strings from source objects to prevent memory bloat
        for s in sources:
            s.raw_content = None
            if s.cleaned_text and len(s.cleaned_text) > 500:
                s.cleaned_text = s.cleaned_text[:500] + "... [TRUNCATED - Full text in raw_cache_path and evidence.json]"

        # Save evidence.json
        evidence_file = research_dir / "evidence.json"
        with open(evidence_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in all_chunks], f, indent=2)
        proc_time = time.time() - proc_t0
        rss_after_proc = get_rss()

        # 7. Index in Qdrant Vector Store
        index_stats = self.evidence_store.index_chunks(
            chunks=all_chunks,
            topic=plan.topic,
            episode_id=episode_id
        )
        rss_after_embed = get_rss()
        rss_after_qdrant = get_rss()

        # 8. Retrieve Evidence per Research Question
        ret_t0 = time.time()
        question_evidence: List[Dict[str, Any]] = []
        for q in plan.research_questions:
            ev_list = self.evidence_store.retrieve_evidence(
                query=q,
                episode_id=episode_id,
                top_k=4
            )
            question_evidence.append({
                "question": q,
                "retrieved_evidence": ev_list
            })
        ret_time = time.time() - ret_t0

        # 9. Build Research Dossier
        dossier = ResearchDossierBuilder.build_dossier(
            episode_id=episode_id,
            plan=plan,
            sources=sources,
            chunks=all_chunks,
            question_evidence=question_evidence,
            research_status="PENDING_QUALITY_GATE"
        )

        # 10. Quality Gate Evaluation
        gate_passed, gate_reasons = self.quality_gate.evaluate(dossier)
        total_time = time.time() - t_start

        # Record system resource snapshot and memory benchmark
        cpu_pct = psutil.cpu_percent() if hasattr(psutil, "cpu_percent") else 0.0
        ram_gb = psutil.virtual_memory().used / (1024 ** 3) if hasattr(psutil, "virtual_memory") else 0.0
        peak_rss = max(rss_before, rss_after_ret, rss_after_proc, rss_after_embed, rss_after_qdrant)

        timing_stats = {
            "planning_time_s": round(plan_time, 2),
            "source_discovery_time_s": round(src_time, 2),
            "document_processing_time_s": round(proc_time, 2),
            "embedding_time_s": index_stats.get("embedding_time_s", 0.0),
            "qdrant_time_s": index_stats.get("qdrant_time_s", 0.0),
            "evidence_retrieval_time_s": round(ret_time, 2),
            "total_research_time_s": round(total_time, 2),
            "total_runtime_seconds": round(total_time, 2),
            "rss_before_mb": round(rss_before, 2),
            "rss_after_retrieval_mb": round(rss_after_ret, 2),
            "rss_after_chunking_mb": round(rss_after_proc, 2),
            "rss_after_embedding_mb": round(rss_after_embed, 2),
            "rss_after_qdrant_mb": round(rss_after_qdrant, 2),
            "peak_rss_mb": round(peak_rss, 2),
            "cpu_usage_pct": cpu_pct,
            "ram_used_gb": round(ram_gb, 2),
            "gpu_usage": "0% (CPU execution only)"
        }

        # 11. Finalize Dossier and Persist Artifacts
        final_status = "RESEARCH_COMPLETE" if gate_passed else "REVIEW_REQUIRED"
        dossier.research_status = final_status

        dossier_data = dossier.to_dict()
        dossier_data["timing_and_resources"] = timing_stats
        dossier_data["quality_gate_passed"] = gate_passed
        dossier_data["quality_gate_reasons"] = gate_reasons

        with open(dossier_path, "w", encoding="utf-8") as f:
            json.dump(dossier_data, f, indent=2)

        # Also write research_evidence.json for backward compatibility with Phase 2 recovery manager
        compat_evidence_file = research_dir / "research_evidence.json"
        with open(compat_evidence_file, "w", encoding="utf-8") as f:
            json.dump({
                "episode_id": episode_id,
                "topic": plan.topic,
                "status": final_status,
                "source_count": len(sources),
                "chunk_count": len(all_chunks),
                "dossier_path": str(dossier_path),
                "sources": [s.to_dict() for s in sources],
                "preliminary_findings": dossier_data["preliminary_findings"]
            }, f, indent=2)

        # 12. Transition Episode State
        if gate_passed:
            confidence = min(1.0, 0.70 + (0.05 * len(sources)) + (0.05 * dossier.source_statistics["high_quality_sources_count"]))
            self.state_manager.update_research_confidence(episode_id, confidence=round(confidence, 2))
            self.state_manager.transition_state(
                episode_id=episode_id,
                target_state=EpisodeState.RESEARCH_COMPLETE,
                stage_name="RESEARCH_COMPLETE"
            )
            logger.info(f"Research successfully passed Quality Gate. State advanced to RESEARCH_COMPLETE.")
        else:
            self.state_manager.transition_state(
                episode_id=episode_id,
                target_state=EpisodeState.REVIEW_REQUIRED,
                stage_name="REVIEW_REQUIRED"
            )
            self.state_manager.record_failure(
                episode_id=episode_id,
                error_message=f"Research Quality Gate failed: {'; '.join(gate_reasons)}"
            )
            logger.warning(f"Research Quality Gate failed for '{episode_id}': {gate_reasons}. Routed to REVIEW_REQUIRED.")

        return {
            "status": final_status,
            "quality_gate_passed": gate_passed,
            "quality_gate_reasons": gate_reasons,
            "sources_count": len(sources),
            "chunks_count": len(all_chunks),
            "timing": timing_stats,
            "dossier_path": str(dossier_path),
            "dossier": dossier_data
        }
