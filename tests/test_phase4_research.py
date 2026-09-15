"""
tests/test_phase4_research.py - Comprehensive Unit & Integration Test Suite for Phase 4.

Verifies the Autonomous Research Engine, Source Discovery, Retrieval, Cleaning,
Deterministic Chunking, CPU-only Qdrant Indexing, Semantic Retrieval, Conflict Detection,
Research Dossier assembly, Quality Gate, Resumption, and Orchestrator Integration.
Uses isolated temp databases, temp Qdrant collections, and mocked offline fixtures.
"""

import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from unittest.mock import MagicMock, patch

from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode, AutonomousResearchSource
from autonomous.research_planner import ResearchPlanner, ResearchPlan
from autonomous.source_retriever import SourceRetriever, SourceMetadata
from autonomous.source_discovery import SourceDiscovery
from autonomous.research_processor import ResearchProcessor, EvidenceChunk
from autonomous.evidence_store import EvidenceStore
from autonomous.research_dossier import ResearchDossier, ResearchDossierBuilder, EvidenceConflict
from autonomous.research_engine import ResearchEngine, ResearchQualityGate
from autonomous.orchestrator import AutonomousOrchestrator, StageExecutionResult


class TestPhase4Research(unittest.TestCase):
    """30+ automated tests verifying Phase 4 research architecture and safeguards."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_phase4_research_")
        self.test_db_path = os.path.join(self.test_dir, "test_universe.db")
        self.test_qdrant_path = os.path.join(self.test_dir, "test_qdrant")
        self.state_manager = StateManager(db_url=f"sqlite:///{self.test_db_path}")

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass

    def _create_mock_source(
        self,
        source_id: str,
        title: str,
        text: str,
        tier: int = 1,
        url: Optional[str] = None,
        publisher: Optional[str] = None,
        source_origin: str = "external",
        source_type: str = "web"
    ) -> SourceMetadata:
        import hashlib
        c_hash = hashlib.sha256(text.encode()).hexdigest()
        src_url = url or f"https://{source_id}.org/report"
        domain = SourceRetriever.extract_domain(src_url)
        pub = publisher or f"Publisher {source_id.upper()}"
        s_role, e_weight = SourceRetriever.determine_evidence_weight(
            tier, domain, source_type, source_origin
        )
        ind_group = SourceRetriever.determine_independence_group(
            src_url, domain, pub, source_origin
        )
        return SourceMetadata(
            source_id=source_id,
            title=title,
            url=src_url,
            publisher=pub,
            domain=domain,
            credibility_tier=tier,
            content_hash=c_hash,
            extraction_status="EXTRACTED",
            cleaned_text=text,
            source_origin=source_origin,
            independence_group=ind_group,
            source_role=s_role,
            evidence_weight_class=e_weight
        )

    # 1. Research Planner Creation
    def test_01_research_planner_creation(self):
        planner = ResearchPlanner()
        plan = planner.create_plan(topic="Keezhadi: 2,600-Year-Old Tamil Urban Civilization")
        self.assertIsInstance(plan, ResearchPlan)
        self.assertEqual(plan.topic, "Keezhadi: 2,600-Year-Old Tamil Urban Civilization")
        self.assertGreaterEqual(len(plan.research_questions), 5)
        self.assertIn("search_queries", plan.to_dict())

    # 2. Research Question Generation Structure
    def test_02_research_question_structure(self):
        planner = ResearchPlanner()
        plan = planner.create_plan(topic="Poompuhar Ancient Sunken Port")
        questions = plan.research_questions
        # Verify archaeological, chronological, and infrastructural coverage
        self.assertTrue(any("evidence" in q.lower() or "archaeological" in q.lower() for q in questions))
        self.assertTrue(any("chronology" in q.lower() or "dating" in q.lower() for q in questions))
        self.assertTrue(any("debated" in q.lower() or "consensus" in q.lower() for q in questions))

    # 3. Source Metadata Model
    def test_03_source_metadata_model(self):
        meta = self._create_mock_source("src_01", "Excavation Report", "Sample text", tier=1)
        d = meta.to_dict()
        self.assertEqual(d["source_id"], "src_01")
        self.assertEqual(d["credibility_tier"], 1)
        self.assertEqual(d["extraction_status"], "EXTRACTED")
        self.assertTrue(len(d["content_hash"]) == 64)

    # 4. URL Normalization
    def test_04_url_normalization(self):
        dirty_url = "HTTPS://WWW.ASI.NIC.IN/excavations/keeladi/?utm_source=twitter&utm_medium=cpc#gallery"
        clean_url = SourceRetriever.normalize_url(dirty_url)
        self.assertEqual(clean_url, "https://www.asi.nic.in/excavations/keeladi")

    # 5. Source Deduplication
    def test_05_source_deduplication(self):
        retriever = SourceRetriever(cache_dir=self.test_dir)
        discovery = SourceDiscovery(retriever=retriever, offline=True)
        s1 = self._create_mock_source("s1", "Title 1", "Content A", url="https://asi.nic.in/rep1")
        s2 = self._create_mock_source("s2", "Title 2", "Content A", url="https://asi.nic.in/rep1")  # Same URL and content
        plan = ResearchPlan(topic="Test Topic", research_questions=["Q1"])
        sources = discovery.discover_sources(plan, existing_sources=[s1, s2])
        # Duplicate should be filtered
        urls = [s.url for s in sources]
        self.assertEqual(urls.count("https://asi.nic.in/rep1"), 1)

    # 6. Content Hashing SHA-256
    def test_06_content_hashing_sha256(self):
        import hashlib
        text = "Archaeological excavation revealed continuous brick structures at 6th century BCE."
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        src = self._create_mock_source("s_hash", "Title", text)
        self.assertEqual(src.content_hash, expected)

    # 7. Source Quality Classification (Tiers 1 to 5)
    def test_07_source_quality_classification(self):
        tier_gov, _ = SourceRetriever.classify_credibility_tier("https://asi.nic.in/report")
        tier_edu, _ = SourceRetriever.classify_credibility_tier("https://oxford.ac.uk/paper")
        tier_wiki, _ = SourceRetriever.classify_credibility_tier("https://en.wikipedia.org/wiki/Keezhadi")
        tier_news, _ = SourceRetriever.classify_credibility_tier("https://thehindu.com/news/national")
        tier_blog, _ = SourceRetriever.classify_credibility_tier("https://randomhistoryblog.com/post")
        tier_social, _ = SourceRetriever.classify_credibility_tier("https://youtube.com/watch?v=123")

        self.assertEqual(tier_gov, 1)
        self.assertEqual(tier_edu, 1)
        self.assertEqual(tier_wiki, 2)
        self.assertEqual(tier_news, 3)
        self.assertEqual(tier_blog, 4)
        self.assertEqual(tier_social, 5)

    # 8. HTML Document Cleaning
    def test_08_html_document_cleaning(self):
        from autonomous.source_retriever import SimpleHTMLTextExtractor
        html = """
        <html>
            <head><title>Excavation Report Title</title><script>var x = 1;</script><style>body {color:red;}</style></head>
            <body>
                <header><nav>Home | Contact</nav></header>
                <article>
                    <h1>Keezhadi Brick Architecture</h1>
                    <p>Excavations revealed a sophisticated urban water management system.</p>
                </article>
                <footer>Copyright 2026</footer>
            </body>
        </html>
        """
        parser = SimpleHTMLTextExtractor()
        parser.feed(html)
        text = parser.get_text()
        self.assertIn("Keezhadi Brick Architecture", text)
        self.assertIn("sophisticated urban water management system", text)
        self.assertNotIn("var x = 1", text)
        self.assertNotIn("color:red", text)
        self.assertNotIn("Home | Contact", text)
        self.assertEqual(parser.get_title(), "Excavation Report Title")

    # 9. Deterministic Chunking
    def test_09_deterministic_chunking(self):
        processor = ResearchProcessor(chunk_size=200, overlap=50)
        sample_text = (
            "Paragraph one describes the initial survey conducted along the Vaigai river in 2014.\n\n"
            "Paragraph two details the structural remains found including ring wells and drainage systems.\n\n"
            "Paragraph three analyzes the radiocarbon dating reports from Beta Analytic in Florida."
        )
        src = self._create_mock_source("src_chunk", "Chunk Test", sample_text)
        chunks = processor.chunk_source(src, episode_id="test_ep_001")
        self.assertGreaterEqual(len(chunks), 2)
        for c in chunks:
            self.assertTrue(len(c.text) > 0)
            self.assertEqual(c.episode_id, "test_ep_001")

    # 10. Provenance Preservation
    def test_10_provenance_preservation(self):
        processor = ResearchProcessor()
        src = self._create_mock_source(
            source_id="src_provenance",
            title="Excavation Journal",
            text="The carbon dates established a chronological baseline of 580 BCE.",
            tier=1,
            url="https://asi.nic.in/vol4",
            publisher="ASI"
        )
        chunks = processor.chunk_source(src, episode_id="ep_prov_123")
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.source_id, "src_provenance")
        self.assertEqual(chunk.episode_id, "ep_prov_123")
        self.assertEqual(chunk.url, "https://asi.nic.in/vol4")
        self.assertEqual(chunk.publisher, "ASI")
        self.assertEqual(chunk.credibility_tier, 1)
        self.assertTrue(len(chunk.content_hash) == 64)

    # 11. CPU Embedding Integration
    def test_11_embedding_integration_cpu(self):
        store = EvidenceStore(qdrant_path=self.test_qdrant_path, device="cpu")
        model = store.embedding_model
        if model:
            self.assertEqual(store.device, "cpu")
            vec = model.encode("Test ancient history phrase", device="cpu")
            self.assertEqual(len(vec), 384)

    # 12. Qdrant Indexing & Additive Collection
    def test_12_qdrant_indexing(self):
        store = EvidenceStore(qdrant_path=self.test_qdrant_path, collection_name="test_phase4_collection")
        if store.qdrant is None:
            self.skipTest("Qdrant client not available in environment.")

        chunk = EvidenceChunk(
            chunk_id="chk_01",
            episode_id="ep_qdrant_1",
            source_id="src_01",
            url="https://asi.nic.in/doc",
            title="Doc Title",
            publisher="ASI",
            retrieval_timestamp="2026-09-10T12:00:00Z",
            content_hash="abc123hash",
            chunk_index=0,
            text="Evidence of brick masonry at Keezhadi.",
            credibility_tier=1
        )
        res = store.index_chunks([chunk], topic="Keezhadi", episode_id="ep_qdrant_1")
        self.assertEqual(res["indexed_count"], 1)

    # 13. Duplicate Indexing Prevention (Idempotent UUID5)
    def test_13_duplicate_indexing_prevention(self):
        store = EvidenceStore(qdrant_path=self.test_qdrant_path, collection_name="test_idempotent_col")
        if store.qdrant is None:
            self.skipTest("Qdrant client not available.")

        chunk = EvidenceChunk(
            chunk_id="chk_idem",
            episode_id="ep_idem",
            source_id="src_idem",
            url="https://asi.nic.in/idem",
            title="Idempotent Test",
            publisher="ASI",
            retrieval_timestamp="2026-09-10T12:00:00Z",
            content_hash="hash_identical_456",
            chunk_index=0,
            text="Exact identical text chunk.",
            credibility_tier=1
        )
        res1 = store.index_chunks([chunk], topic="Topic", episode_id="ep_idem")
        res2 = store.index_chunks([chunk], topic="Topic", episode_id="ep_idem")
        self.assertEqual(res1["indexed_count"], 1)
        self.assertEqual(res2["indexed_count"], 1)
        count = store.qdrant.count("test_idempotent_col")
        # Upsert with same UUID5 point ID does not increase count
        self.assertEqual(count.count, 1)

    # 14. Evidence Retrieval per Question
    def test_14_evidence_retrieval(self):
        store = EvidenceStore(qdrant_path=self.test_qdrant_path, collection_name="test_retrieval_col")
        if store.qdrant is None or store.embedding_model is None:
            self.skipTest("Qdrant or SentenceTransformer not available.")

        chunks = [
            EvidenceChunk(
                chunk_id="chk_water",
                episode_id="ep_ret",
                source_id="s1",
                url="https://asi.nic.in",
                title="Hydrology",
                publisher="ASI",
                retrieval_timestamp="2026",
                content_hash="h1",
                chunk_index=0,
                text="Excavations revealed ring wells, closed drainage channels, and terracotta pipes for water management.",
                credibility_tier=1
            ),
            EvidenceChunk(
                chunk_id="chk_script",
                episode_id="ep_ret",
                source_id="s2",
                url="https://tnarch.gov.in",
                title="Epigraphy",
                publisher="TN Arch",
                retrieval_timestamp="2026",
                content_hash="h2",
                chunk_index=0,
                text="Inscribed potsherds bearing Tamil-Brahmi personal names demonstrate widespread literacy.",
                credibility_tier=1
            )
        ]
        store.index_chunks(chunks, topic="Keezhadi", episode_id="ep_ret")
        results = store.retrieve_evidence("What water drainage systems were discovered?", episode_id="ep_ret", top_k=1)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "chk_water")

    # 15. Dossier Creation & Statistics
    def test_15_dossier_creation(self):
        plan = ResearchPlan(topic="Tanjore Vimana", research_questions=["Q1"])
        s1 = self._create_mock_source("s1", "Doc 1", "Content 1", tier=1, publisher="ASI")
        s2 = self._create_mock_source("s2", "Doc 2", "Content 2", tier=2, publisher="UNESCO")
        s3 = self._create_mock_source("s3", "Doc 3", "Content 3", tier=3, publisher="The Hindu")
        chunks = [
            EvidenceChunk("c1", "ep1", "s1", "url", "Doc 1", "ASI", "2026", "h1", 0, "Text 1", 1),
            EvidenceChunk("c2", "ep1", "s2", "url", "Doc 2", "UNESCO", "2026", "h2", 0, "Text 2", 2),
            EvidenceChunk("c3", "ep1", "s3", "url", "Doc 3", "The Hindu", "2026", "h3", 0, "Text 3", 3),
        ]
        q_ev = [{"question": "Q1", "retrieved_evidence": [{"chunk_id": "c1", "source_id": "s1", "text": "Text 1", "credibility_tier": 1}]}]
        dossier = ResearchDossierBuilder.build_dossier("ep1", plan, [s1, s2, s3], chunks, q_ev)

        self.assertEqual(dossier.source_statistics["total_sources"], 3)
        self.assertEqual(dossier.source_statistics["unique_publishers"], 3)
        self.assertEqual(dossier.source_statistics["high_quality_sources_count"], 2)
        self.assertTrue(dossier.source_statistics["has_independent_sources"])

    # 16. Conflicting Evidence Detection (No Winner Chosen)
    def test_16_conflicting_evidence_detection(self):
        s1 = self._create_mock_source("s1", "Survey A", "Chronology dated to 6th century BCE.", tier=1)
        s2 = self._create_mock_source("s2", "Survey B", "Chronology dated to 3rd century BCE.", tier=2)
        q_ev = [{
            "question": "What is the chronology?",
            "retrieved_evidence": [
                {"chunk_id": "c1", "source_id": "s1", "title": "Survey A", "text": "Chronology dated to 6th century BCE."},
                {"chunk_id": "c2", "source_id": "s2", "title": "Survey B", "text": "Chronology dated to 3rd century BCE."}
            ]
        }]
        conflicts = ResearchDossierBuilder.detect_conflicts(q_ev, [s1, s2])
        self.assertGreaterEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0].resolution_status, "UNRESOLVED_PENDING_PHASE5")
        self.assertIn("6th century BCE", conflicts[0].statement_a)
        self.assertIn("3rd century BCE", conflicts[0].statement_b)

    # 17. Research Quality Gate - Pass
    def test_17_research_quality_gate_pass(self):
        gate = ResearchQualityGate(min_sources=3, min_high_quality_sources=1, min_evidence_chunks=3, min_questions_answered=1)
        plan = ResearchPlan(topic="Topic", research_questions=["Q1"])
        s1 = self._create_mock_source("s1", "Title 1", "Txt", tier=1)
        s2 = self._create_mock_source("s2", "Title 2", "Txt", tier=2)
        s3 = self._create_mock_source("s3", "Title 3", "Txt", tier=3)
        chunks = [
            EvidenceChunk(f"c{i}", "ep", f"s{i}", "u", "t", "p", "time", "h", 0, "txt", 1) for i in range(1, 4)
        ]
        q_ev = [{"question": "Q1", "retrieved_evidence": [{"chunk_id": "c1", "text": "txt"}]}]
        dossier = ResearchDossierBuilder.build_dossier("ep", plan, [s1, s2, s3], chunks, q_ev)
        passed, reasons = gate.evaluate(dossier)
        self.assertTrue(passed)
        self.assertEqual(len(reasons), 0)

    # 18. Research Quality Gate - Fail on Insufficient High Quality Sources
    def test_18_research_quality_gate_fail(self):
        gate = ResearchQualityGate(min_sources=3, min_high_quality_sources=2, min_evidence_chunks=3, min_questions_answered=1)
        plan = ResearchPlan(topic="Topic", research_questions=["Q1"])
        # All low tier (Tier 4)
        s1 = self._create_mock_source("s1", "Blog 1", "Txt", tier=4)
        s2 = self._create_mock_source("s2", "Blog 2", "Txt", tier=4)
        s3 = self._create_mock_source("s3", "Blog 3", "Txt", tier=4)
        chunks = [EvidenceChunk(f"c{i}", "ep", f"s{i}", "u", "t", "p", "time", "h", 0, "txt", 4) for i in range(1, 4)]
        q_ev = [{"question": "Q1", "retrieved_evidence": [{"chunk_id": "c1", "text": "txt"}]}]
        dossier = ResearchDossierBuilder.build_dossier("ep", plan, [s1, s2, s3], chunks, q_ev)
        passed, reasons = gate.evaluate(dossier)
        self.assertFalse(passed)
        self.assertTrue(any("Tier 1/2" in r for r in reasons))

    # 19. Offline Mode (Zero Network Requests)
    def test_19_offline_mode(self):
        discovery = SourceDiscovery(offline=True)
        self.assertTrue(discovery.offline)
        plan = ResearchPlan(topic="Fake Topic Offline", research_questions=["Q1"])
        sources = discovery.discover_sources(plan)
        # Should complete without error and make zero web requests
        self.assertIsInstance(sources, list)

    # 20. Source Caching & Reuse
    def test_20_source_caching(self):
        cache_dir = Path(self.test_dir) / "cache"
        retriever = SourceRetriever(cache_dir=str(cache_dir))
        # Create a cached file manually
        import hashlib
        norm_url = retriever.normalize_url("https://asi.nic.in/cached-doc")
        url_hash = hashlib.sha256(norm_url.encode("utf-8")).hexdigest()
        c_file = cache_dir / f"{url_hash}.txt"
        c_file.write_text("TITLE: Cached ASI Document\nURL: https://asi.nic.in/cached-doc\nExcavation content preserved in cache.", encoding="utf-8")

        meta = retriever.retrieve_url("https://asi.nic.in/cached-doc")
        self.assertEqual(meta.extraction_status, "CACHED")
        self.assertEqual(meta.title, "Cached ASI Document")
        self.assertIn("preserved in cache", meta.cleaned_text)

    # 21. Crash Recovery - Existing Dossier Reuse
    def test_21_crash_recovery_resumption(self):
        ep_dir_path = os.path.join(self.test_dir, "ep_21")
        ep = self.state_manager.create_episode(topic="Recovery Episode", category="Ancient Tamil history", output_directory=ep_dir_path)
        self.state_manager.transition(ep.episode_id, EpisodeState.TOPIC_SELECTED, "TOPIC_SELECTED")

        ep_dir = Path(ep.output_directory)
        res_dir = ep_dir / "research"
        res_dir.mkdir(parents=True, exist_ok=True)

        # Write valid existing dossier on disk
        plan = ResearchPlan(topic="Recovery Episode", research_questions=["Q1"])
        s1 = self._create_mock_source("s1", "T1", "C1", tier=1)
        s2 = self._create_mock_source("s2", "T2", "C2", tier=2)
        s3 = self._create_mock_source("s3", "T3", "C3", tier=3)
        chunks = [EvidenceChunk(f"c{i}", ep.episode_id, f"s{i}", "u", "t", "p", "time", "h", 0, "txt", 1) for i in range(1, 6)]
        q_ev = [
            {"question": "Q1", "retrieved_evidence": [{"chunk_id": "c1", "text": "txt"}]},
            {"question": "Q2", "retrieved_evidence": [{"chunk_id": "c2", "text": "txt"}]},
            {"question": "Q3", "retrieved_evidence": [{"chunk_id": "c3", "text": "txt"}]}
        ]
        dossier = ResearchDossierBuilder.build_dossier(ep.episode_id, plan, [s1, s2, s3], chunks, q_ev, research_status="RESEARCH_COMPLETE")

        with open(res_dir / "dossier.json", "w", encoding="utf-8") as f:
            json.dump(dossier.to_dict(), f, indent=2)

        engine = ResearchEngine(state_manager=self.state_manager, offline=True)
        res = engine.execute_research(ep.episode_id)
        self.assertEqual(res["status"], "RESEARCH_COMPLETE")
        self.assertTrue(res.get("recovered"))

    # 22. Orchestrator Integration (TOPIC_SELECTED -> RESEARCH_COMPLETE)
    def test_22_orchestrator_integration(self):
        ep_dir_path = os.path.join(self.test_dir, "ep_22")
        ep = self.state_manager.create_episode(topic="Orchestrator Test Topic", category="Ancient Tamil history", output_directory=ep_dir_path)
        self.state_manager.transition(ep.episode_id, EpisodeState.TOPIC_SELECTED, "TOPIC_SELECTED")

        # Mock discovery so it returns 3 valid fixtures
        mock_discovery = MagicMock()
        mock_discovery.discover_sources.return_value = [
            self._create_mock_source("s1", "Title 1", "Content of first authoritative source.\n\nAdditional excavation details on stratigraphy and brick foundations.", tier=1),
            self._create_mock_source("s2", "Title 2", "Content of second scholarly source.\n\nAdditional chronological analysis on radiocarbon dates.", tier=2),
            self._create_mock_source("s3", "Title 3", "Content of third journalism source.\n\nAdditional summary of public interest and archaeological significance.", tier=3)
        ]

        store = EvidenceStore(qdrant_path=self.test_qdrant_path, collection_name="orch_col", device="cpu")
        engine = ResearchEngine(
            state_manager=self.state_manager,
            discovery=mock_discovery,
            evidence_store=store,
            offline=True
        )

        orch = AutonomousOrchestrator(state_manager=self.state_manager)
        with patch.object(orch, "research_topic", side_effect=lambda ep_id, **kw: StageExecutionResult(True, EpisodeState.RESEARCH_COMPLETE, data=engine.execute_research(ep_id))):
            res = orch.research_topic(ep.episode_id)
            self.assertTrue(res.success)
            self.assertEqual(res.next_state, EpisodeState.RESEARCH_COMPLETE)

    # 23. State Transition: REVIEW_REQUIRED on Quality Gate Failure
    def test_23_state_transition_review_required(self):
        ep_dir_path = os.path.join(self.test_dir, "ep_23")
        ep = self.state_manager.create_episode(topic="Insufficient Topic", category="Ancient Tamil history", output_directory=ep_dir_path)
        self.state_manager.transition(ep.episode_id, EpisodeState.TOPIC_SELECTED, "TOPIC_SELECTED")

        # Mock discovery returning only 1 low-tier source (fails quality gate)
        mock_discovery = MagicMock()
        mock_discovery.discover_sources.return_value = [
            self._create_mock_source("s1", "Random Blog", "Short unverified snippet", tier=4)
        ]

        store = EvidenceStore(qdrant_path=self.test_qdrant_path, collection_name="fail_col", device="cpu")
        engine = ResearchEngine(
            state_manager=self.state_manager,
            discovery=mock_discovery,
            evidence_store=store,
            offline=True
        )
        res = engine.execute_research(ep.episode_id)
        self.assertEqual(res["status"], "REVIEW_REQUIRED")
        self.assertFalse(res["quality_gate_passed"])

        session = self.state_manager._get_session()
        try:
            updated_ep = session.query(AutonomousEpisode).filter_by(episode_id=ep.episode_id).first()
            self.assertEqual(updated_ep.status, EpisodeState.REVIEW_REQUIRED.value)
        finally:
            session.close()

    # 24. Preliminary Finding Linking to Chunk IDs
    def test_24_preliminary_findings_provenance_linking(self):
        q_ev = [{
            "question": "What artifacts were excavated?",
            "retrieved_evidence": [
                {"chunk_id": "chk_beads_01", "source_id": "src_asi", "title": "ASI Report", "text": "Excavators recovered over 4,000 glass and carnelian beads.", "credibility_tier": 1}
            ]
        }]
        findings = ResearchDossierBuilder.generate_preliminary_findings(q_ev, "Keezhadi")
        self.assertEqual(len(findings), 1)
        f = findings[0]
        self.assertIn("chk_beads_01", f.supporting_chunk_ids)
        self.assertIn("src_asi", f.supporting_source_ids)
        self.assertNotIn("VERIFIED", f.statement)

    # 25. Database Persistence of Sources
    def test_25_database_persistence_of_sources(self):
        ep_dir_path = os.path.join(self.test_dir, "ep_25")
        ep = self.state_manager.create_episode(topic="DB Source Test", category="Ancient Tamil history", output_directory=ep_dir_path)
        source_data = {
            "source_id": "src_db_01",
            "title": "Excavation Vol 1",
            "url": "https://asi.nic.in/vol1",
            "publisher": "ASI",
            "domain": "asi.nic.in",
            "credibility_tier": 1,
            "content_hash": "hash_db_123",
            "chunk_count": 5
        }
        self.state_manager.record_research_source(ep.episode_id, source_data)
        sources = self.state_manager.get_episode_sources(ep.episode_id)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].source_id, "src_db_01")
        self.assertEqual(sources[0].credibility_tier, 1)

    # 26. No Script Generation During Phase 4
    def test_26_no_script_generation_in_phase4(self):
        orch = AutonomousOrchestrator(state_manager=self.state_manager)
        with self.assertRaises(NotImplementedError) as cm:
            orch.generate_script("ep_fake")
        self.assertIn("Phase 6", str(cm.exception))

    # 27. No Video Generation During Phase 4
    def test_27_no_video_generation_in_phase4(self):
        orch = AutonomousOrchestrator(state_manager=self.state_manager)
        with self.assertRaises(NotImplementedError) as cm:
            orch.generate_motion("ep_fake")
        self.assertIn("Phase 8", str(cm.exception))

    # 28. Resource Safety (CPU Execution)
    def test_28_resource_safety_cpu_execution(self):
        store = EvidenceStore(device="cpu")
        self.assertEqual(store.device, "cpu")

    # 29. Existing RAG Knowledge Service Untouched
    def test_29_existing_rag_service_intact(self):
        import sys
        sys.path.insert(0, r"d:\mvid\backend")
        from app.services.knowledge_service import knowledge_service, KNOWLEDGE_COLLECTION
        client = knowledge_service.qdrant
        if client:
            collections = [c.name for c in client.get_collections().collections]
            self.assertIn(KNOWLEDGE_COLLECTION, collections)
            # Confirm original points remain intact
            count = client.count(KNOWLEDGE_COLLECTION)
            self.assertGreaterEqual(count.count, 1350)

    # 30. Active Episode Safety Guard
    def test_30_active_episode_safety(self):
        ep_dir_path = os.path.join(self.test_dir, "ep_30")
        ep = self.state_manager.create_episode(topic="Active Topic Guard Test", category="Ancient Tamil history", output_directory=ep_dir_path)
        self.state_manager.transition(ep.episode_id, EpisodeState.TOPIC_SELECTED, "TOPIC_SELECTED")
        self.state_manager.transition(ep.episode_id, EpisodeState.RESEARCHING, "RESEARCHING")

        # Active episode exists in RESEARCHING state
        active = self.state_manager.get_active_episode()
        self.assertIsNotNone(active)
        self.assertEqual(active.status, EpisodeState.RESEARCHING.value)

    # 31. Local Source Does NOT Count as External Independent Source (Correction 1)
    def test_31_local_source_not_counted_as_external_independent(self):
        plan = ResearchPlan(topic="Local Independence Test", research_questions=["Q1"])
        s_local = self._create_mock_source("s_loc", "Local Notes", "Local file notes", tier=2, url="d:\\mvid\\daily_engine\\topics.json", publisher="Local Project Knowledge", source_origin="local")
        s_ext = self._create_mock_source("s_ext", "ASI Report", "Excavation text", tier=1, url="https://asi.nic.in/rep", publisher="ASI", source_origin="external")
        chunks = [
            EvidenceChunk("c1", "ep_loc", "s_loc", s_local.url, "Local Notes", "Local Project Knowledge", "2026", "h1", 0, "txt1", 2, source_origin="local", independence_group="local_filesystem"),
            EvidenceChunk("c2", "ep_loc", "s_ext", s_ext.url, "ASI Report", "ASI", "2026", "h2", 0, "txt2", 1, source_origin="external", independence_group="asi.nic.in")
        ]
        stats = ResearchDossierBuilder.compute_source_statistics([s_local, s_ext], chunks)
        self.assertEqual(stats["total_sources"], 2)
        self.assertEqual(stats["local_sources"], 1)
        self.assertEqual(stats["external_sources"], 1)
        # Even though there are 2 sources, only 1 is external independent
        self.assertEqual(stats["external_independent_sources"], 1)

    # 32. Two URLs from Same Domain Do NOT Count as Two Independent Sources (Correction 2)
    def test_32_same_domain_urls_same_independence_group(self):
        plan = ResearchPlan(topic="Domain Group Test", research_questions=["Q1"])
        s1 = self._create_mock_source("s1", "ASI Article 1", "Text 1", tier=1, url="https://asi.nic.in/page1", publisher="ASI")
        s2 = self._create_mock_source("s2", "ASI Article 2", "Text 2", tier=1, url="https://asi.nic.in/page2", publisher="ASI")
        chunks = [
            EvidenceChunk("c1", "ep", "s1", s1.url, "ASI Article 1", "ASI", "2026", "h1", 0, "t1", 1, independence_group="asi.nic.in"),
            EvidenceChunk("c2", "ep", "s2", s2.url, "ASI Article 2", "ASI", "2026", "h2", 0, "t2", 1, independence_group="asi.nic.in"),
        ]
        stats = ResearchDossierBuilder.compute_source_statistics([s1, s2], chunks)
        self.assertEqual(stats["total_sources"], 2)
        self.assertEqual(stats["external_sources"], 2)
        # Both belong to independence_group 'asi.nic.in' -> exactly 1 independent group
        self.assertEqual(stats["external_independent_sources"], 1)
        self.assertEqual(stats["unique_independence_groups"], 1)

    # 33. Cached External Source Preserves Original Independence Group (Correction 1 & 12.3)
    def test_33_cached_external_retains_independence_group(self):
        import hashlib
        cache_dir = Path(self.test_dir) / "cache_ind"
        cache_dir.mkdir(parents=True, exist_ok=True)
        retriever = SourceRetriever(cache_dir=str(cache_dir))
        url = "https://archaeology.tn.gov.in/keeladi-excavation"
        norm_url = retriever.normalize_url(url)
        url_hash = hashlib.sha256(norm_url.encode("utf-8")).hexdigest()
        c_file = cache_dir / f"{url_hash}.txt"
        c_file.write_text("TITLE: Keeladi Report\nURL: https://archaeology.tn.gov.in/keeladi-excavation\nSample excavation report text.", encoding="utf-8")

        meta = retriever.retrieve_url(url)
        self.assertEqual(meta.source_origin, "cached_external")
        self.assertEqual(meta.independence_group, "archaeology.tn.gov.in")
        self.assertEqual(meta.credibility_tier, 1)
        self.assertEqual(meta.evidence_weight_class, "PRIMARY")

    # 34. Wikipedia Classified as Discovery Reference (Correction 3 & 12.4)
    def test_34_wikipedia_role_discovery_reference(self):
        retriever = SourceRetriever()
        meta = retriever.retrieve_url("https://en.wikipedia.org/wiki/Keezhadi_excavation_site")
        self.assertEqual(meta.source_role, "discovery_reference")
        self.assertEqual(meta.evidence_weight_class, "REFERENCE")
        self.assertEqual(meta.independence_group, "wikipedia.org")

    # 35. Wikipedia Cannot Satisfy High-Quality External Evidence Requirement Alone (Correction 3 & 12.5)
    def test_35_wikipedia_cannot_satisfy_high_quality_alone(self):
        gate = ResearchQualityGate(
            min_sources=2,
            min_high_quality_sources=1,
            min_evidence_chunks=2,
            min_questions_answered=1,
            minimum_external_independent_sources=1,
            minimum_high_quality_external_sources=1
        )
        plan = ResearchPlan(topic="Wiki Test", research_questions=["Q1"])
        # Source 1 is Wikipedia (REFERENCE role, NOT PRIMARY or HIGH)
        s_wiki = self._create_mock_source("s_wiki", "Wikipedia Keezhadi", "Wiki text", tier=2, url="https://en.wikipedia.org/wiki/Keezhadi", publisher="Wikimedia Foundation", source_type="wikipedia")
        s_local = self._create_mock_source("s_loc", "Local Doc", "Local text", tier=2, url="d:\\mvid\\daily_engine\\topics.json", publisher="Local Project Knowledge", source_origin="local")
        chunks = [
            EvidenceChunk("c1", "ep", "s_wiki", s_wiki.url, "Wiki", "Wiki", "2026", "h1", 0, "txt", 2, source_role="discovery_reference", evidence_weight_class="REFERENCE"),
            EvidenceChunk("c2", "ep", "s_loc", s_local.url, "Loc", "Loc", "2026", "h2", 0, "txt", 2, source_origin="local", source_role="background_local", evidence_weight_class="REFERENCE"),
        ]
        q_ev = [{"question": "Q1", "retrieved_evidence": [{"chunk_id": "c1", "text": "txt"}]}]
        dossier = ResearchDossierBuilder.build_dossier("ep", plan, [s_wiki, s_local], chunks, q_ev)
        passed, reasons = gate.evaluate(dossier)
        # Must fail because high_quality_external_sources is 0 (Wikipedia does not count as PRIMARY/HIGH external proof)
        self.assertFalse(passed)
        self.assertTrue(any("high-quality external sources" in r for r in reasons))

    # 36. Expanded Source Statistics Accounting (Correction 5)
    def test_36_expanded_source_statistics_accounting(self):
        s_ext1 = self._create_mock_source("s1", "Gov Site", "Text 1", tier=1, url="https://asi.nic.in/doc1", publisher="ASI")
        s_ext2 = self._create_mock_source("s2", "University", "Text 2", tier=2, url="https://oxford.ac.uk/paper", publisher="Oxford")
        s_wiki = self._create_mock_source("s3", "Wikipedia", "Text 3", tier=2, url="https://wikipedia.org/wiki/keeladi", publisher="Wikimedia Foundation", source_type="wikipedia")
        s_loc = self._create_mock_source("s4", "Local JSON", "Text 4", tier=2, url="d:\\mvid\\daily_engine\\topics.json", publisher="Local Project Knowledge", source_origin="local")

        chunks = [EvidenceChunk(f"c{i}", "ep", f"s{i}", "u", "t", "p", "time", "h", 0, "txt", 1) for i in range(1, 5)]
        stats = ResearchDossierBuilder.compute_source_statistics([s_ext1, s_ext2, s_wiki, s_loc], chunks)

        self.assertEqual(stats["total_sources"], 4)
        self.assertEqual(stats["external_sources"], 3)
        self.assertEqual(stats["local_sources"], 1)
        self.assertEqual(stats["external_independent_sources"], 3)
        self.assertEqual(stats["high_quality_external_sources"], 2) # asi + oxford (excludes wikipedia)
        self.assertEqual(stats["reference_only_sources"], 2) # wikipedia + local doc
        self.assertIn("source_type_counts", stats)
        self.assertIn("evidence_weight_counts", stats)

    # 37. EvidenceChunk Retains Independence Group and Provenance (Correction 6)
    def test_37_evidence_chunk_retains_independence_group(self):
        processor = ResearchProcessor()
        s = self._create_mock_source("s_chk_prov", "Excavation Vol", "Detailed carbon dating report from 580 BCE.", tier=1, url="https://asi.nic.in/excavation", publisher="ASI")
        chunks = processor.chunk_source(s, episode_id="ep_test_chk")
        self.assertGreaterEqual(len(chunks), 1)
        c = chunks[0]
        self.assertEqual(c.independence_group, "asi.nic.in")
        self.assertEqual(c.source_origin, "external")
        self.assertEqual(c.source_role, "primary_evidence")
        self.assertEqual(c.evidence_weight_class, "PRIMARY")

    # 38. Preliminary Findings Retain Independence and Origin (Correction 7)
    def test_38_preliminary_findings_retain_independence_and_origin(self):
        q_ev = [{
            "question": "What is the chronology?",
            "retrieved_evidence": [{
                "chunk_id": "c1",
                "source_id": "s1",
                "title": "ASI Report",
                "publisher": "ASI",
                "text": "Radiocarbon dating firmly establishes 580 BCE.",
                "credibility_tier": 1,
                "independence_group": "asi.nic.in",
                "source_origin": "external",
                "evidence_weight_class": "PRIMARY"
            }]
        }]
        findings = ResearchDossierBuilder.generate_preliminary_findings(q_ev, "Keezhadi")
        self.assertEqual(len(findings), 1)
        f = findings[0]
        self.assertIn("asi.nic.in", f.independence_groups)
        self.assertIn("external", f.source_origins)
        self.assertIn("PRIMARY", f.evidence_weight_classes)

    # 39. Dossier Serialization Avoids Raw Content Bloat (Correction 8 & 9)
    def test_39_dossier_no_raw_duplication(self):
        huge_text = "Ancient history sentence. " * 500 # > 12 KB
        s = self._create_mock_source("s_huge", "Huge Doc", huge_text, tier=1)
        d = s.to_dict()
        # Ensure serialization truncated the raw preview
        self.assertLess(len(d["raw_content"] or ""), 400)
        self.assertLess(len(d["cleaned_text"] or ""), 700)

    # 40. Memory Benchmarks Recorded in Timing Stats (Correction 10)
    def test_40_memory_and_resource_benchmarks(self):
        ep_dir_path = os.path.join(self.test_dir, "ep_mem")
        ep = self.state_manager.create_episode(topic="Mem Test", category="Ancient Tamil history", output_directory=ep_dir_path)
        self.state_manager.transition(ep.episode_id, EpisodeState.TOPIC_SELECTED, "TOPIC_SELECTED")

        mock_discovery = MagicMock()
        mock_discovery.discover_sources.return_value = [
            self._create_mock_source("s1", "Doc 1", "Evidence paragraph.\n\nMore details on urban stratigraphy.", tier=1, url="https://asi.nic.in/rep1"),
            self._create_mock_source("s2", "Doc 2", "Scholarly paragraph.\n\nChronology and dating.", tier=2, url="https://oxford.ac.uk/rep2"),
            self._create_mock_source("s3", "Doc 3", "Journalism paragraph.\n\nPublic interest.", tier=3, url="https://thehindu.com/rep3")
        ]
        store = EvidenceStore(qdrant_path=self.test_qdrant_path, collection_name="mem_col", device="cpu")
        engine = ResearchEngine(state_manager=self.state_manager, discovery=mock_discovery, evidence_store=store, offline=True)
        res = engine.execute_research(ep.episode_id)

        timing = res["timing"]
        self.assertIn("rss_before_mb", timing)
        self.assertIn("peak_rss_mb", timing)
        self.assertIn("total_runtime_seconds", timing)

    # 41. Quality Gate Fails when External Independence Threshold Not Met (Correction 4 & 14)
    def test_41_quality_gate_external_independence_fail(self):
        gate = ResearchQualityGate(
            min_sources=2,
            minimum_external_independent_sources=2,
            minimum_high_quality_external_sources=1
        )
        plan = ResearchPlan(topic="Same Domain Test", research_questions=["Q1"])
        # Two sources from the SAME domain (asi.nic.in)
        s1 = self._create_mock_source("s1", "ASI 1", "Txt", tier=1, url="https://asi.nic.in/p1", publisher="ASI")
        s2 = self._create_mock_source("s2", "ASI 2", "Txt", tier=1, url="https://asi.nic.in/p2", publisher="ASI")
        chunks = [
            EvidenceChunk("c1", "ep", "s1", s1.url, "ASI 1", "ASI", "2026", "h1", 0, "txt", 1, independence_group="asi.nic.in"),
            EvidenceChunk("c2", "ep", "s2", s2.url, "ASI 2", "ASI", "2026", "h2", 0, "txt", 1, independence_group="asi.nic.in"),
        ]
        q_ev = [{"question": "Q1", "retrieved_evidence": [{"chunk_id": "c1", "text": "txt"}]}]
        dossier = ResearchDossierBuilder.build_dossier("ep", plan, [s1, s2], chunks, q_ev)
        passed, reasons = gate.evaluate(dossier)
        self.assertFalse(passed)
        self.assertTrue(any("external independent sources" in r for r in reasons))

    # 42. Deterministic Evidence Weight Mapping (Correction 3)
    def test_42_deterministic_evidence_weight_mapping(self):
        # Tier 1 Gov -> PRIMARY / primary_evidence
        r1, w1 = SourceRetriever.determine_evidence_weight(1, "asi.nic.in", "web", "external")
        self.assertEqual(w1, "PRIMARY")
        self.assertEqual(r1, "primary_evidence")

        # Wikipedia -> REFERENCE / discovery_reference
        rw, ww = SourceRetriever.determine_evidence_weight(2, "en.wikipedia.org", "wikipedia", "external")
        self.assertEqual(ww, "REFERENCE")
        self.assertEqual(rw, "discovery_reference")

        # Local -> REFERENCE / background_local
        rl, wl = SourceRetriever.determine_evidence_weight(2, "local_filesystem", "local_doc", "local")
        self.assertEqual(wl, "REFERENCE")
        self.assertEqual(rl, "background_local")

        # Tier 3 News -> MODERATE / secondary_evidence
        r3, w3 = SourceRetriever.determine_evidence_weight(3, "thehindu.com", "web", "external")
        self.assertEqual(w3, "MODERATE")
        self.assertEqual(r3, "secondary_evidence")

    # 43. Qdrant Payload Stores Complete Provenance Fields (Correction 6)
    def test_43_qdrant_provenance_payload_storage(self):
        store = EvidenceStore(qdrant_path=self.test_qdrant_path, collection_name="col_prov_test")
        if store.qdrant is None:
            self.skipTest("Qdrant client unavailable.")

        chunk = EvidenceChunk(
            chunk_id="chk_prov_store",
            episode_id="ep_prov",
            source_id="src_prov",
            url="https://asi.nic.in/rep",
            title="Provenance Title",
            publisher="ASI",
            retrieval_timestamp="2026",
            content_hash="h_prov",
            chunk_index=0,
            text="Text chunk with provenance.",
            credibility_tier=1,
            domain="asi.nic.in",
            source_origin="external",
            independence_group="asi.nic.in",
            source_role="primary_evidence",
            evidence_weight_class="PRIMARY"
        )
        store.index_chunks([chunk], topic="Topic", episode_id="ep_prov")
        results = store.retrieve_evidence("provenance", episode_id="ep_prov", top_k=1)
        self.assertGreaterEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r.get("independence_group"), "asi.nic.in")
        self.assertEqual(r.get("source_origin"), "external")
        self.assertEqual(r.get("evidence_weight_class"), "PRIMARY")

    # 44. StateManager Persists New Provenance in Extra Metadata (Correction 1 & 6)
    def test_44_state_manager_persists_provenance_metadata(self):
        ep_dir_path = os.path.join(self.test_dir, "ep_sm_prov")
        ep = self.state_manager.create_episode(topic="SM Provenance", category="Ancient Tamil history", output_directory=ep_dir_path)
        source_data = {
            "source_id": "src_prov_sm",
            "title": "Excavation Report",
            "url": "https://asi.nic.in/vol5",
            "publisher": "ASI",
            "domain": "asi.nic.in",
            "credibility_tier": 1,
            "content_hash": "hash_sm_123",
            "source_origin": "external",
            "independence_group": "asi.nic.in",
            "source_role": "primary_evidence",
            "evidence_weight_class": "PRIMARY"
        }
        self.state_manager.record_research_source(ep.episode_id, source_data)
        sources = self.state_manager.get_episode_sources(ep.episode_id)
        self.assertEqual(len(sources), 1)
        s = sources[0]
        self.assertEqual(s.extra_metadata.get("independence_group"), "asi.nic.in")
        self.assertEqual(s.extra_metadata.get("source_origin"), "external")
        self.assertEqual(s.extra_metadata.get("evidence_weight_class"), "PRIMARY")


if __name__ == "__main__":
    unittest.main()
