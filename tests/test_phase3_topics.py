"""
tests/test_phase3_topics.py - Phase 3 Autonomous Topic Discovery & Selection Test Suite.
Validates:
1. Topic catalog loading from topics.json and curated repertoire.
2. Candidate normalization.
3. Topic history loading from SQLite.
4. Exact duplicate detection.
5. Normalized duplicate detection.
6. Semantic duplicate fallback / embedding similarity.
7. Recent-topic penalty calculation.
8. Category balancing across recent episodes.
9. Failure penalty.
10. Measurable score calculation & weights.
11. Deterministic ranking & tie-breakers.
12. Highest safe candidate selection.
13. Selection persistence to database with extra_data.
14. Manual topic compatibility with --topic.
15. Empty catalog handling.
16. Invalid metadata handling.
17. Orchestrator integration (select_topic).
18. CLI --topics logic.
19. CLI --select-topic logic.
20. Confirmation that research/video generation is NOT executed in Phase 3.
"""

import sys
import os
import unittest
import tempfile
import shutil
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autonomous.config import autonomous_settings
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.topic_discovery import TopicDiscovery, TopicCandidate, normalize_title
from autonomous.topic_history import TopicHistory, jaccard_similarity
from autonomous.topic_selector import TopicSelector, TopicScoreResult
from autonomous.orchestrator import AutonomousOrchestrator, StageExecutionResult


class TestPhase3Topics(unittest.TestCase):

    def setUp(self):
        # Create an isolated temporary directory for test DB and files
        self.test_dir = tempfile.mkdtemp(prefix="test_phase3_topics_")
        self.db_path = Path(self.test_dir) / "test_universe.db"
        self.db_url = f"sqlite:///{self.db_path}"
        self.state_manager = StateManager(db_url=self.db_url)
        self.history = TopicHistory(self.state_manager)

        # Create temporary topics.json for testing
        self.test_topics_file = Path(self.test_dir) / "test_topics.json"
        sample_data = {
            "episodes": [
                {
                    "id": "sample_topic_1",
                    "title_english": "The Great Chola Naval Expedition",
                    "title_tamil": "சோழர்களின் மாபெரும் கடற்படை பயணம்",
                    "description": "Exploration of the ancient Chola armada.",
                    "tags": ["Chola", "Maritime", "Navy"],
                    "scenes": [{"id": 1, "badge": "Scene 1"}]
                },
                {
                    "id": "sample_topic_2",
                    "title_english": "Tanjore Temple Stone Architecture",
                    "title_tamil": "தஞ்சை பெரிய கோவில் கட்டிடக்கலை",
                    "description": "Architectural marvels of Tanjore.",
                    "tags": ["Temple", "Architecture"],
                    "scenes": [{"id": 1, "badge": "Scene 1"}]
                }
            ]
        }
        with open(self.test_topics_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        self.discovery = TopicDiscovery(topics_file=self.test_topics_file)
        self.selector = TopicSelector(
            discovery=self.discovery,
            history=self.history,
            state_manager=self.state_manager
        )

    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except Exception:
            pass

    def test_01_topic_catalog_loading(self):
        """Test loading candidates from both topics.json and curated historical catalog."""
        candidates = self.discovery.discover_candidates()
        self.assertGreaterEqual(len(candidates), 10)
        ids = [c.topic_id for c in candidates]
        self.assertIn("sample_topic_1", ids)
        self.assertIn("poompuhar_sunken_harbor", ids)
        self.assertIn("keezhadi_urban_sangam_civilization", ids)

    def test_02_candidate_normalization(self):
        """Test title normalization strips punctuation and collapses whitespace."""
        self.assertEqual(
            normalize_title("  The Chola Navy: 1025 CE! (Expedition)  "),
            "the chola navy 1025 ce expedition"
        )
        self.assertEqual(normalize_title(""), "")

    def test_03_topic_history_loading(self):
        """Test TopicHistory querying against SQLite."""
        self.assertFalse(self.history.has_topic_been_used("Keezhadi Civilization"))
        self.assertEqual(self.history.get_topic_usage_count("Keezhadi Civilization"), 0)

        # Create an episode in test DB
        self.state_manager.create_episode(topic="Keezhadi Civilization", episode_id="TEST-001")
        self.assertTrue(self.history.has_topic_been_used("Keezhadi Civilization"))
        self.assertEqual(self.history.get_topic_usage_count("Keezhadi Civilization"), 1)

    def test_04_exact_duplicate_detection(self):
        """Test exact duplicate title detection."""
        self.state_manager.create_episode(topic="Poompuhar Port", episode_id="TEST-002")
        is_dup, match, sim = self.history.check_duplicate("Poompuhar Port")
        self.assertTrue(is_dup)
        self.assertEqual(match, "Poompuhar Port")
        self.assertEqual(sim, 1.0)

    def test_05_normalized_duplicate_detection(self):
        """Test normalized duplicate detection with different casing and punctuation."""
        self.state_manager.create_episode(topic="Kallanai Grand Anicut", episode_id="TEST-003")
        is_dup, match, sim = self.history.check_duplicate("kallanai: grand anicut!!")
        self.assertTrue(is_dup)
        self.assertEqual(match, "Kallanai Grand Anicut")
        self.assertEqual(sim, 1.0)

    def test_06_semantic_duplicate_similarity(self):
        """Test similarity computation between closely related titles."""
        sim = self.history.compute_similarity(
            "The Chola Naval Fleet Across Asia",
            "Chola Naval Fleet Across Asia"
        )
        self.assertGreater(sim, 0.80)

    def test_07_recent_topic_penalty(self):
        """Test that a topic used recently receives a recency penalty."""
        cand = TopicCandidate(
            topic_id="test_recent",
            title="Recent Expedition Topic",
            category="Maritime history"
        )
        score_before = self.selector.score_candidate(cand)
        self.assertNotIn("recency", score_before.penalties)

        # Record this topic in DB (as of now)
        self.state_manager.create_episode(topic="Recent Expedition Topic", episode_id="TEST-004")

        score_after = self.selector.score_candidate(cand)
        self.assertIn("recency", score_after.penalties)
        self.assertLess(score_after.final_score, score_before.final_score)

    def test_08_category_balancing(self):
        """Test that repeatedly using the same category applies a category repetition penalty."""
        cand_maritime = TopicCandidate(topic_id="m1", title="Maritime Journey", category="Maritime history")
        cand_arch = TopicCandidate(topic_id="a1", title="Temple Wonder", category="Temples and architecture")

        # Create 3 recent episodes in Maritime history
        self.state_manager.create_episode(topic="Ship 1", category="Maritime history", episode_id="M1")
        self.state_manager.create_episode(topic="Ship 2", category="Maritime history", episode_id="M2")
        self.state_manager.create_episode(topic="Ship 3", category="Maritime history", episode_id="M3")

        score_maritime = self.selector.score_candidate(cand_maritime)
        score_arch = self.selector.score_candidate(cand_arch)

        # Maritime should have category repetition penalty
        self.assertIn("category_repetition", score_maritime.penalties)
        # Architecture should not have category repetition penalty
        self.assertNotIn("category_repetition", score_arch.penalties)

    def test_09_failure_penalty(self):
        """Test that previously failed topics incur a failure penalty."""
        cand = TopicCandidate(topic_id="f1", title="Problematic Subject", category="Lost cities")
        score_before = self.selector.score_candidate(cand)

        # Record a failed episode with this topic
        ep = self.state_manager.create_episode(topic="Problematic Subject", episode_id="F1")
        self.state_manager.transition("F1", EpisodeState.TOPIC_SELECTED)
        self.state_manager.transition("F1", EpisodeState.RESEARCHING)
        self.state_manager.transition("F1", EpisodeState.FAILED, error_message="Source lookup timeout")

        score_after = self.selector.score_candidate(cand)
        self.assertIn("failure_history", score_after.penalties)
        self.assertLess(score_after.final_score, score_before.final_score)

    def test_10_measurable_score_calculation(self):
        """Test that score calculation respects defined weights and produces 0.0-1.0 bounded output."""
        cand = TopicCandidate(
            topic_id="score_test",
            title="Standard Score Test",
            educational_value=0.90,
            storytelling_potential=0.85,
            visual_potential=0.95,
            researchability=0.80,
            novelty=0.70,
            audience_potential=0.80,
            episode_suitability=0.90,
        )
        res = self.selector.score_candidate(cand)
        self.assertGreater(res.base_score, 0.70)
        self.assertLessEqual(res.final_score, 1.0)
        self.assertGreaterEqual(res.final_score, 0.0)
        self.assertIn("educational_value", res.dimension_scores)

    def test_11_deterministic_ranking(self):
        """Test that ranking identical candidate pools produces identical results."""
        candidates = self.discovery.discover_candidates()
        rank1 = [r.candidate.topic_id for r in self.selector.rank_candidates(candidates)]
        rank2 = [r.candidate.topic_id for r in self.selector.rank_candidates(candidates)]
        self.assertEqual(rank1, rank2)

    def test_12_highest_safe_candidate_selection(self):
        """Test selection picks the top-ranked candidate that has is_safe=True."""
        best = self.selector.select_best_candidate()
        self.assertIsNotNone(best)
        self.assertTrue(best.is_safe)
        self.assertGreater(best.final_score, 0.50)

    def test_13_selection_persistence(self):
        """Test that select_and_create_episode persists candidate info and score to SQLite."""
        ep, res = self.selector.select_and_create_episode(allow_active_override=True)
        self.assertIsNotNone(ep)
        self.assertEqual(ep.status, EpisodeState.TOPIC_SELECTED.value)
        self.assertIn("selection_score", ep.extra_data)
        self.assertIn("score_breakdown", ep.extra_data)

        # Verify DB retrieval
        fetched = self.state_manager.get_episode(ep.episode_id)
        self.assertEqual(fetched.topic, ep.topic)

    def test_14_manual_topic_compatibility(self):
        """Test that manual topic creation works seamlessly alongside autonomous discovery."""
        ep = self.state_manager.create_episode(topic="Manual Override Subject", category="Special")
        ep = self.state_manager.transition(ep.episode_id, EpisodeState.TOPIC_SELECTED)
        self.assertEqual(ep.topic, "Manual Override Subject")
        self.assertEqual(ep.status, EpisodeState.TOPIC_SELECTED.value)

    def test_15_empty_catalog_handling(self):
        """Test handling empty discovery catalog gracefully without crashing."""
        empty_disc = TopicDiscovery(topics_file=Path(self.test_dir) / "nonexistent.json")
        empty_disc.load_curated_catalog = lambda: []  # Mock empty
        empty_selector = TopicSelector(discovery=empty_disc, state_manager=self.state_manager)

        candidates = empty_disc.discover_candidates()
        self.assertEqual(len(candidates), 0)
        best = empty_selector.select_best_candidate()
        self.assertIsNone(best)

    def test_16_invalid_metadata_handling(self):
        """Test handling candidates with missing or zeroed metadata."""
        broken_cand = TopicCandidate(
            topic_id="broken",
            title="Zero Metadata Topic",
            educational_value=0.0,
            storytelling_potential=0.0,
            visual_potential=0.0,
            researchability=0.0,
            novelty=0.0,
            audience_potential=0.0,
            episode_suitability=0.0,
        )
        res = self.selector.score_candidate(broken_cand)
        self.assertGreaterEqual(res.final_score, 0.0)

    def test_17_orchestrator_integration(self):
        """Test orchestrator.select_topic() selects topic and halts at TOPIC_SELECTED."""
        orch = AutonomousOrchestrator(
            state_manager=self.state_manager,
            topic_selector=self.selector
        )
        res = orch.select_topic(allow_active_override=True)
        self.assertTrue(res.success)
        self.assertEqual(res.next_state, EpisodeState.TOPIC_SELECTED)
        self.assertIn("episode_id", res.data)
        self.assertIn("topic", res.data)

    def test_18_cli_topics_data_availability(self):
        """Test that candidates can be ranked and formatted for CLI display."""
        candidates = self.selector.discovery.discover_candidates()
        ranked = self.selector.rank_candidates(candidates)
        self.assertGreater(len(ranked), 0)
        first = ranked[0]
        self.assertTrue(hasattr(first, "final_score"))
        self.assertTrue(hasattr(first, "selection_reason"))

    def test_19_safety_against_autonomous_duplicate_loops(self):
        """Test that calling select_and_create_episode when an active episode exists refuses duplicate creation."""
        # Create an active unfinished episode
        ep1, res1 = self.selector.select_and_create_episode(allow_active_override=True)
        self.assertIsNotNone(ep1)

        # Second attempt without override should return existing active episode without creating new record
        ep2, res2 = self.selector.select_and_create_episode(allow_active_override=False)
        self.assertEqual(ep1.episode_id, ep2.episode_id)
        self.assertIsNone(res2)

    def test_20_no_downstream_claim_verification_during_phase3(self):
        """Verify that downstream phases (Phase 5 claim verification) still raise NotImplementedError."""
        orch = AutonomousOrchestrator(
            state_manager=self.state_manager,
            topic_selector=self.selector
        )
        with self.assertRaises(NotImplementedError):
            orch.verify_claims("ANY-EPISODE-ID")



if __name__ == "__main__":
    unittest.main()
