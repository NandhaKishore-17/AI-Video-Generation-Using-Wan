"""
autonomous/topic_history.py - Topic History & Duplicate Detection Engine.
Tracks previously produced, published, and failed topics using the existing
database (autonomous_episodes table). Implements exact, normalized, and
lightweight semantic similarity duplicate detection.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, AutonomousEpisode, EpisodeState
from autonomous.topic_discovery import normalize_title

logger = logging.getLogger("autonomous.topic_history")


def jaccard_similarity(str1: str, str2: str) -> float:
    """Calculate token Jaccard similarity between two strings as a lightweight fallback."""
    set1 = set(normalize_title(str1).split())
    set2 = set(normalize_title(str2).split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection) / union if union > 0 else 0.0


class TopicHistory:
    """Maintains historical awareness of topic frequency, recency, failures, and duplicates."""

    def __init__(self, state_manager: Optional[StateManager] = None):
        self.state_manager = state_manager or StateManager()
        self._embedding_model = None
        self._model_load_attempted = False

    def _get_embedding_model(self):
        """Lazy-load sentence-transformers if available, without failing if missing."""
        if not self._model_load_attempted:
            self._model_load_attempted = True
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading lightweight embedding model (all-MiniLM-L6-v2) for semantic deduplication on CPU...")
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            except Exception as e:
                logger.warning(f"SentenceTransformer not loaded for topic deduplication ({e}). Falling back to token similarity.")
                self._embedding_model = None
        return self._embedding_model

    def compute_similarity(self, title1: str, title2: str) -> float:
        """Compute similarity between two titles using embeddings or token fallback."""
        norm1 = normalize_title(title1)
        norm2 = normalize_title(title2)
        if norm1 == norm2:
            return 1.0

        model = self._get_embedding_model()
        if model is not None:
            try:
                import numpy as np
                emb1 = model.encode(title1, normalize_embeddings=True)
                emb2 = model.encode(title2, normalize_embeddings=True)
                return float(np.dot(emb1, emb2))
            except Exception as e:
                logger.warning(f"Embedding comparison error ({e}). Using token similarity.")

        return jaccard_similarity(title1, title2)

    def get_all_episodes(self) -> List[AutonomousEpisode]:
        """Fetch all stored episode records."""
        session = self.state_manager._get_session()
        try:
            return session.query(AutonomousEpisode).order_by(AutonomousEpisode.created_at.desc()).all()
        finally:
            session.close()

    def get_topic_records(self, topic: str) -> List[AutonomousEpisode]:
        """Get all episode records matching a normalized topic."""
        norm_target = normalize_title(topic)
        all_eps = self.get_all_episodes()
        return [ep for ep in all_eps if normalize_title(ep.topic) == norm_target]

    def has_topic_been_used(self, topic: str) -> bool:
        """Check if a topic has been used previously."""
        records = self.get_topic_records(topic)
        return len(records) > 0

    def get_topic_usage_count(self, topic: str) -> int:
        """Return how many times a topic has been used."""
        return len(self.get_topic_records(topic))

    def get_topic_last_used(self, topic: str) -> Optional[datetime]:
        """Return timestamp of the most recent usage of a topic."""
        records = self.get_topic_records(topic)
        if not records:
            return None
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[0].created_at

    def get_days_since_last_used(self, topic: str) -> Optional[float]:
        """Return days elapsed since the topic was last used."""
        last_used = self.get_topic_last_used(topic)
        if not last_used:
            return None
        now = datetime.now(timezone.utc)
        if last_used.tzinfo is None:
            last_used = last_used.replace(tzinfo=timezone.utc)
        delta = now - last_used
        return max(0.0, delta.total_seconds() / 86400.0)

    def get_topic_failures(self, topic: str) -> int:
        """Return failure count for a specific topic."""
        records = self.get_topic_records(topic)
        return sum(1 for r in records if r.status == EpisodeState.FAILED.value)

    def get_recently_used_categories(self, limit: int = 5) -> List[str]:
        """Return categories of the most recently created/updated episodes."""
        all_eps = self.get_all_episodes()
        categories = []
        for ep in all_eps[:limit]:
            if ep.category and ep.category not in categories:
                categories.append(ep.category)
        return categories

    def get_category_recent_frequency(self, category: str, window: int = 5) -> int:
        """Count how many times a category appears in the last `window` episodes."""
        all_eps = self.get_all_episodes()
        recent_window = all_eps[:window]
        return sum(1 for ep in recent_window if ep.category.lower() == category.lower())

    def get_active_topics(self) -> List[str]:
        """Get titles of currently active unfinished episodes."""
        unfinished = self.state_manager.get_unfinished_episodes()
        return [ep.topic for ep in unfinished]

    def get_published_topics(self) -> List[str]:
        """Get titles of successfully published episodes."""
        published = self.state_manager.get_published_episodes()
        return [ep.topic for ep in published]

    def check_duplicate(
        self,
        candidate_title: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check whether candidate_title matches or is dangerously similar to any past episode,
        benchmark topic, rejected episode, or protected fixture topic.
        Returns: (is_duplicate: bool, matched_existing_title: Optional[str], similarity_score: float)
        """
        if threshold is None:
            threshold = autonomous_settings.topic_duplicate_similarity_threshold

        norm_candidate = normalize_title(candidate_title)

        # 1. Collect all prior topics from DB, output dirs, etc.
        all_eps = self.get_all_episodes()
        comparison_targets = set()
        for ep in all_eps:
            if ep.topic:
                comparison_targets.add(ep.topic)
            if ep.episode_id:
                comparison_targets.add(ep.episode_id)

        # Check directories under outputs/episodes
        ep_dir = BASE_DIR / "outputs" / "episodes"
        if ep_dir.exists():
            for p in ep_dir.iterdir():
                if p.is_dir():
                    comparison_targets.add(p.name)

        # 2. Check exact match first
        for tgt in comparison_targets:
            norm_existing = normalize_title(tgt)
            if norm_existing and norm_candidate == norm_existing:
                return True, tgt, 1.0

        # 3. Explicit protected / forbidden topic stems
        # Values are (normalized form, canonical returned identifier).
        FORBIDDEN_STEMS = [
            ("poompuhar", "poompuhar"),
            ("keezhadi", "keezhadi"),
            ("mamallapuram", "mamallapuram"),
            ("test p9 audit", "test-p9-audit"),
            ("20260910 002", "20260910-002"),
            ("20260913 001", "20260913-001"),
        ]
        for normalized_stem, canonical_stem in FORBIDDEN_STEMS:
            if normalized_stem in norm_candidate or canonical_stem in candidate_title.lower():
                logger.info(f"Novelty gate: '{candidate_title}' matches forbidden stem '{canonical_stem}'. Rejected.")
                return True, canonical_stem, 1.0

        best_match = None
        highest_score = 0.0

        for tgt in comparison_targets:
            norm_existing = normalize_title(tgt)
            if not norm_existing:
                continue

            # Similarity calculation
            sim = self.compute_similarity(candidate_title, tgt)
            if sim > highest_score:
                highest_score = sim
                best_match = tgt

        if highest_score >= threshold:
            logger.info(f"Duplicate detected: '{candidate_title}' is {highest_score:.2f} similar to '{best_match}' (threshold={threshold})")
            return True, best_match, highest_score

        return False, best_match, highest_score
