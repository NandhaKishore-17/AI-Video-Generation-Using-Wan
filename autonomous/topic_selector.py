"""
autonomous/topic_selector.py - Deterministic Topic Scoring & Autonomous Selection Engine.
Calculates measurable dimension scores, enforces category balancing, evaluates
historical recency and failure penalties, ranks candidates, and selects the optimal topic.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from autonomous.config import autonomous_settings
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.topic_discovery import TopicCandidate, TopicDiscovery
from autonomous.topic_history import TopicHistory

logger = logging.getLogger("autonomous.topic_selector")


@dataclass
class TopicScoreResult:
    """Detailed scoring and selection breakdown for a candidate."""
    candidate: TopicCandidate
    base_score: float
    dimension_scores: Dict[str, float]
    penalties: Dict[str, float]
    final_score: float
    selection_reason: str
    is_safe: bool = True
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.candidate.topic_id,
            "title": self.candidate.title,
            "category": self.candidate.category,
            "base_score": round(self.base_score, 4),
            "final_score": round(self.final_score, 4),
            "dimension_scores": {k: round(v, 4) for k, v in self.dimension_scores.items()},
            "penalties": {k: round(v, 4) for k, v in self.penalties.items()},
            "selection_reason": self.selection_reason,
            "is_safe": self.is_safe,
            "warnings": self.warnings,
        }


class TopicSelector:
    """Evaluates candidates, balances categories, applies recency rules, and deterministically selects the top candidate."""

    def __init__(
        self,
        discovery: Optional[TopicDiscovery] = None,
        history: Optional[TopicHistory] = None,
        state_manager: Optional[StateManager] = None,
    ):
        self.state_manager = state_manager or StateManager()
        self.discovery = discovery or TopicDiscovery()
        self.history = history or TopicHistory(self.state_manager)
        self.settings = autonomous_settings

    def score_candidate(self, candidate: TopicCandidate) -> TopicScoreResult:
        """Compute an explainable, deterministic score for a candidate topic."""
        # 1. Calculate category balance factor (1.0 if unused recently, lower if frequently repeated)
        recent_cat_freq = self.history.get_category_recent_frequency(candidate.category, window=5)
        category_balance_dim = max(0.20, 1.0 - (recent_cat_freq * 0.25))

        # 2. Weighted dimension scores
        dimensions = {
            "educational_value": candidate.educational_value,
            "storytelling_potential": candidate.storytelling_potential,
            "visual_potential": candidate.visual_potential,
            "researchability": candidate.researchability,
            "novelty": candidate.novelty,
            "audience_potential": candidate.audience_potential,
            "episode_suitability": candidate.episode_suitability,
            "category_balance": category_balance_dim,
        }

        weights = {
            "educational_value": self.settings.weight_educational_value,
            "storytelling_potential": self.settings.weight_storytelling_potential,
            "visual_potential": self.settings.weight_visual_potential,
            "researchability": self.settings.weight_researchability,
            "novelty": self.settings.weight_novelty,
            "audience_potential": self.settings.weight_audience_potential,
            "episode_suitability": self.settings.weight_episode_suitability,
            "category_balance": self.settings.weight_category_balance,
        }

        base_score = sum(dimensions[k] * weights[k] for k in dimensions)

        # 3. Penalties calculation
        penalties = {}
        warnings = []
        is_safe = True

        # Check for exact or high duplicate match
        is_dup, dup_match, dup_sim = self.history.check_duplicate(
            candidate.title, threshold=self.settings.topic_duplicate_similarity_threshold
        )
        if is_dup:
            penalties["duplicate"] = 0.80
            warnings.append(f"Near-duplicate of existing episode: '{dup_match}' (similarity: {dup_sim:.2f})")
            is_safe = False

        # Recency penalty
        days_since_used = self.history.get_days_since_last_used(candidate.title)
        if days_since_used is not None and days_since_used < self.settings.topic_recent_days:
            recency_decay = max(0.0, 1.0 - (days_since_used / self.settings.topic_recent_days))
            recency_penalty = self.settings.topic_recent_penalty * recency_decay
            penalties["recency"] = recency_penalty
            warnings.append(f"Used recently ({days_since_used:.1f} days ago). Recency penalty: -{recency_penalty:.2f}")

        # Category repetition penalty
        if recent_cat_freq > 1:
            cat_pen = (recent_cat_freq - 1) * self.settings.category_repetition_penalty
            penalties["category_repetition"] = cat_pen
            warnings.append(f"Category '{candidate.category}' appeared {recent_cat_freq} times recently. Penalty: -{cat_pen:.2f}")

        # Failure penalty
        failures = self.history.get_topic_failures(candidate.title)
        if failures > 0:
            fail_pen = min(0.60, failures * self.settings.failure_penalty_base)
            penalties["failure_history"] = fail_pen
            warnings.append(f"Topic has {failures} previous failures. Penalty: -{fail_pen:.2f}")

        # Total score calculation
        total_penalty = sum(penalties.values())
        final_score = max(0.0, min(1.0, base_score - total_penalty))

        # Rationale generation
        reasons = []
        if candidate.visual_potential >= 0.90:
            reasons.append("exceptional visual reconstruction potential")
        if candidate.educational_value >= 0.90:
            reasons.append("high educational/historical depth")
        if candidate.storytelling_potential >= 0.90:
            reasons.append("compelling narrative arc")
        if candidate.researchability >= 0.90:
            reasons.append("strong academic/archaeological documentation")
        if recent_cat_freq == 0:
            reasons.append("excellent category diversity")

        reason_str = f"Selected based on {', '.join(reasons)}." if reasons else "Selected with balanced historical scores."
        if penalties:
            reason_str += f" Penalties applied: {list(penalties.keys())}."

        return TopicScoreResult(
            candidate=candidate,
            base_score=base_score,
            dimension_scores=dimensions,
            penalties=penalties,
            final_score=final_score,
            selection_reason=reason_str,
            is_safe=is_safe,
            warnings=warnings,
        )

    def rank_candidates(self, candidates: Optional[List[TopicCandidate]] = None) -> List[TopicScoreResult]:
        """
        Score and deterministically rank all candidate topics.
        Tie-breakers: final_score (desc) -> novelty (desc) -> researchability (desc) -> topic_id (asc).
        """
        if candidates is None:
            candidates = self.discovery.discover_candidates()

        scored: List[TopicScoreResult] = []
        for cand in candidates:
            res = self.score_candidate(cand)
            scored.append(res)

        # Deterministic sorting
        scored.sort(
            key=lambda r: (
                -r.final_score,
                -r.candidate.novelty,
                -r.candidate.researchability,
                r.candidate.topic_id
            )
        )
        return scored

    def select_best_candidate(
        self,
        candidates: Optional[List[TopicCandidate]] = None,
        allow_unsafe_fallback: bool = False
    ) -> Optional[TopicScoreResult]:
        """
        Select the highest-ranked safe candidate.
        If all candidates have warnings/penalties, picks highest score if allow_unsafe_fallback is True.
        """
        ranked = self.rank_candidates(candidates)
        if not ranked:
            logger.warning("No candidate topics available for selection.")
            return None

        # Pick highest safe candidate
        for r in ranked:
            if r.is_safe:
                logger.info(f"Deterministically selected topic: '{r.candidate.title}' (Score: {r.final_score:.4f})")
                return r

        if allow_unsafe_fallback:
            logger.warning("No completely safe candidates found. Falling back to top ranked candidate.")
            return ranked[0]

        return None

    def select_and_create_episode(self, allow_active_override: bool = False) -> Tuple[Optional[AutonomousEpisode], Optional[TopicScoreResult]]:
        """
        Autonomous Entrypoint: Selects top topic and creates persistent episode in TOPIC_SELECTED state.
        Guards against creating duplicate active episodes.
        """
        # 1. Autonomous loop safety check: Do not create duplicate active episodes
        active_ep = self.state_manager.get_active_episode()
        if active_ep and not allow_active_override:
            logger.warning(
                f"Active episode '{active_ep.episode_id}' ({active_ep.topic}) is currently in progress [{active_ep.status}]. "
                f"Skipping new topic selection to prevent duplicate parallel episodes."
            )
            return active_ep, None

        # 2. Select best topic
        selected = self.select_best_candidate()
        if not selected:
            raise RuntimeError("Topic selection failed: No suitable candidates available.")

        cand = selected.candidate

        # 3. Create episode in persistent database
        ep = self.state_manager.create_episode(
            topic=cand.title,
            category=cand.category,
            extra_data={
                "topic_id": cand.topic_id,
                "title_tamil": cand.title_tamil,
                "selection_score": selected.final_score,
                "score_breakdown": selected.dimension_scores,
                "penalties": selected.penalties,
                "selection_reason": selected.selection_reason,
                "source": cand.source,
                "historical_period": cand.historical_period,
                "geography": cand.geography,
                "keywords": cand.keywords,
            }
        )

        # 4. Transition to TOPIC_SELECTED
        ep = self.state_manager.transition(
            episode_id=ep.episode_id,
            new_state=EpisodeState.TOPIC_SELECTED,
            current_stage="TOPIC_SELECTED"
        )
        logger.info(f"Created episode {ep.episode_id} with topic '{cand.title}' in TOPIC_SELECTED state.")
        return ep, selected
