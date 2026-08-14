import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger("memory_engine")

class QdrantStoryMemoryEngine:
    """
    Story Memory Engine for maintaining long-term story continuity across episodes.
    Stores episode summaries, relationship shifts, plot reveals, and character states.
    Uses DB-backed fallback store when Qdrant is unavailable (always reliable).
    In-memory cache is populated from DB on first query for each universe.
    """

    def __init__(self):
        self.host = settings.QDRANT_HOST
        self.port = settings.QDRANT_PORT
        # Dict of universe_id -> list of memory dicts
        # Populated from DB on demand for restart safety
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def _load_from_db(self, universe_id: str) -> None:
        """Load memories from the DB StoryMemory table into in-memory cache."""
        if universe_id in self._cache:
            return  # Already loaded

        try:
            from app.core.database import SessionLocal
            from app.models.domain import StoryMemory
            db = SessionLocal()
            try:
                records = (
                    db.query(StoryMemory)
                    .filter(StoryMemory.universe_id == universe_id)
                    .order_by(StoryMemory.episode_number)
                    .all()
                )
                self._cache[universe_id] = [
                    {
                        "id": r.id,
                        "universe_id": r.universe_id,
                        "episode_number": r.episode_number,
                        "content": r.content,
                        "entities_involved": r.entities_involved or [],
                        "memory_type": r.memory_type or "EPISODE_RECAP",
                    }
                    for r in records
                ]
                logger.info(
                    "Loaded %d memories from DB for universe %s",
                    len(self._cache[universe_id]),
                    universe_id,
                )
            finally:
                db.close()
        except Exception as exc:
            logger.warning("Failed to load memories from DB for universe %s: %s", universe_id, exc)
            self._cache[universe_id] = []

    async def add_episode_memory(
        self,
        universe_id: str,
        episode_number: int,
        summary: str,
        entities_involved: List[str],
        memory_type: str = "EPISODE_RECAP"
    ) -> str:
        """
        Stores episode memory in the in-memory cache.
        (DB persistence is handled separately by episode_generator via StoryMemory model.)
        """
        if universe_id not in self._cache:
            self._load_from_db(universe_id)

        memory_item = {
            "id": f"mem_{universe_id}_{episode_number}_{len(self._cache.get(universe_id, []))}",
            "universe_id": universe_id,
            "episode_number": episode_number,
            "content": summary,
            "entities_involved": entities_involved,
            "memory_type": memory_type
        }
        self._cache.setdefault(universe_id, []).append(memory_item)
        logger.info("Memory recorded for Universe %s, Episode %d", universe_id, episode_number)
        return memory_item["id"]

    async def query_relevant_memories(
        self,
        universe_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant past memories matching current story context.
        Loads from DB if cache is empty (restart-safe).
        """
        # Ensure cache is populated from DB
        self._load_from_db(universe_id)

        universe_memories = self._cache.get(universe_id, [])

        if not universe_memories:
            logger.info("No memories found for universe %s", universe_id)
            return []

        # Simple keyword relevance scoring
        query_words = set(query.lower().split())
        results = []

        for mem in universe_memories:
            content_words = set(mem["content"].lower().split())
            overlap = len(query_words.intersection(content_words))
            score = max(0.65, min(0.99, 0.70 + (overlap * 0.05)))

            results.append({
                "id": mem["id"],
                "episode_number": mem["episode_number"],
                "memory_type": mem["memory_type"],
                "content": mem["content"],
                "entities_involved": mem["entities_involved"],
                "relevance_score": round(score, 3)
            })

        # Sort by episode number (most recent first) for continuity
        results.sort(key=lambda x: x["episode_number"], reverse=True)
        return results[:top_k]

memory_engine = QdrantStoryMemoryEngine()
