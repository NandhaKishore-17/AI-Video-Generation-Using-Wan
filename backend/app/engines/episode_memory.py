import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import BASE_DIR

logger = logging.getLogger(__name__)


class EpisodeMemoryManager:
    """Stores episode continuity data so each new episode can build on prior lore."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or BASE_DIR / "data" / "episode_memory")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_universe(self, universe_id: str) -> Path:
        safe_name = str(universe_id).replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_name}.json"

    def load_universe_state(self, universe_id: str) -> Dict[str, Any]:
        path = self._path_for_universe(universe_id)
        if not path.exists():
            return {
                "universe_id": universe_id,
                "episode_history": [],
                "characters": [],
                "world_rules": [],
                "active_quests": [],
                "villain_progress": [],
                "memory_notes": {},
            }
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            logger.warning("Unable to load episode memory for %s: %s", universe_id, exc)
            data = {}
        data.setdefault("episode_history", [])
        data.setdefault("characters", [])
        data.setdefault("world_rules", [])
        data.setdefault("active_quests", [])
        data.setdefault("villain_progress", [])
        data.setdefault("memory_notes", {})
        return data

    def save_episode(
        self,
        universe_id: str,
        episode_number: int,
        episode_data: Dict[str, Any],
        characters: Optional[List[Dict[str, Any]]] = None,
        scenes: Optional[List[Dict[str, Any]]] = None,
        memory_notes: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = self.load_universe_state(universe_id)
        episode_entry = {
            "episode_number": episode_number,
            "title": episode_data.get("title") or f"Episode {episode_number}",
            "summary": episode_data.get("summary") or "",
            "cliffhanger": episode_data.get("cliffhanger") or "",
            "characters": characters or [],
            "scenes": scenes or [],
        }
        state.setdefault("episode_history", []).append(episode_entry)
        state["characters"] = characters or state.get("characters", [])
        state["memory_notes"] = {**state.get("memory_notes", {}), **(memory_notes or {})}
        state["last_episode_summary"] = episode_entry["summary"]
        state["previous_cliffhanger"] = episode_entry["cliffhanger"]
        path = self._path_for_universe(universe_id)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        return state

    def build_context(self, universe_id: str, episode_number: int) -> Dict[str, Any]:
        state = self.load_universe_state(universe_id)
        history = state.get("episode_history", [])
        previous_episode = history[-1] if history else None
        return {
            "universe_id": universe_id,
            "episode_number": episode_number,
            "episode_history": history,
            "previous_summary": previous_episode.get("summary", "") if previous_episode else "",
            "previous_cliffhanger": previous_episode.get("cliffhanger", "") if previous_episode else "",
            "characters": state.get("characters", []),
            "world_rules": state.get("world_rules", []),
            "active_quests": state.get("active_quests", []),
            "villain_progress": state.get("villain_progress", []),
            "memory_notes": state.get("memory_notes", {}),
        }


episode_memory_manager = EpisodeMemoryManager()
