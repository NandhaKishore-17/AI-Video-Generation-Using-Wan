import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import BASE_DIR

logger = logging.getLogger(__name__)


class UniverseManager:
    """Persists universe-level state for continuity across episodes."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or BASE_DIR / "data" / "universes")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_universe(self, universe_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        path = self.storage_dir / f"{universe_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def load_universe(self, universe_id: str) -> Dict[str, Any]:
        path = self.storage_dir / f"{universe_id}.json"
        if not path.exists():
            return {"id": universe_id, "characters": [], "world_rules": [], "villains": []}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


universe_manager = UniverseManager()
