import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import BASE_DIR

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Simple JSON-backed persistence helper for the new backend modules."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or BASE_DIR / "data" / "database")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, payload: Dict[str, Any]) -> Path:
        path = self.storage_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    def load_json(self, name: str) -> Dict[str, Any]:
        path = self.storage_dir / f"{name}.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


database_manager = DatabaseManager()
