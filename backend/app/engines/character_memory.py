import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

logger = logging.getLogger(__name__)


class CharacterManager:
    """Persist and reuse character metadata so appearances remain consistent across scenes."""

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = Path(storage_dir or "backend/data/characters")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def load_character(self, character_id: str) -> Optional[Dict[str, Any]]:
        path = self.storage_dir / f"{character_id}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_character(self, character: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(character)
        payload.setdefault("uuid", str(uuid.uuid4()))
        path = self.storage_dir / f"{payload['uuid']}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return payload

    def update_character(self, character_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        existing = self.load_character(character_id) or {}
        existing.update(updates)
        existing.setdefault("uuid", character_id)
        path = self.storage_dir / f"{existing['uuid']}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, indent=2)
        return existing

    def get_prompt(self, character_id: str, scene_description: str) -> str:
        character = self.load_character(character_id)
        if not character:
            raise ValueError(f"Character {character_id} not found")
        return (
            f"A cinematic ultra realistic close-up of {character['name']}, {character['age']}-year-old "
            f"{character['gender']} with {character['hair']} hair wearing {character['clothes']}, "
            f"standing in {scene_description}, volumetric lighting, masterpiece, highly detailed, cinematic."
        )


character_manager = CharacterManager()
