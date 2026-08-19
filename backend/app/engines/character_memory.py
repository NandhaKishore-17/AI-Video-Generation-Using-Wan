import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List
import uuid

logger = logging.getLogger(__name__)


class CharacterManager:
    """Persist and reuse character metadata and memories so character consistency persists."""

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

    # --- Persistent Memories, Relationships, and Knowledge ---

    def _load_memories_file(self, universe_id: str) -> Dict[str, Any]:
        path = self.storage_dir / f"memories_{universe_id}.json"
        if not path.exists():
            return {"memories": {}, "relationships": {}, "knowledge": {}}
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as e:
            logger.warning(f"Error loading memories file {path}: {e}")
            return {"memories": {}, "relationships": {}, "knowledge": {}}

    def _save_memories_file(self, universe_id: str, data: Dict[str, Any]) -> None:
        path = self.storage_dir / f"memories_{universe_id}.json"
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
        except Exception as e:
            logger.warning(f"Error saving memories file {path}: {e}")

    def add_memory(self, character_name: str, universe_id: str, episode_num: int, content: str) -> None:
        # Prevent dialogue memory from being added
        content_lower = content.lower()
        if "said:" in content_lower or "dialogue" in content_lower or '"' in content:
            logger.warning(f"Rejected adding dialogue content to character memory: {content}")
            return
            
        data = self._load_memories_file(universe_id)
        char_mems = data.setdefault("memories", {}).setdefault(character_name, [])
        # Prevent exact duplicates
        if not any(m.get("content") == content for m in char_mems):
            char_mems.append({
                "episode": episode_num,
                "content": content
            })
            self._save_memories_file(universe_id, data)
            logger.info(f"Added memory for {character_name} in universe {universe_id}")

    def get_memories(self, character_name: str, universe_id: str) -> List[str]:
        data = self._load_memories_file(universe_id)
        char_mems = data.get("memories", {}).get(character_name, [])
        return [f"Ep {m['episode']}: {m['content']}" for m in char_mems]

    def update_relationship(self, character_name: str, universe_id: str, target_name: str, relationship: str) -> None:
        data = self._load_memories_file(universe_id)
        data.setdefault("relationships", {}).setdefault(character_name, {})[target_name] = relationship
        self._save_memories_file(universe_id, data)
        logger.info(f"Updated relationship: {character_name} views {target_name} as {relationship}")

    def get_relationship_summary(self, character: Any, all_characters: List[Any]) -> str:
        # Compatibility with old signature: character (dict or name), all_characters (list of dicts or names)
        char_name = character["name"] if isinstance(character, dict) else str(character)
        
        # We can extract universe_id if character has it, else use "default"
        universe_id = "default"
        if isinstance(character, dict) and "universe_id" in character:
            universe_id = character["universe_id"]

        data = self._load_memories_file(universe_id)
        relationships = data.get("relationships", {}).get(char_name, {})
        
        summary = []
        for target in all_characters:
            t_name = target["name"] if isinstance(target, dict) else str(target)
            if t_name == char_name:
                continue
            if t_name in relationships:
                summary.append(f"{char_name} views {t_name} as a {relationships[t_name]}.")
            elif isinstance(character, dict) and "relationships" in character:
                # Fallback to local dict-based relationship if any
                local_rel = character.get("relationships", {})
                t_id = target.get("uuid") or target.get("id") if isinstance(target, dict) else None
                if t_id and t_id in local_rel:
                    summary.append(f"{char_name} views {t_name} as a {local_rel[t_id]}.")
        
        return " ".join(summary) if summary else f"{char_name} has no established relationships."

    def add_knowledge(self, character_name: str, universe_id: str, fact: str) -> None:
        data = self._load_memories_file(universe_id)
        facts = data.setdefault("knowledge", {}).setdefault(character_name, [])
        if fact not in facts:
            facts.append(fact)
            self._save_memories_file(universe_id, data)

    def get_knowledge(self, character_name: str, universe_id: str) -> List[str]:
        data = self._load_memories_file(universe_id)
        return data.get("knowledge", {}).get(character_name, [])

    def get_character_context(self, character_name: str, universe_id: str, all_characters: List[str]) -> str:
        """Compiles a summary of character personality, relationships, and history."""
        mems = self.get_memories(character_name, universe_id)
        know = self.get_knowledge(character_name, universe_id)
        
        personality = "neutral"
        role = "Side Character"
        bio = ""
        for file in self.storage_dir.glob("*.json"):
            if file.name.startswith("memories_"):
                continue
            try:
                with file.open("r", encoding="utf-8") as h:
                    c = json.load(h)
                    if c.get("name") == character_name:
                        personality = c.get("personality", personality)
                        role = c.get("role", role)
                        bio = c.get("bio", bio)
                        break
            except Exception:
                pass

        rel_summary = self.get_relationship_summary(character_name, all_characters)
        
        context = [
            f"Character: {character_name}",
            f"Role: {role}",
            f"Personality: {personality}",
        ]
        if bio:
            context.append(f"Bio: {bio}")
        context.append(f"Relationships: {rel_summary}")
        
        if mems:
            context.append("Memories:\n" + "\n".join(f"- {m}" for m in mems[-5:]))
        if know:
            context.append("Known facts:\n" + "\n".join(f"- {k}" for k in know))
            
        return "\n".join(context)


character_manager = CharacterManager()
