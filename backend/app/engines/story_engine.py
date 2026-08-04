import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import settings
from app.engines.llm.ollama_client import OllamaClient, OllamaError, ollama_client

logger = logging.getLogger(__name__)


class StoryEngine:
    """Generate story structures for complete video production pipelines."""

    def __init__(self, provider: str = "ollama", client: Optional[OllamaClient] = None):
        self.provider = provider or settings.LLM_PROVIDER
        self.client = client or ollama_client

    def generate_story(self, genre: str, theme: str, duration: int, language: str = "English") -> Dict[str, Any]:
        if self.provider.lower() in {"ollama", "openai-compatible", "mock"}:
            try:
                return self._generate_with_llm(genre, theme, duration, language)
            except Exception as exc:
                logger.warning("Story generation via LLM failed, using deterministic fallback: %s", exc)
        return self._generate_fallback_story(genre, theme, duration, language)

    def _generate_with_llm(self, genre: str, theme: str, duration: int, language: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a cinematic story designer. Return ONLY JSON with keys title, characters, scenes. "
            "Each character must have uuid, name, age, gender, hair, face, skin, clothes, personality, voice, speaking_style. "
            "Each scene must have scene_number, description, dialogue, narration, emotion."
        )
        user_prompt = (
            f"Create a {duration}-second {language} story for genre {genre} with theme {theme}. "
            "Return compact JSON with consistent characters across scenes."
        )
        raw = self.client.generate(prompt=user_prompt, system=system_prompt)
        data = self._extract_json(raw)
        if not data.get("title") or not data.get("characters") or not data.get("scenes"):
            raise ValueError("Incomplete story payload")
        return self._normalize_story(data, genre, theme, duration, language)

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("No JSON object found")
        depth = 0
        for idx, char in enumerate(cleaned[start:], start=start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start:idx + 1])
        raise ValueError("Invalid JSON object")

    def _normalize_story(self, data: Dict[str, Any], genre: str, theme: str, duration: int, language: str) -> Dict[str, Any]:
        characters = []
        for index, character in enumerate(data.get("characters", [])[:5]):
            char = {
                "uuid": str(uuid.uuid4()),
                "name": character.get("name", f"Character{index + 1}"),
                "age": int(character.get("age", 18 + index)),
                "gender": character.get("gender", "non-binary"),
                "hair": character.get("hair", "dark"),
                "face": character.get("face", "expressive"),
                "skin": character.get("skin", "warm"),
                "clothes": character.get("clothes", "cinematic outfit"),
                "personality": character.get("personality", "curious"),
                "voice": character.get("voice", "balanced"),
                "speaking_style": character.get("speaking_style", "warm"),
            }
            characters.append(char)

        scenes = []
        for index, scene in enumerate(data.get("scenes", [])[:6], start=1):
            scenes.append(
                {
                    "scene_number": int(scene.get("scene_number", index)),
                    "description": scene.get("description", f"{genre} scene {index}"),
                    "dialogue": scene.get("dialogue", f"{theme} scene {index}"),
                    "narration": scene.get("narration", scene.get("description", f"{theme} scene {index}")),
                    "emotion": scene.get("emotion", "neutral"),
                }
            )

        return {
            "title": data.get("title", f"{theme} Saga"),
            "genre": genre,
            "theme": theme,
            "duration": duration,
            "language": language,
            "characters": characters,
            "scenes": scenes,
        }

    def _generate_fallback_story(self, genre: str, theme: str, duration: int, language: str) -> Dict[str, Any]:
        protagonist = {
            "uuid": str(uuid.uuid4()),
            "name": "Ethan",
            "age": 18,
            "gender": "male",
            "hair": "black",
            "face": "young",
            "skin": "fair",
            "clothes": "blue jacket",
            "personality": "curious",
            "voice": "warm",
            "speaking_style": "gentle",
        }
        mentor = {
            "uuid": str(uuid.uuid4()),
            "name": "Professor Lyra",
            "age": 42,
            "gender": "female",
            "hair": "silver",
            "face": "wise",
            "skin": "light",
            "clothes": "deep purple robes",
            "personality": "wise",
            "voice": "calm",
            "speaking_style": "measured",
        }
        scenes = [
            {
                "scene_number": 1,
                "description": f"A cinematic opening scene in a {genre.lower()} world shaped by {theme}.",
                "dialogue": "Welcome to the academy of wonders.",
                "narration": "The first steps into magic begin.",
                "emotion": "wonder",
            },
            {
                "scene_number": 2,
                "description": f"The protagonist uncovers a hidden secret within the halls of {theme}.",
                "dialogue": "The truth is hidden in plain sight.",
                "narration": "Mystery rises as the path becomes clearer.",
                "emotion": "mystery",
            },
            {
                "scene_number": 3,
                "description": f"A climactic finale where courage and friendship decide the fate of {theme}.",
                "dialogue": "We stand together against the darkness.",
                "narration": "The final challenge is met with hope.",
                "emotion": "hope",
            },
        ]
        return {
            "title": f"The Lost Crystal of {theme}",
            "genre": genre,
            "theme": theme,
            "duration": duration,
            "language": language,
            "characters": [protagonist, mentor],
            "scenes": scenes,
        }


story_engine = StoryEngine()
