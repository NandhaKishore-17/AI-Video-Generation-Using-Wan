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
    """Generate story structures for complete video production pipelines.
    
    All content comes from the LLM — no hardcoded fallback stories.
    """

    def __init__(self, provider: str = "ollama", client: Optional[OllamaClient] = None):
        self.provider = provider or settings.LLM_PROVIDER
        self.client = client or ollama_client

    def generate_story(self, genre: str, theme: str, duration: int, language: str = "English") -> Dict[str, Any]:
        """Generate a story via the LLM.  Raises on failure — never returns hardcoded data."""
        return self._generate_with_llm(genre, theme, duration, language)

    def _generate_with_llm(self, genre: str, theme: str, duration: int, language: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a cinematic story designer. You MUST respond with ONLY valid JSON — no markdown fences, no prose. "
            "The JSON must exactly match this schema:\n"
            "{\n"
            '  "title": "<string>",\n'
            '  "characters": [\n'
            "    {\n"
            '      "uuid": "<string>",\n'
            '      "name": "<string>",\n'
            '      "age": <integer>,\n'
            '      "gender": "<string>",\n'
            '      "hair": "<string>",\n'
            '      "face": "<string>",\n'
            '      "skin": "<string>",\n'
            '      "clothes": "<string>",\n'
            '      "personality": "<string>",\n'
            '      "voice": "<string>",\n'
            '      "speaking_style": "<string>"\n'
            "    }\n"
            "  ],\n"
            '  "scenes": [\n'
            "    {\n"
            '      "scene_number": <integer>,\n'
            '      "description": "<string>",\n'
            '      "dialogue": "<string>",\n'
            '      "narration": "<string>",\n'
            '      "emotion": "<string>"\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "Generate content appropriate to the specified genre — do NOT default to cyberpunk."
        )
        user_prompt = (
            f"Create a {duration}-second {language} story for genre {genre} with theme {theme}. "
            "Return compact JSON with consistent characters across scenes. "
            "All character names and story elements must be original and appropriate to the genre."
        )
        
        last_error = None
        for attempt in range(3):
            try:
                raw = self.client.generate(prompt=user_prompt, system=system_prompt, format="json")
                data = self._extract_json(raw)
                if not data.get("title") or not data.get("characters") or not data.get("scenes"):
                    raise ValueError("Incomplete story payload from LLM")
                return self._normalize_story(data, genre, theme, duration, language)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logging.getLogger(__name__).warning("LLM story generation JSON extraction failed on attempt %d: %s", attempt + 1, exc)
                
        raise ValueError(f"Failed to generate valid story from LLM after 3 attempts: {last_error}")

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


story_engine = StoryEngine()
