import json
import logging
import re
import uuid
from typing import Any, Dict, Optional

from app.core.config import settings
from app.engines.episode_memory import episode_memory_manager
from app.engines.llm.ollama_client import OllamaClient, ollama_client

logger = logging.getLogger(__name__)


class StoryEngine:
    """Generate fresh story structures that continue from prior episode memory."""

    def __init__(self, provider: str = None, client: Optional[OllamaClient] = None):
        self.provider = provider or settings.LLM_PROVIDER
        self.client = client or ollama_client

    def generate_story(
        self,
        genre: str,
        theme: str,
        duration: int,
        language: str = "English",
        episode_number: int = 1,
        characters: Optional[list] = None,
        world_rules: str = "",
        previous_summaries: Optional[list] = None,
        universe_id: str = "default",
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._generate_with_llm(
            genre=genre,
            theme=theme,
            duration=duration,
            language=language,
            episode_number=episode_number,
            characters_input=characters,
            world_rules=world_rules,
            previous_summaries=previous_summaries,
            universe_id=universe_id,
            memory_context=memory_context,
        )

    def _generate_with_llm(
        self,
        genre: str,
        theme: str,
        duration: int,
        language: str,
        episode_number: int,
        characters_input: Optional[list] = None,
        world_rules: str = "",
        previous_summaries: Optional[list] = None,
        universe_id: str = "default",
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        char_names = [c.get("name") for c in characters_input] if characters_input else []
        context = memory_context or episode_memory_manager.build_context(universe_id, episode_number)
        previous_cliffhanger = context.get("previous_cliffhanger", "")
        previous_summary = context.get("previous_summary", "")
        memory_notes = context.get("memory_notes", {})

        print(f"Universe: {theme}")
        print(f"Episode: {episode_number}")
        print(f"Characters: {char_names if char_names else 'Auto-generating characters'}")
        print("Sending prompt to Ollama...")
        print(f"Model: {self.client.model}")

        logger.info("Universe: %s | Episode: %s | Model: %s", theme, episode_number, self.client.model)

        char_context_str = f"Use these existing characters: {', '.join(char_names)}" if char_names else "Create 2-3 original characters tailored specifically to this story world."
        rules_str = f"World Rules: {world_rules or ', '.join(context.get('world_rules', []))}" if world_rules or context.get("world_rules") else ""
        summary_str = f"Previous Episode Summary: {previous_summary}" if previous_summary else ""
        cliffhanger_str = f"Previous Cliffhanger: {previous_cliffhanger}" if previous_cliffhanger else ""
        memory_notes_str = f"Memory Notes: {json.dumps(memory_notes)}" if memory_notes else ""

        system_prompt = (
            "You are a cinematic story designer. Return ONLY raw JSON with keys: title, summary, cliffhanger, characters, scenes. "
            "Each character has name, role, age, personality, voice, appearance, goals, relationships, status, growth. "
            "Each scene has id, title, location, duration, description, camera, lighting, mood, visual_prompt, motion_prompt, dialogues. "
            "Each dialogue has speaker, text, emotion, voice."
        )
        user_prompt = (
            f"Create Episode {episode_number} for '{theme}' ({genre}).\n"
            f"{char_context_str}\n{rules_str}\n{summary_str}\n{cliffhanger_str}\n{memory_notes_str}\n"
            "Continue the story naturally, never restart it, and introduce a new conflict, new location, and a fresh cliffhanger. "
            "Return JSON only with 3 scenes."
        )

        raw = self.client.generate(prompt=user_prompt, system=system_prompt)
        data = self._extract_json(raw)
        if not data.get("title") or not data.get("characters") or not data.get("scenes"):
            raise ValueError(f"Incomplete story payload returned by Ollama: {raw}")

        story = self._normalize_story(data, genre, theme, duration, language, episode_number)
        episode_memory_manager.save_episode(
            universe_id=universe_id,
            episode_number=episode_number,
            episode_data={
                "title": story.get("title"),
                "summary": story.get("summary"),
                "cliffhanger": story.get("cliffhanger"),
            },
            characters=story.get("characters", []),
            scenes=story.get("scenes", []),
            memory_notes=memory_notes,
        )
        return story

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        start = cleaned.find("{")
        if start == -1:
            raise ValueError(f"No JSON object found in Ollama response:\n{raw}")
        depth = 0
        end_idx = -1
        for idx, char in enumerate(cleaned[start:], start=start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = idx + 1
                    break
        if end_idx == -1:
            raise ValueError(f"Invalid JSON object in Ollama response:\n{raw}")

        json_str = cleaned[start:end_idx]

        # Fix unquoted bare words like "age": Unknown, -> "age": "Unknown",
        json_str = re.sub(r':\s*([A-Za-z_][A-Za-z0-9_]*)\s*([,\}\n])', r': "\1"\2', json_str)
        json_str = json_str.replace(': "true"', ': true').replace(': "false"', ': false').replace(': "null"', ': null')
        # Fix trailing commas
        json_str = re.sub(r',\s*([\}\]])', r'\1', json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON from Ollama response:\n{raw}\nError: {exc}") from exc

    @staticmethod
    def _safe_age(raw_age: Any, default: int = 25) -> int:
        """Coerce any age value (str, int, None) to a plain int. Extracts first digit run."""
        if isinstance(raw_age, int):
            return raw_age
        if raw_age is None:
            return default
        digits = re.search(r'\d+', str(raw_age))
        return int(digits.group()) if digits else default

    def _normalize_story(self, data: Dict[str, Any], genre: str, theme: str, duration: int, language: str, episode_number: int) -> Dict[str, Any]:
        characters = []
        for index, character in enumerate(data.get("characters", [])[:5]):
            name = character.get("name") or f"Character{index + 1}"
            char = {
                "uuid": character.get("uuid") or str(uuid.uuid4()),
                "name": name,
                "age": self._safe_age(character.get("age"), default=18 + index),
                "gender": character.get("gender") or "non-binary",
                "hair": character.get("hair") or "dark hair",
                "face": character.get("face") or "expressive face",
                "skin": character.get("skin") or "warm skin tone",
                "clothes": character.get("clothes") or character.get("appearance") or "cinematic outfit",
                "personality": character.get("personality") or "curious",
                "voice": character.get("voice") or "balanced",
                "speaking_style": character.get("speaking_style") or "warm",
                "role": character.get("role") or ("Protagonist" if index == 0 else "Supporting"),
                "bio": character.get("bio") or character.get("background") or f"{name} is a key character in this story.",
            }
            characters.append(char)

        scenes = []
        for index, scene in enumerate(data.get("scenes", [])[:6], start=1):
            dialogues = scene.get("dialogues") or []
            if not dialogues and scene.get("dialogue"):
                dialogues = scene.get("dialogue")
            scenes.append(
                {
                    "scene_number": int(scene.get("scene_number", index)),
                    "id": scene.get("id") or f"scene{index:02d}",
                    "title": scene.get("title") or f"Scene {index}",
                    "location": scene.get("location") or "",
                    "duration": int(scene.get("duration", 10)),
                    "description": scene.get("description", f"{genre} scene {index}"),
                    "camera": scene.get("camera") or "cinematic tracking shot",
                    "lighting": scene.get("lighting") or "dramatic rim lighting",
                    "mood": scene.get("mood") or "neutral",
                    "visual_prompt": scene.get("visual_prompt") or scene.get("image_prompt", ""),
                    "motion_prompt": scene.get("motion_prompt") or scene.get("video_motion_prompt", ""),
                    "negative_prompt": scene.get("negative_prompt", ""),
                    "dialogues": dialogues,
                    "dialogue": dialogues,
                    "narration": scene.get("narration", scene.get("description", f"{theme} scene {index}")),
                    "emotion": scene.get("emotion", "neutral"),
                }
            )

        return {
            "title": data.get("title", f"Episode {episode_number}: {theme} Saga"),
            "genre": genre,
            "theme": theme,
            "duration": duration,
            "language": language,
            "characters": characters,
            "scenes": scenes,
            "summary": data.get("summary", f"Episode {episode_number} of the {theme} series."),
            "cliffhanger": data.get("cliffhanger", f"The story escalates in a new way during Episode {episode_number}."),
        }


story_engine = StoryEngine()
