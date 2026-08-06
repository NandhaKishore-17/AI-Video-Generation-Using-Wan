import asyncio
import json
import logging
import re
from typing import Dict, Any, List
from app.core.config import settings
from app.engines.llm.ollama_client import ollama_client, OllamaError

logger = logging.getLogger("llm_engine")

class QwenLLMEngine:
    """
    LLM Engine powered by local Ollama server.
    Includes prompt structuring for Universe Bible creation and Screenplay writing.
    """

    def __init__(self):
        self.client = ollama_client

    async def generate_universe_bible(self, title: str, genre: str, logline: str, world_rules: str = "") -> Dict[str, Any]:
        """
        Generates lore bible, factions, history, and key initial characters using local Ollama LLM.
        """
        print(f"Universe: {title}")
        print(f"Episode: 0 (Universe Initialization)")
        print(f"Characters: Auto-generating initial roster for universe '{title}'")
        print("Sending prompt to Ollama...")
        print(f"Model: {self.client.model}")

        system_prompt = "You are a Master Worldbuilder & Showrunner. Return ONLY a valid raw JSON object matching the requested keys. No markdown, no commentary."
        prompt = f"""
        Act as a Master Worldbuilder & Showrunner. Create a deep Universe Bible for:
        Title: {title}
        Genre: {genre}
        Logline: {logline}
        World Rules: {world_rules}

        Return a valid JSON object with the following exact keys:
        - "world_summary": Detailed description of the world
        - "factions": List of key organizations/factions
        - "historical_milestones": List of major past events
        - "suggested_characters": Array of 3 key characters specifically tailored to '{title}', each with "name", "role", "personality", "appearance_prompt", "bio", "voice_actor_preset"
        - "initial_story_arcs": Array of 2 story arcs, each with "title", "goal", "episodes_planned"
        """

        raw = await asyncio.to_thread(self.client.generate, prompt=prompt, system=system_prompt)
        print(f"Raw Ollama response:\n{raw}")

        data = self._extract_json(raw)
        print("Story parsed successfully")
        return data

    async def generate_screenplay(
        self,
        universe_title: str,
        universe_genre: str,
        characters: List[Dict[str, Any]],
        past_memories: List[str],
        current_arc: str,
        episode_number: int,
        custom_prompt: str = ""
    ) -> Dict[str, Any]:
        """
        Generates a cinematic episode screenplay with shot-by-shot breakdown using local Ollama LLM.
        """
        char_names = [c.get("name") for c in characters] if characters else []
        char_summary = ", ".join([f"{c.get('name')} ({c.get('role', 'Hero')})" for c in characters]) if characters else "Create 2 unique characters suitable for this universe."
        memories_str = "\n- ".join(past_memories) if past_memories else "No previous episode memory."

        print(f"Universe: {universe_title}")
        print(f"Episode: {episode_number}")
        print(f"Characters: {char_names}")
        print("Sending prompt to Ollama...")
        print(f"Model: {self.client.model}")

        user_dir = custom_prompt or 'Advance the plot with intense drama, cinematic tension, and character revelations.'
        system_prompt = "You are a Hollywood Screenwriter. Return ONLY a raw JSON object matching the requested schema. No markdown, no extra commentary."
        prompt = f"""
        Act as a Hollywood Screenwriter for the Universe '{universe_title}' ({universe_genre}).
        Episode Number: {episode_number}
        Current Story Arc: {current_arc}
        Characters Available: {char_summary}
        Past Context: {memories_str}
        User Direction: {user_dir}

        Return JSON with:
        - "episode_title": String
        - "logline": String
        - "scenes": Array of 3 scenes, each containing:
          - "scene_number": Integer
          - "location": String
          - "visual_description": Short cinematic description
          - "dialogue": Array of objects: [{{"speaker": "Character Name", "line": "Dialogue line"}}]

        Constraint: Use ONLY character names: {char_names if char_names else 'the heroes'}.
        """

        print(f"Prompt sent to Ollama:\n{prompt}")
        logger.info(f"Prompt sent to Ollama:\n{prompt}")

        raw = await asyncio.to_thread(self.client.generate, prompt=prompt, system=system_prompt)
        print(f"Raw Ollama response:\n{raw}")
        logger.info(f"Raw Ollama response:\n{raw}")

        data = self._extract_json(raw)
        print("Story parsed successfully")

        # Step 2: Generate image prompts and motion vectors after screenplay generation
        from app.engines.prompt_generator import prompt_generator
        for scene in data.get("scenes", []):
            if not scene.get("image_prompt"):
                p_info = prompt_generator.build_scene_prompt(scene=scene, characters=characters)
                scene["image_prompt"] = p_info["prompt"]
                scene["video_motion_prompt"] = f"Cinematic camera movement, {scene.get('visual_description', '')}, 24fps"
                scene["negative_prompt"] = p_info["negative_prompt"]

        print(f"Final story passed to voice and video generators:\n{json.dumps(data, indent=2)}")
        logger.info(f"Final story passed to voice and video generators:\n{json.dumps(data, indent=2)}")
        return data

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

llm_engine = QwenLLMEngine()
