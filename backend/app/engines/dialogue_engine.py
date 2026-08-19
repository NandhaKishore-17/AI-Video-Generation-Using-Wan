import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.engines.llm.ollama_client import ollama_client

logger = logging.getLogger("dialogue_engine")


class DialogueEngine:
    """Generate scene-specific dialogue that is fresh, character-aware, and continuity-aware."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.client = ollama_client

    def clear_cache(self):
        logger.info("Dialogue engine cache cleared (no-op as cache is disabled).")

    async def generate_dialogue_for_scene(
        self,
        scene_number: int,
        scene_title: str,
        scene_description: str,
        characters: List[Dict[str, Any]],
        scene_emotion: str = "neutral",
        previous_scene_summary: str = "",
        episode_objective: str = "",
        memories: List[str] = None,
        universe_id: str = "default",
        previous_dialogues: List[Dict[str, Any]] = None,
        episode_number: int = 1,
        scene_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        from app.engines.character_memory import character_manager

        char_names = [c["name"] for c in characters] if characters else []
        char_contexts = []
        for name in char_names:
            ctx = character_manager.get_character_context(name, universe_id, char_names)
            char_contexts.append(ctx)
        char_memory_str = "\n\n".join(char_contexts) if char_contexts else "None."

        prev_diag_str = ""
        if previous_dialogues:
            prev_diag_str = "\n".join([f"- {d['speaker']}: {d.get('text') or d.get('line')}" for d in previous_dialogues])
        else:
            prev_diag_str = "No dialogue yet in this episode."

        scene_context_str = json.dumps(scene_context or {}, ensure_ascii=False)

        system_prompt = (
            "You are a master screenplay dialogue writer. You MUST output ONLY valid JSON without any markdown formatting, code fences, or explanations. "
            'Expected schema: {"dialogue": [{"speaker": "CharacterName", "line": "..."}]}. '
            "Only use characters that are actually present in the current scene. "
            "Do NOT reuse previous dialogue. Generate completely NEW dialogue. "
            "Do NOT invent new characters. Ensure dialogue lines are never empty and sound natural for voice acting."
        )
        user_prompt = (
            f"Generate dialogue for scene {scene_number}: {scene_title}.\n"
            f"Scene description: {scene_description}\n"
            f"Emotion: {scene_emotion}\n"
            f"Episode objective: {episode_objective}\n"
            f"Previous scene summary: {previous_scene_summary}\n"
            f"Dialogue spoken so far in this episode:\n{prev_diag_str}\n"
            f"Character context:\n{char_memory_str}\n"
            f"Scene metadata:\n{scene_context_str}\n"
            "Return 2-3 lines of dialogue that advance the plot and end with a fresh hook."
        )

        res = None
        last_error = None
        for attempt in range(1, 4):
            try:
                logger.info(f"[DIALOGUE] Episode {episode_number}, Scene {scene_number} - Ollama attempt {attempt}/3")
                raw = await asyncio.to_thread(
                    self.client.generate,
                    prompt=user_prompt,
                    system=system_prompt,
                    format="json"
                )
                data = self._extract_json(raw)
                if data and "dialogue" in data and isinstance(data["dialogue"], list):
                    # Ensure 'line' key exists, mapping from 'text' if needed
                    for dlg in data["dialogue"]:
                        if "text" in dlg and "line" not in dlg:
                            dlg["line"] = dlg.pop("text")
                    
                    if len(data["dialogue"]) == 0:
                        raise ValueError("Dialogue array is empty.")
                    
                    # Validate empty lines
                    for dlg in data["dialogue"]:
                        if not dlg.get("line") or str(dlg.get("line")).strip() == "":
                            raise ValueError("Found empty dialogue line.")
                            
                    res = data
                    logger.info("[DIALOGUE] JSON validation: PASS")
                    break
                else:
                    raise ValueError("JSON is missing 'dialogue' array.")
            except Exception as exc:
                last_error = exc
                raw_preview = raw[:200] if 'raw' in locals() else "N/A"
                logger.warning(f"[DIALOGUE] Attempt {attempt} failed: {exc}. Raw response: {raw_preview}...")
                system_prompt += "\nWARNING: Your previous output was invalid. Output ONLY raw JSON."

        if res is None:
            logger.error(f"[DIALOGUE] All attempts failed. Last error: {last_error}")
            raise RuntimeError(f"Dialogue generation failed after 3 attempts: {last_error}")

        logger.info(f"[DIALOGUE] Generated Dialogues for Scene {scene_number}:\n{json.dumps(res, indent=2)}")
        return res

    def _extract_json(self, raw: str) -> Dict[str, Any]:
        import re
        # Try to find a JSON block enclosed in {}
        match = re.search(r'(\{.*\})', raw, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
                
        # Fallback to brace counting
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        start = cleaned.find("{")
        if start == -1:
            raise ValueError("No JSON object found")
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
            raise ValueError("Invalid JSON object")
            
        try:
            return json.loads(cleaned[start:end_idx])
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing error: {e}")




dialogue_engine = DialogueEngine()
