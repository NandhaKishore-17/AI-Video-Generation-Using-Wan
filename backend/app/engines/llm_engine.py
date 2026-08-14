"""
LLM Engine — production-grade interface to Ollama / OpenAI-compatible LLMs.

All story generation MUST go through the LLM.  There are ZERO hardcoded
fallbacks.  If the LLM is unreachable or returns bad JSON the caller gets
a clear LLMUnavailableError with the real reason.
"""

import json
import logging
import re
import time
import traceback
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger("llm_engine")


# ---------------------------------------------------------------------------
# Custom exception — callers catch this to return HTTP 500 with real details
# ---------------------------------------------------------------------------
class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot fulfil a request."""


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
_MAX_RETRIES = 2


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    return re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Find the outermost { … } in *text* and parse it."""
    cleaned = _strip_markdown_fences(text)
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in LLM response")
    depth = 0
    for idx, ch in enumerate(cleaned[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start : idx + 1])
    raise ValueError("JSON object not properly closed in LLM response")


def _safe_parse_json(raw: str, context: str) -> Dict[str, Any]:
    """Try json.loads first, then fallback to extraction + repair."""
    # Fast path
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Repair path
    try:
        return _extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Failed to parse LLM JSON for {context}: {exc}\n"
            f"Raw response (first 500 chars): {raw[:500]}"
        ) from exc


# ---------------------------------------------------------------------------
# Core LLM Engine
# ---------------------------------------------------------------------------
class QwenLLMEngine:
    """
    Production LLM Engine.
    Supports Ollama and any OpenAI-compatible API.
    NO hardcoded content.  NO silent fallbacks.
    """

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.api_base = settings.LLM_API_BASE.rstrip("/")
        self.model_name = settings.LLM_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT
        logger.info(
            "LLM Engine initialised — provider=%s  model=%s  api_base=%s  timeout=%ds",
            self.provider, self.model_name, self.api_base, self.timeout,
        )

    # ------------------------------------------------------------------
    # Internal: call the LLM
    # ------------------------------------------------------------------
    async def _call_llm(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.7,
        context_label: str = "llm_call",
    ) -> Dict[str, Any]:
        """
        Send a prompt to the LLM and return parsed JSON.
        Retries up to _MAX_RETRIES on JSON-parse failures.
        Raises LLMUnavailableError on any permanent failure.
        """
        url = f"{self.api_base}/chat/completions"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
        }

        logger.info(
            "[%s] LLM REQUEST  provider=%s  model=%s  url=%s  prompt_len=%d",
            context_label, self.provider, self.model_name, url, len(prompt),
        )

        last_error: Optional[Exception] = None

        for attempt in range(1, _MAX_RETRIES + 2):  # 1 initial + 2 retries
            t0 = time.time()
            try:
                async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                    resp = await client.post(url, json=payload)

                latency = time.time() - t0

                logger.info(
                    "[%s] LLM RESPONSE  attempt=%d  status=%d  latency=%.1fs  body_len=%d",
                    context_label, attempt, resp.status_code, latency, len(resp.text),
                )

                if resp.status_code != 200:
                    error_detail = resp.text[:500]
                    raise LLMUnavailableError(
                        f"LLM returned HTTP {resp.status_code}: {error_detail}"
                    )

                data = resp.json()
                content = data["choices"][0]["message"]["content"]

                # Log token usage if available
                usage = data.get("usage", {})
                if usage:
                    logger.info(
                        "[%s] TOKEN USAGE  prompt_tokens=%s  completion_tokens=%s  total=%s",
                        context_label,
                        usage.get("prompt_tokens", "?"),
                        usage.get("completion_tokens", "?"),
                        usage.get("total_tokens", "?"),
                    )

                # Parse JSON
                result = _safe_parse_json(content, context_label)
                logger.info("[%s] JSON parsed successfully on attempt %d", context_label, attempt)
                return result

            except LLMUnavailableError:
                raise  # Don't retry HTTP errors
            except httpx.ConnectError as exc:
                raise LLMUnavailableError(
                    f"Cannot connect to LLM at {url}. Is Ollama running? Error: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise LLMUnavailableError(
                    f"LLM request timed out after {self.timeout}s. "
                    f"The model may be too slow or overloaded. Error: {exc}"
                ) from exc
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "[%s] JSON parse failed on attempt %d: %s", context_label, attempt, exc
                )
                if attempt <= _MAX_RETRIES:
                    logger.info("[%s] Retrying LLM call (attempt %d)...", context_label, attempt + 1)
                    continue
            except Exception as exc:
                logger.error(
                    "[%s] Unexpected LLM error:\n%s", context_label, traceback.format_exc()
                )
                raise LLMUnavailableError(
                    f"Unexpected LLM error: {exc}"
                ) from exc

        raise LLMUnavailableError(
            f"LLM returned invalid JSON after {_MAX_RETRIES + 1} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Universe Bible generation
    # ------------------------------------------------------------------
    async def generate_universe_bible(
        self, title: str, genre: str, logline: str, world_rules: str = ""
    ) -> Dict[str, Any]:
        """Generate a comprehensive universe bible via the LLM.  Never returns hardcoded data."""

        system_prompt = """You are an expert story worldbuilder. When given a story concept, you create detailed, original universe bibles.
You respond ONLY with valid JSON objects. No markdown. No explanations. No template markers."""

        prompt = f"""Create a complete universe bible for this story concept:

STORY: "{title}"
GENRE: {genre}
LOGLINE: {logline}
WORLD RULES: {world_rules or "Create rules appropriate to the genre."}

Write a rich, original {genre} universe. Generate completely unique names and content for this specific story.
Return a JSON object with these exact keys filled with real creative content:

{{
  "world_summary": "<write 3 sentences about this world>",
  "geography": "<describe the physical world>",
  "technology_level": "<describe tech/magic level>",
  "politics": "<describe who rules and how>",
  "factions": [
    {{"name": "<original faction name>", "description": "<what they do and believe>", "alignment": "Good"}},
    {{"name": "<original rival faction name>", "description": "<what they do and believe>", "alignment": "Evil"}}
  ],
  "historical_milestones": [
    {{"era": "<period name>", "event": "<important event>", "significance": "<why it matters>"}},
    {{"era": "<second period>", "event": "<another event>", "significance": "<why it matters>"}}
  ],
  "suggested_characters": [
    {{"name": "<protagonist full name>", "role": "Protagonist", "personality": "<3 traits>", "appearance_prompt": "<detailed visual for image gen>", "bio": "<backstory>", "voice_actor_preset": "en-US-ChristopherNeural"}},
    {{"name": "<antagonist full name>", "role": "Antagonist", "personality": "<3 traits>", "appearance_prompt": "<detailed visual for image gen>", "bio": "<backstory>", "voice_actor_preset": "en-AU-WilliamNeural"}},
    {{"name": "<ally full name>", "role": "Supporting", "personality": "<3 traits>", "appearance_prompt": "<detailed visual for image gen>", "bio": "<backstory>", "voice_actor_preset": "en-US-AriaNeural"}}
  ],
  "initial_story_arcs": [
    {{"title": "<compelling arc name>", "goal": "<what must be achieved>", "episodes_planned": 6}}
  ]
}}

Replace ALL angle-bracket placeholders with your original creative content for this {genre} story.
Character names, faction names, and all content must be original and fit the {genre} theme.
Output only the JSON object."""

        result = await self._call_llm(prompt, system_prompt=system_prompt, temperature=0.85, context_label="universe_bible")

        # Validate that the LLM filled in real content (not schema literals)
        self._validate_bible_content(result)
        return result

    def _validate_bible_content(self, data: Dict[str, Any]) -> None:
        """Reject if the LLM returned angle-bracket schema template markers literally."""
        schema_pattern = re.compile(r"^<[^>]{1,50}>$")

        def check_value(val, path: str = "") -> None:
            if isinstance(val, str):
                stripped = val.strip()
                if schema_pattern.match(stripped):
                    raise LLMUnavailableError(
                        f"LLM returned schema placeholder '{stripped}' at '{path}' instead of real content. "
                        f"Model failed to fill in the template. Will retry."
                    )
            elif isinstance(val, dict):
                for k, v in val.items():
                    check_value(v, f"{path}.{k}")
            elif isinstance(val, list):
                for i, item in enumerate(val):
                    check_value(item, f"{path}[{i}]")

        check_value(data)


    # ------------------------------------------------------------------
    # Screenplay generation
    # ------------------------------------------------------------------
    async def generate_screenplay(
        self,
        universe_title: str,
        universe_genre: str,
        characters: List[Dict[str, Any]],
        past_memories: List[str],
        current_arc: str,
        episode_number: int,
        custom_prompt: str = "",
    ) -> Dict[str, Any]:
        """Generate a cinematic episode screenplay via the LLM.  Never returns hardcoded data."""

        char_block = "\n".join(
            f"  - {c['name']} ({c.get('role', 'Unknown')}): {c.get('personality', 'complex character')}. "
            f"Visual: {c.get('appearance_prompt', 'cinematic character')}"
            for c in characters
        ) if characters else "  - No characters defined yet. Invent 2-3 original characters appropriate to the genre."

        memories_block = "\n".join(f"  - {m}" for m in past_memories) if past_memories else "  - This is the very first episode. No prior story history exists."

        user_direction = custom_prompt or "Advance the plot with intense drama, cinematic tension, and meaningful character development."

        # Character names list for the prompt
        char_names = [c['name'] for c in characters] if characters else []
        char_names_str = ", ".join(char_names) if char_names else "create original characters"

        system_prompt = f"""You are an expert cinematic screenwriter creating Episode {episode_number} of "{universe_title}" (Genre: {universe_genre}).
Output ONLY a valid JSON object. No markdown. No explanations. No preamble.

IMPORTANT KEY NAMES — use these exactly:
- dialogue items MUST use "speaker" (character name) and "line" (spoken words)
- scenes MUST use "scene_number", "location", "visual_description", "image_prompt", "video_motion_prompt", "dialogue", "duration_seconds"
- top level MUST have "episode_title", "logline", "scenes"

Write 2 complete scenes with original content. Each scene needs 2-3 dialogue exchanges."""

        user_prompt = f"""Create an original cinematic screenplay for Episode {episode_number} of "{universe_title}".

STORY CONTEXT:
- Current Arc: {current_arc}
- Available Characters: {char_names_str}
- Character Details:
{char_block}

PREVIOUS EPISODE EVENTS (use these for continuity):
{memories_block}

DIRECTOR'S VISION: {user_direction}

Write the JSON screenplay now. Use character names from "{char_names_str}" as the "speaker" values in dialogue.
Make every scene visually rich and genre-appropriate for {universe_genre}.
Each dialogue "line" must be a complete, in-character spoken sentence.

JSON format:
{{
  "episode_title": "your original episode title",
  "logline": "one-sentence story summary",
  "scenes": [
    {{
      "scene_number": 1,
      "location": "specific location name",
      "visual_description": "detailed cinematic description",
      "image_prompt": "detailed image generation prompt",
      "video_motion_prompt": "camera movement and action description",
      "dialogue": [
        {{"speaker": "Character Name", "line": "What they say"}},
        {{"speaker": "Another Name", "line": "Their response"}}
      ],
      "duration_seconds": 8.0
    }},
    {{
      "scene_number": 2,
      "location": "different location",
      "visual_description": "detailed cinematic description",
      "image_prompt": "detailed image generation prompt",
      "video_motion_prompt": "camera movement and action description",
      "dialogue": [
        {{"speaker": "Character Name", "line": "What they say"}},
        {{"speaker": "Another Name", "line": "Their response"}}
      ],
      "duration_seconds": 8.0
    }}
  ]
}}"""

        return await self._call_llm(user_prompt, system_prompt=system_prompt, temperature=0.8, context_label="screenplay")


# Module-level singleton
llm_engine = QwenLLMEngine()
