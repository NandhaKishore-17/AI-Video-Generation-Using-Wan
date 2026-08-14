"""
Story generator.

Takes a user prompt, asks Ollama to produce a structured JSON story,
validates the schema, saves it to the database, and returns the result.
"""
import json
import logging
import re
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.engines.llm.ollama_client import OllamaClient, OllamaError, ollama_client
from app.models.story import Story

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────
# Prompt template
# ────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a professional screenplay writer and story designer.
You MUST respond with ONLY valid JSON — no markdown fences, no prose, no explanations.
The JSON must exactly match this schema:

{
  "title": "<string>",
  "summary": "<string>",
  "characters": [
    {
      "name": "<string>",
      "description": "<string>"
    }
  ],
  "scenes": [
    {
      "scene_number": <integer starting at 1>,
      "prompt": "<cinematic visual description for video generation>",
      "dialogue": "<what characters say in this scene>"
    }
  ]
}

Rules:
- title: concise title for the story
- summary: 2-3 sentence synopsis
- characters: at least 1, at most 5
- scenes: at least 3, at most 6
- Every field is required; never use null
"""

_USER_TEMPLATE = """\
Write a short story for the following idea:

{prompt}

Return ONLY the JSON. No preamble. No markdown.
"""


# ────────────────────────────────────────────
# JSON extraction / validation helpers
# ────────────────────────────────────────────

def _extract_json(raw: str) -> str:
    """Strip markdown fences and extract the first JSON object found."""
    # Remove ```json ... ``` or ``` ... ``` fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the outermost {...}
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    raise ValueError("JSON object not properly closed")


def _validate_story(data: Dict[str, Any]) -> None:
    """Validate the story dict against the required schema."""
    required_top = {"title", "summary", "characters", "scenes"}
    missing = required_top - data.keys()
    if missing:
        raise ValueError(f"Story JSON missing top-level keys: {missing}")

    if not isinstance(data["characters"], list) or len(data["characters"]) == 0:
        raise ValueError("'characters' must be a non-empty list")
    for i, c in enumerate(data["characters"]):
        if "name" not in c or "description" not in c:
            raise ValueError(f"Character {i} missing 'name' or 'description'")

    if not isinstance(data["scenes"], list) or len(data["scenes"]) < 1:
        raise ValueError("'scenes' must be a non-empty list")
    for i, s in enumerate(data["scenes"]):
        for field in ("scene_number", "prompt", "dialogue"):
            if field not in s:
                raise ValueError(f"Scene {i} missing '{field}'")


# ────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────

class StoryGenerator:
    """Generate and persist stories using Ollama."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self._client = client or ollama_client

    def generate(self, prompt: str, job_id: str) -> Dict[str, Any]:
        """Generate a story for *prompt*, save it to the DB, and return the dict.

        Raises StoryGenerationError on failure.
        """
        user_message = _USER_TEMPLATE.format(prompt=prompt.strip())

        last_error = None
        for attempt in range(3):
            # ── Call Ollama ──────────────────────────────────────────────────
            logger.info("Generating story via Ollama for job %s (attempt %d/3) …", job_id, attempt + 1)
            try:
                raw = self._client.generate(prompt=user_message, system=_SYSTEM_PROMPT, format="json")
            except OllamaError as exc:
                raise StoryGenerationError(f"Ollama call failed: {exc}") from exc

            # ── Parse & validate ─────────────────────────────────────────────
            logger.debug("Raw Ollama response: %s", raw[:500])
            try:
                json_str = _extract_json(raw)
                story_data = json.loads(json_str)
                _validate_story(story_data)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning("Invalid JSON from Ollama on attempt %d: %s\nRaw: %s", attempt + 1, exc, raw[:300])
                continue
        else:
            raise StoryGenerationError(f"Failed to generate valid JSON from Ollama after 3 attempts: {last_error}")

        # ── Persist ──────────────────────────────────────────────────────
        story_id = str(uuid.uuid4())
        db: Session = SessionLocal()
        try:
            story = Story(
                id=story_id,
                job_id=job_id,
                prompt=prompt,
                title=story_data.get("title", ""),
                summary=story_data.get("summary", ""),
                story_data=story_data,
            )
            db.add(story)
            db.commit()
            logger.info("Story %s saved to DB (job=%s)", story_id, job_id)
        except Exception as exc:
            db.rollback()
            raise StoryGenerationError(f"DB persist failed: {exc}") from exc
        finally:
            db.close()

        return {"story_id": story_id, "story": story_data}


class StoryGenerationError(RuntimeError):
    """Raised when story generation fails at any stage."""


# Module-level singleton
story_generator = StoryGenerator()
