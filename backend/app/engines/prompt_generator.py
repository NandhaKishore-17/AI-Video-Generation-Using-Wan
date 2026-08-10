import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Transform a story scene into cinematic prompts that reflect the screenplay and continuity context."""

    def __init__(self):
        self.negative_prompt = (
            "blurry, low detail, distorted face, extra limbs, text, watermark, bad anatomy, oversaturated"
        )

    def build_scene_prompt(
        self,
        scene: Dict[str, Any],
        characters: List[Dict[str, Any]],
        emotion: str = "neutral",
        memory_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        character = characters[0] if characters else {"name": "character", "hair": "dark", "clothes": "cinematic outfit"}
        description = scene.get("description", "")
        title = scene.get("title") or scene.get("id") or "scene"
        location = scene.get("location") or scene.get("environment") or ""
        mood = scene.get("mood") or emotion or "neutral"
        camera = scene.get("camera") or "cinematic tracking shot"
        lighting = scene.get("lighting") or "dramatic rim lighting"
        dialogues = scene.get("dialogues") or scene.get("dialogue", [])
        dialogue_text = ""
        if isinstance(dialogues, list):
            dialogue_text = " ".join(
                [item.get("text") or item.get("line") or "" for item in dialogues if item.get("text") or item.get("line")]
            )
        elif dialogues:
            dialogue_text = str(dialogues)

        previous_cliffhanger = (memory_context or {}).get("previous_cliffhanger", "")
        visual_prompt = (
            f"A cinematic scene titled {title} featuring {character['name']} in {location or 'a vivid environment'} "
            f"{description}. Emphasize {mood} atmosphere, {lighting}, and a {camera} that feels intimate and urgent."
        )
        if previous_cliffhanger:
            visual_prompt += f" The scene should echo the previous cliffhanger: {previous_cliffhanger}."
        motion_prompt = (
            f"Slow, deliberate camera motion with {camera}, responsive character movement, subtle environmental motion, "
            f"and a {mood} pacing that builds tension around the current conflict."
        )
        image_prompt = visual_prompt
        video_prompt = f"{image_prompt} {motion_prompt}"
        sound_fx = "low hum, distant thunder, footsteps, subtle ambient tension"

        normalized_dialogues = []
        if isinstance(dialogues, list):
            normalized_dialogues = [
                {"speaker": item.get("speaker") or "Narrator", "text": item.get("text") or item.get("line") or ""}
                for item in dialogues
                if item.get("text") or item.get("line")
            ]

        return {
            "prompt": visual_prompt,
            "visual_prompt": visual_prompt,
            "motion_prompt": motion_prompt,
            "image_prompt": image_prompt,
            "video_prompt": video_prompt,
            "negative_prompt": self.negative_prompt,
            "camera": camera,
            "lighting": lighting,
            "mood": mood,
            "environment": location or description,
            "character_appearance": f"{character['name']} with {character.get('hair', 'dark')} hair and {character.get('clothes', 'cinematic outfit')}",
            "art_style": "cinematic fantasy",
            "narration": scene.get("description", ""),
            "dialogue": dialogue_text,
            "dialogues": normalized_dialogues,
            "sound_fx": sound_fx,
        }


prompt_generator = PromptGenerator()
