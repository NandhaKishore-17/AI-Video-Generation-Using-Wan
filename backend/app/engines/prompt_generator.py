import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Transform a story scene into a cinematic prompt suitable for Wan2.2."""

    def __init__(self):
        self.negative_prompt = (
            "blurry, low detail, distorted face, extra limbs, text, watermark, bad anatomy, oversaturated"
        )

    def build_scene_prompt(self, scene: Dict[str, Any], characters: List[Dict[str, Any]], emotion: str = "neutral") -> Dict[str, Any]:
        character = characters[0] if characters else {"name": "character", "hair": "dark", "clothes": "cinematic outfit"}
        description = scene.get("description", "")
        dialogue = scene.get("dialogue", "")
        narration = scene.get("narration", "")
        prompt = (
            f"A cinematic ultra realistic scene of {character['name']}, {character['age'] if 'age' in character else 'young'}-year-old "
            f"{character['gender'] if 'gender' in character else 'character'} with {character['hair']} hair wearing {character['clothes']}, "
            f"{description}, cinematic lighting, {emotion} mood, volumetric lighting, masterpiece, highly detailed, fantasy."
        )
        return {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "camera": "cinematic tracking shot",
            "lighting": "volumetric rim lighting",
            "emotion": emotion,
            "environment": description,
            "character_appearance": f"{character['name']} with {character['hair']} hair and {character['clothes']}",
            "art_style": "ultra realistic fantasy",
            "narration": narration,
            "dialogue": dialogue,
        }


prompt_generator = PromptGenerator()
