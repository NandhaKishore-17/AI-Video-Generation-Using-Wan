import json
from pathlib import Path

from app.engines.episode_memory import EpisodeMemoryManager
from app.engines.prompt_generator import PromptGenerator


def test_episode_memory_builds_continuity_payload(tmp_path):
    manager = EpisodeMemoryManager(storage_dir=str(tmp_path))

    manager.save_episode(
        universe_id="universe-demo",
        episode_number=1,
        episode_data={
            "title": "The First Breach",
            "summary": "The heroes discover the vault.",
            "cliffhanger": "The vault opens and a shadow rises.",
        },
        characters=[{"name": "Mira", "role": "Protagonist"}],
        scenes=[{"id": "scene01", "title": "Vault Entrance"}],
        memory_notes={"villain_progress": "The villain has found the map."},
    )

    context = manager.build_context(universe_id="universe-demo", episode_number=2)

    assert context["episode_number"] == 2
    assert context["previous_cliffhanger"] == "The vault opens and a shadow rises."
    assert context["episode_history"][0]["episode_number"] == 1
    assert context["memory_notes"]["villain_progress"] == "The villain has found the map."


def test_prompt_generator_builds_scene_assets_from_screenplay():
    generator = PromptGenerator()
    scene = {
        "id": "scene02",
        "title": "Midnight Lab",
        "description": "Mira enters the lab while the lights flicker.",
        "camera": "dolly in",
        "lighting": "cold blue rim light",
        "mood": "tense",
        "dialogues": [{"speaker": "Mira", "text": "The vault is waking up."}],
    }
    payload = generator.build_scene_prompt(
        scene=scene,
        characters=[{"name": "Mira", "role": "Protagonist", "personality": "brave"}],
        memory_context={"previous_cliffhanger": "The vault opens."},
    )

    assert payload["visual_prompt"]
    assert payload["motion_prompt"]
    assert payload["camera"] == "dolly in"
    assert payload["dialogues"][0]["speaker"] == "Mira"
    assert payload["sound_fx"]
