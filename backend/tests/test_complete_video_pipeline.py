import json
import os
import tempfile
from pathlib import Path

from app.engines.story_engine import StoryEngine
from app.engines.character_memory import CharacterManager


def test_story_engine_creates_structured_story():
    engine = StoryEngine(provider="mock")
    story = engine.generate_story(
        genre="Fantasy",
        theme="Magic School",
        duration=60,
        language="English",
    )

    assert story["title"]
    assert story["characters"]
    assert story["scenes"]
    assert story["scenes"][0]["scene_number"] == 1


def test_character_manager_persists_and_builds_prompt(tmp_path):
    manager = CharacterManager(storage_dir=str(tmp_path))
    character = {
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

    saved = manager.save_character(character)
    loaded = manager.load_character(saved["uuid"])

    assert loaded["name"] == "Ethan"
    assert manager.get_prompt(saved["uuid"], scene_description="inside a magical academy")

    prompt_file = tmp_path / f"{saved['uuid']}.json"
    assert prompt_file.exists()
