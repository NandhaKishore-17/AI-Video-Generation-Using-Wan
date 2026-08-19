import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from app.engines.llm_engine import llm_engine
from app.modules.universe_engine import universe_module
from app.modules.story_director import story_director
from app.models.domain import Universe, Episode, Character, RenderTask, Scene
from app.schemas.schemas import ScreenplayData, UniverseGeneratedData
import asyncio
from pathlib import Path

@pytest.fixture
def mock_db():
    db = MagicMock()
    # Ensure db.query().filter().first() doesn't immediately blow up, can configure specific returns in tests
    return db

def test_json_repair_logic():
    # Test markdown stripping and invalid escapes
    bad_json = "```json\n{\n  \"key\": \"value\\_with_escapes\",\n  \"scenes\": [\n    {\"scene_number\": 1}\n  ]\n}\n```"
    # The JSON repair logic should fix the escape and strip the markdown
    repaired = llm_engine._extract_json(bad_json)
    assert repaired["key"] == "value_with_escapes"
    assert repaired["scenes"][0]["scene_number"] == 1

@pytest.mark.asyncio
async def test_bounded_retry_success_second_try():
    # First call returns bad JSON, second call returns good JSON
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        "not json at all",
        '{"world_summary": "Test World"}'
    ]
    with patch("app.engines.llm_engine.ollama_client", mock_client):
        llm_engine.client = mock_client
        data = await llm_engine._generate_with_retry("system", "prompt", UniverseGeneratedData)
        assert data["world_summary"] == "Test World"
        assert mock_client.generate.call_count == 2

@pytest.mark.asyncio
async def test_bounded_retry_failure():
    # Always returns bad JSON
    mock_client = MagicMock()
    mock_client.generate.return_value = "not json at all"
    with patch("app.engines.llm_engine.ollama_client", mock_client):
        llm_engine.client = mock_client
        with pytest.raises(ValueError) as exc:
            await llm_engine._generate_with_retry("system", "prompt", UniverseGeneratedData, max_attempts=2)
        assert "Ollama failed to produce valid JSON after 2 attempts" in str(exc.value)

@pytest.mark.asyncio
async def test_character_validation_failures():
    mock_client = MagicMock()
    # Returns JSON with an invalid speaker
    bad_speaker_json = json.dumps({
        "episode_title": "Title",
        "logline": "Log",
        "summary": "Sum",
        "character_states": {"Hero": "Happy"},
        "scenes": [
            {
                "scene_number": 1,
                "location": "Loc",
                "visual_description": "Vis",
                "emotion": "Emo",
                "image_prompt": "Img",
                "video_motion_prompt": "Vid",
                "negative_prompt": "Neg",
                "dialogue": [{"speaker": "Villain", "line": "Muhaha"}]
            }
        ]
    })
    
    mock_client.generate.side_effect = [
        bad_speaker_json,
        "not json at all"
    ]
    with patch("app.engines.llm_engine.ollama_client", mock_client):
        llm_engine.client = mock_client
        with pytest.raises(ValueError) as exc:
            await llm_engine._generate_with_retry("system", "prompt", ScreenplayData, max_attempts=1, allowed_speakers=["Hero"])
        assert "Invalid speaker 'Villain'" in str(exc.value)

def test_timeline_event_normalization():
    # Test handling of LLM hallucinated dictionary format
    event_dict = {"title": "The Fall", "timestamp": "Year 50", "description": "It fell."}
    title, timestamp, desc = universe_module._normalize_timeline_event(event_dict, 1)
    assert title == "The Fall"
    assert timestamp == "Year 50"
    assert desc == "It fell."

    # Test handling of string
    event_str = "The Fall happened"
    title, timestamp, desc = universe_module._normalize_timeline_event(event_str, 2)
    assert title == "The Fall happened"
    assert timestamp == "Era 1, Year 2020"

@pytest.mark.asyncio
async def test_universe_creation_rollback(mock_db):
    # If llm_engine raises an exception, the universe creation must rollback
    mock_db.add = MagicMock()
    mock_db.flush = MagicMock()
    mock_db.rollback = MagicMock()
    mock_db.commit = MagicMock()

    with patch("app.engines.llm_engine.llm_engine.generate_universe_bible", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = Exception("LLM Error")
        with pytest.raises(Exception):
            await universe_module.create_full_universe(mock_db, "Title", "Genre", "Logline")
        
        # Ensure rollback was called and commit was not called after error
        mock_db.rollback.assert_called_once()
        mock_db.commit.assert_not_called()

def test_media_url_existence_verification():
    from app.api.endpoints import get_episode_detail
    from fastapi import HTTPException
    from datetime import datetime
    
    mock_db = MagicMock()
    mock_episode = Episode(id="ep-1", final_video_url="/media/episodes/ep-1/final.mp4", universe_id="u1", season=1, episode_number=1, title="Test", logline="Log", status="COMPLETED", duration_seconds=120.0, created_at=datetime.now(), updated_at=datetime.now())
    mock_scene = Scene(id="s1", episode_id="ep-1", scene_number=1, location="Loc", time_of_day="Day", visual_description="Vis", image_prompt="Img", video_motion_prompt="Vid", dialogue_script=[], duration_seconds=8.0, image_url="/media/episodes/ep-1/1.png")
    
    mock_db.query().filter().first.return_value = mock_episode
    mock_db.query().filter().order_by().all.return_value = [mock_scene]
    
    # Run the endpoint logic
    # The actual files do not exist on disk, so it should nullify them
    with patch("pathlib.Path.exists", return_value=False):
        resp = get_episode_detail("ep-1", db=mock_db)
        assert resp.final_video_url is None
        assert resp.scenes[0].image_url is None

    # If they do exist, it should keep them
    mock_episode = Episode(id="ep-1", final_video_url="/media/episodes/ep-1/final.mp4", universe_id="u1", season=1, episode_number=1, title="Test", logline="Log", status="COMPLETED", duration_seconds=120.0, created_at=datetime.now(), updated_at=datetime.now())
    mock_scene = Scene(id="s1", episode_id="ep-1", scene_number=1, location="Loc", time_of_day="Day", visual_description="Vis", image_prompt="Img", video_motion_prompt="Vid", dialogue_script=[], duration_seconds=8.0, image_url="/media/episodes/ep-1/1.png")
    mock_db.query().filter().first.return_value = mock_episode
    mock_db.query().filter().order_by().all.return_value = [mock_scene]
    
    with patch("pathlib.Path.exists", return_value=True):
        resp = get_episode_detail("ep-1", db=mock_db)
        assert resp.final_video_url == "/media/episodes/ep-1/final.mp4"
        assert resp.scenes[0].image_url == "/media/episodes/ep-1/1.png"

@pytest.mark.asyncio
async def test_episode_numbering_continuity():
    mock_db = MagicMock()
    
    mock_universe = Universe(id="u1", title="Test", genre="G", logline="L", auto_generate_active=True, total_episodes=5, current_season=1)
    
    def side_effect_query(model):
        query_mock = MagicMock()
        if model == Universe:
            query_mock.filter.return_value.first.return_value = mock_universe
        elif model == Episode:
            # wait, db.query(func.max(...)) is a function call, so model is not Episode, it's a ColumnElement.
            pass
        return query_mock

    # A simpler way to mock:
    mock_db.query.return_value.filter.return_value.first.side_effect = [mock_universe, None, None] # First for Universe, second for StoryArc, third for StoryMemory
    # Mock max episode number
    mock_db.query.return_value.filter.return_value.scalar.return_value = 5
    
    with patch("app.engines.memory_engine.memory_engine.query_relevant_memories", new_callable=AsyncMock) as mock_mem:
        mock_mem.return_value = []
        brief = await story_director.prepare_next_episode_brief(mock_db, "u1")
        # Should be 5 + 1 = 6
        assert brief["episode_number"] == 6
