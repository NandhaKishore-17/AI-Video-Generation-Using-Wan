import asyncio
import os

import pytest

from app.core.config import settings
from app.engines.video.mock_video_engine import MockVideoEngine
from app.engines.video.wan_video_engine import WanVideoEngine, initialize_video_engine


@pytest.mark.asyncio
async def test_initialize_video_engine_returns_mock_when_env_is_mock(monkeypatch, tmp_path):
    import app.engines.video.wan_video_engine
    app.engines.video.wan_video_engine._video_engine_instance = None
    
    settings_video_engine_backup = settings.VIDEO_ENGINE
    settings_video_provider_backup = settings.VIDEO_PROVIDER
    settings_media_dir_backup = settings.MEDIA_OUTPUT_DIR
    
    settings.VIDEO_ENGINE = "mock"
    settings.VIDEO_PROVIDER = "mock"
    settings.MEDIA_OUTPUT_DIR = str(tmp_path)

    try:
        engine = await initialize_video_engine()
        assert isinstance(engine, MockVideoEngine)
    finally:
        settings.VIDEO_ENGINE = settings_video_engine_backup
        settings.VIDEO_PROVIDER = settings_video_provider_backup
        settings.MEDIA_OUTPUT_DIR = settings_media_dir_backup


@pytest.mark.asyncio
async def test_initialize_video_engine_fails_when_wan_init_fails(monkeypatch, tmp_path):
    import app.engines.video.wan_video_engine
    app.engines.video.wan_video_engine._video_engine_instance = None

    settings_video_engine_backup = settings.VIDEO_ENGINE
    settings_model_path_backup = settings.WAN_MODEL_PATH
    settings_media_dir_backup = settings.MEDIA_OUTPUT_DIR

    settings.VIDEO_ENGINE = "wan"
    settings.WAN_MODEL_PATH = str(tmp_path / "missing-model")
    settings.MEDIA_OUTPUT_DIR = str(tmp_path)

    try:
        with pytest.raises(Exception):
            await initialize_video_engine()
    finally:
        settings.VIDEO_ENGINE = settings_video_engine_backup
        settings.WAN_MODEL_PATH = settings_model_path_backup
        settings.MEDIA_OUTPUT_DIR = settings_media_dir_backup


@pytest.mark.asyncio
async def test_wan_engine_can_render_video_to_mp4(monkeypatch, tmp_path):
    import app.engines.video.wan_video_engine
    app.engines.video.wan_video_engine._video_engine_instance = None

    settings_video_engine_backup = settings.VIDEO_ENGINE
    settings_media_dir_backup = settings.MEDIA_OUTPUT_DIR

    settings.VIDEO_ENGINE = "mock"
    settings.MEDIA_OUTPUT_DIR = str(tmp_path)

    try:
        engine = await initialize_video_engine()
        output_path = str(tmp_path / "generated.mp4")

        result_path = await asyncio.to_thread(
            engine.render_video,
            "A calm sky drifting above a quiet lake",
            output_path,
            640,
            360,
            12,
            1.5,
            7,
        )

        assert result_path == output_path
        assert os.path.exists(output_path)
    finally:
        settings.VIDEO_ENGINE = settings_video_engine_backup
        settings.MEDIA_OUTPUT_DIR = settings_media_dir_backup
