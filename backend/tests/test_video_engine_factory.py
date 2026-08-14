import asyncio
import os

import pytest

from app.engines.video.mock_video_engine import MockVideoEngine
from app.engines.video.wan_video_engine import WanVideoEngine, initialize_video_engine


@pytest.mark.asyncio
async def test_initialize_video_engine_returns_mock_when_env_is_mock(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_ENGINE", "mock")
    monkeypatch.setenv("MEDIA_OUTPUT_DIR", str(tmp_path))

    engine = await initialize_video_engine()

    assert isinstance(engine, MockVideoEngine)


@pytest.mark.asyncio
async def test_initialize_video_engine_fails_when_wan_init_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_ENGINE", "wan")
    monkeypatch.setenv("WAN_MODEL_PATH", str(tmp_path / "missing-model"))
    monkeypatch.setenv("MEDIA_OUTPUT_DIR", str(tmp_path))

    with pytest.raises(Exception):
        await initialize_video_engine()


@pytest.mark.asyncio
async def test_wan_engine_can_render_video_to_mp4(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_ENGINE", "mock")
    monkeypatch.setenv("MEDIA_OUTPUT_DIR", str(tmp_path))

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
