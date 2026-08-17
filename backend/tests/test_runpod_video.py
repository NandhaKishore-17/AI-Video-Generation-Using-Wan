import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from app.services.video_service import video_service
from app.core.config import settings
import os

@pytest.fixture
def mock_settings():
    with patch("app.services.runpod_wan_client.settings") as mock_set:
        mock_set.RUNPOD_API_KEY = "test_key"
        mock_set.RUNPOD_WAN_ENDPOINT_URL = "https://api.runpod.ai/v2/test"
        mock_set.WAN_MODEL = "Wan2.2-TI2V-5B"
        mock_set.WAN_TIMEOUT = 300
        mock_set.WAN_POLL_INTERVAL = 0
        yield mock_set

@pytest.mark.asyncio
async def test_generate_video_success(mock_settings, tmp_path):
    submit_response = MagicMock()
    submit_response.status_code = 200
    submit_response.json.return_value = {"id": "123", "status": "IN_QUEUE"}

    poll_response = MagicMock()
    poll_response.status_code = 200
    poll_response.json.return_value = {"id": "123", "status": "COMPLETED", "output": {"video_url": "https://video.mp4"}}

    download_response = MagicMock()
    download_response.status_code = 200
    
    async def mock_aiter():
        yield b"fake_video_data"
        
    download_response.aiter_bytes.return_value = mock_aiter()

    mock_client = AsyncMock()
    mock_client.post.return_value = submit_response
    mock_client.get.side_effect = [poll_response, download_response]

    mock_client_ctx = AsyncMock()
    mock_client_ctx.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client_ctx):
        test_output = str(tmp_path / "test.mp4")
        result = await video_service.generate_video(
            prompt="test prompt",
            output_path=test_output
        )

        assert result.startswith("/media/")
        assert os.path.exists(test_output)

@pytest.mark.asyncio
async def test_generate_video_auth_failure(mock_settings, tmp_path):
    submit_response = MagicMock()
    submit_response.status_code = 401

    mock_client = AsyncMock()
    mock_client.post.return_value = submit_response
    
    mock_client_ctx = AsyncMock()
    mock_client_ctx.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client_ctx):
        test_output = str(tmp_path / "test.mp4")
        with pytest.raises(Exception, match="Runpod authentication failed"):
            await video_service.generate_video(
                prompt="test prompt",
                output_path=test_output
            )

@pytest.mark.asyncio
async def test_generate_video_missing_config():
    with patch("app.services.runpod_wan_client.settings") as mock_set:
        mock_set.RUNPOD_API_KEY = None
        mock_set.RUNPOD_WAN_ENDPOINT_URL = "test"
        
        with pytest.raises(Exception, match="RUNPOD_API_KEY is not configured"):
            await video_service.generate_video(prompt="test prompt")

@pytest.mark.asyncio
async def test_image_to_video_local_url_rejection():
    with pytest.raises(Exception, match="Image-to-video requires an image accessible"):
        await video_service.generate_video(prompt="test", image_url="/media/local.jpg")
        
    with pytest.raises(Exception, match="Image-to-video requires an image accessible"):
        await video_service.generate_video(prompt="test", image_url="http://localhost:8000/image.jpg")
