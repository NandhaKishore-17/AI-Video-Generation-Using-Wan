import pytest
import os
import shutil
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.core.config import settings, BASE_DIR
from app.services.complete_video_service import complete_video_service
from app.engines.voice_engine import voice_engine
from app.engines.dialogue_engine import dialogue_engine
from app.engines.compositor import compositor

@pytest.mark.asyncio
async def test_complete_video_pipeline_bugfix():
    # Set mock providers to avoid remote API calls
    story_engine_provider_backup = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "mock"
    settings.VOICE_PROVIDER = "mock"
    settings.VIDEO_PROVIDER = "mock"

    import app.engines.video.wan_video_engine
    app.engines.video.wan_video_engine._video_engine_instance = None

    # Mock video generation to write dummy files and run quickly
    async def mock_generate_video(prompt, duration, width, height, fps, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("dummy video data")
        return output_path

    # Mock process_scene_dialogues to return simulated audio path
    async def mock_process_scene_dialogues(episode_id, scene_id, dialogue_list, base_output_dir):
        import wave
        ep_clean = str(episode_id).replace("episode_", "").zfill(3)
        sc_clean = str(scene_id).replace("scene_", "").zfill(2)
        scene_dir = os.path.join(base_output_dir, f"episode_{ep_clean}", f"scene_{sc_clean}")
        os.makedirs(scene_dir, exist_ok=True)
        composite_path = os.path.join(scene_dir, f"audio_scene_{sc_clean}.wav")
        # Generate a valid WAV file > 1 KB with 1 second duration
        with wave.open(composite_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00" * 48000) # 48 KB of silence
        return {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "composite_audio_path": composite_path,
            "dialogue_files": [],
            "timed_dialogues": [],
            "srt_content": "1\n00:00:00,000 --> 00:00:02,000\nspeaker: dialogue",
            "total_duration": 6.0
        }

    # Mock compositor.compose to verify what parameters it receives and write a mock final file
    mock_compose_calls = []
    def mock_compose(clips, voice_files, music_files, subtitle_files, output_name):
        mock_compose_calls.append({
            "clips": list(clips),
            "voice_files": list(voice_files),
            "music_files": list(music_files),
            "subtitle_files": list(subtitle_files),
            "output_name": output_name
        })
        output_path = compositor.output_dir / output_name
        with open(output_path, "w") as f:
            f.write("mock composed video")
        return {"output_path": str(output_path)}

    with patch("app.services.video_service.video_service.generate_video", side_effect=mock_generate_video), \
         patch.object(voice_engine.tts_service, "process_scene_dialogues", side_effect=mock_process_scene_dialogues), \
         patch.object(compositor, "compose", side_effect=mock_compose), \
         patch.object(compositor, "verify_composed_streams", return_value=(True, True)):
        
        # Run Episode 1
        job_id_1 = "job1111111111111111111111"
        res_ep1 = await complete_video_service.generate_complete_video(
            request={"genre": "Fantasy", "theme": "Magic School", "duration": 60, "episode_number": 1},
            job_id=job_id_1
        )
        
        # Run Episode 2
        job_id_2 = "job2222222222222222222222"
        res_ep2 = await complete_video_service.generate_complete_video(
            request={"genre": "Fantasy", "theme": "Magic School", "duration": 60, "episode_number": 2},
            job_id=job_id_2
        )

        # ── 1. Episode 1 dialogue != Episode 2 dialogue ──────────────────────────
        job_dir_1 = Path(BASE_DIR) / "media" / "jobs" / job_id_1[:7]
        job_dir_2 = Path(BASE_DIR) / "media" / "jobs" / job_id_2[:7]
        
        with open(job_dir_1 / "script.json", "r") as f:
            script_1 = json.load(f)
        with open(job_dir_2 / "script.json", "r") as f:
            script_2 = json.load(f)

        assert script_1["dialogues"] != script_2["dialogues"], "Episode 1 dialogue and Episode 2 dialogue are identical!"

        # ── 2. Episode 2 audio != Episode 1 audio ─────────────────────────────────
        ep1_audio_files = res_ep1["voice_files"]
        ep2_audio_files = res_ep2["voice_files"]
        assert ep1_audio_files != ep2_audio_files, "Episode 1 and Episode 2 audio files are the same!"
        for path in ep1_audio_files + ep2_audio_files:
            clean_path = path.lstrip("/").replace("media/", "", 1)
            full_path = Path(BASE_DIR) / "media" / clean_path
            assert full_path.exists(), f"Audio file {full_path} was not created"

        # Check unique folders exist under media/
        ep1_dir = Path(BASE_DIR) / "media" / "episode_001"
        ep2_dir = Path(BASE_DIR) / "media" / "episode_002"
        assert ep1_dir.exists(), "media/episode_001 directory was not created"
        assert ep2_dir.exists(), "media/episode_002 directory was not created"
        assert len(list(ep1_dir.glob("*.wav"))) > 0
        assert len(list(ep2_dir.glob("*.wav"))) > 0

        # ── 3. New subtitles generated every episode ──────────────────────────────
        ep1_subtitles = res_ep1["subtitle_files"]
        ep2_subtitles = res_ep2["subtitle_files"]
        assert ep1_subtitles != ep2_subtitles, "Episode 1 and Episode 2 subtitle files are the same!"
        for srt_path in ep1_subtitles + ep2_subtitles:
            assert Path(srt_path).exists(), f"Subtitle file {srt_path} was not created"

        # ── 4. FFmpeg merges current audio only ────────────────────────────────────
        assert len(mock_compose_calls) == 2
        for vf in mock_compose_calls[0]["voice_files"]:
            assert "job1111" in vf
        for vf in mock_compose_calls[1]["voice_files"]:
            assert "job2222" in vf

    # Restore settings
    settings.LLM_PROVIDER = story_engine_provider_backup
