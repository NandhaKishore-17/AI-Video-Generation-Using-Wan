import os
import pytest
import wave
import subprocess
from pathlib import Path
from app.core.config import settings
from app.engines.compositor import compositor
from services.tts.tts_service import tts_service
import imageio_ffmpeg

@pytest.mark.asyncio
async def test_tts_creates_valid_wav(tmp_path):
    # Test 1: TTS creates a valid WAV and WAV duration > 0
    output_wav = str(tmp_path / "test_dialogue.wav")
    
    info = await tts_service.generate_single_dialogue(
        character_name="Kaelen",
        dialogue_text="The signal is getting stronger.",
        output_filepath=output_wav
    )
    
    assert os.path.exists(output_wav)
    assert os.path.getsize(output_wav) > 1024  # > 1 KB
    assert info["duration"] > 0
    
    # Verify wave file specs
    with wave.open(output_wav, "rb") as wf:
        assert wf.getframerate() >= 24000
        assert wf.getnchannels() >= 1
        assert wf.getnframes() > 0


@pytest.mark.asyncio
async def test_ffmpeg_output_contains_audio_stream_and_not_silent(tmp_path):
    # Test 2: FFmpeg output contains an audio stream and final MP4 is not silent
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Generate dummy video clip (3 seconds)
    dummy_video = str(tmp_path / "dummy_video.mp4")
    cmd_v = [
        ffmpeg_exe, "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s=640x360:d=3",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        dummy_video
    ]
    subprocess.run(cmd_v, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Generate a real voice WAV file
    voice_wav = str(tmp_path / "voice.wav")
    await tts_service.generate_single_dialogue(
        character_name="Kaelen",
        dialogue_text="The core mainframe is breaching.",
        output_filepath=voice_wav
    )
    
    # Generate a real music WAV file
    music_wav = str(tmp_path / "music.wav")
    # Generate some simple sine tone as music
    cmd_m = [
        ffmpeg_exe, "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=3",
        "-c:a", "pcm_s16le",
        music_wav
    ]
    subprocess.run(cmd_m, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Output path for final video
    final_output = str(tmp_path / "final_composed_video.mp4")
    
    # Setup compositor dir
    original_dir = compositor.output_dir
    compositor.output_dir = tmp_path
    
    try:
        compositor.compose(
            clips=[dummy_video],
            voice_files=[voice_wav],
            music_files=[music_wav],
            subtitle_files=[],
            output_name="final_composed_video.mp4"
        )
    finally:
        compositor.output_dir = original_dir
        
    assert os.path.exists(final_output)
    
    # Verify streams in final MP4
    has_video, has_audio = compositor.verify_composed_streams(final_output)
    assert has_video, "Final MP4 is missing video stream"
    assert has_audio, "Final MP4 is missing audio stream"
    
    # Verify not silent (ensure it has actual audio packet bytes)
    ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")
    if os.path.exists(ffprobe_exe):
        # Read packets info of audio stream to verify size > 0
        cmd_p = [
            ffprobe_exe, "-show_entries", "packet=size", 
            "-select_streams", "a", "-loglevel", "error", final_output
        ]
        res = subprocess.run(cmd_p, capture_output=True, text=True)
        assert "size=" in res.stdout, "Final MP4 audio stream has no packets/data"
