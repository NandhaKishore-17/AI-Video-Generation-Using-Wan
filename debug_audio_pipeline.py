import os
import sys
import asyncio
import logging
import wave
import subprocess

# Set up logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("debug_audio")

# Add backend directory to sys.path
backend_dir = os.path.abspath("backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.core.config import settings
from app.engines.story_engine import story_engine
from app.engines.voice_engine import voice_engine
from app.engines.music_engine import music_engine
from app.engines.subtitle_engine import subtitle_engine
from app.engines.compositor import compositor
from app.services.complete_video_service import complete_video_service
from services.tts.tts_service import tts_service

async def test_pipeline():
    logger.info("=== STEP 1: Tracing Audio Flow ===")
    
    # 1. Generate Story
    story = story_engine.generate_story(
        genre="Fantasy",
        theme="Magic School",
        duration=15,
        language="English"
    )
    logger.info(f"Generated Story Title: {story.get('title')}")
    logger.info(f"Number of scenes: {len(story.get('scenes', []))}")
    
    # 2. Verify Voice Engine & TTS
    logger.info("=== STEP 2 & 3: Verifying Voice Engine & TTS ===")
    backend = tts_service.get_backend()
    logger.info(f"TTS Provider: {tts_service.provider}")
    logger.info(f"TTS Backend Class: {type(backend).__name__}")
    
    scene = story.get("scenes", [])[0]
    dialogue_list = [
        {"speaker": "Narrator", "line": scene.get("narration", "Once upon a time in a magical academy."), "context": "Intro"},
        {"speaker": "Luna", "line": "Look at that glowing spellbook!", "context": "Excited"}
    ]
    
    audio_rel_url, srt_content, total_duration = await voice_engine.generate_scene_audio_and_srt(
        scene_id="scene_01",
        dialogue_list=dialogue_list,
        episode_id="ep_test"
    )
    logger.info(f"Voice engine returned URL: {audio_rel_url}")
    logger.info(f"SRT Content generated:\n{srt_content}")
    logger.info(f"Reported total duration: {total_duration:.2f}s")
    
    # Check generated WAV files
    media_dir = settings.MEDIA_OUTPUT_DIR
    audio_filename = os.path.basename(audio_rel_url)
    wav_path = os.path.join(media_dir, audio_filename)
    logger.info(f"Audio file path on disk: {wav_path}")
    
    # 3. Verify Audio Output (Task 4)
    logger.info("=== STEP 4: Verifying Audio File Details ===")
    if os.path.exists(wav_path):
        size = os.path.getsize(wav_path)
        with wave.open(wav_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            dur = frames / float(sample_rate)
        logger.info(f"WAV File: {audio_filename}")
        logger.info(f"  Sample Rate: {sample_rate} Hz")
        logger.info(f"  Channels   : {channels}")
        logger.info(f"  Duration   : {dur:.2f} s")
        logger.info(f"  File Size  : {size} bytes")
    else:
        logger.error(f"WAV File DOES NOT EXIST at: {wav_path}")

    # 4. Verify Background Music (Task 5)
    logger.info("=== STEP 5: Verifying Background Music ===")
    music_info = music_engine.generate_music(scene, duration_seconds=total_duration)
    music_path = music_info["path"]
    logger.info(f"Music File Path: {music_path}")
    if os.path.exists(music_path):
        size = os.path.getsize(music_path)
        with wave.open(music_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            dur = frames / float(sample_rate)
        logger.info(f"Music WAV File: {os.path.basename(music_path)}")
        logger.info(f"  Sample Rate: {sample_rate} Hz")
        logger.info(f"  Channels   : {channels}")
        logger.info(f"  Duration   : {dur:.2f} s")
        logger.info(f"  File Size  : {size} bytes")

    # 5. Run Complete Video Service to trace FFmpeg composition
    logger.info("=== STEP 6 & 7: Testing Complete Video Service Generation ===")
    request = {"genre": "Fantasy", "theme": "Magic School", "duration": 15, "language": "English"}
    res = await complete_video_service.generate_complete_video(request, job_id="test_job_123")
    final_output = res["output_path"]
    logger.info(f"Final output video path: {final_output}")
    
    # Run ffprobe on final video
    if os.path.exists(final_output):
        cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=index,codec_name,codec_type,duration,channels,sample_rate", "-of", "default=noprint_wrappers=1", final_output]
        logger.info(f"Running ffprobe command: {' '.join(cmd)}")
        probe_res = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"ffprobe output:\n{probe_res.stdout}")
        if probe_res.stderr:
            logger.warning(f"ffprobe stderr:\n{probe_res.stderr}")

if __name__ == "__main__":
    asyncio.run(test_pipeline())
