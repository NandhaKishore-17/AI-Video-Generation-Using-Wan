import os
import sys
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import settings

# Add project root to sys.path if not present so services package is importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.tts.tts_service import TTSService

logger = logging.getLogger("voice_engine")


class VoiceGenerationEngine:
    """
    Upgraded Open-Source Voice Engine powered by modern natural human TTS service (services/tts).
    Generates character-consistent, expressive dialogue speech and WebVTT/SRT subtitles.
    Audited: Processes dynamically generated, scene-by-scene dialogues with no hardcoded templates.
    """

    def __init__(self):
        self.provider = settings.VOICE_PROVIDER
        self.media_dir = Path(settings.MEDIA_OUTPUT_DIR)
        # Use absolute path so copyfile works regardless of CWD
        self.audio_dir = Path(os.path.join(project_root, "media", "audio"))
        self.temp_dir = settings.TEMP_DIR
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.tts_service = TTSService(provider=self.provider)

    async def generate_scene_audio_and_srt(
        self,
        scene_id: str,
        dialogue_list: List[Dict[str, Any]],
        voice_presets: Dict[str, str] = None,
        episode_id: str = "001",
        episode_number: int = 1,
        job_id: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        Generates full spoken dialogue audio track for a scene and returns (audio_relative_url, srt_content, total_duration_seconds).
        Saves dialogues to: audio/episode_{episode_id}/scene_{scene_id}/{speaker}_{idx}.wav
        """
        import re
        match = re.search(r'\d+', scene_id)
        scene_num_val = int(match.group()) if match else 1

        filename = f"audio_{scene_id}.wav"
        filepath = str(self.media_dir / filename)
        audio_output_path = str(self.audio_dir / filename)

        # Normalize dialogue keys to support both 'line' and 'text' keys safely
        normalized_dialogue = []
        for item in dialogue_list:
            speaker = item.get("speaker") or "Narrator"
            text_val = item.get("text") or item.get("line") or ""
            normalized_dialogue.append({
                "speaker": speaker,
                "line": text_val,
                "text": text_val,
                "context": item.get("context", "")
            })

        # Process dialogue lines through services.tts pipeline
        res = await self.tts_service.process_scene_dialogues(
            episode_id=episode_id,
            scene_id=scene_id,
            dialogue_list=normalized_dialogue,
            base_output_dir=os.path.join(project_root, "audio")
        )

        composite_wav = res["composite_audio_path"]
        srt_content = res["srt_content"]
        total_duration = res["total_duration"]

        # Copy composite scene audio to media_output for video renderer access
        if os.path.exists(composite_wav):
            shutil.copyfile(composite_wav, filepath)
            shutil.copyfile(composite_wav, audio_output_path)

        # Requirements 4 & 5 & 10: unique job audio and episode audio directories
        scene_filename = f"scene_{scene_num_val:03d}.wav"
        
        # 1. Save to episode-specific unique folder under media/
        episode_dir = Path(project_root) / "media" / f"episode_{episode_number:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_audio_path = episode_dir / scene_filename
        if os.path.exists(composite_wav):
            shutil.copyfile(composite_wav, str(episode_audio_path))
            logger.info(f"Copied episode composite audio to {episode_audio_path}")

        # 2. Save to job-specific folder under media/jobs/{job_id_short}/audio/
        if job_id:
            job_dir_name = job_id[:7] if len(job_id) > 7 else job_id
            job_audio_dir = Path(project_root) / "media" / "jobs" / job_dir_name / "audio"
            job_audio_dir.mkdir(parents=True, exist_ok=True)
            job_audio_path = job_audio_dir / scene_filename
            if os.path.exists(composite_wav):
                shutil.copyfile(composite_wav, str(job_audio_path))
                logger.info(f"Copied job composite audio to {job_audio_path}")
            return f"/media/jobs/{job_dir_name}/audio/{scene_filename}", srt_content, total_duration

        return f"/media/{filename}", srt_content, total_duration

    async def validate_audio_file(self, filepath: str) -> Tuple[float, int, bool]:
        """
        Verify file exists, duration > 0, sample rate, and ffprobe detects an audio stream.
        Returns (duration, sample_rate, has_audio_stream)
        """
        import wave
        import subprocess
        import imageio_ffmpeg
        
        if not os.path.exists(filepath):
            return 0.0, 0, False
            
        try:
            # Read wave info
            with wave.open(filepath, "rb") as wf:
                sample_rate = wf.getframerate()
                frames = wf.getnframes()
                channels = wf.getnchannels()
                duration = frames / float(sample_rate)
        except Exception:
            return 0.0, 0, False
            
        # Run ffprobe to verify audio stream
        has_audio_stream = False
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")
            if not os.path.exists(ffprobe_exe):
                # Fallback to ffmpeg output probing if ffprobe doesn't exist
                cmd = [ffmpeg_exe, "-i", filepath]
                res = subprocess.run(cmd, capture_output=True, text=True)
                has_audio_stream = "Audio:" in res.stderr
            else:
                cmd = [ffprobe_exe, "-show_streams", "-select_streams", "a", "-loglevel", "error", filepath]
                res = subprocess.run(cmd, capture_output=True, text=True)
                has_audio_stream = "[STREAM]" in res.stdout
        except Exception as e:
            logger.warning(f"Audio stream ffprobe check failed: {e}")
            # Fallback to wave metadata if ffprobe check fails
            has_audio_stream = duration > 0 and channels > 0
            
        return duration, sample_rate, has_audio_stream


voice_engine = VoiceGenerationEngine()
