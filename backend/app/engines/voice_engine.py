import os
import sys
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
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
        episode_id: str = "001"
    ) -> Tuple[str, str, float]:
        """
        Generates full spoken dialogue audio track for a scene and returns (audio_relative_url, srt_content, total_duration_seconds).
        Saves dialogues to: audio/episode_{episode_id}/scene_{scene_id}/{speaker}_{idx}.wav
        """
        filename = f"audio_{scene_id}.wav"
        filepath = str(self.media_dir / filename)
        audio_output_path = str(self.audio_dir / filename)

        # Process dialogue lines through services.tts pipeline
        res = await self.tts_service.process_scene_dialogues(
            episode_id=episode_id,
            scene_id=scene_id,
            dialogue_list=dialogue_list,
            base_output_dir=os.path.join(project_root, "audio")
        )

        composite_wav = res["composite_audio_path"]
        srt_content = res["srt_content"]
        total_duration = res["total_duration"]

        # Copy composite scene audio to media_output for video renderer access
        if os.path.exists(composite_wav):
            shutil.copyfile(composite_wav, filepath)
            shutil.copyfile(composite_wav, audio_output_path)

        import time
        timestamp = int(time.time())
        return f"/media/{filename}?t={timestamp}", srt_content, total_duration


voice_engine = VoiceGenerationEngine()
