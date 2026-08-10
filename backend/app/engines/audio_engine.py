import logging
from typing import Any, Dict, List, Optional, Tuple

from app.engines.voice_engine import voice_engine

logger = logging.getLogger(__name__)


class AudioEngine:
    """Thin wrapper around the voice pipeline so audio assets are emitted with fresh metadata."""

    def __init__(self):
        self.voice_engine = voice_engine

    async def generate_audio_assets(
        self,
        scene_id: str,
        dialogue_list: List[Dict[str, Any]],
        episode_id: str = "001",
        episode_number: int = 1,
        job_id: Optional[str] = None,
    ) -> Tuple[str, str, float]:
        return await self.voice_engine.generate_scene_audio_and_srt(
            scene_id=scene_id,
            dialogue_list=dialogue_list,
            episode_id=episode_id,
            episode_number=episode_number,
            job_id=job_id,
        )


audio_engine = AudioEngine()
