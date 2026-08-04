import logging
import wave
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings

logger = logging.getLogger("music_engine")


class MusicGenerationEngine:
    """Generate scene-adaptive background music and ambience files."""

    def __init__(self):
        self.provider = settings.MUSIC_PROVIDER
        self.media_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.mood_map = {
            "adventure": "Adventure",
            "fantasy": "Fantasy",
            "sad": "Sad",
            "horror": "Horror",
            "action": "Action",
            "mystery": "Mystery",
            "peaceful": "Peaceful",
        }

    async def generate_score_and_sfx(
        self,
        scene_id: str,
        genre: str,
        mood: str = "Tense / Mysterious",
        duration_seconds: float = 6.0,
    ) -> str:
        filename = f"music_{scene_id}.wav"
        filepath = self.media_dir / filename
        self._generate_synthetic_cinematic_score(filepath, mood, duration_seconds)
        return f"/media/{filename}"

    def generate_music(self, scene: Dict[str, Any], duration_seconds: float = 6.0) -> Dict[str, Any]:
        emotion = (scene.get("emotion") or "neutral").lower()
        mood = self.mood_map.get(emotion, "Fantasy")
        prompt = f"{mood} cinematic background music, ambient textures, emotional, high quality"
        path = self.media_dir / f"music_{scene.get('scene_number', 1):02d}.wav"
        self._generate_synthetic_cinematic_score(path, mood, duration_seconds)
        return {"prompt": prompt, "path": str(path), "mood": mood}

    def _generate_synthetic_cinematic_score(self, filepath: Path | str, mood: str, duration: float):
        sample_rate = 22050
        num_samples = int(sample_rate * duration)
        wav_file = wave.open(str(filepath), "wb")
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00" * num_samples * 4)
        wav_file.close()


music_engine = MusicGenerationEngine()
