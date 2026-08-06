import logging
import wave
from pathlib import Path
from typing import Any, Dict, Optional

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

    def generate_music(self, scene: Dict[str, Any], duration_seconds: float = 6.0, job_id: Optional[str] = None) -> Dict[str, Any]:
        from typing import Optional
        raw_emotion = scene.get("emotion") or "neutral"
        if isinstance(raw_emotion, list) and raw_emotion:
            raw_emotion = raw_emotion[0]
        emotion = str(raw_emotion).lower()
        mood = self.mood_map.get(emotion, "Fantasy")
        prompt = f"{mood} cinematic background music, ambient textures, emotional, high quality"
        
        scene_num = scene.get('scene_number', 1)
        if job_id:
            from app.core.config import BASE_DIR
            job_dir_name = job_id[:7] if len(job_id) > 7 else job_id
            job_audio_dir = Path(BASE_DIR) / "media" / "jobs" / job_dir_name / "audio"
            job_audio_dir.mkdir(parents=True, exist_ok=True)
            path = job_audio_dir / f"music_scene_{scene_num:03d}.wav"
        else:
            path = self.media_dir / f"music_{scene_num:02d}.wav"
            
        self._generate_synthetic_cinematic_score(path, mood, duration_seconds)
        return {"prompt": prompt, "path": str(path), "mood": mood}

    def _generate_synthetic_cinematic_score(self, filepath: Path | str, mood: str, duration: float):
        import math
        import struct
        sample_rate = 22050
        num_samples = int(sample_rate * duration)

        # Mood-to-frequency mapping: root note + harmonic overtones
        mood_freqs = {
            "Adventure": (220.0, 330.0, 440.0),
            "Fantasy":   (196.0, 294.0, 392.0),
            "Sad":       (174.6, 261.6, 349.2),
            "Horror":    (130.8, 185.0, 277.2),
            "Action":    (246.9, 370.0, 493.9),
            "Mystery":   (164.8, 220.0, 329.6),
            "Peaceful":  (261.6, 392.0, 523.3),
        }
        freqs = mood_freqs.get(mood, (220.0, 330.0, 440.0))

        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # Fade in over first 0.3 s, fade out over last 0.5 s
            env = 1.0
            if t < 0.3:
                env = t / 0.3
            elif t > duration - 0.5:
                env = (duration - t) / 0.5
            env = max(0.0, min(1.0, env))

            # Layered sine oscillators
            sample = (
                0.40 * math.sin(2 * math.pi * freqs[0] * t) +          # root
                0.25 * math.sin(2 * math.pi * freqs[1] * t) +          # fifth
                0.15 * math.sin(2 * math.pi * freqs[2] * t) +          # octave
                0.10 * math.sin(2 * math.pi * freqs[0] * 0.5 * t) +   # sub-bass
                0.10 * math.sin(2 * math.pi * freqs[1] * 2.0 * t + math.sin(2 * math.pi * 0.5 * t))  # pad sweep
            ) * env * 0.7

            # Clamp and convert to 16-bit little-endian stereo
            val = int(max(-32767, min(32767, sample * 32767)))
            packed = struct.pack("<hh", val, val)  # L + R
            frames.extend(packed)

        wav_file = wave.open(str(filepath), "wb")
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))
        wav_file.close()


music_engine = MusicGenerationEngine()
