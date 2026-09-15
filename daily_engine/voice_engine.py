"""
voice_engine.py - Edge-TTS Tamil Voice Generation with Loudness Normalization.
Generates crystal-clear, broadcast-loud (-14 LUFS) female Tamil voiceover for Host Yaazhini.
"""

import asyncio
import os
import subprocess
from pathlib import Path
import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_VOICE = "ta-IN-PallaviNeural"  # Professional female Tamil voice


async def generate_tamil_voice_async(
    text: str,
    output_wav: Path,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    volume: str = "+0%",
) -> float:
    """
    Generate Tamil speech from text and normalize to broadcast standard (-14 LUFS).
    Returns exact duration in seconds.
    """
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    temp_mp3 = output_wav.with_suffix(".temp.mp3")

    # 1. Synthesize audio via Edge-TTS
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume)
    await communicate.save(str(temp_mp3))

    # 2. Convert to 16kHz mono WAV (compatible with SadTalker) + normalize loudness
    cmd = [
        FFMPEG, "-y",
        "-i", str(temp_mp3),
        "-ar", "16000",
        "-ac", "1",
        "-filter:a", "loudnorm=I=-14:LRA=7:TP=-1.5",
        "-c:a", "pcm_s16le",
        str(output_wav),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    if temp_mp3.exists():
        temp_mp3.unlink()

    # 3. Get exact duration via ffprobe / python wave
    import wave
    with wave.open(str(output_wav), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)

    return duration


def generate_tamil_voice(text: str, output_wav: Path, voice: str = DEFAULT_VOICE) -> float:
    """Synchronous wrapper for generate_tamil_voice_async."""
    return asyncio.run(generate_tamil_voice_async(text, output_wav, voice))


if __name__ == "__main__":
    test_text = "வணக்கம்! இன்னைக்கு நாம ஆயிரம் வருஷத்துக்கு முன்னாடி... ராஜேந்திர சோழனின் கடற்படைக்குள்ள வந்திருக்கோம்!"
    test_out = Path("D:/mvid/daily_engine/scratch/test_voice.wav")
    dur = generate_tamil_voice(test_text, test_out)
    print(f"Generated test voice: {test_out} (Duration: {dur:.2f}s)")
