import os
import wave
import math
import struct
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

from services.tts.voice_manager import voice_manager, VoiceManager
from services.tts.emotion_mapper import emotion_mapper, EmotionMapper
from services.tts.audio_utils import audio_utils, AudioUtils, TARGET_SAMPLE_RATE
from services.tts.base import BaseTTSBackend

logger = logging.getLogger("tts_service")


class EdgeTTSBackend(BaseTTSBackend):
    """
    High-quality open-source EdgeTTS backend (Microsoft Azure Neural voices).
    Generates 24kHz+ human-like natural speech with realistic breathing and dynamic prosody.
    """

    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_filepath: str,
        pitch: str = "+0Hz",
        rate: str = "+0%",
        volume: str = "+0%",
        ssml_style: str = "chat"
    ) -> bool:
        try:
            import edge_tts
            if not voice_id or not str(voice_id).strip():
                raise ValueError(f"VOICE LOST BEFORE EDGE TTS")
            
            clean_voice = voice_id.strip()

            
            logger.info(f"Generating EdgeTTS speech with model/voice: '{clean_voice}'")
            communicate = edge_tts.Communicate(
                text=text,
                voice=clean_voice,
                pitch=pitch,
                rate=rate,
                volume=volume
            )
            temp_mp3 = output_filepath.replace(".wav", ".mp3")
            await asyncio.wait_for(communicate.save(temp_mp3), timeout=15.0)

            # Convert generated MP3 to 24kHz WAV PCM
            success = self._convert_mp3_to_wav(temp_mp3, output_filepath)
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)

            return success
        except Exception as e:
            logger.warning(f"EdgeTTS generation failed for voice '{voice_id}': {e}. Exception details: {e}")
            return False

    def _convert_mp3_to_wav(self, mp3_path: str, wav_path: str) -> bool:
        """Converts MP3 to 24kHz WAV using ffmpeg if available."""
        if not os.path.exists(mp3_path) or os.path.getsize(mp3_path) == 0:
            return False

        try:
            import subprocess
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-i", mp3_path,
                "-ar", str(TARGET_SAMPLE_RATE),
                "-ac", "1",
                "-c:a", "pcm_s16le",
                wav_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                return True
        except Exception as e:
            logger.debug(f"FFmpeg MP3 to WAV conversion fallback: {e}")

        return False


class PyTTSx3Backend(BaseTTSBackend):
    """
    Offline system TTS backend powered by pyttsx3.
    """

    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_filepath: str,
        pitch: str = "+0Hz",
        rate: str = "+0%",
        volume: str = "+0%",
        ssml_style: str = "chat"
    ) -> bool:
        def _run_pyttsx3():
            try:
                import pyttsx3
                eng = pyttsx3.init()
                if voice_id:
                    try:
                        eng.setProperty('voice', voice_id)
                    except Exception:
                        pass
                
                base_rate = 160
                if "+" in rate:
                    base_rate += int(rate.replace("+", "").replace("%", "")) * 1.5
                elif "-" in rate:
                    base_rate -= int(rate.replace("-", "").replace("%", "")) * 1.5
                eng.setProperty('rate', max(100, min(250, int(base_rate))))

                eng.save_to_file(text, output_filepath)
                eng.runAndWait()
                return os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0
            except Exception as e:
                logger.warning(f"pyttsx3 backend error: {e}")
                return False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run_pyttsx3)


class HumanFallbackTTSBackend(BaseTTSBackend):
    """
    High-reliability natural human voice fallback generator.
    Ensures zero robotic audio generation by using high-definition neural speech synthesis.
    """

    def __init__(self):
        self.edge_backend = EdgeTTSBackend()

    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_filepath: str,
        pitch: str = "+0Hz",
        rate: str = "+0%",
        volume: str = "+0%",
        ssml_style: str = "chat"
    ) -> bool:
        clean_voice = voice_id

        return await self.edge_backend.generate_speech(
            text=text,
            voice_id=clean_voice,
            output_filepath=output_filepath,
            pitch=pitch,
            rate=rate,
            volume=volume,
            ssml_style=ssml_style
        )


class MockTTSBackend(HumanFallbackTTSBackend):
    """Alias to HumanFallbackTTSBackend maintaining backwards compatibility while removing robotic voices."""
    pass


class ElevenLabsBackend(BaseTTSBackend):
    """
    Hyper-realistic ElevenLabs cloud voice synthesis backend.
    Requires ELEVENLABS_API_KEY set in environment or config.
    """
    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_filepath: str,
        pitch: str = "+0Hz",
        rate: str = "+0%",
        volume: str = "+0%",
        ssml_style: str = "chat"
    ) -> bool:
        try:
            from app.core.config import settings
            api_key = os.environ.get("ELEVENLABS_API_KEY") or settings.ELEVENLABS_API_KEY
            if not api_key:
                logger.warning("ElevenLabs API Key not configured. Falling back...")
                return False

            import urllib.request
            import json
            eleven_voice_id = voice_id if len(voice_id) == 20 else "21m00Tcm4TlvDq8ikWAM"
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{eleven_voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            data = json.dumps({
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
            }).encode('utf-8')

            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            temp_mp3 = output_filepath.replace(".wav", ".mp3")
            with urllib.request.urlopen(req) as response, open(temp_mp3, 'wb') as out_f:
                out_f.write(response.read())

            edge_backend = EdgeTTSBackend()
            success = edge_backend._convert_mp3_to_wav(temp_mp3, output_filepath)
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            return success
        except Exception as e:
            logger.warning(f"ElevenLabs TTS generation failed: {e}")
            return False


class OpenAITTSBackend(BaseTTSBackend):
    """
    OpenAI TTS backend (tts-1 / tts-1-hd) with realistic natural voices: alloy, echo, fable, onyx, nova, shimmer.
    """
    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        output_filepath: str,
        pitch: str = "+0Hz",
        rate: str = "+0%",
        volume: str = "+0%",
        ssml_style: str = "chat"
    ) -> bool:
        try:
            from app.core.config import settings
            api_key = os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
            if not api_key:
                logger.warning("OpenAI API Key not configured. Falling back...")
                return False

            import urllib.request
            import json
            openai_voice = voice_id.lower() if voice_id.lower() in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"] else "alloy"
            url = "https://api.openai.com/v1/audio/speech"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = json.dumps({
                "model": "tts-1",
                "input": text,
                "voice": openai_voice,
                "response_format": "mp3"
            }).encode('utf-8')

            req = urllib.request.Request(url, data=data, headers=headers, method='POST')
            temp_mp3 = output_filepath.replace(".wav", ".mp3")
            with urllib.request.urlopen(req) as response, open(temp_mp3, 'wb') as out_f:
                out_f.write(response.read())

            edge_backend = EdgeTTSBackend()
            success = edge_backend._convert_mp3_to_wav(temp_mp3, output_filepath)
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            return success
        except Exception as e:
            logger.warning(f"OpenAI TTS generation failed: {e}")
            return False


class TTSService:
    """
    Unified Text-to-Speech Service coordinating VoiceManager, EmotionMapper,
    AudioUtils, and modular open-source F5-TTS / Edge-TTS backends.
    """

    def __init__(self, provider: str = "f5-tts"):
        self.provider = provider
        self.voice_mgr = voice_manager
        self.emotion_map = emotion_mapper
        self.utils = audio_utils

        # Import F5TTSBackend dynamically or from services.tts.f5_tts_engine
        from services.tts.f5_tts_engine import F5TTSBackend

        f5_backend = F5TTSBackend()
        edge_backend = EdgeTTSBackend()

        self.backends: Dict[str, BaseTTSBackend] = {
            "f5-tts": f5_backend,
            "f5tts": f5_backend,
            "edge-tts": edge_backend,
            "elevenlabs": ElevenLabsBackend(),
            "openai-tts": OpenAITTSBackend(),
            "piper": f5_backend,
            "kokoro": edge_backend,
            "pyttsx3": PyTTSx3Backend(),
            "mock": HumanFallbackTTSBackend()
        }
        self.fallback_backend = PyTTSx3Backend()
        self.mock_backend = HumanFallbackTTSBackend()

    def get_backend(self) -> BaseTTSBackend:
        """Returns configured active TTS backend or fallback."""
        return self.backends.get(self.provider.lower(), self.backends.get("f5-tts", self.backends["edge-tts"]))

    async def generate_single_dialogue(
        self,
        character_name: str,
        dialogue_text: str,
        output_filepath: str,
        scene_context: str = "",
        language: str = "en",
        force_voice_id: str = None,
        character_id: str = None
    ) -> Dict[str, Any]:
        """
        Generates expressive spoken audio file for a single character dialogue line.
        """
        clean_text = self.emotion_map.clean_dialogue_text(dialogue_text)
        if not clean_text:
            clean_text = dialogue_text

        # 1. Get consistent voice assignment for character
        voice_info = self.voice_mgr.get_or_assign_voice(character_name, language=language)
        
        # Determine fallback mapping for known internal voice presets if used
        FALLBACK_VOICE_MAP = {
            "Piper-Male-Cinematic-1": "en-US-ChristopherNeural",
            "Idris-Old-Man-Royal": "en-GB-RyanNeural"
        }
        
        raw_voice_id = force_voice_id.strip() if force_voice_id and force_voice_id.strip() else None
        
        if raw_voice_id:
            if raw_voice_id in FALLBACK_VOICE_MAP:
                final_voice_id = FALLBACK_VOICE_MAP[raw_voice_id]
                logger.info(f"[TTS] Mapped custom preset '{raw_voice_id}' to '{final_voice_id}'")
            elif "Neural" not in raw_voice_id and "en-" not in raw_voice_id:
                # If it's a completely unrecognized non-Edge voice, fallback to voice_mgr
                final_voice_id = voice_info.get("voice_id")
                logger.warning(f"[TTS] Unrecognized EdgeTTS voice '{raw_voice_id}'. Falling back to '{final_voice_id}'")
            else:
                final_voice_id = raw_voice_id
        else:
            final_voice_id = voice_info.get("voice_id")

        if not final_voice_id or not str(final_voice_id).strip():
            raise ValueError(
                f"VOICE RESOLUTION FAILED: "
                f"speaker={repr(character_name)}, "
                f"character_id={repr(character_id)}, "
                f"character_name={repr(character_name)}"
            )
            
        print("\n" + "=" * 100)
        print("FINAL TTS REQUEST")
        print("speaker:", repr(character_name))
        print("character_id:", repr(character_id))
        print("character_name:", repr(character_name))
        print("resolved_voice:", repr(final_voice_id))
        print("text:", repr(dialogue_text[:150]))
        print("=" * 100)
        
        logger.info(f"[TTS] Generating dialogue | speaker='{character_name}' | character_id='{character_id}' | resolved_voice='{final_voice_id}'")

        # 2. Infer emotion and prosody adjustments from dialogue
        emotion_name, prosody = self.emotion_map.infer_emotion(dialogue_text, scene_context=scene_context)

        final_pitch = prosody["pitch"] if prosody["pitch"] != "+0Hz" else voice_info["pitch"]
        final_rate = prosody["rate"] if prosody["rate"] != "+0%" else voice_info["rate"]

        # 3. Generate speech via active backend
        backend = self.get_backend()
        success = await backend.generate_speech(
            text=clean_text,
            voice_id=final_voice_id,
            output_filepath=output_filepath,
            pitch=final_pitch,
            rate=final_rate,
            volume=prosody["volume"],
            ssml_style=prosody["ssml_style"]
        )

        # Fall back if primary backend fails
        if not success:
            logger.info(f"Primary backend '{self.provider}' failed. Trying pyttsx3 fallback...")
            success = await self.fallback_backend.generate_speech(
                text=clean_text,
                voice_id=final_voice_id,
                output_filepath=output_filepath,
                rate=final_rate
            )

        if not success:
            logger.info("pyttsx3 fallback failed. Using synthetic mock voice generator...")
            await self.mock_backend.generate_speech(
                text=clean_text,
                voice_id=final_voice_id,
                output_filepath=output_filepath
            )

        # Post-process WAV file to ensure 24kHz+ sample rate & natural punctuation micro-pauses
        if os.path.exists(output_filepath):
            try:
                sr, dur, pcm = self.utils.read_wav_info(output_filepath)
                if sr < TARGET_SAMPLE_RATE:
                    pcm, sr = self.utils.resample_pcm(pcm, sr, TARGET_SAMPLE_RATE)
                enhanced_pcm = self.utils.insert_punctuation_pauses(pcm, clean_text, sr)
                self.utils.write_wav_file(output_filepath, enhanced_pcm, sr)
                sr, dur, _ = self.utils.read_wav_info(output_filepath)
            except Exception as e:
                logger.warning(f"Error post-processing WAV audio pauses: {e}")
                dur = 3.0
        else:
            dur = 3.0

        return {
            "character": character_name,
            "line": clean_text,
            "voice_name": voice_info["name"],
            "archetype": voice_info["archetype"],
            "emotion": emotion_name,
            "filepath": output_filepath,
            "duration": dur
        }

    async def process_scene_dialogues(
        self,
        episode_id: str,
        scene_id: str,
        dialogue_list: List[Dict[str, Any]],
        base_output_dir: str = "./audio"
    ) -> Dict[str, Any]:
        """
        Processes screenplay dialogue list for a scene, generates speech WAV files in structured hierarchy:
        `audio/episode_{episode_id}/scene_{scene_id}/{speaker_slug}_{index}.wav`,
        concatenates full scene audio, and builds SRT subtitles.
        """
        ep_clean = str(episode_id).replace("episode_", "").zfill(3)
        sc_clean = str(scene_id).replace("scene_", "").zfill(2)

        scene_dir = os.path.join(base_output_dir, f"episode_{ep_clean}", f"scene_{sc_clean}")
        os.makedirs(scene_dir, exist_ok=True)

        generated_dialogues = []
        dialogue_file_paths = []
        timed_dialogues = []
        current_time = 0.5

        for idx, item in enumerate(dialogue_list, 1):
            speaker = str(item.get("speaker") or "Narrator")
            line = str(item.get("line") or "").strip()

            if not line:
                continue

            speaker_slug = speaker.lower().replace(" ", "_")
            char_id_str = item.get("character_id", "no_id")[:8] if item.get("character_id") else "no_id"
            file_name = f"{char_id_str}_{sc_clean}_{speaker_slug}_{idx:03d}.wav"
            line_filepath = os.path.join(scene_dir, file_name)

            info = await self.generate_single_dialogue(
                character_name=speaker,
                dialogue_text=line,
                output_filepath=line_filepath,
                scene_context=item.get("context", ""),
                force_voice_id=item.get("voice_id"),
                character_id=item.get("character_id")
            )

            # Stage 2 & 3: Audio validation for individual dialogue WAV file
            try:
                if not os.path.exists(line_filepath):
                    raise FileNotFoundError(f"Voice generation failed: Audio file not found at {line_filepath}")
                
                file_size = os.path.getsize(line_filepath)
                if file_size < 1024:
                    raise ValueError(f"WAV file validation failed: {line_filepath} size is {file_size} bytes (under 1 KB)")

                try:
                    with wave.open(line_filepath, "rb") as wf:
                        sr_val = wf.getframerate()
                        frames = wf.getnframes()
                        channels = wf.getnchannels()
                        dur_val = frames / float(sr_val)
                except Exception as e:
                    raise ValueError(f"WAV file validation failed: Failed to read wave metadata from {line_filepath}: {e}")

                if dur_val <= 0:
                    raise ValueError(f"WAV file validation failed: {line_filepath} has duration {dur_val}s (must be > 0)")
                
                has_stream = False
                try:
                    import imageio_ffmpeg
                    import subprocess
                    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                    ffprobe_exe = ffmpeg_exe.replace("ffmpeg", "ffprobe")
                    if not os.path.exists(ffprobe_exe):
                        cmd = [ffmpeg_exe, "-i", line_filepath]
                        res = subprocess.run(cmd, capture_output=True, text=True)
                        has_stream = "Audio:" in res.stderr
                    else:
                        cmd = [ffprobe_exe, "-show_streams", "-select_streams", "a", "-loglevel", "error", line_filepath]
                        res = subprocess.run(cmd, capture_output=True, text=True)
                        has_stream = "[STREAM]" in res.stdout
                except Exception as e:
                    logger.warning(f"Audio stream ffprobe check failed: {e}")
                    has_stream = dur_val > 0 and channels > 0
                
                if not has_stream:
                    raise ValueError(f"WAV file validation failed: No valid audio stream detected by ffprobe in {line_filepath}")
            except Exception as e:
                logger.warning(f"Audio generation completely failed for dialogue {idx}: {e}")
                continue # Skip this dialogue line instead of crashing

            # Log stage 2 & 3 details
            logger.info(
                "=== VOICE GENERATION REPORT (DIALOGUE LINE) ===\n"
                f"Dialogue: {line}\n"
                f"Speaker: {speaker}\n"
                f"Wav Path: {line_filepath}\n"
                f"File Size: {file_size} bytes\n"
                f"Duration: {dur_val:.2f}s\n"
                f"Sample Rate: {sr_val} Hz\n"
                "=============================================="
            )

            start_t = current_time
            end_t = start_t + info["duration"]
            current_time = end_t + 0.30

            timed_dialogues.append({
                "speaker": speaker,
                "line": line,
                "start": start_t,
                "end": end_t
            })

            generated_dialogues.append({
                "speaker": speaker,
                "file_path": line_filepath,
                "relative_path": os.path.relpath(line_filepath, base_output_dir),
                "duration": info["duration"],
                "emotion": info["emotion"],
                "voice": info["voice_name"]
            })
            dialogue_file_paths.append(line_filepath)

        # Build full scene composite audio file
        composite_filename = f"audio_scene_{sc_clean}.wav"
        composite_path = os.path.join(scene_dir, composite_filename)
        if not dialogue_file_paths:
            logger.warning(f"No valid audio generated for scene {sc_clean}. Skipping audio compilation.")
            composite_path = None
            total_duration = 6.0
        else:
            total_duration = self.utils.concatenate_wav_files(dialogue_file_paths, composite_path)

        srt_content = self.utils.generate_srt_subtitles(timed_dialogues)

        return {
            "episode_id": f"episode_{ep_clean}",
            "scene_id": f"scene_{sc_clean}",
            "composite_audio_path": composite_path,
            "dialogue_files": generated_dialogues,
            "timed_dialogues": timed_dialogues,
            "srt_content": srt_content,
            "total_duration": max(6.0, total_duration)
        }


tts_service = TTSService(provider="edge-tts")
