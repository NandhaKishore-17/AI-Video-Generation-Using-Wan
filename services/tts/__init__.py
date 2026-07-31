"""
Text-to-Speech Service Package.
Provides natural human voice synthesis, character voice library management,
emotion inference, audio quality post-processing, and modular TTS backends.
"""

from services.tts.voice_manager import VoiceManager, voice_manager
from services.tts.emotion_mapper import EmotionMapper, emotion_mapper
from services.tts.audio_utils import AudioUtils, audio_utils
from services.tts.f5_tts_engine import F5TTSBackend
from services.tts.tts_service import TTSService, tts_service, BaseTTSBackend, EdgeTTSBackend, PyTTSx3Backend, HumanFallbackTTSBackend, MockTTSBackend

__all__ = [
    "VoiceManager",
    "voice_manager",
    "EmotionMapper",
    "emotion_mapper",
    "AudioUtils",
    "audio_utils",
    "TTSService",
    "tts_service",
    "BaseTTSBackend",
    "F5TTSBackend",
    "EdgeTTSBackend",
    "PyTTSx3Backend",
    "HumanFallbackTTSBackend",
    "MockTTSBackend",
]
