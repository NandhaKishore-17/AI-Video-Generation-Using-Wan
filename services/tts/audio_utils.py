import os
import wave
import math
import struct
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger("audio_utils")

TARGET_SAMPLE_RATE = 24000  # 24 kHz minimum requirement
CHANNELS = 1                # Mono audio for clean voice track
SAMPLE_WIDTH = 2            # 16-bit PCM WAV (2 bytes per sample)


class AudioUtils:
    """
    Audio processing utilities for voice generation, punctuation pause insertion,
    volume normalization, sample rate conversion to 24kHz+, WAV formatting,
    and WebVTT/SRT subtitle generation.
    """

    def resample_pcm(self, pcm_data: bytearray, src_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> Tuple[bytearray, int]:
        """
        Resamples PCM 16-bit audio to target sample rate (e.g. 22050Hz -> 24000Hz) using linear interpolation.
        Returns (resampled_pcm_data, target_rate).
        """
        if src_rate == target_rate or not pcm_data:
            return pcm_data, src_rate

        num_src_samples = len(pcm_data) // SAMPLE_WIDTH
        if num_src_samples == 0:
            return pcm_data, target_rate

        try:
            samples = list(struct.unpack(f"<{num_src_samples}h", pcm_data))
        except Exception:
            return pcm_data, src_rate

        ratio = float(src_rate) / float(target_rate)
        num_target_samples = int(num_src_samples / ratio)
        resampled = []

        for i in range(num_target_samples):
            src_index = i * ratio
            idx0 = int(src_index)
            idx1 = min(idx0 + 1, num_src_samples - 1)
            frac = src_index - idx0

            val0 = samples[idx0]
            val1 = samples[idx1]
            interpolated = val0 + frac * (val1 - val0)
            resampled.append(int(max(-32768, min(32767, interpolated))))

        resampled_pcm = bytearray(struct.pack(f"<{len(resampled)}h", *resampled))
        return resampled_pcm, target_rate

    def create_silence_pcm(self, duration_seconds: float, sample_rate: int = TARGET_SAMPLE_RATE) -> bytearray:
        """Generates silent PCM audio frames for natural micro-pauses."""
        num_samples = int(sample_rate * duration_seconds)
        return bytearray(b'\x00' * (num_samples * SAMPLE_WIDTH * CHANNELS))

    def normalize_pcm_volume(self, pcm_data: bytearray, target_peak_ratio: float = 0.90) -> bytearray:
        """
        Normalizes peak PCM audio volume to prevent clipping and ensure consistent dialogue volume.
        """
        if not pcm_data:
            return pcm_data

        num_samples = len(pcm_data) // SAMPLE_WIDTH
        if num_samples == 0:
            return pcm_data

        fmt = f"<{num_samples}h"
        try:
            samples = list(struct.unpack(fmt, pcm_data))
        except Exception:
            return pcm_data

        max_sample = max(abs(s) for s in samples)
        if max_sample == 0 or max_sample >= 32767 * target_peak_ratio:
            return pcm_data

        gain = (32767.0 * target_peak_ratio) / float(max_sample)
        gain = min(gain, 3.5)

        normalized_samples = [int(max(-32768, min(32767, s * gain))) for s in samples]
        return bytearray(struct.pack(fmt, *normalized_samples))

    def insert_punctuation_pauses(self, pcm_data: bytearray, text: str, sample_rate: int = TARGET_SAMPLE_RATE) -> bytearray:
        """
        Appends natural breath/pause tail silence based on ending punctuation.
        """
        text_strip = text.strip()
        pause_duration = 0.20

        if text_strip.endswith((".", "!", "?")):
            pause_duration = 0.40
        elif text_strip.endswith((",", ";", "-", "—")):
            pause_duration = 0.22
        elif text_strip.endswith("..."):
            pause_duration = 0.50

        tail_silence = self.create_silence_pcm(pause_duration, sample_rate)
        combined = bytearray(pcm_data)
        combined.extend(tail_silence)
        return combined

    def write_wav_file(self, filepath: str, pcm_frames: bytearray, sample_rate: int = TARGET_SAMPLE_RATE):
        """Writes raw PCM bytearray to standard 24kHz 16-bit WAV file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_frames)

    def read_wav_info(self, filepath: str) -> Tuple[int, float, bytearray]:
        """Reads WAV file and returns (sample_rate, duration_seconds, pcm_data)."""
        with wave.open(filepath, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            dur = max(0.5, float(n_frames) / float(sample_rate))
            pcm_data = bytearray(wf.readframes(n_frames))
            return sample_rate, dur, pcm_data

    def concatenate_wav_files(self, input_files: List[str], output_filepath: str) -> float:
        """
        Concatenates multiple dialogue WAV files into a single 24kHz master audio track.
        Enforces minimum 24kHz sample rate on output.
        """
        combined_pcm = bytearray()
        target_sr = TARGET_SAMPLE_RATE

        for idx, file_path in enumerate(input_files):
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                try:
                    sr, dur, pcm = self.read_wav_info(file_path)
                    if sr < TARGET_SAMPLE_RATE:
                        pcm, sr = self.resample_pcm(pcm, sr, TARGET_SAMPLE_RATE)
                    target_sr = sr
                    combined_pcm.extend(pcm)

                    if idx < len(input_files) - 1:
                        gap = self.create_silence_pcm(0.30, target_sr)
                        combined_pcm.extend(gap)
                except Exception as e:
                    logger.warning(f"Error reading WAV file {file_path}: {e}")

        if not combined_pcm:
            combined_pcm = self.create_silence_pcm(3.0, target_sr)

        normalized_pcm = self.normalize_pcm_volume(combined_pcm)
        self.write_wav_file(output_filepath, normalized_pcm, target_sr)

        total_samples = len(normalized_pcm) // (SAMPLE_WIDTH * CHANNELS)
        return float(total_samples) / float(target_sr)

    def generate_srt_subtitles(self, timed_dialogues: List[Dict[str, Any]]) -> str:
        """Converts timed dialogue list to WebVTT / SRT subtitle format."""
        srt_lines = []
        for idx, item in enumerate(timed_dialogues, 1):
            start_str = self._format_timestamp(item["start"])
            end_str = self._format_timestamp(item["end"])
            speaker = item.get("speaker", "NARRATOR").upper()
            line = item.get("line", "")

            srt_lines.append(f"{idx}\n{start_str} --> {end_str}\n{speaker}: {line}\n")

        return "\n".join(srt_lines)

    def _format_timestamp(self, seconds: float) -> str:
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


audio_utils = AudioUtils()
