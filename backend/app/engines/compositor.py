import logging
import subprocess
import os
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_ffmpeg_exe() -> str:
    """Locate ffmpeg via imageio_ffmpeg or fall back to system PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


class VideoCompositor:
    """Compose a final MP4 from scene clips, voice, music, and subtitles using ffmpeg."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or settings.MEDIA_OUTPUT_DIR).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(settings.TEMP_DIR).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_path(p: str) -> Optional[str]:
        """Resolve URL path or relative path to absolute local filesystem path."""
        if not p:
            return None
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return os.path.abspath(p)
        clean_p = p.lstrip("/").replace("media/", "", 1) if p.startswith("/media/") else p.lstrip("/")
        candidate = Path(settings.MEDIA_OUTPUT_DIR) / os.path.basename(clean_p)
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate.resolve())
        candidate2 = Path(settings.MEDIA_OUTPUT_DIR).parent / clean_p
        if candidate2.exists() and candidate2.stat().st_size > 0:
            return str(candidate2.resolve())
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose(
        self,
        clips: List[str],
        voice_files: List[str],
        music_files: List[str],
        subtitle_files: List[str],
        output_name: str = "final_video.mp4",
    ) -> Dict[str, Any]:
        output_path = self.output_dir / output_name
        resolved_clips = [self._resolve_path(c) for c in clips if self._resolve_path(c)]
        if not resolved_clips:
            raise ValueError("At least one valid clip is required")

        ffmpeg = _get_ffmpeg_exe()

        # Step 1 – concatenate video clips into a single silent video
        silent_video = str(self.temp_dir / f"_silent_{output_name}")
        self._concat_video_clips(ffmpeg, resolved_clips, silent_video)

        # Step 2 – resolve and combine voice, music, and subtitle files
        valid_voice_paths = [self._resolve_path(v) for v in voice_files if self._resolve_path(v)]
        valid_music_paths = [self._resolve_path(m) for m in music_files if self._resolve_path(m)]
        valid_sub_paths = [self._resolve_path(s) for s in subtitle_files if self._resolve_path(s)]

        combined_voice_wav = self._combine_audio_files(ffmpeg, valid_voice_paths, str(self.temp_dir / f"_voice_{output_name}.wav"))
        combined_music_wav = self._combine_audio_files(ffmpeg, valid_music_paths, str(self.temp_dir / f"_music_{output_name}.wav"))
        combined_srt = self._combine_srt_files(valid_sub_paths, str(self.temp_dir / f"_subtitles_{output_name}.srt"))

        # Step 3 – mux audio into video
        if combined_voice_wav or combined_music_wav:
            self._mux_audio(ffmpeg, silent_video, combined_voice_wav, combined_music_wav, combined_srt, str(output_path))
        else:
            logger.warning("No valid audio files found; output will have no audio track.")
            import shutil
            shutil.copy2(silent_video, str(output_path))

        # Clean up temp file
        try:
            if os.path.exists(silent_video):
                os.remove(silent_video)
        except Exception:
            pass

        return {
            "output_path": str(output_path),
            "clips": resolved_clips,
            "voice_files": valid_voice_paths,
            "music_files": valid_music_paths,
            "subtitle_files": valid_sub_paths,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _combine_audio_files(self, ffmpeg: str, paths: List[str], output_path: str) -> Optional[str]:
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0]
        
        # Concat multiple audio files via FFmpeg concat filter
        cmd = [ffmpeg, "-y"]
        for p in paths:
            cmd.extend(["-i", p])
        filter_str = "".join([f"[{i}:a]" for i in range(len(paths))]) + f"concat=n={len(paths)}:v=0:a=1[aout]"
        cmd.extend([
            "-filter_complex", filter_str,
            "-map", "[aout]",
            "-c:a", "pcm_s16le",
            output_path
        ])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        return paths[0]

    def _combine_srt_files(self, paths: List[str], output_path: str) -> Optional[str]:
        if not paths:
            return None
        combined_lines = []
        for p in paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        combined_lines.append(content)
            except Exception as e:
                logger.warning(f"Error reading SRT file {p}: {e}")
        if not combined_lines:
            return None
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(combined_lines))
        return output_path if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else None

    def _concat_video_clips(self, ffmpeg: str, clips: List[str], output: str) -> None:
        """Concatenate video clips; no audio stream is required at this stage."""
        if len(clips) == 1:
            cmd = [
                ffmpeg, "-y",
                "-i", clips[0],
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-an",
                output,
            ]
        else:
            cmd = [ffmpeg, "-y"]
            for clip in clips:
                cmd.extend(["-i", clip])
            filter_complex = self._build_video_filter(len(clips))
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output,
            ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg concat failed: %s", result.stderr)
            raise RuntimeError(result.stderr or "ffmpeg concat failed")

    def _mux_audio(
        self,
        ffmpeg: str,
        video_path: str,
        voice_wav: Optional[str],
        music_wav: Optional[str],
        subtitle_srt: Optional[str],
        output: str,
    ) -> None:
        """
        Mux audio and burn subtitles into video using FFmpeg.
        - If both voice and music exist: mix them (voice at full volume, music at attenuated volume).
        - If only one exists: use it directly.
        - If subtitle_srt exists: burn subtitles using vf subtitles filter.
        """
        cmd = [ffmpeg, "-y", "-i", video_path]

        if voice_wav and music_wav:
            cmd.extend(["-i", voice_wav, "-i", music_wav])
            cmd.extend([
                "-filter_complex",
                "[1:a]aformat=sample_rates=44100:channel_layouts=stereo[v];[2:a]volume=0.25,aformat=sample_rates=44100:channel_layouts=stereo[m];[v][m]amix=inputs=2:duration=longest[aout]",
                "-map", "0:v:0",
                "-map", "[aout]",
            ])
        elif voice_wav:
            cmd.extend(["-i", voice_wav])
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
        else:
            cmd.extend(["-i", music_wav])
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])

        if subtitle_srt and os.path.exists(subtitle_srt):
            # Escape path for FFmpeg subtitles filter on Windows
            clean_srt = subtitle_srt.replace("\\", "/").replace(":", "\\:")
            cmd.extend(["-vf", f"subtitles='{clean_srt}'"])

        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output,
        ])

        logger.info("=" * 80)
        logger.info("EXACT FFMPEG COMPOSITION COMMAND:")
        logger.info(" ".join(cmd))
        logger.info("=" * 80)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("ffmpeg audio mux failed: %s", result.stderr)
            # Fallback without subtitle filter if subtitle burning fails
            if subtitle_srt and "-vf" in cmd:
                logger.warning("Retrying FFmpeg muxing without subtitle burn-in...")
                cmd_nosub = [c for idx, c in enumerate(cmd) if c != "-vf" and (idx == 0 or cmd[idx-1] != "-vf")]
                result2 = subprocess.run(cmd_nosub, capture_output=True, text=True)
                if result2.returncode == 0:
                    return
            raise RuntimeError(result.stderr or "ffmpeg audio mux failed")

    @staticmethod
    def _build_video_filter(clip_count: int) -> str:
        """Build a filter_complex string for concatenating N video-only streams."""
        parts = [f"[{i}:v]" for i in range(clip_count)]
        parts.append(f"concat=n={clip_count}:v=1:a=0[vout]")
        return "".join(parts)


compositor = VideoCompositor()
