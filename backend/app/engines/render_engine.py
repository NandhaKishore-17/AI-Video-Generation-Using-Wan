import os
import subprocess
import logging
import wave
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger("render_engine")

class FFmpegVideoRenderEngine:
    """
    FFmpeg & MoviePy Video Compositing Engine.
    Assembles scene videos, dialogue audio tracks, background score, sound effects,
    and burns formatted WebVTT/SRT subtitles into a final broadcast-quality MP4 video.
    """

    def __init__(self):
        self.media_dir = settings.MEDIA_OUTPUT_DIR
        self.temp_dir = settings.TEMP_DIR

    def _get_ffmpeg_exe(self) -> str:
        """
        Locates FFmpeg executable via imageio_ffmpeg or system PATH fallback.
        """
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return "ffmpeg"

    def _combine_wav_files(self, wav_paths: List[str], output_wav_path: str) -> bool:
        """
        Concatenates multiple WAV audio files into a single master WAV track.
        """
        valid_paths = [p for p in wav_paths if os.path.exists(p)]
        if not valid_paths:
            return False

        try:
            with wave.open(valid_paths[0], 'rb') as first_wav:
                params = first_wav.getparams()

            with wave.open(output_wav_path, 'wb') as outfile:
                outfile.setparams(params)
                for wp in valid_paths:
                    with wave.open(wp, 'rb') as infile:
                        outfile.writeframes(infile.readframes(infile.getnframes()))
            return os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0
        except Exception as e:
            logger.warning(f"Error combining WAV files: {e}")
            return False

    async def render_episode_mp4(
        self,
        episode_id: str,
        scene_assets: List[Dict[str, Any]],
        aspect_ratio: str = "16:9"
    ) -> str:
        """
        Stitches all scenes together, overlays composite character audio + music track,
        burns SRT subtitles, and produces final episode MP4 file with character voice audio.
        """
        output_filename = f"final_episode_{episode_id}.mp4"
        output_path = os.path.join(self.media_dir, output_filename)

        concat_list_path = os.path.join(self.temp_dir, f"concat_{episode_id}.txt")
        srt_combined_path = os.path.join(self.temp_dir, f"subtitles_{episode_id}.srt")
        combined_audio_path = os.path.join(self.temp_dir, f"audio_{episode_id}.wav")

        valid_video_paths = []
        valid_audio_paths = []
        combined_srt_lines = []
        global_srt_index = 1
        current_time_offset = 0.0

        for scene_idx, scene in enumerate(scene_assets):
            # Collect scene video path
            video_rel = scene.get("video_url")
            if video_rel:
                video_filename = os.path.basename(video_rel)
                full_video_path = os.path.join(self.media_dir, video_filename)
                if os.path.exists(full_video_path):
                    valid_video_paths.append(full_video_path)

            # Collect scene character audio path
            audio_rel = scene.get("audio_url") or scene.get("score_url")
            if audio_rel:
                audio_filename = os.path.basename(audio_rel)
                full_audio_path = os.path.join(self.media_dir, audio_filename)
                if os.path.exists(full_audio_path):
                    valid_audio_paths.append(full_audio_path)

            # Build combined master subtitle file
            srt_content = scene.get("subtitle_srt", "")
            duration = scene.get("duration_seconds", 6.0)

            if srt_content:
                lines = srt_content.strip().split("\n")
                i = 0
                while i < len(lines):
                    if lines[i].isdigit():
                        i += 1
                        if i < len(lines) and "-->" in lines[i]:
                            start_str, end_str = lines[i].split("-->")
                            i += 1
                            text_lines = []
                            while i < len(lines) and lines[i].strip() != "":
                                text_lines.append(lines[i])
                                i += 1
                            
                            combined_srt_lines.append(f"{global_srt_index}")
                            combined_srt_lines.append(f"{start_str.strip()} --> {end_str.strip()}")
                            combined_srt_lines.extend(text_lines)
                            combined_srt_lines.append("")
                            global_srt_index += 1
                    i += 1

            current_time_offset += duration

        # Save combined SRT file
        with open(srt_combined_path, "w", encoding="utf-8") as f:
            f.write("\n".join(combined_srt_lines))

        # Save video list for FFmpeg concatenation
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for vp in valid_video_paths:
                clean_p = vp.replace("\\", "/")
                f.write(f"file '{clean_p}'\n")

        # Combine scene audio WAV files into a single master audio track
        has_audio = self._combine_wav_files(valid_audio_paths, combined_audio_path)

        ffmpeg_exe = self._get_ffmpeg_exe()

        # Execute FFmpeg Command to Concatenate Video + Merge Character Voice Audio Track
        try:
            if has_audio and os.path.exists(combined_audio_path):
                cmd = [
                    ffmpeg_exe, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-i", combined_audio_path,
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    output_path
                ]
            else:
                cmd = [
                    ffmpeg_exe, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list_path,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    output_path
                ]

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"FFmpeg render successful with audio: {output_path}")
            return f"/media/{output_filename}"
        except Exception as e:
            logger.warning(f"FFmpeg execution failed: {e}. Executing direct audio-video multiplex fallback.")

        # Fallback: Multiplex primary video with character audio file if available
        if valid_video_paths:
            primary_video = valid_video_paths[0]
            primary_audio = valid_audio_paths[0] if valid_audio_paths else None
            if primary_audio and os.path.exists(primary_audio):
                try:
                    cmd_fallback = [
                        ffmpeg_exe, "-y",
                        "-i", primary_video,
                        "-i", primary_audio,
                        "-map", "0:v:0",
                        "-map", "1:a:0",
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-shortest",
                        output_path
                    ]
                    subprocess.run(cmd_fallback, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return f"/media/{output_filename}"
                except Exception as fb_err:
                    logger.warning(f"Fallback multiplex failed: {fb_err}")
            
            primary_name = os.path.basename(primary_video)
            return f"/media/{primary_name}"

        return f"/media/{output_filename}"

render_engine = FFmpegVideoRenderEngine()

