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
        Stitches all scenes together by first muxing each scene's video, audio, and subtitles,
        then concatenating the fully formed scene clips into a final episode MP4.
        """
        output_filename = f"final_episode_{episode_id}.mp4"
        output_path = os.path.join(self.media_dir, output_filename)
        concat_list_path = os.path.join(self.temp_dir, f"concat_{episode_id}.txt")

        ffmpeg_exe = self._get_ffmpeg_exe()
        valid_muxed_paths = []

        for scene_idx, scene in enumerate(scene_assets):
            # Collect scene video path — strip query strings (?t=timestamp) before file lookup
            video_rel = scene.get("video_url")
            if not video_rel:
                continue
            video_filename = os.path.basename(video_rel.split("?")[0])
            full_video_path = os.path.join(self.media_dir, video_filename)
            if not os.path.exists(full_video_path):
                logger.warning("Video file not found: %s (from URL: %s)", full_video_path, video_rel)
                continue

            # Collect scene character audio path — strip query strings before lookup
            audio_rel = scene.get("audio_url")
            full_audio_path = None
            if audio_rel:
                audio_filename = os.path.basename(audio_rel.split("?")[0])
                candidate_paths = [
                    os.path.join(self.media_dir, audio_filename),
                    os.path.join(self.media_dir, "audio", audio_filename),
                    audio_rel.split("?")[0],
                ]
                for candidate in candidate_paths:
                    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                        full_audio_path = candidate
                        break

            # Collect scene music/score path — strip query strings before lookup
            score_rel = scene.get("score_url")
            full_score_path = None
            if score_rel:
                score_filename = os.path.basename(score_rel.split("?")[0])
                candidate_paths = [
                    os.path.join(self.media_dir, score_filename),
                    os.path.join(self.media_dir, "audio", score_filename),
                    score_rel.split("?")[0],
                ]
                for candidate in candidate_paths:
                    if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                        full_score_path = candidate
                        break

            # Save scene-specific subtitles
            srt_content = scene.get("subtitle_srt", "")
            scene_srt_path = os.path.join(self.temp_dir, f"subtitles_{episode_id}_{scene_idx}.srt")
            has_subtitles = False
            if srt_content.strip():
                with open(scene_srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content.strip() + "\n")
                has_subtitles = True

            muxed_filename = f"muxed_scene_{episode_id}_{scene_idx}.mp4"
            muxed_path = os.path.join(self.temp_dir, muxed_filename)

            # Build FFmpeg command to loop video, mix audio/music, and burn subtitles
            cmd = [ffmpeg_exe, "-y", "-stream_loop", "-1", "-i", full_video_path]
            
            has_a = full_audio_path and os.path.exists(full_audio_path)
            has_m = full_score_path and os.path.exists(full_score_path)

            if has_a and has_m:
                cmd.extend(["-i", full_audio_path, "-i", full_score_path])
                cmd.extend([
                    "-filter_complex",
                    "[1:a]aformat=sample_rates=44100:channel_layouts=stereo[v];[2:a]volume=0.25,aformat=sample_rates=44100:channel_layouts=stereo[m];[v][m]amix=inputs=2:duration=longest[aout]",
                    "-map", "0:v:0",
                    "-map", "[aout]"
                ])
            elif has_a:
                cmd.extend(["-i", full_audio_path, "-map", "0:v:0", "-map", "1:a:0"])
            elif has_m:
                cmd.extend(["-i", full_score_path, "-map", "0:v:0", "-map", "1:a:0"])
            else:
                cmd.extend(["-map", "0:v:0"])

            if has_subtitles:
                clean_srt = scene_srt_path.replace("\\", "/").replace(":", "\\:")
                cmd.extend(["-vf", f"subtitles='{clean_srt}'"])

            cmd.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                muxed_path
            ])

            logger.info(f"Muxing scene {scene_idx} with command: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(muxed_path) and os.path.getsize(muxed_path) > 0:
                    valid_muxed_paths.append(muxed_path)
                else:
                    valid_muxed_paths.append(full_video_path)
            except Exception as e:
                logger.warning(f"Muxing scene {scene_idx} failed: {e}. Falling back to raw video.")
                valid_muxed_paths.append(full_video_path)

        if not valid_muxed_paths:
            logger.warning("No valid video scenes to concatenate.")
            return ""

        # Concatenate all muxed scene videos
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for vp in valid_muxed_paths:
                clean_p = vp.replace("\\", "/")
                f.write(f"file '{clean_p}'\n")

        import time
        timestamp = int(time.time())
        try:
            concat_cmd = [
                ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
                "-c", "copy", output_path
            ]
            logger.info(f"Concatenating episode with command: {' '.join(concat_cmd)}")
            subprocess.run(concat_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"FFmpeg render successful: {output_path}")
            return f"/media/{output_filename}?t={timestamp}"
        except Exception as e:
            logger.warning(f"Final concatenation failed: {e}.")
            if valid_muxed_paths:
                return f"/media/{os.path.basename(valid_muxed_paths[0])}?t={timestamp}"

        return f"/media/{output_filename}?t={timestamp}"

render_engine = FFmpegVideoRenderEngine()

