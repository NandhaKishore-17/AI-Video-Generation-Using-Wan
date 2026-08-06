import subprocess
from pathlib import Path

from app.core.config import settings
from app.engines.video.video_engine import VideoEngine


class MockVideoEngine(VideoEngine):
    """A mock video engine that generates a simple placeholder MP4 using ffmpeg."""

    def render_placeholder(self, job_id: str) -> str:
        output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}.mp4"
        return self.render_video(
            scene_prompt=f"placeholder for {job_id}",
            output_path=str(output_path),
            width=640,
            height=360,
            fps=12,
            duration=2.0,
            seed=42,
        )

    def render_video(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        return self._render_video_impl(output_path, width, height, fps, duration)

    async def render_video_async(
        self,
        scene_prompt: str,
        output_path: str,
        width: int,
        height: int,
        fps: int,
        duration: float,
        seed: int,
    ) -> str:
        return self._render_video_impl(output_path, width, height, fps, duration)

    def _render_video_impl(self, output_path: str, width: int, height: int, fps: int, duration: float) -> str:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        output_path = str(Path(output_path))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={width}x{height}:d={duration}",
            "-vf",
            f"fps={fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
