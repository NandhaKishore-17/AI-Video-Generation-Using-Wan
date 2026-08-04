import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from app.core.config import settings

logger = logging.getLogger(__name__)

class VideoCompositor:
    """Compose a final MP4 from scene clips, voice, music, and subtitles using ffmpeg."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir or settings.MEDIA_OUTPUT_DIR).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compose(self, clips: List[str], voice_files: List[str], music_files: List[str], subtitle_files: List[str], output_name: str = "final_video.mp4") -> Dict[str, Any]:
        output_path = self.output_dir / output_name
        if not clips:
            raise ValueError("At least one clip is required")
        if len(clips) == 1:
            command = ["ffmpeg", "-y", "-i", clips[0], "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)]
        else:
            command = ["ffmpeg", "-y"]
            for clip in clips:
                command.extend(["-i", clip])
            command.extend(["-filter_complex", self._build_filter_complex(len(clips)), "-map", "[vout]", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "ffmpeg composition failed")
        return {"output_path": str(output_path), "clips": clips, "voice_files": voice_files, "music_files": music_files, "subtitle_files": subtitle_files}

    def _build_filter_complex(self, clip_count: int) -> str:
        parts = []
        for index in range(clip_count):
            parts.append(f"[{index}:v]")
        parts.append(f"concat=n={clip_count}:v=1:a=0[vout]")
        return "".join(parts)


compositor = VideoCompositor()
