import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SubtitleEngine:
    """Generate SRT, ASS, and JSON subtitle files from dialogue timings."""

    def __init__(self, output_dir: str = "media/subtitles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_subtitles(self, scene_id: str, dialogue_items: List[Dict[str, Any]], base_name: str = "scene") -> Dict[str, Any]:
        srt_lines = []
        ass_lines = ["[Script Info]", "Title: Subtitle", "ScriptType: v4.00+", "WrapStyle: 2", "Collisions: Normal", "Timer: 100.0", "\n[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding", "Style: Default,Arial,24,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1", "\n[Events]"]
        json_payload = []
        for idx, item in enumerate(dialogue_items, start=1):
            start = item.get("start", 0.0)
            end = item.get("end", start + 2.0)
            speaker = item.get("speaker", "Narrator")
            text = item.get("text", item.get("line", ""))
            srt_lines.append(f"{idx}")
            srt_lines.append(f"{self._format_timestamp(start)} --> {self._format_timestamp(end)}")
            srt_lines.append(f"{speaker}: {text}")
            srt_lines.append("")
            ass_lines.append(f"Dialogue: 0,{self._format_ass_timestamp(start)},{self._format_ass_timestamp(end)},Default,,0,0,0,,{speaker}: {text}")
            json_payload.append({"index": idx, "speaker": speaker, "text": text, "start": round(start, 3), "end": round(end, 3)})

        srt_path = self.output_dir / f"{base_name}_{scene_id}.srt"
        ass_path = self.output_dir / f"{base_name}_{scene_id}.ass"
        json_path = self.output_dir / f"{base_name}_{scene_id}.json"
        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
        ass_path.write_text("\n".join(ass_lines), encoding="utf-8")
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        return {"srt_path": str(srt_path), "ass_path": str(ass_path), "json_path": str(json_path)}

    def _format_timestamp(self, seconds: float) -> str:
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        milliseconds = int(round((seconds % 1) * 1000))
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def _format_ass_timestamp(self, seconds: float) -> str:
        return f"{seconds:.2f}"


subtitle_engine = SubtitleEngine()
