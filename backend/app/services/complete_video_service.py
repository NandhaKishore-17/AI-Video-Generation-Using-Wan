import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.core.config import settings
from app.engines.character_memory import character_manager
from app.engines.compositor import compositor
from app.engines.prompt_generator import prompt_generator
from app.engines.story_engine import story_engine
from app.engines.subtitle_engine import subtitle_engine
from app.engines.music_engine import music_engine
from app.engines.voice_engine import voice_engine
from app.services.video_service import video_service

logger = logging.getLogger(__name__)


class CompleteVideoService:
    """Orchestrate story generation, character memory, prompts, audio, subtitles, music, and video composition."""

    def __init__(self):
        self.output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_complete_video(
        self,
        request: Dict[str, Any],
        job_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        if progress_callback:
            progress_callback("story", 10.0)
        story = story_engine.generate_story(
            genre=request.get("genre", "Fantasy"),
            theme=request.get("theme", "Magic School"),
            duration=int(request.get("duration", 60)),
            language=request.get("language", "English"),
        )

        for character in story.get("characters", []):
            character_manager.save_character(character)

        if progress_callback:
            progress_callback("characters", 20.0)

        scene_paths: list[str] = []
        voice_files: list[str] = []
        music_files: list[str] = []
        subtitle_files: list[str] = []

        total_scenes = max(1, len(story.get("scenes", [])))
        for scene in story.get("scenes", []):
            prompt_payload = prompt_generator.build_scene_prompt(
                scene=scene,
                characters=story.get("characters", []),
                emotion=scene.get("emotion", "neutral"),
            )
            scene_number = scene.get("scene_number", 1)
            if progress_callback:
                progress_callback("prompts", 35.0 + (scene_number / total_scenes * 15.0))

            clip_path = self.output_dir / f"scene_{scene_number:02d}.mp4"
            await video_service.generate_video(
                prompt=prompt_payload["prompt"],
                duration=max(2.0, min(6.0, int(request.get("duration", 60)) / 20.0)),
                width=1280,
                height=720,
                fps=24,
                output_path=str(clip_path),
            )
            scene_paths.append(str(clip_path))

            dialogue_list = [{"speaker": "Narrator", "line": scene.get("narration", scene.get("dialogue", "")), "context": scene.get("description", "")}] 
            for character in story.get("characters", []):
                if character.get("name") and character.get("name") in scene.get("dialogue", ""):
                    dialogue_list.append({"speaker": character.get("name"), "line": scene.get("dialogue", ""), "context": scene.get("description", "")})
            if not dialogue_list:
                dialogue_list.append({"speaker": "Narrator", "line": scene.get("dialogue", ""), "context": scene.get("description", "")})

            audio_path, srt_content, _ = await voice_engine.generate_scene_audio_and_srt(
                scene_id=f"scene_{scene_number:02d}",
                dialogue_list=dialogue_list,
                episode_id=f"ep_{job_id[:6]}",
            )
            voice_files.append(audio_path)
            if progress_callback:
                progress_callback("voice", 80.0)

            subtitle_payload = subtitle_engine.generate_subtitles(
                scene_id=f"scene_{scene_number:02d}",
                dialogue_items=[{"speaker": item["speaker"], "text": item["line"], "start": 0.0 + idx * 2.0, "end": 2.0 + idx * 2.0} for idx, item in enumerate(dialogue_list)],
                base_name=f"scene_{scene_number:02d}",
            )
            subtitle_files.append(subtitle_payload["srt_path"])

            music_payload = music_engine.generate_music(scene, duration_seconds=6.0)
            music_files.append(music_payload["path"])
            if progress_callback:
                progress_callback("music", 90.0)

            if progress_callback:
                progress_callback("video", 70.0 + (scene_number / total_scenes * 10.0))

        final_output = self.output_dir / f"{job_id}_final.mp4"
        compositor.compose(
            clips=scene_paths,
            voice_files=voice_files,
            music_files=music_files,
            subtitle_files=subtitle_files,
            output_name=final_output.name,
        )

        if progress_callback:
            progress_callback("composition", 100.0)

        return {
            "story": story,
            "output_path": str(final_output),
            "scene_paths": scene_paths,
            "voice_files": voice_files,
            "music_files": music_files,
            "subtitle_files": subtitle_files,
        }


complete_video_service = CompleteVideoService()
