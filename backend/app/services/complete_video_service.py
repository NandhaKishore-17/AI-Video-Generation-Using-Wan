import logging
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.core.config import settings, BASE_DIR
from app.engines.character_memory import character_manager
from app.engines.compositor import compositor
from app.engines.prompt_generator import prompt_generator
from app.engines.story_engine import story_engine
from app.engines.subtitle_engine import subtitle_engine
from app.engines.music_engine import music_engine
from app.engines.voice_engine import voice_engine
from app.engines.dialogue_engine import dialogue_engine
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
        # Requirement 8 & 9: Clear all dialogue caches before starting
        dialogue_engine.clear_cache()

        # Requirement 10: Create a new UUID-based working directory under media/jobs/
        job_dir_name = job_id[:7] if len(job_id) > 7 else job_id
        job_dir = Path(BASE_DIR) / "media" / "jobs" / job_dir_name
        audio_dir = job_dir / "audio"
        subtitles_dir = job_dir / "subtitles"
        video_dir = job_dir / "video"

        audio_dir.mkdir(parents=True, exist_ok=True)
        subtitles_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback:
            progress_callback("Generating screenplay...", 10.0)
            
        episode_number = int(request.get("episode_number", 1))
        universe_id = f"job_{job_id}"

        story = story_engine.generate_story(
            genre=request.get("genre", "Fantasy"),
            theme=request.get("theme", request.get("universe_title", "Magic School")),
            duration=int(request.get("duration", 60)),
            language=request.get("language", "English"),
            episode_number=episode_number,
            characters=request.get("characters"),
            world_rules=request.get("world_rules", ""),
            previous_summaries=request.get("previous_summaries"),
        )

        print(f"Final story passed to voice and video generators:\n{json.dumps(story, indent=2)}")
        logger.info(f"Final story passed to voice and video generators:\n{json.dumps(story, indent=2)}")

        # Requirement 10: Save story.json in working directory
        with open(job_dir / "story.json", "w", encoding="utf-8") as f:
            json.dump(story, f, indent=2)

        for character in story.get("characters", []):
            character_manager.save_character(character)

        if progress_callback:
            progress_callback("characters", 20.0)

        scene_paths: list[str] = []
        voice_files: list[str] = []
        music_files: list[str] = []
        subtitle_files: list[str] = []
        accumulated_dialogue = []

        total_scenes = max(1, len(story.get("scenes", [])))
        for scene in story.get("scenes", []):
            prompt_payload = prompt_generator.build_scene_prompt(
                scene=scene,
                characters=story.get("characters", []),
                emotion=scene.get("emotion", "neutral"),
            )
            scene_number = scene.get("scene_number", 1)
            
            # Print required Stage 5 log: "Generating Scene {scene_number}"
            print(f"Generating Scene {scene_number}")
            logger.info(f"Generating Scene {scene_number}")

            if progress_callback:
                progress_callback("prompts", 35.0 + (scene_number / total_scenes * 15.0))

            # Requirement 10: Save scene videos under job video/ folder
            clip_path = video_dir / f"scene_{scene_number:03d}.mp4"
            await video_service.generate_video(
                prompt=prompt_payload["prompt"],
                duration=max(2.0, min(6.0, int(request.get("duration", 60)) / 20.0)),
                width=1280,
                height=720,
                fps=24,
                output_path=str(clip_path),
            )
            scene_paths.append(str(clip_path))

            # Requirement 2 & 12: Generate dialogue dynamically scene-by-scene with correct episode_number
            dialogue_res = await dialogue_engine.generate_dialogue_for_scene(
                scene_number=scene_number,
                scene_title=f"Scene {scene_number}",
                scene_description=scene.get("description", ""),
                characters=story.get("characters", []),
                scene_emotion=scene.get("emotion", "neutral"),
                previous_scene_summary="" if scene_number == 1 else story.get("scenes", [])[scene_number - 2].get("description", ""),
                episode_objective=story.get("summary", request.get("theme", "")),
                universe_id=universe_id,
                previous_dialogues=accumulated_dialogue,
                episode_number=episode_number
            )
            dialogue_list = dialogue_res.get("dialogue", [])
            accumulated_dialogue.extend(dialogue_list)
            
            # Stage 1 Verification: Ensure dialogue exists for every scene
            if not dialogue_list:
                raise ValueError(f"Dialogue generation failed: No dialogue produced for Scene {scene_number}")

            # Log every dialogue string before TTS
            logger.info(f"Dialogue to process for TTS in Scene {scene_number}: {dialogue_list}")

            # Print required Stage 5 log: "Generating voice..."
            print("Generating voice...")
            logger.info("Generating voice...")

            # Requirement 3 & 4 & 5: Voice generation for scene dialogue with unique audio paths
            audio_path, srt_content, _ = await voice_engine.generate_scene_audio_and_srt(
                scene_id=f"scene_{scene_number:03d}",
                dialogue_list=dialogue_list,
                episode_id=f"ep_{job_id[:6]}",
                episode_number=episode_number,
                job_id=job_id,
            )

            # Stage 3 Verification: Resolve absolute path and run audio file checks
            resolved_audio_path = compositor._resolve_path(audio_path)
            if not resolved_audio_path or not os.path.exists(resolved_audio_path):
                raise FileNotFoundError(f"Voice generation failed: Audio file not found at {audio_path}")

            # If a WAV file is missing, empty, or under 1 KB, raise an exception
            file_size = os.path.getsize(resolved_audio_path)
            if file_size < 1024:
                raise ValueError(f"WAV file validation failed: {resolved_audio_path} size is {file_size} bytes (under 1 KB)")

            # Validate audio duration, sample rate, and audio stream existence via ffprobe
            dur_seconds, sr_val, has_stream = await voice_engine.validate_audio_file(resolved_audio_path)
            if dur_seconds <= 0:
                raise ValueError(f"WAV file validation failed: {resolved_audio_path} has duration {dur_seconds}s (must be > 0)")
            if not has_stream:
                raise ValueError(f"WAV file validation failed: No valid audio stream detected by ffprobe in {resolved_audio_path}")

            # Print required Stage 5 logs: "Saved audio:" and "Audio duration:"
            print(f"Saved audio: {resolved_audio_path}")
            print(f"Audio duration: {dur_seconds:.2f}")

            # Log stage 2 details: dialogue, speaker, output wav path, file size, duration
            logger.info(
                "=== VOICE GENERATION REPORT ===\n"
                f"Dialogue: {dialogue_list}\n"
                f"Speakers: {[item.get('speaker') for item in dialogue_list]}\n"
                f"Wav Path: {resolved_audio_path}\n"
                f"File Size: {file_size} bytes\n"
                f"Duration: {dur_seconds:.2f}s\n"
                "=============================="
            )

            voice_files.append(audio_path)
            if progress_callback:
                progress_callback("voice", 80.0)

            # Requirement 6 & 10: Subtitle generation uses current scene dialogue, saved under job subtitles/ folder
            subtitle_payload = subtitle_engine.generate_subtitles(
                scene_id=f"scene_{scene_number:03d}",
                dialogue_items=[
                    {
                        "speaker": item["speaker"],
                        "text": item.get("text") or item.get("line") or "",
                        "start": 0.0 + idx * 2.0,
                        "end": 2.0 + idx * 2.0
                    }
                    for idx, item in enumerate(dialogue_list)
                ],
                base_name=f"scene_{scene_number:03d}",
                output_dir=subtitles_dir,
            )
            subtitle_files.append(subtitle_payload["srt_path"])

            # Requirement 10: Generate background music unique path
            music_payload = music_engine.generate_music(scene, duration_seconds=6.0, job_id=job_id)
            music_files.append(music_payload["path"])
            if progress_callback:
                progress_callback("music", 90.0)

            # Requirement 13: Structured log statement
            logger.info(
                "=== PIPELINE LOG ===\n"
                f"Current Episode: {episode_number}\n"
                f"Current Scene: {scene_number}\n"
                f"Dialogue Generated: {dialogue_list}\n"
                f"Voice Generated: {[d.get('voice', 'N/A') or d.get('speaker', 'N/A') for d in dialogue_list]}\n"
                f"Audio Path: {audio_path}\n"
                "===================="
            )

            if progress_callback:
                progress_callback("video", 70.0 + (scene_number / total_scenes * 10.0))

        # Requirement 1 & 10: Save script.json in working directory
        with open(job_dir / "script.json", "w", encoding="utf-8") as f:
            json.dump({
                "episode_number": episode_number,
                "dialogues": accumulated_dialogue
            }, f, indent=2)

        # Save general episode summary to character memories
        for character in story.get("characters", []):
            character_manager.add_memory(
                character_name=character["name"],
                universe_id=universe_id,
                episode_num=episode_number,
                content=f"Successfully finished Episode {episode_number}: {story.get('title')}. Objective: {story.get('summary')}."
            )

        # Requirement 10: Save composed video under job video/ folder
        final_output = video_dir / f"{job_id}_final.mp4"
        
        # Print required Stage 5 log: "Merging audio..."
        print("Merging audio...")
        logger.info("Merging audio...")

        # Override output_dir on compositor so composition happens inside video_dir
        original_compositor_dir = compositor.output_dir
        compositor.output_dir = video_dir
        try:
            compositor.compose(
                clips=scene_paths,
                voice_files=voice_files,
                music_files=music_files,
                subtitle_files=subtitle_files,
                output_name=final_output.name,
            )
        finally:
            compositor.output_dir = original_compositor_dir

        # Stage 4 stream verification: Verify composed MP4 contains video and audio streams
        has_video_stream, has_audio_stream = compositor.verify_composed_streams(str(final_output))
        
        # Print required Stage 5 logs: "Final MP4:" and "Audio stream detected: YES/NO"
        print(f"Final MP4: {final_output}")
        print(f"Audio stream detected: {'YES' if has_audio_stream else 'NO'}")
        
        if not has_video_stream or not has_audio_stream:
            raise RuntimeError(f"Composition validation failed: Final MP4 is missing valid streams. Video stream: {has_video_stream}, Audio stream: {has_audio_stream}")

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

    async def generate_story_only(
        self,
        request: Dict[str, Any],
        job_id: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """
        Story-only pipeline: Ollama LLM story → dialogue (Ollama) → TTS (voice.wav) → subtitles (.srt)
        Skips: Wan2.2, FLUX, SDXL, image generation, video generation, FFmpeg composition.
        Saves debug files: episode.json, screenplay.txt, dialogues.json, subtitles.srt, voice.wav
        """
        dialogue_engine.clear_cache()

        job_dir_name = job_id[:7] if len(job_id) > 7 else job_id
        job_dir = Path(BASE_DIR) / "media" / "jobs" / job_dir_name
        audio_dir = job_dir / "audio"
        subtitles_dir = job_dir / "subtitles"

        audio_dir.mkdir(parents=True, exist_ok=True)
        subtitles_dir.mkdir(parents=True, exist_ok=True)

        episode_number = int(request.get("episode_number", 1))
        universe_id = f"job_{job_id}"

        # ── Step 1: Generate Story via Ollama ─────────────────────────────────
        if progress_callback:
            progress_callback("Generating screenplay...", 5.0)

        logger.info("=== STORY-ONLY PIPELINE START ===")
        logger.info("Universe: %s | Genre: %s | Episode: %d", request.get("theme", "?"), request.get("genre", "?"), episode_number)
        print(f"=== STORY-ONLY PIPELINE START | Universe: {request.get('theme')} | Episode: {episode_number} ===")

        story = story_engine.generate_story(
            genre=request.get("genre", "Fantasy"),
            theme=request.get("theme", request.get("universe_title", "Magic School")),
            duration=int(request.get("duration", 60)),
            language=request.get("language", "English"),
            episode_number=episode_number,
            characters=request.get("characters"),
            world_rules=request.get("world_rules", ""),
            previous_summaries=request.get("previous_summaries"),
        )

        # Save episode.json
        with open(job_dir / "episode.json", "w", encoding="utf-8") as f:
            json.dump({"episode_number": episode_number, "universe_id": universe_id, **story}, f, indent=2)

        # Save screenplay.txt
        screenplay_lines = [
            f"TITLE: {story.get('title', '')}",
            f"GENRE: {story.get('genre', '')}",
            f"THEME: {story.get('theme', '')}",
            f"SUMMARY: {story.get('summary', '')}",
            "",
            "CHARACTERS:",
        ]
        for c in story.get("characters", []):
            screenplay_lines.append(f"  - {c['name']} ({c.get('role','')}) | {c.get('personality','')}")
        screenplay_lines.append("")
        screenplay_lines.append("SCENES:")
        for s in story.get("scenes", []):
            screenplay_lines.append(f"\n  SCENE {s['scene_number']}: {s.get('emotion', '')} mood")
            screenplay_lines.append(f"  {s.get('description', '')}")
        with open(job_dir / "screenplay.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(screenplay_lines))

        logger.info("Parsed screenplay:\n%s", "\n".join(screenplay_lines))

        for character in story.get("characters", []):
            character_manager.save_character(character)

        if progress_callback:
            progress_callback("Story ready, generating dialogue...", 20.0)

        # ── Step 2: Generate Dialogue scene-by-scene via Ollama ───────────────
        all_dialogues: list = []
        all_srt_blocks: list = []
        all_voice_paths: list = []
        accumulated_dialogue = []
        srt_index = 1
        srt_time_cursor = 0.0

        total_scenes = max(1, len(story.get("scenes", [])))
        for scene in story.get("scenes", []):
            scene_number = scene.get("scene_number", 1)
            scene_progress = 20.0 + (scene_number / total_scenes) * 60.0

            print(f"\n--- Scene {scene_number}/{total_scenes} | {scene.get('emotion','neutral')} ---")
            logger.info("Processing scene %d/%d", scene_number, total_scenes)

            if progress_callback:
                progress_callback(f"Generating dialogue for scene {scene_number}...", scene_progress)

            dialogue_res = await dialogue_engine.generate_dialogue_for_scene(
                scene_number=scene_number,
                scene_title=f"Scene {scene_number}: {scene.get('description', '')[:50]}",
                scene_description=scene.get("description", ""),
                characters=story.get("characters", []),
                scene_emotion=scene.get("emotion", "neutral"),
                previous_scene_summary="" if scene_number == 1 else story.get("scenes", [])[scene_number - 2].get("description", ""),
                episode_objective=story.get("summary", request.get("theme", "")),
                universe_id=universe_id,
                previous_dialogues=accumulated_dialogue,
                episode_number=episode_number,
            )
            dialogue_list = dialogue_res.get("dialogue", [])
            if not dialogue_list:
                raise ValueError(f"Dialogue generation failed: no dialogue for scene {scene_number}")

            accumulated_dialogue.extend(dialogue_list)
            all_dialogues.extend([{"scene": scene_number, **d} for d in dialogue_list])

            logger.info("Generated dialogue for scene %d:\n%s", scene_number, json.dumps(dialogue_list, indent=2))
            print(f"Generated dialogue:\n{json.dumps(dialogue_list, indent=2)}")

            # ── Step 3: TTS per scene ─────────────────────────────────────────
            if progress_callback:
                progress_callback(f"TTS for scene {scene_number}...", scene_progress + 5.0)

            audio_path, srt_content, scene_duration = await voice_engine.generate_scene_audio_and_srt(
                scene_id=f"scene_{scene_number:03d}",
                dialogue_list=dialogue_list,
                episode_id=f"ep_{episode_number:03d}",
                episode_number=episode_number,
                job_id=job_id,
            )

            print(f"TTS complete for scene {scene_number}: audio={audio_path} duration={scene_duration:.2f}s")
            logger.info("TTS scene %d: path=%s duration=%.2fs", scene_number, audio_path, scene_duration)
            all_voice_paths.append(audio_path)

            # Build SRT blocks for this scene
            for d in dialogue_list:
                start_ts = srt_time_cursor
                end_ts = srt_time_cursor + max(2.0, len((d.get("text") or d.get("line") or "").split()) * 0.4)
                h_s, m_s = divmod(int(start_ts), 3600)
                m_s, s_s = divmod(m_s, 60)
                h_e, m_e = divmod(int(end_ts), 3600)
                m_e, s_e = divmod(m_e, 60)
                ts_start = f"{h_s:02d}:{m_s:02d}:{s_s:02d},{int((start_ts%1)*1000):03d}"
                ts_end = f"{h_e:02d}:{m_e:02d}:{s_e:02d},{int((end_ts%1)*1000):03d}"
                text = d.get("text") or d.get("line") or ""
                all_srt_blocks.append(f"{srt_index}\n{ts_start} --> {ts_end}\n{d.get('speaker','')}: {text}\n")
                srt_index += 1
                srt_time_cursor = end_ts

        # ── Save debug files ──────────────────────────────────────────────────
        with open(job_dir / "dialogues.json", "w", encoding="utf-8") as f:
            json.dump(all_dialogues, f, indent=2)

        srt_text = "\n".join(all_srt_blocks)
        with open(job_dir / "subtitles.srt", "w", encoding="utf-8") as f:
            f.write(srt_text)

        # Copy first voice file to voice.wav for easy inspection
        if all_voice_paths:
            import shutil
            first_audio = all_voice_paths[0]
            try:
                from app.engines.compositor import compositor as _comp
                first_audio_abs = _comp._resolve_path(first_audio) or first_audio
                if os.path.exists(first_audio_abs):
                    shutil.copy2(first_audio_abs, str(job_dir / "voice.wav"))
            except Exception as copy_err:
                logger.warning("Could not copy voice.wav for inspection: %s", copy_err)

        if progress_callback:
            progress_callback("Story pipeline complete.", 100.0)

        logger.info("=== STORY-ONLY PIPELINE COMPLETE | job_dir=%s ===", job_dir)
        print(f"=== STORY-ONLY PIPELINE COMPLETE | Debug files at: {job_dir} ===")

        return {
            "status": "complete",
            "job_id": job_id,
            "job_dir": str(job_dir),
            "story": story,
            "episode_number": episode_number,
            "universe_id": universe_id,
            "dialogues": all_dialogues,
            "voice_files": all_voice_paths,
            "subtitles_srt": srt_text,
            "debug_files": {
                "episode_json": str(job_dir / "episode.json"),
                "screenplay_txt": str(job_dir / "screenplay.txt"),
                "dialogues_json": str(job_dir / "dialogues.json"),
                "subtitles_srt": str(job_dir / "subtitles.srt"),
                "voice_wav": str(job_dir / "voice.wav"),
            },
        }


complete_video_service = CompleteVideoService()
