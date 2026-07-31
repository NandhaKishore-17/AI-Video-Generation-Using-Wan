from sqlalchemy.orm import Session
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any

from app.models.domain import Universe, Episode, Scene, RenderTask, StoryMemory, TimelineEvent, StoryArc
from app.modules.story_director import story_director
from app.engines.llm_engine import llm_engine
from app.engines.image_engine import image_engine
from app.engines.video_engine import video_engine
from app.engines.voice_engine import voice_engine
from app.engines.music_engine import music_engine
from app.engines.memory_engine import memory_engine
from app.engines.render_engine import render_engine

logger = logging.getLogger("episode_generator")

class EpisodeGeneratorModule:
    """
    Core Module 4 & 10: Autonomous Episode Generator & Video Renderer.
    Runs full autonomous pipeline: Screenplay -> Scene Breakdown -> Images -> Videos -> Voice & SRT -> Music -> Render MP4 -> Update Memory -> Increment Universe State.
    """

    async def generate_episode_pipeline(
        self,
        db: Session,
        universe_id: str,
        custom_prompt: str = "",
        scene_duration_seconds: float = 8.5
    ) -> Episode:
        # 1. Obtain Story Brief from Story Director
        brief = await story_director.prepare_next_episode_brief(db, universe_id, custom_prompt)
        
        # 2. Create Episode Record & RenderTask
        episode = Episode(
            universe_id=universe_id,
            season=brief["season"],
            episode_number=brief["episode_number"],
            title=f"Episode {brief['episode_number']}: Crafting Script...",
            logline="Generating automated screenplay breakdown...",
            status="GENERATING"
        )
        db.add(episode)
        db.commit()
        db.refresh(episode)

        render_task = RenderTask(
            episode_id=episode.id,
            stage="SCREENPLAY",
            progress_percentage=10,
            current_step_details="Generating screenplay & scene breakdown..."
        )
        db.add(render_task)
        db.commit()

        try:
            # 3. Generate Screenplay via Qwen LLM Engine
            screenplay_data = await llm_engine.generate_screenplay(
                universe_title=brief["universe_title"],
                universe_genre=brief["universe_genre"],
                characters=brief["characters"],
                past_memories=brief["past_memories"],
                current_arc=brief["active_arc"],
                episode_number=brief["episode_number"],
                custom_prompt=custom_prompt
            )

            episode.title = screenplay_data.get("episode_title", f"Episode {brief['episode_number']}")
            episode.logline = screenplay_data.get("logline", "An epic new chapter unfolds.")
            episode.screenplay = screenplay_data
            db.commit()

            # 4. Generate Scenes, Keyframe Images, Motion Video Clips, Audio & Subtitles
            scenes_data = screenplay_data.get("scenes", [])
            total_duration = 0.0
            scene_assets_list = []

            for idx, scene_info in enumerate(scenes_data, 1):
                render_task.stage = "MEDIA_GEN"
                render_task.progress_percentage = 20 + int((idx / len(scenes_data)) * 50)
                render_task.current_step_details = f"Processing Scene {idx}/{len(scenes_data)}: Keyframes, Video, TTS, Music..."
                db.commit()

                target_duration = float(scene_duration_seconds) if scene_duration_seconds else scene_info.get("duration_seconds", 8.0)

                scene_obj = Scene(
                    episode_id=episode.id,
                    scene_number=scene_info.get("scene_number", idx),
                    location=scene_info.get("location", "LOCATION UNKNOWN"),
                    time_of_day="NIGHT" if "NIGHT" in scene_info.get("location", "") else "DAY",
                    visual_description=scene_info.get("visual_description", ""),
                    image_prompt=scene_info.get("image_prompt", ""),
                    video_motion_prompt=scene_info.get("video_motion_prompt", ""),
                    dialogue_script=scene_info.get("dialogue", []),
                    duration_seconds=target_duration
                )
                db.add(scene_obj)
                db.commit()
                db.refresh(scene_obj)

                # Character Visual Anchor String
                char_anchors = ", ".join([c["appearance_prompt"] for c in brief["characters"]])

                # Generate Image Keyframe (FLUX / SDXL)
                img_url = await image_engine.generate_scene_image(
                    scene_id=scene_obj.id,
                    prompt=scene_obj.image_prompt,
                    character_anchors=char_anchors
                )
                scene_obj.image_url = img_url

                # Generate Video Motion (CogVideoX / Wan)
                vid_url = await video_engine.generate_scene_video(
                    scene_id=scene_obj.id,
                    image_relative_url=img_url,
                    motion_prompt=scene_obj.video_motion_prompt,
                    duration_seconds=scene_obj.duration_seconds
                )
                scene_obj.video_url = vid_url

                # Generate TTS Voice Dialogue Narration & WebVTT SRT Subtitles
                audio_url, srt_content, scene_dur = await voice_engine.generate_scene_audio_and_srt(
                    scene_id=scene_obj.id,
                    dialogue_list=scene_obj.dialogue_script
                )
                scene_obj.audio_url = audio_url
                scene_obj.subtitle_srt = srt_content
                scene_obj.duration_seconds = scene_dur
                total_duration += scene_dur

                # Generate Music Score (MusicGen)
                score_url = await music_engine.generate_score_and_sfx(
                    scene_id=scene_obj.id,
                    genre=brief["universe_genre"],
                    duration_seconds=scene_dur
                )

                db.commit()

                scene_assets_list.append({
                    "scene_id": scene_obj.id,
                    "video_url": vid_url,
                    "audio_url": audio_url,
                    "score_url": score_url,
                    "subtitle_srt": srt_content,
                    "duration_seconds": scene_dur
                })

            episode.duration_seconds = total_duration
            if scene_assets_list and scene_assets_list[0].get("image_url"):
                episode.thumbnail_url = scene_assets_list[0]["image_url"]

            # 5. Composite Final MP4 Video using FFmpeg Render Engine
            render_task.stage = "FFMPEG_RENDERING"
            render_task.progress_percentage = 85
            render_task.current_step_details = "FFmpeg compositing video clips, audio tracks, and burning subtitles..."
            db.commit()

            final_video_url = await render_engine.render_episode_mp4(
                episode_id=episode.id,
                scene_assets=scene_assets_list
            )
            episode.final_video_url = final_video_url
            episode.status = "COMPLETED"

            # 6. Store Long-Term Story Memory in Vector Store (Qdrant) & DB
            memory_summary = f"Episode {episode.episode_number} ({episode.title}): {episode.logline}"
            mem_id = await memory_engine.add_episode_memory(
                universe_id=universe_id,
                episode_number=episode.episode_number,
                summary=memory_summary,
                entities_involved=[c["name"] for c in brief["characters"]]
            )

            story_mem = StoryMemory(
                universe_id=universe_id,
                episode_number=episode.episode_number,
                memory_type="EPISODE_RECAP",
                content=memory_summary,
                entities_involved=[c["name"] for c in brief["characters"]],
                vector_id=mem_id
            )
            db.add(story_mem)

            # Update Universe state
            universe = db.query(Universe).filter(Universe.id == universe_id).first()
            if universe:
                universe.total_episodes += 1

            # Update Story Arc completion status
            active_arc = db.query(StoryArc).filter(
                StoryArc.universe_id == universe_id,
                StoryArc.status == "ACTIVE"
            ).first()
            if active_arc:
                active_arc.episodes_completed += 1
                if active_arc.episodes_completed >= active_arc.episodes_planned:
                    active_arc.status = "COMPLETED"

            # Update timeline
            tl_event = TimelineEvent(
                universe_id=universe_id,
                timestamp_in_universe=f"Season {episode.season}, Ep {episode.episode_number}",
                title=episode.title,
                description=episode.logline,
                importance_score=7,
                season=episode.season,
                episode_number=episode.episode_number
            )
            db.add(tl_event)

            render_task.stage = "DONE"
            render_task.progress_percentage = 100
            render_task.current_step_details = "Episode rendered, memories stored, and universe updated."
            render_task.completed_at = datetime.utcnow()

            db.commit()
            db.refresh(episode)
            logger.info(f"Episode {episode.episode_number} pipeline completed successfully!")
            return episode

        except Exception as e:
            logger.error(f"Error during episode generation pipeline: {e}", exc_info=True)
            episode.status = "FAILED"
            render_task.stage = "FAILED"
            render_task.error_log = str(e)
            db.commit()
            raise e

episode_generator = EpisodeGeneratorModule()
