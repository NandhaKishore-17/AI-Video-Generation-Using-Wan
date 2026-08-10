from sqlalchemy.orm import Session
import logging
import asyncio
import json
from datetime import datetime
import hashlib
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
from app.engines.dialogue_engine import dialogue_engine
from app.engines.character_memory import character_manager
from app.core.database import SessionLocal

logger = logging.getLogger("episode_generator")

class EpisodeGeneratorModule:
    """
    Core Module 4 & 10: Autonomous Episode Generator & Video Renderer.
    Runs full autonomous pipeline: Screenplay -> Scene Breakdown -> Dialogue Generation -> Images -> Videos -> Voice & SRT -> Music -> Render MP4 -> Update Memory -> Increment Universe State.
    """

    def _set_episode_failed(self, episode_id: str, error_msg: str):
        db = SessionLocal()
        try:
            episode = db.query(Episode).filter(Episode.id == episode_id).first()
            if episode:
                episode.status = "FAILED"
            render_task = db.query(RenderTask).filter(RenderTask.episode_id == episode_id).first()
            if render_task:
                render_task.stage = "FAILED"
                render_task.error_log = error_msg
            db.commit()
        except Exception as e:
            logger.error(f"Failed to update episode failure state: {e}", exc_info=True)
        finally:
            db.close()

    async def generate_episode_pipeline(
        self,
        episode_id: str,
        universe_id: str,
        brief: Dict[str, Any],
        custom_prompt: str = "",
        scene_duration_seconds: float = 8.5
    ) -> None:
        # Phase 1: Setup
        db = SessionLocal()
        try:
            episode = db.query(Episode).filter(Episode.id == episode_id).first()
            render_task = db.query(RenderTask).filter(RenderTask.episode_id == episode_id).first()
            
            if not episode or not render_task:
                logger.error(f"Cannot run pipeline: Episode {episode_id} not found.")
                return
                
            episode.status = "GENERATING_SCREENPLAY"
            render_task.stage = "GENERATING_SCREENPLAY"
            db.commit()
            
            # extract needed info for the LLM phase before closing session
            episode_number = episode.episode_number
            episode_season = episode.season
            
            print(f"GENERATION START")
            print(f"episode_id={episode_id}")
            print(f"season={episode_season}")
            print(f"episode_number={episode_number}")
            
            # Fetch previous episode for uniqueness check
            previous_episode = db.query(Episode).filter(
                Episode.universe_id == universe_id,
                Episode.episode_number == episode_number - 1
            ).first()
            previous_hash = ""
            if previous_episode and previous_episode.screenplay:
                prev_sp = previous_episode.screenplay
                prev_text = f"{prev_sp.get('episode_title', '')}{prev_sp.get('logline', '')}{prev_sp.get('summary', '')}"
                for s in prev_sp.get("scenes", []):
                    prev_text += str(s.get("dialogue", []))
                previous_hash = hashlib.sha256(prev_text.encode('utf-8')).hexdigest()
        except Exception as e:
            logger.error(f"Error during episode generation pipeline setup: {e}", exc_info=True)
            db.rollback()
            return
        finally:
            db.close()

        # Phase 2: Generate Screenplay (NO active db session)
        try:
            screenplay_data = None
            for attempt in range(3):
                attempt_prompt = custom_prompt
                if attempt > 0:
                    attempt_prompt = f"MUST BE COMPLETELY DIFFERENT FROM PREVIOUS EPISODE. {custom_prompt}"
                    
                screenplay_data = await llm_engine.generate_screenplay(
                    universe_title=brief["universe_title"],
                    universe_genre=brief["universe_genre"],
                    characters=brief["characters"],
                    past_memories=brief["past_memories"],
                    previous_episode_summaries=brief.get("previous_episode_summaries", []),
                    current_arc=brief["active_arc"],
                    episode_number=episode_number,
                    custom_prompt=attempt_prompt,
                    universe_lore=brief.get("universe_lore", "")
                )
                
                # Check uniqueness
                curr_text = f"{screenplay_data.get('episode_title', '')}{screenplay_data.get('logline', '')}{screenplay_data.get('summary', '')}"
                for s in screenplay_data.get("scenes", []):
                    curr_text += str(s.get("dialogue", []))
                current_hash = hashlib.sha256(curr_text.encode('utf-8')).hexdigest()
                
                print("OLLAMA GENERATION")
                print(f"prompt_hash=N/A")
                print(f"response_hash={current_hash}")
                
                if previous_hash and current_hash == previous_hash:
                    logger.warning("WARNING: Ollama generated identical screenplay to previous episode.")
                    if attempt < 2:
                        continue
                    else:
                        raise ValueError("Ollama repeatedly generated identical screenplay to previous episode")
                break
                
            print("SCREENPLAY GENERATED")
            print(f"title={screenplay_data.get('episode_title', 'Unknown')}")
            print(f"logline={screenplay_data.get('logline', 'Unknown')}")
            print(f"scene_count={len(screenplay_data.get('scenes', []))}")
            
            print(f"Final story passed to voice and video generators:\n{json.dumps(screenplay_data, indent=2)}")
            logger.info(f"Final story passed to voice and video generators:\n{json.dumps(screenplay_data, indent=2)}")
        except Exception as e:
            logger.error(f"Error generating screenplay: {e}", exc_info=True)
            self._set_episode_failed(episode_id, str(e))
            return

        # Phase 3: Save Screenplay
        db = SessionLocal()
        try:
            episode = db.query(Episode).filter(Episode.id == episode_id).first()
            episode.title = screenplay_data.get("episode_title", f"Episode {episode_number}")
            episode.logline = screenplay_data.get("logline", "An epic new chapter unfolds.")
            episode.summary = screenplay_data.get("summary", episode.logline)
            episode.screenplay = screenplay_data
            episode.status = "GENERATING_MEDIA"
            db.commit()
            episode_logline = episode.logline
            episode_title = episode.title
            episode_summary = episode.summary
            logger.info(f"[SCREENPLAY] Saved screenplay for Episode {episode_number}")
        except Exception as e:
            logger.error(f"[SCREENPLAY] Error saving screenplay: {e}", exc_info=True)
            db.rollback()
            self._set_episode_failed(episode_id, str(e))
            return
        finally:
            db.close()

        # Phase 4: Generate Media
        try:
            scenes_data = screenplay_data.get("scenes", [])
            total_duration = 0.0
            scene_assets_list = []
            accumulated_dialogue = []

            chars_for_dialogue = []
            for c in brief["characters"]:
                chars_for_dialogue.append({
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "role": c.get("role"),
                    "personality": c.get("personality")
                })

            for idx, scene_info in enumerate(scenes_data, 1):
                # Update progress
                db = SessionLocal()
                try:
                    render_task = db.query(RenderTask).filter(RenderTask.episode_id == episode_id).first()
                    render_task.stage = "MEDIA_GEN"
                    render_task.progress_percentage = 20 + int((idx / max(len(scenes_data), 1)) * 50)
                    render_task.current_step_details = f"Processing Scene {idx}/{len(scenes_data)}..."
                    db.commit()
                except Exception:
                    db.rollback()
                finally:
                    db.close()

                target_duration = float(scene_duration_seconds) if scene_duration_seconds else float(scene_info.get("duration_seconds", 8.0))

                # Dialogue Gen (NO active db session)
                logger.info(f"[DIALOGUE] Generating dialogue for scene {idx}...")
                active_char_names = scene_info.get("characters", [])
                if active_char_names:
                    scene_chars = [c for c in chars_for_dialogue if c["name"] in active_char_names]
                else:
                    scene_chars = chars_for_dialogue

                try:
                    dialogue_res = await dialogue_engine.generate_dialogue_for_scene(
                        scene_number=idx,
                        scene_title=scene_info.get("location", f"Scene {idx}"),
                        scene_description=scene_info.get("visual_description", ""),
                        characters=scene_chars,
                        scene_emotion=scene_info.get("emotion", "neutral") or "neutral",
                        previous_scene_summary="" if idx == 1 else scenes_data[idx - 2].get("visual_description", ""),
                        episode_objective=episode_logline,
                        universe_id=universe_id,
                        previous_dialogues=accumulated_dialogue,
                        episode_number=episode_number,
                        scene_context={
                            "episode_title": episode_title,
                            "episode_summary": episode_summary
                        }
                    )
                    dialogue_script = dialogue_res.get("dialogue", [])
                    if not dialogue_script:
                        raise ValueError("No valid dialogue returned by Ollama.")
                    accumulated_dialogue.extend(dialogue_script)
                except Exception as e:
                    logger.error(f"[DIALOGUE] Validation failed for scene {idx}: {e}")
                    self._set_episode_failed(episode_id, f"Dialogue generation failed for scene {idx}: {e}")
                    return

                # Save Scene row
                logger.info(f"[DATABASE] Saving Scene {idx} record...")
                db = SessionLocal()
                try:
                    scene_obj = Scene(
                        episode_id=episode_id,
                        scene_number=scene_info.get("scene_number", idx),
                        location=scene_info.get("location", "LOCATION UNKNOWN"),
                        time_of_day="NIGHT" if "NIGHT" in scene_info.get("location", "") else "DAY",
                        visual_description=scene_info.get("visual_description", ""),
                        image_prompt=scene_info.get("image_prompt", ""),
                        video_motion_prompt=scene_info.get("video_motion_prompt", ""),
                        dialogue_script=dialogue_script,
                        duration_seconds=target_duration
                    )
                    db.add(scene_obj)
                    db.commit()
                    db.refresh(scene_obj)
                    scene_id = scene_obj.id
                    scene_img_prompt = scene_obj.image_prompt
                    scene_vid_prompt = scene_obj.video_motion_prompt
                except Exception as e:
                    db.rollback()
                    raise e
                finally:
                    db.close()

                # Generate Assets (NO active db session)
                char_anchors = ", ".join([c.get("appearance_prompt", "") for c in brief["characters"] if c.get("appearance_prompt")])
                
                logger.info(f"[IMAGE] Generating image for Scene {idx}...")
                img_url = await image_engine.generate_scene_image(scene_id=scene_id, prompt=scene_img_prompt, character_anchors=char_anchors)
                
                logger.info(f"[VIDEO] Generating video for Scene {idx}...")
                vid_url = await video_engine.generate_scene_video(scene_id=scene_id, image_relative_url=img_url, motion_prompt=scene_vid_prompt, duration_seconds=target_duration)
                
                logger.info(f"[TTS] Generating audio for Scene {idx}...")
                try:
                    audio_url, srt_content, scene_dur = await voice_engine.generate_scene_audio_and_srt(scene_id=scene_id, dialogue_list=dialogue_script)
                    logger.info(f"[TTS] Audio generated for Scene {idx}: {audio_url}")
                except Exception as e:
                    logger.error(f"[TTS] Audio generation failed for Scene {idx}: {e}")
                    self._set_episode_failed(episode_id, f"TTS generation failed for scene {idx}: {e}")
                    return
                    
                score_url = await music_engine.generate_score_and_sfx(scene_id=scene_id, genre=brief["universe_genre"], duration_seconds=scene_dur)

                # Save Asset URLs
                logger.info(f"[DATABASE] Updating asset URLs for Scene {idx}...")
                db = SessionLocal()
                try:
                    scene_obj = db.query(Scene).filter(Scene.id == scene_id).first()
                    if scene_obj:
                        scene_obj.image_url = img_url
                        scene_obj.video_url = vid_url
                        scene_obj.audio_url = audio_url
                        scene_obj.subtitle_srt = srt_content
                        scene_obj.duration_seconds = scene_dur
                        db.commit()
                except Exception:
                    db.rollback()
                finally:
                    db.close()

                total_duration += scene_dur
                scene_assets_list.append({
                    "scene_id": scene_id,
                    "image_url": img_url,
                    "video_url": vid_url,
                    "audio_url": audio_url,
                    "score_url": score_url,
                    "subtitle_srt": srt_content,
                    "duration_seconds": scene_dur
                })

            logger.info(f"[DATABASE] Saved all scene records for Episode {episode_id}")
            print("DATABASE SAVE")
            print(f"episode_id={episode_id}")
            print(f"scene_ids={[s['scene_id'] for s in scene_assets_list]}")

            # Phase 5: FFmpeg Rendering
            db = SessionLocal()
            try:
                episode = db.query(Episode).filter(Episode.id == episode_id).first()
                episode.duration_seconds = total_duration
                if scene_assets_list and scene_assets_list[0].get("image_url"):
                    episode.thumbnail_url = scene_assets_list[0]["image_url"]
                render_task = db.query(RenderTask).filter(RenderTask.episode_id == episode_id).first()
                render_task.stage = "FFMPEG_RENDERING"
                render_task.progress_percentage = 85
                render_task.current_step_details = "FFmpeg compositing video clips, audio tracks, and burning subtitles..."
                db.commit()
                episode_season = episode.season
                episode_title = episode.title
                episode_summary = episode.summary
            except Exception as e:
                db.rollback()
                raise e
            finally:
                db.close()

            # Generate final video (NO active db session)
            logger.info(f"[MUX] Starting render for Episode {episode_id}...")
            try:
                final_video_url = await render_engine.render_episode_mp4(episode_id=episode_id, scene_assets=scene_assets_list)
                logger.info(f"[MUX] Render successful. Final video: {final_video_url}")
            except Exception as e:
                logger.error(f"[MUX] Render failed: {e}")
                self._set_episode_failed(episode_id, f"FFmpeg rendering failed: {e}")
                return

            # Phase 6: Finalization
            db = SessionLocal()
            try:
                episode = db.query(Episode).filter(Episode.id == episode_id).first()
                episode.final_video_url = final_video_url
                episode.status = "COMPLETED"

                memory_summary = f"Episode {episode_number} ({episode_title}): {episode_summary or episode_logline}"
                
                completed_events = screenplay_data.get("completed_events", [])
                unresolved_events = screenplay_data.get("unresolved_events", [])
                character_states = screenplay_data.get("character_states", {})
                new_locations = screenplay_data.get("new_locations", [])
                new_items = screenplay_data.get("new_items", [])

                mem_id = await memory_engine.add_episode_memory(
                    universe_id=universe_id,
                    episode_number=episode_number,
                    summary=memory_summary,
                    entities_involved=[c["name"] for c in brief["characters"]],
                    completed_events=completed_events,
                    unresolved_events=unresolved_events,
                    character_states=character_states,
                    new_locations=new_locations,
                    new_items=new_items
                )
                
                extended_db_memory = memory_summary
                if unresolved_events:
                    extended_db_memory += f"\nUnresolved: {', '.join(unresolved_events)}"
                if character_states:
                    extended_db_memory += f"\nStates: {json.dumps(character_states)}"

                story_mem = StoryMemory(
                    universe_id=universe_id,
                    episode_number=episode_number,
                    memory_type="EPISODE_RECAP",
                    content=extended_db_memory,
                    entities_involved=[c["name"] for c in brief["characters"]],
                    vector_id=mem_id
                )
                db.add(story_mem)

                for c in brief["characters"]:
                    character_manager.add_memory(
                        character_name=c["name"],
                        universe_id=universe_id,
                        episode_num=episode_number,
                        content=f"Successfully finished Episode {episode_number} ({episode_title}): {episode_logline}."
                    )

                universe = db.query(Universe).filter(Universe.id == universe_id).first()
                if universe:
                    universe.total_episodes += 1

                active_arc = db.query(StoryArc).filter(StoryArc.universe_id == universe_id, StoryArc.status == "ACTIVE").first()
                if active_arc:
                    active_arc.episodes_completed += 1
                    if active_arc.episodes_completed >= active_arc.episodes_planned:
                        active_arc.status = "COMPLETED"

                tl_event = TimelineEvent(
                    universe_id=universe_id,
                    timestamp_in_universe=f"Season {episode_season}, Ep {episode_number}",
                    title=episode_title,
                    description=episode_logline,
                    importance_score=7,
                    season=episode_season,
                    episode_number=episode_number
                )
                db.add(tl_event)

                render_task = db.query(RenderTask).filter(RenderTask.episode_id == episode_id).first()
                render_task.stage = "DONE"
                render_task.progress_percentage = 100
                render_task.current_step_details = "Episode rendered, memories stored, and universe updated."
                render_task.completed_at = datetime.utcnow()

                db.commit()
                logger.info(f"Episode {episode_number} pipeline completed successfully!")
            except Exception as e:
                db.rollback()
                raise e
            finally:
                db.close()

        except Exception as e:
            logger.error(f"Error during episode media generation phase: {e}", exc_info=True)
            self._set_episode_failed(episode_id, str(e))

episode_generator = EpisodeGeneratorModule()
