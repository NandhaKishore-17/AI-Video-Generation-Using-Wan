from sqlalchemy.orm import Session
import logging
from typing import Any
from app.models.domain import Universe, Character, TimelineEvent, StoryArc
from app.engines.llm_engine import llm_engine
from app.services.knowledge_service import knowledge_service

logger = logging.getLogger("universe_engine")

class UniverseEngineModule:
    """
    Core Module 1: Universe Engine.
    Creates and initializes story universe, generates universe bible, character roster, timeline events, and story arcs.
    """

    def _normalize_timeline_event(self, event: Any, default_idx: int) -> tuple[str, str, str]:
        """Safely extracts title, timestamp, and description from any type returned by LLM."""
        default_title = f"Historical Event {default_idx}"
        default_timestamp = f"Era 1, Year {2000 + default_idx*10}"
        default_desc = f"Historical milestone event: {event}"

        if not event:
            return default_title, default_timestamp, default_desc

        if isinstance(event, dict):
            title = event.get("event") or event.get("title") or event.get("name") or default_title
            timestamp = event.get("date") or event.get("timestamp") or event.get("time") or default_timestamp
            desc = event.get("description") or event.get("details") or f"Historical milestone event: {title}"
            return str(title), str(timestamp), str(desc)
        elif isinstance(event, str):
            return event, default_timestamp, f"Historical milestone event: {event}"
        else:
            # Fallback for unexpected primitives (int, float, list, etc)
            str_val = str(event)
            return str_val, default_timestamp, f"Historical milestone event: {str_val}"

    async def create_full_universe(
        self,
        db: Session,
        title: str,
        genre: str,
        logline: str,
        world_rules: str = "",
        use_reference_knowledge: bool = False,
        reference_document_id: str = None
    ) -> Universe:
        try:
            # 1. Create base Universe record
            universe = Universe(
                title=title,
                genre=genre,
                logline=logline,
                world_rules=world_rules,
                auto_generate_active=True
            )
            db.add(universe)
            db.flush()  # Populate universe.id without committing

            # 2. Build Optional RAG Context
            rag_context = None
            if use_reference_knowledge and reference_document_id:
                try:
                    user_prompt = f"{title}\n{genre}\n{logline}\n{world_rules}"
                    retrieved_chunks = knowledge_service.retrieve_relevant_themes(
                        query=user_prompt,
                        document_ids=[reference_document_id],
                        top_k=5
                    )
                    
                    if retrieved_chunks:
                        context_parts = []
                        context_parts.append("REFERENCE CONTEXT")
                        
                        source_name = retrieved_chunks[0].get("metadata", {}).get("document_name", "Reference Document")
                        context_parts.append(f"Source: {source_name}")
                        
                        for i, chunk in enumerate(retrieved_chunks, 1):
                            context_parts.append(f"\nRelevant Passage {i}:\n{chunk.get('text', '')}")
                        
                        context_parts.append("\nEND REFERENCE CONTEXT")
                        rag_context = "\n".join(context_parts)
                        logger.info(f"RAG Context successfully built for {title} (Retrieved {len(retrieved_chunks)} chunks)")
                    else:
                        logger.warning(f"RAG was enabled for {title} but no relevant chunks were retrieved.")
                except Exception as e:
                    logger.error(f"Failed to retrieve reference knowledge: {e}")
                    # Fail open: proceed without context if retrieval fails but continue logging

            # 3. Generate detailed Bible via Qwen LLM Engine
            bible_data = await llm_engine.generate_universe_bible(title, genre, logline, world_rules, rag_context)
            
            # Since the LLM returns Pydantic models in our updated llm_engine, we need to access via attributes.
            # Wait, our llm_engine returns a dictionary because _generate_with_retry ends with `return data`.
            # Let's ensure it handles both dict and Pydantic object if it was changed.
            # In our llm_engine._generate_with_retry we return `data` which is a dict.
            universe.lore_bible = bible_data

            # 3. Create Characters
            suggested_chars = bible_data.get("suggested_characters", [])
            for char_info in suggested_chars:
                character = Character(
                    universe_id=universe.id,
                    name=char_info.get("name", "Unknown Hero"),
                    role=char_info.get("role", "Protagonist"),
                    personality=char_info.get("personality", "Brave and resilient"),
                    appearance_prompt=char_info.get("appearance_prompt", f"Cinematic photo of {char_info.get('name')}, highly detailed 8k"),
                    voice_actor_preset=char_info.get("voice_actor_preset", "Piper-Male-Cinematic-1"),
                    bio=char_info.get("bio", "Hero of the universe")
                )
                db.add(character)

            # 4. Create Initial Timeline Events
            historical_milestones = bible_data.get("historical_milestones", ["Universe Founded", "The Great Event"])
            for idx, milestone in enumerate(historical_milestones, 1):
                t_title, t_timestamp, t_desc = self._normalize_timeline_event(milestone, idx)
                evt = TimelineEvent(
                    universe_id=universe.id,
                    timestamp_in_universe=t_timestamp,
                    title=t_title,
                    description=t_desc,
                    importance_score=8,
                    season=1
                )
                db.add(evt)

            # 5. Create Initial Story Arcs
            story_arcs = bible_data.get("initial_story_arcs", [])
            for arc_info in story_arcs:
                arc = StoryArc(
                    universe_id=universe.id,
                    title=arc_info.get("title", "Beginning of the Journey"),
                    goal=arc_info.get("goal", "Establish peace in the galaxy"),
                    status="ACTIVE" if arc_info == story_arcs[0] else "PLANNING",
                    season=1,
                    episodes_planned=arc_info.get("episodes_planned", 5)
                )
                db.add(arc)

            db.commit()
            db.refresh(universe)
            logger.info(f"Universe created successfully: {universe.title} (ID: {universe.id})")
            return universe
        
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create full universe: {e}")
            raise e

universe_module = UniverseEngineModule()
