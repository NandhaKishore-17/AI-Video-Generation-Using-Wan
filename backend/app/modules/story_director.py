from sqlalchemy.orm import Session
import logging
import json
from typing import List, Dict, Any
from app.models.domain import Universe, StoryArc, Episode, StoryMemory
from app.engines.memory_engine import memory_engine
from app.services.knowledge_service import knowledge_service

logger = logging.getLogger("story_director")

class StoryDirectorModule:
    """
    Core Module 3: Story Director & Continuity Guardian.
    Analyzes historical memory, ensures narrative consistency, tracks character arcs,
    and determines the plot direction for the next episode.
    """

    async def prepare_next_episode_brief(
        self,
        db: Session,
        universe_id: str,
        custom_prompt: str = "",
        reference_document_id: str = None,
        reference_influence: str = "Medium"
    ) -> Dict[str, Any]:
        from sqlalchemy import func
        universe = db.query(Universe).filter(Universe.id == universe_id).first()
        if not universe:
            raise ValueError(f"Universe with ID {universe_id} not found.")

        max_ep = db.query(func.max(Episode.episode_number)).filter(Episode.universe_id == universe_id).scalar()
        next_ep_num = (max_ep or 0) + 1

        # Retrieve active story arc
        active_arc = db.query(StoryArc).filter(
            StoryArc.universe_id == universe_id,
            StoryArc.status == "ACTIVE"
        ).first()

        if active_arc:
            arc_title = f"{active_arc.title}: {active_arc.goal}"
        else:
            arc_title = "The Unfolding Odyssey"

        lore_bible_summary = ""
        if universe.lore_bible:
            try:
                lore_bible_summary = json.dumps(universe.lore_bible, indent=2)
            except Exception:
                lore_bible_summary = str(universe.lore_bible)

        # Build concise history from stored episode summaries
        previous_episodes = db.query(Episode).filter(
            Episode.universe_id == universe_id,
            Episode.episode_number < next_ep_num
        ).order_by(Episode.season, Episode.episode_number).all()

        previous_summaries = []
        for ep in previous_episodes:
            if ep.summary:
                previous_summaries.append(f"Episode {ep.episode_number}: {ep.title} — {ep.summary}")
            else:
                previous_summaries.append(f"Episode {ep.episode_number}: {ep.title} — {ep.logline}")

        # Enhance the most recent episode summary with its extended StoryMemory (states, unresolved events)
        if previous_episodes:
            last_ep = previous_episodes[-1]
            last_ep_memory = db.query(StoryMemory).filter(
                StoryMemory.universe_id == universe_id,
                StoryMemory.episode_number == last_ep.episode_number,
                StoryMemory.memory_type == "EPISODE_RECAP"
            ).first()
            if last_ep_memory:
                previous_summaries[-1] = f"Episode {last_ep.episode_number} (DETAILED MEMORY): {last_ep_memory.content}"

        if not previous_summaries:
            previous_summaries = ["No prior episodes have been generated yet."]

        # Query relevant past memories
        past_memories_res = await memory_engine.query_relevant_memories(
            universe_id=universe_id,
            query=custom_prompt or arc_title,
            top_k=5
        )
        memories_text = [m["content"] for m in past_memories_res]

        # Extract character roster
        characters_data = []
        for c in universe.characters:
            characters_data.append({
                "id": c.id,
                "name": c.name,
                "role": c.role,
                "personality": c.personality,
                "appearance_prompt": c.appearance_prompt,
                "voice_actor_preset": c.voice_actor_preset
            })
            
        # Fallback to lore_bible if DB characters are empty
        if not characters_data and universe.lore_bible and isinstance(universe.lore_bible, dict):
            suggested = universe.lore_bible.get("suggested_characters", [])
            for c in suggested:
                characters_data.append({
                    "id": "fallback",
                    "name": c.get("name", "Unknown Character"),
                    "role": c.get("role", "Protagonist"),
                    "personality": c.get("personality", ""),
                    "appearance_prompt": c.get("appearance_prompt", ""),
                    "voice_actor_preset": c.get("voice_actor_preset", "")
                })

        # Process Reference RAG if requested
        reference_themes = []
        if reference_document_id:
            logger.info(f"Retrieving reference themes from document {reference_document_id}")
            # The top_k can vary based on influence level
            top_k = 3 if reference_influence == "Low" else (5 if reference_influence == "Medium" else 8)
            search_query = custom_prompt or arc_title
            
            theme_results = knowledge_service.retrieve_relevant_themes(
                query=search_query,
                document_ids=[reference_document_id],
                top_k=top_k
            )
            for res in theme_results:
                reference_themes.append(f"Theme Context ({res['metadata'].get('category', 'THEME')}): {res['text']}")

        brief = {
            "universe_id": universe.id,
            "universe_title": universe.title,
            "universe_genre": universe.genre,
            "episode_number": next_ep_num,
            "season": universe.current_season,
            "active_arc": arc_title,
            "characters": characters_data,
            "past_memories": memories_text,
            "previous_episode_summaries": previous_summaries,
            "universe_lore": lore_bible_summary,
            "custom_prompt": custom_prompt,
            "reference_themes": reference_themes,
            "reference_influence": reference_influence
        }

        logger.info(f"Story Director brief created for Episode {next_ep_num} of {universe.title}")
        return brief

story_director = StoryDirectorModule()
