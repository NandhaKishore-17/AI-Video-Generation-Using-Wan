from sqlalchemy.orm import Session
import logging
from typing import List, Dict, Any
from app.models.domain import Universe, StoryArc, StoryMemory
from app.engines.memory_engine import memory_engine

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
        custom_prompt: str = ""
    ) -> Dict[str, Any]:
        universe = db.query(Universe).filter(Universe.id == universe_id).first()
        if not universe:
            raise ValueError(f"Universe with ID {universe_id} not found.")

        next_ep_num = universe.total_episodes + 1

        # Retrieve active story arc
        active_arc = db.query(StoryArc).filter(
            StoryArc.universe_id == universe_id,
            StoryArc.status == "ACTIVE"
        ).first()

        arc_title = active_arc.title if active_arc else "The Unfolding Odyssey"

        # Query relevant past memories
        past_memories_res = await memory_engine.query_relevant_memories(
            universe_id=universe_id,
            query=custom_prompt or arc_title,
            top_k=3
        )
        memories_text = [m["content"] for m in past_memories_res]

        # Extract character roster
        characters_data = [
            {
                "id": c.id,
                "name": c.name,
                "role": c.role,
                "personality": c.personality,
                "appearance_prompt": c.appearance_prompt,
                "voice_actor_preset": c.voice_actor_preset
            }
            for c in universe.characters
        ]

        brief = {
            "universe_id": universe.id,
            "universe_title": universe.title,
            "universe_genre": universe.genre,
            "episode_number": next_ep_num,
            "season": universe.current_season,
            "active_arc": arc_title,
            "characters": characters_data,
            "past_memories": memories_text,
            "custom_prompt": custom_prompt
        }

        logger.info(f"Story Director brief created for Episode {next_ep_num} of {universe.title}")
        return brief

story_director = StoryDirectorModule()
