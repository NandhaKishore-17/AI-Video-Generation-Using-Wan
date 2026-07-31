from sqlalchemy.orm import Session
import logging
from app.models.domain import Universe, Character, TimelineEvent, StoryArc
from app.engines.llm_engine import llm_engine

logger = logging.getLogger("universe_engine")

class UniverseEngineModule:
    """
    Core Module 1: Universe Engine.
    Creates and initializes story universe, generates universe bible, character roster, timeline events, and story arcs.
    """

    async def create_full_universe(
        self,
        db: Session,
        title: str,
        genre: str,
        logline: str,
        world_rules: str = ""
    ) -> Universe:
        # 1. Create base Universe record
        universe = Universe(
            title=title,
            genre=genre,
            logline=logline,
            world_rules=world_rules,
            auto_generate_active=True
        )
        db.add(universe)
        db.commit()
        db.refresh(universe)

        # 2. Generate detailed Bible via Qwen LLM Engine
        bible_data = await llm_engine.generate_universe_bible(title, genre, logline, world_rules)
        universe.lore_bible = bible_data
        db.commit()

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
            evt = TimelineEvent(
                universe_id=universe.id,
                timestamp_in_universe=f"Era 1, Year {2000 + idx*10}",
                title=milestone,
                description=f"Historical milestone event: {milestone}",
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

universe_module = UniverseEngineModule()
