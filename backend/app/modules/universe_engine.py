from sqlalchemy.orm import Session
import logging
from app.models.domain import Universe, Character, TimelineEvent, StoryArc
from app.engines.llm_engine import llm_engine, LLMUnavailableError

logger = logging.getLogger("universe_engine")

class UniverseEngineModule:
    """
    Core Module 1: Universe Engine.
    Creates and initializes story universe, generates universe bible, character roster, timeline events, and story arcs.
    All content comes from the LLM — no hardcoded data.
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

        # 2. Generate detailed Bible via LLM — NO FALLBACK, up to 3 attempts
        bible_data = None
        last_exc = None
        for attempt in range(1, 4):
            try:
                logger.info("Generating universe bible for '%s' — attempt %d/3", title, attempt)
                bible_data = await llm_engine.generate_universe_bible(title, genre, logline, world_rules)
                chars = bible_data.get("suggested_characters", [])
                factions = bible_data.get("factions", [])
                arcs = bible_data.get("initial_story_arcs", [])
                logger.info(
                    "Universe bible generated: %d characters, %d factions, %d arcs",
                    len(chars) if isinstance(chars, list) else 0,
                    len(factions) if isinstance(factions, list) else 0,
                    len(arcs) if isinstance(arcs, list) else 0,
                )
                # Validate that required arrays are non-empty
                if not isinstance(chars, list) or len(chars) == 0:
                    raise LLMUnavailableError(
                        f"LLM returned bible with no characters (type={type(chars).__name__}). "
                        f"The model may have used different field names. Will retry."
                    )
                break  # Success
            except LLMUnavailableError as exc:
                last_exc = exc
                logger.warning("Bible generation attempt %d failed: %s", attempt, exc)
                bible_data = None
                if attempt == 3:
                    # Roll back the universe record since we can't populate it
                    db.delete(universe)
                    db.commit()
                    logger.error("All 3 bible generation attempts failed for '%s'", title)
                    raise exc  # Will be caught by the API endpoint and returned as HTTP 500

        if bible_data is None:
            db.delete(universe)
            db.commit()
            raise LLMUnavailableError(f"Universe bible generation failed for '{title}': {last_exc}")

        # Normalize array fields — the LLM sometimes returns strings instead of arrays
        for key in ("suggested_characters", "historical_milestones", "initial_story_arcs", "factions"):
            val = bible_data.get(key)
            if not isinstance(val, list):
                logger.warning("LLM returned non-list for '%s' (type=%s), defaulting to empty list", key, type(val).__name__)
                bible_data[key] = []

        universe.lore_bible = bible_data
        db.commit()

        # 3. Create Characters from LLM response
        suggested_chars = bible_data.get("suggested_characters", [])
        for char_info in suggested_chars:
            if not isinstance(char_info, dict):
                continue
            character = Character(
                universe_id=universe.id,
                name=char_info.get("name", "Unnamed Character"),
                role=char_info.get("role", "Supporting"),
                personality=char_info.get("personality", "Complex and layered"),
                appearance_prompt=char_info.get("appearance_prompt", f"Cinematic portrait of {char_info.get('name', 'a character')}, highly detailed 8k"),
                voice_actor_preset=char_info.get("voice_actor_preset", "cinematic voice"),
                bio=char_info.get("bio", "A character in this story universe.")
            )
            db.add(character)

        # 4. Create Initial Timeline Events from LLM response
        historical_milestones = bible_data.get("historical_milestones", [])
        for idx, milestone in enumerate(historical_milestones, 1):
            if isinstance(milestone, dict):
                evt = TimelineEvent(
                    universe_id=universe.id,
                    timestamp_in_universe=milestone.get("era", f"Era {idx}"),
                    title=milestone.get("event", f"Historical Event {idx}"),
                    description=milestone.get("significance", "A pivotal moment in history."),
                    importance_score=8,
                    season=1
                )
            elif isinstance(milestone, str):
                evt = TimelineEvent(
                    universe_id=universe.id,
                    timestamp_in_universe=f"Era {idx}",
                    title=milestone,
                    description=f"Historical milestone: {milestone}",
                    importance_score=8,
                    season=1
                )
            else:
                continue
            db.add(evt)

        # 5. Create Initial Story Arcs from LLM response
        story_arcs = bible_data.get("initial_story_arcs", [])
        for arc_idx, arc_info in enumerate(story_arcs):
            if not isinstance(arc_info, dict):
                continue
            arc = StoryArc(
                universe_id=universe.id,
                title=arc_info.get("title", f"Story Arc {arc_idx + 1}"),
                goal=arc_info.get("goal", "An unfolding narrative journey"),
                status="ACTIVE" if arc_idx == 0 else "PLANNING",
                season=1,
                episodes_planned=arc_info.get("episodes_planned", 5)
            )
            db.add(arc)

        db.commit()
        db.refresh(universe)
        logger.info("Universe created successfully: %s (ID: %s) — %d characters, %d milestones, %d arcs",
                     universe.title, universe.id,
                     len(suggested_chars), len(historical_milestones), len(story_arcs))
        return universe

universe_module = UniverseEngineModule()
