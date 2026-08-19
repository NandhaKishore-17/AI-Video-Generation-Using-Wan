from sqlalchemy import Column, String, Integer, Float, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from app.core.database import Base

def generate_uuid():
    return str(uuid.uuid4())

class Universe(Base):
    __tablename__ = "universes"

    id = Column(String, primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False)
    genre = Column(String(100), nullable=False)
    logline = Column(Text, nullable=False)
    world_rules = Column(Text, nullable=True)
    lore_bible = Column(JSON, nullable=True)  # Detailed world background, factions, history
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Continuous Generation state
    auto_generate_active = Column(Boolean, default=True)
    total_episodes = Column(Integer, default=0)
    current_season = Column(Integer, default=1)

    characters = relationship("Character", back_populates="universe", cascade="all, delete-orphan")
    timeline_events = relationship("TimelineEvent", back_populates="universe", cascade="all, delete-orphan")
    story_arcs = relationship("StoryArc", back_populates="universe", cascade="all, delete-orphan")
    episodes = relationship("Episode", back_populates="universe", cascade="all, delete-orphan")
    memories = relationship("StoryMemory", back_populates="universe", cascade="all, delete-orphan")


class Character(Base):
    __tablename__ = "characters"

    id = Column(String, primary_key=True, default=generate_uuid)
    universe_id = Column(String, ForeignKey("universes.id"), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(String(100), nullable=False)  # Protagonist, Antagonist, Sidekick, Mentor
    personality = Column(Text, nullable=False)
    appearance_prompt = Column(Text, nullable=False)  # Consistency visual anchor for FLUX / SDXL
    voice_actor_preset = Column(String(100), default="Piper-Male-Cinematic-1")
    voice_pitch = Column(Float, default=1.0)
    voice_speed = Column(Float, default=1.0)
    relationships = Column(JSON, default=dict)  # {"Char_ID": "Ally", "Char_ID_2": "Rival"}
    bio = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    universe = relationship("Universe", back_populates="characters")


class TimelineEvent(Base):
    __tablename__ = "timeline_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    universe_id = Column(String, ForeignKey("universes.id"), nullable=False)
    timestamp_in_universe = Column(String(100), nullable=False)  # e.g., "Year 2099, Month 4"
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    importance_score = Column(Integer, default=5)  # 1 to 10
    season = Column(Integer, default=1)
    episode_number = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    universe = relationship("Universe", back_populates="timeline_events")


class StoryArc(Base):
    __tablename__ = "story_arcs"

    id = Column(String, primary_key=True, default=generate_uuid)
    universe_id = Column(String, ForeignKey("universes.id"), nullable=False)
    title = Column(String(255), nullable=False)
    goal = Column(Text, nullable=False)
    status = Column(String(50), default="PLANNING")  # PLANNING, ACTIVE, COMPLETED, RESOLVED
    season = Column(Integer, default=1)
    episodes_planned = Column(Integer, default=5)
    episodes_completed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    universe = relationship("Universe", back_populates="story_arcs")


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(String, primary_key=True, default=generate_uuid)
    universe_id = Column(String, ForeignKey("universes.id"), nullable=False)
    season = Column(Integer, default=1)
    episode_number = Column(Integer, nullable=False)
    title = Column(String(255), nullable=False)
    logline = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    status = Column(String(50), default="DRAFT")  # DRAFT, GENERATING, READY, RENDERING, COMPLETED, FAILED
    screenplay = Column(JSON, nullable=True)  # Complete JSON breakdown of screenplay scenes
    duration_seconds = Column(Float, default=0.0)
    final_video_url = Column(String(500), nullable=True)
    thumbnail_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    universe = relationship("Universe", back_populates="episodes")
    scenes = relationship("Scene", back_populates="episode", cascade="all, delete-orphan")
    render_tasks = relationship("RenderTask", back_populates="episode", cascade="all, delete-orphan")


class Scene(Base):
    __tablename__ = "scenes"

    id = Column(String, primary_key=True, default=generate_uuid)
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    scene_number = Column(Integer, nullable=False)
    location = Column(String(255), nullable=False)
    time_of_day = Column(String(100), default="DAY")
    visual_description = Column(Text, nullable=False)
    image_prompt = Column(Text, nullable=False)
    video_motion_prompt = Column(Text, nullable=False)
    dialogue_script = Column(JSON, default=list)  # [{"speaker": "A", "line": "...", "audio_url": "..."}]
    image_url = Column(String(500), nullable=True)
    video_url = Column(String(500), nullable=True)
    audio_url = Column(String(500), nullable=True)
    subtitle_srt = Column(Text, nullable=True)
    duration_seconds = Column(Float, default=5.0)

    episode = relationship("Episode", back_populates="scenes")


class StoryMemory(Base):
    __tablename__ = "story_memories"

    id = Column(String, primary_key=True, default=generate_uuid)
    universe_id = Column(String, ForeignKey("universes.id"), nullable=False)
    episode_number = Column(Integer, nullable=False)
    memory_type = Column(String(50), default="EPISODE_RECAP")  # EVENT, RELATIONSHIP_CHANGE, REVEAL, DEATH
    content = Column(Text, nullable=False)
    entities_involved = Column(JSON, default=list)
    vector_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    universe = relationship("Universe", back_populates="memories")


class RenderTask(Base):
    __tablename__ = "render_tasks"

    id = Column(String, primary_key=True, default=generate_uuid)
    episode_id = Column(String, ForeignKey("episodes.id"), nullable=False)
    stage = Column(String(100), default="QUEUED")  # QUEUED, SCREENPLAY, IMAGES, VIDEOS, AUDIO, SUBTITLES, FFMPEG, DONE, FAILED
    progress_percentage = Column(Integer, default=0)
    current_step_details = Column(String(255), default="Initializing engine...")
    error_log = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    episode = relationship("Episode", back_populates="render_tasks")


class SchedulerConfig(Base):
    __tablename__ = "scheduler_config"

    id = Column(String, primary_key=True, default="default")
    is_running = Column(Boolean, default=False)
    interval_minutes = Column(Integer, default=60)
    last_run = Column(DateTime, nullable=True)
    next_run = Column(DateTime, nullable=True)
    auto_publish = Column(Boolean, default=True)
    mode = Column(String(20), default="interval")  # "interval" or "continuous"
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    status = Column(String(50), default="PROCESSING")  # UPLOADED, PROCESSING, COMPLETED, FAILED
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("KnowledgeChunk", back_populates="document", cascade="all, delete-orphan")


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(String, primary_key=True, default=generate_uuid)
    document_id = Column(String, ForeignKey("knowledge_documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)  # THEME, CHARACTER_ARCHETYPE, etc.
    metadata_json = Column(JSON, nullable=True)
    
    document = relationship("KnowledgeDocument", back_populates="chunks")

