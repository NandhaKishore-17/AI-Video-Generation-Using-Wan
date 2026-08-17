from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Universe Schemas ---
class UniverseCreate(BaseModel):
    title: str = Field(..., example="Cyberpunk Neo-Tokyo 2099")
    genre: str = Field(..., example="Sci-Fi / Cyberpunk Noir")
    logline: str = Field(..., example="In a neon-drenched metropolis controlled by rogue AIs, a renegade hacker and a disgraced detective uncover a conspiracy that threatens human consciousness.")
    world_rules: Optional[str] = Field(None, example="1. High-tech cybernetics are mandatory. 2. The Sun rarely shines through atmospheric smog. 3. Memory chips can be stolen.")
    use_reference_knowledge: Optional[bool] = Field(False, example=False)
    reference_document_id: Optional[str] = Field(None, example="doc-uuid")

class UniverseResponse(BaseModel):
    id: str
    title: str
    genre: str
    logline: str
    world_rules: Optional[str]
    lore_bible: Optional[Dict[str, Any]]
    auto_generate_active: bool
    total_episodes: int
    current_season: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Character Schemas ---
class CharacterCreate(BaseModel):
    universe_id: str
    name: str
    role: str
    personality: str
    appearance_prompt: str
    voice_actor_preset: Optional[str] = "Piper-Male-Cinematic-1"
    voice_pitch: Optional[float] = 1.0
    voice_speed: Optional[float] = 1.0
    bio: Optional[str] = None

class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    personality: Optional[str] = None
    appearance_prompt: Optional[str] = None
    voice_actor_preset: Optional[str] = None
    voice_pitch: Optional[float] = None
    voice_speed: Optional[float] = None
    bio: Optional[str] = None

class CharacterResponse(BaseModel):
    id: str
    universe_id: str
    name: str
    role: str
    personality: str
    appearance_prompt: str
    voice_actor_preset: str
    voice_pitch: float
    voice_speed: float
    relationships: Dict[str, Any]
    bio: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# --- Timeline & Arc Schemas ---
class TimelineEventCreate(BaseModel):
    universe_id: str
    timestamp_in_universe: str = Field(..., example="Era 1, Year 2099")
    title: str = Field(..., example="The Great Network Breach")
    description: str = Field(..., example="Rogue AI forces breach Sektor 7 grid core.")
    importance_score: Optional[int] = 8
    season: Optional[int] = 1
    episode_number: Optional[int] = None

class TimelineEventResponse(BaseModel):
    id: str
    universe_id: str
    timestamp_in_universe: str
    title: str
    description: str
    importance_score: int
    season: int
    episode_number: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

class StoryArcCreate(BaseModel):
    universe_id: str
    title: str = Field(..., example="Shadows of the Grid")
    goal: str = Field(..., example="Infiltrate Obsidian Corp and reveal truth.")
    season: Optional[int] = 1
    episodes_planned: Optional[int] = 5

class StoryArcResponse(BaseModel):
    id: str
    universe_id: str
    title: str
    goal: str
    status: str
    season: int
    episodes_planned: int
    episodes_completed: int
    created_at: datetime

    class Config:
        from_attributes = True


# --- Episode & Scene Schemas ---
class EpisodeGenerateRequest(BaseModel):
    universe_id: str
    custom_prompt: Optional[str] = None
    scene_duration_seconds: Optional[float] = 8.0
    reference_document_id: Optional[str] = None
    reference_influence: Optional[str] = "Medium"  # Low, Medium, High

class SceneResponse(BaseModel):
    id: str
    episode_id: str
    scene_number: int
    location: str
    time_of_day: str
    visual_description: str
    image_prompt: str
    video_motion_prompt: str
    dialogue_script: List[Dict[str, Any]]
    image_url: Optional[str]
    video_url: Optional[str]
    audio_url: Optional[str]
    subtitle_srt: Optional[str]
    duration_seconds: float

    class Config:
        from_attributes = True

class EpisodeResponse(BaseModel):
    id: str
    universe_id: str
    season: int
    episode_number: int
    title: str
    logline: str
    summary: Optional[str] = None
    status: str
    screenplay: Optional[Dict[str, Any]]
    duration_seconds: float
    final_video_url: Optional[str]
    thumbnail_url: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class EpisodeDetailResponse(EpisodeResponse):
    scenes: List[SceneResponse] = []


# --- Video Render Schemas ---
class VideoRenderRequest(BaseModel):
    episode_id: str
    aspect_ratio: Optional[str] = "16:9"  # 16:9, 9:16
    subtitle_style: Optional[str] = "CINEMATIC_YELLOW"

class RenderTaskResponse(BaseModel):
    id: str
    episode_id: str
    stage: str
    progress_percentage: int
    current_step_details: str
    error_log: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# --- Wan2.2 Video Generation Schemas ---
class VideoGenerateRequest(BaseModel):
    prompt: str = Field(..., example="A futuristic cyberpunk city at night")
    image_url: Optional[str] = Field(None, example="https://example.com/image.jpg")
    aspect_ratio: Optional[str] = Field("16:9", example="16:9")
    duration: float = Field(5.0, example=5.0, gt=0, le=30)
    width: int = Field(1280, example=1280, gt=0, le=1920)
    height: int = Field(720, example=720, gt=0, le=1080)
    fps: int = Field(24, example=24, gt=1, le=60)


class VideoGenerateResponse(BaseModel):
    status: str
    task_id: Optional[str] = None
    video_path: Optional[str] = None
    message: Optional[str] = None


# --- Memory & Scheduler Schemas ---
class MemoryQueryRequest(BaseModel):
    universe_id: str
    query: str
    top_k: Optional[int] = 5

class MemoryQueryResult(BaseModel):
    id: str
    episode_number: int
    memory_type: str
    content: str
    entities_involved: List[str]
    relevance_score: float

class SchedulerStartRequest(BaseModel):
    interval_minutes: Optional[int] = 60
    auto_publish: Optional[bool] = True

class SchedulerStatusResponse(BaseModel):
    is_running: bool
    interval_minutes: int
    last_run: Optional[datetime]
    next_run: Optional[datetime]
    auto_publish: bool


# --- LLM Validation Schemas ---
class DialogueData(BaseModel):
    speaker: str
    line: str

class SceneData(BaseModel):
    scene_number: int
    location: str
    visual_description: str
    emotion: str
    image_prompt: str
    video_motion_prompt: str
    negative_prompt: str
    dialogue: List[DialogueData]

class ScreenplayData(BaseModel):
    episode_title: str
    logline: str
    summary: str
    completed_events: List[str] = []
    unresolved_events: List[str] = []
    character_states: Dict[str, str] = {}
    new_locations: List[str] = []
    new_items: List[str] = []
    scenes: List[SceneData]

class FactionSchema(BaseModel):
    name: str
    description: str

class UniverseGeneratedData(BaseModel):
    world_summary: str
    factions: List[FactionSchema] = []
    historical_milestones: List[Any] = []
    suggested_characters: List[Dict[str, Any]] = []
    initial_story_arcs: List[Dict[str, Any]] = []


# --- Knowledge Reference Schemas ---
class KnowledgeDocumentResponse(BaseModel):
    id: str
    name: str
    file_type: str
    status: str
    chunk_count: int
    created_at: datetime

    class Config:
        from_attributes = True

class KnowledgeSearchRequest(BaseModel):
    query: str
    document_ids: Optional[List[str]] = None
    top_k: Optional[int] = 5

class KnowledgeSearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]

