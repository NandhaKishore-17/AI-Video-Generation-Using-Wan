export interface Universe {
  id: string;
  title: string;
  genre: string;
  logline: string;
  world_rules?: string;
  lore_bible?: Record<string, any>;
  auto_generate_active: boolean;
  total_episodes: number;
  current_season: number;
  created_at: string;
  updated_at: string;
}

export interface Character {
  id: string;
  universe_id: string;
  name: string;
  role: string;
  personality: string;
  appearance_prompt: string;
  voice_actor_preset: string;
  voice_pitch: number;
  voice_speed: number;
  relationships: Record<string, any>;
  bio?: string;
  created_at: string;
}

export interface TimelineEvent {
  id: string;
  timestamp: string;
  title: string;
  description: string;
  importance_score: number;
  season: number;
  episode_number?: number;
}

export interface StoryArc {
  id: string;
  title: string;
  goal: string;
  status: 'PLANNING' | 'ACTIVE' | 'COMPLETED';
  season: number;
  episodes_planned: number;
  episodes_completed: number;
}

export interface Scene {
  id: string;
  episode_id: string;
  scene_number: number;
  location: string;
  time_of_day: string;
  visual_description: string;
  image_prompt: string;
  video_motion_prompt: string;
  dialogue_script: Array<{ speaker: string; line: string }>;
  image_url?: string;
  video_url?: string;
  audio_url?: string;
  subtitle_srt?: string;
  duration_seconds: number;
}

export interface Episode {
  id: string;
  universe_id: string;
  season: number;
  episode_number: number;
  title: string;
  logline: string;
  status: 'DRAFT' | 'GENERATING' | 'RENDERING' | 'COMPLETED' | 'FAILED';
  screenplay?: Record<string, any>;
  duration_seconds: number;
  final_video_url?: string;
  thumbnail_url?: string;
  created_at: string;
  updated_at: string;
  scenes?: Scene[];
}

export interface RenderTask {
  id: string;
  episode_id: string;
  stage: string;
  progress_percentage: number;
  current_step_details: string;
  error_log?: string;
  started_at: string;
  completed_at?: string;
}

export interface SchedulerStatus {
  is_running: boolean;
  interval_minutes: number;
  last_run?: string;
  next_run?: string;
  auto_publish: boolean;
}

export interface MemoryQueryResult {
  id: string;
  episode_number: number;
  memory_type: string;
  content: string;
  entities_involved: string[];
  relevance_score: number;
}

export interface AnalyticsData {
  total_universes: number;
  total_episodes: number;
  total_characters: number;
  completed_renders: number;
  engine_benchmarks: {
    llm_provider?: string;
    llm_model?: string;
    image_provider?: string;
    video_provider?: string;
    voice_provider?: string;
  };
  system_status: {
    llm_api_base?: string;
    ollama_model?: string;
  };
}
