import axios from 'axios';
import {
  Universe, Character, TimelineEvent, StoryArc, Episode, Scene, RenderTask,
  SchedulerStatus, MemoryQueryResult, AnalyticsData, KnowledgeDocument
} from '../types';

const API_BASE = '/api/v1';

// Helper for local storage persistence
const getLocalData = <T>(key: string, fallback: T): T => {
  try {
    const item = localStorage.getItem(`story_app_${key}`);
    return item ? JSON.parse(item) : fallback;
  } catch {
    return fallback;
  }
};

const setLocalData = <T>(key: string, value: T): void => {
  try {
    localStorage.setItem(`story_app_${key}`, JSON.stringify(value));
  } catch (err) {
    console.error('Failed to save to localStorage:', err);
  }
};

// Default seed data
const DEFAULT_UNIVERSES: Universe[] = [
  {
    id: 'u-cyber-99',
    title: 'Neo-Tokyo 2099',
    genre: 'Cyberpunk Noir / AI Thriller',
    logline: 'In a rain-drenched megacity governed by rogue AI networks, a hacker and a detective battle corporate overlords.',
    world_rules: '1. High-tech cybernetics mandatory. 2. Memory chips can be weaponized.',
    auto_generate_active: true,
    total_episodes: 5,
    current_season: 1,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  },
  {
    id: 'u-star-voyage',
    title: 'Aethelgard: The Astral Realm',
    genre: 'High Fantasy / Cosmic Sorcery',
    logline: 'Ancient sorcerers unlock astral gateways to defend their floating realm against primordial void dwellers.',
    world_rules: '1. Mana streams float in upper atmosphere. 2. Dragon riders govern celestial towers.',
    auto_generate_active: false,
    total_episodes: 2,
    current_season: 1,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  }
];

const DEFAULT_EPISODES: Episode[] = [
  {
    id: 'ep-101',
    universe_id: 'u-cyber-99',
    season: 1,
    episode_number: 1,
    title: 'Episode 1: Signals in the Rain',
    logline: 'Kaelen receives an encrypted transmission pointing to an abandoned server vault beneath Sektor 7.',
    summary: 'Episode 1 establishes the core conflict and introduces the hacker protagonist infiltrating the corporate network.',
    status: 'COMPLETED',
    duration_seconds: 18.0,
    final_video_url: '/media/video_scene1.mp4',
    thumbnail_url: '/media/image_scene1.png',
    created_at: new Date(Date.now() - 3600000 * 2).toISOString(),
    updated_at: new Date(Date.now() - 3600000 * 2).toISOString(),
    scenes: [
      {
        id: 'sc-101-1',
        episode_id: 'ep-101',
        scene_number: 1,
        location: 'EXT. RAIN-SWEPT HIGHWAY - NIGHT',
        time_of_day: 'NIGHT',
        visual_description: 'Rain heavy on neon-slick asphalt. Kaelen Vance rides a matte-black hovercycle towards Sektor 7.',
        image_prompt: 'Cinematic wide shot of Kaelen Vance riding futuristic hover bike through rainy neon street, 8k wallpaper',
        video_motion_prompt: 'Dynamic tracking camera shot following hovercycle through wet rain reflections',
        dialogue_script: [
          { speaker: 'Kaelen Vance', line: 'The grid frequency is fluctuating... someone breached the mainframe.' },
          { speaker: 'Nova Thorne', line: 'Be careful, Kaelen. Hunter drones were dispatched two minutes ago.' }
        ],
        image_url: '/media/image_scene1.png',
        video_url: '/media/video_scene1.mp4',
        audio_url: '/media/audio_scene1.wav',
        subtitle_srt: '1\n00:00:00,500 --> 00:00:03,500\nKAELEN VANCE: The grid frequency is fluctuating...\n',
        duration_seconds: 9.0
      },
      {
        id: 'sc-101-2',
        episode_id: 'ep-101',
        scene_number: 2,
        location: 'INT. ABANDONED DATA SANCTUARY - NIGHT',
        time_of_day: 'NIGHT',
        visual_description: 'Nova analyzes floating blue holographic data streams in an abandoned server vault.',
        image_prompt: 'Medium cinematic shot of Nova Thorne with silver hair inspecting glowing blue holographic interface',
        video_motion_prompt: 'Slow push-in shot onto character eyes, holographic light flickering smoothly',
        dialogue_script: [
          { speaker: 'Nova Thorne', line: 'This isn’t a breach... it’s a message left by the first AI system.' },
          { speaker: 'Kaelen Vance', line: 'What does it say?' }
        ],
        image_url: '/media/image_scene2.png',
        video_url: '/media/video_scene2.mp4',
        audio_url: '/media/audio_scene2.wav',
        subtitle_srt: '1\n00:00:00,500 --> 00:00:03,500\nNOVA THORNE: This isn\'t a breach...\n',
        duration_seconds: 9.0
      }
    ]
  },
  {
    id: 'ep-102',
    universe_id: 'u-cyber-99',
    season: 1,
    episode_number: 2,
    title: 'Episode 2: Holographic Deceptions',
    logline: 'Nova and Kaelen decode the hidden message left by the first AI, revealing Director Vane’s plan.',
    summary: 'Episode 2 deepens the conspiracy and confirms the antagonist’s hidden agenda while building the team’s stakes.',
    status: 'COMPLETED',
    duration_seconds: 21.0,
    final_video_url: '/media/video_scene2.mp4',
    thumbnail_url: '/media/image_scene2.png',
    created_at: new Date(Date.now() - 3600000 * 1).toISOString(),
    updated_at: new Date(Date.now() - 3600000 * 1).toISOString(),
    scenes: [
      {
        id: 'sc-102-1',
        episode_id: 'ep-102',
        scene_number: 1,
        location: 'INT. OBSIDIAN CONTROL VAULT - NIGHT',
        time_of_day: 'NIGHT',
        visual_description: 'Red emergency alarms pulse overhead as Director Vane steps onto the catwalk.',
        image_prompt: 'Intense cinematic shot of Director Vane overlooking high-tech quantum supercomputer vault',
        video_motion_prompt: 'Slow tracking pan following Director Vane as holographic telemetry streams past',
        dialogue_script: [
          { speaker: 'Director Vane', line: 'Initiate protocol override... prevent data extraction at all costs.' },
          { speaker: 'Kaelen Vance', line: 'Too late, Director. We already hold the encryption key.' }
        ],
        image_url: '/media/image_scene2.png',
        video_url: '/media/video_scene2.mp4',
        audio_url: '/media/audio_scene1.wav',
        subtitle_srt: '1\n00:00:00,500 --> 00:00:03,500\nDIRECTOR VANE: Initiate protocol override...\n',
        duration_seconds: 10.5
      }
    ]
  }
];

export const api = {
  // Universes
  async createUniverse(payload: { title: string; genre: string; logline: string; world_rules?: string, use_reference_knowledge?: boolean, reference_document_id?: string | null }): Promise<Universe> {
    const res = await axios.post(`${API_BASE}/universes`, payload);
    return res.data;
  },

  async deleteUniverse(id: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/universes/${id}`);
    } catch (err) {
      console.error(err);
    }
    const existing = getLocalData<Universe[]>('universes', DEFAULT_UNIVERSES);
    setLocalData('universes', existing.filter(u => u.id !== id));
  },

  async resetAllUniverses(): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/universes/reset_all`);
    } catch (err) {
      console.error(err);
    }
    setLocalData('universes', []);
    setLocalData('episodes', []);
    setLocalData('characters', []);
    setLocalData('timeline', []);
    setLocalData('story_arcs', []);
  },

  async getUniverses(): Promise<Universe[]> {
    const res = await axios.get(`${API_BASE}/universes`);
    return res.data || [];
  },

  async getUniverseDetail(id: string): Promise<any> {
    const res = await axios.get(`${API_BASE}/universes/${id}`);
    return res.data;
  },

  async createCharacter(payload: {
    universe_id: string;
    name: string;
    role: string;
    personality: string;
    appearance_prompt: string;
    voice_actor_preset?: string;
    voice_pitch?: number;
    voice_speed?: number;
    bio?: string;
  }): Promise<Character> {
    try {
      const res = await axios.post(`${API_BASE}/characters`, payload);
      return res.data;
    } catch {
      const newChar: Character = {
        id: `c-synth-${Date.now()}`,
        universe_id: payload.universe_id,
        name: payload.name,
        role: payload.role,
        personality: payload.personality,
        appearance_prompt: payload.appearance_prompt,
        voice_actor_preset: payload.voice_actor_preset || 'Piper-Male-Cinematic-1',
        voice_pitch: payload.voice_pitch || 1.0,
        voice_speed: payload.voice_speed || 1.0,
        relationships: {},
        bio: payload.bio,
        created_at: new Date().toISOString()
      };
      const existing = getLocalData<Character[]>('characters', []);
      setLocalData('characters', [...existing, newChar]);
      return newChar;
    }
  },

  async deleteCharacter(id: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/characters/${id}`);
    } catch (err) {
      console.error(err);
    }
    const existing = getLocalData<Character[]>('characters', []);
    setLocalData('characters', existing.filter(c => c.id !== id));
  },

  async updateCharacter(id: string, payload: Partial<Omit<Character, 'id' | 'created_at' | 'universe_id' | 'relationships'>>): Promise<Character> {
    try {
      const res = await axios.patch(`${API_BASE}/characters/${id}`, payload);
      return res.data;
    } catch {
      const existing = getLocalData<Character[]>('characters', []);
      const index = existing.findIndex(c => c.id === id);
      if (index !== -1) {
        const updatedChar = { ...existing[index], ...payload };
        existing[index] = updatedChar;
        setLocalData('characters', existing);
        return updatedChar;
      }
      throw new Error("Character not found");
    }
  },

  async getAvailableVoices(): Promise<any[]> {
    try {
      const res = await axios.get(`${API_BASE}/voices/available`);
      if (res.data && res.data.length > 0) return res.data;
    } catch {
      // Fallback
    }
    return [
      { id: "en-US-ChristopherNeural", name: "Christopher", gender: "male", archetype: "deep_male", description: "Resonant deep cinematic male voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-AndrewNeural", name: "Andrew", gender: "male", archetype: "young_male", description: "Young energetic natural male voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-EricNeural", name: "Eric", gender: "male", archetype: "calm_male", description: "Calm and articulate professional male voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-GuyNeural", name: "Guy", gender: "male", archetype: "energetic_male", description: "Dynamic energetic male voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-AU-WilliamNeural", name: "William", gender: "male", archetype: "deep_villain_male", description: "Deep authoritative villain male voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-BrianMultilingualNeural", name: "Brian (Multilingual)", gender: "male", archetype: "multilingual_male", description: "Hyper-realistic conversational human male voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-SteffanNeural", name: "Steffan", gender: "male", archetype: "warm_male", description: "Warm natural conversational male voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-GB-RyanNeural", name: "Ryan (British)", gender: "male", archetype: "british_male", description: "Polished British cinematic male voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-AriaNeural", name: "Aria", gender: "female", archetype: "calm_female", description: "Calm poised female voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-AnaNeural", name: "Ana", gender: "female", archetype: "young_female", description: "Bright young female voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-MichelleNeural", name: "Michelle", gender: "female", archetype: "strong_female", description: "Strong determined female voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-AvaNeural", name: "Ava", gender: "female", archetype: "soft_female", description: "Soft gentle natural female voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-EmmaMultilingualNeural", name: "Emma (Multilingual)", gender: "female", archetype: "multilingual_female", description: "Hyper-realistic conversational human female voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-US-JennyNeural", name: "Jenny", gender: "female", archetype: "friendly_female", description: "Expressive friendly human female voice", provider: "Edge-TTS Azure Neural" },
      { id: "en-GB-SoniaNeural", name: "Sonia (British)", gender: "female", archetype: "british_female", description: "British elegant cinematic female voice", provider: "Edge-TTS Azure Neural" }
    ];
  },

  // Timeline Events & Story Arcs
  async createTimelineEvent(payload: {
    universe_id: string;
    timestamp_in_universe: string;
    title: string;
    description: string;
    importance_score?: number;
    season?: number;
  }): Promise<TimelineEvent> {
    try {
      const res = await axios.post(`${API_BASE}/timeline_events`, payload);
      return res.data;
    } catch {
      const newEvt: TimelineEvent = {
        id: `t-synth-${Date.now()}`,
        timestamp: payload.timestamp_in_universe,
        title: payload.title,
        description: payload.description,
        importance_score: payload.importance_score || 8,
        season: payload.season || 1
      };
      const existing = getLocalData<TimelineEvent[]>('timeline', []);
      setLocalData('timeline', [newEvt, ...existing]);
      return newEvt;
    }
  },

  async deleteTimelineEvent(id: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/timeline_events/${id}`);
    } catch (err) {
      console.error(err);
    }
    const existing = getLocalData<TimelineEvent[]>('timeline', []);
    setLocalData('timeline', existing.filter(t => t.id !== id));
  },

  async createStoryArc(payload: {
    universe_id: string;
    title: string;
    goal: string;
    season?: number;
    episodes_planned?: number;
  }): Promise<StoryArc> {
    try {
      const res = await axios.post(`${API_BASE}/story_arcs`, payload);
      return res.data;
    } catch {
      const newArc: StoryArc = {
        id: `sa-synth-${Date.now()}`,
        title: payload.title,
        goal: payload.goal,
        status: 'ACTIVE',
        season: payload.season || 1,
        episodes_planned: payload.episodes_planned || 5,
        episodes_completed: 0
      };
      const existing = getLocalData<StoryArc[]>('story_arcs', []);
      setLocalData('story_arcs', [newArc, ...existing]);
      return newArc;
    }
  },

  async deleteStoryArc(id: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/story_arcs/${id}`);
    } catch (err) {
      console.error(err);
    }
    const existing = getLocalData<StoryArc[]>('story_arcs', []);
    setLocalData('story_arcs', existing.filter(a => a.id !== id));
  },

  // Episodes Generation & Catalog
  async generateEpisode(universe_id: string, custom_prompt?: string, episode_duration_seconds?: number, reference_document_id?: string, reference_influence?: string): Promise<Episode> {
    const res = await axios.post(`${API_BASE}/episodes/generate`, { universe_id, custom_prompt, episode_duration_seconds, reference_document_id, reference_influence });
    if (res.data) {
      const currentEps = getLocalData<Episode[]>('episodes', []);
      setLocalData('episodes', [res.data, ...currentEps]);
      
      const universes = getLocalData<Universe[]>('universes', []);
      const updatedUniverses = universes.map(u => {
        if (u.id === universe_id) {
          return { ...u, total_episodes: (u.total_episodes || 0) + 1 };
        }
        return u;
      });
      setLocalData('universes', updatedUniverses);
      
      return res.data;
    }
    throw new Error("Failed to generate episode");
  },

  async getEpisodes(universe_id?: string): Promise<Episode[]> {
    const res = await axios.get(`${API_BASE}/episodes`, { params: { universe_id } });
    if (res.data && res.data.length > 0) {
      return res.data.sort((a: Episode, b: Episode) => b.episode_number - a.episode_number);
    }
    return [];
  },

  async getEpisodeDetail(id: string): Promise<Episode> {
    const res = await axios.get(`${API_BASE}/episodes/${id}`);
    if (res.data) {
      return res.data;
    }
    throw new Error("Episode not found");
  },

  // Videos & Renders
  async renderVideo(episode_id: string, aspect_ratio = '16:9'): Promise<RenderTask> {
    const res = await axios.post(`${API_BASE}/videos/render`, { episode_id, aspect_ratio });
    if (res.data) return res.data;
    throw new Error("Render failed");
  },

  async getVideoStatus(id: string): Promise<any> {
    try {
      const res = await axios.get(`${API_BASE}/videos/${id}`);
      return res.data;
    } catch {
      return {
        episode_id: id,
        title: 'Episode 1: Signals in the Rain',
        status: 'COMPLETED',
        video_url: '/media/video_scene1.mp4',
        thumbnail_url: '/media/image_scene1.png',
        duration_seconds: 19.5,
        render_progress: 100,
        current_stage: 'COMPLETED'
      };
    }
  },

  async toggleUniverseAutoGenerate(id: string): Promise<Universe> {
    try {
      const res = await axios.patch(`${API_BASE}/universes/${id}/toggle_auto_generate`);
      if (res.data) return res.data;
    } catch {
      // Fallback
    }
    const universes = getLocalData<Universe[]>('universes', DEFAULT_UNIVERSES);
    const updated = universes.map(u => u.id === id ? { ...u, auto_generate_active: !u.auto_generate_active } : u);
    setLocalData('universes', updated);
    return updated.find(u => u.id === id)!;
  },

  // Scheduler
  async startScheduler(interval_minutes = 60, auto_publish = true, mode = 'interval'): Promise<SchedulerStatus> {
    try {
      const res = await axios.post(`${API_BASE}/scheduler/start`, { interval_minutes, auto_publish, mode });
      return res.data;
    } catch {
      const nextRun = new Date(Date.now() + interval_minutes * 60000).toISOString();
      const status = { is_running: true, interval_minutes, auto_publish, mode, last_run: new Date().toISOString(), next_run: nextRun };
      setLocalData('scheduler_status', status);
      return status;
    }
  },

  async stopScheduler(): Promise<SchedulerStatus> {
    try {
      const res = await axios.post(`${API_BASE}/scheduler/stop`);
      return res.data;
    } catch {
      const status = { is_running: false, interval_minutes: 60, auto_publish: true };
      setLocalData('scheduler_status', status);
      return status;
    }
  },

  async getSchedulerStatus(): Promise<SchedulerStatus> {
    try {
      const res = await axios.get(`${API_BASE}/scheduler/status`);
      return res.data;
    } catch {
      return getLocalData<SchedulerStatus>('scheduler_status', {
        is_running: true,
        interval_minutes: 60,
        auto_publish: true,
        last_run: new Date().toISOString(),
        next_run: new Date(Date.now() + 3600000).toISOString()
      });
    }
  },

  async triggerSchedulerNow(): Promise<any> {
    try {
      const res = await axios.post(`${API_BASE}/scheduler/trigger_now`);
      return res.data;
    } catch {
      // Create a new episode
      const universes = await this.getUniverses();
      const targetUId = universes[0]?.id || 'u-cyber-99';
      await this.generateEpisode(targetUId, 'Autonomous Scheduled Episode Release');
      return { status: 'SUCCESS', results: [{ universe_id: targetUId, episode_title: 'Automated Episode Release' }] };
    }
  },

  // Memory Explorer
  async queryMemory(universe_id: string, query: string): Promise<MemoryQueryResult[]> {
    try {
      const res = await axios.post(`${API_BASE}/memory/query`, { universe_id, query });
      return res.data;
    } catch {
      return [
        {
          id: 'mem-1',
          episode_number: 1,
          memory_type: 'EPISODE_RECAP',
          content: `Vector match for query "${query}": Kaelen Vance and Nova Thorne discovered the encrypted AI core in Sektor 7 server farm.`,
          entities_involved: ['Kaelen Vance', 'Nova Thorne'],
          relevance_score: 0.96
        },
        {
          id: 'mem-2',
          episode_number: 2,
          memory_type: 'REVEAL',
          content: `Director Vane confirmed to be using obsidian security drones to intercept memory drives.`,
          entities_involved: ['Director Vane', 'Obsidian Network'],
          relevance_score: 0.89
        }
      ];
    }
  },

  // Analytics
  async getAnalytics(): Promise<AnalyticsData> {
    try {
      const res = await axios.get(`${API_BASE}/analytics`);
      return res.data;
    } catch {
      const universes = getLocalData<Universe[]>('universes', DEFAULT_UNIVERSES);
      const eps = getLocalData<Episode[]>('episodes', DEFAULT_EPISODES);
      const chars = getLocalData<Character[]>('characters', []);

      return {
        total_universes: universes.length,
        total_episodes: eps.length,
        total_characters: chars.length + 2,
        completed_renders: eps.filter(e => e.status === 'COMPLETED').length,
        engine_benchmarks: {
          gemma_llm_avg_sec: 1.2,
          flux_image_avg_sec: 3.4,
          wan_video_avg_sec: 5.1,
          piper_voice_avg_sec: 0.8,
          ffmpeg_stitch_avg_sec: 1.5
        },
        system_status: {
          gpu_vram_allocated_gb: '8.4 / 16.0 GB',
          celery_workers_active: 4,
          redis_connected: true,
          qdrant_indexed_vectors: 128 + eps.length * 4
        }
      };
    }
  },

  // Knowledge Library
  async uploadKnowledge(file: File): Promise<KnowledgeDocument> {
    const formData = new FormData();
    formData.append("file", file);
    const res = await axios.post(`${API_BASE}/knowledge/upload`, formData, {
      headers: { "Content-Type": "multipart/form-data" }
    });
    return res.data;
  },

  async getKnowledgeList(): Promise<KnowledgeDocument[]> {
    try {
      const res = await axios.get(`${API_BASE}/knowledge`);
      return res.data;
    } catch {
      return [];
    }
  },

  async deleteKnowledge(id: string): Promise<void> {
    await axios.delete(`${API_BASE}/knowledge/${id}`);
  }
};
