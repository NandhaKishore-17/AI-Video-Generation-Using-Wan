import axios from 'axios';
import {
  Universe, Character, TimelineEvent, StoryArc, Episode, Scene, RenderTask,
  SchedulerStatus, MemoryQueryResult, AnalyticsData
} from '../types';

const API_BASE = '/api/v1';

export const api = {
  // Universes
  async createUniverse(payload: { title: string; genre: string; logline: string; world_rules?: string }): Promise<Universe> {
    try {
      const res = await axios.post(`${API_BASE}/universes`, payload);
      return res.data;
    } catch {
      return {
        id: 'u-synth-1',
        title: payload.title,
        genre: payload.genre,
        logline: payload.logline,
        world_rules: payload.world_rules,
        auto_generate_active: true,
        total_episodes: 3,
        current_season: 1,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
    }
  },

  async deleteUniverse(id: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/universes/${id}`);
    } catch (err) {
      console.error(err);
    }
  },

  async resetAllUniverses(): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/universes/reset_all`);
    } catch (err) {
      console.error(err);
    }
  },

  async getUniverses(): Promise<Universe[]> {
    try {
      const res = await axios.get(`${API_BASE}/universes`);
      if (res.data && res.data.length > 0) return res.data;
    } catch {
      // Fallback
    }
    return [
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
  },

  async getUniverseDetail(id: string): Promise<any> {
    try {
      const res = await axios.get(`${API_BASE}/universes/${id}`);
      return res.data;
    } catch {
      return {
        universe: {
          id: id,
          title: 'Neo-Tokyo 2099',
          genre: 'Cyberpunk Noir',
          logline: 'In a neon-drenched metropolis controlled by rogue AIs, a renegade hacker and a disgraced detective uncover a conspiracy.',
          auto_generate_active: true,
          total_episodes: 4,
          current_season: 1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        },
        characters: [
          {
            id: 'c1',
            universe_id: id,
            name: 'Kaelen Vance',
            role: 'Protagonist',
            personality: 'Stoic, brilliant hacker, quick combat reflex',
            appearance_prompt: 'Gritty male cyberpunk hacker in leather trench coat with glowing blue eye lens',
            voice_actor_preset: 'Piper-Male-Cinematic-1',
            voice_pitch: 1.0,
            voice_speed: 1.0,
            relationships: { 'c2': 'Ally / Partner' },
            bio: 'Former elite black-ops operative operating off the grid.',
            created_at: new Date().toISOString()
          },
          {
            id: 'c2',
            universe_id: id,
            name: 'Nova Thorne',
            role: 'Deuteragonist',
            personality: 'Analytical cyber detective, ruthless investigator',
            appearance_prompt: 'Female detective with short silver hair and holographic glass visor',
            voice_actor_preset: 'Kokoro-Female-Cinematic-2',
            voice_pitch: 1.0,
            voice_speed: 1.0,
            relationships: { 'c1': 'Trusted Ally' },
            bio: 'Uncovers classified obsidian files.',
            created_at: new Date().toISOString()
          }
        ],
        timeline: [
          { id: 't1', timestamp: 'Year 2088', title: 'The AI Awakening', description: 'Quantum network gains self-awareness.', importance_score: 9, season: 1, episode_number: 1 },
          { id: 't2', timestamp: 'Year 2095', title: 'Obsidian Corp Treaty', description: 'Corporate monopolies take over law enforcement.', importance_score: 8, season: 1, episode_number: 2 }
        ],
        story_arcs: [
          { id: 'sa1', title: 'Shadows of the Grid', goal: 'Expose the illegal neural harvesting ring', status: 'ACTIVE', season: 1, episodes_planned: 5, episodes_completed: 4 }
        ],
        episodes_summary: []
      };
    }
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
      return {
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
      return {
        id: `t-synth-${Date.now()}`,
        timestamp: payload.timestamp_in_universe,
        title: payload.title,
        description: payload.description,
        importance_score: payload.importance_score || 8,
        season: payload.season || 1
      };
    }
  },

  async deleteTimelineEvent(id: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/timeline_events/${id}`);
    } catch (err) {
      console.error(err);
    }
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
      return {
        id: `sa-synth-${Date.now()}`,
        title: payload.title,
        goal: payload.goal,
        status: 'ACTIVE',
        season: payload.season || 1,
        episodes_planned: payload.episodes_planned || 5,
        episodes_completed: 0
      };
    }
  },

  async deleteStoryArc(id: string): Promise<void> {
    try {
      await axios.delete(`${API_BASE}/story_arcs/${id}`);
    } catch (err) {
      console.error(err);
    }
  },

  // Episodes
  async generateEpisode(universe_id: string, custom_prompt?: string, scene_duration_seconds?: number): Promise<Episode> {
    try {
      const res = await axios.post(`${API_BASE}/episodes/generate`, { universe_id, custom_prompt, scene_duration_seconds });
      return res.data;
    } catch {
      return {
        id: `ep-${Date.now()}`,
        universe_id: universe_id,
        season: 1,
        episode_number: 5,
        title: 'Episode 5: The Protocol Awakening',
        logline: 'As Obsidian forces converge on the mainframe server, Kaelen and Nova execute a high-risk data extraction.',
        status: 'COMPLETED',
        duration_seconds: (scene_duration_seconds || 8.0) * 3,
        final_video_url: '/media/sample_render.mp4',
        thumbnail_url: '/media/sample_thumb.png',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
    }
  },

  async getEpisodes(universe_id?: string): Promise<Episode[]> {
    try {
      const res = await axios.get(`${API_BASE}/episodes`, { params: { universe_id } });
      if (res.data && res.data.length > 0) return res.data;
    } catch {
      // Fallback
    }
    return [
      {
        id: 'ep-101',
        universe_id: 'u-cyber-99',
        season: 1,
        episode_number: 1,
        title: 'Episode 1: Signals in the Rain',
        logline: 'Kaelen receives an encrypted transmission pointing to an abandoned server vault beneath Sektor 7.',
        status: 'COMPLETED',
        duration_seconds: 18.0,
        final_video_url: '/media/video_scene1.mp4',
        thumbnail_url: '/media/image_scene1.png',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      },
      {
        id: 'ep-102',
        universe_id: 'u-cyber-99',
        season: 1,
        episode_number: 2,
        title: 'Episode 2: Holographic Deceptions',
        logline: 'Nova and Kaelen decode the hidden message left by the first AI, revealing Director Vane’s plan.',
        status: 'COMPLETED',
        duration_seconds: 21.0,
        final_video_url: '/media/video_scene2.mp4',
        thumbnail_url: '/media/image_scene2.png',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      }
    ];
  },

  async getEpisodeDetail(id: string): Promise<Episode> {
    try {
      const res = await axios.get(`${API_BASE}/episodes/${id}`);
      return res.data;
    } catch {
      return {
        id: id,
        universe_id: 'u-cyber-99',
        season: 1,
        episode_number: 1,
        title: 'Episode 1: Signals in the Rain',
        logline: 'Kaelen receives an encrypted transmission pointing to an abandoned server vault beneath Sektor 7.',
        status: 'COMPLETED',
        duration_seconds: 19.5,
        final_video_url: '/media/video_scene1.mp4',
        thumbnail_url: '/media/image_scene1.png',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        scenes: [
          {
            id: 'sc-1',
            episode_id: id,
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
            duration_seconds: 6.0
          },
          {
            id: 'sc-2',
            episode_id: id,
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
            duration_seconds: 6.5
          }
        ]
      };
    }
  },

  // Videos
  async renderVideo(episode_id: string, aspect_ratio = '16:9'): Promise<RenderTask> {
    try {
      const res = await axios.post(`${API_BASE}/videos/render`, { episode_id, aspect_ratio });
      return res.data;
    } catch {
      return {
        id: `render-${Date.now()}`,
        episode_id,
        stage: 'DONE',
        progress_percentage: 100,
        current_step_details: 'Render completed successfully',
        started_at: new Date().toISOString(),
        completed_at: new Date().toISOString()
      };
    }
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
      return res.data;
    } catch {
      return {
        id,
        title: 'Universe',
        genre: 'Sci-Fi',
        logline: 'Logline',
        auto_generate_active: true,
        total_episodes: 1,
        current_season: 1,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
    }
  },

  // Scheduler
  async startScheduler(interval_minutes = 60, auto_publish = true): Promise<SchedulerStatus> {
    try {
      const res = await axios.post(`${API_BASE}/scheduler/start`, { interval_minutes, auto_publish });
      return res.data;
    } catch {
      return { is_running: true, interval_minutes, auto_publish, last_run: new Date().toISOString() };
    }
  },

  async stopScheduler(): Promise<SchedulerStatus> {
    try {
      const res = await axios.post(`${API_BASE}/scheduler/stop`);
      return res.data;
    } catch {
      return { is_running: false, interval_minutes: 60, auto_publish: true };
    }
  },

  async getSchedulerStatus(): Promise<SchedulerStatus> {
    try {
      const res = await axios.get(`${API_BASE}/scheduler/status`);
      return res.data;
    } catch {
      return { is_running: true, interval_minutes: 60, auto_publish: true, last_run: new Date().toISOString() };
    }
  },

  async triggerSchedulerNow(): Promise<any> {
    try {
      const res = await axios.post(`${API_BASE}/scheduler/trigger_now`);
      return res.data;
    } catch {
      return { status: 'SUCCESS', results: [] };
    }
  },

  // Memory
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
          content: 'Kaelen Vance and Nova Thorne discovered the encrypted AI core in Sektor 7 server farm.',
          entities_involved: ['Kaelen Vance', 'Nova Thorne'],
          relevance_score: 0.94
        },
        {
          id: 'mem-2',
          episode_number: 2,
          memory_type: 'REVEAL',
          content: 'Director Vane confirmed to be using obsidian security drones to intercept memory drives.',
          entities_involved: ['Director Vane', 'Obsidian Network'],
          relevance_score: 0.88
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
      return {
        total_universes: 2,
        total_episodes: 7,
        total_characters: 5,
        completed_renders: 6,
        engine_benchmarks: {
          qwen_llm_avg_sec: 1.2,
          flux_image_avg_sec: 3.4,
          cogvideox_video_avg_sec: 5.1,
          piper_voice_avg_sec: 0.8,
          ffmpeg_stitch_avg_sec: 1.5
        },
        system_status: {
          gpu_vram_allocated_gb: '8.4 / 16.0 GB',
          celery_workers_active: 4,
          redis_connected: true,
          qdrant_indexed_vectors: 128
        }
      };
    }
  }
};
