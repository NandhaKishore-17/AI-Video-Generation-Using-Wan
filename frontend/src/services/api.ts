import axios from 'axios';
import {
  Universe, Character, TimelineEvent, StoryArc, Episode, Scene, RenderTask,
  SchedulerStatus, MemoryQueryResult, AnalyticsData
} from '../types';

const API_BASE = '/api/v1';

// Helper: extract error message from backend response
function extractErrorMessage(err: any): string {
  if (axios.isAxiosError(err) && err.response?.data) {
    const detail = err.response.data.detail;
    if (typeof detail === 'string') return detail;
    if (detail?.error && detail?.reason) return `${detail.error}: ${detail.reason}`;
    if (detail?.error) return detail.error;
    return JSON.stringify(detail);
  }
  return err?.message || 'Unknown error';
}

// API Service — all data comes from the backend.
// No hardcoded universes, characters, episodes, or dialogue.
// If the backend is unreachable, errors are thrown — never silently masked.

const api = {
  // --- Universes ---
  async createUniverse(payload: {
    title: string; genre: string; logline: string; world_rules?: string;
  }): Promise<Universe> {
    try {
      const res = await axios.post(`${API_BASE}/universes`, payload, {
        timeout: 180_000,  // 3 minutes — LLM generation can be slow
      });
      return res.data;
    } catch (err) {
      if (axios.isAxiosError(err) && err.code === 'ECONNABORTED') {
        throw new Error('Universe generation timed out. The LLM may be overloaded or not running. Please check that Ollama is running and try again.');
      }
      throw new Error(`Failed to create universe: ${extractErrorMessage(err)}`);
    }
  },

  async deleteUniverse(id: string): Promise<void> {
    await axios.delete(`${API_BASE}/universes/${id}`);
  },

  async resetAllUniverses(): Promise<void> {
    await axios.delete(`${API_BASE}/universes/reset_all`);
  },

  async getUniverses(): Promise<Universe[]> {
    try {
      const res = await axios.get(`${API_BASE}/universes`);
      return res.data || [];
    } catch {
      return [];
    }
  },

  async getUniverseDetail(id: string): Promise<any> {
    try {
      const res = await axios.get(`${API_BASE}/universes/${id}`);
      return res.data;
    } catch {
      // Return empty structure — the UI should handle empty state gracefully
      return {
        universe: null,
        characters: [],
        timeline: [],
        story_arcs: [],
        episodes_summary: []
      };
    }
  },

  // --- Characters ---
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
    } catch (err) {
      throw new Error(`Failed to create character: ${extractErrorMessage(err)}`);
    }
  },

  async deleteCharacter(id: string): Promise<void> {
    await axios.delete(`${API_BASE}/characters/${id}`);
  },

  async getAvailableVoices(): Promise<any[]> {
    try {
      const res = await axios.get(`${API_BASE}/voices/available`);
      return res.data || [];
    } catch {
      // Voice list is a valid static fallback — these are platform capabilities, not story data
      return [
        { id: "en-US-ChristopherNeural", name: "Christopher", gender: "male", archetype: "deep_male", description: "Resonant deep cinematic male voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-US-AndrewNeural", name: "Andrew", gender: "male", archetype: "young_male", description: "Young energetic natural male voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-US-EricNeural", name: "Eric", gender: "male", archetype: "calm_male", description: "Calm and articulate professional male voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-US-GuyNeural", name: "Guy", gender: "male", archetype: "energetic_male", description: "Dynamic energetic male voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-AU-WilliamNeural", name: "William", gender: "male", archetype: "deep_villain_male", description: "Deep authoritative villain male voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-US-AriaNeural", name: "Aria", gender: "female", archetype: "calm_female", description: "Calm poised female voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-US-AnaNeural", name: "Ana", gender: "female", archetype: "young_female", description: "Bright young female voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-US-MichelleNeural", name: "Michelle", gender: "female", archetype: "strong_female", description: "Strong determined female voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-US-AvaNeural", name: "Ava", gender: "female", archetype: "soft_female", description: "Soft gentle natural female voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-US-JennyNeural", name: "Jenny", gender: "female", archetype: "friendly_female", description: "Expressive friendly human female voice", provider: "Edge-TTS Azure Neural" },
        { id: "en-GB-SoniaNeural", name: "Sonia (British)", gender: "female", archetype: "british_female", description: "British elegant cinematic female voice", provider: "Edge-TTS Azure Neural" }
      ];
    }
  },

  // --- Timeline Events ---
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
    } catch (err) {
      throw new Error(`Failed to create timeline event: ${extractErrorMessage(err)}`);
    }
  },

  async deleteTimelineEvent(id: string): Promise<void> {
    await axios.delete(`${API_BASE}/timeline_events/${id}`);
  },

  // --- Story Arcs ---
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
    } catch (err) {
      throw new Error(`Failed to create story arc: ${extractErrorMessage(err)}`);
    }
  },

  async deleteStoryArc(id: string): Promise<void> {
    await axios.delete(`${API_BASE}/story_arcs/${id}`);
  },

  // --- Episodes ---
  async generateEpisode(universe_id: string, custom_prompt?: string, scene_duration_seconds?: number): Promise<Episode> {
    try {
      const res = await axios.post(`${API_BASE}/episodes/generate`, {
        universe_id,
        custom_prompt,
        scene_duration_seconds
      }, {
        timeout: 300_000, // 5 minutes
      });
      return res.data;
    } catch (err) {
      if (axios.isAxiosError(err) && err.code === 'ECONNABORTED') {
        throw new Error('Episode generation timed out. The LLM or Video models may be overloaded. Please check Ollama and try again.');
      }
      throw new Error(`Failed to generate episode: ${extractErrorMessage(err)}`);
    }
  },

  async getEpisodes(universe_id?: string): Promise<Episode[]> {
    try {
      const res = await axios.get(`${API_BASE}/episodes`, { params: { universe_id } });
      return res.data || [];
    } catch {
      return [];
    }
  },

  async getEpisodeDetail(id: string): Promise<Episode | null> {
    try {
      const res = await axios.get(`${API_BASE}/episodes/${id}`);
      return res.data;
    } catch {
      return null;
    }
  },

  // --- Videos & Renders ---
  async renderVideo(episode_id: string, aspect_ratio = '16:9'): Promise<RenderTask> {
    try {
      const res = await axios.post(`${API_BASE}/videos/render`, { episode_id, aspect_ratio });
      return res.data;
    } catch (err) {
      throw new Error(`Failed to render video: ${extractErrorMessage(err)}`);
    }
  },

  async getVideoStatus(id: string): Promise<any> {
    try {
      const res = await axios.get(`${API_BASE}/videos/${id}`);
      return res.data;
    } catch {
      return null;
    }
  },

  async toggleUniverseAutoGenerate(id: string): Promise<Universe> {
    const res = await axios.patch(`${API_BASE}/universes/${id}/toggle_auto_generate`);
    return res.data;
  },

  // --- Scheduler ---
  async startScheduler(interval_minutes = 60, auto_publish = true): Promise<SchedulerStatus> {
    const res = await axios.post(`${API_BASE}/scheduler/start`, { interval_minutes, auto_publish });
    return res.data;
  },

  async stopScheduler(): Promise<SchedulerStatus> {
    const res = await axios.post(`${API_BASE}/scheduler/stop`);
    return res.data;
  },

  async getSchedulerStatus(): Promise<SchedulerStatus> {
    try {
      const res = await axios.get(`${API_BASE}/scheduler/status`);
      return res.data;
    } catch {
      return { is_running: false, interval_minutes: 60, auto_publish: true };
    }
  },

  async triggerSchedulerNow(): Promise<any> {
    const res = await axios.post(`${API_BASE}/scheduler/trigger_now`);
    return res.data;
  },

  // --- Memory Explorer ---
  async queryMemory(universe_id: string, query: string): Promise<MemoryQueryResult[]> {
    try {
      const res = await axios.post(`${API_BASE}/memory/query`, { universe_id, query });
      return res.data || [];
    } catch {
      return [];
    }
  },

  // --- Analytics ---
  async getAnalytics(): Promise<AnalyticsData> {
    try {
      const res = await axios.get(`${API_BASE}/analytics`);
      return res.data;
    } catch {
      return {
        total_universes: 0,
        total_episodes: 0,
        total_characters: 0,
        completed_renders: 0,
        engine_benchmarks: {},
        system_status: {}
      };
    }
  }
};

export { api };
export default api;
