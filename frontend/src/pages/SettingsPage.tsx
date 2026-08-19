import React, { useState, useEffect } from 'react';
import { GlassCard } from '../components/GlassCard';
import { Settings, Cpu, Save, Sliders, CheckCircle2, Cloud, HardDrive, Key } from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const [gpuMode, setGpuMode] = useState(() => localStorage.getItem('gpu_mode') || 'LOCAL_GPU');
  const [llmModel, setLlmModel] = useState(() => localStorage.getItem('llm_model') || 'Qwen/Qwen2.5-72B-Instruct');
  const [imageModel, setImageModel] = useState(() => localStorage.getItem('image_model') || 'black-forest-labs/FLUX.1-schnell');
  const [videoModel, setVideoModel] = useState(() => localStorage.getItem('video_model') || 'Wan-AI/Wan2.1-T2V-1.3B (CogVideoX)');
  const [voiceModel, setVoiceModel] = useState(() => localStorage.getItem('voice_model') || 'F5-TTS Open-Source Zero-Shot (GPU/CPU)');
  const [hfToken, setHfToken] = useState(() => localStorage.getItem('hf_token') || '');
  const [saved, setSaved] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    localStorage.setItem('gpu_mode', gpuMode);
    localStorage.setItem('llm_model', llmModel);
    localStorage.setItem('image_model', imageModel);
    localStorage.setItem('video_model', videoModel);
    localStorage.setItem('voice_model', voiceModel);
    localStorage.setItem('hf_token', hfToken);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };


  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-text-primary tracking-tight flex items-center gap-2">
          <Settings className="w-6 h-6 text-text-secondary" />
          <span>GPU ENGINE & CLOUD CONNECTOR SETTINGS</span>
        </h1>
        <p className="text-xs text-text-secondary mt-1">
          Configure local GPU (Nvidia CUDA / PyTorch) execution or switch to open-source Cloud GPU endpoints (HuggingFace / RunPod / Modal).
        </p>
      </div>

      <GlassCard glow className="max-w-2xl">
        <form onSubmit={handleSave} className="space-y-5 text-xs font-mono">
          {/* GPU Mode Selector */}
          <div className="p-4 rounded-2xl bg-neutral-soft border border-neutral-border space-y-2">
            <label className="block text-accent-primary font-bold flex items-center gap-2">
              <Cpu className="w-4 h-4 text-accent-primary" />
              <span>GPU EXECUTION MODE</span>
            </label>
            <select
              value={gpuMode}
              onChange={(e) => setGpuMode(e.target.value)}
              className="w-full bg-white border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary font-bold focus:outline-none focus:border-accent-primary cursor-pointer"
            >
              <option value="LOCAL_GPU">Local Open Source GPU (Nvidia CUDA / PyTorch / Ollama / ComfyUI)</option>
              <option value="CLOUD_GPU">Cloud GPU (HuggingFace Serverless / RunPod / Modal / Replicate)</option>
              <option value="HYBRID">Hybrid (Local LLM + Cloud Diffusers Video GPU)</option>
            </select>
            <p className="text-[11px] text-text-secondary pt-1">
              Currently running in <span className="text-accent-primary font-bold">{gpuMode}</span> mode.
            </p>
          </div>

          {/* HuggingFace API Token for Cloud GPU */}
          {gpuMode !== 'LOCAL_GPU' && (
            <div className="p-4 rounded-2xl bg-neutral-soft border border-neutral-border space-y-2">
              <label className="block text-text-secondary font-bold flex items-center gap-2">
                <Key className="w-4 h-4 text-text-secondary" />
                <span>HUGGINGFACE / CLOUD GPU API TOKEN</span>
              </label>
              <input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
                className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-neutral-border"
              />
              <p className="text-[10px] text-text-secondary">
                Required for serverless HuggingFace Cloud GPU video inference (Wan2.1 / CogVideoX / FLUX).
              </p>
            </div>
          )}

          <div>
            <label className="block text-text-secondary font-bold mb-1">STORY GENERATION LLM MODEL</label>
            <input
              type="text"
              value={llmModel}
              onChange={(e) => setLlmModel(e.target.value)}
              className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
            />
          </div>

          <div>
            <label className="block text-text-secondary font-bold mb-1">IMAGE GENERATOR DIFFUSER (FLUX / SDXL)</label>
            <input
              type="text"
              value={imageModel}
              onChange={(e) => setImageModel(e.target.value)}
              className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
            />
          </div>

          <div>
            <label className="block text-text-secondary font-bold mb-1">VIDEO MOTION SYNTHESIZER (COGVIDEOX / WAN 2.1)</label>
            <input
              type="text"
              value={videoModel}
              onChange={(e) => setVideoModel(e.target.value)}
              className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
            />
          </div>

          <div className="p-4 rounded-2xl bg-neutral-soft border border-accent-primary space-y-3">
            <label className="block text-accent-primary font-bold flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-accent-primary" />
                <span>HUMAN VOICE SYNTHESIS PROVIDER</span>
              </span>
              <span className="text-[10px] text-accent-primary font-bold bg-neutral-soft px-2 py-0.5 rounded-full border border-accent-primary">
                24kHz+ HD Audio
              </span>
            </label>
            <select
              value={voiceModel}
              onChange={(e) => setVoiceModel(e.target.value)}
              className="w-full bg-white border border-accent-primary rounded-xl px-3 py-2.5 text-text-primary font-bold focus:outline-none focus:border-accent-primary cursor-pointer"
            >
              <option value="F5-TTS Open-Source Zero-Shot (GPU/CPU)">F5-TTS Open-Source Zero-Shot Model (GPU CUDA + CPU Fallback)</option>
              <option value="Edge-TTS Azure Neural (Realistic Free)">Edge-TTS Azure Neural (Realistic Free & Zero-Setup)</option>
              <option value="ElevenLabs Multilingual v2">ElevenLabs Voice Cloning (Ultra-Realistic Human Voice API)</option>
              <option value="OpenAI TTS (tts-1-hd)">OpenAI HD Voice Engine (Natural Voice API)</option>
            </select>
            <p className="text-[11px] text-text-secondary">
              Active Voice Provider: <span className="text-accent-primary font-bold">{voiceModel}</span>
            </p>

            {voiceModel.includes('ElevenLabs') && (
              <div className="pt-2">
                <label className="block text-text-secondary font-bold mb-1">ELEVENLABS API KEY</label>
                <input
                  type="password"
                  placeholder="xi-api-key-xxxxxxxxxxxxxxxx"
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-neutral-border"
                />
              </div>
            )}

            {voiceModel.includes('OpenAI') && (
              <div className="pt-2">
                <label className="block text-text-secondary font-bold mb-1">OPENAI API KEY</label>
                <input
                  type="password"
                  placeholder="sk-proj-xxxxxxxxxxxxxxxx"
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-neutral-border"
                />
              </div>
            )}
          </div>

          <div className="pt-4 flex items-center justify-between">
            <button
              type="submit"
              className="px-6 py-3 rounded-xl font-bold bg-neutral-soft hover:brightness-110 text-text-primary shadow-sm flex items-center space-x-2 text-xs"
            >
              <Save className="w-4 h-4" />
              <span>SAVE GPU CONFIGURATION</span>
            </button>

            {saved && (
              <span className="text-accent-primary font-bold flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4" /> GPU CONFIG SAVED
              </span>
            )}
          </div>
        </form>
      </GlassCard>
    </div>
  );
};
