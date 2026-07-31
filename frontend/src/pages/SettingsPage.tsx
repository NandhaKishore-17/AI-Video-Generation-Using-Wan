import React, { useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { Settings, Cpu, Save, Sliders, CheckCircle2, Cloud, HardDrive, Key } from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const [gpuMode, setGpuMode] = useState('LOCAL_GPU');
  const [llmModel, setLlmModel] = useState('Qwen/Qwen2.5-72B-Instruct');
  const [imageModel, setImageModel] = useState('black-forest-labs/FLUX.1-schnell');
  const [videoModel, setVideoModel] = useState('Wan-AI/Wan2.1-T2V-1.3B (CogVideoX)');
  const [voiceModel, setVoiceModel] = useState('F5-TTS Open-Source Zero-Shot (GPU/CPU)');
  const [hfToken, setHfToken] = useState('');
  const [saved, setSaved] = useState(false);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <Settings className="w-6 h-6 text-purple-400" />
          <span>GPU ENGINE & CLOUD CONNECTOR SETTINGS</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Configure local GPU (Nvidia CUDA / PyTorch) execution or switch to open-source Cloud GPU endpoints (HuggingFace / RunPod / Modal).
        </p>
      </div>

      <GlassCard glow className="max-w-2xl">
        <form onSubmit={handleSave} className="space-y-5 text-xs font-mono">
          {/* GPU Mode Selector */}
          <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 space-y-2">
            <label className="block text-cyan-400 font-bold flex items-center gap-2">
              <Cpu className="w-4 h-4 text-cyan-400" />
              <span>GPU EXECUTION MODE</span>
            </label>
            <select
              value={gpuMode}
              onChange={(e) => setGpuMode(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2.5 text-white font-bold focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="LOCAL_GPU">Local Open Source GPU (Nvidia CUDA / PyTorch / Ollama / ComfyUI)</option>
              <option value="CLOUD_GPU">Cloud GPU (HuggingFace Serverless / RunPod / Modal / Replicate)</option>
              <option value="HYBRID">Hybrid (Local LLM + Cloud Diffusers Video GPU)</option>
            </select>
            <p className="text-[11px] text-slate-400 pt-1">
              Currently running in <span className="text-cyan-300 font-bold">{gpuMode}</span> mode.
            </p>
          </div>

          {/* HuggingFace API Token for Cloud GPU */}
          {gpuMode !== 'LOCAL_GPU' && (
            <div className="p-4 rounded-2xl bg-purple-950/40 border border-purple-800 space-y-2">
              <label className="block text-purple-300 font-bold flex items-center gap-2">
                <Key className="w-4 h-4 text-purple-400" />
                <span>HUGGINGFACE / CLOUD GPU API TOKEN</span>
              </label>
              <input
                type="password"
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
                className="w-full bg-slate-950 border border-purple-800/60 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-purple-500"
              />
              <p className="text-[10px] text-purple-400">
                Required for serverless HuggingFace Cloud GPU video inference (Wan2.1 / CogVideoX / FLUX).
              </p>
            </div>
          )}

          <div>
            <label className="block text-slate-300 font-bold mb-1">STORY GENERATION LLM MODEL</label>
            <input
              type="text"
              value={llmModel}
              onChange={(e) => setLlmModel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
            />
          </div>

          <div>
            <label className="block text-slate-300 font-bold mb-1">IMAGE GENERATOR DIFFUSER (FLUX / SDXL)</label>
            <input
              type="text"
              value={imageModel}
              onChange={(e) => setImageModel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
            />
          </div>

          <div>
            <label className="block text-slate-300 font-bold mb-1">VIDEO MOTION SYNTHESIZER (COGVIDEOX / WAN 2.1)</label>
            <input
              type="text"
              value={videoModel}
              onChange={(e) => setVideoModel(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
            />
          </div>

          <div className="p-4 rounded-2xl bg-cyan-950/30 border border-cyan-800 space-y-3">
            <label className="block text-cyan-300 font-bold flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <span>HUMAN VOICE SYNTHESIS PROVIDER</span>
              </span>
              <span className="text-[10px] text-emerald-400 font-bold bg-emerald-950 px-2 py-0.5 rounded-full border border-emerald-700">
                24kHz+ HD Audio
              </span>
            </label>
            <select
              value={voiceModel}
              onChange={(e) => setVoiceModel(e.target.value)}
              className="w-full bg-slate-900 border border-cyan-800 rounded-xl px-3 py-2.5 text-white font-bold focus:outline-none focus:border-cyan-500 cursor-pointer"
            >
              <option value="F5-TTS Open-Source Zero-Shot (GPU/CPU)">F5-TTS Open-Source Zero-Shot Model (GPU CUDA + CPU Fallback)</option>
              <option value="Edge-TTS Azure Neural (Realistic Free)">Edge-TTS Azure Neural (Realistic Free & Zero-Setup)</option>
              <option value="ElevenLabs Multilingual v2">ElevenLabs Voice Cloning (Ultra-Realistic Human Voice API)</option>
              <option value="OpenAI TTS (tts-1-hd)">OpenAI HD Voice Engine (Natural Voice API)</option>
            </select>
            <p className="text-[11px] text-slate-400">
              Active Voice Provider: <span className="text-cyan-300 font-bold">{voiceModel}</span>
            </p>

            {voiceModel.includes('ElevenLabs') && (
              <div className="pt-2">
                <label className="block text-purple-300 font-bold mb-1">ELEVENLABS API KEY</label>
                <input
                  type="password"
                  placeholder="xi-api-key-xxxxxxxxxxxxxxxx"
                  className="w-full bg-slate-950 border border-purple-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-purple-500"
                />
              </div>
            )}

            {voiceModel.includes('OpenAI') && (
              <div className="pt-2">
                <label className="block text-purple-300 font-bold mb-1">OPENAI API KEY</label>
                <input
                  type="password"
                  placeholder="sk-proj-xxxxxxxxxxxxxxxx"
                  className="w-full bg-slate-950 border border-purple-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-purple-500"
                />
              </div>
            )}
          </div>

          <div className="pt-4 flex items-center justify-between">
            <button
              type="submit"
              className="px-6 py-3 rounded-xl font-bold bg-gradient-to-r from-cyan-500 via-purple-600 to-pink-500 hover:brightness-110 text-white shadow-glow-cyan flex items-center space-x-2 text-xs"
            >
              <Save className="w-4 h-4" />
              <span>SAVE GPU CONFIGURATION</span>
            </button>

            {saved && (
              <span className="text-emerald-400 font-bold flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4" /> GPU CONFIG SAVED
              </span>
            )}
          </div>
        </form>
      </GlassCard>
    </div>
  );
};
