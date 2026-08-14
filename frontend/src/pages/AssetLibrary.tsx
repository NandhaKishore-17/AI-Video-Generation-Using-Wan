import React from 'react';
import { GlassCard } from '../components/GlassCard';
import { FileVideo, Image as ImageIcon, Music, Mic, Film } from 'lucide-react';

export const AssetLibrary: React.FC = () => {
  const sampleAssets = [
    { id: 'a1', type: 'IMAGE', name: 'FLUX Keyframe — Scene 1', model: 'FLUX.1-dev', url: '/media/image_scene1.png' },
    { id: 'a2', type: 'IMAGE', name: 'SDXL Keyframe — Scene 2', model: 'SDXL 1.0', url: '/media/image_scene2.png' },
    { id: 'a3', type: 'AUDIO', name: 'TTS Voice Dialogue — Scene 1', model: 'Edge-TTS / Piper', url: '/media/audio_scene1.wav' },
    { id: 'a4', type: 'MUSIC', name: 'MusicGen Score — Ambient', model: 'MusicGen', url: '/media/music_scene1.wav' },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <FileVideo className="w-6 h-6 text-pink-400" />
          <span>MEDIA ASSET VAULT</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Inspect generated FLUX keyframe images, CogVideoX video clips, Piper TTS voice narration stems, and MusicGen audio scores.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {sampleAssets.map((asset) => (
          <GlassCard key={asset.id} className="space-y-3">
            <div className="aspect-video bg-slate-950 rounded-xl border border-slate-800 overflow-hidden flex flex-col items-center justify-center relative group p-3">
              {asset.type === 'IMAGE' ? (
                <img src={asset.url} alt={asset.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
              ) : (
                <div className="flex flex-col items-center space-y-2 text-cyan-400 w-full">
                  <div className="flex items-center space-x-2">
                    <Music className="w-6 h-6 text-purple-400 animate-pulse" />
                    <span className="text-[11px] font-mono text-cyan-300 font-bold">WAV AUDIO STEM</span>
                  </div>
                  <audio controls src={asset.url} className="w-full h-8 accent-cyan-400 rounded" />
                </div>
              )}
            </div>

            <div className="space-y-1">
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                {asset.type} • {asset.model}
              </span>
              <h4 className="text-xs font-bold text-white truncate">{asset.name}</h4>
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
};
