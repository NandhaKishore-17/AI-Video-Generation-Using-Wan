import React from 'react';
import { GlassCard } from '../components/GlassCard';
import { FileVideo, Image as ImageIcon, Music, Mic, Film } from 'lucide-react';

export const AssetLibrary: React.FC = () => {
  const sampleAssets = [
    { id: 'a1', type: 'IMAGE', name: 'FLUX Keyframe - Hovercycle Highway', model: 'FLUX.1-dev', url: '/media/image_scene1.png' },
    { id: 'a2', type: 'IMAGE', name: 'SDXL Keyframe - Holographic Server Vault', model: 'SDXL 1.0', url: '/media/image_scene2.png' },
    { id: 'a3', type: 'AUDIO', name: 'Piper Voice Dialogue - Kaelen Vance', model: 'Piper TTS', url: '/media/audio_scene1.wav' },
    { id: 'a4', type: 'MUSIC', name: 'MusicGen Score - Cyberpunk Ambient Drone', model: 'MusicGen', url: '/media/music_scene1.wav' },
  ];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-text-primary tracking-tight flex items-center gap-2">
          <FileVideo className="w-6 h-6 text-text-secondary" />
          <span>MEDIA ASSET VAULT</span>
        </h1>
        <p className="text-xs text-text-secondary mt-1">
          Inspect generated FLUX keyframe images, CogVideoX video clips, Piper TTS voice narration stems, and MusicGen audio scores.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        {sampleAssets.map((asset) => (
          <GlassCard key={asset.id} className="space-y-3">
            <div className="aspect-video bg-neutral-soft rounded-xl border border-neutral-border overflow-hidden flex flex-col items-center justify-center relative group p-3">
              {asset.type === 'IMAGE' ? (
                <img src={asset.url} alt={asset.name} className="w-full h-full object-cover group-hover:scale-105 transition-transform" />
              ) : (
                <div className="flex flex-col items-center space-y-2 text-accent-primary w-full">
                  <div className="flex items-center space-x-2">
                    <Music className="w-6 h-6 text-text-secondary animate-pulse" />
                    <span className="text-[11px] font-mono text-accent-primary font-bold">WAV AUDIO STEM</span>
                  </div>
                  <audio controls src={asset.url} className="w-full h-8 accent-cyan-400 rounded" />
                </div>
              )}
            </div>

            <div className="space-y-1">
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-soft text-accent-primary border border-accent-primary">
                {asset.type} • {asset.model}
              </span>
              <h4 className="text-xs font-bold text-text-primary truncate">{asset.name}</h4>
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
};
