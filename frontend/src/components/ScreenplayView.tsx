import React from 'react';
import { Scene } from '../types';
import { Sparkles, Video, Mic, Image as ImageIcon } from 'lucide-react';

interface ScreenplayViewProps {
  scenes: Scene[];
  title: string;
  logline: string;
}

export const ScreenplayView: React.FC<ScreenplayViewProps> = ({ scenes, title, logline }) => {
  return (
    <div className="space-y-6 font-mono text-sm">
      {/* Title Card */}
      <div className="border-b border-neutral-border pb-4">
        <h2 className="text-xl font-bold text-text-primary tracking-wider font-sans uppercase">{title}</h2>
        <p className="text-xs text-text-secondary font-sans mt-1">{logline}</p>
      </div>

      {/* Scenes List */}
      <div className="space-y-8">
        {scenes.map((scene, idx) => (
          <div key={scene.id || idx} className="p-5 rounded-xl bg-white/60 border border-neutral-border space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between text-accent-primary font-bold border-b border-neutral-border/60 pb-2">
              <span className="text-xs bg-neutral-soft px-2.5 py-1 rounded-md border border-accent-primary">
                SCENE {scene.scene_number || idx + 1}: {scene.location}
              </span>
              <span className="text-xs text-text-secondary font-mono">{scene.duration_seconds || 6.0}s DURATION</span>
            </div>

            {/* Visual Description */}
            <p className="text-xs text-text-secondary italic pl-3 border-l-2 border-neutral-border">
              {scene.visual_description}
            </p>

            {/* Image & Video Prompts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px] font-mono">
              <div className="p-3 rounded-lg bg-neutral-soft border border-neutral-border text-text-secondary space-y-1">
                <div className="flex items-center space-x-1.5 text-accent-primary font-semibold">
                  <ImageIcon className="w-3.5 h-3.5" />
                  <span>FLUX / SDXL KEYFRAME PROMPT</span>
                </div>
                <p className="text-text-secondary">{scene.image_prompt}</p>
              </div>

              <div className="p-3 rounded-lg bg-neutral-soft border border-neutral-border text-text-secondary space-y-1">
                <div className="flex items-center space-x-1.5 text-text-secondary font-semibold">
                  <Video className="w-3.5 h-3.5" />
                  <span>COGVIDEOX / WAN MOTION PROMPT</span>
                </div>
                <p className="text-text-secondary">{scene.video_motion_prompt}</p>
              </div>
            </div>

            {/* Dialogue Block */}
            <div className="space-y-3 pt-2">
              <div className="text-[11px] text-accent-primary font-semibold flex items-center space-x-1">
                <Mic className="w-3.5 h-3.5" />
                <span>PIPER / KOKORO DIALOGUE SCRIPT</span>
              </div>

              {scene.dialogue_script && scene.dialogue_script.map((dlg, dIdx) => (
                <div key={dIdx} className="pl-6 space-y-0.5">
                  <div className="text-xs font-bold text-yellow-400 tracking-wider font-sans">{dlg.speaker.toUpperCase()}</div>
                  <div className="text-xs text-text-secondary font-sans">{dlg.line}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
