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
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-xl font-bold text-white tracking-wider font-sans uppercase">{title}</h2>
        <p className="text-xs text-slate-400 font-sans mt-1">{logline}</p>
      </div>

      {/* Scenes List */}
      <div className="space-y-8">
        {scenes.map((scene, idx) => (
          <div key={scene.id || idx} className="p-5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between text-cyan-400 font-bold border-b border-slate-800/60 pb-2">
              <span className="text-xs bg-cyan-950 px-2.5 py-1 rounded-md border border-cyan-800">
                SCENE {scene.scene_number || idx + 1}: {scene.location}
              </span>
              <span className="text-xs text-purple-400 font-mono">{scene.duration_seconds || 6.0}s DURATION</span>
            </div>

            {/* Visual Description */}
            <p className="text-xs text-slate-300 italic pl-3 border-l-2 border-slate-700">
              {scene.visual_description}
            </p>

            {/* Image & Video Prompts */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px] font-mono">
              <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-slate-400 space-y-1">
                <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold">
                  <ImageIcon className="w-3.5 h-3.5" />
                  <span>FLUX / SDXL KEYFRAME PROMPT</span>
                </div>
                <p className="text-slate-300">{scene.image_prompt}</p>
              </div>

              <div className="p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-slate-400 space-y-1">
                <div className="flex items-center space-x-1.5 text-pink-400 font-semibold">
                  <Video className="w-3.5 h-3.5" />
                  <span>COGVIDEOX / WAN MOTION PROMPT</span>
                </div>
                <p className="text-slate-300">{scene.video_motion_prompt}</p>
              </div>
            </div>

            {/* Dialogue Block */}
            <div className="space-y-3 pt-2">
              <div className="text-[11px] text-emerald-400 font-semibold flex items-center space-x-1">
                <Mic className="w-3.5 h-3.5" />
                <span>PIPER / KOKORO DIALOGUE SCRIPT</span>
              </div>

              {scene.dialogue_script && scene.dialogue_script.map((dlg, dIdx) => (
                <div key={dIdx} className="pl-6 space-y-0.5">
                  <div className="text-xs font-bold text-yellow-400 tracking-wider font-sans">{dlg.speaker.toUpperCase()}</div>
                  <div className="text-xs text-slate-200 font-sans">{dlg.line}</div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
