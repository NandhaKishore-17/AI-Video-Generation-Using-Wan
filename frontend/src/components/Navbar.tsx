import React from 'react';
import { Film, Zap, Globe, Cpu, Clock } from 'lucide-react';
import { Universe } from '../types';

interface NavbarProps {
  onGenerateClick: (sceneDuration?: number) => void;
  isGenerating: boolean;
  universes: Universe[];
  activeUniverseId: string;
  onSelectUniverse: (id: string) => void;
  sceneDuration: number;
  onSceneDurationChange: (sec: number) => void;
  onLogoClick?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  onGenerateClick,
  isGenerating,
  universes,
  activeUniverseId,
  onSelectUniverse,
  sceneDuration,
  onSceneDurationChange,
  onLogoClick
}) => {
  return (
    <header className="h-16 border-b border-slate-800/80 bg-[#0B0F19]/95 backdrop-blur-xl sticky top-0 z-50 px-6 flex items-center justify-between shadow-2xl">
      {/* Left Section: Logo, Title & Universe Switcher */}
      <div className="flex items-center space-x-6">
        <div onClick={onLogoClick} className="flex items-center space-x-3 cursor-pointer group">

          <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-500 via-purple-600 to-pink-500 shadow-glow-cyan">
            <Film className="w-5 h-5 text-white" />
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="font-black text-lg tracking-tight text-white">
              STORY<span className="text-cyan-400">UNIVERSE</span>
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-700/60 font-mono font-bold tracking-wider uppercase">
              AUTONOMOUS OS
            </span>
          </div>
        </div>

        {/* Universe Selector Dropdown */}
        <div className="hidden md:flex items-center space-x-2 text-xs pl-6 border-l border-slate-800">
          <Globe className="w-4 h-4 text-cyan-400 shrink-0" />
          <span className="text-slate-400 font-mono font-bold shrink-0">UNIVERSE:</span>
          <select
            value={activeUniverseId}
            onChange={(e) => onSelectUniverse(e.target.value)}
            className="bg-slate-950 text-cyan-300 font-semibold px-3 py-1.5 rounded-xl border border-slate-800 focus:border-cyan-500 focus:outline-none cursor-pointer text-xs max-w-[240px] truncate shadow-inner"
          >
            {universes.length === 0 ? (
              <option value="">No Universes Created</option>
            ) : (
              universes.map((u) => (
                <option key={u.id} value={u.id} className="bg-slate-900 text-white">
                  {u.title} ({u.genre})
                </option>
              ))
            )}
          </select>
        </div>
      </div>

      {/* Right Section: Timing, AI Models Badge & Action Button */}
      <div className="flex items-center space-x-4">
        {/* Timing Selector */}
        <div className="hidden sm:flex items-center space-x-2 bg-slate-950/90 px-3 py-1.5 rounded-xl border border-slate-800/80 text-xs font-mono">
          <Clock className="w-3.5 h-3.5 text-purple-400" />
          <span className="text-slate-400 font-bold">TIMING:</span>
          <select
            value={sceneDuration}
            onChange={(e) => onSceneDurationChange(Number(e.target.value))}
            className="bg-transparent text-purple-300 font-bold focus:outline-none cursor-pointer text-xs"
          >
            <option value={5} className="bg-slate-900 text-white">5s / scene</option>
            <option value={8} className="bg-slate-900 text-white">8s / scene</option>
            <option value={12} className="bg-slate-900 text-white">12s / scene</option>
            <option value={15} className="bg-slate-900 text-white">15s / scene</option>
          </select>
        </div>

        {/* AI Stack Engine Badges */}
        <div className="hidden xl:flex items-center space-x-2 text-[11px] font-mono bg-slate-950/80 px-3.5 py-1.5 rounded-xl border border-slate-800/80">
          <Cpu className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-cyan-400 font-bold">Qwen 2.5</span>
          <span className="text-slate-700">•</span>
          <span className="text-purple-400 font-medium">FLUX / SDXL</span>
          <span className="text-slate-700">•</span>
          <span className="text-pink-400 font-medium">CogVideoX</span>
          <span className="text-slate-700">•</span>
          <span className="text-emerald-400 font-medium">Piper / Kokoro</span>
        </div>

        {/* Primary Auto-Generate Episode Button */}
        <button
          onClick={() => onGenerateClick(sceneDuration)}
          disabled={isGenerating}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl font-bold text-xs tracking-wider transition-all duration-300 shadow-xl ${
            isGenerating
              ? 'bg-purple-950 text-purple-300 border border-purple-700 cursor-not-allowed animate-pulse'
              : 'bg-gradient-to-r from-cyan-500 via-purple-600 to-pink-500 hover:brightness-110 text-white shadow-glow-cyan active:scale-95'
          }`}
        >
          <Zap className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} />
          <span>{isGenerating ? 'GENERATING EPISODE...' : 'AUTO-GENERATE EPISODE'}</span>
        </button>
      </div>
    </header>
  );
};
