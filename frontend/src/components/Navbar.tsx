import React from 'react';
import { Leaf, Globe, Clock, Cpu, Film, Mic, Zap } from 'lucide-react';
import { Universe } from '../types';

interface NavbarProps {
  onGenerateClick: (episodeDuration?: number) => void;
  isGenerating: boolean;
  universes: Universe[];
  activeUniverseId: string;
  onSelectUniverse: (id: string) => void;
  episodeDuration: number;
  onEpisodeDurationChange: (sec: number) => void;
  onLogoClick?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  onGenerateClick,
  isGenerating,
  universes,
  activeUniverseId,
  onSelectUniverse,
  episodeDuration,
  onEpisodeDurationChange,
  onLogoClick
}) => {
  return (
    <header className="h-16 border-b border-neutral-border bg-background sticky top-0 z-50 px-6 flex items-center justify-between shadow-subtle">
      {/* Left Section: Logo & Badges */}
      <div className="flex items-center space-x-6">
        <div onClick={onLogoClick} className="flex items-center space-x-3 cursor-pointer group">
          <Leaf className="w-8 h-8 text-accent-primary" strokeWidth={1.5} />
          <div className="flex flex-col">
            <span className="font-serif font-bold text-xl tracking-tight text-text-primary">
              STORYUNIVERSE
            </span>
            <span className="text-[10px] text-text-secondary tracking-widest font-medium uppercase -mt-1">
              AI Story Platform
            </span>
          </div>
        </div>

        <div className="hidden lg:block pl-6 border-l border-neutral-border">
          <span className="px-3 py-1 rounded-md bg-accent-light text-accent-dark text-[10px] font-bold tracking-widest uppercase">
            AUTONOMOUS OS
          </span>
        </div>
      </div>

      {/* Right Section: Controls & Models */}
      <div className="flex items-center space-x-4">
        
        {/* Universe Selector Dropdown */}
        <div className="hidden md:flex items-center space-x-2 text-xs">
          <Globe className="w-4 h-4 text-text-secondary" strokeWidth={1.5} />
          <span className="text-text-secondary font-medium tracking-wide">UNIVERSE:</span>
          <select
            value={activeUniverseId}
            onChange={(e) => onSelectUniverse(e.target.value)}
            className="bg-card text-text-primary font-semibold px-3 py-1.5 rounded-lg border border-neutral-border focus:border-accent-primary focus:ring-1 focus:ring-accent-primary focus:outline-none cursor-pointer text-xs"
          >
            {universes.length === 0 ? (
              <option value="">No Universes Created</option>
            ) : (
              universes.map((u) => (
                <option key={u.id} value={u.id}>
                  {u.title} ({u.genre})
                </option>
              ))
            )}
          </select>
        </div>

        {/* Timing Selector */}
        <div className="hidden sm:flex items-center space-x-2 text-xs border-l border-neutral-border pl-4">
          <Clock className="w-4 h-4 text-text-secondary" strokeWidth={1.5} />
          <span className="text-text-secondary font-medium tracking-wide uppercase text-[10px]">Episode Length:</span>
          <select
            value={episodeDuration}
            onChange={(e) => onEpisodeDurationChange(Number(e.target.value))}
            className="bg-card text-text-primary font-semibold px-3 py-1.5 rounded-lg border border-neutral-border focus:border-accent-primary focus:outline-none cursor-pointer text-xs"
          >
            <option value={30}>30 sec</option>
            <option value={45}>45 sec</option>
            <option value={60}>60 sec</option>
          </select>
        </div>

        {/* AI Stack Engine Badges */}
        <div className="hidden xl:flex items-center space-x-4 px-4 py-1.5 rounded-lg border border-neutral-border bg-card">
          <div className="flex items-center space-x-1.5">
            <Cpu className="w-3.5 h-3.5 text-text-secondary" />
            <div className="flex flex-col">
              <span className="text-[9px] uppercase tracking-wider text-text-secondary">LLM</span>
              <span className="text-[11px] font-bold text-text-primary">Gemma 3:4B</span>
            </div>
          </div>
          <div className="w-px h-6 bg-neutral-border"></div>
          <div className="flex items-center space-x-1.5">
            <Film className="w-3.5 h-3.5 text-text-secondary" />
            <div className="flex flex-col">
              <span className="text-[9px] uppercase tracking-wider text-text-secondary">Video</span>
              <span className="text-[11px] font-bold text-text-primary">Wan 2.2 5B</span>
            </div>
          </div>
          <div className="w-px h-6 bg-neutral-border"></div>
          <div className="flex items-center space-x-1.5">
            <Mic className="w-3.5 h-3.5 text-text-secondary" />
            <div className="flex flex-col">
              <span className="text-[9px] uppercase tracking-wider text-text-secondary">Voice</span>
              <span className="text-[11px] font-bold text-text-primary">Piper / Kokoro</span>
            </div>
          </div>
        </div>

        {/* Primary Auto-Generate Episode Button */}
        <button
          onClick={() => onGenerateClick(episodeDuration)}
          disabled={isGenerating}
          className={`flex items-center space-x-2 px-5 py-2.5 rounded-lg font-bold text-[11px] tracking-wide transition-all shadow-sm ${
            isGenerating
              ? 'bg-neutral-soft text-text-secondary border border-neutral-border cursor-not-allowed opacity-80'
              : 'bg-accent-primary hover:bg-accent-dark text-white border border-accent-dark'
          }`}
        >
          <Zap className={`w-3.5 h-3.5 ${isGenerating ? 'animate-spin' : ''}`} fill="currentColor" />
          <span className="uppercase">{isGenerating ? 'GENERATING EPISODE...' : 'AUTO-GENERATE EPISODE'}</span>
        </button>
      </div>
    </header>
  );
};
