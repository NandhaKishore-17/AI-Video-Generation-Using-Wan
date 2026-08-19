import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { VideoPlayer } from '../components/VideoPlayer';
import { ScreenplayView } from '../components/ScreenplayView';
import { api } from '../services/api';
import { Episode, Scene } from '../types';
import { Library, Film, FileText, Play, CheckCircle2, Download } from 'lucide-react';

interface EpisodeLibraryProps {
  activeUniverseId?: string;
  refreshTrigger?: number;
}

export const EpisodeLibrary: React.FC<EpisodeLibraryProps> = ({ activeUniverseId, refreshTrigger }) => {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [selectedEp, setSelectedEp] = useState<Episode | null>(null);
  const [activeTab, setActiveTab] = useState<'video' | 'screenplay'>('video');

  useEffect(() => {
    loadEpisodes();
  }, [activeUniverseId, refreshTrigger]);


  const loadEpisodes = async () => {
    const data = await api.getEpisodes(activeUniverseId);
    setEpisodes(data);
    if (data.length > 0) {
      const detail = await api.getEpisodeDetail(data[0].id);
      setSelectedEp(detail);
    } else {
      setSelectedEp(null);
    }
  };

  const handleSelect = async (ep: Episode) => {
    const detail = await api.getEpisodeDetail(ep.id);
    setSelectedEp(detail);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-text-primary tracking-tight flex items-center gap-2">
          <Library className="w-6 h-6 text-accent-primary" />
          <span>EPISODE LIBRARY & SCREENPLAY VAULT</span>
        </h1>
        <p className="text-xs text-text-secondary mt-1">
          Browse generated story episodes, view shot-by-shot screenplays, and play final MP4 renders.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Episode Selector List */}
        <div className="lg:col-span-1 space-y-4">
          <h3 className="text-base font-bold text-text-primary">EPISODES</h3>
          <div className="space-y-3">
            {episodes.map((ep) => (
              <GlassCard
                key={ep.id}
                glow={ep.id === selectedEp?.id}
                className="cursor-pointer hover:border-cyan-500/60 space-y-3"
                onClick={() => handleSelect(ep)}
              >
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-accent-primary font-bold">
                      SEASON {ep.season} EPISODE {ep.episode_number}
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-accent-light text-accent-dark border border-accent-primary">
                      {ep.status}
                    </span>
                  </div>
                  <h4 className="text-sm font-bold text-text-primary">{ep.title}</h4>
                  <p className="text-xs text-text-secondary line-clamp-2">{ep.logline}</p>
                  {ep.summary && (
                    <p className="text-[11px] text-text-secondary line-clamp-2">{ep.summary}</p>
                  )}
                </div>

                <div className="flex items-center space-x-2 pt-3 border-t border-neutral-border/60 text-[11px] font-mono">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSelect(ep);
                      setActiveTab('video');
                    }}
                    className="flex-1 py-1.5 rounded-lg bg-accent-primary hover:bg-accent-dark border border-transparent text-white font-bold flex items-center justify-center space-x-1 shadow-sm transition-colors"
                  >
                    <Film className="w-3 h-3" />
                    <span>WATCH VIDEO</span>
                  </button>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSelect(ep);
                      setActiveTab('screenplay');
                    }}
                    className="flex-1 py-1.5 rounded-lg bg-card hover:bg-neutral-soft border border-neutral-border text-text-primary font-bold flex items-center justify-center space-x-1 shadow-sm transition-colors"
                  >
                    <FileText className="w-3 h-3" />
                    <span>READ SCRIPT</span>
                  </button>

                  {ep.final_video_url && (
                    <a
                      href={ep.final_video_url}
                      download
                      onClick={(e) => e.stopPropagation()}
                      className="p-1.5 rounded-lg bg-card hover:bg-neutral-soft text-text-secondary hover:text-accent-primary border border-neutral-border transition-colors shadow-sm"
                      title="Download MP4 Video"
                    >
                      <Download className="w-3.5 h-3.5" />
                    </a>
                  )}
                </div>
              </GlassCard>
            ))}
          </div>
        </div>

        {/* Selected Episode Detail Area */}
        <div className="lg:col-span-2 space-y-6">
          {selectedEp && (
            <GlassCard glow className="space-y-6">
              {/* Tab Selector */}
              <div className="flex items-center justify-between border-b border-neutral-border pb-4">
                <div>
                  <h2 className="text-xl font-serif font-bold text-text-primary">{selectedEp.title}</h2>
                  <p className="text-xs text-text-secondary mt-0.5">{selectedEp.logline}</p>
                </div>

                <div className="flex items-center space-x-2 bg-neutral-soft p-1 rounded-xl border border-neutral-border">
                  <button
                    onClick={() => setActiveTab('video')}
                    className={`flex items-center space-x-1.5 px-4 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                      activeTab === 'video' ? 'bg-accent-primary text-white font-bold shadow-sm' : 'text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    <Film className="w-3.5 h-3.5" />
                    <span>FINAL VIDEO</span>
                  </button>

                  <button
                    onClick={() => setActiveTab('screenplay')}
                    className={`flex items-center space-x-1.5 px-4 py-1.5 rounded-lg text-xs font-mono transition-colors ${
                      activeTab === 'screenplay' ? 'bg-accent-primary text-white font-bold shadow-sm' : 'text-text-secondary hover:text-text-primary'
                    }`}
                  >
                    <FileText className="w-3.5 h-3.5" />
                    <span>SCREENPLAY</span>
                  </button>
                </div>
              </div>

              {/* Tab Content */}
              {activeTab === 'video' ? (
                <VideoPlayer
                  videoUrl={selectedEp.final_video_url}
                  posterUrl={selectedEp.thumbnail_url}
                  title={selectedEp.title}
                  subtitles="SRT Subtitles Burnt-in"
                  scenes={selectedEp.scenes}
                />
              ) : (
                <ScreenplayView
                  scenes={selectedEp.scenes || []}
                  title={selectedEp.title}
                  logline={selectedEp.logline}
                />
              )}
            </GlassCard>
          )}
        </div>
      </div>
    </div>
  );
};
