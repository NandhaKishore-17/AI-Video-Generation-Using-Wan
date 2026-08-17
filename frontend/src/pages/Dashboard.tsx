import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { VideoPlayer } from '../components/VideoPlayer';
import { EpisodeModal } from '../components/EpisodeModal';
import { api } from '../services/api';
import { Universe, Episode, AnalyticsData } from '../types';
import { Film, Zap, Play, CheckCircle2, Cpu, Activity, Clock, RefreshCw, FileText, Eye, BookOpen } from 'lucide-react';
import { KnowledgeDocument } from '../types';

interface DashboardProps {
  onNavigatePage: (page: any) => void;
  onGenerateClick: () => void;
  isGenerating: boolean;
  activeUniverseId?: string;
  refreshTrigger?: number;
}

export const Dashboard: React.FC<DashboardProps> = ({ onNavigatePage, onGenerateClick, isGenerating, activeUniverseId, refreshTrigger }) => {
  const [universes, setUniverses] = useState<Universe[]>([]);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [selectedEpForModal, setSelectedEpForModal] = useState<Episode | null>(null);
  
  // Reference RAG State
  const [useReference, setUseReference] = useState(() => {
    return localStorage.getItem('rag_reference_enabled') === 'true';
  });
  const [knowledgeList, setKnowledgeList] = useState<KnowledgeDocument[]>([]);
  const [selectedReference, setSelectedReference] = useState<string>(() => {
    return localStorage.getItem('rag_reference_id') || '';
  });
  const [referenceInfluence, setReferenceInfluence] = useState<string>(() => {
    return localStorage.getItem('rag_reference_influence') || 'Medium';
  });

  useEffect(() => {
    loadData();
  }, [activeUniverseId, refreshTrigger]);


  const loadData = async () => {
    const uData = await api.getUniverses();
    setUniverses(uData);
    const targetUId = activeUniverseId || (uData.length > 0 ? uData[0].id : undefined);
    const epData = await api.getEpisodes(targetUId);
    setEpisodes(epData);
    const aData = await api.getAnalytics();
    setAnalytics(aData);
    
    // Load knowledge documents
    const kData = await api.getKnowledgeList();
    setKnowledgeList(kData);
    if (kData.length > 0 && !selectedReference) {
      setSelectedReference(kData[0].id);
    }
  };

  const handleGenerateWithRAG = async () => {
    if (!activeUniverseId) return;
    
    onGenerateClick(); // Keep parent tracking if needed, wait actually the parent does the generation...
    // Let's modify the parent's handler or just do it here. 
    // It's cleaner to dispatch the generation here.
  };

  const activeUniverse = universes.find(u => u.id === activeUniverseId) || universes[0];
  const latestEpisode = episodes[0];

  return (
    <div className="space-y-8">
      {/* Modal Inspector */}
      {selectedEpForModal && (
        <EpisodeModal episode={selectedEpForModal} onClose={() => setSelectedEpForModal(null)} />
      )}

      {/* Hero Banner / Active Universe Card */}
      <div className="relative rounded-3xl overflow-hidden p-8 bg-gradient-to-r from-slate-900 via-purple-950 to-cyan-950 border border-slate-700/60 shadow-glow-cyan">
        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-3 max-w-2xl">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-cyan-950/80 border border-cyan-800 text-cyan-400 text-xs font-mono">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              <span>AUTONOMOUS PERSISTENT STORY UNIVERSE ACTIVE</span>
            </div>
            <h1 className="text-3xl lg:text-4xl font-extrabold text-white tracking-tight">
              {activeUniverse?.title || 'Neo-Tokyo 2099'}
            </h1>
            <p className="text-sm text-slate-300 leading-relaxed">
              {activeUniverse?.logline || 'In a rain-drenched megacity governed by rogue AI networks, a hacker and a detective battle corporate overlords.'}
            </p>
            {latestEpisode?.summary && (
              <p className="text-xs text-slate-400 mt-3 border-l-2 border-cyan-500 pl-3">
                Latest episode summary: {latestEpisode.summary}
              </p>
            )}

            <div className="flex flex-wrap gap-4 pt-2 text-xs font-mono">
              <div className="bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300">
                <span className="text-slate-500">Genre:</span> {activeUniverse?.genre || 'Cyberpunk Noir'}
              </div>
              <div className="bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300">
                <span className="text-slate-500">Episodes Generated:</span> {activeUniverse?.total_episodes || 5}
              </div>
              <div className="bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300">
                <span className="text-slate-500">Season:</span> {activeUniverse?.current_season || 1}
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row gap-3">
              <button
                onClick={() => {
                  const btn = document.getElementById('dash-gen-btn');
                  if (btn) btn.click();
                }}
                disabled={isGenerating}
                className="px-6 py-3.5 rounded-xl font-bold text-sm bg-gradient-to-r from-cyan-500 via-purple-600 to-pink-500 text-white shadow-glow-cyan hover:scale-105 transition-all duration-300 flex items-center justify-center space-x-2"
              >
                <Zap className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} />
                <span>{isGenerating ? 'GENERATING EPISODE...' : 'GENERATE NEXT EPISODE'}</span>
              </button>
  
              <button
                onClick={() => onNavigatePage('episodes')}
                className="px-5 py-3.5 rounded-xl font-semibold text-sm bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 flex items-center justify-center space-x-2"
              >
                <Film className="w-4 h-4 text-cyan-400" />
                <span>EXPLORE EPISODES</span>
              </button>
            </div>
            
            {/* Reference Knowledge Options */}
            <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-700/50 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <BookOpen className="w-4 h-4 text-emerald-400" />
                  <span className="text-sm font-semibold text-slate-200">Use reference knowledge:</span>
                </div>
                <div className="flex bg-slate-800 rounded-lg p-1 text-xs font-bold">
                  <button 
                    onClick={() => {
                        setUseReference(true);
                        localStorage.setItem('rag_reference_enabled', 'true');
                    }}
                    className={`px-3 py-1 rounded-md transition-colors ${useReference ? 'bg-emerald-500 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                  >
                    ON
                  </button>
                  <button 
                    onClick={() => {
                        setUseReference(false);
                        localStorage.setItem('rag_reference_enabled', 'false');
                    }}
                    className={`px-3 py-1 rounded-md transition-colors ${!useReference ? 'bg-slate-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                  >
                    OFF
                  </button>
                </div>
              </div>
              
              {useReference && (
                <div className="space-y-3 pt-2 border-t border-slate-700/50">
                  <div className="flex flex-col space-y-1 text-sm">
                    <label className="text-slate-400 text-xs font-bold">Reference Document:</label>
                    <select 
                      value={selectedReference}
                      onChange={(e) => {
                          setSelectedReference(e.target.value);
                          localStorage.setItem('rag_reference_id', e.target.value);
                      }}
                      className="bg-slate-950 text-emerald-300 font-semibold px-3 py-2 rounded-xl border border-slate-700 focus:border-emerald-500 focus:outline-none"
                    >
                      {knowledgeList.length === 0 ? (
                        <option value="">No reference uploaded (Upload in Knowledge Library)</option>
                      ) : (
                        knowledgeList.map(k => (
                          <option key={k.id} value={k.id}>✓ {k.name}</option>
                        ))
                      )}
                    </select>
                  </div>
                  
                  <div className="flex flex-col space-y-1 text-sm">
                    <label className="text-slate-400 text-xs font-bold">Influence:</label>
                    <div className="flex items-center space-x-2 text-xs font-bold">
                      {['Low', 'Medium', 'High'].map(level => (
                        <button
                          key={level}
                          onClick={() => {
                              setReferenceInfluence(level);
                              localStorage.setItem('rag_reference_influence', level);
                          }}
                          className={`flex-1 py-1.5 rounded-lg border transition-colors ${
                            referenceInfluence === level 
                            ? 'bg-emerald-900/60 border-emerald-500 text-emerald-400' 
                            : 'bg-slate-800 border-slate-700 text-slate-400 hover:bg-slate-700'
                          }`}
                        >
                          {referenceInfluence === level ? '●' : '○'} {level}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            {/* Hidden button for triggering generation from App.tsx via DOM hack or we just pass the values up */}
            <button
              id="dash-gen-btn"
              className="hidden"
              onClick={() => {
                onGenerateClick();
              }}
            />
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <GlassCard>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 font-mono">TOTAL UNIVERSES</p>
              <h3 className="text-2xl font-extrabold text-white mt-1">{analytics?.total_universes || 2}</h3>
            </div>
            <div className="p-3 rounded-xl bg-cyan-950/60 text-cyan-400 border border-cyan-800">
              <Film className="w-6 h-6" />
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 font-mono">EPISODES RENDERED</p>
              <h3 className="text-2xl font-extrabold text-purple-400 mt-1">{analytics?.total_episodes || 7}</h3>
            </div>
            <div className="p-3 rounded-xl bg-purple-950/60 text-purple-400 border border-purple-800">
              <CheckCircle2 className="w-6 h-6" />
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 font-mono">STORY MEMORY ENTRIES</p>
              <h3 className="text-2xl font-extrabold text-pink-400 mt-1">{analytics?.system_status.qdrant_indexed_vectors || 128}</h3>
            </div>
            <div className="p-3 rounded-xl bg-pink-950/60 text-pink-400 border border-pink-800">
              <Activity className="w-6 h-6" />
            </div>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 font-mono">VRAM ALLOCATION</p>
              <h3 className="text-2xl font-extrabold text-emerald-400 mt-1">{analytics?.system_status.gpu_vram_allocated_gb || '8.4 GB'}</h3>
            </div>
            <div className="p-3 rounded-xl bg-emerald-950/60 text-emerald-400 border border-emerald-800">
              <Cpu className="w-6 h-6" />
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Main Grid: Video Player + Recent Episodes */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-4">
          <h2 className="text-lg font-bold text-white flex items-center justify-between">
            <span>LATEST RENDERED EPISODE</span>
            <button onClick={loadData} className="p-1.5 rounded-lg bg-slate-900 text-slate-400 hover:text-white border border-slate-800">
              <RefreshCw className="w-4 h-4" />
            </button>
          </h2>
          <VideoPlayer
            videoUrl={latestEpisode?.final_video_url}
            posterUrl={latestEpisode?.thumbnail_url}
            title={latestEpisode?.title || 'Episode 1: Signals in the Rain'}
            subtitles="SRT Subtitles Burnt-in"
            scenes={latestEpisode?.scenes}
          />
        </div>

        {/* Recent Episode Feed with Watch Video & Read Script Actions */}
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-white">EPISODES FEED</h2>
          <div className="space-y-3">
            {episodes.map((ep) => (
              <GlassCard
                key={ep.id}
                className="hover:border-cyan-500/50 cursor-pointer space-y-3"
                glow={ep.id === latestEpisode?.id}
                onClick={() => setSelectedEpForModal(ep)}
              >
                <div className="flex items-start space-x-3">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 to-purple-600 flex items-center justify-center font-bold text-white text-xs shrink-0">
                    E{ep.episode_number}
                  </div>
                  <div className="space-y-1 flex-1">
                    <h4 className="font-bold text-sm text-white hover:text-cyan-400 transition-colors">{ep.title}</h4>
                    <p className="text-xs text-slate-400 line-clamp-2">{ep.logline}</p>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-slate-800/60 text-xs font-mono">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedEpForModal(ep);
                    }}
                    className="flex items-center space-x-1 text-cyan-400 hover:text-cyan-300"
                  >
                    <Eye className="w-3.5 h-3.5" />
                    <span>WATCH & READ SCRIPT</span>
                  </button>

                  <span className="text-[10px] text-purple-400">{ep.duration_seconds || 18.0}s</span>
                </div>
              </GlassCard>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
