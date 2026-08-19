import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { VideoPlayer } from '../components/VideoPlayer';
import { EpisodeModal } from '../components/EpisodeModal';
import { api } from '../services/api';
import { Universe, Episode, AnalyticsData } from '../types';
import { Film, Zap, Play, CheckCircle2, Cpu, Activity, Clock, RefreshCw, FileText, Eye, BookOpen, ChevronDown, Globe, Brain } from 'lucide-react';
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
    onGenerateClick();
  };

  const activeUniverse = universes.find(u => u.id === activeUniverseId) || universes[0];
  const latestEpisode = episodes[0];

  return (
    <div className="space-y-10 pb-12">
      {/* Modal Inspector */}
      {selectedEpForModal && (
        <EpisodeModal episode={selectedEpForModal} onClose={() => setSelectedEpForModal(null)} />
      )}

      {/* Hero Banner / Active Universe Card */}
      <div className="relative rounded-3xl overflow-hidden p-10 bg-secondary border border-neutral-border shadow-subtle min-h-[400px]">
        {/* Background blend */}
        <div 
          className="absolute inset-0 right-0 left-1/3 opacity-30 mix-blend-multiply pointer-events-none"
          style={{ 
            backgroundImage: 'none',
            backgroundSize: 'cover',
            backgroundPosition: 'center right',
            maskImage: 'linear-gradient(to right, transparent, black 40%)',
            WebkitMaskImage: 'linear-gradient(to right, transparent, black 40%)'
          }}
        />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-start justify-between gap-12">
          <div className="space-y-5 max-w-2xl pt-2">
            <div className="inline-flex items-center space-x-2 text-[10px] font-bold tracking-widest text-accent-dark uppercase">
              <span className="w-2 h-2 rounded-full bg-accent-primary" />
              <span>AUTONOMOUS PERSISTENT STORY UNIVERSE ACTIVE</span>
            </div>
            
            <h1 className="text-5xl lg:text-6xl font-serif font-bold text-text-primary tracking-tight">
              {activeUniverse?.title || 'Mistborn'}
            </h1>
            
            <p className="text-lg text-text-primary font-medium leading-relaxed">
              {activeUniverse?.logline || 'Hero escaped from prison to fight the emperor'}
            </p>
            
            {latestEpisode?.summary && (
              <div className="flex">
                <div className="w-1 bg-neutral-border rounded-full mr-4" />
                <p className="text-sm text-text-secondary leading-relaxed">
                  Latest episode summary: {latestEpisode.summary}
                </p>
              </div>
            )}

            <div className="flex flex-wrap gap-3 pt-4 text-xs font-medium">
              <div className="bg-card px-4 py-1.5 rounded-full border border-neutral-border text-text-primary">
                <span className="text-text-secondary mr-1">Genre:</span> {activeUniverse?.genre || 'Fantasy'}
              </div>
              <div className="bg-card px-4 py-1.5 rounded-full border border-neutral-border text-text-primary">
                <span className="text-text-secondary mr-1">Episodes Generated:</span> {activeUniverse?.total_episodes || 1}
              </div>
              <div className="bg-card px-4 py-1.5 rounded-full border border-neutral-border text-text-primary">
                <span className="text-text-secondary mr-1">Season:</span> {activeUniverse?.current_season || 1}
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-5 min-w-[340px] bg-card p-6 rounded-2xl border border-neutral-border shadow-card">
            <div className="flex flex-col gap-3">
              <button
                onClick={() => {
                  const btn = document.getElementById('dash-gen-btn');
                  if (btn) btn.click();
                }}
                disabled={isGenerating}
                className="w-full px-6 py-4 rounded-xl font-bold text-[13px] bg-accent-primary text-white shadow-sm hover:bg-accent-dark transition-colors flex items-center justify-center space-x-2 tracking-wide uppercase"
              >
                <Zap className={`w-4 h-4 ${isGenerating ? 'animate-spin' : ''}`} fill="currentColor" />
                <span>{isGenerating ? 'GENERATING...' : 'GENERATE NEXT EPISODE'}</span>
              </button>
  
              <button
                onClick={() => onNavigatePage('episodes')}
                className="w-full px-6 py-4 rounded-xl font-bold text-[13px] bg-white hover:bg-neutral-soft text-text-primary border border-neutral-border transition-colors flex items-center justify-center space-x-2 tracking-wide uppercase"
              >
                <Film className="w-4 h-4 text-text-secondary" />
                <span>EXPLORE EPISODES</span>
              </button>
            </div>
            
            {/* Reference Knowledge Options */}
            <div className="pt-4 border-t border-neutral-border space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <BookOpen className="w-4 h-4 text-text-secondary" />
                  <span className="text-sm font-semibold text-text-primary">Use reference knowledge:</span>
                </div>
                <div className="flex bg-neutral-soft rounded-full p-1 text-[11px] font-bold border border-neutral-border">
                  <button 
                    onClick={() => {
                        setUseReference(true);
                        localStorage.setItem('rag_reference_enabled', 'true');
                    }}
                    className={`px-4 py-1.5 rounded-full transition-colors ${useReference ? 'bg-accent-primary text-white' : 'text-text-secondary hover:text-text-primary'}`}
                  >
                    ON
                  </button>
                  <button 
                    onClick={() => {
                        setUseReference(false);
                        localStorage.setItem('rag_reference_enabled', 'false');
                    }}
                    className={`px-4 py-1.5 rounded-full transition-colors ${!useReference ? 'bg-white text-text-primary shadow-sm border border-neutral-border' : 'text-text-secondary hover:text-text-primary'}`}
                  >
                    OFF
                  </button>
                </div>
              </div>
              
              {useReference && (
                <div className="space-y-4">
                  <div className="flex flex-col space-y-2">
                    <label className="text-text-secondary text-xs font-medium">Reference Document:</label>
                    <div className="relative">
                      <select 
                        value={selectedReference}
                        onChange={(e) => {
                            setSelectedReference(e.target.value);
                            localStorage.setItem('rag_reference_id', e.target.value);
                        }}
                        className="w-full appearance-none bg-white text-text-primary font-medium text-sm px-4 py-2.5 rounded-xl border border-neutral-border focus:border-accent-primary focus:ring-1 focus:ring-accent-primary focus:outline-none"
                      >
                        {knowledgeList.length === 0 ? (
                          <option value="">No reference uploaded</option>
                        ) : (
                          knowledgeList.map(k => (
                            <option key={k.id} value={k.id}>{k.name}</option>
                          ))
                        )}
                      </select>
                      <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary pointer-events-none" />
                    </div>
                  </div>
                  
                  <div className="flex flex-col space-y-2">
                    <label className="text-text-secondary text-xs font-medium">Influence:</label>
                    <div className="flex items-center space-x-2 text-[11px] font-semibold">
                      {['Low', 'Medium', 'High'].map(level => (
                        <button
                          key={level}
                          onClick={() => {
                              setReferenceInfluence(level);
                              localStorage.setItem('rag_reference_influence', level);
                          }}
                          className={`flex-1 py-2 rounded-lg border transition-colors ${
                            referenceInfluence === level 
                            ? 'bg-accent-primary border-accent-primary text-white' 
                            : 'bg-white border-neutral-border text-text-secondary hover:bg-neutral-soft'
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
            
            {/* Hidden button for triggering generation */}
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
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <GlassCard className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 rounded-full bg-accent-light flex items-center justify-center text-accent-dark">
                <Globe className="w-5 h-5" strokeWidth={1.5} />
              </div>
              <div className="flex flex-col">
                <p className="text-[10px] text-text-secondary font-bold tracking-widest uppercase">TOTAL UNIVERSES</p>
                <h3 className="text-3xl font-serif font-bold text-text-primary mt-1">{analytics?.total_universes || 1}</h3>
              </div>
            </div>
          </div>
        </GlassCard>

        <GlassCard className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 rounded-full bg-neutral-soft flex items-center justify-center text-text-secondary">
                <Film className="w-5 h-5" strokeWidth={1.5} />
              </div>
              <div className="flex flex-col">
                <p className="text-[10px] text-text-secondary font-bold tracking-widest uppercase">EPISODES RENDERED</p>
                <h3 className="text-3xl font-serif font-bold text-text-primary mt-1">{analytics?.total_episodes || 1}</h3>
              </div>
            </div>
          </div>
        </GlassCard>

        <GlassCard className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 rounded-full bg-accent-light flex items-center justify-center text-accent-dark">
                <Brain className="w-5 h-5" strokeWidth={1.5} />
              </div>
              <div className="flex flex-col">
                <p className="text-[10px] text-text-secondary font-bold tracking-widest uppercase">STORY MEMORY ENTRIES</p>
                <h3 className="text-3xl font-serif font-bold text-text-primary mt-1">{analytics?.system_status.qdrant_indexed_vectors || 128}</h3>
              </div>
            </div>
          </div>
        </GlassCard>

        <GlassCard className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="w-12 h-12 rounded-full bg-neutral-soft flex items-center justify-center text-text-secondary">
                <Cpu className="w-5 h-5" strokeWidth={1.5} />
              </div>
              <div className="flex flex-col">
                <p className="text-[10px] text-text-secondary font-bold tracking-widest uppercase">VRAM ALLOCATION</p>
                <h3 className="text-3xl font-serif font-bold text-text-primary mt-1">{analytics?.system_status.gpu_vram_allocated_gb || '8.4 / 16.0 GB'}</h3>
              </div>
            </div>
          </div>
        </GlassCard>
      </div>

      {/* Main Grid: Video Player + Recent Episodes */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        <div className="lg:col-span-2 space-y-5">
          <h2 className="text-sm font-bold text-text-secondary tracking-widest uppercase flex items-center justify-between">
            <span>LATEST RENDERED EPISODE</span>
            <button onClick={loadData} className="text-text-secondary hover:text-accent-primary transition-colors">
              <RefreshCw className="w-4 h-4" />
            </button>
          </h2>
          
          <div className="rounded-2xl overflow-hidden shadow-card border border-neutral-border bg-card">
            {latestEpisode ? (
              <VideoPlayer
                videoUrl={latestEpisode.final_video_url}
                posterUrl={latestEpisode.thumbnail_url}
                title={latestEpisode.title || 'Episode 1: Signals in the Rain'}
                subtitles="SRT Subtitles Burnt-in"
                scenes={latestEpisode.scenes}
              />
            ) : (
              <div className="aspect-video bg-neutral-soft flex items-center justify-center">
                <div className="text-center text-text-secondary">
                  <Film className="w-12 h-12 mx-auto mb-3 opacity-20" />
                  <p className="font-medium">No episodes generated yet</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Recent Episode Feed with Watch Video & Read Script Actions */}
        <div className="space-y-5">
          <h2 className="text-sm font-bold text-text-secondary tracking-widest uppercase">EPISODES FEED</h2>
          <div className="space-y-4">
            {episodes.map((ep) => (
              <GlassCard
                key={ep.id}
                className={`hover:border-accent-primary cursor-pointer space-y-4 transition-all duration-200 ${ep.id === latestEpisode?.id ? 'border-accent-primary/50 bg-secondary/30' : ''}`}
                onClick={() => setSelectedEpForModal(ep)}
              >
                <div className="flex items-start space-x-4">
                  <div className="w-12 h-12 rounded-xl bg-accent-primary flex items-center justify-center font-bold text-white text-sm shrink-0">
                    E{ep.episode_number}
                  </div>
                  <div className="space-y-1.5 flex-1 pt-0.5">
                    <h4 className="font-serif font-bold text-base text-text-primary hover:text-accent-primary transition-colors">{ep.title}</h4>
                    <p className="text-xs text-text-secondary line-clamp-2 leading-relaxed">{ep.logline}</p>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-3 border-t border-neutral-border/60 text-xs font-medium text-text-secondary">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-1.5">
                      <Clock className="w-3.5 h-3.5" />
                      <span>{new Date(ep.created_at || Date.now()).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <span className="bg-neutral-soft px-2 py-1 rounded-md">{ep.duration_seconds || 30.0}s</span>
                </div>
              </GlassCard>
            ))}
            
            {episodes.length === 0 && (
              <div className="text-center py-12 text-text-secondary border border-dashed border-neutral-border rounded-2xl">
                <p className="font-medium">No episodes found.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

