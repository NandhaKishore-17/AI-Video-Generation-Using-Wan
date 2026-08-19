import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { api } from '../services/api';
import { TimelineEvent, StoryArc } from '../types';
import { Clock, Flame, Plus, Trash2, RefreshCw, X, Sparkles, CheckCircle2 } from 'lucide-react';

interface TimelineProps {
  activeUniverseId?: string;
}

export const Timeline: React.FC<TimelineProps> = ({ activeUniverseId }) => {
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [arcs, setArcs] = useState<StoryArc[]>([]);
  const [currentUniverseId, setCurrentUniverseId] = useState<string>('');
  const [loading, setLoading] = useState(false);

  // Modal states
  const [showEventModal, setShowEventModal] = useState(false);
  const [showArcModal, setShowArcModal] = useState(false);

  // Event form
  const [evtTimestamp, setEvtTimestamp] = useState('Year 2099');
  const [evtTitle, setEvtTitle] = useState('');
  const [evtDesc, setEvtDesc] = useState('');
  const [evtImportance, setEvtImportance] = useState(8);

  // Arc form
  const [arcTitle, setArcTitle] = useState('');
  const [arcGoal, setArcGoal] = useState('');
  const [arcEpisodesPlanned, setArcEpisodesPlanned] = useState(5);

  useEffect(() => {
    loadData();
  }, [activeUniverseId]);

  const loadData = async () => {
    setLoading(true);
    const universes = await api.getUniverses();
    const targetUId = activeUniverseId || (universes.length > 0 ? universes[0].id : 'u-cyber-99');
    setCurrentUniverseId(targetUId);
    const detail = await api.getUniverseDetail(targetUId);
    setTimeline(detail.timeline || []);
    setArcs(detail.story_arcs || []);
    setLoading(false);
  };

  const handleCreateEvent = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!evtTitle || !evtDesc) return;
    await api.createTimelineEvent({
      universe_id: currentUniverseId,
      timestamp_in_universe: evtTimestamp,
      title: evtTitle,
      description: evtDesc,
      importance_score: evtImportance,
      season: 1
    });
    setEvtTitle('');
    setEvtDesc('');
    setShowEventModal(false);
    loadData();
  };

  const handleCreateArc = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!arcTitle || !arcGoal) return;
    await api.createStoryArc({
      universe_id: currentUniverseId,
      title: arcTitle,
      goal: arcGoal,
      season: 1,
      episodes_planned: arcEpisodesPlanned
    });
    setArcTitle('');
    setArcGoal('');
    setShowArcModal(false);
    loadData();
  };

  const handleDeleteEvent = async (id: string) => {
    if (window.confirm("Delete this timeline event?")) {
      await api.deleteTimelineEvent(id);
      loadData();
    }
  };

  const handleDeleteArc = async (id: string) => {
    if (window.confirm("Delete this story arc?")) {
      await api.deleteStoryArc(id);
      loadData();
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-text-primary tracking-tight flex items-center gap-2">
            <Clock className="w-6 h-6 text-text-secondary" />
            <span>CHRONOLOGICAL TIMELINE & STORY ARCS</span>
          </h1>
          <p className="text-xs text-text-secondary mt-1">
            Track universe milestone history, season progression, and story tension arcs across infinite episodes.
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={loadData}
            disabled={loading}
            className="p-2.5 rounded-xl bg-white hover:bg-neutral-soft text-text-secondary border border-neutral-border transition-colors"
            title="Refresh Timeline Data"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>

          <button
            onClick={() => setShowArcModal(true)}
            className="px-3.5 py-2 rounded-xl text-xs font-mono font-bold bg-neutral-soft hover:bg-neutral-soft text-text-secondary border border-neutral-border flex items-center space-x-1.5 transition-all shadow-lg"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>ADD STORY ARC</span>
          </button>

          <button
            onClick={() => setShowEventModal(true)}
            className="px-3.5 py-2 rounded-xl text-xs font-mono font-bold bg-accent-primary hover:brightness-110 text-text-primary flex items-center space-x-1.5 transition-all shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>ADD MILESTONE EVENT</span>
          </button>
        </div>
      </div>

      {/* Story Arcs Section */}
      <div className="space-y-4">
        <h3 className="text-base font-bold text-text-primary flex items-center space-x-2">
          <Flame className="w-4 h-4 text-text-secondary" />
          <span>ACTIVE & PLANNED STORY ARCS ({arcs.length})</span>
        </h3>

        {arcs.length === 0 ? (
          <GlassCard className="p-6 text-center text-text-secondary text-xs">
            No story arcs defined yet. Click "ADD STORY ARC" above to create your first narrative arc!
          </GlassCard>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {arcs.map((arc) => {
              const progressPct = Math.min(100, Math.max(0, (arc.episodes_completed / (arc.episodes_planned || 1)) * 100));
              return (
                <GlassCard key={arc.id} glow={arc.status === 'ACTIVE'} className="space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="font-bold text-text-primary text-sm">{arc.title}</h4>
                    <div className="flex items-center space-x-2">
                      <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${
                        arc.status === 'ACTIVE'
                          ? 'bg-neutral-soft text-accent-primary border-accent-primary'
                          : arc.status === 'COMPLETED'
                          ? 'bg-neutral-soft text-accent-primary border-accent-primary'
                          : 'bg-white text-text-secondary border-neutral-border'
                      }`}>
                        {arc.status}
                      </span>
                      <button
                        onClick={() => handleDeleteArc(arc.id)}
                        className="p-1 rounded bg-white hover:bg-neutral-soft text-text-secondary hover:text-text-secondary transition-colors"
                        title="Delete Story Arc"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  <p className="text-xs text-text-secondary leading-relaxed">{arc.goal}</p>

                  <div className="space-y-1.5 pt-2 border-t border-neutral-border/60">
                    <div className="flex items-center justify-between text-[11px] font-mono text-text-secondary">
                      <span>SEASON {arc.season}</span>
                      <span>PROGRESS: {arc.episodes_completed} / {arc.episodes_planned} EPISODES</span>
                    </div>

                    <div className="w-full bg-neutral-soft h-2 rounded-full overflow-hidden border border-neutral-border">
                      <div
                        className="bg-accent-primary h-full rounded-full transition-all duration-500"
                        style={{ width: `${progressPct}%` }}
                      />
                    </div>
                  </div>
                </GlassCard>
              );
            })}
          </div>
        )}
      </div>

      {/* Chronological Timeline Feed */}
      <div className="space-y-4">
        <h3 className="text-base font-bold text-text-primary">HISTORICAL EVENTS TIMELINE ({timeline.length})</h3>

        {timeline.length === 0 ? (
          <GlassCard className="p-6 text-center text-text-secondary text-xs">
            No historical timeline events yet. Add milestone events or generate episodes to build world lore history!
          </GlassCard>
        ) : (
          <div className="relative pl-6 border-l-2 border-neutral-border space-y-6">
            {timeline.map((evt) => (
              <div key={evt.id} className="relative group">
                <div className="absolute -left-[31px] top-1.5 w-4 h-4 rounded-full bg-accent-primary border-4 border-[#0B0F19] group-hover:scale-125 transition-transform" />
                <GlassCard className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-accent-primary font-bold">{evt.timestamp}</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-[10px] font-mono bg-neutral-soft px-2 py-0.5 rounded text-text-secondary border border-neutral-border">
                        IMPORTANCE: {evt.importance_score}/10
                      </span>
                      <button
                        onClick={() => handleDeleteEvent(evt.id)}
                        className="p-1 rounded bg-white hover:bg-neutral-soft text-text-secondary hover:text-text-secondary transition-colors"
                        title="Delete Timeline Event"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                  <h4 className="text-base font-bold text-text-primary">{evt.title}</h4>
                  <p className="text-xs text-text-secondary leading-relaxed">{evt.description}</p>
                </GlassCard>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Add Event Modal */}
      {showEventModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <GlassCard className="max-w-lg w-full space-y-4">
            <div className="flex items-center justify-between border-b border-neutral-border pb-3">
              <h3 className="font-bold text-text-primary text-base flex items-center space-x-2">
                <Clock className="w-4 h-4 text-accent-primary" />
                <span>ADD MILESTONE TIMELINE EVENT</span>
              </h3>
              <button onClick={() => setShowEventModal(false)} className="text-text-secondary hover:text-text-primary">
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateEvent} className="space-y-4 text-xs font-mono">
              <div>
                <label className="block text-text-secondary mb-1">TIMESTAMPS IN UNIVERSE</label>
                <input
                  type="text"
                  value={evtTimestamp}
                  onChange={(e) => setEvtTimestamp(e.target.value)}
                  placeholder="e.g. Era 1, Year 2099"
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
                  required
                />
              </div>

              <div>
                <label className="block text-text-secondary mb-1">EVENT TITLE</label>
                <input
                  type="text"
                  value={evtTitle}
                  onChange={(e) => setEvtTitle(e.target.value)}
                  placeholder="e.g. The Great Network Breach"
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
                  required
                />
              </div>

              <div>
                <label className="block text-text-secondary mb-1">EVENT DESCRIPTION</label>
                <textarea
                  value={evtDesc}
                  onChange={(e) => setEvtDesc(e.target.value)}
                  placeholder="Describe what occurred during this milestone event..."
                  rows={3}
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
                  required
                />
              </div>

              <div>
                <label className="block text-text-secondary mb-1">IMPORTANCE SCORE (1 - 10): {evtImportance}</label>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={evtImportance}
                  onChange={(e) => setEvtImportance(Number(e.target.value))}
                  className="w-full cursor-pointer accent-cyan-400"
                />
              </div>

              <button
                type="submit"
                className="w-full py-3 rounded-xl font-bold bg-accent-primary text-text-primary shadow-sm flex items-center justify-center space-x-2"
              >
                <Sparkles className="w-4 h-4" />
                <span>SAVE TIMELINE EVENT</span>
              </button>
            </form>
          </GlassCard>
        </div>
      )}

      {/* Add Story Arc Modal */}
      {showArcModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <GlassCard className="max-w-lg w-full space-y-4">
            <div className="flex items-center justify-between border-b border-neutral-border pb-3">
              <h3 className="font-bold text-text-primary text-base flex items-center space-x-2">
                <Flame className="w-4 h-4 text-text-secondary" />
                <span>CREATE NEW STORY ARC</span>
              </h3>
              <button onClick={() => setShowArcModal(false)} className="text-text-secondary hover:text-text-primary">
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleCreateArc} className="space-y-4 text-xs font-mono">
              <div>
                <label className="block text-text-secondary mb-1">ARC TITLE</label>
                <input
                  type="text"
                  value={arcTitle}
                  onChange={(e) => setArcTitle(e.target.value)}
                  placeholder="e.g. Shadows of the Grid"
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-neutral-border"
                  required
                />
              </div>

              <div>
                <label className="block text-text-secondary mb-1">ARC NARRATIVE GOAL</label>
                <textarea
                  value={arcGoal}
                  onChange={(e) => setArcGoal(e.target.value)}
                  placeholder="What is the central conflict or quest for this story arc?"
                  rows={3}
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-neutral-border"
                  required
                />
              </div>

              <div>
                <label className="block text-text-secondary mb-1">PLANNED EPISODES: {arcEpisodesPlanned}</label>
                <input
                  type="range"
                  min={1}
                  max={12}
                  value={arcEpisodesPlanned}
                  onChange={(e) => setArcEpisodesPlanned(Number(e.target.value))}
                  className="w-full cursor-pointer accent-amber-400"
                />
              </div>

              <button
                type="submit"
                className="w-full py-3 rounded-xl font-bold bg-accent-primary text-text-primary shadow-lg flex items-center justify-center space-x-2"
              >
                <Sparkles className="w-4 h-4" />
                <span>CREATE STORY ARC</span>
              </button>
            </form>
          </GlassCard>
        </div>
      )}
    </div>
  );
};
