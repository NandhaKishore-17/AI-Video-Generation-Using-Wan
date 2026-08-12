import React, { useState, useEffect } from 'react';
import { GlassCard } from '../components/GlassCard';
import { api } from '../services/api';
import { Universe, KnowledgeDocument } from '../types';
import { Plus, Globe, Sparkles, BookOpen, Layers, Trash2, RefreshCw, Database } from 'lucide-react';

interface UniverseManagerProps {
  activeUniverseId?: string;
  onSelectUniverse?: (id: string) => void;
  onUniversesChanged?: () => void;
}

export const UniverseManager: React.FC<UniverseManagerProps> = ({ activeUniverseId, onSelectUniverse, onUniversesChanged }) => {
  const [universes, setUniverses] = useState<Universe[]>([]);
  const [title, setTitle] = useState('');
  const [genre, setGenre] = useState('');
  const [logline, setLogline] = useState('');
  const [rules, setRules] = useState('');
  const [useReference, setUseReference] = useState(false);
  const [referenceId, setReferenceId] = useState('');
  const [knowledgeDocs, setKnowledgeDocs] = useState<KnowledgeDocument[]>([]);
  const [loading, setLoading] = useState(false);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    loadUniverses();
    loadKnowledgeDocs();
  }, []);

  const loadKnowledgeDocs = async () => {
    const docs = await api.getKnowledgeList();
    const readyDocs = docs.filter(d => d.status === 'COMPLETED' || d.status === 'READY');
    setKnowledgeDocs(readyDocs);
  };

  const loadUniverses = async () => {
    const data = await api.getUniverses();
    setUniverses(data);
    if (onUniversesChanged) onUniversesChanged();
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title || !genre || !logline) return;
    if (useReference && !referenceId) {
      alert("Please select a reference document or turn off Reference Knowledge.");
      return;
    }
    setLoading(true);
    const created = await api.createUniverse({ 
      title, 
      genre, 
      logline, 
      world_rules: rules,
      use_reference_knowledge: useReference,
      reference_document_id: useReference ? referenceId : null
    });
    setTitle('');
    setGenre('');
    setLogline('');
    setRules('');
    setUseReference(false);
    setReferenceId('');
    setLoading(false);
    await loadUniverses();
    if (created && created.id && onSelectUniverse) {
      onSelectUniverse(created.id);
    }
  };

  const handleDeleteUniverse = async (id: string) => {
    if (window.confirm("Are you sure you want to delete this universe? All episodes and memory will be deleted.")) {
      await api.deleteUniverse(id);
      await loadUniverses();
    }
  };

  const handleResetAll = async () => {
    if (window.confirm("Are you sure you want to DELETE ALL UNIVERSES and start fresh?")) {
      setResetting(true);
      await api.resetAllUniverses();
      setUniverses([]);
      setResetting(false);
      if (onUniversesChanged) onUniversesChanged();
    }
  };

  const handleToggleAutoGenerate = async (id: string) => {
    await api.toggleUniverseAutoGenerate(id);
    await loadUniverses();
  };


  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Globe className="w-6 h-6 text-cyan-400" />
            <span>UNIVERSE MANAGER & BIBLE GENERATOR</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Create persistent story universes with auto-generated lore bibles, rules, factions, and initial story arcs.
          </p>
        </div>

        <button
          onClick={handleResetAll}
          disabled={resetting}
          className="px-4 py-2 rounded-xl text-xs font-mono font-bold bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800 flex items-center space-x-1.5 transition-all shadow-lg"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${resetting ? 'animate-spin' : ''}`} />
          <span>{resetting ? 'RESETTING...' : 'RESET & START FRESH'}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Create Universe Form */}
        <GlassCard className="lg:col-span-1 space-y-4">
          <h3 className="text-base font-bold text-white flex items-center space-x-2">
            <Plus className="w-4 h-4 text-cyan-400" />
            <span>CREATE NEW UNIVERSE</span>
          </h3>

          <form onSubmit={handleCreate} className="space-y-4 text-xs">
            <div>
              <label className="block text-slate-400 font-mono mb-1">UNIVERSE TITLE</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Cyberpunk Neo-Tokyo 2099"
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 font-mono mb-1">GENRE & STYLE</label>
              <input
                type="text"
                value={genre}
                onChange={(e) => setGenre(e.target.value)}
                placeholder="e.g. Sci-Fi Noir / Cosmic Sorcery"
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 font-mono mb-1">CORE LOGLINE</label>
              <textarea
                value={logline}
                onChange={(e) => setLogline(e.target.value)}
                placeholder="Describe the central conflict and world premise..."
                rows={3}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                required
              />
            </div>

            <div>
              <label className="block text-slate-400 font-mono mb-1">WORLD RULES & LAWS (OPTIONAL)</label>
              <textarea
                value={rules}
                onChange={(e) => setRules(e.target.value)}
                placeholder="1. Magic depletes at midnight. 2. Cybernetic chips can be hacked."
                rows={2}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
              />
            </div>

            <div className="pt-2 border-t border-slate-800/50">
              <label className="block text-slate-400 font-mono mb-2 flex items-center space-x-2">
                <Database className="w-3.5 h-3.5" />
                <span>REFERENCE KNOWLEDGE</span>
              </label>
              <div className="flex space-x-4 mb-3">
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    name="use_rag"
                    checked={!useReference}
                    onChange={() => setUseReference(false)}
                    className="text-cyan-500 focus:ring-cyan-500 focus:ring-offset-slate-900 bg-slate-900 border-slate-700"
                  />
                  <span className="text-slate-300">OFF</span>
                </label>
                <label className="flex items-center space-x-2 cursor-pointer">
                  <input
                    type="radio"
                    name="use_rag"
                    checked={useReference}
                    onChange={() => setUseReference(true)}
                    className="text-cyan-500 focus:ring-cyan-500 focus:ring-offset-slate-900 bg-slate-900 border-slate-700"
                  />
                  <span className="text-slate-300">ON</span>
                </label>
              </div>

              {useReference && (
                <div>
                  <select
                    value={referenceId}
                    onChange={(e) => setReferenceId(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="" disabled>Select Reference Document ▼</option>
                    {knowledgeDocs.map(doc => (
                      <option key={doc.id} value={doc.id}>{doc.name}</option>
                    ))}
                  </select>
                  {knowledgeDocs.length === 0 && (
                    <p className="text-[10px] text-amber-500 mt-1">
                      No ready documents found. Go to Reference Knowledge Library to upload and process documents first.
                    </p>
                  )}
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 rounded-xl font-bold bg-gradient-to-r from-cyan-500 via-purple-600 to-pink-500 text-white shadow-glow-cyan flex items-center justify-center space-x-2"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'GENERATING UNIVERSE BIBLE...' : 'GENERATE STORY UNIVERSE'}</span>
            </button>
          </form>
        </GlassCard>

        {/* Existing Universes Roster */}
        <div className="lg:col-span-2 space-y-4">
          <h3 className="text-base font-bold text-white flex items-center space-x-2">
            <Layers className="w-4 h-4 text-purple-400" />
            <span>ACTIVE STORY UNIVERSES ({universes.length})</span>
          </h3>

          <div className="space-y-4">
            {universes.length === 0 ? (
              <GlassCard className="p-8 text-center text-slate-400 space-y-2">
                <Globe className="w-10 h-10 mx-auto text-slate-600" />
                <p className="text-sm">No universes found. Create your first Story Universe to start fresh!</p>
              </GlassCard>
            ) : (
              universes.map((u) => (
                <GlassCard key={u.id} glow className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="text-lg font-bold text-white">{u.title}</h4>
                        <button
                          onClick={() => handleToggleAutoGenerate(u.id)}
                          className={`px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold border transition-colors ${
                            u.auto_generate_active
                              ? 'bg-emerald-950/80 text-emerald-400 border-emerald-700'
                              : 'bg-slate-900 text-slate-500 border-slate-800'
                          }`}
                          title="Click to toggle continuous auto-generation for this universe"
                        >
                          {u.auto_generate_active ? 'AUTO-GEN: ON' : 'AUTO-GEN: OFF'}
                        </button>
                      </div>
                      <span className="text-xs px-2.5 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-mono mt-1 inline-block">
                        {u.genre}
                      </span>
                    </div>

                    <div className="flex items-center space-x-2">
                      {activeUniverseId === u.id ? (
                        <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-700 shadow-glow-cyan">
                          ★ ACTIVE UNIVERSE
                        </span>
                      ) : (
                        <button
                          onClick={() => onSelectUniverse && onSelectUniverse(u.id)}
                          className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-slate-900 hover:bg-cyan-950 text-slate-300 hover:text-cyan-300 border border-slate-700 hover:border-cyan-700 transition-all"
                        >
                          SET AS ACTIVE
                        </button>
                      )}

                      <button
                        onClick={() => handleDeleteUniverse(u.id)}
                        className="p-2 rounded-xl bg-slate-900 hover:bg-rose-900/80 text-slate-400 hover:text-rose-300 border border-slate-800 transition-colors"
                        title="Delete Universe"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  <p className="text-xs text-slate-300 leading-relaxed">{u.logline}</p>

                  {u.world_rules && (
                    <div className="p-3 rounded-lg bg-slate-950 border border-slate-800/80 text-xs font-mono">
                      <span className="text-purple-400 font-bold block mb-1">WORLD RULES & LAWS:</span>
                      <p className="text-slate-400">{u.world_rules}</p>
                    </div>
                  )}
                </GlassCard>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
