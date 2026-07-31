import React, { useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { api } from '../services/api';
import { MemoryQueryResult } from '../types';
import { Brain, Search, Sparkles, Database, Network } from 'lucide-react';

interface MemoryExplorerProps {
  activeUniverseId?: string;
}

export const MemoryExplorer: React.FC<MemoryExplorerProps> = ({ activeUniverseId }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<MemoryQueryResult[]>([]);
  const [searching, setSearching] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query) return;
    setSearching(true);
    const universes = await api.getUniverses();
    const targetUId = activeUniverseId || (universes.length > 0 ? universes[0].id : 'u-cyber-99');
    const res = await api.queryMemory(targetUId, query);
    setResults(res);
    setSearching(false);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <Brain className="w-6 h-6 text-cyan-400" />
          <span>QDRANT LONG-TERM VECTOR STORY MEMORY EXPLORER</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Perform semantic search over vector memory index to verify continuity, past reveals, and character relationship history.
        </p>
      </div>

      {/* Vector Search Input */}
      <GlassCard glow>
        <form onSubmit={handleSearch} className="flex gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 absolute left-3.5 top-3.5 text-slate-500" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask memory: e.g. What happened to Kaelen in Episode 1?"
              className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-3 text-sm text-white focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>
          <button
            type="submit"
            disabled={searching}
            className="px-6 py-3 rounded-xl font-bold bg-gradient-to-r from-cyan-500 to-purple-600 text-white shadow-glow-cyan flex items-center space-x-2 text-xs"
          >
            <Sparkles className="w-4 h-4" />
            <span>{searching ? 'QUERYING VECTORS...' : 'QUERY MEMORY'}</span>
          </button>
        </form>
      </GlassCard>

      {/* Memory Graph & Results */}
      <div className="space-y-4">
        <h3 className="text-base font-bold text-white flex items-center space-x-2">
          <Database className="w-4 h-4 text-purple-400" />
          <span>RETRIEVED STORY MEMORY VECTORS</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {results.length > 0 ? (
            results.map((res) => (
              <GlassCard key={res.id}>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-cyan-400 font-bold">
                      EPISODE {res.episode_number} • {res.memory_type}
                    </span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">
                      SIMILARITY: {(res.relevance_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="text-xs text-slate-200">{res.content}</p>
                  <div className="flex flex-wrap gap-1.5 pt-2">
                    {res.entities_involved.map((entity, idx) => (
                      <span key={idx} className="text-[10px] font-mono bg-slate-950 px-2 py-0.5 rounded text-slate-400 border border-slate-800">
                        {entity}
                      </span>
                    ))}
                  </div>
                </div>
              </GlassCard>
            ))
          ) : (
            <div className="col-span-2 text-center p-8 rounded-2xl bg-slate-900/40 border border-slate-800 text-slate-500 text-xs font-mono">
              Enter a prompt above to search long-term memory embeddings.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
