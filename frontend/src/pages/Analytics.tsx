import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { api } from '../services/api';
import { AnalyticsData } from '../types';
import { BarChart3, Cpu, Zap, Activity, Clock } from 'lucide-react';

export const Analytics: React.FC = () => {
  const [data, setData] = useState<AnalyticsData | null>(null);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    const res = await api.getAnalytics();
    setData(res);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-cyan-400" />
          <span>ENGINE BENCHMARKS & PLATFORM ANALYTICS</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Monitor open-source model configuration and system setup.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <GlassCard>
          <div className="space-y-2">
            <span className="text-xs font-mono text-cyan-400">LLM PROVIDER</span>
            <h3 className="text-2xl font-extrabold text-white uppercase">{data?.engine_benchmarks?.llm_provider || 'N/A'}</h3>
            <p className="text-[11px] text-slate-400 font-mono">Model: {data?.engine_benchmarks?.llm_model || 'N/A'}</p>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="space-y-2">
            <span className="text-xs font-mono text-purple-400">IMAGE PROVIDER</span>
            <h3 className="text-2xl font-extrabold text-white uppercase">{data?.engine_benchmarks?.image_provider || 'N/A'}</h3>
            <p className="text-[11px] text-slate-400 font-mono">Generates Keyframes</p>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="space-y-2">
            <span className="text-xs font-mono text-pink-400">VIDEO PROVIDER</span>
            <h3 className="text-2xl font-extrabold text-white uppercase">{data?.engine_benchmarks?.video_provider || 'N/A'}</h3>
            <p className="text-[11px] text-slate-400 font-mono">Generates Motion Video</p>
          </div>
        </GlassCard>
      </div>

      <GlassCard glow>
        <h3 className="font-bold text-white text-base mb-4">SYSTEM CONFIGURATION</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-center font-mono text-xs">
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block">LLM API BASE URL</span>
            <span className="text-cyan-400 font-bold">{data?.system_status?.llm_api_base || 'N/A'}</span>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-slate-500 text-[10px] block">OLLAMA MODEL</span>
            <span className="text-purple-400 font-bold">{data?.system_status?.ollama_model || 'N/A'}</span>
          </div>
        </div>
      </GlassCard>
    </div>
  );
};
