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
        <h1 className="text-2xl font-extrabold text-text-primary tracking-tight flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-accent-primary" />
          <span>ENGINE BENCHMARKS & PLATFORM ANALYTICS</span>
        </h1>
        <p className="text-xs text-text-secondary mt-1">
          Monitor open-source model inference speeds, rendering pipeline latency, and VRAM utilization metrics.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <GlassCard>
          <div className="space-y-2">
            <span className="text-xs font-mono text-accent-primary">GEMMA 3:4B LLM INFERENCE</span>
            <span className="text-xl font-extrabold text-text-primary">{data?.engine_benchmarks?.gemma_llm_avg_sec || '1.2'}s</span>
            <p className="text-[11px] text-text-secondary font-mono">Average time for Screenplay & Scene Breakdown</p>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="space-y-2">
            <span className="text-xs font-mono text-text-secondary">FLUX.1 KEYFRAME GENERATION</span>
            <h3 className="text-2xl font-extrabold text-text-primary">{data?.engine_benchmarks.flux_image_avg_sec}s</h3>
            <p className="text-[11px] text-text-secondary font-mono">Average time per 8k scene keyframe image</p>
          </div>
        </GlassCard>

        <GlassCard>
          <div className="space-y-2">
            <span className="text-xs font-mono text-text-secondary">WAN 2.2 5B MOTION SYNTHESIS</span>
            <span className="text-xl font-extrabold text-text-primary">{data?.engine_benchmarks?.wan_video_avg_sec || '5.1'}s</span>
            <p className="text-[11px] text-text-secondary font-mono">Average time per 24fps motion clip</p>
          </div>
        </GlassCard>
      </div>

      <GlassCard glow>
        <h3 className="font-bold text-text-primary text-base mb-4">SYSTEM HARDWARE ALLOCATION</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-center font-mono text-xs">
          <div className="p-3 bg-neutral-soft rounded-xl border border-neutral-border">
            <span className="text-text-secondary text-[10px] block">GPU VRAM USED</span>
            <span className="text-accent-primary font-bold">{data?.system_status.gpu_vram_allocated_gb}</span>
          </div>
          <div className="p-3 bg-neutral-soft rounded-xl border border-neutral-border">
            <span className="text-text-secondary text-[10px] block">CELERY WORKERS</span>
            <span className="text-text-secondary font-bold">{data?.system_status.celery_workers_active}</span>
          </div>
          <div className="p-3 bg-neutral-soft rounded-xl border border-neutral-border">
            <span className="text-text-secondary text-[10px] block">REDIS STATUS</span>
            <span className="text-accent-primary font-bold">{data?.system_status.redis_connected ? 'CONNECTED' : 'STANDBY'}</span>
          </div>
          <div className="p-3 bg-neutral-soft rounded-xl border border-neutral-border">
            <span className="text-text-secondary text-[10px] block">QDRANT VECTORS</span>
            <span className="text-text-secondary font-bold">{data?.system_status.qdrant_indexed_vectors}</span>
          </div>
        </div>
      </GlassCard>
    </div>
  );
};
