import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { api } from '../services/api';
import { RenderTask } from '../types';
import { ListOrdered, CheckCircle2, Clock, Cpu, AlertTriangle } from 'lucide-react';

export const RenderQueue: React.FC = () => {
  const [tasks, setTasks] = useState<RenderTask[]>([]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    const episodes = await api.getEpisodes();
    const mockTasks: RenderTask[] = episodes.map((ep, idx) => ({
      id: `task-${ep.id}`,
      episode_id: ep.id,
      stage: ep.status === 'COMPLETED' ? 'DONE' : 'MEDIA_GEN',
      progress_percentage: ep.status === 'COMPLETED' ? 100 : 65,
      current_step_details: ep.status === 'COMPLETED' ? 'Episode rendered & memory stored.' : 'Generating CogVideoX motion clips...',
      started_at: ep.created_at,
      completed_at: ep.updated_at
    }));
    setTasks(mockTasks);
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <ListOrdered className="w-6 h-6 text-cyan-400" />
          <span>MULTI-STAGE RENDER QUEUE MONITOR</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Real-time tracking of generation stages: Qwen Screenplay → FLUX Keyframes → CogVideoX Motion → Piper TTS → FFmpeg Compositing.
        </p>
      </div>

      <div className="space-y-4">
        {tasks.map((task) => (
          <GlassCard key={task.id} glow={task.stage !== 'DONE'}>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-xl border ${
                    task.stage === 'DONE' ? 'bg-emerald-950 text-emerald-400 border-emerald-800' : 'bg-cyan-950 text-cyan-400 border-cyan-800 animate-pulse'
                  }`}>
                    {task.stage === 'DONE' ? <CheckCircle2 className="w-5 h-5" /> : <Cpu className="w-5 h-5" />}
                  </div>
                  <div>
                    <h4 className="font-bold text-white text-sm">RENDER TASK #{task.id}</h4>
                    <p className="text-xs text-slate-400 font-mono">{task.current_step_details}</p>
                  </div>
                </div>

                <div className="text-right font-mono">
                  <span className="text-sm font-bold text-cyan-400">{task.progress_percentage}%</span>
                  <span className="text-[10px] text-slate-500 block">STAGE: {task.stage}</span>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-2 rounded-full bg-slate-950 overflow-hidden border border-slate-800">
                <div
                  className={`h-full transition-all duration-500 ${
                    task.stage === 'DONE'
                      ? 'bg-gradient-to-r from-emerald-500 to-cyan-500'
                      : 'bg-gradient-to-r from-cyan-500 via-purple-500 to-pink-500 animate-pulse'
                  }`}
                  style={{ width: `${task.progress_percentage}%` }}
                />
              </div>
            </div>
          </GlassCard>
        ))}
      </div>
    </div>
  );
};
