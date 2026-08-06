import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { api } from '../services/api';
import { SchedulerStatus } from '../types';
import { Calendar, Play, Square, Clock, Zap, CheckCircle2 } from 'lucide-react';

interface SchedulerPageProps {
  onTriggerGenerate?: (sec?: number, prompt?: string) => void;
  refreshTrigger?: number;
}

export const SchedulerPage: React.FC<SchedulerPageProps> = ({ onTriggerGenerate, refreshTrigger }) => {
  const [status, setStatus] = useState<SchedulerStatus | null>(null);
  const [intervalMins, setIntervalMins] = useState(60);
  const [loading, setLoading] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [triggerMessage, setTriggerMessage] = useState<string | null>(null);
  const [timeLeft, setTimeLeft] = useState<{ hours: string; minutes: string; seconds: string; totalSec: number }>({
    hours: '00',
    minutes: '00',
    seconds: '00',
    totalSec: 0
  });

  useEffect(() => {
    loadStatus();
  }, [refreshTrigger]);


  useEffect(() => {
    if (!status?.next_run || !status?.is_running) {
      setTimeLeft({ hours: '00', minutes: '00', seconds: '00', totalSec: 0 });
      return;
    }

    const targetIso = status.next_run.endsWith('Z') || status.next_run.includes('+')
      ? status.next_run
      : `${status.next_run}Z`;

    const calcTime = () => {
      const targetTime = new Date(targetIso).getTime();
      const diffSec = Math.max(0, Math.floor((targetTime - Date.now()) / 1000));
      const h = Math.floor(diffSec / 3600);
      const m = Math.floor((diffSec % 3600) / 60);
      const s = diffSec % 60;
      const pad = (n: number) => n.toString().padStart(2, '0');
      return { hours: pad(h), minutes: pad(m), seconds: pad(s), totalSec: diffSec };
    };

    setTimeLeft(calcTime());
    const interval = setInterval(() => {
      const updated = calcTime();
      setTimeLeft(updated);
      if (updated.totalSec === 0) {
        loadStatus();
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [status?.next_run, status?.is_running]);

  const loadStatus = async () => {
    const data = await api.getSchedulerStatus();
    setStatus(data);
    if (data.interval_minutes) setIntervalMins(data.interval_minutes);
  };

  const handleStart = async () => {
    setLoading(true);
    const updated = await api.startScheduler(intervalMins);
    setStatus(updated);
    setLoading(false);
  };

  const handleStop = async () => {
    setLoading(true);
    const updated = await api.stopScheduler();
    setStatus(updated);
    setLoading(false);
  };

  const handleTriggerNow = async () => {
    setTriggering(true);
    setTriggerMessage(null);
    try {
      if (onTriggerGenerate) {
        await onTriggerGenerate(8, "Autonomous Scheduler Trigger Release");
      } else {
        await api.triggerSchedulerNow();
      }
      setTriggerMessage(`Successfully triggered story script auto-generation for active universes!`);
      await loadStatus();
    } catch (e: any) {
      setTriggerMessage(`Trigger failed: ${e?.message || 'Error occurred'}`);
    } finally {
      setTriggering(false);
    }
  };


  const formatLocalDate = (isoStr?: string) => {
    if (!isoStr) return null;
    const str = isoStr.endsWith('Z') || isoStr.includes('+') ? isoStr : `${isoStr}Z`;
    return new Date(str).toLocaleString(undefined, {
      dateStyle: 'short',
      timeStyle: 'medium'
    });
  };

  const totalIntervalSec = (status?.interval_minutes || 60) * 60;
  const elapsedSec = Math.max(0, totalIntervalSec - timeLeft.totalSec);
  const progressPercent = status?.is_running
    ? Math.min(100, Math.max(0, (elapsedSec / totalIntervalSec) * 100))
    : 0;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
          <Calendar className="w-6 h-6 text-emerald-400" />
          <span>AUTONOMOUS CONTINUOUS SCHEDULER</span>
        </h1>
        <p className="text-xs text-slate-400 mt-1">
          Configure continuous background generation loops to automatically write screenplays and produce new episodes.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls */}
        <GlassCard glow className="lg:col-span-1 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-white text-base">LOOP CONTROLLER</h3>
            <span className={`px-3 py-1 rounded-full text-xs font-mono font-bold flex items-center gap-1.5 border ${
              status?.is_running
                ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                : 'bg-slate-900 text-slate-400 border-slate-800'
            }`}>
              <span className={`w-2 h-2 rounded-full ${status?.is_running ? 'bg-emerald-400 animate-ping' : 'bg-slate-500'}`} />
              {status?.is_running ? 'RUNNING' : 'STOPPED'}
            </span>
          </div>

          <div className="space-y-4 text-xs font-mono">
            <div>
              <label className="block text-slate-400 mb-1">GENERATION INTERVAL (MINUTES)</label>
              <select
                value={intervalMins}
                onChange={(e) => setIntervalMins(Number(e.target.value))}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
              >
                <option value={15}>Every 15 Minutes (Turbo Fast)</option>
                <option value={60}>Every 1 Hour (Recommended)</option>
                <option value={360}>Every 6 Hours</option>
                <option value={1440}>Every 24 Hours (Daily Show)</option>
              </select>
            </div>

            <div className="pt-2 space-y-2">
              {status?.is_running ? (
                <button
                  onClick={handleStop}
                  disabled={loading}
                  className="w-full py-3 rounded-xl font-bold bg-red-950 hover:bg-red-900 border border-red-800 text-red-300 transition-all flex items-center justify-center space-x-2"
                >
                  <Square className="w-4 h-4 fill-red-400" />
                  <span>PAUSE SCHEDULER LOOP</span>
                </button>
              ) : (
                <button
                  onClick={handleStart}
                  disabled={loading}
                  className="w-full py-3 rounded-xl font-bold bg-gradient-to-r from-emerald-500 to-cyan-600 hover:brightness-110 text-white shadow-glow-cyan transition-all flex items-center justify-center space-x-2"
                >
                  <Play className="w-4 h-4 fill-white" />
                  <span>START AUTONOMOUS LOOP</span>
                </button>
              )}

              <button
                onClick={handleTriggerNow}
                disabled={triggering}
                className="w-full py-3 rounded-xl font-bold bg-purple-950/80 hover:bg-purple-900 border border-purple-800 text-purple-200 transition-all flex items-center justify-center space-x-2"
              >
                <Zap className={`w-4 h-4 text-purple-400 ${triggering ? 'animate-bounce' : ''}`} />
                <span>{triggering ? 'GENERATING STORY SCRIPT...' : 'TRIGGER INSTANT SCRIPT NOW'}</span>
              </button>
            </div>

            {triggerMessage && (
              <div className="p-3 rounded-xl bg-slate-950 border border-purple-800 text-[11px] text-purple-300">
                {triggerMessage}
              </div>
            )}
          </div>
        </GlassCard>

        {/* Schedule Timeline Status & Live Countdown */}
        <GlassCard className="lg:col-span-2 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-white text-base">SCHEDULED EXECUTIONS</h3>
            {status?.is_running && (
              <span className="text-[11px] font-mono text-purple-400 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-purple-400 animate-ping" />
                LIVE COUNTDOWN ACTIVE
              </span>
            )}
          </div>

          {/* Large Digital Clock Countdown Banner */}
          <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-950 via-purple-950/40 to-slate-950 border border-purple-800/60 shadow-lg text-center space-y-4">
            <span className="text-xs font-mono text-slate-400 tracking-wider block">
              NEXT AUTOMATED STORY SCRIPT GENERATION IN
            </span>

            <div className="flex items-center justify-center space-x-3 md:space-x-4 font-mono">
              <div className="bg-slate-900/90 border border-purple-700/60 rounded-xl px-4 py-3 min-w-[70px] md:min-w-[85px]">
                <span className="text-3xl md:text-4xl font-extrabold text-purple-300 block">{timeLeft.hours}</span>
                <span className="text-[10px] text-slate-400 tracking-widest uppercase block mt-1">HOURS</span>
              </div>
              <span className="text-2xl font-bold text-purple-500">:</span>
              <div className="bg-slate-900/90 border border-purple-700/60 rounded-xl px-4 py-3 min-w-[70px] md:min-w-[85px]">
                <span className="text-3xl md:text-4xl font-extrabold text-cyan-300 block">{timeLeft.minutes}</span>
                <span className="text-[10px] text-slate-400 tracking-widest uppercase block mt-1">MINS</span>
              </div>
              <span className="text-2xl font-bold text-purple-500">:</span>
              <div className="bg-slate-900/90 border border-purple-700/60 rounded-xl px-4 py-3 min-w-[70px] md:min-w-[85px]">
                <span className="text-3xl md:text-4xl font-extrabold text-emerald-400 block">{timeLeft.seconds}</span>
                <span className="text-[10px] text-slate-400 tracking-widest uppercase block mt-1">SECS</span>
              </div>
            </div>

            {/* Visual Progress Bar */}
            <div className="space-y-1.5 pt-2 max-w-md mx-auto">
              <div className="w-full bg-slate-900 h-2 rounded-full overflow-hidden border border-slate-800">
                <div
                  className="bg-gradient-to-r from-cyan-500 via-purple-500 to-emerald-400 h-full transition-all duration-1000 ease-linear rounded-full"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-500 px-1">
                <span>PROGRESS</span>
                <span>{progressPercent.toFixed(1)}%</span>
              </div>
            </div>
          </div>

          <div className="space-y-3 font-mono text-xs">
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
              <div className="space-y-1">
                <span className="text-slate-500 block text-[10px]">LAST GENERATED EPISODE SCRIPT</span>
                <span className="text-cyan-400 font-bold">{status?.last_run ? formatLocalDate(status.last_run) : 'Just now'}</span>
              </div>
              <CheckCircle2 className="w-5 h-5 text-emerald-400" />
            </div>

            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between">
              <div className="space-y-1">
                <span className="text-slate-500 block text-[10px]">NEXT AUTOMATED TRIGGER TIME</span>
                <span className="text-purple-400 font-bold">{status?.next_run ? formatLocalDate(status.next_run) : 'In 60 mins'}</span>
              </div>
              <Clock className="w-5 h-5 text-purple-400 animate-spin" />
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  );
};
