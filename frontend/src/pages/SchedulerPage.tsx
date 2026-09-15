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
  const [mode, setMode] = useState('interval');
  const [loading, setLoading] = useState(false);
  const [triggering, setTriggering] = useState(false);
  const [triggerMessage, setTriggerMessage] = useState<string | null>(null);
  const [timeLeft, setTimeLeft] = useState<{ hours: string; minutes: string; seconds: string; totalSec: number }>({
    hours: '00',
    minutes: '00',
    seconds: '00',
    totalSec: 0
  });

  // Load status on mount / when refreshTrigger changes
  useEffect(() => {
    loadStatus();
  }, [refreshTrigger]);

  // Background poll every 30 seconds — keeps status fresh without hammering the API
  useEffect(() => {
    const pollInterval = setInterval(() => {
      loadStatus();
    }, 30000);
    return () => clearInterval(pollInterval);
  }, []);

  // Countdown timer — runs every second but only calls loadStatus() ONCE when it reaches 0
  useEffect(() => {
    if (!status?.next_run || !status?.is_running || status?.mode === 'continuous') {
      setTimeLeft({ hours: '00', minutes: '00', seconds: '00', totalSec: 0 });
      return;
    }

    const targetIso = status.next_run.endsWith('Z') || status.next_run.includes('+')
      ? status.next_run
      : `${status.next_run}Z`;

    let hasFiredReload = false;

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
      // Only trigger one reload when the countdown first reaches 0
      if (updated.totalSec === 0 && !hasFiredReload) {
        hasFiredReload = true;
        loadStatus();
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [status?.next_run, status?.is_running, status?.mode]);

  const loadStatus = async () => {
    const data = await api.getSchedulerStatus();
    setStatus(data);
    if (data.interval_minutes) setIntervalMins(data.interval_minutes);
    if (data.mode) setMode(data.mode);
  };

  const handleStart = async () => {
    setLoading(true);
    const updated = await api.startScheduler(intervalMins, true, mode);
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
        await onTriggerGenerate(30, "Autonomous Scheduler Trigger Release");
      } else {
        await api.triggerSchedulerNow();
      }
      setTriggerMessage(`Successfully triggered story script auto-generation!`);
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
        <h1 className="text-2xl font-extrabold text-text-primary tracking-tight flex items-center gap-2">
          <Calendar className="w-6 h-6 text-accent-primary" />
          <span>AUTONOMOUS SCHEDULER</span>
        </h1>
        <p className="text-xs text-text-secondary mt-1">
          Configure continuous background generation loops to automatically write screenplays and produce new episodes.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Controls */}
        <GlassCard className="lg:col-span-1 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-text-primary text-base">LOOP CONTROLLER</h3>
            <span className={`px-3 py-1 rounded-full text-xs font-mono font-bold flex items-center gap-1.5 border ${
              status?.is_running
                ? 'bg-accent-light text-accent-primary border-emerald-200'
                : 'bg-neutral-soft text-text-secondary border-neutral-border'
            }`}>
              <span className={`w-2 h-2 rounded-full ${status?.is_running ? 'bg-accent-light0 animate-ping' : 'bg-stone-400'}`} />
              {status?.is_running ? 'RUNNING' : 'STOPPED'}
            </span>
          </div>

          <div className="space-y-4 text-xs font-mono">
            <div>
              <label className="block text-text-secondary font-bold mb-1">MODE</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                disabled={status?.is_running}
                className="w-full bg-white border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary disabled:opacity-50 shadow-sm"
              >
                <option value="interval">Interval</option>
                <option value="continuous">Continuous</option>
              </select>
            </div>

            {mode === 'interval' && (
              <div>
                <label className="block text-text-secondary font-bold mb-1">GENERATION INTERVAL (MINUTES)</label>
                <select
                  value={intervalMins}
                  onChange={(e) => setIntervalMins(Number(e.target.value))}
                  disabled={status?.is_running}
                  className="w-full bg-white border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary disabled:opacity-50 shadow-sm"
                >
                  <option value={15}>Every 15 Minutes (Turbo Fast)</option>
                  <option value={60}>Every 1 Hour (Recommended)</option>
                  <option value={360}>Every 6 Hours</option>
                  <option value={1440}>Every 24 Hours (Daily Show)</option>
                </select>
              </div>
            )}

            <div className="pt-2 space-y-2">
              {status?.is_running ? (
                <button
                  onClick={handleStop}
                  disabled={loading}
                  className="w-full py-3 rounded-xl font-bold bg-accent-light hover:bg-rose-200 border border-neutral-border text-text-secondary transition-all flex items-center justify-center space-x-2"
                >
                  <Square className="w-4 h-4 fill-rose-600" />
                  <span>PAUSE SCHEDULER LOOP</span>
                </button>
              ) : (
                <button
                  onClick={handleStart}
                  disabled={loading}
                  className="w-full py-3 rounded-xl font-bold bg-neutral-soft hover:bg-accent-primary text-text-primary shadow-sm transition-all flex items-center justify-center space-x-2"
                >
                  <Play className="w-4 h-4 fill-white" />
                  <span>START AUTONOMOUS LOOP</span>
                </button>
              )}

              <button
                onClick={handleTriggerNow}
                disabled={triggering}
                className="w-full py-3 rounded-xl font-bold bg-neutral-soft hover:bg-neutral-soft border border-neutral-border text-text-primary transition-all flex items-center justify-center space-x-2"
              >
                <Zap className={`w-4 h-4 text-accent-primary ${triggering ? 'animate-bounce' : ''}`} />
                <span>{triggering ? 'GENERATING STORY SCRIPT...' : 'TRIGGER INSTANT SCRIPT NOW'}</span>
              </button>
            </div>

            {triggerMessage && (
              <div className="p-3 rounded-xl bg-neutral-soft border border-neutral-border text-[11px] text-text-primary">
                {triggerMessage}
              </div>
            )}
          </div>
        </GlassCard>

        {/* Schedule Timeline Status & Live Countdown */}
        <GlassCard className="lg:col-span-2 space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="font-bold text-text-primary text-base">SCHEDULED EXECUTIONS</h3>
            {status?.is_running && (
              <span className="text-[11px] font-mono text-accent-primary flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-accent-light0 animate-ping" />
                {status.mode === 'continuous' ? 'CONTINUOUS MODE ACTIVE' : 'LIVE COUNTDOWN ACTIVE'}
              </span>
            )}
          </div>

          {/* Large Status Display / Digital Clock Countdown Banner */}
          <div className="p-6 rounded-2xl bg-white border border-neutral-border shadow-sm text-center space-y-4">
            
            {status?.mode === 'continuous' && status?.is_running ? (
              <div className="py-8 space-y-4">
                <div className="flex justify-center">
                  <div className="p-4 bg-accent-light text-accent-primary rounded-full animate-pulse">
                    <Zap className="w-10 h-10" />
                  </div>
                </div>
                <h2 className="text-xl font-bold text-text-primary">CONTINUOUS GENERATION ACTIVE</h2>
                <p className="text-sm font-mono text-text-secondary max-w-md mx-auto">
                  NEXT EPISODE WILL START AUTOMATICALLY AFTER CURRENT EPISODE COMPLETES
                </p>
              </div>
            ) : (
              <>
                <span className="text-xs font-mono text-text-secondary tracking-wider block">
                  NEXT AUTOMATED STORY SCRIPT GENERATION IN
                </span>

                <div className="flex items-center justify-center space-x-3 md:space-x-4 font-mono">
                  <div className="bg-neutral-soft border border-neutral-border rounded-xl px-4 py-3 min-w-[70px] md:min-w-[85px]">
                    <span className="text-3xl md:text-4xl font-extrabold text-accent-primary block">{timeLeft.hours}</span>
                    <span className="text-[10px] text-text-secondary tracking-widest uppercase block mt-1">HOURS</span>
                  </div>
                  <span className="text-2xl font-bold text-text-secondary">:</span>
                  <div className="bg-neutral-soft border border-neutral-border rounded-xl px-4 py-3 min-w-[70px] md:min-w-[85px]">
                    <span className="text-3xl md:text-4xl font-extrabold text-accent-primary block">{timeLeft.minutes}</span>
                    <span className="text-[10px] text-text-secondary tracking-widest uppercase block mt-1">MINS</span>
                  </div>
                  <span className="text-2xl font-bold text-text-secondary">:</span>
                  <div className="bg-neutral-soft border border-neutral-border rounded-xl px-4 py-3 min-w-[70px] md:min-w-[85px]">
                    <span className="text-3xl md:text-4xl font-extrabold text-accent-primary block">{timeLeft.seconds}</span>
                    <span className="text-[10px] text-text-secondary tracking-widest uppercase block mt-1">SECS</span>
                  </div>
                </div>

                {/* Visual Progress Bar for Interval Mode */}
                <div className="space-y-1.5 pt-2 max-w-md mx-auto">
                  <div className="w-full bg-neutral-soft h-2 rounded-full overflow-hidden border border-neutral-border">
                    <div
                      className="bg-accent-light0 h-full transition-all duration-1000 ease-linear rounded-full"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                  <div className="flex items-center justify-between text-[10px] font-mono text-text-secondary px-1">
                    <span>PROGRESS</span>
                    <span>{progressPercent.toFixed(1)}%</span>
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="space-y-3 font-mono text-xs">
            <div className="p-4 rounded-xl bg-white border border-neutral-border shadow-sm flex items-center justify-between">
              <div className="space-y-1">
                <span className="text-text-secondary block text-[10px]">LAST GENERATED EPISODE SCRIPT</span>
                <span className="text-accent-primary font-bold">{status?.last_run ? formatLocalDate(status.last_run) : 'Just now'}</span>
              </div>
              <CheckCircle2 className="w-5 h-5 text-accent-primary" />
            </div>

            <div className="p-4 rounded-xl bg-white border border-neutral-border shadow-sm flex items-center justify-between">
              <div className="space-y-1">
                <span className="text-text-secondary block text-[10px]">NEXT AUTOMATED TRIGGER TIME</span>
                <span className="text-text-primary font-bold">
                  {status?.mode === 'continuous' && status.is_running 
                    ? 'Continuous Mode' 
                    : (status?.next_run ? formatLocalDate(status.next_run) : 'In 60 mins')}
                </span>
              </div>
              <Clock className={`w-5 h-5 text-text-secondary ${status?.is_running && status?.mode !== 'continuous' ? 'animate-spin' : ''}`} />
            </div>
          </div>
        </GlassCard>
      </div>
    </div>
  );
};
