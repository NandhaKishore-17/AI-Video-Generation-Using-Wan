import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { VideoPlayer } from '../components/VideoPlayer';
import { api } from '../services/api';
import { Episode } from '../types';
import { Video, Download, Play, Monitor, Smartphone, CheckCircle2 } from 'lucide-react';

interface VideoLibraryProps {
  activeUniverseId?: string;
  refreshTrigger?: number;
}

export const VideoLibrary: React.FC<VideoLibraryProps> = ({ activeUniverseId, refreshTrigger }) => {
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<Episode | null>(null);
  const [aspectRatio, setAspectRatio] = useState<'16:9' | '9:16'>('16:9');

  useEffect(() => {
    loadData();
  }, [activeUniverseId, refreshTrigger]);


  const loadData = async () => {
    const data = await api.getEpisodes(activeUniverseId);
    setEpisodes(data);
    if (data.length > 0) {
      handleSelectVideo(data[0]);
    } else {
      setSelectedVideo(null);
    }
  };

  const handleSelectVideo = async (ep: Episode) => {
    try {
      const detail = await api.getEpisodeDetail(ep.id);
      setSelectedVideo(detail);
    } catch {
      setSelectedVideo(ep);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header & Aspect Ratio Switcher */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-extrabold text-text-primary tracking-tight flex items-center gap-2">
            <Video className="w-6 h-6 text-text-secondary" />
            <span>FINAL RENDERED MP4 VIDEO LIBRARY</span>
          </h1>
          <p className="text-xs text-text-secondary mt-1">
            Export broadcast-ready MP4 video files formatted for 16:9 cinematic widescreen or 9:16 vertical social shorts.
          </p>
        </div>

        {/* Aspect Ratio Toggles */}
        <div className="flex items-center space-x-2 bg-neutral-soft p-1.5 rounded-xl border border-neutral-border text-xs font-mono">
          <button
            onClick={() => setAspectRatio('16:9')}
            className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 font-bold transition-all ${
              aspectRatio === '16:9'
                ? 'bg-neutral-soft text-text-secondary border border-neutral-border shadow-sm'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Monitor className="w-3.5 h-3.5" />
            <span>16:9 CINEMATIC</span>
          </button>

          <button
            onClick={() => setAspectRatio('9:16')}
            className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 font-bold transition-all ${
              aspectRatio === '9:16'
                ? 'bg-neutral-soft text-text-secondary border border-neutral-border shadow-sm'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            <Smartphone className="w-3.5 h-3.5" />
            <span>9:16 SHORTS</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Video Viewport */}
        <div className="lg:col-span-2 space-y-4">
          {selectedVideo ? (
            <>
              <VideoPlayer
                videoUrl={selectedVideo.final_video_url}
                posterUrl={selectedVideo.thumbnail_url}
                title={selectedVideo.title}
                subtitles={selectedVideo.logline}
                scenes={selectedVideo.scenes}
                aspectRatio={aspectRatio}
              />

              <GlassCard className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h3 className="font-bold text-text-primary text-base">{selectedVideo.title}</h3>
                  <p className="text-xs text-text-secondary font-mono mt-0.5">
                    Duration: {selectedVideo.duration_seconds || 19.5}s • Format: MP4 (H.264 / AAC) • Season {selectedVideo.season}, Ep {selectedVideo.episode_number}
                  </p>
                </div>

                <div className="flex items-center space-x-3 text-xs">
                  <a
                    href={selectedVideo.final_video_url || '#'}
                    download
                    className="px-4 py-2 rounded-xl bg-accent-primary hover:brightness-110 text-text-primary font-bold flex items-center space-x-2 shadow-sm transition-all"
                  >
                    <Download className="w-4 h-4" />
                    <span>DOWNLOAD MP4</span>
                  </a>
                </div>
              </GlassCard>
            </>
          ) : (
            <GlassCard className="p-12 text-center text-text-secondary text-sm space-y-2">
              <Video className="w-10 h-10 mx-auto text-slate-600" />
              <p>No rendered videos found for this universe. Generate your first episode to watch rendered videos!</p>
            </GlassCard>
          )}
        </div>

        {/* Video Select Roster */}
        <div className="space-y-4">
          <h3 className="text-base font-bold text-text-primary flex items-center space-x-2">
            <Play className="w-4 h-4 text-accent-primary" />
            <span>RENDERED EPISODES ({episodes.length})</span>
          </h3>

          <div className="space-y-3">
            {episodes.length === 0 ? (
              <GlassCard className="p-6 text-center text-text-secondary text-xs">
                No episodes rendered yet.
              </GlassCard>
            ) : (
              episodes.map((ep) => (
                <GlassCard
                  key={ep.id}
                  glow={ep.id === selectedVideo?.id}
                  className="cursor-pointer hover:border-neutral-border transition-all space-y-2"
                  onClick={() => handleSelectVideo(ep)}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono text-text-secondary font-bold bg-neutral-soft px-2 py-0.5 rounded border border-neutral-border">
                      S{ep.season} E{ep.episode_number}
                    </span>
                    {ep.id === selectedVideo?.id && (
                      <span className="text-[10px] font-mono text-accent-primary flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3 text-accent-primary" /> PLAYING
                      </span>
                    )}
                  </div>
                  <h4 className="text-sm font-bold text-text-primary hover:text-accent-primary transition-colors">{ep.title}</h4>
                  <p className="text-xs text-text-secondary line-clamp-2">{ep.logline}</p>
                </GlassCard>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
