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
          <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Video className="w-6 h-6 text-purple-400" />
            <span>FINAL RENDERED MP4 VIDEO LIBRARY</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Export broadcast-ready MP4 video files formatted for 16:9 cinematic widescreen or 9:16 vertical social shorts.
          </p>
        </div>

        {/* Aspect Ratio Toggles */}
        <div className="flex items-center space-x-2 bg-slate-950 p-1.5 rounded-xl border border-slate-800 text-xs font-mono">
          <button
            onClick={() => setAspectRatio('16:9')}
            className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 font-bold transition-all ${
              aspectRatio === '16:9'
                ? 'bg-purple-950 text-purple-300 border border-purple-700 shadow-glow-cyan'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <Monitor className="w-3.5 h-3.5" />
            <span>16:9 CINEMATIC</span>
          </button>

          <button
            onClick={() => setAspectRatio('9:16')}
            className={`px-3 py-1.5 rounded-lg flex items-center space-x-1.5 font-bold transition-all ${
              aspectRatio === '9:16'
                ? 'bg-purple-950 text-purple-300 border border-purple-700 shadow-glow-cyan'
                : 'text-slate-400 hover:text-white'
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
                  <h3 className="font-bold text-white text-base">{selectedVideo.title}</h3>
                  <p className="text-xs text-slate-400 font-mono mt-0.5">
                    Duration: {selectedVideo.duration_seconds || 19.5}s • Format: MP4 (H.264 / AAC) • Season {selectedVideo.season}, Ep {selectedVideo.episode_number}
                  </p>
                </div>

                <div className="flex items-center space-x-3 text-xs">
                  <a
                    href={selectedVideo.final_video_url || '#'}
                    download
                    className="px-4 py-2 rounded-xl bg-gradient-to-r from-cyan-600 to-purple-600 hover:brightness-110 text-white font-bold flex items-center space-x-2 shadow-glow-cyan transition-all"
                  >
                    <Download className="w-4 h-4" />
                    <span>DOWNLOAD MP4</span>
                  </a>
                </div>
              </GlassCard>
            </>
          ) : (
            <GlassCard className="p-12 text-center text-slate-400 text-sm space-y-2">
              <Video className="w-10 h-10 mx-auto text-slate-600" />
              <p>No rendered videos found for this universe. Generate your first episode to watch rendered videos!</p>
            </GlassCard>
          )}
        </div>

        {/* Video Select Roster */}
        <div className="space-y-4">
          <h3 className="text-base font-bold text-white flex items-center space-x-2">
            <Play className="w-4 h-4 text-cyan-400" />
            <span>RENDERED EPISODES ({episodes.length})</span>
          </h3>

          <div className="space-y-3">
            {episodes.length === 0 ? (
              <GlassCard className="p-6 text-center text-slate-400 text-xs">
                No episodes rendered yet.
              </GlassCard>
            ) : (
              episodes.map((ep) => (
                <GlassCard
                  key={ep.id}
                  glow={ep.id === selectedVideo?.id}
                  className="cursor-pointer hover:border-purple-500/60 transition-all space-y-2"
                  onClick={() => handleSelectVideo(ep)}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono text-purple-400 font-bold bg-purple-950 px-2 py-0.5 rounded border border-purple-800">
                      S{ep.season} E{ep.episode_number}
                    </span>
                    {ep.id === selectedVideo?.id && (
                      <span className="text-[10px] font-mono text-cyan-300 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3 text-cyan-400" /> PLAYING
                      </span>
                    )}
                  </div>
                  <h4 className="text-sm font-bold text-white hover:text-cyan-300 transition-colors">{ep.title}</h4>
                  <p className="text-xs text-slate-400 line-clamp-2">{ep.logline}</p>
                </GlassCard>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
