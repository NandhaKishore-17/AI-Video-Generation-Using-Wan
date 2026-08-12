import React, { useState, useEffect } from 'react';
import { Episode } from '../types';
import { VideoPlayer } from './VideoPlayer';
import { ScreenplayView } from './ScreenplayView';
import { api } from '../services/api';
import { X, Film, FileText, Download, Play, Sparkles } from 'lucide-react';

interface EpisodeModalProps {
  episode: Episode | null;
  onClose: () => void;
}

export const EpisodeModal: React.FC<EpisodeModalProps> = ({ episode, onClose }) => {
  const [activeTab, setActiveTab] = useState<'video' | 'script'>('video');
  const [fullEpisodeDetail, setFullEpisodeDetail] = useState<Episode | null>(null);

  const pollCountRef = React.useRef(0);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (episode) {
      loadDetail(episode.id);
      pollCountRef.current = 0;
      
      interval = setInterval(() => {
        pollCountRef.current += 1;
        if (fullEpisodeDetail && (fullEpisodeDetail.status === 'COMPLETED' || fullEpisodeDetail.status === 'FAILED') || pollCountRef.current > 200) {
          clearInterval(interval);
        } else {
          loadDetail(episode.id);
        }
      }, 3000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [episode, fullEpisodeDetail?.status]);

  const loadDetail = async (id: string) => {
    const detail = await api.getEpisodeDetail(id);
    setFullEpisodeDetail(detail);
  };

  if (!episode) return null;

  const currentEp = fullEpisodeDetail || episode;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-[#0E1424] border border-slate-700/80 rounded-3xl overflow-hidden flex flex-col shadow-2xl">
        {/* Header */}
        <div className="p-5 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-2 text-xs font-mono text-cyan-400 font-bold">
              <span>SEASON {currentEp.season} EPISODE {currentEp.episode_number}</span>
              <span>•</span>
              <span className="text-purple-400 font-sans">{currentEp.status}</span>
            </div>
            <h2 className="text-xl font-extrabold text-white tracking-tight">{currentEp.title}</h2>
          </div>

          <div className="flex items-center space-x-3">
            {/* View Switcher */}
            <div className="flex items-center space-x-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs">
              <button
                onClick={() => setActiveTab('video')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-mono transition-all ${
                  activeTab === 'video' ? 'bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-bold shadow-glow-cyan' : 'text-slate-400 hover:text-white'
                }`}
              >
                <Film className="w-3.5 h-3.5" />
                <span>WATCH VIDEO</span>
              </button>

              <button
                onClick={() => setActiveTab('script')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-mono transition-all ${
                  activeTab === 'script' ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white font-bold' : 'text-slate-400 hover:text-white'
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>READ SCRIPT</span>
              </button>
            </div>

            {currentEp.final_video_url && (
              <button
                onClick={async () => {
                  try {
                    const downloadUrl = `/api/v1/videos/download?path=${encodeURIComponent(currentEp.final_video_url!)}`;
                    const res = await fetch(downloadUrl);
                    if (!res.ok) {
                      const errorText = await res.text();
                      throw new Error(`Video download failed: ${res.status} ${errorText}`);
                    }
                    const blob = await res.blob();
                    
                    console.log("VIDEO DOWNLOAD");
                    console.log("status:", res.status);
                    console.log("contentType:", res.headers.get("content-type"));
                    console.log("blobType:", blob.type);
                    console.log("blobSize:", blob.size);

                    if (blob.size === 0) {
                      throw new Error("Downloaded video is empty");
                    }
                    
                    const objectUrl = window.URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = objectUrl;
                    link.download = `${(currentEp.title || 'episode').replace(/[^a-zA-Z0-9_-]/g, '_')}.mp4`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    window.URL.revokeObjectURL(objectUrl);
                  } catch (err) {
                    console.error('Failed to download video:', err);
                    alert('Failed to download video file. The file may not exist yet or an error occurred.');
                  }
                }}
                className="px-3.5 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs font-mono flex items-center space-x-1 shadow-glow-cyan"
              >
                <Download className="w-4 h-4" />
                <span>DOWNLOAD MP4</span>
              </button>
            )}

            <button
              onClick={onClose}
              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body Scroll Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <div className="space-y-3">
            <p className="text-xs text-slate-300 italic border-l-2 border-cyan-500 pl-3">
              {currentEp.logline}
            </p>
            {currentEp.summary && (
              <p className="text-xs text-slate-400 bg-slate-950/80 border border-slate-800 rounded-2xl p-3">
                {currentEp.summary}
              </p>
            )}
          </div>

          {activeTab === 'video' ? (
            <div className="space-y-4">
              {currentEp.status === 'COMPLETED' ? (
                <VideoPlayer
                  videoUrl={currentEp.final_video_url}
                  posterUrl={currentEp.thumbnail_url}
                  title={currentEp.title}
                  subtitles="Burnt-in Subtitles"
                  scenes={currentEp.scenes}
                />
              ) : currentEp.status === 'FAILED' ? (
                <div className="w-full aspect-video bg-rose-950/30 border border-rose-800 rounded-2xl flex flex-col items-center justify-center space-y-4">
                  <X className="w-12 h-12 text-rose-500" />
                  <p className="text-rose-400 font-mono text-sm">Episode Generation Failed</p>
                </div>
              ) : (
                <div className="w-full aspect-video bg-slate-900 border border-cyan-800/50 rounded-2xl flex flex-col items-center justify-center space-y-6">
                  <Sparkles className="w-12 h-12 text-cyan-400 animate-spin" />
                  <div className="text-center">
                    <p className="text-cyan-400 font-bold tracking-widest animate-pulse">AUTONOMOUS GENERATION IN PROGRESS</p>
                    <p className="text-slate-400 text-sm mt-2 font-mono">Current Status: {currentEp.status}</p>
                    <p className="text-slate-500 text-xs mt-1 italic">Writing screenplay, parsing scenes, rendering video...</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <ScreenplayView
              scenes={currentEp.scenes || []}
              title={currentEp.title}
              logline={currentEp.logline}
            />
          )}
        </div>
      </div>
    </div>
  );
};
