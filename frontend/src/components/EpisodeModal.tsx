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
      
      interval = setInterval(async () => {
        pollCountRef.current += 1;
        if (pollCountRef.current > 300) {
          clearInterval(interval);
          return;
        }
        
        try {
          const detail = await api.getEpisodeDetail(episode.id);
          setFullEpisodeDetail(detail);
          if (detail && (detail.status === 'COMPLETED' || detail.status === 'FAILED')) {
            clearInterval(interval);
          }
        } catch (err) {
          console.error("Polling error:", err);
        }
      }, 10000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [episode]);

  const loadDetail = async (id: string) => {
    const detail = await api.getEpisodeDetail(id);
    setFullEpisodeDetail(detail);
  };

  if (!episode) return null;

  const currentEp = fullEpisodeDetail || episode;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="relative w-full max-w-4xl max-h-[90vh] bg-[#0E1424] border border-neutral-border/80 rounded-3xl overflow-hidden flex flex-col shadow-2xl">
        {/* Header */}
        <div className="p-5 bg-neutral-soft border-b border-neutral-border flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-2 text-xs font-mono text-accent-primary font-bold">
              <span>SEASON {currentEp.season} EPISODE {currentEp.episode_number}</span>
              <span>•</span>
              <span className="text-text-secondary font-sans">{currentEp.status}</span>
            </div>
            <h2 className="text-xl font-extrabold text-text-primary tracking-tight">{currentEp.title}</h2>
          </div>

          <div className="flex items-center space-x-3">
            {/* View Switcher */}
            <div className="flex items-center space-x-1.5 bg-neutral-soft p-1 rounded-xl border border-neutral-border text-xs">
              <button
                onClick={() => setActiveTab('video')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-mono transition-all ${
                  activeTab === 'video' ? 'bg-accent-primary text-text-primary font-bold shadow-sm' : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                <Film className="w-3.5 h-3.5" />
                <span>WATCH VIDEO</span>
              </button>

              <button
                onClick={() => setActiveTab('script')}
                className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg font-mono transition-all ${
                  activeTab === 'script' ? 'bg-accent-primary text-text-primary font-bold' : 'text-text-secondary hover:text-text-primary'
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
                className="px-3.5 py-2 rounded-xl bg-accent-primary hover:bg-accent-primary text-text-primary font-bold text-xs font-mono flex items-center space-x-1 shadow-sm"
              >
                <Download className="w-4 h-4" />
                <span>DOWNLOAD MP4</span>
              </button>
            )}

            <button
              onClick={onClose}
              className="p-2 rounded-xl bg-neutral-soft hover:bg-neutral-soft text-text-secondary hover:text-text-primary transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Modal Body Scroll Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          <div className="space-y-3">
            <p className="text-xs text-text-secondary italic border-l-2 border-accent-primary pl-3">
              {currentEp.logline}
            </p>
            {currentEp.summary && (
              <p className="text-xs text-text-secondary bg-neutral-soft border border-neutral-border rounded-2xl p-3">
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
                <div className="w-full aspect-video bg-neutral-soft border border-neutral-border rounded-2xl flex flex-col items-center justify-center space-y-4">
                  <X className="w-12 h-12 text-text-secondary" />
                  <p className="text-text-secondary font-mono text-sm">Episode Generation Failed</p>
                </div>
              ) : (
                <div className="w-full aspect-video bg-white border border-emerald-200 rounded-2xl flex flex-col items-center justify-center space-y-6">
                  <Sparkles className="w-12 h-12 text-accent-primary animate-spin" />
                  <div className="text-center">
                    <p className="text-accent-primary font-bold tracking-widest animate-pulse">AUTONOMOUS GENERATION IN PROGRESS</p>
                    <p className="text-text-secondary text-sm mt-2 font-mono">Current Status: {currentEp.status}</p>
                    <p className="text-text-secondary text-xs mt-1 italic">Writing screenplay, parsing scenes, rendering video...</p>
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
