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

  useEffect(() => {
    if (episode) {
      loadDetail(episode.id);
    }
  }, [episode]);

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
              <a
                href={currentEp.final_video_url}
                download
                className="px-3.5 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs font-mono flex items-center space-x-1 shadow-glow-cyan"
              >
                <Download className="w-4 h-4" />
                <span>DOWNLOAD MP4</span>
              </a>
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
          <p className="text-xs text-slate-300 italic border-l-2 border-cyan-500 pl-3">
            {currentEp.logline}
          </p>

          {activeTab === 'video' ? (
            <div className="space-y-4">
              <VideoPlayer
                videoUrl={currentEp.final_video_url}
                posterUrl={currentEp.thumbnail_url}
                title={currentEp.title}
                subtitles="Burnt-in Subtitles"
                scenes={currentEp.scenes}
              />
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
