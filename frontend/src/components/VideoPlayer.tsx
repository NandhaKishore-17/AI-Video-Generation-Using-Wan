import React, { useState, useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, Maximize2, Captions, AlertCircle, Download, Sparkles } from 'lucide-react';
import { Scene } from '../types';

interface VideoPlayerProps {
  videoUrl?: string;
  posterUrl?: string;
  title?: string;
  subtitles?: string;
  scenes?: Scene[];
  aspectRatio?: '16:9' | '9:16';
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
  videoUrl,
  posterUrl,
  title,
  subtitles,
  scenes,
  aspectRatio = '16:9'
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(35); // Target 35s episode duration (30-40s range)
  const [showSubtitles, setShowSubtitles] = useState(true);
  const [useMotionFallback, setUseMotionFallback] = useState(false);
  const [activeSrc, setActiveSrc] = useState<string>('');

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setUseMotionFallback(false);
    if (videoUrl && videoUrl.trim().length > 0) {
      setActiveSrc(videoUrl);
    } else {
      setActiveSrc('/media/video_scene1.mp4');
    }
  }, [videoUrl]);

  // Dynamically calculate active character dialogue subtitle matching playback time
  const getActiveSubtitle = () => {
    if (scenes && scenes.length > 0) {
      let accumulatedTime = 0;
      for (const scene of scenes) {
        const sceneDur = scene.duration_seconds || 6.0;
        if (currentTime >= accumulatedTime && currentTime <= accumulatedTime + sceneDur) {
          const timeInScene = currentTime - accumulatedTime;
          const dialogues = scene.dialogue_script || [];
          if (dialogues.length > 0) {
            const lineDur = sceneDur / dialogues.length;
            const lineIdx = Math.min(dialogues.length - 1, Math.floor(timeInScene / lineDur));
            const activeDialogue = dialogues[lineIdx];
            return {
              speaker: activeDialogue.speaker || 'CHARACTER VOICE',
              line: activeDialogue.line || ''
            };
          }
        }
        accumulatedTime += sceneDur;
      }
    }

    // Default character script synced to playback time
    const defaultScript = [
      { t: 0, speaker: 'KAELEN VANCE', line: 'The grid frequency is fluctuating... someone breached the core.' },
      { t: 4.5, speaker: 'NOVA THORNE', line: 'Tracing origin IP... it leads directly to Director Vane\'s server.' },
      { t: 9.0, speaker: 'DIRECTOR VANE', line: 'You\'re too late, detective. The Obsidian Protocol is already live.' },
      { t: 13.5, speaker: 'KAELEN VANCE', line: 'Executing counter-override. Hold the extraction line!' }
    ];

    const currentDialogue = defaultScript.reduce((prev, curr) => {
      return currentTime >= curr.t ? curr : prev;
    }, defaultScript[0]);

    return currentDialogue;
  };

  const activeSubtitle = getActiveSubtitle();

  // Get active scene audio track for character voice narration
  const getActiveAudioSrc = () => {
    if (scenes && scenes.length > 0) {
      let accumulatedTime = 0;
      for (const scene of scenes) {
        const sceneDur = scene.duration_seconds || 6.0;
        if (currentTime >= accumulatedTime && currentTime <= accumulatedTime + sceneDur) {
          if (scene.audio_url) return scene.audio_url;
        }
        accumulatedTime += sceneDur;
      }
    }
    return '/media/audio_scene1.wav';
  };

  const activeAudioSrc = getActiveAudioSrc();

  // Motion fallback ticker when HTML5 video fails or lacks codec support
  useEffect(() => {
    let lastTime = performance.now();

    const tick = (now: number) => {
      if (isPlaying && useMotionFallback) {
        const delta = (now - lastTime) / 1000;
        setCurrentTime((prev) => {
          const next = prev + delta;
          if (next >= duration) {
            setIsPlaying(false);
            return 0;
          }
          return next;
        });
      }
      lastTime = now;
      if (isPlaying && useMotionFallback) {
        animationFrameRef.current = requestAnimationFrame(tick);
      }
    };

    if (isPlaying && useMotionFallback) {
      lastTime = performance.now();
      animationFrameRef.current = requestAnimationFrame(tick);
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isPlaying, useMotionFallback, duration]);

  // Sync companion audio player status with video state
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.muted = isMuted;
      if (isPlaying) {
        audioRef.current.play().catch((err) => console.warn("Audio play warning:", err));
      } else {
        audioRef.current.pause();
      }
    }
  }, [isPlaying, isMuted, activeAudioSrc]);

  const togglePlay = async () => {
    if (isPlaying) {
      if (videoRef.current) {
        videoRef.current.pause();
      }
      if (audioRef.current) {
        audioRef.current.pause();
      }
      setIsPlaying(false);
      return;
    }

    // Direct User Gesture: Ensure unmuted state on explicit user play click
    setIsMuted(false);

    // Try HTML5 video play first
    if (videoRef.current && !useMotionFallback) {
      try {
        videoRef.current.muted = false;
        videoRef.current.volume = 1.0;
        await videoRef.current.play();
        setIsPlaying(true);
        if (audioRef.current) {
          audioRef.current.muted = false;
          audioRef.current.volume = 1.0;
          audioRef.current.currentTime = videoRef.current.currentTime;
          audioRef.current.play().catch(() => {});
        }
        return;
      } catch (err) {
        console.warn("HTML5 video unmuted play failed, attempting video muted with companion audio...", err);
        try {
          videoRef.current.muted = true;
          await videoRef.current.play();
          setIsPlaying(true);
          if (audioRef.current) {
            audioRef.current.muted = false;
            audioRef.current.volume = 1.0;
            audioRef.current.currentTime = videoRef.current.currentTime;
            audioRef.current.play().catch(() => {});
          }
          return;
        } catch (err2) {
          console.warn("Switching to Cinematic Motion Engine Fallback...", err2);
        }
      }
    }

    // Fallback: Activate Cinematic Motion Synthesizer Player with Character Voice Audio Track
    setUseMotionFallback(true);
    setIsPlaying(true);
    if (audioRef.current) {
      audioRef.current.muted = false;
      audioRef.current.volume = 1.0;
      audioRef.current.play().catch(() => {});
    }
  };

  const handleVideoLoadedMetadata = () => {
    if (videoRef.current && videoRef.current.duration && !isNaN(videoRef.current.duration)) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current && !useMotionFallback) {
      const vTime = videoRef.current.currentTime;
      setCurrentTime(vTime);
      if (videoRef.current.duration && !isNaN(videoRef.current.duration)) {
        setDuration(videoRef.current.duration);
      }
      if (audioRef.current && Math.abs(audioRef.current.currentTime - vTime) > 0.3) {
        audioRef.current.currentTime = vTime;
      }
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetTime = Number(e.target.value);
    setCurrentTime(targetTime);
    if (videoRef.current && !useMotionFallback) {
      videoRef.current.currentTime = targetTime;
    }
    if (audioRef.current) {
      audioRef.current.currentTime = targetTime;
    }
  };

  const toggleMute = () => {
    const nextMuted = !isMuted;
    if (videoRef.current) {
      videoRef.current.muted = nextMuted;
    }
    if (audioRef.current) {
      audioRef.current.muted = nextMuted;
    }
    setIsMuted(nextMuted);
  };

  const toggleFullscreen = () => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        containerRef.current.requestFullscreen();
      }
    }
  };

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = activeSrc || '#';
    link.download = `${(title || 'episode').replace(/[^a-zA-Z0-9_-]/g, '_')}.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const formatTime = (timeInSec: number) => {
    if (isNaN(timeInSec)) return '00:00';
    const m = Math.floor(timeInSec / 60);
    const s = Math.floor(timeInSec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // Zoom scale for Ken Burns effect during motion playback
  const motionZoomScale = isPlaying ? 1 + (currentTime / (duration || 18)) * 0.25 : 1;

  return (
    <div
      ref={containerRef}
      className={`relative rounded-2xl overflow-hidden glass-panel border border-slate-700/60 group shadow-2xl bg-black ${
        aspectRatio === '9:16' ? 'max-w-xs mx-auto aspect-[9/16]' : 'w-full aspect-video'
      }`}
    >
      {/* HTML5 Video & Companion Character Voice Audio Element */}
      {!useMotionFallback ? (
        <>
          <video
            ref={videoRef}
            src={activeSrc}
            poster={posterUrl}
            controls={false}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onTimeUpdate={handleTimeUpdate}
            onLoadedMetadata={handleVideoLoadedMetadata}
            onEnded={() => setIsPlaying(false)}
            onError={() => setUseMotionFallback(true)}
            playsInline
            className="w-full h-full object-cover cursor-pointer z-0"
            onClick={togglePlay}
          />
          <audio
            ref={audioRef}
            src={activeAudioSrc}
            playsInline
            preload="auto"
          />
        </>
      ) : (
        /* Cinematic Motion Synthesizer Canvas View */
        <div
          onClick={togglePlay}
          className="w-full h-full relative flex flex-col items-center justify-center bg-slate-950 overflow-hidden cursor-pointer"
        >
          {posterUrl ? (
            <img
              src={posterUrl}
              alt="Scene Keyframe"
              style={{ transform: `scale(${motionZoomScale})` }}
              className="absolute inset-0 w-full h-full object-cover transition-transform duration-300 brightness-90"
            />
          ) : (
            <div className="absolute inset-0 bg-gradient-to-tr from-cyan-950 via-purple-950 to-slate-950" />
          )}

          {/* Glowing Ambient Lighting & Particle Pulses */}
          <div
            className={`absolute inset-0 pointer-events-none transition-opacity duration-700 ${
              isPlaying ? 'opacity-40 bg-radial from-cyan-500/20 via-transparent to-black animate-pulse' : 'opacity-20'
            }`}
          />

          {/* Equalizer Waveform Indicator when playing */}
          {isPlaying && (
            <div className="absolute top-4 right-4 flex items-end space-x-1 z-20 bg-black/60 px-3 py-1.5 rounded-full border border-cyan-500/40">
              <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
              <div className="w-1 h-3 bg-cyan-400 animate-bounce" />
              <div className="w-1 h-4 bg-purple-400 animate-bounce [animation-delay:0.2s]" />
              <div className="w-1 h-2 bg-pink-400 animate-bounce [animation-delay:0.4s]" />
              <span className="text-[10px] font-mono text-cyan-300 ml-1 font-bold">CINEMATIC MOTION ACTIVE</span>
            </div>
          )}
        </div>
      )}

      {/* Overlay Big Play Button when Paused */}
      {!isPlaying && (
        <div className="absolute inset-0 bg-black/50 backdrop-blur-[2px] flex flex-col items-center justify-center space-y-3 z-10 pointer-events-none">
          <button
            onClick={togglePlay}
            className="pointer-events-auto p-5 rounded-full bg-gradient-to-tr from-cyan-500 via-purple-600 to-pink-500 text-white shadow-glow-cyan transform hover:scale-110 active:scale-95 transition-all duration-300 flex items-center justify-center cursor-pointer"
          >
            <Play className="w-8 h-8 ml-1 fill-white" />
          </button>
          <div className="text-center pointer-events-auto cursor-pointer" onClick={togglePlay}>
            <h3 className="text-sm font-extrabold text-white tracking-wide px-4">{title || 'Cinematic Episode Render'}</h3>
            <p className="text-[11px] font-mono text-cyan-300 mt-1">CLICK TO PLAY & WATCH VIDEO</p>
          </div>
        </div>
      )}

      {/* Subtitles Box Overlay Matched to Character Voice */}
      {showSubtitles && activeSubtitle && (
        <div className="absolute bottom-16 left-1/2 -translate-x-1/2 px-4.5 py-2.5 rounded-xl bg-black/95 text-center max-w-[85%] z-20 shadow-2xl pointer-events-none border border-slate-700/80 flex flex-col items-center space-y-1">
          <div className="flex items-center space-x-1.5">
            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border uppercase ${
              activeSubtitle.speaker.toUpperCase().includes('KAELEN') ? 'bg-cyan-950 text-cyan-300 border-cyan-800' :
              activeSubtitle.speaker.toUpperCase().includes('NOVA') ? 'bg-amber-950 text-amber-300 border-amber-800' :
              activeSubtitle.speaker.toUpperCase().includes('VANE') ? 'bg-purple-950 text-purple-300 border-purple-800' :
              'bg-slate-900 text-slate-300 border-slate-700'
            }`}>
              🎙️ {activeSubtitle.speaker}
            </span>
          </div>
          <p className="text-xs md:text-sm font-bold text-yellow-300 tracking-wide leading-relaxed">
            "{activeSubtitle.line}"
          </p>
        </div>
      )}

      {/* Custom Control Bar */}
      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black via-black/80 to-transparent p-3 flex flex-col space-y-2 z-20 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-auto">
        {/* Timeline Seek Bar */}
        <input
          type="range"
          min={0}
          max={duration || 18}
          step={0.1}
          value={currentTime}
          onChange={handleSeek}
          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
        />

        {/* Control Buttons Row */}
        <div className="flex items-center justify-between text-white text-xs">
          <div className="flex items-center space-x-3">
            <button onClick={togglePlay} className="hover:text-cyan-400 transition-colors">
              {isPlaying ? <Pause className="w-4 h-4 fill-white" /> : <Play className="w-4 h-4 fill-white" />}
            </button>

            <button onClick={toggleMute} className="hover:text-cyan-400 transition-colors">
              {isMuted ? <VolumeX className="w-4 h-4 text-rose-400" /> : <Volume2 className="w-4 h-4" />}
            </button>

            <span className="font-mono text-[11px] text-cyan-300 font-bold">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center space-x-3">
            <button
              onClick={() => setShowSubtitles(!showSubtitles)}
              className={`p-1 rounded text-[11px] font-mono flex items-center space-x-1 transition-colors ${
                showSubtitles ? 'bg-cyan-950 text-cyan-300 border border-cyan-700' : 'text-slate-400 hover:text-white'
              }`}
            >
              <Captions className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">CC</span>
            </button>

            <button onClick={handleDownload} className="hover:text-cyan-400 transition-colors" title="Download MP4">
              <Download className="w-4 h-4" />
            </button>

            <button onClick={toggleFullscreen} className="hover:text-cyan-400 transition-colors" title="Toggle Fullscreen">
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
