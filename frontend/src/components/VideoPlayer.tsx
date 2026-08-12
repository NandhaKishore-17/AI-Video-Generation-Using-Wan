import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Play, Pause, Volume2, VolumeX, Maximize2, Captions, Download, Sparkles, Mic } from 'lucide-react';
import { Scene } from '../types';

interface VideoPlayerProps {
  videoUrl?: string;
  posterUrl?: string;
  title?: string;
  subtitles?: string;
  scenes?: Scene[];
  aspectRatio?: '16:9' | '9:16';
}

// Voice config for character TTS matching
const CHARACTER_VOICES: Record<string, { pitch: number; rate: number; name?: string }> = {
  'kaelen': { pitch: 0.85, rate: 0.95 },
  'nova': { pitch: 1.25, rate: 1.0 },
  'director': { pitch: 0.7, rate: 0.85 },
  'vane': { pitch: 0.7, rate: 0.85 },
  'default': { pitch: 1.0, rate: 0.9 },
};

function getVoiceConfig(speaker: string) {
  const lower = speaker.toLowerCase();
  for (const key of Object.keys(CHARACTER_VOICES)) {
    if (lower.includes(key)) return CHARACTER_VOICES[key];
  }
  return CHARACTER_VOICES['default'];
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
  const containerRef = useRef<HTMLDivElement>(null);
  const animationFrameRef = useRef<number | null>(null);

  // TTS state refs
  const synthRef = useRef<SpeechSynthesis | null>(null);
  const lastSpokenLineRef = useRef<string>('');
  const isTTSSpeakingRef = useRef<boolean>(false);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(35);
  const [showSubtitles, setShowSubtitles] = useState(true);
  const [useMotionFallback, setUseMotionFallback] = useState(false);
  const [activeSrc, setActiveSrc] = useState<string>('');
  const [ttsReady, setTtsReady] = useState(false);

  // Init Web Speech API
  useEffect(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      synthRef.current = window.speechSynthesis;
      // Ensure voices are loaded
      const loadVoices = () => {
        const voices = window.speechSynthesis.getVoices();
        if (voices.length > 0) setTtsReady(true);
      };
      window.speechSynthesis.onvoiceschanged = loadVoices;
      loadVoices();
    }
  }, []);

  useEffect(() => {
    setIsPlaying(false);
    setCurrentTime(0);
    setUseMotionFallback(false);
    lastSpokenLineRef.current = '';
    stopTTS();
    if (videoUrl && videoUrl.trim().length > 0) {
      setActiveSrc(videoUrl);
    } else if (scenes && scenes.length > 0 && scenes[0].video_url) {
      setActiveSrc(scenes[0].video_url);
    } else {
      setActiveSrc('');
    }
  }, [videoUrl, scenes]);

  // Stop TTS
  const stopTTS = useCallback(() => {
    if (synthRef.current && synthRef.current.speaking) {
      synthRef.current.cancel();
    }
    isTTSSpeakingRef.current = false;
  }, []);

  // Speak a dialogue line using Web Speech API
  const speakDialogue = useCallback((speaker: string, line: string) => {
    if (!synthRef.current || isMuted) return;
    const lineKey = `${speaker}::${line}`;
    if (lastSpokenLineRef.current === lineKey) return; // already spoken this line
    lastSpokenLineRef.current = lineKey;

    // Cancel current speech before new one
    synthRef.current.cancel();

    const utterance = new SpeechSynthesisUtterance(line);
    const voiceConf = getVoiceConfig(speaker);
    utterance.pitch = voiceConf.pitch;
    utterance.rate = voiceConf.rate;
    utterance.volume = 1.0;

    // Pick best available voice
    const voices = synthRef.current.getVoices();
    const engVoices = voices.filter(v => v.lang.startsWith('en'));
    if (engVoices.length > 0) {
      // Try to assign gender-appropriate voice
      const isFemale = speaker.toLowerCase().includes('nova') || speaker.toLowerCase().includes('thorne');
      const femaleVoices = engVoices.filter(v => /female|woman|girl/i.test(v.name));
      const maleVoices = engVoices.filter(v => /male|man|david|james|microsoft/i.test(v.name));
      if (isFemale && femaleVoices.length > 0) {
        utterance.voice = femaleVoices[0];
      } else if (!isFemale && maleVoices.length > 0) {
        utterance.voice = maleVoices[0];
      } else {
        utterance.voice = engVoices[0];
      }
    }

    utterance.onstart = () => { isTTSSpeakingRef.current = true; };
    utterance.onend = () => { isTTSSpeakingRef.current = false; };
    utterance.onerror = () => { isTTSSpeakingRef.current = false; };

    synthRef.current.speak(utterance);
  }, [isMuted]);

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
      { t: 4.5, speaker: 'NOVA THORNE', line: "Tracing origin IP... it leads directly to Director Vane's server." },
      { t: 9.0, speaker: 'DIRECTOR VANE', line: "You're too late, detective. The Obsidian Protocol is already live." },
      { t: 13.5, speaker: 'KAELEN VANCE', line: 'Executing counter-override. Hold the extraction line!' }
    ];

    const currentDialogue = defaultScript.reduce((prev, curr) => {
      return currentTime >= curr.t ? curr : prev;
    }, defaultScript[0]);

    return currentDialogue;
  };

  const activeSubtitle = getActiveSubtitle();

  // TTS playback: Speak dialogue lines when they become active during playback
  useEffect(() => {
    if (isPlaying && !isMuted && activeSubtitle && activeSubtitle.line) {
      speakDialogue(activeSubtitle.speaker, activeSubtitle.line);
    }
  }, [activeSubtitle?.speaker, activeSubtitle?.line, isPlaying, isMuted, speakDialogue]);

  // Stop TTS when paused / muted / unmounted
  useEffect(() => {
    if (!isPlaying || isMuted) {
      stopTTS();
    }
  }, [isPlaying, isMuted, stopTTS]);

  useEffect(() => {
    return () => {
      stopTTS();
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [stopTTS]);

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
            stopTTS();
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
  }, [isPlaying, useMotionFallback, duration, stopTTS]);

  const togglePlay = async () => {
    if (isPlaying) {
      if (videoRef.current) videoRef.current.pause();
      stopTTS();
      setIsPlaying(false);
      return;
    }

    setIsMuted(false);
    lastSpokenLineRef.current = ''; // Reset so first line speaks again

    // Try HTML5 video play
    if (videoRef.current && !useMotionFallback) {
      try {
        videoRef.current.muted = false;
        videoRef.current.volume = 1.0;
        await videoRef.current.play();
        setIsPlaying(true);
        return;
      } catch (err) {
        console.warn('Unmuted video play failed, trying muted video + TTS...', err);
        try {
          videoRef.current.muted = true;
          await videoRef.current.play();
          setIsPlaying(true);
          return;
        } catch (err2) {
          console.warn('Switching to Cinematic Motion Engine Fallback...', err2);
        }
      }
    }

    // Fallback: Motion Synthesizer + TTS narration
    setUseMotionFallback(true);
    setIsPlaying(true);
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
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const targetTime = Number(e.target.value);
    setCurrentTime(targetTime);
    lastSpokenLineRef.current = ''; // Reset TTS on seek
    stopTTS();
    if (videoRef.current && !useMotionFallback) {
      videoRef.current.currentTime = targetTime;
    }
  };

  const toggleMute = () => {
    const nextMuted = !isMuted;
    if (videoRef.current) videoRef.current.muted = nextMuted;
    setIsMuted(nextMuted);
    if (nextMuted) {
      stopTTS();
    }
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

  const handleDownload = async () => {
    if (!activeSrc) return;
    
    try {
      const downloadUrl = `/api/v1/videos/download?path=${encodeURIComponent(activeSrc)}`;
      const response = await fetch(downloadUrl);
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Video download failed: ${response.status} ${errorText}`);
      }
      
      const blob = await response.blob();
      console.log("VIDEO DOWNLOAD");
      console.log("status:", response.status);
      console.log("contentType:", response.headers.get("content-type"));
      console.log("blobType:", blob.type);
      console.log("blobSize:", blob.size);

      if (blob.size === 0) {
        throw new Error("Downloaded video is empty");
      }
      
      const objectUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = objectUrl;
      link.download = `${(title || 'episode').replace(/[^a-zA-Z0-9_-]/g, '_')}.mp4`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(objectUrl);
    } catch (err) {
      console.error("Failed to download video:", err);
      alert("Failed to download video file. The file may not exist yet or an error occurred.");
    }
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
      {/* HTML5 Video Element */}
      {!useMotionFallback ? (
        <video
          ref={videoRef}
          src={activeSrc}
          poster={posterUrl}
          controls={false}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleVideoLoadedMetadata}
          onEnded={() => { setIsPlaying(false); stopTTS(); }}
          onError={() => setUseMotionFallback(true)}
          playsInline
          muted={isMuted}
          className="w-full h-full object-cover cursor-pointer z-0"
          onClick={togglePlay}
        />
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

          {/* Glowing Ambient Lighting */}
          <div
            className={`absolute inset-0 pointer-events-none transition-opacity duration-700 ${
              isPlaying ? 'opacity-40 animate-pulse' : 'opacity-20'
            }`}
            style={{ background: 'radial-gradient(circle at center, rgba(6,182,212,0.2) 0%, transparent 70%)' }}
          />

          {/* Audio/TTS waveform indicator when playing */}
          {isPlaying && (
            <div className="absolute top-4 right-4 flex items-end space-x-1 z-20 bg-black/60 px-3 py-1.5 rounded-full border border-cyan-500/40">
              <Mic className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
              <div className="w-1 h-3 bg-cyan-400 animate-bounce" />
              <div className="w-1 h-4 bg-purple-400 animate-bounce [animation-delay:0.2s]" />
              <div className="w-1 h-2 bg-pink-400 animate-bounce [animation-delay:0.4s]" />
              <span className="text-[10px] font-mono text-cyan-300 ml-1 font-bold">CHARACTER VOICE ACTIVE</span>
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
            <p className="text-[11px] font-mono text-cyan-300 mt-1">CLICK TO PLAY WITH VOICE DIALOGUE</p>
          </div>
        </div>
      )}

      {/* Subtitles Box Overlay Matched to Character Voice */}
      {showSubtitles && isPlaying && activeSubtitle && activeSubtitle.line && (
        <div className="absolute bottom-16 left-1/2 -translate-x-1/2 px-4 py-2.5 rounded-xl bg-black/95 text-center max-w-[85%] z-20 shadow-2xl pointer-events-none border border-slate-700/80 flex flex-col items-center space-y-1">
          <div className="flex items-center space-x-1.5">
            <Mic className="w-3 h-3 text-cyan-400 animate-pulse" />
            <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border uppercase ${
              activeSubtitle.speaker.toUpperCase().includes('KAELEN') ? 'bg-cyan-950 text-cyan-300 border-cyan-800' :
              activeSubtitle.speaker.toUpperCase().includes('NOVA') ? 'bg-amber-950 text-amber-300 border-amber-800' :
              activeSubtitle.speaker.toUpperCase().includes('VANE') || activeSubtitle.speaker.toUpperCase().includes('DIRECTOR') ? 'bg-purple-950 text-purple-300 border-purple-800' :
              'bg-slate-900 text-slate-300 border-slate-700'
            }`}>
              {activeSubtitle.speaker}
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
            {/* TTS Active indicator */}
            {ttsReady && isPlaying && !isMuted && (
              <span className="text-[9px] font-mono text-emerald-400 flex items-center gap-1">
                <Mic className="w-3 h-3 animate-pulse" /> TTS ON
              </span>
            )}

            <button
              onClick={() => setShowSubtitles(!showSubtitles)}
              className={`p-1 rounded text-[11px] font-mono flex items-center space-x-1 transition-colors ${
                showSubtitles ? 'bg-cyan-950 text-cyan-300 border border-cyan-700' : 'text-slate-400 hover:text-white'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
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
