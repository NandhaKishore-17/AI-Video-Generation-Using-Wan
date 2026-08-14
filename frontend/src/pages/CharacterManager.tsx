import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { api } from '../services/api';
import { Character, Universe } from '../types';
import { Users, Mic, Eye, UserPlus, Sparkles, X, CheckCircle2, Trash2 } from 'lucide-react';

interface CharacterManagerProps {
  activeUniverseId?: string;
}

export const CharacterManager: React.FC<CharacterManagerProps> = ({ activeUniverseId }) => {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [universeId, setUniverseId] = useState<string>('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [availableVoices, setAvailableVoices] = useState<any[]>([]);

  // Form states
  const [name, setName] = useState('');
  const [role, setRole] = useState('Protagonist');
  const [personality, setPersonality] = useState('');
  const [appearancePrompt, setAppearancePrompt] = useState('');
  const [voicePreset, setVoicePreset] = useState('en-US-ChristopherNeural');
  const [voicePitch, setVoicePitch] = useState(1.0);
  const [voiceSpeed, setVoiceSpeed] = useState(1.0);
  const [bio, setBio] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadData();
  }, [activeUniverseId]);

  const loadData = async () => {
    const universes = await api.getUniverses();
    const targetUId = activeUniverseId || (universes.length > 0 ? universes[0].id : 'u-cyber-99');
    setUniverseId(targetUId);
    const detail = await api.getUniverseDetail(targetUId);
    setCharacters(detail.characters || []);

    const voices = await api.getAvailableVoices();
    setAvailableVoices(voices || []);
    if (voices && voices.length > 0) {
      setVoicePreset(voices[0].id);
    }
  };

  const handleDeleteCharacter = async (id: string) => {
    if (window.confirm("Are you sure you want to delete this character profile?")) {
      await api.deleteCharacter(id);
      loadData();
    }
  };

  const handleAddCharacter = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !personality || !appearancePrompt) return;
    setSaving(true);
    try {
      const newChar = await api.createCharacter({
        universe_id: universeId || 'u-cyber-99',
        name,
        role,
        personality,
        appearance_prompt: appearancePrompt,
        voice_actor_preset: voicePreset,
        voice_pitch: voicePitch,
        voice_speed: voiceSpeed,
        bio
      });
      setCharacters([...characters, newChar]);
      setShowAddModal(false);
      // Reset form
      setName('');
      setPersonality('');
      setAppearancePrompt('');
      setBio('');
    } catch (err) {
      console.error(err);
    } finally {
      setSaving(false);
    }
  };


  return (
    <div className="space-y-8">
      {/* Add Character Modal */}
      {showAddModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
          <div className="relative w-full max-w-xl bg-[#0E1424] border border-slate-700/80 rounded-3xl overflow-hidden shadow-2xl p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <h2 className="text-xl font-bold text-white flex items-center space-x-2">
                <UserPlus className="w-5 h-5 text-cyan-400" />
                <span>ADD NEW CHARACTER DETAILS</span>
              </h2>
              <button
                onClick={() => setShowAddModal(false)}
                className="p-1.5 rounded-lg bg-slate-800 text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleAddCharacter} className="space-y-4 text-xs font-mono">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-slate-400 mb-1">CHARACTER NAME</label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. Commander Vex"
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                    required
                  />
                </div>

                <div>
                  <label className="block text-slate-400 mb-1">ROLE</label>
                  <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                  >
                    <option value="Protagonist">Protagonist</option>
                    <option value="Deuteragonist">Deuteragonist</option>
                    <option value="Antagonist">Antagonist</option>
                    <option value="Sidekick">Sidekick</option>
                    <option value="Mentor">Mentor</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">PERSONALITY TRAITS</label>
                <input
                  type="text"
                  value={personality}
                  onChange={(e) => setPersonality(e.target.value)}
                  placeholder="e.g. Tactical genius, quiet, deeply loyal"
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">FLUX / SDXL VISUAL APPEARANCE PROMPT</label>
                <textarea
                  value={appearancePrompt}
                  onChange={(e) => setAppearancePrompt(e.target.value)}
                  placeholder="Detailed visual prompt anchor (e.g. Futuristic commander in chrome armor, glowing cyber eye, 8k portrait)"
                  rows={3}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-slate-400 mb-1 flex items-center justify-between">
                    <span>REALISTIC HUMAN VOICE PRESET</span>
                    <span className="text-[10px] text-cyan-400 font-bold">F5-TTS Zero-Shot / 24kHz HD</span>
                  </label>
                  <select
                    value={voicePreset}
                    onChange={(e) => setVoicePreset(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                  >
                    {availableVoices.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.gender.toUpperCase()} - {v.name} ({v.description})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-slate-400 mb-1">VOICE PITCH: {voicePitch}x</label>
                  <input
                    type="range"
                    min="0.5"
                    max="1.5"
                    step="0.1"
                    value={voicePitch}
                    onChange={(e) => setVoicePitch(Number(e.target.value))}
                    className="w-full mt-2 accent-cyan-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">BIO & BACKSTORY</label>
                <textarea
                  value={bio}
                  onChange={(e) => setBio(e.target.value)}
                  placeholder="Character origin story and background..."
                  rows={2}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  disabled={saving}
                  className="w-full py-3 rounded-xl font-bold bg-gradient-to-r from-cyan-500 via-purple-600 to-pink-500 text-white shadow-glow-cyan flex items-center justify-center space-x-2"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>{saving ? 'SAVING CHARACTER...' : 'CREATE CHARACTER'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight flex items-center gap-2">
            <Users className="w-6 h-6 text-purple-400" />
            <span>CHARACTER CONSISTENCY & VOICE HUB</span>
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Maintain exact visual appearance anchors for FLUX diffusers and voice actor presets for Piper/Kokoro TTS.
          </p>
        </div>

        <button
          onClick={() => setShowAddModal(true)}
          className="px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-cyan-500 to-purple-600 hover:brightness-110 text-white shadow-glow-cyan flex items-center space-x-2"
        >
          <UserPlus className="w-4 h-4" />
          <span>ADD NEW CHARACTER</span>
        </button>
      </div>

      {/* Character Roster */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {characters.map((char) => (
          <GlassCard key={char.id} className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-bold text-white">{char.name}</h3>
                <span className="text-xs px-2.5 py-0.5 rounded-md bg-purple-950 text-purple-300 border border-purple-800 font-mono">
                  {char.role}
                </span>
              </div>
              <button
                onClick={() => handleDeleteCharacter(char.id)}
                className="p-2 rounded-xl bg-slate-900 hover:bg-rose-900/80 text-slate-400 hover:text-rose-300 border border-slate-800 transition-colors"
                title="Delete Character"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>


            <p className="text-xs text-slate-300 italic">{char.personality}</p>

            <div className="space-y-2 text-xs font-mono">
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold">
                  <Eye className="w-3.5 h-3.5" />
                  <span>FLUX VISUAL ANCHOR PROMPT</span>
                </div>
                <p className="text-slate-300 text-[11px]">{char.appearance_prompt}</p>
              </div>

              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 space-y-1">
                <div className="flex items-center space-x-1.5 text-emerald-400 font-semibold">
                  <Mic className="w-3.5 h-3.5" />
                  <span>PIPER / KOKORO VOICE PRESET</span>
                </div>
                <p className="text-slate-300 text-[11px]">{char.voice_actor_preset} (Pitch: {char.voice_pitch}x)</p>
              </div>
            </div>

            {char.bio && (
              <p className="text-xs text-slate-400 border-t border-slate-800/80 pt-2">{char.bio}</p>
            )}
          </GlassCard>
        ))}
      </div>
    </div>
  );
};
