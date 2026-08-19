import React, { useEffect, useState } from 'react';
import { GlassCard } from '../components/GlassCard';
import { api } from '../services/api';
import { Character, Universe } from '../types';
import { Users, Mic, Eye, UserPlus, Sparkles, X, CheckCircle2, Trash2, Edit2 } from 'lucide-react';

interface CharacterManagerProps {
  activeUniverseId?: string;
}

export const CharacterManager: React.FC<CharacterManagerProps> = ({ activeUniverseId }) => {
  const [characters, setCharacters] = useState<Character[]>([]);
  const [universeId, setUniverseId] = useState<string>('');
  const [showAddModal, setShowAddModal] = useState(false);
  const [availableVoices, setAvailableVoices] = useState<any[]>([]);
  const [editingCharacterId, setEditingCharacterId] = useState<string | null>(null);

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

  const handleResetForm = () => {
    setEditingCharacterId(null);
    setName('');
    setRole('Protagonist');
    setPersonality('');
    setAppearancePrompt('');
    if (availableVoices.length > 0) {
      setVoicePreset(availableVoices[0].id);
    }
    setVoicePitch(1.0);
    setVoiceSpeed(1.0);
    setBio('');
  };

  const handleStartEdit = (char: Character) => {
    setEditingCharacterId(char.id);
    setName(char.name);
    setRole(char.role);
    setPersonality(char.personality);
    setAppearancePrompt(char.appearance_prompt);
    setVoicePreset(char.voice_actor_preset);
    setVoicePitch(char.voice_pitch);
    setVoiceSpeed(char.voice_speed);
    setBio(char.bio || '');
    setShowAddModal(true);
  };

  const handleAddCharacter = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !personality || !appearancePrompt) return;
    setSaving(true);
    try {
      if (editingCharacterId) {
        const updatedChar = await api.updateCharacter(editingCharacterId, {
          name,
          role,
          personality,
          appearance_prompt: appearancePrompt,
          voice_actor_preset: voicePreset,
          voice_pitch: voicePitch,
          voice_speed: voiceSpeed,
          bio
        });
        setCharacters(characters.map(c => c.id === editingCharacterId ? updatedChar : c));
      } else {
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
      }
      setShowAddModal(false);
      handleResetForm();
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
          <div className="relative w-full max-w-xl bg-[#0E1424] border border-neutral-border/80 rounded-3xl overflow-hidden shadow-2xl p-6 space-y-6">
            <div className="flex items-center justify-between border-b border-neutral-border pb-4">
              <h2 className="text-xl font-bold text-text-primary flex items-center space-x-2">
                <UserPlus className="w-5 h-5 text-accent-primary" />
                <span>{editingCharacterId ? 'EDIT CHARACTER DETAILS' : 'ADD NEW CHARACTER DETAILS'}</span>
              </h2>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  handleResetForm();
                }}
                className="p-1.5 rounded-lg bg-neutral-soft text-text-secondary hover:text-text-primary"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleAddCharacter} className="space-y-4 text-xs font-mono">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-text-secondary mb-1">CHARACTER NAME</label>
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. Commander Vex"
                    className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
                    required
                  />
                </div>

                <div>
                  <label className="block text-text-secondary mb-1">ROLE</label>
                  <select
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                    className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
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
                <label className="block text-text-secondary mb-1">PERSONALITY TRAITS</label>
                <input
                  type="text"
                  value={personality}
                  onChange={(e) => setPersonality(e.target.value)}
                  placeholder="e.g. Tactical genius, quiet, deeply loyal"
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
                  required
                />
              </div>

              <div>
                <label className="block text-text-secondary mb-1">FLUX / SDXL VISUAL APPEARANCE PROMPT</label>
                <textarea
                  value={appearancePrompt}
                  onChange={(e) => setAppearancePrompt(e.target.value)}
                  placeholder="Detailed visual prompt anchor (e.g. Futuristic commander in chrome armor, glowing cyber eye, 8k portrait)"
                  rows={3}
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-text-secondary mb-1 flex items-center justify-between">
                    <span>REALISTIC HUMAN VOICE PRESET</span>
                    <span className="text-[10px] text-accent-primary font-bold">F5-TTS Zero-Shot / 24kHz HD</span>
                  </label>
                  <select
                    value={voicePreset}
                    onChange={(e) => setVoicePreset(e.target.value)}
                    className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
                  >
                    {availableVoices.map((v) => (
                      <option key={v.id} value={v.id}>
                        {v.gender.toUpperCase()} - {v.name} ({v.description})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-text-secondary mb-1">VOICE PITCH: {voicePitch}x</label>
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
                <label className="block text-text-secondary mb-1">BIO & BACKSTORY</label>
                <textarea
                  value={bio}
                  onChange={(e) => setBio(e.target.value)}
                  placeholder="Character origin story and background..."
                  rows={2}
                  className="w-full bg-neutral-soft border border-neutral-border rounded-xl px-3 py-2.5 text-text-primary focus:outline-none focus:border-accent-primary"
                />
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  disabled={saving}
                  className="w-full py-3 rounded-xl font-bold bg-neutral-soft text-text-primary shadow-sm flex items-center justify-center space-x-2"
                >
                  <Sparkles className="w-4 h-4" />
                  <span>
                    {saving 
                      ? (editingCharacterId ? 'SAVING CHANGES...' : 'SAVING CHARACTER...') 
                      : (editingCharacterId ? 'UPDATE CHARACTER' : 'CREATE CHARACTER')}
                  </span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-extrabold text-text-primary tracking-tight flex items-center gap-2">
            <Users className="w-6 h-6 text-text-secondary" />
            <span>CHARACTER CONSISTENCY & VOICE HUB</span>
          </h1>
          <p className="text-xs text-text-secondary mt-1">
            Maintain exact visual appearance anchors for FLUX diffusers and voice actor presets for Piper/Kokoro TTS.
          </p>
        </div>

        <button
          onClick={() => {
            handleResetForm();
            setShowAddModal(true);
          }}
          className="px-5 py-2.5 rounded-xl font-bold text-xs bg-accent-primary hover:brightness-110 text-text-primary shadow-sm flex items-center space-x-2"
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
                <h3 className="text-lg font-bold text-text-primary">{char.name}</h3>
                <span className="text-xs px-2.5 py-0.5 rounded-md bg-neutral-soft text-text-secondary border border-neutral-border font-mono">
                  {char.role}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  onClick={() => handleStartEdit(char)}
                  className="p-2 rounded-xl bg-white hover:bg-neutral-soft text-text-secondary hover:text-accent-primary border border-neutral-border transition-colors"
                  title="Edit Character"
                >
                  <Edit2 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => handleDeleteCharacter(char.id)}
                  className="p-2 rounded-xl bg-white hover:bg-neutral-soft text-text-secondary hover:text-text-secondary border border-neutral-border transition-colors"
                  title="Delete Character"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>


            <p className="text-xs text-text-secondary italic">{char.personality}</p>

            <div className="space-y-2 text-xs font-mono">
              <div className="p-3 rounded-lg bg-neutral-soft border border-neutral-border space-y-1">
                <div className="flex items-center space-x-1.5 text-accent-primary font-semibold">
                  <Eye className="w-3.5 h-3.5" />
                  <span>FLUX VISUAL ANCHOR PROMPT</span>
                </div>
                <p className="text-text-secondary text-[11px]">{char.appearance_prompt}</p>
              </div>

              <div className="p-3 rounded-lg bg-neutral-soft border border-neutral-border space-y-1">
                <div className="flex items-center space-x-1.5 text-accent-primary font-semibold">
                  <Mic className="w-3.5 h-3.5" />
                  <span>PIPER / KOKORO VOICE PRESET</span>
                </div>
                <p className="text-text-secondary text-[11px]">{char.voice_actor_preset} (Pitch: {char.voice_pitch}x)</p>
              </div>
            </div>

            {char.bio && (
              <p className="text-xs text-text-secondary border-t border-neutral-border pt-2">{char.bio}</p>
            )}
          </GlassCard>
        ))}
      </div>
    </div>
  );
};
