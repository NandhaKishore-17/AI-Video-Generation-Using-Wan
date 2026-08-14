import os
import sys
import asyncio
import wave

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.tts.voice_manager import voice_manager, PRESET_CHARACTER_ARCHETYPES
from services.tts.emotion_mapper import emotion_mapper
from services.tts.audio_utils import audio_utils
from services.tts.tts_service import tts_service


async def run_tests():
    print("--- 1. Testing Voice Manager ---")
    kaelen_v = voice_manager.get_or_assign_voice("Kaelen Vance")
    nova_v = voice_manager.get_or_assign_voice("Nova Thorne")
    vane_v = voice_manager.get_or_assign_voice("Director Vane")

    print(f"Kaelen Vance -> {kaelen_v['name']} ({kaelen_v['archetype']})")
    print(f"Nova Thorne  -> {nova_v['name']} ({nova_v['archetype']})")
    print(f"Director Vane -> {vane_v['name']} ({vane_v['archetype']})")

    assert kaelen_v["archetype"] == "deep_male", f"Expected deep_male, got {kaelen_v['archetype']}"
    assert nova_v["archetype"] == "calm_female", f"Expected calm_female, got {nova_v['archetype']}"
    assert vane_v["archetype"] == "deep_villain_male", f"Expected deep_villain_male, got {vane_v['archetype']}"

    # Test new character voice assignment
    new_char_v = voice_manager.get_or_assign_voice("Agent Sterling")
    print(f"New Character 'Agent Sterling' -> {new_char_v['name']} ({new_char_v['archetype']})")

    # Verify voice consistency (same character returns exact same voice)
    kaelen_v_again = voice_manager.get_or_assign_voice("Kaelen Vance")
    assert kaelen_v["voice_id"] == kaelen_v_again["voice_id"], "Voice consistency check failed!"
    print("Voice Consistency Check: PASSED")

    print("\n--- 2. Testing Emotion Mapper ---")
    test_lines = [
        ("I'm not leaving you!", "Urgent extraction scene"),
        ("We've been compromised.", "High security room"),
        ("Run!", "Chased by drones"),
        ("(whispering) Shh, don't make a sound.", "Stealth mode"),
        ("We won the battle!", "Victory party")
    ]

    for text, ctx in test_lines:
        emo, prosody = emotion_mapper.infer_emotion(text, scene_context=ctx)
        print(f"Line: '{text}' -> Emotion: {emo} (rate: {prosody['rate']}, pitch: {prosody['pitch']})")

    print("\n--- 3. Testing Scene Dialogue Generation ---")
    sample_scene_dialogues = [
        {"speaker": "Kaelen Vance", "line": "I'm not leaving you!"},
        {"speaker": "Nova Thorne", "line": "We've been compromised."},
        {"speaker": "Director Vane", "line": "Run! The network is collapsing."}
    ]

    res = await tts_service.process_scene_dialogues(
        episode_id="001",
        scene_id="01",
        dialogue_list=sample_scene_dialogues,
        base_output_dir=os.path.join(project_root, "audio")
    )

    print(f"\nEpisode ID: {res['episode_id']}")
    print(f"Scene ID: {res['scene_id']}")
    print(f"Composite Audio Path: {res['composite_audio_path']}")
    print(f"Total Duration: {res['total_duration']:.2f}s")
    print("\nDialogue Audio Files Generated:")
    for df in res["dialogue_files"]:
        print(f" - {df['speaker']}: {df['file_path']} (Duration: {df['duration']:.2f}s, Emotion: {df['emotion']})")
        assert os.path.exists(df['file_path']), f"File not found: {df['file_path']}"

    # Verify Audio Quality (sample rate >= 24000 Hz)
    with wave.open(res['composite_audio_path'], 'rb') as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth() * 8
        print(f"\nAudio File Specs: {sr} Hz, {sw}-bit, {ch} channel(s)")
        assert sr >= 24000, f"Sample rate too low: {sr} Hz"

    print("\nSRT Subtitles Sample:\n" + res["srt_content"])
    print("\n[SUCCESS] ALL TTS PIPELINE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    asyncio.run(run_tests())
