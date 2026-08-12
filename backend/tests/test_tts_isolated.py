import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.tts.tts_service import TTSService

async def test_tts():
    tts = TTSService()
    
    # Test 1: Aethon Darkhaven using en-US-EricNeural
    print("Testing Voice 1: en-US-EricNeural")
    res1 = await tts.generate_single_dialogue(
        character_name="Aethon Darkhaven",
        dialogue_text="I am Aethon Darkhaven. I command the shadows.",
        output_filepath="aethon_test.wav",
        scene_context="Testing",
        force_voice_id="en-US-EricNeural",
        character_id="aethon_id_123"
    )
    print("Result 1:", res1)
    
    # Test 2: Lysander Moonwhisper using en-US-AriaNeural
    print("\nTesting Voice 2: en-US-AriaNeural")
    res2 = await tts.generate_single_dialogue(
        character_name="Lysander Moonwhisper",
        dialogue_text="I am Lysander Moonwhisper. The stars guide my path.",
        output_filepath="lysander_test.wav",
        scene_context="Testing",
        force_voice_id="en-US-AriaNeural",
        character_id="lysander_id_456"
    )
    print("Result 2:", res2)

    if os.path.exists("aethon_test.wav"):
        print("aethon_test.wav exists, size:", os.path.getsize("aethon_test.wav"))
    if os.path.exists("lysander_test.wav"):
        print("lysander_test.wav exists, size:", os.path.getsize("lysander_test.wav"))

if __name__ == "__main__":
    asyncio.run(test_tts())
