import asyncio
import sys
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.engines.voice_engine import voice_engine
from app.engines.render_engine import render_engine
from app.engines.wan_pipeline import wan_pipeline
from app.core.config import settings
from PIL import Image

async def test_e2e():
    print("\nStarting E2E Voice and Video pipeline test...")
    
    # Fake Scene data
    scene_id = "test_scene_001"
    
    # 1. Voice Engine
    dialogue_list = [
        {
            "speaker": "Aethon Darkhaven",
            "line": "Witness the power of the dark sun.",
            "voice_id": "en-US-EricNeural",
            "character_id": "c1"
        },
        {
            "speaker": "Lysander Moonwhisper",
            "line": "Your darkness will never extinguish the stars.",
            "voice_id": "en-US-AriaNeural",
            "character_id": "c2"
        }
    ]
    print("\n--- Generating Audio ---")
    audio_url, srt_content, scene_dur = await voice_engine.generate_scene_audio_and_srt(
        scene_id=scene_id, dialogue_list=dialogue_list
    )
    print("Audio URL:", audio_url)
    print("Scene Duration:", scene_dur)
    
    # 2. Wan Pipeline Video
    print("\n--- Generating Video (Fallback Mock) ---")
    media_output_dir = os.path.join(settings.MEDIA_OUTPUT_DIR)
    os.makedirs(media_output_dir, exist_ok=True)
    temp_img_path = os.path.join(media_output_dir, f"{scene_id}.png")
    temp_vid_path = os.path.join(media_output_dir, f"{scene_id}_vid.mp4")
    
    Image.new("RGB", (640, 360), color=(255, 0, 0)).save(temp_img_path)
    
    # using fallback of wan_pipeline directly because model might not load in test
    vid_url = wan_pipeline._fallback_generate_video(
        image_path=temp_img_path,
        output_path=temp_vid_path,
        width=640,
        height=360,
        fps=24,
        num_frames=int(scene_dur * 24) if scene_dur > 0 else 48
    )
    print("Generated Video Path:", vid_url)
    
    # 3. Render Engine Compositing
    print("\n--- Compositing Final MP4 ---")
    scene_assets = [
        {
            "scene_id": scene_id,
            "video_url": vid_url,
            "audio_url": audio_url,
            "subtitle_srt": srt_content,
            "duration_seconds": scene_dur if scene_dur > 0 else 2.0
        }
    ]
    
    # This should trigger FFmpeg composite, then fallback to muxing if needed, and finally trigger the validation logs
    episode_id = "test_episode_999"
    try:
        final_video = await render_engine.render_episode_mp4(episode_id, scene_assets)
        print("Final Composite Rendered:", final_video)
    except Exception as e:
        print("Final Composite Failed:", str(e))
    
    print("E2E Test Completed.")

if __name__ == "__main__":
    asyncio.run(test_e2e())
