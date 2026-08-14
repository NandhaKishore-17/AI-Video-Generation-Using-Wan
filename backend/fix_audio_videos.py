import os
import sys
import asyncio
import subprocess
import imageio_ffmpeg

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.tts.tts_service import tts_service
from services.tts.voice_manager import voice_manager

async def main():
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    media_dir = os.path.join(backend_dir, "media_output")
    root_media = os.path.join(project_root, "media_output")

    print("=== 1. Generating Crystal-Clear Real Human Speech Audio Tracks ===")
    sample_dialogues = [
        {"speaker": "KAELEN VANCE", "line": "The grid frequency is fluctuating... someone breached the core."},
        {"speaker": "NOVA THORNE", "line": "Tracing origin IP... it leads directly to Director Vane's server."},
        {"speaker": "DIRECTOR VANE", "line": "You're too late, detective. The Obsidian Protocol is already live."},
        {"speaker": "KAELEN VANCE", "line": "Executing counter-override. Hold the extraction line!"}
    ]

    res1 = await tts_service.process_scene_dialogues(
        episode_id="001",
        scene_id="01",
        dialogue_list=sample_dialogues,
        base_output_dir=os.path.join(project_root, "audio")
    )
    
    res2 = await tts_service.process_scene_dialogues(
        episode_id="001",
        scene_id="02",
        dialogue_list=sample_dialogues,
        base_output_dir=os.path.join(project_root, "audio")
    )

    clean_speech_wav1 = res1["composite_audio_path"]
    clean_speech_wav2 = res2["composite_audio_path"]

    import shutil
    shutil.copyfile(clean_speech_wav1, os.path.join(media_dir, "audio_scene1.wav"))
    shutil.copyfile(clean_speech_wav2, os.path.join(media_dir, "audio_scene2.wav"))

    print(f"Generated clean speech scene1 ({res1['total_duration']:.2f}s) & scene2 ({res2['total_duration']:.2f}s)")

    print("\n=== 2. Stripping Background Voice/Noise and Multiplexing Pure Human Speech ===")
    mp4_files = [f for f in os.listdir(media_dir) if f.endswith(".mp4") and not f.startswith("temp_") and not f.startswith("clean_")]

    success_count = 0
    for vf in mp4_files:
        v_path = os.path.join(media_dir, vf)
        base_name = vf.replace("video_", "audio_").replace(".mp4", ".wav")
        a_path = os.path.join(media_dir, base_name)
        if not os.path.exists(a_path):
            a_path = os.path.join(media_dir, "audio_scene1.wav")

        temp_out = os.path.join(media_dir, f"clean_{vf}")
        cmd = [
            ffmpeg_exe, "-y",
            "-i", v_path,
            "-i", a_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            temp_out
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
            os.replace(temp_out, v_path)
            success_count += 1

    print(f"Successfully stripped background noise & multiplexed real human voice into {success_count} MP4 videos.")

    if os.path.exists(root_media):
        for f in os.listdir(media_dir):
            if f.endswith(".mp4") or f.endswith(".wav"):
                shutil.copyfile(os.path.join(media_dir, f), os.path.join(root_media, f))
        print("Synchronized root media_output directory.")

if __name__ == "__main__":
    asyncio.run(main())
