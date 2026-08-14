import os
import sys
import asyncio
import subprocess
import imageio_ffmpeg

# Ensure project root is in python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.tts.tts_service import tts_service


def rebuild_speech():
    media_dir = os.path.join(backend_dir, "media_output")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    print("Re-synthesizing natural human character speech dialogues using services/tts...")

    sample_dialogues = [
        {"speaker": "KAELEN VANCE", "line": "The grid frequency is fluctuating... someone breached the core."},
        {"speaker": "NOVA THORNE", "line": "Tracing origin IP... it leads directly to Director Vane's server."},
        {"speaker": "DIRECTOR VANE", "line": "You're too late, detective. The Obsidian Protocol is already live."},
        {"speaker": "KAELEN VANCE", "line": "Executing counter-override. Hold the extraction line!"}
    ]

    # Process scenes audio using upgraded TTS pipeline
    for sc_idx, scene_id in enumerate(["scene1", "scene2"], 1):
        res = asyncio.run(tts_service.process_scene_dialogues(
            episode_id="001",
            scene_id=f"0{sc_idx}",
            dialogue_list=sample_dialogues,
            base_output_dir=os.path.join(project_root, "audio")
        ))
        comp_path = res["composite_audio_path"]
        out_wav = os.path.join(media_dir, f"audio_{scene_id}.wav")
        if os.path.exists(comp_path):
            import shutil
            shutil.copyfile(comp_path, out_wav)
            print(f"Generated natural voice speech track for audio_{scene_id}.wav ({res['total_duration']:.2f}s)")

    # Multiplex audio into video_scene1.mp4, video_scene2.mp4, and final_episode_*.mp4
    for vf in os.listdir(media_dir):
        if vf.endswith(".mp4") and not vf.startswith("temp_"):
            v_path = os.path.join(media_dir, vf)
            a_path = os.path.join(media_dir, "audio_scene1.wav")

            temp_output = os.path.join(media_dir, f"sp_{vf}")
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
                temp_output
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(temp_output):
                os.replace(temp_output, v_path)
                print(f"Multiplexed real character voice speech into {vf}")

    # Copy to root media_output if exists
    root_media_dir = os.path.join(project_root, "media_output")
    if os.path.exists(root_media_dir):
        for f in os.listdir(media_dir):
            if f.endswith(".mp4") or f.endswith(".wav"):
                src = os.path.join(media_dir, f)
                dst = os.path.join(root_media_dir, f)
                import shutil
                shutil.copyfile(src, dst)
        print("Synchronized root media_output directory with natural human voice speech.")


if __name__ == "__main__":
    rebuild_speech()
