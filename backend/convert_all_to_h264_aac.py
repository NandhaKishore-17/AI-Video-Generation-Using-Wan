import os
import subprocess
import imageio_ffmpeg

def convert_all():
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    media_dir = os.path.join(backend_dir, "media_output")
    root_media_dir = os.path.join(os.path.dirname(backend_dir), "media_output")

    print(f"Converting all videos to standard H.264 + AAC 44.1kHz stereo/mono...")

    for target_dir in [media_dir, root_media_dir]:
        if not os.path.exists(target_dir):
            continue

        for f in os.listdir(target_dir):
            if f.endswith(".mp4") and not f.startswith("conv_"):
                v_path = os.path.join(target_dir, f)
                temp_path = os.path.join(target_dir, f"conv_{f}")
                
                # Check matching audio file if available
                base_id = f.replace("video_", "").replace("final_episode_", "").replace(".mp4", "")
                audio_file = os.path.join(target_dir, f"audio_{base_id}.wav")
                if not os.path.exists(audio_file):
                    audio_file = os.path.join(target_dir, "audio_scene1.wav")

                if os.path.exists(audio_file):
                    cmd = [
                        exe, "-y",
                        "-i", v_path,
                        "-i", audio_file,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac",
                        "-b:a", "192k",
                        "-ar", "44100",
                        "-shortest",
                        temp_path
                    ]
                else:
                    cmd = [
                        exe, "-y",
                        "-i", v_path,
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        temp_path
                    ]

                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and os.path.exists(temp_path):
                    os.replace(temp_path, v_path)
                    print(f"Successfully converted {f} to H.264 + AAC")
                else:
                    print(f"Failed to convert {f}: {res.stderr[:150]}")

if __name__ == "__main__":
    convert_all()
