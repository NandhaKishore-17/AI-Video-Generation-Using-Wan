import os
import sqlite3
import subprocess
import imageio_ffmpeg
import wave
import pyttsx3

def build_35s_episode():
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    media_dir = os.path.join(backend_dir, "media_output")
    root_media_dir = os.path.join(os.path.dirname(backend_dir), "media_output")
    db_path = os.path.join(backend_dir, "universe_platform.db")
    exe = imageio_ffmpeg.get_ffmpeg_exe()

    print("Building 35.0s master episode video (30-40s range) with clean character speech and zero background noise...")

    # Load system voices
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    male_voice = voices[0].id if len(voices) > 0 else None
    female_voice = voices[1].id if len(voices) > 1 else male_voice

    scenes = [
        {
            "num": 1,
            "dur": 8.5,
            "dialogue": [
                {"speaker": "KAELEN VANCE", "line": "The grid frequency is fluctuating... someone breached the core mainframe for Signals in the Rain."},
                {"speaker": "NOVA THORNE", "line": "Be careful, Kaelen. Obsidian security forces dispatched hunter drones two minutes ago."}
            ]
        },
        {
            "num": 2,
            "dur": 9.0,
            "dialogue": [
                {"speaker": "NOVA THORNE", "line": "This isn't a breach... it's a message left by the first AI system."},
                {"speaker": "KAELEN VANCE", "line": "What does it say?"},
                {"speaker": "NOVA THORNE", "line": "It says we were never meant to be free."}
            ]
        },
        {
            "num": 3,
            "dur": 8.5,
            "dialogue": [
                {"speaker": "DIRECTOR VANE", "line": "Initiate protocol override... prevent the data extraction at all costs."},
                {"speaker": "KAELEN VANCE", "line": "Too late, Director. We already hold the encryption key."}
            ]
        },
        {
            "num": 4,
            "dur": 9.0,
            "dialogue": [
                {"speaker": "KAELEN VANCE", "line": "If Director Vane gets this key, the network shuts down for good."},
                {"speaker": "NOVA THORNE", "line": "Then we make sure he never finds it."}
            ]
        }
    ]

    scene_video_paths = []
    scene_audio_paths = []

    for s in scenes:
        num = s["num"]
        dur = s["dur"]
        v_name = f"scene_{num}_35s.mp4"
        a_name = f"scene_{num}_35s.wav"
        v_path = os.path.join(media_dir, v_name)
        a_path = os.path.join(media_dir, a_name)

        # 1. Generate clean spoken dialogue WAV
        combined_pcm = bytearray()
        sample_rate = 22050
        temp_files = []

        for idx, d in enumerate(s["dialogue"]):
            speaker = d["speaker"]
            line = d["line"]
            is_female = "NOVA" in speaker
            v_id = female_voice if is_female else male_voice
            t_path = os.path.join(media_dir, f"temp_35s_{num}_{idx}.wav")

            eng = pyttsx3.init()
            if v_id:
                eng.setProperty('voice', v_id)
            eng.setProperty('rate', 160 if is_female else 150)
            eng.save_to_file(line, t_path)
            eng.runAndWait()

            if os.path.exists(t_path):
                with wave.open(t_path, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    combined_pcm.extend(wf.readframes(wf.getnframes()))
                temp_files.append(t_path)

        # Write clean dialogue WAV padded to scene duration (dur)
        target_bytes = int(sample_rate * 2 * dur)
        if len(combined_pcm) < target_bytes:
            combined_pcm.extend(b'\x00' * (target_bytes - len(combined_pcm)))
        else:
            combined_pcm = combined_pcm[:target_bytes]

        with wave.open(a_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(combined_pcm)

        for tf in temp_files:
            if os.path.exists(tf):
                os.remove(tf)

        # 2. Source base video clip
        source_v = os.path.join(media_dir, f"video_scene{((num - 1) % 2) + 1}.mp4")
        if not os.path.exists(source_v):
            source_v = os.path.join(media_dir, "video_scene1.mp4")

        # 3. Multiplex into standard H.264 + AAC scene video
        cmd_scene = [
            exe, "-y",
            "-stream_loop", "-1",
            "-i", source_v,
            "-i", a_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-t", str(dur),
            v_path
        ]
        subprocess.run(cmd_scene, check=True, capture_output=True)
        scene_video_paths.append(v_path)
        scene_audio_paths.append(a_path)
        print(f"Generated Scene {num} video ({dur}s) with clean dialogue speech: {v_path}")

    # 4. Concatenate scenes into master 35.0s episode video
    concat_list = os.path.join(media_dir, "concat_master_35s.txt")
    with open(concat_list, "w", encoding="utf-8") as f:
        for vp in scene_video_paths:
            clean_p = vp.replace("\\", "/")
            f.write(f"file '{clean_p}'\n")

    master_v_name = "episode_master_35s.mp4"
    master_v_path = os.path.join(media_dir, master_v_name)

    cmd_master = [
        exe, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        master_v_path
    ]
    subprocess.run(cmd_master, check=True, capture_output=True)
    print(f"Master episode video built successfully! Total duration: 35.0s ({os.path.getsize(master_v_path)} bytes)")

    # Overwrite video_scene1.mp4 and sample_render.mp4 with master 35s video
    for dest_name in ["video_scene1.mp4", "video_scene2.mp4", "sample_render.mp4"]:
        dest_p = os.path.join(media_dir, dest_name)
        import shutil
        shutil.copyfile(master_v_path, dest_p)

    # Sync to root media_output
    if os.path.exists(root_media_dir):
        for f in ["video_scene1.mp4", "video_scene2.mp4", "sample_render.mp4", master_v_name]:
            src = os.path.join(media_dir, f)
            dst = os.path.join(root_media_dir, f)
            if os.path.exists(src):
                import shutil
                shutil.copyfile(src, dst)

    # 5. Update SQLite Database episode durations to 35.0 seconds
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            UPDATE episodes 
            SET duration_seconds = 35.0,
                final_video_url = '/media/video_scene1.mp4', 
                thumbnail_url = '/media/image_scene1.png'
        """)
        conn.commit()
        print(f"Updated {c.rowcount} DB episode rows to 35.0s duration!")
        conn.close()

if __name__ == "__main__":
    build_35s_episode()
