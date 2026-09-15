"""
compositor.py - 16:9 Broadcast Finishing & Compositing Engine.
Applies official Kaalapadhivugal branding, bilingual typography, ducked BGM, and master audio.
"""

import os
import subprocess
from pathlib import Path
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FONT_TAMIL = "D:/mvid/daily_engine/assets/fonts/NotoSansTamil-Bold.ttf"
FONT_EN = "C:/Windows/Fonts/segoeui.ttf"


def style_scene_clip(
    video_path: Path,
    badge_text: str,
    tamil_sub: str,
    english_sub: str,
    output_path: Path,
    target_width: int = 1280,
    target_height: int = 720,
    fps: int = 25,
) -> Path:
    """
    Ensures 1280x720 16:9 scaling and burns in channel watermark,
    top badge, and bilingual subtitles.
    """
    video_path = Path(video_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Escape fonts for ffmpeg filtergraph on Windows
    f_tamil = FONT_TAMIL.replace(":", r"\:")
    f_en = FONT_EN.replace(":", r"\:")

    # Clean text: replace single quotes to avoid breaking ffmpeg syntax
    b_txt = badge_text.replace("'", "").replace(":", "-")
    t_txt = tamil_sub.replace("'", "")
    e_txt = english_sub.replace("'", "")

    vf = (
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
        f"crop={target_width}:{target_height},"
        f"fps={fps},"
        # Top Badge: Gold text on semi-transparent dark box
        f"drawtext=fontfile='{f_tamil}':text='{b_txt}':fontsize=20:fontcolor=0xFFD700:x=(w-text_w)/2:y=35:box=1:boxcolor=black@0.75:boxborderw=10,"
        # Watermark: top-left channel handle
        f"drawtext=fontfile='{f_en}':text='@kaalapadhivugal  |  Kaalapadhivugal':fontsize=16:fontcolor=white@0.65:x=40:y=38,"
        # Bottom subtitle gradient box
        f"drawbox=y=ih-125:color=black@0.70:width=iw:height=125:t=fill,"
        f"drawbox=y=ih-125:color=0xD4AF37:width=iw:height=2:t=fill,"
        # Tamil subtitle (white with yellow tint)
        f"drawtext=fontfile='{f_tamil}':text='{t_txt}':fontsize=21:fontcolor=white:x=(w-text_w)/2:y=h-92,"
        # English translation subtitle (gold)
        f"drawtext=fontfile='{f_en}':text='{e_txt}':fontsize=16:fontcolor=0xFFD700:x=(w-text_w)/2:y=h-48"
    )

    cmd = [
        FFMPEG, "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_path


def concatenate_scenes(scene_video_paths: list[Path], output_video: Path) -> Path:
    """Concatenate multiple scene videos seamlessly using FFmpeg concat demuxer."""
    output_video = Path(output_video).resolve()
    concat_list_file = output_video.parent / "concat_list.txt"

    with open(concat_list_file, "w", encoding="utf-8") as f:
        for p in scene_video_paths:
            f.write(f"file '{str(p).replace('\\', '/')}'\n")

    cmd = [
        FFMPEG, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        str(output_video),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_video


def build_master_audio(
    scene_audio_paths: list[Path],
    bgm_path: Path,
    output_audio: Path,
    bgm_volume_db: float = -18.0,
    voice_volume_db: float = 2.0,
) -> Path:
    """
    Concatenate scene dialogue WAVs and mix with ducked background music.
    Normalizes final audio to -14 LUFS.
    """
    output_audio = Path(output_audio).resolve()
    concat_list_file = output_audio.parent / "audio_concat_list.txt"

    with open(concat_list_file, "w", encoding="utf-8") as f:
        for p in scene_audio_paths:
            f.write(f"file '{str(p).replace('\\', '/')}'\n")

    combined_voice_wav = output_audio.parent / "combined_voice.wav"
    cmd_concat = [
        FFMPEG, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        str(combined_voice_wav),
    ]
    subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Mix combined voice with ducked BGM
    filter_complex = (
        f"[0:a]volume={voice_volume_db}dB[voice];"
        f"[1:a]volume={bgm_volume_db}dB[bgm];"
        f"[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2,"
        f"loudnorm=I=-14:LRA=7:TP=-1.5[outa]"
    )

    cmd_mix = [
        FFMPEG, "-y",
        "-i", str(combined_voice_wav),
        "-stream_loop", "10",
        "-i", str(bgm_path),
        "-filter_complex", filter_complex,
        "-map", "[outa]",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_audio),
    ]
    subprocess.run(cmd_mix, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output_audio


def assemble_final_episode(
    video_concat_path: Path,
    master_audio_path: Path,
    final_output_mp4: Path,
) -> Path:
    """Mux video track and master audio track together."""
    final_output_mp4 = Path(final_output_mp4).resolve()
    final_output_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG, "-y",
        "-i", str(video_concat_path),
        "-i", str(master_audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(final_output_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return final_output_mp4
