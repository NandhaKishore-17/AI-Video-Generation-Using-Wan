"""
sadtalker_runner.py - Full-Frame SadTalker Lip-Sync Runner for Host Yaazhini.
Generates animated talking-head video and seamlessly pastes it back onto the full 16:9 scene.
"""

import os
import sys
import glob
import subprocess
from pathlib import Path
import cv2
import numpy as np

SADTALKER_DIR = Path(r"D:\mvid\libs\sadtalker")
CHECKPOINT_DIR = SADTALKER_DIR / "checkpoints"


def generate_talking_host_video(
    source_image: Path,
    driven_audio: Path,
    output_mp4: Path,
    face_size: int = 256,
) -> Path:
    """
    Run SadTalker on a host portrait with driven audio, then seamlessly paste the
    animated face back onto the original 16:9 widescreen keyframe.
    """
    source_image = Path(source_image).resolve()
    driven_audio = Path(driven_audio).resolve()
    output_mp4 = Path(output_mp4).resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    temp_res_dir = output_mp4.parent / f"sadtalker_temp_{output_mp4.stem}"
    temp_res_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SADTALKER] Generating talking head for {source_image.name}...")

    # Step 1: Run SadTalker CLI
    cmd = [
        sys.executable,
        "-u",
        str(SADTALKER_DIR / "inference.py"),
        "--driven_audio", str(driven_audio),
        "--source_image", str(source_image),
        "--result_dir", str(temp_res_dir),
        "--size", str(face_size),
        "--preprocess", "crop",
        "--still",
        "--checkpoint_dir", str(CHECKPOINT_DIR),
    ]

    subprocess.run(cmd, cwd=str(SADTALKER_DIR), check=True)

    # Find the generated video in temp_res_dir
    mp4_files = sorted(glob.glob(str(temp_res_dir / "*.mp4")))
    if not mp4_files:
        raise RuntimeError(f"SadTalker did not generate an MP4 in {temp_res_dir}")

    face_video = mp4_files[-1]
    print(f"[SADTALKER] Face animation rendered: {face_video}")

    # Step 2: Paste animated face back onto full original keyframe (1280x720)
    print("[SADTALKER] Pasting animated face onto full 16:9 keyframe...")
    if str(SADTALKER_DIR) not in sys.path:
        sys.path.insert(0, str(SADTALKER_DIR))

    from src.utils.paste_pic import paste_pic
    from src.utils.croper import Preprocesser

    preprocessor = Preprocesser("cuda")
    img_bgr = cv2.imread(str(source_image))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    frames, crop, quad = preprocessor.crop([img_rgb], still=False, xsize=512)
    clx, cly, crx, cry = crop
    lx, ly, rx, ry = [int(v) for v in quad]
    oy1, oy2, ox1, ox2 = cly + ly, cly + ry, clx + lx, clx + rx
    crop_info = ((ox2 - ox1, oy2 - oy1), crop, quad)

    paste_pic(
        face_video,
        str(source_image),
        crop_info,
        str(driven_audio),
        str(output_mp4),
        extended_crop=False,
    )

    print(f"[SADTALKER] Full-frame composite complete: {output_mp4}")
    return output_mp4


if __name__ == "__main__":
    test_img = Path(r"D:\mvid\daily_engine\assets\hosts\yaazhini_presenter.jpg")
    test_wav = Path(r"D:\mvid\daily_engine\scratch\test_voice.wav")
    test_out = Path(r"D:\mvid\daily_engine\scratch\test_talking_host.mp4")
    if test_img.exists() and test_wav.exists():
        generate_talking_host_video(test_img, test_wav, test_out)
