import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("upscale_engine")


def upscale_video(
    input_path: str,
    output_path: Optional[str] = None,
    scale: float = 2.0,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> str:
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python-headless is required for upscaling.")

    import imageio.v2 as imageio
    import numpy as np

    input_path = str(Path(input_path).resolve())
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if output_path is None:
        stem = Path(input_path).stem
        parent = Path(input_path).parent
        output_path = str(parent / f"{stem}_upscaled.mp4")
    output_path = str(Path(output_path).resolve())
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info("Upscale pass: %s  ->  %s", input_path, output_path)

    reader = imageio.get_reader(input_path)
    meta = reader.get_meta_data()
    source_fps = meta.get("fps", 12.0)

    first_frame = np.array(reader.get_data(0))
    src_h, src_w = first_frame.shape[:2]

    if target_width and target_height:
        out_w, out_h = int(target_width), int(target_height)
    elif target_width:
        ratio = target_width / src_w
        out_w, out_h = int(target_width), int(src_h * ratio)
    elif target_height:
        ratio = target_height / src_h
        out_w, out_h = int(src_w * ratio), int(target_height)
    else:
        out_w = int(src_w * scale)
        out_h = int(src_h * scale)

    out_w = out_w if out_w % 2 == 0 else out_w + 1
    out_h = out_h if out_h % 2 == 0 else out_h + 1

    logger.info("Upscaling %dx%d -> %dx%d  (%.1fx)  fps=%.1f", src_w, src_h, out_w, out_h, out_w / src_w, source_fps)

    # Use a temporary output file if in-place overwrite to prevent file truncation
    is_inplace = Path(input_path).resolve() == Path(output_path).resolve()
    actual_out_path = str(Path(output_path).with_suffix(".tmp.mp4")) if is_inplace else output_path

    writer = imageio.get_writer(actual_out_path, fps=source_fps, codec="libx264", quality=9, output_params=["-crf", "18"])

    reader.set_image_index(0)
    frame_count = 0
    for raw_frame in reader:
        frame_bgr = cv2.cvtColor(np.array(raw_frame), cv2.COLOR_RGB2BGR)
        upscaled_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        upscaled_rgb = cv2.cvtColor(upscaled_bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(upscaled_rgb)
        frame_count += 1

    writer.close()
    reader.close()

    if is_inplace:
        import time
        time.sleep(0.1)
        os.replace(actual_out_path, output_path)

    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Upscale complete: %d frames -> %s  (%.1f MB)", frame_count, output_path, output_size_mb)
    return output_path
