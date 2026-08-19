import os
import gc
import subprocess
import torch
from dotenv import load_dotenv
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

load_dotenv("backend/.env")

MODEL_PATH = os.environ["WAN_MODEL_PATH"]

OUTPUT_DIR = "/root/itsme/media_output/wan_30s"
FINAL_VIDEO = "/root/itsme/media_output/wan_30_seconds.mp4"

WIDTH = 480
HEIGHT = 256
FPS = 16
FRAMES = 49
STEPS = 20

# ============================================================
# YOUR 30-SECOND SCRIPT
# ============================================================

scenes = [
    """
    A cinematic futuristic cyberpunk city at night.
    Neon signs illuminate wet streets as the camera slowly
    moves forward through the crowded futuristic city.
    Atmospheric rain, realistic reflections, cinematic lighting.
    """,

    """
    The same futuristic city continues.
    A mysterious person wearing a dark futuristic coat walks
    slowly through the neon streets while rain falls around them.
    Smooth cinematic camera movement, realistic lighting.
    """,

    """
    The person enters a narrow neon-lit alley.
    Holographic advertisements glow on the walls while rain
    falls onto the pavement. The camera follows behind the person.
    Cinematic atmosphere, realistic movement.
    """,

    """
    The person reaches a futuristic underground train station.
    Neon lights reflect across the polished floor as a futuristic
    train arrives. Slow cinematic camera movement, realistic details.
    """,

    """
    The person enters the futuristic train.
    Through the train windows the enormous cyberpunk city passes by,
    filled with skyscrapers, neon advertisements and flying vehicles.
    Smooth realistic motion, cinematic lighting.
    """,

    """
    The train arrives at a massive futuristic tower.
    The person exits and looks upward toward the enormous glowing
    skyscrapers. The camera slowly tilts upward.
    Epic cinematic composition, realistic lighting.
    """,

    """
    The person walks across a futuristic rooftop.
    Hundreds of neon skyscrapers surround them and flying vehicles
    move between the buildings. Strong cinematic atmosphere.
    """,

    """
    A massive futuristic storm approaches the cyberpunk city.
    Lightning illuminates the skyscrapers while rain becomes heavier.
    The person continues walking across the rooftop.
    Dramatic cinematic lighting.
    """,

    """
    The storm begins to clear.
    The futuristic city glows beneath the remaining clouds while
    the person stands at the edge of the rooftop looking over the city.
    Slow camera movement, realistic atmosphere.
    """,

    """
    Dawn begins over the futuristic cyberpunk city.
    The neon lights slowly fade as sunlight appears between the
    skyscrapers. The person walks toward the horizon.
    Epic cinematic ending, highly detailed realistic environment.
    """
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("WAN 2.2 — 30 SECOND VIDEO GENERATOR")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:",
      round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
      "GB")

print("\nLoading Wan 2.2...")

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

pipe.enable_model_cpu_offload()

print("Wan loaded with CPU offloading.")

# Starting conditioning image
current_image = Image.new(
    "RGB",
    (WIDTH, HEIGHT),
    (20, 20, 30)
)

scene_files = []

for i, prompt in enumerate(scenes, start=1):

    print("\n" + "=" * 60)
    print(f"SCENE {i}/10")
    print("=" * 60)

    output = pipe(
        image=current_image,
        prompt=prompt.strip(),
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=5.0,
    )

    frames = output.frames[0]

    scene_path = os.path.join(
        OUTPUT_DIR,
        f"scene_{i:02d}.mp4"
    )

    export_to_video(
        frames,
        scene_path,
        fps=FPS
    )

    scene_files.append(scene_path)

    print(f"Scene {i} saved:")
    print(scene_path)

    # Use final frame as conditioning image for next scene.
    import numpy as np

    last_frame = frames[-1]

    # Wan may return float32 frames in [0, 1].
    # Convert them to uint8 RGB for Pillow.
    if last_frame.dtype != np.uint8:
        if last_frame.max() <= 1.0:
            last_frame = last_frame * 255.0

        last_frame = np.clip(last_frame, 0, 255).astype(np.uint8)

    current_image = Image.fromarray(last_frame, mode="RGB").convert("RGB")

    del output
    del frames

    gc.collect()
    torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("ALL 10 SCENES GENERATED")
print("=" * 60)

# Create ffmpeg concat file
concat_file = os.path.join(
    OUTPUT_DIR,
    "concat.txt"
)

with open(concat_file, "w") as f:
    for path in scene_files:
        f.write(f"file '{path}'\n")

print("\nConcatenating scenes...")

subprocess.run([
    "ffmpeg",
    "-y",
    "-f", "concat",
    "-safe", "0",
    "-i", concat_file,
    "-c", "copy",
    FINAL_VIDEO
], check=True)

print("\n" + "=" * 60)
print("SUCCESS!")
print("=" * 60)
print("FINAL VIDEO:")
print(FINAL_VIDEO)

