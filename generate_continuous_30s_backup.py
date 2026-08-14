import os
import gc
import subprocess

import numpy as np
import torch

from dotenv import load_dotenv
from PIL import Image

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video


# ============================================================
# CONFIG
# ============================================================

load_dotenv("backend/.env")

MODEL_PATH = os.environ["WAN_MODEL_PATH"]

OUTPUT_DIR = "/root/itsme/media_output/historical_mystery_30s"
FINAL_VIDEO = "/root/itsme/media_output/wan_historical_mystery_30s.mp4"

WIDTH = 832
HEIGHT = 480

FRAMES = 49
FPS = 16

# IMPORTANT:
# Official Wan 2.2 TI2V-5B configuration uses 50 steps.
STEPS = 50

NUM_SCENES = 10


# ============================================================
# STORY
# ============================================================

SCENES = [

    """
    A vast golden desert at sunset surrounding an enormous ancient
    sandstone temple half buried beneath the dunes. A lone archaeologist
    wearing practical dusty expedition clothing stands far from the entrance
    and slowly walks toward the monumental doorway. Ancient stone carvings,
    weathered pillars and drifting desert dust are visible.
    Natural warm sunlight, realistic atmospheric haze, physically accurate
    shadows, photorealistic textures, realistic human proportions,
    restrained natural colors, cinematic 35mm photography,
    slow smooth forward camera movement, highly detailed,
    realistic historical adventure film.
    """,

    """
    The same archaeologist continues walking toward the enormous sandstone
    temple entrance. The same clothing, same appearance and same environment
    remain consistent. The camera slowly follows from behind and slightly
    to the side. Wind gently moves the archaeologist's clothing while fine
    dust moves across the ground. Ancient carvings become more visible.
    Warm natural sunset illumination, realistic sandstone texture,
    subtle atmospheric depth, natural colors, photorealistic cinematic
    historical adventure cinematography, smooth controlled movement.
    """,

    """
    The archaeologist reaches the ancient temple entrance and slowly walks
    beneath its enormous stone pillars. The same person and clothing remain
    consistent. The camera performs a slow cinematic dolly forward into the
    dark entrance. Warm sunlight from outside gradually fades into cool
    natural shadow inside the temple. Detailed stone walls, ancient symbols,
    dust particles floating in shafts of light, realistic textures,
    restrained color palette, photorealistic historical mystery film.
    """,

    """
    Inside the ancient temple, the archaeologist carefully walks through a
    long stone corridor illuminated by narrow shafts of sunlight from cracks
    above. The same character remains consistent. Ancient inscriptions cover
    the walls. The camera moves slowly alongside the character while the
    archaeologist examines the carvings with curiosity. Realistic stone,
    realistic dust, subtle shadows, physically accurate lighting,
    natural muted colors, cinematic 35mm photography, photorealistic.
    """,

    """
    The archaeologist discovers a huge circular stone door covered in
    mysterious ancient symbols. The same person stands before it and slowly
    approaches the center of the door. The camera gradually pushes closer.
    Dust falls gently from the ceiling and small particles float through
    the light. The stone door is massive, ancient and physically realistic.
    Serious historical mystery atmosphere, realistic materials,
    natural cinematic lighting, restrained colors, photorealistic detail.
    """,

    """
    The enormous circular stone door slowly begins opening, revealing a vast
    underground chamber beyond it. The same archaeologist watches in awe.
    The camera slowly moves behind the character toward the opening.
    Ancient stone architecture fills the chamber. Soft golden light from
    an unseen source illuminates the distant interior. Dust particles drift
    naturally through the air. Realistic scale, realistic shadows,
    natural colors, cinematic historical mystery, photorealistic film look.
    """,

    """
    The archaeologist enters the enormous underground chamber and walks
    carefully across the ancient stone floor. A gigantic mysterious stone
    artifact stands in the center of the chamber. The character remains
    visually consistent. The camera slowly circles around the character and
    reveals more of the artifact. Natural volumetric light, realistic stone
    textures, subtle atmospheric dust, physically accurate shadows,
    restrained colors, cinematic composition, photorealistic historical
    adventure movie.
    """,

    """
    The archaeologist slowly approaches the gigantic ancient artifact.
    Detailed carvings and symbols cover its surface. The same character,
    clothing and environment remain consistent. The camera performs a slow
    controlled push toward the artifact while keeping the archaeologist in
    the foreground. Subtle dust movement, realistic material surfaces,
    natural warm and neutral lighting, realistic shadows,
    photorealistic cinematic mystery film, no exaggerated colors.
    """,

    """
    The archaeologist reaches the ancient artifact and carefully places one
    hand near its carved surface without touching it. The artifact begins
    producing a very subtle warm glow that illuminates the surrounding stone.
    The character reacts naturally with quiet astonishment. The camera slowly
    moves around them. Realistic human proportions, realistic face,
    realistic hands, physically plausible lighting, restrained warm colors,
    detailed ancient architecture, photorealistic cinematic cinematography.
    """,

    """
    Final cinematic reveal inside the enormous underground chamber.
    The archaeologist stands before the gigantic ancient artifact while
    the entire chamber gradually becomes illuminated by its subtle golden
    light. The character remains consistent and small relative to the
    enormous architecture. The camera slowly pulls backward, revealing the
    full scale of the chamber. Dust floats through the illuminated air.
    Monumental ancient architecture, realistic materials, natural colors,
    realistic lighting, cinematic 35mm film composition,
    photorealistic historical mystery ending.
    """
]


# ============================================================
# NEGATIVE PROMPT
# ============================================================

NEGATIVE_PROMPT = """
low quality, worst quality, very low resolution, blurry, soft image,
pixelated, compression artifacts, oversharpened, excessive saturation,
neon colors, psychedelic colors, oversaturated colors, blown highlights,
crushed blacks, unnatural contrast, color banding, flickering,
unstable exposure, changing lighting, unstable colors,
deformed human, distorted face, asymmetrical face, malformed hands,
extra fingers, missing fingers, fused fingers, extra limbs,
duplicate person, multiple heads, duplicate objects,
warped architecture, melting objects, floating objects,
unnatural movement, jitter, camera shake, rapid camera movement,
fast zoom, sudden camera movement, text, subtitles, watermark,
logo, cartoon, anime, illustration, painting, CGI appearance
"""


# ============================================================
# DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("WAN 2.2 — HIGH QUALITY CONTINUOUS 30 SECOND VIDEO")
print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB"
)

print()
print("Resolution:", WIDTH, "x", HEIGHT)
print("Frames/scene:", FRAMES)
print("FPS:", FPS)
print("Steps:", STEPS)
print("Scenes:", NUM_SCENES)

print()
print("Loading Wan 2.2...")


# ============================================================
# LOAD MODEL
# ============================================================

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

pipe.enable_model_cpu_offload()

print("Wan loaded with CPU offloading.")


# ============================================================
# INITIAL IMAGE
# ============================================================

# IMPORTANT:
# Replace this with a REAL cinematic keyframe for substantially
# better results.
#
# Put your image here:
#
# /root/itsme/media_output/historical_mystery_start.jpg
#
# If it does not exist, use a neutral fallback.

INITIAL_IMAGE = "/root/itsme/media_output/historical_mystery_start.jpg"

if os.path.exists(INITIAL_IMAGE):

    current_image = Image.open(INITIAL_IMAGE).convert("RGB")

    current_image = current_image.resize(
        (WIDTH, HEIGHT),
        Image.Resampling.LANCZOS
    )

    print()
    print("Using cinematic starting image:")
    print(INITIAL_IMAGE)

else:

    print()
    print("WARNING: Starting image not found.")
    print("Using generated dark fallback.")
    print("For best quality, provide:")
    print(INITIAL_IMAGE)

    current_image = Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (80, 65, 45)
    )


# ============================================================
# GENERATE SCENES
# ============================================================

scene_files = []

for scene_index, prompt in enumerate(SCENES, start=1):

    print()
    print("=" * 70)
    print(f"SCENE {scene_index}/{NUM_SCENES}")
    print("=" * 70)

    output = pipe(
        image=current_image,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=5.0,
        guidance_scale_2=5.0,
    )

    frames = output.frames[0]

    scene_path = os.path.join(
        OUTPUT_DIR,
        f"scene_{scene_index:02d}.mp4"
    )

    export_to_video(
        frames,
        scene_path,
        fps=FPS
    )

    scene_files.append(scene_path)

    print()
    print("Scene saved:")
    print(scene_path)

    # ========================================================
    # LAST FRAME → NEXT SCENE
    # ========================================================

    last_frame = frames[-1]

    if isinstance(last_frame, Image.Image):

        current_image = last_frame.convert("RGB")

    else:

        last_frame = np.asarray(last_frame)

        # Wan output can be float [0,1] or uint8 [0,255].
        if last_frame.dtype != np.uint8:

            last_frame = np.clip(last_frame, 0.0, 1.0)

            last_frame = (
                last_frame * 255.0
            ).round().astype(np.uint8)

        current_image = Image.fromarray(
            last_frame,
            "RGB"
        )

    current_image = current_image.resize(
        (WIDTH, HEIGHT),
        Image.Resampling.LANCZOS
    )

    print("Continuity frame prepared.")

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# CONCATENATE
# ============================================================

print()
print("=" * 70)
print("ALL SCENES GENERATED")
print("=" * 70)

concat_file = os.path.join(
    OUTPUT_DIR,
    "concat.txt"
)

with open(concat_file, "w") as f:

    for scene in scene_files:

        f.write(
            "file '" +
            scene.replace("'", "'\\''") +
            "'\n"
        )


print()
print("Concatenating...")

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        FINAL_VIDEO,
    ],
    check=True,
)


# ============================================================
# VALIDATE
# ============================================================

print()
print("=" * 70)
print("FINAL VIDEO")
print("=" * 70)

print(FINAL_VIDEO)

subprocess.run(
    [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "default=noprint_wrappers=1",
        FINAL_VIDEO,
    ],
    check=True,
)

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)
