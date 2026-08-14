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

OUTPUT_DIR = "/root/itsme/media_output/fantasy_30s"
FINAL_VIDEO = "/root/itsme/media_output/wan_fantasy_30s.mp4"

# RTX 3090 SAFE QUALITY TARGET
WIDTH = 832
HEIGHT = 480

FRAMES = 49
FPS = 16

# Quality over speed
STEPS = 50

NUM_SCENES = 10

SEED = 12345


# ============================================================
# PERSISTENT CHARACTER
# ============================================================

CHARACTER = """
the same 28-year-old female elf mage throughout the entire film,
pale natural skin,
long silver-white hair,
emerald green eyes,
small pointed ears,
dark emerald-green leather cloak,
black leather boots,
brown leather belt,
silver crescent pendant,
wooden staff containing a glowing blue crystal,
realistic human proportions,
consistent facial identity,
consistent clothing,
consistent hairstyle
"""


# ============================================================
# COMMON CINEMATIC STYLE
# ============================================================

STYLE = """
cinematic dark fantasy feature film,
photorealistic,
extremely detailed environment,
realistic materials,
natural skin texture,
physically based lighting,
volumetric lighting,
soft atmospheric fog,
subtle depth of field,
natural motion blur,
high dynamic range,
professional cinematic color grading,
35mm cinema lens,
realistic shadows,
natural contrast,
rich but believable colors,
professional cinematography,
no cartoon appearance,
no anime appearance,
no illustration,
no plastic skin
"""


# ============================================================
# NEGATIVE PROMPT
# ============================================================

NEGATIVE = """
cartoon,
anime,
illustration,
painting,
CGI look,
plastic skin,
oversaturated colors,
neon colors,
low resolution,
blurry,
soft image,
deformed face,
extra fingers,
extra limbs,
duplicate person,
mutated hands,
bad anatomy,
distorted eyes,
cross-eyed,
warped body,
floating objects,
text,
logo,
watermark,
flicker,
camera shake,
rapid motion,
unnatural motion
"""


# ============================================================
# STORY
# ============================================================

SCENES = [

"""
A vast ancient enchanted forest at dawn.
The elf mage walks slowly along a moss-covered stone path.
Huge ancient trees surround her.
Small particles of magical light float naturally through the air.
The camera slowly follows behind her.
The movement is calm and realistic.
""",

"""
The same elf mage continues deeper into the ancient forest.
She approaches an enormous ancient stone arch covered with glowing
blue magical runes.
She raises her staff slightly and the runes begin glowing.
The camera slowly moves from behind her toward a three-quarter view.
""",

"""
The same elf mage walks through a misty forest valley.
A magnificent waterfall can be seen in the distance.
Mist rises naturally from the water.
She walks slowly beside a crystal-clear stream.
The camera makes a slow cinematic side tracking movement.
""",

"""
The same elf mage discovers enormous ancient ruins hidden among the trees.
Broken stone statues of forgotten kings surround her.
She walks carefully between the ruins.
Her glowing staff illuminates the ancient carvings.
Slow cinematic camera movement.
""",

"""
The same elf mage reaches a gigantic ruined staircase leading upward.
Ancient torches ignite one after another as she walks past them.
Warm firelight mixes naturally with the blue magical glow from her staff.
The camera slowly rises behind her.
""",

"""
The same elf mage reaches the top of the ruined fortress.
A gigantic ancient castle is visible through the fog in the distance.
Dark storm clouds move slowly above the mountains.
She looks toward the castle.
The camera slowly pushes toward her face.
""",

"""
The same elf mage enters an enormous underground chamber beneath the castle.
A gigantic circular stone seal lies on the floor.
Ancient blue symbols glow around its edge.
She approaches the center cautiously.
The camera slowly circles around her.
""",

"""
The same elf mage raises her wooden staff toward the ancient stone seal.
Blue magical energy begins flowing from the staff into the symbols.
Dust particles move naturally through the chamber.
The stone floor begins trembling subtly.
The camera slowly moves closer.
""",

"""
The same elf mage stands before the awakened ancient dragon.
The enormous dragon slowly emerges from the darkness behind the stone ruins.
Its scales are dark obsidian with subtle blue reflections.
Its eyes glow with ancient intelligence.
The elf remains calm and looks upward.
The camera slowly pulls backward to reveal the enormous scale.
""",

"""
The elf mage and the ancient dragon stand together on a mountain
overlooking the enchanted kingdom at sunrise.
Golden sunlight breaks through the clouds.
The dragon spreads its enormous wings slowly.
The elf mage stands beside it holding her glowing staff.
The camera slowly pulls backward into a magnificent wide cinematic shot.
Epic but realistic fantasy cinematography.
"""
]


# ============================================================
# UTILITY
# ============================================================

def cleanup():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def prepare_frame(frame):

    frame = np.asarray(frame)

    # Handle float output
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = frame * 255.0

        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Remove accidental batch dimension
    if frame.ndim == 4:
        frame = frame[0]

    return Image.fromarray(frame, "RGB")


# ============================================================
# START
# ============================================================

print("=" * 70)
print("WAN 2.2 — CINEMATIC FANTASY 30 SECOND VIDEO")
print("=" * 70)

print("GPU:", torch.cuda.get_device_name(0))

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2
    ),
    "GB"
)

print()
print("Resolution:", WIDTH, "x", HEIGHT)
print("Frames/scene:", FRAMES)
print("FPS:", FPS)
print("Steps:", STEPS)
print("Scenes:", NUM_SCENES)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print()
print("Loading Wan 2.2...")

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

current_image = Image.new(
    "RGB",
    (WIDTH, HEIGHT),
    (15, 18, 20)
)


# ============================================================
# GENERATE SCENES
# ============================================================

scene_files = []

for index, scene in enumerate(SCENES):

    scene_number = index + 1

    print()
    print("=" * 70)
    print(f"SCENE {scene_number}/{NUM_SCENES}")
    print("=" * 70)

    prompt = f"""
{CHARACTER}

{STYLE}

SCENE:
{scene}

Maintain exactly the same character identity, clothing,
hair, face, proportions and staff design from the previous scene.

The motion should be slow, physically realistic and cinematic.

Do not introduce unnecessary characters.

{NEGATIVE}
"""

    print("Generating...")

    generator = torch.Generator(device="cpu").manual_seed(
        SEED + scene_number
    )

    output = pipe(
        image=current_image,
        prompt=prompt,
        negative_prompt=NEGATIVE,
        height=HEIGHT,
        width=WIDTH,
        num_frames=FRAMES,
        num_inference_steps=STEPS,
        guidance_scale=5.0,
        generator=generator,
    )

    frames = output.frames[0]

    scene_path = os.path.join(
        OUTPUT_DIR,
        f"scene_{scene_number:02d}.mp4"
    )

    export_to_video(
        frames,
        scene_path,
        fps=FPS
    )

    scene_files.append(scene_path)

    print("Scene saved:")
    print(scene_path)

    # Last generated frame becomes the next scene's conditioning image
    last_frame = frames[-1]

    current_image = prepare_frame(last_frame)

    print("Continuity frame prepared.")

    cleanup()


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

    for scene_file in scene_files:

        f.write(
            "file '" +
            scene_file.replace("'", "'\\''") +
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
        "14",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        FINAL_VIDEO,
    ],
    check=True
)


# ============================================================
# VALIDATION
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
    check=True
)

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)
