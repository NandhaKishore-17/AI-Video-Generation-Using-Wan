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
# WAN 2.2 — CONTINUOUS 30 SECOND FANTASY ADVENTURE
# ============================================================

load_dotenv("backend/.env")

MODEL_PATH = os.environ["WAN_MODEL_PATH"]

OUTPUT_DIR = "/root/itsme/media_output/fantasy_30s"
FINAL_VIDEO = "/root/itsme/media_output/wan_fantasy_30s.mp4"

# Stable configuration for RTX 3090
WIDTH = 832
HEIGHT = 480

FRAMES = 49
FPS = 16
STEPS = 50

NUM_SCENES = 10


# ============================================================
# GLOBAL VISUAL IDENTITY
# ============================================================

STYLE = """
cinematic fantasy adventure film,
photorealistic live-action appearance,
natural realistic human anatomy,
realistic skin and fabric,
physically realistic lighting,
natural colors,
subtle cinematic color grading,
high dynamic range,
detailed environment,
realistic atmospheric depth,
soft volumetric light,
natural motion,
slow deliberate camera movement,
stable camera,
professional cinematography,
35mm cinema lens,
shallow depth of field when appropriate,
fine environmental detail,
no text,
no subtitles,
no watermark,
no fantasy illustration,
no cartoon,
no anime
"""


CHARACTER = """
the same lone female explorer throughout the entire story,
approximately 30 years old,
dark brown shoulder-length hair,
weathered olive-green expedition jacket,
dark brown hiking trousers,
brown leather boots,
small dark backpack,
brass compass hanging from her belt,
realistic human proportions,
consistent face,
consistent hairstyle,
consistent clothing,
consistent body proportions
"""


# ============================================================
# STORY
# ============================================================

SCENES = [

"""
The explorer walks slowly along an ancient forest path at dawn.
Huge moss-covered trees surround her.
Thin golden sunlight passes through the branches.
Small particles of dust float in the sunlight.
She carefully looks around while walking forward.
The camera follows behind her with a slow smooth tracking movement.
The forest feels mysterious but peaceful.
""",

"""
The explorer continues deeper into the same ancient forest.
The same trees, same clothing and same morning lighting remain consistent.
She notices strange glowing blue plants beside the path.
She stops and carefully kneels beside one glowing plant.
The camera slowly moves closer to her.
The movement is natural and realistic.
""",

"""
The explorer stands and continues along the same forest path.
The glowing blue plants become more numerous.
A gentle mist begins appearing between the trees.
She walks slowly through the mist.
The camera moves beside her at walking speed.
Her backpack, jacket and hair remain consistent.
""",

"""
The explorer reaches a small ancient stone bridge crossing a narrow stream.
The same forest surrounds the bridge.
She slowly walks across the bridge.
Water flows naturally beneath the stones.
Sunlight reflects softly on the water.
The camera follows from behind and slightly to the side.
""",

"""
The explorer reaches the other side of the bridge.
She discovers enormous ancient stone ruins covered in moss.
She slowly approaches the ruins.
Vines move gently in the wind.
She raises her hand and touches an old carved stone.
The camera makes a slow cinematic push toward the carving.
""",

"""
The ancient carving begins emitting a very subtle blue glow.
The explorer looks at it with surprise.
The same character and environment remain consistent.
Soft blue light illuminates her face.
She slowly steps backward and looks toward the forest.
The camera slowly rotates around her.
""",

"""
A hidden path opens between two enormous ancient trees.
The explorer cautiously walks through the opening.
A warm golden light is visible in the distance.
The camera follows directly behind her.
Leaves move naturally in a gentle breeze.
The atmosphere becomes increasingly magical but remains photorealistic.
""",

"""
The explorer emerges into a vast hidden clearing.
At the center stands an enormous ancient tree with glowing blue branches.
She stops several meters away and looks toward the tree.
The camera slowly moves from behind her toward the enormous tree.
Mist floats close to the ground.
The scene feels majestic and mysterious.
""",

"""
The explorer slowly walks toward the enormous glowing tree.
Its blue light gently illuminates the surrounding forest.
She reaches the base of the tree and places one hand against its bark.
The glow becomes slightly brighter.
Her face shows quiet wonder.
The camera slowly moves closer.
""",

"""
The explorer stands beneath the enormous glowing tree.
Blue and golden light gently fills the clearing.
The same character, clothing and environment remain consistent.
She looks upward toward the branches.
The camera slowly pulls backward and upward,
revealing the enormous tree and the surrounding ancient forest.
A peaceful cinematic ending.
"""
]


# ============================================================
# HELPERS
# ============================================================

def normalize_frame(frame):

    arr = np.asarray(frame)

    if arr.dtype != np.uint8:

        if arr.max() <= 1.0:
            arr = arr * 255.0

        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 4:
        arr = arr[0]

    return arr


def extract_last_frame(video_frames):

    last = video_frames[-1]

    last = normalize_frame(last)

    return Image.fromarray(last).convert("RGB")


def make_start_image():

    # Neutral dark image for first scene.
    # The first generated frame will establish the actual scene.
    return Image.new(
        "RGB",
        (WIDTH, HEIGHT),
        (15, 18, 16)
    )


def concatenate_videos():

    concat_file = os.path.join(
        OUTPUT_DIR,
        "concat.txt"
    )

    with open(concat_file, "w") as f:

        for i in range(1, NUM_SCENES + 1):

            path = os.path.join(
                OUTPUT_DIR,
                f"scene_{i:02d}.mp4"
            )

            f.write(
                "file '" +
                path.replace("'", "'\\''") +
                "'\n"
            )

    print("\nConcatenating scenes...")

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

            FINAL_VIDEO
        ],
        check=True
    )


# ============================================================
# MAIN
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("=" * 70)
print("WAN 2.2 — CONTINUOUS FANTASY ADVENTURE")
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

print()
print("Loading Wan 2.2...")

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

pipe.enable_model_cpu_offload()

print("Wan loaded with CPU offloading.")


current_image = make_start_image()

generated_frames = []


for scene_index in range(NUM_SCENES):

    print()
    print("=" * 70)
    print(
        f"SCENE {scene_index + 1}/{NUM_SCENES}"
    )
    print("=" * 70)

    prompt = (
        STYLE
        + "\n"
        + CHARACTER
        + "\n"
        + SCENES[scene_index]
    )

    print("Generating...")

    with torch.inference_mode():

        result = pipe(
            image=current_image,

            prompt=prompt,

            height=HEIGHT,
            width=WIDTH,

            num_frames=FRAMES,

            num_inference_steps=STEPS,

            guidance_scale=5.0,

        )

    frames = result.frames[0]

    generated_frames.append(frames)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"scene_{scene_index + 1:02d}.mp4"
    )

    export_to_video(
        frames,
        output_path,
        fps=FPS
    )

    print()
    print("Scene saved:")
    print(output_path)

    # --------------------------------------------------------
    # CONTINUITY
    # --------------------------------------------------------

    current_image = extract_last_frame(frames)

    # Force exact generation resolution.
    current_image = current_image.resize(
        (WIDTH, HEIGHT),
        Image.Resampling.LANCZOS
    )

    current_image.save(
        os.path.join(
            OUTPUT_DIR,
            f"continuity_{scene_index + 1:02d}.png"
        )
    )

    print("Continuity frame prepared.")

    # --------------------------------------------------------
    # MEMORY CLEANUP
    # --------------------------------------------------------

    del result
    del frames

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# CONCATENATE
# ============================================================

print()
print("=" * 70)
print("ALL SCENES GENERATED")
print("=" * 70)

concatenate_videos()


# ============================================================
# FINAL VALIDATION
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
        FINAL_VIDEO
    ],
    check=True
)

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)
