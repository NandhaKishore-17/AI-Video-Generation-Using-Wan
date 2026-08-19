import os
import torch
from dotenv import load_dotenv
from PIL import Image
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video

load_dotenv("backend/.env")

model_path = os.environ["WAN_MODEL_PATH"]

print("=" * 60)
print("WAN 2.2 — HIGH QUALITY TEST")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB"
)

print("\nLoading Wan 2.2...")

pipe = WanImageToVideoPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

pipe.enable_model_cpu_offload()

print("CPU offloading enabled.")

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

WIDTH = 832
HEIGHT = 480

# 49 frames at 16 FPS = 3.0625 seconds
FRAMES = 49

FPS = 16

# Better quality than the previous 20-step test
STEPS = 30

# ------------------------------------------------------------
# INPUT IMAGE
# ------------------------------------------------------------

image = Image.new(
    "RGB",
    (WIDTH, HEIGHT),
    (18, 18, 22)
)

prompt = """
cinematic photorealistic film scene, a lone young man walking slowly
through an ancient forest at dawn, enormous ancient trees, detailed
moss-covered rocks, soft golden sunlight passing through the leaves,
volumetric light rays, realistic skin and clothing, natural human
movement, subtle wind moving leaves and hair, realistic shadows,
cinematic depth of field, physically realistic lighting, rich natural
colors, high dynamic range, detailed environment, professional
cinematography, smooth slow camera tracking shot, 35mm cinema lens,
beautiful composition, realistic texture, sharp details
"""

negative_prompt = """
blurry, low quality, low resolution, oversaturated, undersaturated,
washed out colors, gray image, cartoon, anime, illustration,
deformed person, distorted face, extra limbs, duplicate person,
flickering, jitter, unstable camera, unnatural movement,
bad anatomy, distorted environment, artifacts, noise
"""

print("\nGenerating...")
print("Resolution:", WIDTH, "x", HEIGHT)
print("Frames:", FRAMES)
print("FPS:", FPS)
print("Steps:", STEPS)

output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=HEIGHT,
    width=WIDTH,
    num_frames=FRAMES,
    num_inference_steps=STEPS,
    guidance_scale=5.0,
)

frames = output.frames[0]

output_path = "/root/itsme/media_output/wan_high_quality_test.mp4"

export_to_video(
    frames,
    output_path,
    fps=FPS
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)
print("Output:", output_path)
print("Frames:", len(frames))

torch.cuda.empty_cache()
