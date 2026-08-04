"""
verify_wan22_inference.py
=========================
Complete end-to-end verification of the Wan 2.2 TI2V-5B pipeline.

Steps performed:
  1.  Environment check  (PyTorch version, CUDA, GPU name, device)
  2.  Model source resolution (local path or HF repo)
  3-6.  Pipeline load + component listing
  7.  Scheduler config
  8.  Real inference: "A futuristic cyberpunk city at night with neon lights"
  9.  File validation  (exists + size > 0)
  9b. API round-trip   (optional, requires server running)

Run from the backend/ directory:
    cd backend
    python verify_wan22_inference.py
"""

from __future__ import annotations
import os, sys, time, uuid, traceback
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; RESET  = "\033[0m"

def ok(m):   print(f"  {GREEN}OK  {m}{RESET}")
def err(m):  print(f"  {RED}ERR {m}{RESET}")
def info(m): print(f"  {CYAN}>>  {m}{RESET}")
def warn(m): print(f"  {YELLOW}WRN {m}{RESET}")
def hdr(t):
    print(f"\n{BOLD}{'='*70}\n  {t}\n{'='*70}{RESET}")

# ── STEP 1: Environment ──────────────────────────────────────────────────────
def step1_environment():
    hdr("STEP 1/9 — Environment Check")
    import torch
    pv = torch.__version__
    cuda = torch.cuda.is_available()
    ok(f"PyTorch version : {pv}")
    result = {"pytorch_version": pv, "cuda_available": cuda}
    if cuda:
        gpu_name = torch.cuda.get_device_name(0)
        ok(f"CUDA available  : True")
        ok(f"GPU name        : {gpu_name}")
        ok(f"Device          : cuda:0")
        result.update({"device": "cuda:0", "gpu_name": gpu_name, "torch_dtype": torch.bfloat16})
    else:
        warn("CUDA NOT available — CPU mode (slow!)")
        ok(f"Device          : cpu")
        result.update({"device": "cpu", "gpu_name": "N/A", "torch_dtype": __import__("torch").float32})
    return result

# ── STEP 2: Model source ─────────────────────────────────────────────────────
def step2_resolve_model(env):
    hdr("STEP 2/9 — Resolving Model Source")
    env_val = os.environ.get("WAN_MODEL_PATH", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    local = Path(env_val)
    if local.exists() and local.is_dir():
        resolved = str(local.resolve())
        ok(f"LOCAL dir: {resolved}")
    elif "Wan-AI/" in env_val:
        resolved = env_val
        ok(f"HuggingFace repo: {resolved}")
    else:
        resolved = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        warn(f"'{env_val}' not found -> fallback to {resolved}")
    return resolved

# ── STEPS 3-7: Load pipeline ─────────────────────────────────────────────────
def step3_7_load_pipeline(model_id, env):
    hdr("STEPS 3-7/9 — Loading WanImageToVideoPipeline")
    import torch
    from diffusers import WanImageToVideoPipeline
    from diffusers.schedulers import UniPCMultistepScheduler

    dtype = env["torch_dtype"]
    info(f"Model source : {model_id}")
    info(f"torch_dtype  : {dtype}")
    info("Loading weights (may take minutes) ...")

    t0 = time.time()
    pipeline = WanImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
    ok(f"Pipeline loaded in {time.time()-t0:.1f}s  [{type(pipeline).__name__}]")

    # Scheduler
    try:
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config, flow_shift=5.0)
        ok("Scheduler : UniPCMultistepScheduler (flow_shift=5.0)")
    except Exception as e:
        warn(f"flow_shift not supported: {e}")
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        ok("Scheduler : UniPCMultistepScheduler (default)")

    # Component listing
    hdr("Loaded Components")
    for name in ["tokenizer","text_encoder","transformer","vae","scheduler","image_encoder"]:
        comp = getattr(pipeline, name, None)
        if comp is not None:
            ok(f"{name:<18}: {type(comp).__name__}")
        else:
            warn(f"{name:<18}: MISSING / None")

    # Fallback guard
    ptype = type(pipeline).__name__
    if any(w in ptype.lower() for w in ["mock","placeholder","fake","dummy"]):
        raise RuntimeError(f"Fallback/mock pipeline detected: {ptype}")
    ok(f"REAL pipeline confirmed: {ptype} (no fallback)")

    # Device placement
    cpu_offload = os.environ.get("WAN_ENABLE_CPU_OFFLOAD","true").lower() == "true"
    if cpu_offload and env["cuda_available"]:
        pipeline.enable_model_cpu_offload()
        ok("model_cpu_offload ENABLED")
    else:
        pipeline = pipeline.to(env["device"])
        ok(f"Pipeline moved to {env['device']}")

    # VAE opts
    if getattr(pipeline, "vae", None):
        try:
            pipeline.vae.enable_slicing(); pipeline.vae.enable_tiling()
            ok("VAE slicing + tiling enabled")
        except Exception:
            pass

    return pipeline

# ── STEP 8: Real inference ───────────────────────────────────────────────────
def step8_inference(pipeline, env, job_id):
    hdr("STEP 8/9 — Real Wan2.2 Inference")
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    PROMPT     = "A futuristic cyberpunk city at night with neon lights"
    NEG_PROMPT = "Distorted, discontinuous, ugly, blurry, low resolution, motionless, static, disfigured"
    WIDTH=480; HEIGHT=272; FPS=16; FRAMES=25; STEPS=20; GUIDANCE=5.0; SEED=42

    out_dir = BACKEND_DIR.parent / "media_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / f"{job_id}.mp4")

    info(f"Prompt      : {PROMPT}")
    info(f"Resolution  : {WIDTH}x{HEIGHT}  frames={FRAMES}  fps={FPS}  steps={STEPS}")
    info(f"Output path : {out_path}")

    cond = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    dev = env["device"]
    gen = torch.Generator(device="cpu" if dev == "cpu" else dev).manual_seed(SEED)

    ok("STARTING inference loop ...")
    t0 = time.time()
    with torch.inference_mode():
        output = pipeline(
            image=cond, prompt=PROMPT, negative_prompt=NEG_PROMPT,
            num_inference_steps=STEPS, guidance_scale=GUIDANCE,
            height=HEIGHT, width=WIDTH, num_frames=FRAMES, generator=gen,
        )
    ok(f"Inference done in {time.time()-t0:.1f}s")

    frames = output.frames[0]
    ok(f"Generated {len(frames)} frames")

    hdr("STEP 9/9 — Exporting MP4")
    export_to_video(frames, out_path, fps=FPS)
    ok(f"Video saved -> {out_path}")
    return out_path

# ── STEP 9a: File validation ─────────────────────────────────────────────────
def step9a_validate(out_path):
    hdr("File Validation")
    p = Path(out_path)
    assert p.exists(), f"File NOT found: {out_path}"
    size = p.stat().st_size
    assert size > 0, f"File is EMPTY: {out_path}"
    ok(f"File exists: {p}")
    ok(f"File size  : {size:,} bytes ({size/1024:.1f} KB)")

# ── STEP 9b: API round-trip ──────────────────────────────────────────────────
def step9b_api(job_id, out_path):
    hdr("STEP 9b — API Round-Trip (POST /generate + GET /status)")
    try:
        import httpx
    except ImportError:
        warn("httpx not available — skip API test"); return

    API = "http://127.0.0.1:8000/api/v1"
    try:
        r = httpx.post(f"{API}/generate",
            json={"prompt":"A futuristic cyberpunk city at night with neon lights"},
            timeout=10)
        if r.status_code == 200:
            aid = r.json()["job_id"]
            ok(f"POST /generate -> 200 OK  job_id={aid}")
        else:
            warn(f"POST /generate -> {r.status_code}: {r.text[:200]}"); return
    except Exception as e:
        warn(f"Server not reachable: {e}")
        info("Start server: uvicorn app.main:app --port 8000"); return

    info(f"Polling GET /api/v1/status/{aid} ...")
    for _ in range(90):
        time.sleep(2)
        try:
            s = httpx.get(f"{API}/status/{aid}", timeout=10).json()
            status = s.get("status","?"); prog = s.get("progress",0)
            info(f"  status={status}  progress={prog}")
            if status in ("completed","failed"):
                break
        except Exception: pass

    if s.get("status") == "completed":
        ok(f"API job COMPLETED successfully!")
        ok(f"  video_path    = {s.get('video_path')}")
        ok(f"  error_message = {s.get('error_message')}")
    else:
        warn(f"Final status: {s.get('status')} | error: {str(s.get('error_message',''))[:300]}")

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{CYAN}{'#'*70}")
    print(f"  Wan 2.2 TI2V-5B — Real Inference Verification")
    print(f"{'#'*70}{RESET}\n")

    job_id = str(uuid.uuid4())
    info(f"Verification job_id: {job_id}")

    env_file = BACKEND_DIR.parent / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(str(env_file))
            ok(f"Loaded .env from {env_file}")
        except ImportError:
            pass

    try:
        env      = step1_environment()
        model_id = step2_resolve_model(env)
        pipeline = step3_7_load_pipeline(model_id, env)
        out_path = step8_inference(pipeline, env, job_id)
        step9a_validate(out_path)
        step9b_api(job_id, out_path)

        hdr("VERIFICATION COMPLETE")
        ok(f"Real Wan2.2 MP4 at: {out_path}")
        ok(f"Job ID: {job_id}")
        print(f"\n{BOLD}{GREEN}  ALL CHECKS PASSED — Wan2.2 inference is working!{RESET}\n")

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted.{RESET}")
        sys.exit(1)
    except Exception as exc:
        hdr("FAILURE — Full Traceback")
        print(f"{RED}{traceback.format_exc()}{RESET}")
        print(f"\n{BOLD}{RED}Root cause: {exc}{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
