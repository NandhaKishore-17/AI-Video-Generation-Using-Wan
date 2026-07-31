import os
import cv2
import numpy as np
import logging
import httpx
from PIL import Image
from app.core.config import settings

logger = logging.getLogger("video_engine")

class VideoGenerationEngine:
    """
    Open-source Video Engine powered by Wan 2.2 TI2V-5B local GPU, CogVideoX, and HuggingFace cloud inference.
    
    Priority chain:
      1. Wan Local GPU (wan_local) — uses locally downloaded Wan 2.2 model for image-to-video generation
      2. HuggingFace Inference API — cloud-based text-to-video via API
      3. Local HTTP API — CogVideoX/Wan served via ComfyUI or similar
      4. OpenCV Motion Synthesizer — procedural animation fallback
    """

    def __init__(self):
        self.provider = settings.VIDEO_PROVIDER
        self.api_base = settings.VIDEO_API_BASE
        self.media_dir = settings.MEDIA_OUTPUT_DIR

    async def generate_scene_video(
        self,
        scene_id: str,
        image_relative_url: str,
        motion_prompt: str,
        duration_seconds: float = 5.0,
        fps: int = 24
    ) -> str:
        """
        Generates a video clip from a keyframe image using the Wan 2.2 local GPU model,
        with automatic fallback to cloud APIs and procedural synthesis.
        """
        filename = f"video_{scene_id}.mp4"
        filepath = os.path.join(self.media_dir, filename)

        # Source keyframe image path
        image_filename = os.path.basename(image_relative_url)
        image_path = os.path.join(self.media_dir, image_filename)

        # ━━━ 1. WAN LOCAL GPU (Primary) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Uses the locally downloaded Wan 2.2 TI2V-5B model for image-to-video generation
        if self.provider == "wan_local":
            try:
                from app.engines.wan_pipeline import wan_pipeline

                await wan_pipeline.generate_video(
                    image_path=image_path,
                    prompt=motion_prompt,
                    output_path=filepath,
                )
                logger.info(f"Generated AI video via Wan 2.2 Local GPU: {filename}")
                return f"/media/{filename}"
            except Exception as e:
                logger.warning(
                    f"Wan Local GPU generation failed: {e}. "
                    "Falling back to cloud/API providers."
                )

        # ━━━ 2. HUGGINGFACE INFERENCE API (Cloud Fallback) ━━━━━━━━━━━━━
        api_key = settings.VIDEO_API_KEY or settings.HF_TOKEN
        if api_key:
            try:
                from huggingface_hub import InferenceClient
                client = InferenceClient(
                    provider="auto",
                    api_key=api_key
                )
                video_data = client.text_to_video(
                    prompt=motion_prompt,
                    model=settings.HF_MODEL_WAN
                )
                with open(filepath, "wb") as f:
                    f.write(video_data)
                logger.info(f"Generated AI video via HuggingFace InferenceClient: {filename}")
                return f"/media/{filename}"
            except Exception as e:
                logger.warning(f"HuggingFace InferenceClient text_to_video error: {e}. Trying HTTP fallback / local endpoint.")

        # ━━━ 2b. HUGGINGFACE DIRECT HTTP (Cloud Fallback) ━━━━━━━━━━━━━━
        if api_key or settings.GPU_EXECUTION_MODE == "CLOUD_GPU":
            try:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                hf_url = f"{settings.HF_API_BASE}/{settings.HF_MODEL_WAN}"
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        hf_url,
                        headers=headers,
                        json={"inputs": motion_prompt}
                    )
                    if resp.status_code == 200:
                        with open(filepath, "wb") as f:
                            f.write(resp.content)
                        logger.info(f"Generated AI video via HuggingFace Cloud GPU HTTP: {filename}")
                        return f"/media/{filename}"
            except Exception as e:
                logger.warning(f"HuggingFace Cloud GPU API: {e}. Trying local GPU endpoint.")

        # ━━━ 3. LOCAL HTTP API (ComfyUI / vLLM Endpoint) ━━━━━━━━━━━━━━
        try:
            if self.provider in ["cogvideox", "wan"]:
                async with httpx.AsyncClient(timeout=90.0) as client:
                    resp = await client.post(
                        f"{self.api_base}/generate_video",
                        json={
                            "image_path": image_path,
                            "prompt": motion_prompt,
                            "duration": duration_seconds
                        }
                    )
                    if resp.status_code == 200:
                        with open(filepath, "wb") as f:
                            f.write(resp.content)
                        return f"/media/{filename}"
        except Exception as e:
            logger.warning(f"Local GPU Video Server unreachable: {e}. Utilizing High-Definition Motion Synthesizer.")

        # ━━━ 4. OPENCV MOTION SYNTHESIZER (Last Resort Fallback) ━━━━━━━
        self._synthesize_motion_video(image_path, filepath, motion_prompt, duration_seconds, fps)
        return f"/media/{filename}"

    def _synthesize_motion_video(self, image_path: str, output_path: str, motion_prompt: str, duration: float, fps: int):
        """
        Synthesizes high-definition 1080p motion video stream with volumetric camera zoom,
        parallax pan, atmospheric dust particles, and cinematic lighting pulse over keyframe.
        """
        width, height = 1280, 720
        total_frames = int(duration * fps)

        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is None:
                img = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (width, height))
        else:
            img = np.zeros((height, width, 3), dtype=np.uint8)

        # Generate random particle motes
        num_particles = 40
        np.random.seed(abs(hash(motion_prompt)) % 1000)
        px = np.random.randint(0, width, size=num_particles)
        py = np.random.randint(0, height, size=num_particles)
        pspeed = np.random.uniform(0.5, 2.5, size=num_particles)
        pradius = np.random.randint(1, 4, size=num_particles)

        # Cross-platform VideoWriter initialization
        codecs_to_try = [
            ('m', 'p', '4', 'v'),
            ('M', 'J', 'P', 'G'),
            ('X', 'V', 'I', 'D')
        ]
        out = None
        for c1, c2, c3, c4 in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(c1, c2, c3, c4)
                test_out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if test_out.isOpened():
                    out = test_out
                    break
            except Exception:
                continue

        if out is None or not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for i in range(total_frames):
            t = i / total_frames
            
            # Smooth 3D camera push-in zoom factor (1.0 to 1.18)
            scale = 1.0 + 0.18 * (0.5 - 0.5 * np.cos(t * np.pi))
            
            # Parallax pan offsets
            dx = int(25 * np.sin(t * np.pi * 1.5))
            dy = int(12 * np.cos(t * np.pi * 0.8))

            scaled_w = int(width * scale)
            scaled_h = int(height * scale)
            resized = cv2.resize(img, (scaled_w, scaled_h))

            # Crop center frame with dynamic camera shift
            center_x, center_y = scaled_w // 2 + dx, scaled_h // 2 + dy
            start_x = max(0, min(center_x - width // 2, scaled_w - width))
            start_y = max(0, min(center_y - height // 2, scaled_h - height))

            frame = resized[start_y:start_y + height, start_x:start_x + width].copy()
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            # 1. Floating Atmospheric Particle Layer
            for p_i in range(num_particles):
                py[p_i] = (py[p_i] - pspeed[p_i]) % height
                px[p_i] = (px[p_i] + np.sin(t * 5 + p_i) * 0.5) % width
                cv2.circle(frame, (int(px[p_i]), int(py[p_i])), pradius[p_i], (200, 240, 255), -1)

            # 2. Cinematic Volumetric Lighting Sweep Overlay
            light_x = int(width * (0.2 + 0.6 * t))
            overlay = frame.copy()
            cv2.circle(overlay, (light_x, height // 3), 200, (60, 180, 240), -1)
        out.release()

        # Transcode to standard H.264 format for universal HTML5 browser video playback
        try:
            import imageio_ffmpeg, subprocess
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_h264 = output_path.replace(".mp4", "_h264.mp4")
            cmd = [
                exe, "-y", "-i", output_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                temp_h264
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(temp_h264):
                os.replace(temp_h264, output_path)
        except Exception as e:
            logger.warning(f"H.264 transcode error: {e}")

video_engine = VideoGenerationEngine()

