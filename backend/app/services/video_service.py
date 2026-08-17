import logging
import asyncio
import os
from pathlib import Path
from typing import Optional
from uuid import uuid4
import httpx

from app.core.config import settings
from app.services.runpod_wan_client import runpod_client

logger = logging.getLogger("video_service")

class VideoGenerationService:
    def __init__(self):
        self.output_dir = Path(settings.MEDIA_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_video(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        duration: float = 2.0,
        width: int = 640,
        height: int = 360,
        fps: int = 24,
        output_path: Optional[str] = None,
        aspect_ratio: str = "16:9"
    ) -> str:
        """Generate video remotely using Runpod Serverless GPU."""
        
        if not output_path:
            filename = f"scene_{uuid4().hex}.mp4"
            output_path = str(self.output_dir / filename)
            
        logger.info("==================================================")
        logger.info("STARTING REMOTE WAN2.2 VIDEO GENERATION VIA RUNPOD")
        logger.info(f"Prompt: {prompt[:500]}")
        logger.info(f"Output: {output_path}")
        logger.info("==================================================")

        payload = {
            "prompt": prompt,
            "model": getattr(settings, "WAN_MODEL", "Wan2.2-TI2V-5B"),
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "width": width,
            "height": height,
            "fps": fps
        }

        if image_url:
            if image_url.startswith("/") or "localhost" in image_url or "127.0.0.1" in image_url:
                logger.error(f"Local/relative image URL detected: {image_url}")
                raise Exception("Image-to-video requires an image accessible by the configured Runpod endpoint.")
            payload["image_url"] = image_url

        try:
            # 1. Generate Video
            video_url = await runpod_client.generate(payload)
            
            # 2. Download Video
            logger.info("Video download started")
            
            timeout = httpx.Timeout(
                60.0,
                connect=15.0,
                read=60.0,
                write=15.0,
                pool=10.0
            )
            
            async with httpx.AsyncClient(timeout=timeout) as client:
                video_resp = await client.get(video_url)
                video_resp.raise_for_status()
                
                with open(output_path, "wb") as f:
                    async for chunk in video_resp.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                raise Exception("Video download failed or file is empty")
                
            logger.info("Video saved successfully")
            return f"/media/{Path(output_path).name}"

        except httpx.RequestError as e:
            logger.exception("Remote API connection failure or timeout during video download")
            raise Exception("Video download or network connection failure") from e
        except IOError as e:
            logger.exception("Disk write failure during video download")
            raise Exception("Video download or disk write failure") from e
        except Exception as e:
            # Let the runpod client's specific errors bubble up cleanly
            raise

video_service = VideoGenerationService()
