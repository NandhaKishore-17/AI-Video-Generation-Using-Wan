import logging
import asyncio
import time
import httpx
from typing import Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger("runpod_client")

class RunpodWanClient:
    """Isolated client for authenticating and polling Runpod Serverless Wan2.2."""
    def __init__(self):
        pass

    def _get_api_key(self) -> str:
        key = getattr(settings, "RUNPOD_API_KEY", None)
        if not key:
            raise Exception("RUNPOD_API_KEY is not configured.")
        return key.strip().strip('"').strip("'")

    def _get_endpoint_url(self) -> str:
        url = getattr(settings, "RUNPOD_WAN_ENDPOINT_URL", None)
        if not url:
            raise Exception("RUNPOD_WAN_ENDPOINT_URL is not configured.")
        return url.rstrip("/")

    def _normalize_runpod_result(self, data: Dict[str, Any]) -> dict:
        """Extract job ID, status, and video URL from Runpod response."""
        normalized = {
            "job_id": data.get("id"),
            "status": "pending",
            "video_url": None,
            "error": None
        }

        status = data.get("status")
        if status:
            status = str(status).upper()
            if status == "COMPLETED":
                normalized["status"] = "completed"
            elif status == "FAILED":
                normalized["status"] = "failed"
            elif status in ["IN_QUEUE", "IN_PROGRESS"]:
                normalized["status"] = "pending"

        # Check for output
        output = data.get("output")
        if output:
            if isinstance(output, dict) and "video_url" in output:
                normalized["video_url"] = output["video_url"]
            elif isinstance(output, str) and output.startswith("http"):
                normalized["video_url"] = output
                
        # Check for errors
        if data.get("error"):
            normalized["error"] = data["error"]

        return normalized

    async def generate(self, payload: dict) -> str:
        """Submit generation request, poll, and return video URL."""
        api_key = self._get_api_key()
        base_url = self._get_endpoint_url()
        run_url = f"{base_url}/run"
        status_url = f"{base_url}/status"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        timeout_seconds = getattr(settings, "WAN_TIMEOUT", 900)
        poll_interval = getattr(settings, "WAN_POLL_INTERVAL", 5)

        timeout = httpx.Timeout(
            timeout_seconds,
            connect=15.0,
            read=timeout_seconds,
            write=15.0,
            pool=10.0
        )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info("Runpod generation submitted")
                run_resp = await client.post(run_url, json={"input": payload}, headers=headers)
                
                if run_resp.status_code in (401, 403):
                    raise Exception("Runpod authentication failed (401/403)")
                elif run_resp.status_code == 429:
                    raise Exception("Runpod rate limit exceeded")
                    
                run_resp.raise_for_status()
                run_data = run_resp.json()
                norm = self._normalize_runpod_result(run_data)
                
                job_id = norm.get("job_id")
                if not job_id:
                    logger.error(f"Invalid Runpod Wan response: {run_data}")
                    raise Exception("Missing job ID in Runpod response")
                    
                logger.info(f"Runpod job ID: {job_id}")
                
                # If synchronous
                if norm["status"] == "completed" and norm["video_url"]:
                    return norm["video_url"]
                    
                # Poll
                start_time = time.time()
                last_logged_status = None
                
                while True:
                    if time.time() - start_time > timeout_seconds:
                        raise Exception("Runpod generation timeout")
                        
                    await asyncio.sleep(poll_interval)
                    
                    poll_resp = await client.get(f"{status_url}/{job_id}", headers=headers)
                    if poll_resp.status_code in (401, 403):
                        raise Exception("Runpod authentication failed during polling")
                    poll_resp.raise_for_status()
                    
                    poll_data = poll_resp.json()
                    poll_norm = self._normalize_runpod_result(poll_data)
                    
                    if poll_norm["status"] != last_logged_status:
                        logger.info(f"Runpod generation status: {poll_norm['status'].upper()}")
                        last_logged_status = poll_norm["status"]
                        
                    if poll_norm["status"] == "completed":
                        video_url = poll_norm["video_url"]
                        if not video_url:
                            logger.error(f"Missing video URL in completed response: {poll_data}")
                            raise Exception("Runpod API completed but no video URL was found")
                        return video_url
                    elif poll_norm["status"] == "failed":
                        err = poll_norm["error"] or "Unknown failure"
                        raise Exception(f"Runpod generation failed: {err}")

        except httpx.RequestError as e:
            logger.exception("Runpod connection failure or timeout")
            raise Exception("Runpod API connection failure or timeout") from e
        except Exception as e:
            if "authentication failed" in str(e).lower() or "timeout" in str(e).lower() or "is not configured" in str(e).lower() or "rate limit" in str(e).lower() or "missing job" in str(e).lower():
                raise
            logger.exception("Video generation failed with an unexpected error")
            raise Exception("Video generation failed") from e

runpod_client = RunpodWanClient()
