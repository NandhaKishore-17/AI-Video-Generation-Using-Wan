"""
Job Manager — orchestrates the full generation pipeline.
"""
import asyncio
import json
import logging
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.job import Job, JobStatus

logger = logging.getLogger(__name__)


def _job_to_dict(job: Job) -> Dict[str, Any]:
    """Snapshot a Job ORM row as a plain dict while the session is open."""
    return {
        "id": job.id,
        "prompt": job.prompt,
        "status": job.status.value if hasattr(job.status, "value") else job.status,
        "progress": job.progress,
        "result_path": job.result_path,
        "story_id": job.story_id,
        "error_message": job.error_message,
    }


class JobManager:
    """Async job manager backed by an asyncio.Queue."""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start_worker(self):
        if self._running:
            return
        self._running = True
        logger.info("JobManager worker started")
        while self._running:
            job_id = await self.queue.get()
            try:
                await self._process_job(job_id)
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error("Unhandled error processing job %s:\n%s", job_id, tb)
                self._update_job(
                    job_id,
                    status=JobStatus.FAILED,
                    progress=0.0,
                    error_message=tb,
                )
            finally:
                self.queue.task_done()

    async def stop_worker(self):
        self._running = False

    def enqueue_job(self, prompt: str) -> str:
        return self._enqueue_payload({"job_type": "text", "prompt": prompt})

    def enqueue_complete_video_job(self, request: Dict[str, Any]) -> str:
        return self._enqueue_payload({"job_type": "complete_video", "request": request})

    def enqueue_story_only_job(self, request: Dict[str, Any]) -> str:
        return self._enqueue_payload({"job_type": "story_only", "request": request})

    def _enqueue_payload(self, payload: Dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        db: Session = SessionLocal()
        try:
            job = Job(
                id=job_id,
                prompt=json.dumps(payload),
                status=JobStatus.QUEUED,
                progress=0.0,
            )
            db.add(job)
            db.commit()
        finally:
            db.close()
        self.queue.put_nowait(job_id)
        logger.info("Job %s enqueued", job_id)
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return job data as a plain dict, or None if not found."""
        db: Session = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job is None:
                return None
            return _job_to_dict(job)
        finally:
            db.close()

    def _update_job(self, job_id: str, **kwargs) -> None:
        """Patch fields on a job row inside a fresh session."""
        db: Session = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job is None:
                logger.warning("_update_job: job %s not found", job_id)
                return
            for key, value in kwargs.items():
                setattr(job, key, value)
            db.commit()
        finally:
            db.close()

    async def _process_job(self, job_id: str) -> None:
        logger.info("Processing job %s", job_id)
        payload = self._get_payload(job_id)
        self._update_job(job_id, status=JobStatus.RUNNING, progress=5.0)

        if payload.get("job_type") == "complete_video":
            await self._process_complete_video_job(job_id, payload.get("request", {}))
            return

        if payload.get("job_type") == "story_only":
            await self._process_story_only_job(job_id, payload.get("request", {}))
            return

        await self._process_text_job(job_id, payload)

    async def _process_complete_video_job(self, job_id: str, request: Dict[str, Any]) -> None:
        try:
            from app.services.complete_video_service import complete_video_service

            async def progress_callback(stage: str, value: float) -> None:
                self._update_job(job_id, progress=value)

            logger.info("Job %s: running complete video pipeline", job_id)
            result = await complete_video_service.generate_complete_video(
                request=request,
                job_id=job_id,
                progress_callback=progress_callback,
            )
            self._update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100.0,
                result_path=result.get("output_path"),
            )
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Job %s: complete video pipeline failed: %s", job_id, tb)
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                progress=0.0,
                error_message=tb,
            )

    async def _process_story_only_job(self, job_id: str, request: Dict[str, Any]) -> None:
        """Execute the story-only pipeline: LLM story → dialogue → TTS → subtitles."""
        try:
            from app.services.complete_video_service import complete_video_service

            async def progress_callback(stage: str, value: float) -> None:
                self._update_job(job_id, progress=value)

            logger.info("Job %s: running story-only pipeline", job_id)
            result = await complete_video_service.generate_story_only(
                request=request,
                job_id=job_id,
                progress_callback=progress_callback,
            )
            self._update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress=100.0,
                result_path=result.get("job_dir"),  # debug dir instead of mp4
            )
            logger.info("Job %s: story-only pipeline complete. debug_dir=%s", job_id, result.get("job_dir"))
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Job %s: story-only pipeline failed: %s", job_id, tb)
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                progress=0.0,
                error_message=tb,
            )

    async def _process_text_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        story_id: Optional[str] = None
        try:
            from app.engines.llm.story_generator import story_generator

            logger.info("Job %s: Step 2 - Calling Ollama story generator...", job_id)
            self._update_job(job_id, progress=15.0)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, story_generator.generate, self._get_prompt(job_id), job_id
            )
            story_id = result.get("story_id") if isinstance(result, dict) else str(result)
            logger.info("Job %s: story %s generated successfully", job_id, story_id)
            self._update_job(job_id, story_id=story_id, progress=60.0)

        except Exception:
            tb = traceback.format_exc()
            logger.warning("Job %s: Ollama story generator raised exception:\n%s", job_id, tb)
            logger.info("Continuing job %s to video generation phase with default story ID", job_id)
            story_id = f"fallback_{job_id}"
            self._update_job(job_id, story_id=story_id, progress=60.0)

        try:
            from app.engines.video.wan_video_engine import get_video_engine

            logger.info("Job %s: Step 3 - Generating video via local WAN engine...", job_id)
            output_file = self._video_output_path(job_id)
            prompt = self._get_prompt(job_id)

            engine = await get_video_engine()
            await engine.render_video_async(
                scene_prompt=prompt,
                output_path=output_file,
                width=640,
                height=360,
                fps=12,
                duration=2.0,
                seed=42,
            )

            output_path = output_file
            logger.info("Job %s: video generated at %s", job_id, output_path)
            self._update_job(job_id, progress=90.0)

        except Exception:
            tb = traceback.format_exc()
            logger.error("Job %s: video generation failed with traceback:\n%s", job_id, tb)
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                progress=60.0,
                error_message=tb,
            )
            return

        logger.info("Job %s: Step 4 - Marking COMPLETED with result_path=%s", job_id, output_path)
        self._update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100.0,
            result_path=output_path,
        )
        logger.info("Job %s COMPLETED successfully", job_id)

    def _get_payload(self, job_id: str) -> Dict[str, Any]:
        db: Session = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if not job:
                return {}
            try:
                return json.loads(job.prompt)
            except (TypeError, ValueError):
                return {"job_type": "text", "prompt": job.prompt}
        finally:
            db.close()

    def _get_prompt(self, job_id: str) -> str:
        payload = self._get_payload(job_id)
        if isinstance(payload, dict):
            if payload.get("request") and isinstance(payload.get("request"), dict):
                return str(payload["request"].get("prompt") or payload["request"].get("theme") or "Default prompt")
            return str(payload.get("prompt") or "Default prompt")
        return str(payload)

    def _video_output_path(self, job_id: str) -> str:
        output_dir = Path("./media_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        return str((output_dir / f"{job_id}.mp4").resolve())


job_manager = JobManager()
