"""
Job Manager — orchestrates the full generation pipeline:

  User Prompt (POST /api/v1/generate)
      ↓
  API router (generation.py)
      ↓
  job_manager.enqueue_job(prompt)
      ↓
  video_service.generate_video(...)
      ↓
  WanVideoEngine (wan_video_engine.py)
      ↓
  Wan22Wrapper (wan22.py)
      ↓
  wan_pipeline (wan_pipeline.py)
      ↓
  Job marked COMPLETED
"""
import asyncio
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

    # ── Lifecycle ────────────────────────────────────────────────────────

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

    # ── Public API ───────────────────────────────────────────────────────

    def enqueue_job(self, prompt: str) -> str:
        job_id = str(uuid.uuid4())
        db: Session = SessionLocal()
        try:
            job = Job(
                id=job_id,
                prompt=prompt,
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

    # ── Internal pipeline ────────────────────────────────────────────────

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

        # ── Step 1: Mark RUNNING ─────────────────────────────────────────
        logger.info("Job %s: Step 1 - Marking RUNNING", job_id)
        self._update_job(
            job_id,
            status=JobStatus.RUNNING,
            progress=5.0,
        )

        # ── Step 2: Generate story via Ollama ────────────────────────────
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

        except Exception as exc:
            tb = traceback.format_exc()
            logger.warning("Job %s: Ollama story generator raised exception:\n%s", job_id, tb)
            logger.info("Continuing job %s to video generation phase with default story ID", job_id)
            story_id = f"fallback_{job_id}"
            self._update_job(job_id, story_id=story_id, progress=60.0)

        # ── Step 3: Generate video via video_service ─────────────────────
        try:
            from app.services.video_service import video_service

            logger.info("Job %s: Step 3 - Routing video generation via video_service...", job_id)
            output_file = self._video_output_path(job_id)
            prompt = self._get_prompt(job_id)

            relative_video_path = await video_service.generate_video(
                prompt=prompt,
                duration=2.0,
                width=640,
                height=360,
                fps=12,
                output_path=output_file,
            )

            output_path = relative_video_path
            logger.info("Job %s: video generated at %s", job_id, output_path)
            self._update_job(job_id, progress=90.0)

        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Job %s: video generation failed with traceback:\n%s", job_id, tb)
            self._update_job(
                job_id,
                status=JobStatus.FAILED,
                progress=60.0,
                error_message=tb,
            )
            return

        # ── Step 4: Mark COMPLETED ────────────────────────────────────────
        logger.info("Job %s: Step 4 - Marking COMPLETED with result_path=%s", job_id, output_path)
        self._update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100.0,
            result_path=output_path,
        )
        logger.info("Job %s COMPLETED successfully", job_id)

    def _get_prompt(self, job_id: str) -> str:
        """Read the prompt from the DB."""
        db: Session = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            return job.prompt if job else "Default prompt"
        finally:
            db.close()

    def _video_output_path(self, job_id: str) -> str:
        output_dir = Path("./media_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        return str((output_dir / f"{job_id}.mp4").resolve())


# Singleton instance
job_manager = JobManager()
