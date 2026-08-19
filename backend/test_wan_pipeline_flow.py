import asyncio
import logging
import sys
from pathlib import Path

# Add app directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_flow")

async def test_end_to_end():
    from app.core.database import engine, Base
    from app.models.job import Job  # noqa: F401
    from app.engines.job_manager import job_manager

    # Ensure DB tables are created
    Base.metadata.create_all(bind=engine)

    logger.info("Testing complete Wan2.2 execution flow...")
    prompt = "A majestic lion standing on a cliff at sunset, cinematic 4k"
    
    # 1. Enqueue job
    job_id = job_manager.enqueue_job(prompt)
    logger.info("1. Enqueued job ID: %s", job_id)
    
    # Check initial status
    job_initial = job_manager.get_job(job_id)
    logger.info("2. Initial job state: %s", job_initial)
    assert job_initial["status"] == "queued"
    
    # 3. Process job
    logger.info("3. Executing _process_job(job_id)...")
    await job_manager._process_job(job_id)
    
    # 4. Check final status
    job_final = job_manager.get_job(job_id)
    logger.info("4. Final job state: %s", job_final)
    
    assert job_final["status"] == "completed", f"Job failed with status={job_final['status']}, error={job_final.get('error_message')}"
    assert job_final.get("result_path"), "Job completed but result_path is missing"
    assert Path(job_final["result_path"]).exists() or Path(f"backend/{job_final['result_path']}").exists() or Path(f"./{job_final['result_path']}").exists(), f"Output file does not exist on disk: {job_final['result_path']}"
    
    print("\n" + "=" * 60)
    print("SUCCESS: Pipeline completed successfully!")
    print(f"Status: {job_final['status']}")
    print(f"Video path: {job_final['result_path']}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_end_to_end())
