import uvicorn
from fastapi import FastAPI
from app.core.config import settings
from app.core.database import engine, Base
from app.api.generation import router as generation_router
from app.engines.job_manager import job_manager
# Import all models so Base.metadata is fully populated before create_all()
from app.models import job as _job_model  # noqa: F401
from app.models import story as _story_model  # noqa: F401

# Initialize FastAPI app
app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# Include generation router
app.include_router(generation_router, prefix=settings.API_V1_STR)

# Create DB tables on startup
@app.on_event("startup")
async def on_startup():
    Base.metadata.create_all(bind=engine)
    # Start background job worker
    import asyncio
    asyncio.create_task(job_manager.start_worker())

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
