import enum
from sqlalchemy import Column, String, Float, DateTime, Enum as SAEnum, Text
from sqlalchemy.sql import func
from app.core.database import Base


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    status = Column(SAEnum(JobStatus), default=JobStatus.QUEUED, nullable=False)
    progress = Column(Float, default=0.0, nullable=False)
    result_path = Column(String, nullable=True)
    # Foreign key to the generated story (populated after LLM step)
    story_id = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
