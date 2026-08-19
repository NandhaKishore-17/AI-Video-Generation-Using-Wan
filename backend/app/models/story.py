"""
Story model — stores LLM-generated story JSON in the database.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, JSON
from app.core.database import Base


def _gen_uuid() -> str:
    return str(uuid.uuid4())


class Story(Base):
    __tablename__ = "stories"

    id = Column(String, primary_key=True, default=_gen_uuid)
    job_id = Column(String, nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    title = Column(String(512), nullable=True)
    summary = Column(Text, nullable=True)
    # Full story JSON stored as a single JSON blob
    story_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
