"""
autonomous/state_manager.py - Persistent State Manager & Finite State Machine.
Manages persistent episode records in SQLite, validates state transitions,
and tracks pipeline stages, errors, retries, and artifacts across system restarts.
"""

import os
import logging
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from autonomous.config import autonomous_settings, EPISODES_DIR

logger = logging.getLogger("autonomous.state_manager")


class EpisodeState(str, Enum):
    """Explicit Lifecycle States for Autonomous Episodes."""
    CREATED = "CREATED"
    TOPIC_SELECTED = "TOPIC_SELECTED"
    RESEARCHING = "RESEARCHING"
    RESEARCH_COMPLETE = "RESEARCH_COMPLETE"
    RESEARCH_EXPANDING = "RESEARCH_EXPANDING"
    VERIFYING = "VERIFYING"
    VERIFIED = "VERIFIED"
    SCRIPTING = "SCRIPTING"
    SCRIPT_VALIDATED = "SCRIPT_VALIDATED"
    VISUAL_PLANNING = "VISUAL_PLANNING"
    GENERATING_VISUALS = "GENERATING_VISUALS"
    STATIC_VISUALS_READY = "STATIC_VISUALS_READY"
    GENERATING_MOTION = "GENERATING_MOTION"
    MOTION_READY = "MOTION_READY"
    GENERATING_AUDIO = "GENERATING_AUDIO"
    AUDIO_READY = "AUDIO_READY"
    MASTERING = "MASTERING"
    MASTER_READY = "MASTER_READY"
    METADATA_GENERATING = "METADATA_GENERATING"
    METADATA_READY = "METADATA_READY"
    THUMBNAIL_GENERATING = "THUMBNAIL_GENERATING"
    THUMBNAIL_READY = "THUMBNAIL_READY"
    PUBLISH_PACKAGE_READY = "PUBLISH_PACKAGE_READY"
    # Phase 12: Autonomous YouTube Publication Architecture (Mock / Offline-first Gate)
    YOUTUBE_VALIDATING = "YOUTUBE_VALIDATING"
    YOUTUBE_READY = "YOUTUBE_READY"
    YOUTUBE_UPLOADING = "YOUTUBE_UPLOADING"
    YOUTUBE_UPLOADED = "YOUTUBE_UPLOADED"
    THUMBNAIL_UPLOADING = "THUMBNAIL_UPLOADING"
    THUMBNAIL_UPLOADED = "THUMBNAIL_UPLOADED"
    PUBLICATION_VERIFIED = "PUBLICATION_VERIFIED"
    PUBLICATION_REVIEW_REQUIRED = "PUBLICATION_REVIEW_REQUIRED"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    RENDERING = "RENDERING"
    QC = "QC"
    SEO = "SEO"
    READY_TO_PUBLISH = "READY_TO_PUBLISH"
    UPLOADING = "UPLOADING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    CANCELLED = "CANCELLED"


# Valid State Transitions Table
VALID_TRANSITIONS: Dict[EpisodeState, List[EpisodeState]] = {
    EpisodeState.CREATED: [
        EpisodeState.TOPIC_SELECTED,
        EpisodeState.CANCELLED,
    ],
    EpisodeState.TOPIC_SELECTED: [
        EpisodeState.RESEARCHING,
        EpisodeState.CANCELLED,
    ],
    EpisodeState.RESEARCHING: [
        EpisodeState.RESEARCH_COMPLETE,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.RESEARCH_EXPANDING: [
        EpisodeState.RESEARCH_COMPLETE,
        EpisodeState.REVIEW_REQUIRED,
        EpisodeState.FAILED,
    ],
    EpisodeState.RESEARCH_COMPLETE: [
        EpisodeState.VERIFYING,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.VERIFYING: [
        EpisodeState.VERIFIED,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.VERIFIED: [
        EpisodeState.SCRIPTING,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.SCRIPTING: [
        EpisodeState.SCRIPT_VALIDATED,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.SCRIPT_VALIDATED: [
        EpisodeState.VISUAL_PLANNING,
        EpisodeState.SCRIPTING,  # Re-script loop if validation rejected
        EpisodeState.VERIFYING,  # Re-verification / quality hardening pass
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.VISUAL_PLANNING: [
        EpisodeState.GENERATING_VISUALS,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.GENERATING_VISUALS: [
        EpisodeState.STATIC_VISUALS_READY,
        EpisodeState.GENERATING_MOTION,
        EpisodeState.VISUAL_PLANNING,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.STATIC_VISUALS_READY: [
        EpisodeState.GENERATING_MOTION,
        EpisodeState.GENERATING_VISUALS,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.GENERATING_MOTION: [
        EpisodeState.MOTION_READY,
        EpisodeState.GENERATING_AUDIO,
        EpisodeState.GENERATING_MOTION,  # Retry with reduced motion
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.MOTION_READY: [
        EpisodeState.GENERATING_AUDIO,
        EpisodeState.GENERATING_MOTION,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.GENERATING_AUDIO: [
        EpisodeState.AUDIO_READY,
        EpisodeState.RENDERING,  # Preserved for backward compatibility with early phase test harnesses
        EpisodeState.GENERATING_AUDIO,  # Retry TTS
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.AUDIO_READY: [
        EpisodeState.MASTERING,
        EpisodeState.RENDERING,
        EpisodeState.GENERATING_AUDIO,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.MASTERING: [
        EpisodeState.MASTER_READY,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.MASTER_READY: [
        EpisodeState.METADATA_GENERATING,
        EpisodeState.MASTERING,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.METADATA_GENERATING: [
        EpisodeState.METADATA_READY,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.METADATA_READY: [
        EpisodeState.THUMBNAIL_GENERATING,
        EpisodeState.METADATA_GENERATING,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.THUMBNAIL_GENERATING: [
        EpisodeState.THUMBNAIL_READY,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.THUMBNAIL_READY: [
        EpisodeState.PUBLISH_PACKAGE_READY,
        EpisodeState.THUMBNAIL_GENERATING,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.PUBLISH_PACKAGE_READY: [],  # Preserves Phase 11 terminal invariant; consumed via PHASE_CONSUMPTION_TRANSITIONS
    # Phase 12: Publication Architecture Transitions
    EpisodeState.YOUTUBE_VALIDATING: [
        EpisodeState.YOUTUBE_READY,
        EpisodeState.PUBLICATION_REVIEW_REQUIRED,
        EpisodeState.FAILED,
    ],
    EpisodeState.YOUTUBE_READY: [
        EpisodeState.YOUTUBE_UPLOADING,
        EpisodeState.PUBLICATION_REVIEW_REQUIRED,
        EpisodeState.FAILED,
    ],
    EpisodeState.YOUTUBE_UPLOADING: [
        EpisodeState.YOUTUBE_UPLOADED,
        EpisodeState.UPLOAD_FAILED,
        EpisodeState.PUBLICATION_REVIEW_REQUIRED,
        EpisodeState.FAILED,
    ],
    EpisodeState.YOUTUBE_UPLOADED: [
        EpisodeState.THUMBNAIL_UPLOADING,
        EpisodeState.PUBLICATION_REVIEW_REQUIRED,
        EpisodeState.FAILED,
    ],
    EpisodeState.THUMBNAIL_UPLOADING: [
        EpisodeState.THUMBNAIL_UPLOADED,
        EpisodeState.UPLOAD_FAILED,
        EpisodeState.PUBLICATION_REVIEW_REQUIRED,
        EpisodeState.FAILED,
    ],
    EpisodeState.THUMBNAIL_UPLOADED: [
        EpisodeState.PUBLICATION_VERIFIED,
        EpisodeState.PUBLICATION_REVIEW_REQUIRED,
        EpisodeState.FAILED,
    ],
    EpisodeState.PUBLICATION_VERIFIED: [],  # Terminal for Phase 12 (Mock verification)
    EpisodeState.PUBLICATION_REVIEW_REQUIRED: [],
    EpisodeState.UPLOAD_FAILED: [
        EpisodeState.YOUTUBE_UPLOADING,
        EpisodeState.THUMBNAIL_UPLOADING,
        EpisodeState.YOUTUBE_VALIDATING,
        EpisodeState.FAILED,
        EpisodeState.PUBLICATION_REVIEW_REQUIRED,
    ],
    EpisodeState.RENDERING: [
        EpisodeState.QC,
        EpisodeState.MASTER_READY,
        EpisodeState.RENDERING,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.QC: [
        EpisodeState.SEO,
        EpisodeState.MASTER_READY,
        EpisodeState.RENDERING,  # Re-render if QC failed recoverable visual
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.SEO: [
        EpisodeState.READY_TO_PUBLISH,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.READY_TO_PUBLISH: [
        EpisodeState.UPLOADING,
        EpisodeState.REVIEW_REQUIRED,
        EpisodeState.CANCELLED,
    ],
    EpisodeState.UPLOADING: [
        EpisodeState.PUBLISHED,
        EpisodeState.FAILED,
        EpisodeState.REVIEW_REQUIRED,
    ],
    EpisodeState.FAILED: [
        EpisodeState.TOPIC_SELECTED,
        EpisodeState.RESEARCHING,
        EpisodeState.SCRIPTING,
        EpisodeState.GENERATING_VISUALS,
        EpisodeState.GENERATING_MOTION,
        EpisodeState.GENERATING_AUDIO,
        EpisodeState.MASTERING,
        EpisodeState.METADATA_GENERATING,
        EpisodeState.THUMBNAIL_GENERATING,
        EpisodeState.YOUTUBE_VALIDATING,
        EpisodeState.YOUTUBE_UPLOADING,
        EpisodeState.THUMBNAIL_UPLOADING,
        EpisodeState.RENDERING,
        EpisodeState.UPLOADING,
        EpisodeState.CANCELLED,
    ],
    EpisodeState.REVIEW_REQUIRED: [
        EpisodeState.RESEARCH_EXPANDING,
        EpisodeState.VERIFYING,
        EpisodeState.TOPIC_SELECTED,
        EpisodeState.CANCELLED,
    ],
    EpisodeState.PUBLISHED: [],
    EpisodeState.CANCELLED: [],
}

# Phase Boundary Consumption Transitions:
# Preserves Phase 11 terminal state invariant (VALID_TRANSITIONS[PUBLISH_PACKAGE_READY] == [])
# while permitting Phase 12 pipeline to consume PUBLISH_PACKAGE_READY -> YOUTUBE_VALIDATING.
PHASE_CONSUMPTION_TRANSITIONS: Dict[EpisodeState, List[EpisodeState]] = {
    EpisodeState.PUBLISH_PACKAGE_READY: [EpisodeState.YOUTUBE_VALIDATING],
}

TERMINAL_STATES = {EpisodeState.PUBLISHED, EpisodeState.CANCELLED, EpisodeState.PUBLICATION_VERIFIED}
UNFINISHED_STATES = {
    s for s in EpisodeState if s not in TERMINAL_STATES and s != EpisodeState.FAILED
}

StateBase = declarative_base()


class AutonomousEpisode(StateBase):
    """Database model for an autonomous episode record."""
    __tablename__ = "autonomous_episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(64), unique=True, index=True, nullable=False)
    topic = Column(String(500), nullable=False)
    category = Column(String(200), default="Ancient Tamil history", nullable=False)
    status = Column(String(50), default=EpisodeState.CREATED.value, nullable=False)
    current_stage = Column(String(50), default="CREATED", nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    retry_count = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)

    output_directory = Column(String(1000), nullable=False)
    script_path = Column(String(1000), nullable=True)
    video_path = Column(String(1000), nullable=True)
    thumbnail_path = Column(String(1000), nullable=True)

    youtube_video_id = Column(String(100), nullable=True)
    youtube_status = Column(String(50), default="NOT_UPLOADED", nullable=False)  # NOT_UPLOADED, UPLOADING, UPLOADED, FAILED

    research_confidence = Column(Float, nullable=True)
    script_validation_status = Column(String(50), nullable=True)
    motion_qc_status = Column(String(50), nullable=True)
    video_qc_status = Column(String(50), nullable=True)
    audio_qc_status = Column(String(50), nullable=True)
    metadata_path = Column(String(1000), nullable=True)

    extra_data = Column(JSON, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a dictionary representation."""
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "topic": self.topic,
            "category": self.category,
            "status": self.status,
            "current_stage": self.current_stage,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "output_directory": self.output_directory,
            "script_path": self.script_path,
            "video_path": self.video_path,
            "thumbnail_path": self.thumbnail_path,
            "youtube_video_id": self.youtube_video_id,
            "youtube_status": self.youtube_status,
            "research_confidence": self.research_confidence,
            "script_validation_status": self.script_validation_status,
            "motion_qc_status": self.motion_qc_status,
            "video_qc_status": self.video_qc_status,
            "audio_qc_status": self.audio_qc_status,
            "metadata_path": self.metadata_path,
            "extra_data": self.extra_data or {},
        }


class AutonomousResearchSource(StateBase):
    """Database model for an authoritative research source attached to an episode."""
    __tablename__ = "autonomous_research_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(64), ForeignKey("autonomous_episodes.episode_id"), index=True, nullable=False)
    source_id = Column(String(64), unique=True, index=True, nullable=False)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=True)
    publisher = Column(String(255), nullable=True)
    domain = Column(String(255), nullable=True)
    author = Column(String(255), nullable=True)
    publication_date = Column(String(100), nullable=True)
    retrieval_timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    source_type = Column(String(50), default="web", nullable=False)
    credibility_tier = Column(Integer, default=3, nullable=False)  # Tier 1 to 5
    content_hash = Column(String(64), index=True, nullable=False)
    language = Column(String(20), default="en", nullable=False)
    extraction_status = Column(String(50), default="EXTRACTED", nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    extra_metadata = Column(JSON, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "publisher": self.publisher,
            "domain": self.domain,
            "author": self.author,
            "publication_date": self.publication_date,
            "retrieval_timestamp": self.retrieval_timestamp.isoformat() if self.retrieval_timestamp else None,
            "source_type": self.source_type,
            "credibility_tier": self.credibility_tier,
            "content_hash": self.content_hash,
            "language": self.language,
            "extraction_status": self.extraction_status,
            "chunk_count": self.chunk_count,
            "extra_metadata": self.extra_metadata or {},
        }


class AutonomousEpisodeClaim(StateBase):
    """Database model for a verified factual claim attached to an episode."""
    __tablename__ = "autonomous_episode_claims"

    id = Column(Integer, primary_key=True, autoincrement=True)
    episode_id = Column(String(64), ForeignKey("autonomous_episodes.episode_id"), index=True, nullable=False)
    claim_id = Column(String(64), index=True, nullable=False)
    statement = Column(Text, nullable=False)
    normalized_statement = Column(Text, nullable=False)
    claim_type = Column(String(64), default="HISTORICAL_FACT", nullable=False)
    importance = Column(String(32), default="MEDIUM", nullable=False)
    confidence_score = Column(Float, default=0.0, nullable=False)
    classification = Column(String(64), default="UNVERIFIED_CLAIM", nullable=False)
    epistemic_status = Column(String(64), default="UNCERTAIN", nullable=False)
    epistemic_constraint = Column(String(64), default="FORBIDDEN", nullable=False)
    extra_data = Column(JSON, default=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "claim_id": self.claim_id,
            "statement": self.statement,
            "normalized_statement": self.normalized_statement,
            "claim_type": self.claim_type,
            "importance": self.importance,
            "confidence_score": self.confidence_score,
            "classification": self.classification,
            "epistemic_status": self.epistemic_status,
            "epistemic_constraint": self.epistemic_constraint,
            "extra_data": self.extra_data or {},
        }


class StateManager:
    """Manages persistent episode state and validates stage transitions."""

    def __init__(self, db_url: Optional[str] = None):
        if db_url is None:
            # Use universe_platform.db configured in autonomous_settings
            db_path = autonomous_settings.db_path
            db_url = f"sqlite:///{db_path}"

        self.db_url = db_url
        is_sqlite = db_url.startswith("sqlite")
        connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}

        self.engine = create_engine(db_url, connect_args=connect_args, echo=False)

        if is_sqlite:
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()

        # Create autonomous_episodes table non-destructively
        StateBase.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(autocommit=False, autoflush=False, expire_on_commit=False, bind=self.engine)
        logger.info(f"StateManager initialized on database: {db_url}")

    def _get_session(self) -> Session:
        return self.Session()

    def generate_episode_id(self) -> str:
        """Generate a sequential, human-readable episode ID for today (YYYYMMDD-###)."""
        now = datetime.now(timezone.utc)
        date_prefix = now.strftime("%Y%m%d")

        session = self._get_session()
        try:
            today_count = session.query(AutonomousEpisode).filter(
                AutonomousEpisode.episode_id.like(f"{date_prefix}-%")
            ).count()
            return f"{date_prefix}-{today_count + 1:03d}"
        finally:
            session.close()

    def create_episode(
        self,
        topic: str,
        category: str = "Ancient Tamil history",
        episode_id: Optional[str] = None,
        output_directory: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> AutonomousEpisode:
        """Create a new autonomous episode in CREATED state."""
        session = self._get_session()
        try:
            if not episode_id:
                episode_id = self.generate_episode_id()

            # Verify uniqueness
            existing = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if existing:
                raise ValueError(f"Episode with ID '{episode_id}' already exists.")

            if output_directory:
                ep_dir = Path(output_directory)
            else:
                ep_dir = EPISODES_DIR / episode_id
            ep_dir.mkdir(parents=True, exist_ok=True)

            now = datetime.now(timezone.utc)
            episode = AutonomousEpisode(
                episode_id=episode_id,
                topic=topic.strip(),
                category=category.strip(),
                status=EpisodeState.CREATED.value,
                current_stage="CREATED",
                created_at=now,
                updated_at=now,
                started_at=now,
                output_directory=str(ep_dir),
                extra_data=extra_data or {},
            )
            session.add(episode)
            session.commit()
            session.refresh(episode)
            logger.info(f"Created episode {episode_id}: '{topic}' ({category})")
            return episode
        finally:
            session.close()

    def transition(
        self,
        episode_id: str,
        new_state: EpisodeState,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> AutonomousEpisode:
        """
        Validate and persist a controlled state transition for an episode.
        Raises ValueError if transition is invalid according to the state machine.
        """
        session = self._get_session()
        try:
            episode = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if not episode:
                raise KeyError(f"Episode '{episode_id}' not found.")

            current_state = EpisodeState(episode.status)

            # Idempotent transition: If already in requested state, do not corrupt state or raise error
            if current_state == new_state:
                logger.info(f"Episode {episode_id} is already in state {new_state.value} (idempotent transition).")
                if error_message is not None:
                    episode.error_message = error_message
                if current_stage is not None:
                    episode.current_stage = current_stage
                session.commit()
                session.refresh(episode)
                return episode

            # Validate transition (incorporating phase consumption boundaries)
            allowed_next = list(VALID_TRANSITIONS.get(current_state, []))
            if current_state in PHASE_CONSUMPTION_TRANSITIONS:
                allowed_next.extend(PHASE_CONSUMPTION_TRANSITIONS[current_state])
            if new_state not in allowed_next:
                raise ValueError(
                    f"Invalid state transition: Cannot transition episode {episode_id} "
                    f"from {current_state.value} to {new_state.value}. "
                    f"Allowed transitions from {current_state.value}: {[s.value for s in allowed_next]}"
                )

            # Update state
            now = datetime.now(timezone.utc)
            episode.status = new_state.value
            episode.current_stage = current_stage or new_state.value
            episode.updated_at = now

            if error_message is not None:
                episode.error_message = error_message

            if new_state == EpisodeState.PUBLISHED:
                episode.completed_at = now
                episode.youtube_status = "UPLOADED"
            elif new_state == EpisodeState.PUBLICATION_VERIFIED:
                episode.completed_at = now
                episode.youtube_status = "MOCK_VERIFIED"

            session.commit()
            session.refresh(episode)
            logger.info(f"Episode {episode_id} transitioned: {current_state.value} -> {new_state.value}")
            return episode
        finally:
            session.close()

    def get_episode(self, episode_id: str) -> Optional[AutonomousEpisode]:
        """Fetch an episode record by ID."""
        session = self._get_session()
        try:
            ep = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if ep is not None:
                session.expunge(ep)
            return ep
        finally:
            session.close()

    def get_active_episode(self) -> Optional[AutonomousEpisode]:
        """
        Get the most recent active unfinished episode, if one exists.
        Returns None if all episodes are in terminal or failed states.
        """
        session = self._get_session()
        try:
            unfinished_values = [s.value for s in UNFINISHED_STATES]
            return session.query(AutonomousEpisode).filter(
                AutonomousEpisode.status.in_(unfinished_values)
            ).order_by(AutonomousEpisode.updated_at.desc()).first()
        finally:
            session.close()

    def get_unfinished_episodes(self) -> List[AutonomousEpisode]:
        """Get all currently unfinished episodes ordered by creation date."""
        session = self._get_session()
        try:
            unfinished_values = [s.value for s in UNFINISHED_STATES]
            return session.query(AutonomousEpisode).filter(
                AutonomousEpisode.status.in_(unfinished_values)
            ).order_by(AutonomousEpisode.created_at.asc()).all()
        finally:
            session.close()

    def get_published_episodes(self) -> List[AutonomousEpisode]:
        """Get all published episodes."""
        session = self._get_session()
        try:
            return session.query(AutonomousEpisode).filter_by(
                status=EpisodeState.PUBLISHED.value
            ).order_by(AutonomousEpisode.completed_at.desc()).all()
        finally:
            session.close()

    def record_error(self, episode_id: str, error: str, increment_retry: bool = True) -> AutonomousEpisode:
        """Record an error message and optionally increment the retry count."""
        session = self._get_session()
        try:
            episode = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if not episode:
                raise KeyError(f"Episode '{episode_id}' not found.")

            episode.error_message = error
            if increment_retry:
                episode.retry_count += 1
            episode.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(episode)
            return episode
        finally:
            session.close()

    def update_artifacts(self, episode_id: str, **kwargs) -> AutonomousEpisode:
        """Update artifact paths (script_path, video_path, thumbnail_path, etc.)."""
        session = self._get_session()
        try:
            episode = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if not episode:
                raise KeyError(f"Episode '{episode_id}' not found.")

            for k, v in kwargs.items():
                if hasattr(episode, k):
                    setattr(episode, k, v)
                else:
                    logger.warning(f"Unknown attribute '{k}' on AutonomousEpisode.")

            episode.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(episode)
            return episode
        finally:
            session.close()

    # Alias for update_artifacts
    update_episode = update_artifacts

    def get_system_summary(self) -> Dict[str, Any]:
        """Return high-level summary counts across all states."""
        session = self._get_session()
        try:
            total = session.query(AutonomousEpisode).count()
            published = session.query(AutonomousEpisode).filter_by(status=EpisodeState.PUBLISHED.value).count()
            failed = session.query(AutonomousEpisode).filter_by(status=EpisodeState.FAILED.value).count()
            review = session.query(AutonomousEpisode).filter_by(status=EpisodeState.REVIEW_REQUIRED.value).count()
            active = self.get_active_episode()

            return {
                "total_episodes": total,
                "published_episodes": published,
                "failed_episodes": failed,
                "review_required_episodes": review,
                "active_episode": active.to_dict() if active else None,
            }
        finally:
            session.close()

    def record_research_source(self, episode_id: str, source_data: Dict[str, Any]) -> AutonomousResearchSource:
        """Record or update a research source for an episode."""
        session = self._get_session()
        try:
            source_id = source_data.get("source_id")
            existing = session.query(AutonomousResearchSource).filter_by(source_id=source_id).first()
            if existing:
                for k, v in source_data.items():
                    if hasattr(existing, k) and k != "id":
                        if k == "retrieval_timestamp" and isinstance(v, str):
                            try:
                                v = datetime.fromisoformat(v)
                            except Exception:
                                continue
                        elif k == "extra_metadata" and isinstance(v, dict):
                            existing_extra = dict(existing.extra_metadata or {})
                            existing_extra.update(v)
                            v = existing_extra
                        setattr(existing, k, v)
                extra = dict(existing.extra_metadata or {})
                extra["source_origin"] = source_data.get("source_origin", extra.get("source_origin", "external"))
                extra["independence_group"] = source_data.get("independence_group", extra.get("independence_group", source_data.get("domain") or ""))
                extra["source_role"] = source_data.get("source_role", extra.get("source_role", "primary_evidence"))
                extra["evidence_weight_class"] = source_data.get("evidence_weight_class", extra.get("evidence_weight_class", "PRIMARY"))
                if source_data.get("raw_cache_path"):
                    extra["raw_cache_path"] = source_data.get("raw_cache_path")
                existing.extra_metadata = extra
                session.commit()
                session.refresh(existing)
                return existing

            extra = dict(source_data.get("extra_metadata", {}))
            extra["source_origin"] = source_data.get("source_origin", "external")
            extra["independence_group"] = source_data.get("independence_group", source_data.get("domain") or "")
            extra["source_role"] = source_data.get("source_role", "primary_evidence")
            extra["evidence_weight_class"] = source_data.get("evidence_weight_class", "PRIMARY")
            if source_data.get("raw_cache_path"):
                extra["raw_cache_path"] = source_data.get("raw_cache_path")

            ret_ts = source_data.get("retrieval_timestamp")
            if isinstance(ret_ts, str):
                try:
                    ret_ts = datetime.fromisoformat(ret_ts)
                except Exception:
                    ret_ts = datetime.now(timezone.utc)
            elif not isinstance(ret_ts, datetime):
                ret_ts = datetime.now(timezone.utc)

            source = AutonomousResearchSource(
                episode_id=episode_id,
                source_id=source_data["source_id"],
                title=source_data.get("title", "Untitled Source"),
                url=source_data.get("url"),
                publisher=source_data.get("publisher"),
                domain=source_data.get("domain"),
                author=source_data.get("author"),
                publication_date=source_data.get("publication_date"),
                retrieval_timestamp=ret_ts,
                source_type=source_data.get("source_type", "web"),
                credibility_tier=source_data.get("credibility_tier", 3),
                content_hash=source_data.get("content_hash", ""),
                language=source_data.get("language", "en"),
                extraction_status=source_data.get("extraction_status", "EXTRACTED"),
                chunk_count=source_data.get("chunk_count", 0),
                extra_metadata=extra,
            )
            session.add(source)
            session.commit()
            session.refresh(source)
            return source
        finally:
            session.close()

    def get_episode_sources(self, episode_id: str) -> List[AutonomousResearchSource]:
        """Retrieve all recorded sources for an episode."""
        session = self._get_session()
        try:
            return session.query(AutonomousResearchSource).filter_by(episode_id=episode_id).all()
        finally:
            session.close()

    def get_source_by_hash(self, content_hash: str) -> Optional[AutonomousResearchSource]:
        """Find an already retrieved source by its content hash across any episode."""
        session = self._get_session()
        try:
            return session.query(AutonomousResearchSource).filter_by(content_hash=content_hash).first()
        finally:
            session.close()

    def get_source_by_url(self, url: str) -> Optional[AutonomousResearchSource]:
        """Find an already retrieved source by its URL."""
        session = self._get_session()
        try:
            return session.query(AutonomousResearchSource).filter_by(url=url).first()
        finally:
            session.close()

    def update_research_confidence(self, episode_id: str, confidence: float) -> AutonomousEpisode:
        """Update research confidence score on the episode."""
        return self.update_artifacts(episode_id, research_confidence=confidence)

    def transition_state(self, episode_id: str, target_state: Optional[EpisodeState] = None, stage_name: str = "", new_state: Optional[EpisodeState] = None, **kwargs) -> AutonomousEpisode:
        """Alias for transition() for explicit state stage transitions."""
        st = target_state or new_state
        return self.transition(episode_id=episode_id, new_state=st, current_stage=stage_name, **kwargs)

    def record_failure(self, episode_id: str, error_message: str) -> AutonomousEpisode:
        """Record an error message on the episode."""
        return self.update_artifacts(episode_id, error_message=error_message)

    def record_verified_claims(self, episode_id: str, claims: List[Any]) -> None:
        """Persist a list of VerifiedClaim objects into autonomous_episode_claims."""
        session = self._get_session()
        try:
            session.query(AutonomousEpisodeClaim).filter_by(episode_id=episode_id).delete()
            for c in claims:
                c_dict = c.to_dict() if hasattr(c, "to_dict") else dict(c)
                record = AutonomousEpisodeClaim(
                    episode_id=episode_id,
                    claim_id=c_dict["claim_id"],
                    statement=c_dict["statement"],
                    normalized_statement=c_dict.get("normalized_statement", c_dict["statement"]),
                    claim_type=c_dict.get("claim_type", "HISTORICAL_FACT"),
                    importance=c_dict.get("importance", "MEDIUM"),
                    confidence_score=float(c_dict.get("confidence_score", 0.0)),
                    classification=c_dict.get("classification", "UNVERIFIED_CLAIM"),
                    epistemic_status=c_dict.get("epistemic_status", "UNCERTAIN"),
                    epistemic_constraint=c_dict.get("epistemic_constraint", "FORBIDDEN"),
                    extra_data={
                        "breakdown": c_dict.get("confidence_breakdown", {}),
                        "supporting_chunk_ids": c_dict.get("supporting_chunk_ids", []),
                        "independence_groups": c_dict.get("independence_groups", []),
                        "extracted_entities": c_dict.get("extracted_entities", []),
                        "extracted_dates": c_dict.get("extracted_dates", []),
                        "extracted_numbers": c_dict.get("extracted_numbers", []),
                    }
                )
                session.add(record)
            session.commit()
            logger.info(f"Persisted {len(claims)} verified claims to autonomous_episode_claims for episode '{episode_id}'")
        finally:
            session.close()

    def get_verified_claims(self, episode_id: str) -> List[AutonomousEpisodeClaim]:
        """Retrieve all recorded claims for an episode."""
        session = self._get_session()
        try:
            return session.query(AutonomousEpisodeClaim).filter_by(episode_id=episode_id).all()
        finally:
            session.close()



