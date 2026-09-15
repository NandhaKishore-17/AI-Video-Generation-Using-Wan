"""
autonomous/youtube_client.py - YouTube Client Abstraction for Kaalapadhivugal Production Pipeline.

Provides a unified interface for YouTube publication with two implementations:
1. MockYouTubeClient: Strict offline implementation using deterministic fake resources.
   Guarantees zero network calls, zero OAuth flows, and clear provider="mock" provenance.
2. RealYouTubeClient: Dormant YouTube Data API v3 client architecture with a hard runtime
   safety lock (YouTubeIntegrationDisabledError) whenever YOUTUBE_INTEGRATION_ENABLED=false.
"""

import os
import re
import time
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from autonomous.config import autonomous_settings

logger = logging.getLogger("autonomous.youtube_client")


# ==============================================================================
# Error Hierarchy & Classifications
# ==============================================================================

class ErrorClassification(str, Enum):
    """Classification of publication failure modes."""
    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    SAFE_CONFIGURATION_BLOCK = "SAFE_CONFIGURATION_BLOCK"


class YouTubeClientError(Exception):
    """Base exception for all YouTube client errors."""
    classification: ErrorClassification = ErrorClassification.NON_RETRYABLE

    def __init__(self, message: str, classification: ErrorClassification = ErrorClassification.NON_RETRYABLE):
        super().__init__(message)
        self.classification = classification


class YouTubeIntegrationDisabledError(YouTubeClientError):
    """Raised when any real API call is attempted while integration is disabled."""
    def __init__(self, message: str = "YouTube integration is DISABLED (YOUTUBE_INTEGRATION_ENABLED=false). Real API access refused."):
        super().__init__(message, classification=ErrorClassification.SAFE_CONFIGURATION_BLOCK)


class YouTubeAuthenticationError(YouTubeClientError):
    """Raised when authentication credentials are missing or invalid."""
    def __init__(self, message: str):
        super().__init__(message, classification=ErrorClassification.REVIEW_REQUIRED)


class YouTubeUploadError(YouTubeClientError):
    """Raised when video or thumbnail upload fails."""
    def __init__(self, message: str, retryable: bool = False):
        classification = ErrorClassification.RETRYABLE if retryable else ErrorClassification.NON_RETRYABLE
        super().__init__(message, classification=classification)


class YouTubeDuplicateError(YouTubeClientError):
    """Raised when duplicate resource creation is blocked."""
    def __init__(self, message: str):
        super().__init__(message, classification=ErrorClassification.NON_RETRYABLE)


class YouTubeVerificationError(YouTubeClientError):
    """Raised when published resource verification fails."""
    def __init__(self, message: str):
        super().__init__(message, classification=ErrorClassification.REVIEW_REQUIRED)


# ==============================================================================
# Base Abstract Client
# ==============================================================================

class BaseYouTubeClient(ABC):
    """Abstract interface for YouTube publication clients."""

    @abstractmethod
    def authenticate(self) -> Dict[str, Any]:
        """Verify or establish authentication."""
        pass

    @abstractmethod
    def upload_video(
        self,
        episode_id: str,
        video_path: Path,
        title: str,
        description: str,
        tags: List[str],
        category_id: str = "27",
        privacy_status: str = "private",
        made_for_kids: bool = False,
    ) -> Dict[str, Any]:
        """Upload video file and metadata."""
        pass

    @abstractmethod
    def upload_thumbnail(self, video_id: str, thumbnail_path: Path) -> Dict[str, Any]:
        """Upload thumbnail for existing video resource."""
        pass

    @abstractmethod
    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Fetch video resource metadata."""
        pass

    @abstractmethod
    def verify_video(
        self,
        video_id: str,
        expected_title: Optional[str] = None,
        expected_privacy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify uploaded video resource exists and matches expectation."""
        pass

    @abstractmethod
    def verify_thumbnail(self, video_id: str) -> Dict[str, Any]:
        """Verify thumbnail resource exists for video."""
        pass

    @abstractmethod
    def check_connection(self) -> Dict[str, Any]:
        """Check client operational readiness."""
        pass


# ==============================================================================
# Mock YouTube Client (Strictly Offline & Deterministic)
# ==============================================================================

class MockYouTubeClient(BaseYouTubeClient):
    """
    In-memory mock YouTube client for Phase 12 verification.
    Guarantees 100% offline operation: does not import or call any networking library.
    Generates deterministic fake resources with explicit mock provenance.
    """

    def __init__(self, settings=None):
        self.settings = settings or autonomous_settings
        # In-memory store: {video_id: video_data}
        self._store: Dict[str, Dict[str, Any]] = {}
        # Simulation controls for test harnesses
        self.fail_next_upload: bool = False
        self.fail_next_thumbnail: bool = False
        self.fail_next_verification: bool = False
        self.transient_fail_count: int = 0
        self._call_log: List[Dict[str, Any]] = []

    def check_connection(self) -> Dict[str, Any]:
        return {
            "provider": "mock",
            "mock": True,
            "real_api_called": False,
            "status": "OPERATIONAL",
            "offline": True,
            "channel_id": None,
        }

    def authenticate(self) -> Dict[str, Any]:
        """Simulate successful offline mock authentication."""
        self._call_log.append({"op": "authenticate", "timestamp": datetime.now(timezone.utc).isoformat()})
        return {
            "provider": "mock",
            "mock": True,
            "real_api_called": False,
            "authenticated": True,
            "channel_id": None,
            "auth_type": "MOCK_LOCAL",
        }

    def upload_video(
        self,
        episode_id: str,
        video_path: Path,
        title: str,
        description: str,
        tags: List[str],
        category_id: str = "27",
        privacy_status: str = "MOCK_PRIVATE",
        made_for_kids: bool = False,
    ) -> Dict[str, Any]:
        """Simulate video upload, creating a deterministic fake video resource."""
        self._call_log.append({"op": "upload_video", "episode_id": episode_id})

        # Test failure injection
        if self.fail_next_upload:
            self.fail_next_upload = False
            raise YouTubeUploadError("Simulated mock video upload failure (transient network timeout)", retryable=True)

        if self.transient_fail_count > 0:
            self.transient_fail_count -= 1
            raise YouTubeUploadError(f"Simulated transient error ({self.transient_fail_count + 1} remaining)", retryable=True)

        # Enforce file existence locally
        if not video_path.exists():
            raise YouTubeUploadError(f"Video file not found at path: {video_path}", retryable=False)

        # Check duplicate protection: if episode already in store, block duplicate creation
        clean_ep_id = re.sub(r"[^A-Za-z0-9_]", "_", episode_id).upper()
        mock_id = f"MOCK_{clean_ep_id}_001"

        if mock_id in self._store:
            raise YouTubeDuplicateError(
                f"Duplicate mock upload blocked: Resource '{mock_id}' already exists for episode '{episode_id}'."
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        video_record = {
            "id": mock_id,
            "episode_id": episode_id,
            "provider": "mock",
            "mock": True,
            "real_api_called": False,
            "channel_id": None,
            "title": title,
            "description": description,
            "tags": tags,
            "category_id": category_id,
            "privacy_status": privacy_status,
            "made_for_kids": made_for_kids,
            "upload_timestamp": now_iso,
            "status": "UPLOADED",
            "thumbnail_uploaded": False,
            "thumbnail_path": None,
            "file_size": video_path.stat().st_size,
        }

        self._store[mock_id] = video_record
        logger.info(f"MockYouTubeClient: Created mock video resource '{mock_id}' for episode '{episode_id}'.")
        return video_record

    def upload_thumbnail(self, video_id: str, thumbnail_path: Path) -> Dict[str, Any]:
        """Simulate thumbnail upload for an existing mock video."""
        self._call_log.append({"op": "upload_thumbnail", "video_id": video_id})

        if self.fail_next_thumbnail:
            self.fail_next_thumbnail = False
            raise YouTubeUploadError("Simulated mock thumbnail upload failure", retryable=True)

        if video_id not in self._store:
            raise YouTubeUploadError(f"Cannot upload thumbnail: Video resource '{video_id}' does not exist in mock store.", retryable=False)

        if not thumbnail_path.exists():
            raise YouTubeUploadError(f"Thumbnail file not found at path: {thumbnail_path}", retryable=False)

        now_iso = datetime.now(timezone.utc).isoformat()
        self._store[video_id]["thumbnail_uploaded"] = True
        self._store[video_id]["thumbnail_path"] = str(thumbnail_path)
        self._store[video_id]["thumbnail_timestamp"] = now_iso

        logger.info(f"MockYouTubeClient: Attached thumbnail to mock video '{video_id}'.")
        return {
            "provider": "mock",
            "mock": True,
            "real_api_called": False,
            "video_id": video_id,
            "thumbnail_status": "UPLOADED",
            "timestamp": now_iso,
        }

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored mock video record."""
        self._call_log.append({"op": "get_video", "video_id": video_id})
        return self._store.get(video_id)

    def verify_video(
        self,
        video_id: str,
        expected_title: Optional[str] = None,
        expected_privacy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify that the mock video exists and satisfies expectations."""
        self._call_log.append({"op": "verify_video", "video_id": video_id})

        if self.fail_next_verification:
            self.fail_next_verification = False
            raise YouTubeVerificationError("Simulated verification failure.")

        video = self._store.get(video_id)
        if not video:
            raise YouTubeVerificationError(f"Verification failed: Mock video '{video_id}' not found.")

        if expected_title and video.get("title") != expected_title:
            raise YouTubeVerificationError(
                f"Verification failed: Title mismatch. Expected '{expected_title}', got '{video.get('title')}'."
            )

        if expected_privacy and video.get("privacy_status") != expected_privacy:
            raise YouTubeVerificationError(
                f"Verification failed: Privacy mismatch. Expected '{expected_privacy}', got '{video.get('privacy_status')}'."
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "provider": "mock",
            "mock": True,
            "real_api_called": False,
            "video_id": video_id,
            "verification_status": "VERIFIED",
            "title_matched": expected_title is None or video.get("title") == expected_title,
            "privacy_matched": expected_privacy is None or video.get("privacy_status") == expected_privacy,
            "timestamp": now_iso,
        }

    def verify_thumbnail(self, video_id: str) -> Dict[str, Any]:
        """Verify that thumbnail upload succeeded for the mock resource."""
        self._call_log.append({"op": "verify_thumbnail", "video_id": video_id})
        video = self._store.get(video_id)
        if not video:
            raise YouTubeVerificationError(f"Thumbnail verification failed: Mock video '{video_id}' not found.")

        if not video.get("thumbnail_uploaded", False):
            raise YouTubeVerificationError(f"Thumbnail verification failed: No thumbnail attached to '{video_id}'.")

        return {
            "provider": "mock",
            "mock": True,
            "real_api_called": False,
            "video_id": video_id,
            "thumbnail_verified": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ==============================================================================
# Real YouTube Client (Dormant with Hard Runtime Safety Lock)
# ==============================================================================

class RealYouTubeClient(BaseYouTubeClient):
    """
    Dormant client implementing the architecture for YouTube Data API v3.
    Contains a strict runtime safety lock: if YOUTUBE_INTEGRATION_ENABLED is False,
    all operations immediately raise YouTubeIntegrationDisabledError before making
    any network request or attempting authentication.
    """

    def __init__(self, settings=None):
        self.settings = settings or autonomous_settings
        self._check_safety_lock("initialization")

    def _check_safety_lock(self, operation: str) -> None:
        """Enforce the hard runtime safety lock."""
        if not getattr(self.settings, "youtube_integration_enabled", False):
            msg = (
                f"YouTube integration is DISABLED (YOUTUBE_INTEGRATION_ENABLED=false). "
                f"Attempted operation '{operation}' was blocked with ZERO network requests. "
                "Real Google/YouTube API calls are strictly forbidden in Phase 12."
            )
            logger.warning(f"SAFETY LOCK ACTIVATED: {msg}")
            raise YouTubeIntegrationDisabledError(msg)

    def check_connection(self) -> Dict[str, Any]:
        self._check_safety_lock("check_connection")
        raise NotImplementedError("Real YouTube Data API connector is dormant and not configured.")

    def authenticate(self) -> Dict[str, Any]:
        self._check_safety_lock("authenticate")
        raise NotImplementedError("Real YouTube OAuth flow is dormant and not configured.")

    def upload_video(
        self,
        episode_id: str,
        video_path: Path,
        title: str,
        description: str,
        tags: List[str],
        category_id: str = "27",
        privacy_status: str = "private",
        made_for_kids: bool = False,
    ) -> Dict[str, Any]:
        self._check_safety_lock("upload_video")
        raise NotImplementedError("Real YouTube video upload is dormant and not configured.")

    def upload_thumbnail(self, video_id: str, thumbnail_path: Path) -> Dict[str, Any]:
        self._check_safety_lock("upload_thumbnail")
        raise NotImplementedError("Real YouTube thumbnail upload is dormant and not configured.")

    def get_video(self, video_id: str) -> Optional[Dict[str, Any]]:
        self._check_safety_lock("get_video")
        raise NotImplementedError("Real YouTube get_video is dormant and not configured.")

    def verify_video(
        self,
        video_id: str,
        expected_title: Optional[str] = None,
        expected_privacy: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._check_safety_lock("verify_video")
        raise NotImplementedError("Real YouTube verify_video is dormant and not configured.")

    def verify_thumbnail(self, video_id: str) -> Dict[str, Any]:
        self._check_safety_lock("verify_thumbnail")
        raise NotImplementedError("Real YouTube verify_thumbnail is dormant and not configured.")


# ==============================================================================
# Client Factory
# ==============================================================================

def get_youtube_client(settings=None, force_mock: bool = True) -> BaseYouTubeClient:
    """
    Factory function for obtaining a YouTube client.
    By default, returns MockYouTubeClient.
    If force_mock is False, instantiates RealYouTubeClient (which triggers safety lock if disabled).
    """
    active_settings = settings or autonomous_settings
    if force_mock or not active_settings.youtube_integration_enabled:
        return MockYouTubeClient(settings=active_settings)
    return RealYouTubeClient(settings=active_settings)
