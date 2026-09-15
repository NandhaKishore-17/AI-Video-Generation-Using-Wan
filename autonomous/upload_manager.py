"""
autonomous/upload_manager.py - YouTube Video & Thumbnail Upload Coordinator.

Coordinates upload operations across Mock and Real clients:
1. Assembles upload payload directly from validated publish package (no re-generation)
2. Enforces duplicate protection: verifies existing resource if video ID is already known
3. Executes video upload with configured privacy mode (default: MOCK_PRIVATE / private)
4. Executes thumbnail upload independently, preserving video if thumbnail fails
5. Classifies errors into RETRYABLE, NON_RETRYABLE, or REVIEW_REQUIRED
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.youtube_client import (
    BaseYouTubeClient, MockYouTubeClient, get_youtube_client,
    YouTubeClientError, YouTubeUploadError, YouTubeDuplicateError,
    ErrorClassification
)

logger = logging.getLogger("autonomous.upload_manager")


@dataclass
class UploadResult:
    """Outcome of video and thumbnail upload operations."""
    success: bool
    episode_id: str
    video_id: Optional[str] = None
    video_uploaded: bool = False
    thumbnail_uploaded: bool = False
    error: Optional[str] = None
    error_classification: Optional[ErrorClassification] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    video_record: Dict[str, Any] = field(default_factory=dict)


class YouTubeUploadManager:
    """Manages video and thumbnail upload workflows."""

    def __init__(
        self,
        client: Optional[BaseYouTubeClient] = None,
        state_manager: Optional[StateManager] = None,
        settings=None
    ):
        self.settings = settings or autonomous_settings
        self.client = client or get_youtube_client(settings=self.settings, force_mock=True)
        self.state_manager = state_manager or StateManager()

    def prepare_payload(self, ep_dir: Path, package_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assemble the exact upload payload using ONLY validated package data.
        Does NOT re-generate SEO or call an LLM.
        """
        seo = package_data.get("seo", {})
        title = seo.get("selected_title", "")
        description = seo.get("description", "")
        tags = seo.get("tags", [])

        # Strict Made-For-Kids default from configuration
        made_for_kids = bool(self.settings.youtube_made_for_kids)

        # Mode determination
        privacy_mode = "MOCK_PRIVATE" if isinstance(self.client, MockYouTubeClient) else "private"

        category_id = str(seo.get("category_id", "27"))  # Default Education = 27

        master_path = ep_dir / package_data.get("master", {}).get("path", "master/master.mp4")
        thumbnail_path = ep_dir / package_data.get("thumbnail", {}).get("path", "publish/thumbnail.jpg")

        return {
            "title": title,
            "description": description,
            "tags": tags,
            "category_id": category_id,
            "privacy_status": privacy_mode,
            "made_for_kids": made_for_kids,
            "master_video_path": master_path,
            "thumbnail_path": thumbnail_path,
            "channel_id": None,  # Channel ID is unconfigured/unknown
        }

    def execute_upload(
        self,
        episode_id: str,
        ep_dir: Path,
        package_data: Dict[str, Any],
        existing_video_id: Optional[str] = None
    ) -> UploadResult:
        """
        Execute video and thumbnail upload with duplicate protection and safe error handling.
        """
        payload = self.prepare_payload(ep_dir, package_data)
        master_video_path = payload["master_video_path"]
        thumbnail_path = payload["thumbnail_path"]

        # Duplicate protection check: If existing video ID is already known
        video_id = existing_video_id
        video_record: Dict[str, Any] = {}

        if video_id:
            logger.info(f"Duplicate check: Existing video resource '{video_id}' found for episode '{episode_id}'.")
            existing_res = self.client.get_video(video_id)
            if existing_res:
                logger.info(f"Existing video resource '{video_id}' verified in store. Reusing without re-uploading.")
                video_record = existing_res
            else:
                logger.warning(f"Existing video ID '{video_id}' not found in client store. Will upload anew.")
                video_id = None

        # Step 1: Upload Video (if not already existing)
        if not video_id:
            try:
                logger.info(f"Uploading video for episode '{episode_id}' (Title: '{payload['title'][:50]}...')...")
                video_record = self.client.upload_video(
                    episode_id=episode_id,
                    video_path=master_video_path,
                    title=payload["title"],
                    description=payload["description"],
                    tags=payload["tags"],
                    category_id=payload["category_id"],
                    privacy_status=payload["privacy_status"],
                    made_for_kids=payload["made_for_kids"],
                )
                video_id = video_record.get("id")
            except YouTubeDuplicateError as e:
                logger.info(f"Duplicate blocked: {e}")
                # Extract ID and fetch existing
                clean_ep_id = episode_id.replace("-", "_").upper()
                fallback_id = f"MOCK_{clean_ep_id}_001"
                existing = self.client.get_video(fallback_id)
                if existing:
                    video_id = fallback_id
                    video_record = existing
                else:
                    return UploadResult(
                        success=False,
                        episode_id=episode_id,
                        error=str(e),
                        error_classification=ErrorClassification.NON_RETRYABLE,
                        payload=payload
                    )
            except YouTubeClientError as e:
                logger.error(f"Video upload failed for '{episode_id}': {e}")
                return UploadResult(
                    success=False,
                    episode_id=episode_id,
                    error=str(e),
                    error_classification=e.classification,
                    payload=payload
                )
            except Exception as e:
                logger.exception(f"Unexpected video upload error for '{episode_id}': {e}")
                return UploadResult(
                    success=False,
                    episode_id=episode_id,
                    error=str(e),
                    error_classification=ErrorClassification.NON_RETRYABLE,
                    payload=payload
                )

        # Step 2: Upload Thumbnail (Independent step: video is preserved if thumbnail fails)
        try:
            logger.info(f"Uploading thumbnail for video '{video_id}' from '{thumbnail_path}'...")
            thumb_res = self.client.upload_thumbnail(
                video_id=video_id,
                thumbnail_path=thumbnail_path
            )
            thumbnail_uploaded = True
        except YouTubeClientError as e:
            logger.error(f"Thumbnail upload failed for video '{video_id}': {e} (Video resource is PRESERVED).")
            return UploadResult(
                success=False,
                episode_id=episode_id,
                video_id=video_id,
                video_uploaded=True,
                thumbnail_uploaded=False,
                error=f"Thumbnail upload failed: {e}",
                error_classification=e.classification,
                payload=payload,
                video_record=video_record
            )
        except Exception as e:
            logger.exception(f"Unexpected thumbnail upload error for '{video_id}': {e}")
            return UploadResult(
                success=False,
                episode_id=episode_id,
                video_id=video_id,
                video_uploaded=True,
                thumbnail_uploaded=False,
                error=f"Unexpected thumbnail error: {e}",
                error_classification=ErrorClassification.NON_RETRYABLE,
                payload=payload,
                video_record=video_record
            )

        return UploadResult(
            success=True,
            episode_id=episode_id,
            video_id=video_id,
            video_uploaded=True,
            thumbnail_uploaded=thumbnail_uploaded,
            payload=payload,
            video_record=video_record
        )
