"""
autonomous/publication_manager.py - High-Level Publication Manager & State Coordinator.

Coordinates Phase 12 operations:
- dry-run audit (--youtube-dry-run)
- mock upload and verification (--mock-youtube-upload)
- publication status inspection (--youtube-status)
- idempotency: recognizes existing valid publication manifests
- state transitions:
  PUBLISH_PACKAGE_READY -> YOUTUBE_VALIDATING -> YOUTUBE_READY ->
  YOUTUBE_UPLOADING -> YOUTUBE_UPLOADED -> THUMBNAIL_UPLOADING ->
  THUMBNAIL_UPLOADED -> PUBLICATION_VERIFIED
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.youtube_client import (
    BaseYouTubeClient, MockYouTubeClient, RealYouTubeClient, get_youtube_client,
    YouTubeClientError, YouTubeUploadError, YouTubeDuplicateError, ErrorClassification
)
from autonomous.youtube_validator import YouTubePublicationValidator, YouTubeValidationReport
from autonomous.upload_manager import YouTubeUploadManager, UploadResult
from autonomous.publication_verifier import YouTubePublicationVerifier

logger = logging.getLogger("autonomous.publication_manager")


class PublicationManager:
    """Coordinates YouTube publication workflows, dry runs, and state transitions."""

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        client: Optional[BaseYouTubeClient] = None,
        settings=None
    ):
        self.state_manager = state_manager or StateManager()
        self.settings = settings or autonomous_settings
        self.client = client or get_youtube_client(settings=self.settings, force_mock=True)
        self.validator = YouTubePublicationValidator(self.state_manager, self.settings)
        self.upload_manager = YouTubeUploadManager(self.client, self.state_manager, self.settings)
        self.verifier = YouTubePublicationVerifier(self.client, self.settings)

    def execute_dry_run(self, episode_id: str) -> Dict[str, Any]:
        """
        Perform a dry-run check of the publication package without any network or state mutation.
        """
        report = self.validator.validate_package(episode_id)
        episode = self.state_manager.get_episode(episode_id)

        ep_dir = Path(episode.output_directory) if episode else BASE_DIR / "outputs" / "episodes" / episode_id
        if not ep_dir.is_absolute():
            ep_dir = BASE_DIR / ep_dir

        payload = {}
        if report.passed and report.package_data:
            payload = self.upload_manager.prepare_payload(ep_dir, report.package_data)

        return {
            "episode_id": episode_id,
            "validation_passed": report.passed,
            "errors": report.errors,
            "warnings": report.warnings,
            "checks": report.checks,
            "payload": {
                "title": payload.get("title"),
                "description_preview": payload.get("description", "")[:100] + "...",
                "tags_count": len(payload.get("tags", [])),
                "category_id": payload.get("category_id"),
                "privacy_status": payload.get("privacy_status"),
                "made_for_kids": payload.get("made_for_kids"),
            } if payload else None,
            "integration_status": {
                "youtube_integration_enabled": bool(self.settings.youtube_integration_enabled),
                "channel_id": self.settings.youtube_channel_id,
                "oauth_configured": False,
                "client_mode": "MOCK" if isinstance(self.client, MockYouTubeClient) else "REAL",
                "real_api_calls": 0,
            },
            "verdict": "READY_FOR_FUTURE_YOUTUBE_INTEGRATION" if report.passed else "VALIDATION_FAILED",
        }

    def execute_mock_upload(
        self,
        episode_id: str,
        publication_mode: str = "MOCK_PRIVATE_TEST"
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Execute full Phase 12 mock publication pipeline:
        PUBLISH_PACKAGE_READY -> YOUTUBE_VALIDATING -> YOUTUBE_READY ->
        YOUTUBE_UPLOADING -> YOUTUBE_UPLOADED -> THUMBNAIL_UPLOADING ->
        THUMBNAIL_UPLOADED -> PUBLICATION_VERIFIED.

        Idempotent: If already PUBLICATION_VERIFIED and manifest exists, verifies and reuses it.
        """
        episode = self.state_manager.get_episode(episode_id)
        if not episode:
            return False, None, f"Episode '{episode_id}' not found in database."

        ep_dir = Path(episode.output_directory)
        if not ep_dir.is_absolute():
            ep_dir = BASE_DIR / ep_dir

        publish_dir = ep_dir / "publish"
        manifest_path = publish_dir / "youtube_publication_manifest.json"

        # 1. Idempotency check: If already PUBLICATION_VERIFIED and manifest exists, verify & return
        if episode.status == EpisodeState.PUBLICATION_VERIFIED.value and manifest_path.exists():
            logger.info(f"Episode '{episode_id}' is already in PUBLICATION_VERIFIED. Verifying existing manifest...")
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    existing_manifest = json.load(f)
                if existing_manifest.get("verification_status") == "VERIFIED":
                    logger.info(f"Existing publication manifest for '{episode_id}' verified. Reusing without re-uploading.")
                    return True, existing_manifest, "Episode is already published and verified in mock service."
            except Exception as e:
                logger.warning(f"Failed to read existing publication manifest: {e}. Re-evaluating publication.")

        # 2. Transition: PUBLISH_PACKAGE_READY -> YOUTUBE_VALIDATING
        if episode.status == EpisodeState.PUBLISH_PACKAGE_READY.value:
            self.state_manager.transition(episode_id, EpisodeState.YOUTUBE_VALIDATING)
            episode = self.state_manager.get_episode(episode_id)
        elif episode.status not in (
            EpisodeState.YOUTUBE_VALIDATING.value,
            EpisodeState.YOUTUBE_READY.value,
            EpisodeState.YOUTUBE_UPLOADING.value,
            EpisodeState.YOUTUBE_UPLOADED.value,
            EpisodeState.THUMBNAIL_UPLOADING.value,
            EpisodeState.THUMBNAIL_UPLOADED.value,
            EpisodeState.UPLOAD_FAILED.value,
        ):
            return False, None, f"Episode '{episode_id}' is in invalid state '{episode.status}' for publication."

        # 3. Input Gate Validation
        validation_report = self.validator.validate_package(episode_id)
        if not validation_report.passed:
            err_msg = f"Publication input gate failed: {'; '.join(validation_report.errors)}"
            logger.error(err_msg)
            self.state_manager.transition(episode_id, EpisodeState.PUBLICATION_REVIEW_REQUIRED, error_message=err_msg)
            return False, None, err_msg

        package_data = validation_report.package_data
        if not package_data:
            with open(publish_dir / "publish_package.json", "r", encoding="utf-8") as f:
                package_data = json.load(f)

        # 4. Transition: YOUTUBE_VALIDATING -> YOUTUBE_READY
        self.state_manager.transition(episode_id, EpisodeState.YOUTUBE_READY)

        # 5. Transition: YOUTUBE_READY -> YOUTUBE_UPLOADING
        self.state_manager.transition(episode_id, EpisodeState.YOUTUBE_UPLOADING)

        # Check existing video ID if present in existing manifest
        existing_vid = None
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    existing_vid = m.get("youtube_video_id")
            except Exception:
                pass

        # 6. Execute Video & Thumbnail Upload
        upload_res = self.upload_manager.execute_upload(
            episode_id=episode_id,
            ep_dir=ep_dir,
            package_data=package_data,
            existing_video_id=existing_vid
        )

        if not upload_res.video_uploaded:
            target_state = (
                EpisodeState.PUBLICATION_REVIEW_REQUIRED
                if upload_res.error_classification == ErrorClassification.REVIEW_REQUIRED
                else EpisodeState.UPLOAD_FAILED
            )
            self.state_manager.transition(episode_id, target_state, error_message=upload_res.error)
            return False, None, f"Video upload failed: {upload_res.error}"

        # Transition: YOUTUBE_UPLOADING -> YOUTUBE_UPLOADED
        self.state_manager.transition(episode_id, EpisodeState.YOUTUBE_UPLOADED)

        # Transition: YOUTUBE_UPLOADED -> THUMBNAIL_UPLOADING
        self.state_manager.transition(episode_id, EpisodeState.THUMBNAIL_UPLOADING)

        if not upload_res.thumbnail_uploaded:
            target_state = (
                EpisodeState.PUBLICATION_REVIEW_REQUIRED
                if upload_res.error_classification == ErrorClassification.REVIEW_REQUIRED
                else EpisodeState.UPLOAD_FAILED
            )
            self.state_manager.transition(episode_id, target_state, error_message=upload_res.error)
            return False, None, f"Thumbnail upload failed: {upload_res.error}"

        # Transition: THUMBNAIL_UPLOADING -> THUMBNAIL_UPLOADED
        self.state_manager.transition(episode_id, EpisodeState.THUMBNAIL_UPLOADED)

        # 7. Verification & Manifest Generation
        ok, manifest, ver_err = self.verifier.verify_and_seal(
            episode_id=episode_id,
            ep_dir=ep_dir,
            video_id=upload_res.video_id,
            package_data=package_data,
            publication_mode=publication_mode
        )

        if not ok or not manifest:
            self.state_manager.transition(episode_id, EpisodeState.PUBLICATION_REVIEW_REQUIRED, error_message=ver_err)
            return False, None, f"Verification failed: {ver_err}"

        # 8. Terminal Transition: THUMBNAIL_UPLOADED -> PUBLICATION_VERIFIED
        self.state_manager.transition(episode_id, EpisodeState.PUBLICATION_VERIFIED)

        logger.info(f"Phase 12: PUBLICATION_VERIFIED reached for episode '{episode_id}' (Mock ID: {upload_res.video_id}).")
        return True, manifest, "Publication verified in mock environment successfully."

    def get_publication_status(self, episode_id: str) -> Dict[str, Any]:
        """Inspect and return current publication status for an episode."""
        episode = self.state_manager.get_episode(episode_id)
        if not episode:
            return {"error": f"Episode '{episode_id}' not found."}

        ep_dir = Path(episode.output_directory)
        if not ep_dir.is_absolute():
            ep_dir = BASE_DIR / ep_dir

        publish_dir = ep_dir / "publish"
        manifest_file = publish_dir / "youtube_publication_manifest.json"
        manifest_data = None

        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading manifest: {e}")

        is_mock_verified = (episode.status == EpisodeState.PUBLICATION_VERIFIED.value)

        return {
            "episode_id": episode_id,
            "state": episode.status,
            "publication_status": "MOCK_VERIFIED" if is_mock_verified else episode.status,
            "real_youtube_upload": "NOT PERFORMED",
            "mock_youtube_upload": "YES" if (is_mock_verified or manifest_data) else "NO",
            "youtube_channel": "NOT CONFIGURED",
            "google_oauth": "NOT CONFIGURED",
            "real_integration": "DORMANT / DISABLED",
            "video_id": manifest_data.get("youtube_video_id") if manifest_data else None,
            "privacy_status": manifest_data.get("privacy_status") if manifest_data else None,
            "manifest_exists": manifest_file.exists(),
            "manifest_data": manifest_data,
        }
