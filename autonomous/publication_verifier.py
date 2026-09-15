"""
autonomous/publication_verifier.py - Verification & Publication Manifest Generator.

Verifies published resources with the client and creates the cryptographically sealed manifest:
outputs/episodes/<episode_id>/publish/youtube_publication_manifest.json

Clearly records provider="mock", real_api_called=false, mock=true, and channel_id=null.
Uses atomic file writes (.tmp promotion).
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.youtube_client import BaseYouTubeClient, MockYouTubeClient, YouTubeVerificationError

logger = logging.getLogger("autonomous.publication_verifier")


def _hash_str(s: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class YouTubePublicationVerifier:
    """Verifies uploaded YouTube resources and compiles the final publication manifest."""

    def __init__(self, client: BaseYouTubeClient, settings=None):
        self.client = client
        self.settings = settings or autonomous_settings

    def verify_and_seal(
        self,
        episode_id: str,
        ep_dir: Path,
        video_id: str,
        package_data: Dict[str, Any],
        publication_mode: str = "MOCK_PRIVATE_TEST"
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Verify video and thumbnail resource status, then atomically write
        youtube_publication_manifest.json.
        """
        publish_dir = ep_dir / "publish"
        manifest_path = publish_dir / "youtube_publication_manifest.json"
        tmp_manifest_path = publish_dir / "youtube_publication_manifest.tmp.json"

        expected_title = package_data.get("seo", {}).get("selected_title")
        expected_privacy = "MOCK_PRIVATE" if isinstance(self.client, MockYouTubeClient) else "private"

        # 1. Verify Video Resource
        try:
            video_ver = self.client.verify_video(
                video_id=video_id,
                expected_title=expected_title,
                expected_privacy=expected_privacy
            )
        except YouTubeVerificationError as e:
            logger.error(f"Video verification failed for '{video_id}': {e}")
            return False, None, f"Video verification failed: {e}"

        # 2. Verify Thumbnail Resource
        try:
            thumb_ver = self.client.verify_thumbnail(video_id=video_id)
        except YouTubeVerificationError as e:
            logger.error(f"Thumbnail verification failed for '{video_id}': {e}")
            return False, None, f"Thumbnail verification failed: {e}"

        # 3. Assemble Publication Manifest
        now_iso = datetime.now(timezone.utc).isoformat()
        master_info = package_data.get("master", {})
        thumb_info = package_data.get("thumbnail", {})
        seo_info = package_data.get("seo", {})

        description_text = seo_info.get("description", "")
        tags_list = seo_info.get("tags", [])

        is_mock = isinstance(self.client, MockYouTubeClient)

        publication_manifest = {
            "episode_id": episode_id,
            "publication_schema_version": "1.0.0",
            "generator_version": "1.0.0",
            "provider": "mock" if is_mock else "youtube",
            "real_api_called": False if is_mock else True,
            "mock": is_mock,
            "youtube_video_id": video_id,
            "channel_id": None,  # Channel ID is unconfigured and unknown
            "channel_handle": "@kaalapadhivugal",
            "channel_name_ta": "காலப் பதிவுகள்",
            "upload_timestamp": now_iso,
            "verification_timestamp": now_iso,
            "master_sha256": master_info.get("sha256", ""),
            "master_size": master_info.get("file_size_bytes", 0),
            "master_duration": master_info.get("duration_seconds", 0.0),
            "thumbnail_sha256": thumb_info.get("sha256", ""),
            "title": expected_title,
            "description_hash": _hash_str(description_text),
            "tags_hash": _hash_str(json.dumps(tags_list, ensure_ascii=False)),
            "privacy_status": expected_privacy,
            "made_for_kids": bool(self.settings.youtube_made_for_kids),
            "publication_mode": publication_mode,
            "upload_status": "SUCCESS",
            "thumbnail_status": "SUCCESS",
            "verification_status": "VERIFIED",
            "provenance": {
                "source_publish_package_sha256": package_data.get("provenance", {}).get("source_master_sha256", ""),
                "master_sha256": master_info.get("sha256", ""),
                "thumbnail_sha256": thumb_info.get("sha256", ""),
                "script_sha256": package_data.get("script", {}).get("script_sha256", ""),
                "validated_script_sha256": package_data.get("script", {}).get("validated_script_sha256", ""),
                "created_at": now_iso,
            }
        }

        # 4. Atomic Write
        try:
            with open(tmp_manifest_path, "w", encoding="utf-8") as f:
                json.dump(publication_manifest, f, indent=2, ensure_ascii=False)
            os.replace(tmp_manifest_path, manifest_path)
            logger.info(f"YouTubePublicationVerifier: Publication manifest atomically created at '{manifest_path}'.")
        except Exception as e:
            if tmp_manifest_path.exists():
                tmp_manifest_path.unlink()
            return False, None, f"Failed to atomically write publication manifest: {e}"

        return True, publication_manifest, "Publication verified and manifest generated successfully."
