"""
autonomous/publish_package.py - Autonomous Post-Production Publishing Package Builder.
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).

Consumes MASTER_READY, SEO metadata, and Thumbnail artifacts.
Cryptographically seals and validates the publishing package in:
outputs/episodes/<episode_id>/publish/publish_package.json
Terminal gate for Phase 11 (PUBLISH_PACKAGE_READY).
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.seo_engine import AutonomousSEOEngine, SEOResult
from autonomous.thumbnail_engine import AutonomousThumbnailEngine, ThumbnailManifest, ThumbnailQCReport
from autonomous.seo_validator import SEOValidator, SEOValidationReport, OBSOLETE_IDENTITY_TOKENS

logger = logging.getLogger("autonomous.publish_package")


def _sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class PublishPackageBuilder:
    """Builds, validates, and seals the publishing package for an episode."""

    def __init__(self, state_manager: Optional[StateManager] = None, settings=None):
        self.state_manager = state_manager or StateManager()
        self.settings = settings or autonomous_settings
        self.seo_engine = AutonomousSEOEngine(self.state_manager, self.settings)
        self.thumbnail_engine = AutonomousThumbnailEngine(self.state_manager, self.settings)
        self.seo_validator = SEOValidator(self.settings)

    def build_publish_package(
        self,
        episode_id: str,
        offline: bool = False,
        force_host_thumbnail: Optional[bool] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Execute full Phase 11 publishing package generation & validation:
        MASTER_READY -> METADATA_GENERATING -> METADATA_READY ->
        THUMBNAIL_GENERATING -> THUMBNAIL_READY -> PUBLISH_PACKAGE_READY.
        Idempotent: If already PUBLISH_PACKAGE_READY, verifies and returns existing package.
        """
        episode = self.state_manager.get_episode(episode_id)
        if not episode:
            return False, None, f"Episode '{episode_id}' not found in database."

        ep_dir = Path(episode.output_directory)
        if not ep_dir.is_absolute():
            ep_dir = BASE_DIR / ep_dir

        publish_dir = ep_dir / "publish"
        publish_package_path = publish_dir / "publish_package.json"

        # 1. Idempotency Check: If already PUBLISH_PACKAGE_READY, verify and return
        if (episode.status == EpisodeState.PUBLISH_PACKAGE_READY.value or
            episode.current_stage == EpisodeState.PUBLISH_PACKAGE_READY.value) and publish_package_path.exists():
            logger.info(f"Episode '{episode_id}' is already PUBLISH_PACKAGE_READY. Verifying existing package integrity...")
            valid, pkg, msg = self.verify_package_integrity(episode_id)
            if valid:
                logger.info(f"Existing publish package for '{episode_id}' is cryptographically verified.")
                return True, pkg, "Publish package is already complete and verified."
            else:
                logger.warning(f"Existing publish package failed verification ({msg}). Re-evaluating package.")

        # 2. Check input gate
        if episode.status == EpisodeState.REVIEW_REQUIRED.value:
            return False, None, f"Episode '{episode_id}' is in REVIEW_REQUIRED state. Rejected."
        if episode.status == EpisodeState.FAILED.value:
            return False, None, f"Episode '{episode_id}' is in FAILED state. Rejected."

        # 3. Step 1: SEO Generation (if not already METADATA_READY or THUMBNAIL_READY)
        seo_json_path = publish_dir / "seo.json"
        if episode.status in (EpisodeState.MASTER_READY.value, EpisodeState.METADATA_GENERATING.value) or not seo_json_path.exists():
            logger.info(f"Phase 11: Executing SEO generation for '{episode_id}'...")
            ok, seo_res, seo_val, err = self.seo_engine.generate_seo(episode_id, offline=offline)
            if not ok:
                return False, None, f"SEO Generation failed: {err}"

        # 4. Step 2: Thumbnail Generation (if not already THUMBNAIL_READY)
        thumb_path = publish_dir / "thumbnail.jpg"
        episode = self.state_manager.get_episode(episode_id)
        if episode.status in (EpisodeState.METADATA_READY.value, EpisodeState.THUMBNAIL_GENERATING.value) or not thumb_path.exists():
            logger.info(f"Phase 11: Executing Thumbnail generation for '{episode_id}'...")
            ok, thumb_manifest, thumb_qc, err = self.thumbnail_engine.generate_thumbnail(
                episode_id,
                force_host=force_host_thumbnail
            )
            if not ok:
                return False, None, f"Thumbnail Generation failed: {err}"

        # 5. Read all generated artifacts
        try:
            with open(publish_dir / "seo.json", "r", encoding="utf-8") as f:
                seo_dict = json.load(f)
            with open(publish_dir / "seo_validation_report.json", "r", encoding="utf-8") as f:
                seo_report_dict = json.load(f)
            with open(publish_dir / "thumbnail_manifest.json", "r", encoding="utf-8") as f:
                thumb_manifest_dict = json.load(f)
            with open(publish_dir / "thumbnail_qc_report.json", "r", encoding="utf-8") as f:
                thumb_qc_dict = json.load(f)
        except Exception as e:
            return False, None, f"Failed to load generated Phase 11 artifacts: {e}"

        # 6. Read Master & Script artifacts for cryptographic sealing
        master_mp4 = ep_dir / "master" / "master.mp4"
        master_manifest_path = ep_dir / "master" / "master_manifest.json"
        validated_script_path = ep_dir / "script" / "validated_script.json"
        script_path = ep_dir / "script" / "script.json"

        if not master_mp4.exists():
            return False, None, "master.mp4 does not exist."
        if not master_manifest_path.exists():
            return False, None, "master_manifest.json does not exist."

        master_sha = _sha256_file(master_mp4)
        master_manifest_sha = _sha256_file(master_manifest_path)
        with open(master_manifest_path, "r", encoding="utf-8") as f:
            master_manifest_data = json.load(f)

        script_sha = _sha256_file(script_path) if script_path.exists() else ""
        val_script_sha = _sha256_file(validated_script_path) if validated_script_path.exists() else script_sha

        thumb_sha = _sha256_file(publish_dir / "thumbnail.jpg")
        thumb_manifest_sha = _sha256_file(publish_dir / "thumbnail_manifest.json")
        thumb_qc_sha = _sha256_file(publish_dir / "thumbnail_qc_report.json")
        seo_report_sha = _sha256_file(publish_dir / "seo_validation_report.json")

        now_iso = datetime.now(timezone.utc).isoformat()
        channel_cfg = self.settings.channel

        # 7. Assemble Complete Publishing Package Schema
        publish_package = {
            "episode_id": episode_id,
            "schema_version": "1.0.0",
            "generator_version": "1.0.0",
            "terminal_state": EpisodeState.PUBLISH_PACKAGE_READY.value,
            "created_at": now_iso,
            "channel_identity": {
                "channel_name_ta": channel_cfg.channel_name_ta,
                "channel_name_en": channel_cfg.channel_name_en,
                "channel_handle": channel_cfg.channel_handle,
                "channel_description_en": channel_cfg.channel_description_en,
                "tagline_ta": channel_cfg.tagline_ta,
                "tagline_en": channel_cfg.tagline_en,
            },
            "host_identity": {
                "host_name_ta": channel_cfg.host_name_ta,
                "host_name_en": channel_cfg.host_name_en,
                "character_id": channel_cfg.host_character_id,
                "consistency_token": channel_cfg.host_consistency_token,
            },
            "master": {
                "path": "master/master.mp4",
                "sha256": master_sha,
                "manifest_path": "master/master_manifest.json",
                "manifest_sha256": master_manifest_sha,
                "duration_seconds": master_manifest_data.get("duration_seconds", 0.0),
                "resolution": master_manifest_data.get("resolution", "1280x720"),
                "fps": master_manifest_data.get("fps", 25.0),
            },
            "script": {
                "script_path": "script/script.json" if script_path.exists() else None,
                "script_sha256": script_sha,
                "validated_script_path": "script/validated_script.json" if validated_script_path.exists() else None,
                "validated_script_sha256": val_script_sha,
            },
            "seo": {
                "selected_title": seo_dict.get("selected_title"),
                "description": seo_dict.get("description"),
                "tags": seo_dict.get("tags", []),
                "tag_count": len(seo_dict.get("tags", [])),
                "total_tag_chars": seo_dict.get("total_tag_chars", 0),
                "hashtags": seo_dict.get("hashtags", []),
                "language": seo_dict.get("language", "ta"),
                "secondary_language": seo_dict.get("secondary_language", "en"),
                "category": seo_dict.get("category", "Education"),
                "category_id": seo_dict.get("category_id", 27),
                "validation_report_sha256": seo_report_sha,
            },
            "thumbnail": {
                "path": "publish/thumbnail.jpg",
                "sha256": thumb_sha,
                "dimensions": thumb_manifest_dict.get("dimensions", [1280, 720]),
                "aspect_ratio": thumb_manifest_dict.get("aspect_ratio", "16:9"),
                "selected_text": thumb_manifest_dict.get("selected_text"),
                "manifest_sha256": thumb_manifest_sha,
                "qc_report_sha256": thumb_qc_sha,
            },
            "provenance": {
                "llm_used": seo_dict.get("provenance", {}).get("llm_used", False),
                "model_name": seo_dict.get("provenance", {}).get("model_name"),
                "fallback_used": seo_dict.get("provenance", {}).get("fallback_used", True),
                "fallback_reason": seo_dict.get("provenance", {}).get("fallback_reason"),
                "generator_version": "1.0.0",
                "schema_version": "1.0.0",
                "source_master_sha256": master_sha,
                "source_script_sha256": script_sha,
                "source_validated_script_sha256": val_script_sha,
            }
        }

        # 8. Final Package Gate (Verify all 13 conditions)
        passed_gate, gate_errors = self._evaluate_final_package_gate(
            ep_dir=ep_dir,
            package=publish_package,
            seo_report=seo_report_dict,
            thumb_qc=thumb_qc_dict
        )

        if not passed_gate:
            err_str = "; ".join(gate_errors)
            logger.error(f"Final publishing package gate failed for '{episode_id}': {err_str}")
            try:
                self.state_manager.transition(
                    episode_id,
                    EpisodeState.REVIEW_REQUIRED,
                    error_message=f"PUBLISH_PACKAGE_GATE_FAILED: {err_str}"
                )
            except Exception:
                pass
            return False, None, f"Final gate validation failed: {err_str}"

        # 9. Atomic Write of publish_package.json
        tmp_package_path = publish_dir / "publish_package.tmp.json"
        with open(tmp_package_path, "w", encoding="utf-8") as f:
            json.dump(publish_package, f, indent=2, ensure_ascii=False)
        os.replace(tmp_package_path, publish_package_path)

        # 10. Transition to PUBLISH_PACKAGE_READY
        try:
            self.state_manager.transition(episode_id, EpisodeState.PUBLISH_PACKAGE_READY)
            self.state_manager.update_episode(
                episode_id,
                metadata_path=str(publish_package_path)
            )
        except Exception as e:
            logger.error(f"Failed to transition to PUBLISH_PACKAGE_READY: {e}")
            return False, publish_package, str(e)

        logger.info(f"Phase 11: PUBLISH_PACKAGE_READY successfully reached for '{episode_id}'.")
        return True, publish_package, "Publishing package created and verified successfully."

    def _evaluate_final_package_gate(
        self,
        ep_dir: Path,
        package: Dict[str, Any],
        seo_report: Dict[str, Any],
        thumb_qc: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Verify all 13 conditions required for PUBLISH_PACKAGE_READY."""
        errors = []

        # 1. SEO generation succeeded
        if not package.get("seo", {}).get("selected_title"):
            errors.append("Gate 1 failed: SEO selected_title is missing.")

        # 2. SEO validation passed
        if not seo_report.get("passed", False):
            errors.append(f"Gate 2 failed: SEO validation report status is false ({seo_report.get('errors')}).")

        # 3. Thumbnail generation succeeded
        thumb_path = ep_dir / "publish" / "thumbnail.jpg"
        if not thumb_path.exists() or thumb_path.stat().st_size == 0:
            errors.append("Gate 3 failed: thumbnail.jpg does not exist or is empty.")

        # 4. Thumbnail QC passed
        if not thumb_qc.get("passed", False):
            errors.append(f"Gate 4 failed: Thumbnail QC report status is false ({thumb_qc.get('errors')}).")

        # 5. All package files exist
        required_files = [
            ep_dir / "publish" / "seo.json",
            ep_dir / "publish" / "thumbnail.jpg",
            ep_dir / "publish" / "thumbnail_manifest.json",
            ep_dir / "publish" / "thumbnail_qc_report.json",
            ep_dir / "publish" / "seo_validation_report.json",
        ]
        for rf in required_files:
            if not rf.exists() or rf.stat().st_size == 0:
                errors.append(f"Gate 5 failed: Required artifact missing: {rf.name}")

        # 6. All JSON files parse
        for rf in required_files:
            if rf.suffix == ".json":
                try:
                    with open(rf, "r", encoding="utf-8") as f:
                        json.load(f)
                except Exception as e:
                    errors.append(f"Gate 6 failed: Failed to parse {rf.name}: {e}")

        # 7. SHA-256 hashes match internal package records
        actual_thumb_hash = _sha256_file(thumb_path)
        if actual_thumb_hash != package["thumbnail"]["sha256"]:
            errors.append("Gate 7 failed: thumbnail hash mismatch with package record.")

        # 8. Master hash matches actual master.mp4
        master_mp4 = ep_dir / "master" / "master.mp4"
        actual_master_hash = _sha256_file(master_mp4)
        if actual_master_hash != package["master"]["sha256"]:
            errors.append("Gate 8 failed: Master MP4 actual hash does not match package record.")

        # 9. Script hashes match
        val_script = ep_dir / "script" / "validated_script.json"
        if val_script.exists():
            actual_val_script_hash = _sha256_file(val_script)
            if actual_val_script_hash != package["script"]["validated_script_sha256"]:
                errors.append("Gate 9 failed: validated_script.json hash mismatch.")

        # 10. Thumbnail hash matches actual thumbnail
        if actual_thumb_hash != _sha256_file(ep_dir / "publish" / "thumbnail.jpg"):
            errors.append("Gate 10 failed: thumbnail file hash mismatch.")

        # 11. No obsolete identity in package
        package_str = json.dumps(package).lower()
        for obsolete in OBSOLETE_IDENTITY_TOKENS:
            if obsolete in package_str:
                errors.append(f"Gate 11 failed: Obsolete identity token found in package: '{obsolete}'.")

        # 12. No unsupported factual claim remains
        title = package["seo"]["selected_title"]
        for pattern in [r"\b100%\s*proof\b", r"\bscientists\s+are\s+terrified\b"]:
            if re.search(pattern, title, re.IGNORECASE):
                errors.append(f"Gate 12 failed: Sensational claim pattern detected: {pattern}")

        # 13. All required provenance fields exist
        prov = package.get("provenance", {})
        required_prov_keys = [
            "llm_used", "fallback_used", "generator_version", "schema_version",
            "source_master_sha256", "source_script_sha256"
        ]
        for k in required_prov_keys:
            if k not in prov:
                errors.append(f"Gate 13 failed: Missing required provenance field: '{k}'.")

        passed = len(errors) == 0
        return passed, errors

    def verify_package_integrity(self, episode_id: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """Independently verify an existing publish package for idempotency."""
        episode = self.state_manager.get_episode(episode_id)
        if not episode:
            return False, None, "Episode not found."

        ep_dir = Path(episode.output_directory)
        if not ep_dir.is_absolute():
            ep_dir = BASE_DIR / ep_dir

        publish_package_path = ep_dir / "publish" / "publish_package.json"
        if not publish_package_path.exists():
            return False, None, "publish_package.json does not exist."

        try:
            with open(publish_package_path, "r", encoding="utf-8") as f:
                package = json.load(f)
            with open(ep_dir / "publish" / "seo_validation_report.json", "r", encoding="utf-8") as f:
                seo_report = json.load(f)
            with open(ep_dir / "publish" / "thumbnail_qc_report.json", "r", encoding="utf-8") as f:
                thumb_qc = json.load(f)
        except Exception as e:
            return False, None, f"Failed to load existing package files: {e}"

        passed, errors = self._evaluate_final_package_gate(ep_dir, package, seo_report, thumb_qc)
        if not passed:
            return False, package, "; ".join(errors)
        return True, package, "Package integrity verified."
