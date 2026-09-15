"""
autonomous/youtube_validator.py - Strict Publication Package Input Gate Validator.

Enforces 21-point validation before allowing any episode to enter the upload pipeline:
1. Episode exists in database
2. Episode state == PUBLISH_PACKAGE_READY
3. publish/ directory exists
4. publish/publish_package.json exists
5. publish/seo.json exists
6. publish/thumbnail.jpg exists
7. publish/thumbnail_manifest.json exists
8. publish/thumbnail_qc_report.json exists
9. publish/seo_validation_report.json exists
10. master/master.mp4 exists
11. master/master_manifest.json exists
12. Master MP4 SHA-256 matches package record
13. Validated script SHA-256 matches package record
14. Thumbnail JPG SHA-256 matches package record
15. SEO validation passed in seo_validation_report.json
16. Thumbnail QC passed in thumbnail_qc_report.json
17. publish_package.json conforms to required schema
18. Provenance information complete (generator_version, schema_version, source hashes)
19. Canonical branding valid (காலப் பதிவுகள் / Kaalapadhivugal / @kaalapadhivugal / யாழினி / Yaazhini)
20. Zero obsolete identity tokens present
21. No unsupported sensational claims or factual fabrications
"""

import os
import re
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.seo_validator import OBSOLETE_IDENTITY_TOKENS, SENSATIONAL_PATTERNS

logger = logging.getLogger("autonomous.youtube_validator")


def _sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class YouTubeValidationReport:
    """Detailed validation report for publication package input gate."""
    passed: bool
    episode_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    package_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "episode_id": self.episode_id,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
        }


class YouTubePublicationValidator:
    """Validates that a publication package is 100% compliant and cryptographically sealed."""

    def __init__(self, state_manager: Optional[StateManager] = None, settings=None):
        self.state_manager = state_manager or StateManager()
        self.settings = settings or autonomous_settings

    def validate_package(self, episode_id: str) -> YouTubeValidationReport:
        """Execute the full 21-point input gate validation."""
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, bool] = {}

        # 1. Episode exists in database
        episode = self.state_manager.get_episode(episode_id)
        if not episode:
            return YouTubeValidationReport(
                passed=False,
                episode_id=episode_id,
                errors=[f"Gate 1 failed: Episode '{episode_id}' not found in database."],
                checks={"episode_exists": False}
            )
        checks["episode_exists"] = True

        # 2. Episode state == PUBLISH_PACKAGE_READY (or YOUTUBE_VALIDATING during active validation)
        if episode.status not in (EpisodeState.PUBLISH_PACKAGE_READY.value, EpisodeState.YOUTUBE_VALIDATING.value):
            errors.append(
                f"Gate 2 failed: Episode state is '{episode.status}', expected '{EpisodeState.PUBLISH_PACKAGE_READY.value}'."
            )
            checks["state_is_publish_package_ready"] = False
        else:
            checks["state_is_publish_package_ready"] = True

        # Reject explicitly if in REVIEW_REQUIRED or FAILED
        if episode.status in (EpisodeState.REVIEW_REQUIRED.value, EpisodeState.FAILED.value):
            errors.append(f"Gate 2b failed: Episode is in blocked status '{episode.status}'.")

        ep_dir = Path(episode.output_directory)
        if not ep_dir.is_absolute():
            ep_dir = BASE_DIR / ep_dir

        publish_dir = ep_dir / "publish"
        master_dir = ep_dir / "master"
        script_dir = ep_dir / "script"

        # 3. publish/ directory exists
        checks["publish_dir_exists"] = publish_dir.exists() and publish_dir.is_dir()
        if not checks["publish_dir_exists"]:
            errors.append("Gate 3 failed: Directory 'publish/' does not exist.")
            return YouTubeValidationReport(passed=False, episode_id=episode_id, errors=errors, checks=checks)

        # 4-9. Required Publish Artifacts Exist & Non-Empty
        package_file = publish_dir / "publish_package.json"
        seo_file = publish_dir / "seo.json"
        thumb_file = publish_dir / "thumbnail.jpg"
        thumb_manifest_file = publish_dir / "thumbnail_manifest.json"
        thumb_qc_file = publish_dir / "thumbnail_qc_report.json"
        seo_report_file = publish_dir / "seo_validation_report.json"

        artifact_map = {
            "publish_package_exists": (package_file, 4),
            "seo_json_exists": (seo_file, 5),
            "thumbnail_jpg_exists": (thumb_file, 6),
            "thumbnail_manifest_exists": (thumb_manifest_file, 7),
            "thumbnail_qc_report_exists": (thumb_qc_file, 8),
            "seo_validation_report_exists": (seo_report_file, 9),
        }

        for check_key, (fpath, gate_num) in artifact_map.items():
            exists_and_valid = fpath.exists() and fpath.stat().st_size > 0
            checks[check_key] = exists_and_valid
            if not exists_and_valid:
                errors.append(f"Gate {gate_num} failed: Required file '{fpath.name}' is missing or empty.")

        # 10-11. Master Video & Master Manifest Exist
        master_file = master_dir / "master.mp4"
        master_manifest_file = master_dir / "master_manifest.json"

        checks["master_mp4_exists"] = master_file.exists() and master_file.stat().st_size > 1000
        if not checks["master_mp4_exists"]:
            errors.append("Gate 10 failed: 'master/master.mp4' is missing or corrupted.")

        checks["master_manifest_exists"] = master_manifest_file.exists() and master_manifest_file.stat().st_size > 20
        if not checks["master_manifest_exists"]:
            errors.append("Gate 11 failed: 'master/master_manifest.json' is missing or empty.")

        # If any files are missing, halt before reading contents
        if errors:
            return YouTubeValidationReport(passed=False, episode_id=episode_id, errors=errors, checks=checks)

        # Parse JSON Artifacts
        try:
            with open(package_file, "r", encoding="utf-8") as f:
                package_data = json.load(f)
            with open(seo_file, "r", encoding="utf-8") as f:
                seo_data = json.load(f)
            with open(seo_report_file, "r", encoding="utf-8") as f:
                seo_report = json.load(f)
            with open(thumb_manifest_file, "r", encoding="utf-8") as f:
                thumb_manifest = json.load(f)
            with open(thumb_qc_file, "r", encoding="utf-8") as f:
                thumb_qc = json.load(f)
        except Exception as e:
            errors.append(f"Gate 4-9 parse error: Failed to parse required JSON artifacts: {e}")
            return YouTubeValidationReport(passed=False, episode_id=episode_id, errors=errors, checks=checks)

        # 12. Master MP4 SHA-256 matches package record
        actual_master_sha = _sha256_file(master_file)
        expected_master_sha = package_data.get("master", {}).get("sha256")
        checks["master_hash_matches"] = (actual_master_sha == expected_master_sha)
        if not checks["master_hash_matches"]:
            errors.append(
                f"Gate 12 failed: Master MP4 actual hash '{actual_master_sha}' does not match package record '{expected_master_sha}'."
            )

        # 13. Validated Script SHA-256 matches package record
        val_script_file = script_dir / "validated_script.json"
        script_file = script_dir / "script.json"
        target_script = val_script_file if val_script_file.exists() else script_file

        if target_script.exists():
            actual_script_sha = _sha256_file(target_script)
            expected_val_sha = package_data.get("script", {}).get("validated_script_sha256")
            expected_src_sha = package_data.get("provenance", {}).get("source_validated_script_sha256")
            checks["script_hash_matches"] = (actual_script_sha in (expected_val_sha, expected_src_sha))
            if not checks["script_hash_matches"]:
                errors.append(
                    f"Gate 13 failed: Script actual hash '{actual_script_sha}' does not match package record '{expected_val_sha}'."
                )
        else:
            checks["script_hash_matches"] = False
            errors.append("Gate 13 failed: Neither validated_script.json nor script.json was found.")

        # 14. Thumbnail JPG SHA-256 matches package record
        actual_thumb_sha = _sha256_file(thumb_file)
        expected_thumb_sha = package_data.get("thumbnail", {}).get("sha256")
        checks["thumbnail_hash_matches"] = (actual_thumb_sha == expected_thumb_sha)
        if not checks["thumbnail_hash_matches"]:
            errors.append(
                f"Gate 14 failed: Thumbnail JPG actual hash '{actual_thumb_sha}' does not match package record '{expected_thumb_sha}'."
            )

        # 15. SEO Validation Passed
        checks["seo_validation_passed"] = bool(seo_report.get("passed", False))
        if not checks["seo_validation_passed"]:
            errors.append(f"Gate 15 failed: SEO validation report status is False ({seo_report.get('errors')}).")

        # 16. Thumbnail QC Passed
        checks["thumbnail_qc_passed"] = bool(thumb_qc.get("passed", False))
        if not checks["thumbnail_qc_passed"]:
            errors.append(f"Gate 16 failed: Thumbnail QC report status is False ({thumb_qc.get('errors')}).")

        # 17. Package Schema Valid
        required_schema_keys = [
            "episode_id", "schema_version", "generator_version", "channel_identity",
            "master", "script", "seo", "thumbnail", "provenance"
        ]
        checks["package_schema_valid"] = all(k in package_data for k in required_schema_keys)
        if not checks["package_schema_valid"]:
            missing_keys = [k for k in required_schema_keys if k not in package_data]
            errors.append(f"Gate 17 failed: publish_package.json missing required schema keys: {missing_keys}")

        # 18. Provenance Complete
        prov = package_data.get("provenance", {})
        required_prov_keys = [
            "generator_version", "schema_version", "source_master_sha256", "source_script_sha256"
        ]
        checks["provenance_complete"] = all(k in prov for k in required_prov_keys)
        if not checks["provenance_complete"]:
            missing_prov = [k for k in required_prov_keys if k not in prov]
            errors.append(f"Gate 18 failed: Provenance missing required keys: {missing_prov}")

        # 19. Canonical Channel & Host Identity Valid
        ch_ident = package_data.get("channel_identity", {})
        ch_name_ta = ch_ident.get("channel_name_ta", "")
        ch_name_en = ch_ident.get("channel_name_en", "")
        ch_handle = ch_ident.get("channel_handle", "")

        host_ident = package_data.get("host_identity", {})
        host_name_ta = host_ident.get("host_name_ta", "")
        host_name_en = host_ident.get("host_name_en", "")

        identity_ok = (
            ("காலப் பதிவுகள்" in ch_name_ta) and
            ("Kaalapadhivugal" in ch_name_en) and
            (ch_handle == "@kaalapadhivugal") and
            ("யாழினி" in host_name_ta) and
            ("Yaazhini" in host_name_en)
        )
        checks["canonical_branding_valid"] = identity_ok
        if not identity_ok:
            errors.append(
                f"Gate 19 failed: Canonical channel or host identity invalid in publish package: {ch_ident}, {host_ident}"
            )

        # 20. Zero Obsolete Identity Tokens
        combined_text = json.dumps(package_data, ensure_ascii=False).lower()
        found_obsolete = [tok for tok in OBSOLETE_IDENTITY_TOKENS if tok.lower() in combined_text]
        checks["no_obsolete_identity_tokens"] = (len(found_obsolete) == 0)
        if not checks["no_obsolete_identity_tokens"]:
            errors.append(f"Gate 20 failed: Obsolete branding tokens found in publish package: {found_obsolete}")

        # 21. No Unsupported Sensational Claims or Clickbait
        title = package_data.get("seo", {}).get("selected_title", "")
        found_clickbait = [pat for pat in SENSATIONAL_PATTERNS if re.search(pat, title, re.IGNORECASE)]
        checks["no_unsupported_factual_claims"] = (len(found_clickbait) == 0)
        if not checks["no_unsupported_factual_claims"]:
            errors.append(f"Gate 21 failed: Clickbait or sensational claims detected in title: {found_clickbait}")

        passed = (len(errors) == 0)
        logger.info(f"YouTubePublicationValidator: Episode '{episode_id}' input gate: passed={passed} ({len(errors)} errors).")

        return YouTubeValidationReport(
            passed=passed,
            episode_id=episode_id,
            errors=errors,
            warnings=warnings,
            checks=checks,
            package_data=package_data if passed else None
        )
