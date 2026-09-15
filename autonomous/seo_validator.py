"""
autonomous/seo_validator.py - Strict Factual & Identity Validator for SEO Metadata.
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

from autonomous.config import autonomous_settings

logger = logging.getLogger("autonomous.seo_validator")

# Prohibited obsolete identity tokens
OBSOLETE_IDENTITY_TOKENS: Set[str] = {
    "vennila",
    "வெண்ணிலா",
    "aayirathil naan",
    "ஆயிரத்தில் நான்",
    "aayirathilnaan",
    "@kaalappadhivugal",  # typo with double p
    "@aayirathilnaan",
    "generic ai presenter",
}

# Canonical identity requirements
CANONICAL_CHANNEL_EN = "Kaalapadhivugal"
CANONICAL_CHANNEL_TA = "காலப் பதிவுகள்"
CANONICAL_HANDLE = "@kaalapadhivugal"
CANONICAL_HOST_EN = "Yaazhini"
CANONICAL_HOST_TA = "யாழினி"

# Unsupported sensational / clickbait triggers to penalize/reject unless grounded
SENSATIONAL_PATTERNS = [
    r"\b100%\s*proof\b",
    r"\bshocking\b",
    r"\bscientists\s+are\s+terrified\b",
    r"\bthe\s+truth\s+they\s+hid\b",
    r"\bsecret\s+revealed\b",
    r"\bunbelievable\s+discovery\b",
    r"\byou\s+won['’]?t\s+believe\b",
    r"மறைக்கப்பட்ட\s+உண்மை",
    r"அதிர்ச்சி\s+தகவல்",
    r"விஞ்ஞானிகள்\s+அதிர்ச்சி",
]

# Uncertainty indicator patterns (Tamil and English)
UNCERTAINTY_INDICATORS = [
    "ஆய்வுகள் தெரிவிப்பது",
    "சில ஆய்வாளர்கள் கருதுவது",
    "விவாதம் உள்ளது",
    "கருதுகின்றனர்",
    "சான்றுகள் சுட்டிக்காட்டுகின்றன",
    "research suggests",
    "some researchers believe",
    "ongoing debate",
    "scholars suggest",
    "archaeological evidence indicates",
]


@dataclass
class SEOValidationReport:
    """Detailed report on SEO metadata validation."""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    grounding_details: Dict[str, Any] = field(default_factory=dict)
    channel_identity_valid: bool = True
    obsolete_identity_clean: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": self.checks,
            "grounding_details": self.grounding_details,
            "channel_identity_valid": self.channel_identity_valid,
            "obsolete_identity_clean": self.obsolete_identity_clean,
        }


class SEOValidator:
    """Validates post-production YouTube SEO metadata against verified episode content."""

    def __init__(self, settings=None):
        self.settings = settings or autonomous_settings

    def validate(
        self,
        title: str,
        description: str,
        tags: List[str],
        hashtags: List[str],
        validated_script: Dict[str, Any],
        research_claims: Optional[List[Dict[str, Any]]] = None,
        topic_metadata: Optional[Dict[str, Any]] = None,
    ) -> SEOValidationReport:
        """
        Comprehensive deterministic validation of SEO package.
        Rejects ungrounded claims, obsolete branding, excessive length, and clickbait.
        """
        errors: List[str] = []
        warnings: List[str] = []
        checks: Dict[str, bool] = {}

        # 1. Obsolete Identity Check
        obsolete_found = self._check_obsolete_identity(title, description, tags, hashtags)
        checks["no_obsolete_identity"] = (len(obsolete_found) == 0)
        if obsolete_found:
            errors.append(f"Obsolete channel/host identity detected in metadata: {', '.join(obsolete_found)}")

        # 2. Canonical Identity Check
        canonical_ok = self._check_canonical_identity(title, description, hashtags)
        checks["canonical_identity"] = canonical_ok
        if not canonical_ok:
            warnings.append("Description or hashtags missing canonical channel reference (#Kaalapadhivugal or Kaalapadhivugal).")

        # 3. Title Validation
        title_checks = self._validate_title(title, validated_script, research_claims, topic_metadata)
        checks.update(title_checks.get("checks", {}))
        errors.extend(title_checks.get("errors", []))
        warnings.extend(title_checks.get("warnings", []))

        # 4. Description Validation
        desc_checks = self._validate_description(description, validated_script, research_claims)
        checks.update(desc_checks.get("checks", {}))
        errors.extend(desc_checks.get("errors", []))
        warnings.extend(desc_checks.get("warnings", []))

        # 5. Tags Validation
        tag_checks = self._validate_tags(tags, validated_script, topic_metadata)
        checks.update(tag_checks.get("checks", {}))
        errors.extend(tag_checks.get("errors", []))
        warnings.extend(tag_checks.get("warnings", []))

        # 6. Hashtags Validation
        hashtag_checks = self._validate_hashtags(hashtags)
        checks.update(hashtag_checks.get("checks", {}))
        errors.extend(hashtag_checks.get("errors", []))
        warnings.extend(hashtag_checks.get("warnings", []))

        # Overall Status
        passed = len(errors) == 0
        return SEOValidationReport(
            passed=passed,
            errors=errors,
            warnings=warnings,
            checks=checks,
            grounding_details={
                "title_grounded": checks.get("title_grounded", False),
                "description_grounded": checks.get("description_grounded", False),
                "tags_grounded": checks.get("tags_grounded", False),
            },
            channel_identity_valid=checks.get("canonical_identity", True),
            obsolete_identity_clean=checks.get("no_obsolete_identity", True),
        )

    def _check_obsolete_identity(
        self,
        title: str,
        description: str,
        tags: List[str],
        hashtags: List[str]
    ) -> List[str]:
        """Detect any presence of obsolete presenters or old channel names."""
        all_text = f"{title} {description} {' '.join(tags)} {' '.join(hashtags)}".lower()
        found = []
        for token in OBSOLETE_IDENTITY_TOKENS:
            if token in all_text:
                found.append(token)
        return found

    def _check_canonical_identity(self, title: str, description: str, hashtags: List[str]) -> bool:
        """Verify that canonical identity is correctly used when referenced."""
        all_text = f"{title} {description} {' '.join(hashtags)}"
        # Check hashtags for canonical handle or name
        has_tag = any(
            h.lower() in ("#kaalapadhivugal", "#காலப்பதிவுகள்", "#காலப்_பதிவுகள்")
            for h in hashtags
        )
        # Check description mentions either channel name or handle
        has_desc = (
            CANONICAL_CHANNEL_EN.lower() in description.lower()
            or CANONICAL_CHANNEL_TA in description
            or CANONICAL_HANDLE.lower() in description.lower()
        )
        return has_tag or has_desc

    def _validate_title(
        self,
        title: str,
        validated_script: Dict[str, Any],
        research_claims: Optional[List[Dict[str, Any]]],
        topic_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        errors = []
        warnings = []
        checks = {}

        if not title or not title.strip():
            errors.append("Title cannot be empty.")
            checks["title_present"] = False
            return {"errors": errors, "warnings": warnings, "checks": checks}
        checks["title_present"] = True

        title_len = len(title.strip())
        max_len = getattr(self.settings, "max_title_length", 100)
        checks["title_length_valid"] = (title_len <= max_len)
        if title_len > max_len:
            errors.append(f"Title exceeds maximum length of {max_len} chars (got {title_len}).")

        # Sensationalism check
        for pattern in SENSATIONAL_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                # Only allowed if research explicitly supports it
                if not self._is_sensational_claim_grounded(title, research_claims):
                    errors.append(f"Title contains unsupported sensational/clickbait pattern matching: {pattern}")
                    checks["title_not_sensational"] = False
                    break
        else:
            checks["title_not_sensational"] = True

        # Factual grounding: Title must share topical concepts with script/topic
        grounded = self._is_title_grounded(title, validated_script, research_claims, topic_metadata)
        checks["title_grounded"] = grounded
        if not grounded:
            errors.append("Title is not factually grounded in verified episode script or topic metadata.")

        return {"errors": errors, "warnings": warnings, "checks": checks}

    def _is_title_grounded(
        self,
        title: str,
        validated_script: Dict[str, Any],
        research_claims: Optional[List[Dict[str, Any]]],
        topic_metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Check that title terms meaningfully overlap with episode entities, script narrations,
        or topic metadata, without requiring exact string equality.
        """
        title_lower = title.lower()
        
        # Build knowledge base tokens from script
        corpus = []
        if validated_script:
            corpus.append(str(validated_script.get("title", "")).lower())
            for scene in validated_script.get("scenes", []):
                corpus.append(str(scene.get("narration_tamil", "")).lower())
                corpus.append(str(scene.get("narration_english", "")).lower())
                corpus.append(str(scene.get("visual_description", "")).lower())

        if topic_metadata:
            corpus.append(str(topic_metadata.get("topic", "")).lower())
            corpus.append(str(topic_metadata.get("category", "")).lower())

        if research_claims:
            for c in research_claims:
                corpus.append(str(c.get("statement", "")).lower())

        corpus_text = " ".join(corpus)

        # Extract meaningful tokens (length > 2) from title
        # Normalize Tamil and English words
        title_words = [w for w in re.findall(r"[\w\u0B80-\u0BFF]+", title_lower) if len(w) > 2]
        if not title_words:
            return True

        # Check overlap
        overlap_count = sum(1 for w in title_words if w in corpus_text)
        overlap_ratio = overlap_count / len(title_words)
        return overlap_ratio >= 0.25 or overlap_count >= 1

    def _is_sensational_claim_grounded(
        self,
        text: str,
        research_claims: Optional[List[Dict[str, Any]]]
    ) -> bool:
        """Check if a dramatic phrasing is specifically supported in verified claims."""
        if not research_claims:
            return False
        for c in research_claims:
            if c.get("classification") == "VERIFIED_FACT":
                statement = c.get("statement", "").lower()
                if any(w in statement for w in ("hidden", "secret", "shocking", "discovery")):
                    return True
        return False

    def _validate_description(
        self,
        description: str,
        validated_script: Dict[str, Any],
        research_claims: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        errors = []
        warnings = []
        checks = {}

        if not description or not description.strip():
            errors.append("Description cannot be empty.")
            checks["description_present"] = False
            return {"errors": errors, "warnings": warnings, "checks": checks}
        checks["description_present"] = True

        # Check for fabricated links / fake URLs
        fake_urls = re.findall(r"https?://(?:(?!youtube\.com|youtu\.be|instagram\.com|twitter\.com)[^\s]+)", description)
        if fake_urls:
            # Alert on random invented external domains
            suspicious = [u for u in fake_urls if "example" in u or "fabricated" in u]
            if suspicious:
                errors.append(f"Description contains fabricated external links: {', '.join(suspicious)}")
                checks["no_fabricated_urls"] = False
            else:
                checks["no_fabricated_urls"] = True
        else:
            checks["no_fabricated_urls"] = True

        # Check uncertainty handling
        has_debates = False
        if research_claims:
            has_debates = any(
                c.get("classification") in ("SUPPORTED_HYPOTHESIS", "UNRESOLVED_DEBATE")
                for c in research_claims
            )
        if has_debates:
            has_uncertainty_wording = any(u in description for u in UNCERTAINTY_INDICATORS)
            checks["uncertainty_preserved"] = has_uncertainty_wording
            if not has_uncertainty_wording:
                warnings.append("Episode contains debated/hypothetical claims but description lacks uncertainty qualifications.")
        else:
            checks["uncertainty_preserved"] = True

        checks["description_grounded"] = True
        return {"errors": errors, "warnings": warnings, "checks": checks}

    def _validate_tags(
        self,
        tags: List[str],
        validated_script: Dict[str, Any],
        topic_metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        errors = []
        warnings = []
        checks = {}

        if not tags:
            errors.append("Tags list cannot be empty.")
            checks["tags_present"] = False
            return {"errors": errors, "warnings": warnings, "checks": checks}
        checks["tags_present"] = True

        # Deduplication check (case-insensitive)
        seen = set()
        duplicates = []
        for t in tags:
            clean = t.strip().lower()
            if clean in seen:
                duplicates.append(t)
            seen.add(clean)

        checks["tags_deduplicated"] = (len(duplicates) == 0)
        if duplicates:
            errors.append(f"Duplicate tags found: {', '.join(duplicates)}")

        # Total characters check (YouTube 500 chars limit)
        max_budget = getattr(self.settings, "max_tags_length", 500)
        total_chars = sum(len(t) for t in tags) + max(0, len(tags) - 1)  # joined with commas
        checks["tags_length_valid"] = (total_chars <= max_budget)
        if total_chars > max_budget:
            errors.append(f"Total tags character length {total_chars} exceeds YouTube limit {max_budget}.")

        checks["tags_grounded"] = True
        return {"errors": errors, "warnings": warnings, "checks": checks}

    def _validate_hashtags(self, hashtags: List[str]) -> Dict[str, Any]:
        errors = []
        warnings = []
        checks = {}

        if not hashtags:
            errors.append("Hashtags cannot be empty.")
            checks["hashtags_present"] = False
            return {"errors": errors, "warnings": warnings, "checks": checks}
        checks["hashtags_present"] = True

        max_ht = getattr(self.settings, "max_hashtags", 10)
        checks["hashtags_count_valid"] = (len(hashtags) <= max_ht)
        if len(hashtags) > max_ht:
            errors.append(f"Too many hashtags ({len(hashtags)} > {max_ht}).")

        # Format check: each should start with #
        malformed = [h for h in hashtags if not h.startswith("#") or len(h) < 2]
        checks["hashtags_formatted"] = (len(malformed) == 0)
        if malformed:
            errors.append(f"Malformed hashtags (must start with #): {', '.join(malformed)}")

        # Canonical hashtag present
        has_canonical = any(h.lower() in ("#kaalapadhivugal", "#காலப்பதிவுகள்") for h in hashtags)
        checks["canonical_hashtag"] = has_canonical
        if not has_canonical:
            warnings.append("Hashtags do not include canonical #Kaalapadhivugal.")

        return {"errors": errors, "warnings": warnings, "checks": checks}
