"""
autonomous/seo_engine.py - Autonomous SEO & Metadata Generation Engine.
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).

Consumes MASTER_READY episodes, validated scripts, and verified claims.
Generates structured, grounded, and verified YouTube publishing metadata.
"""

import os
import re
import json
import time
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

import requests

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.seo_validator import SEOValidator, SEOValidationReport

logger = logging.getLogger("autonomous.seo_engine")


@dataclass
class TitleCandidate:
    """A scored candidate YouTube title."""
    title: str
    language: str
    score: float
    reasons: List[str] = field(default_factory=list)
    is_clickbait: bool = False
    is_grounded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DescriptionResult:
    """Structured YouTube description."""
    hook: str
    summary: str
    context: str
    uncertainty_note: str
    channel_branding: str
    cta: str
    full_description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TagSet:
    """Structured and deduplicated tags."""
    accepted_tags: List[str]
    rejected_tags: List[Dict[str, str]]
    total_characters: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HashtagSet:
    """Structured hashtags."""
    hashtags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SEOResult:
    """Complete, verified SEO metadata package."""
    episode_id: str
    selected_title: str
    title_candidates: List[TitleCandidate]
    description: str
    description_structured: Dict[str, str]
    tags: List[str]
    rejected_tags: List[Dict[str, str]]
    total_tag_chars: int
    hashtags: List[str]
    language: str = "ta"
    secondary_language: Optional[str] = "en"
    category: str = "Education"
    category_id: int = 27
    llm_used: bool = False
    model_name: Optional[str] = None
    fallback_used: bool = True
    fallback_reason: Optional[str] = None
    source_master_sha256: str = ""
    source_script_sha256: str = ""
    source_validated_script_sha256: str = ""
    generator_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "selected_title": self.selected_title,
            "title_candidates": [c.to_dict() for c in self.title_candidates],
            "description": self.description,
            "description_structured": self.description_structured,
            "tags": self.tags,
            "rejected_tags": self.rejected_tags,
            "total_tag_chars": self.total_tag_chars,
            "tag_count": len(self.tags),
            "hashtags": self.hashtags,
            "language": self.language,
            "secondary_language": self.secondary_language,
            "category": self.category,
            "category_id": self.category_id,
            "provenance": {
                "llm_used": self.llm_used,
                "model_name": self.model_name,
                "fallback_used": self.fallback_used,
                "fallback_reason": self.fallback_reason,
                "source_master_sha256": self.source_master_sha256,
                "source_script_sha256": self.source_script_sha256,
                "source_validated_script_sha256": self.source_validated_script_sha256,
                "generator_version": self.generator_version,
                "schema_version": self.schema_version,
                "created_at": self.created_at,
            }
        }


def _sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class AutonomousSEOEngine:
    """Generates and validates autonomous post-production SEO metadata for YouTube."""

    def __init__(self, state_manager: Optional[StateManager] = None, settings=None):
        self.state_manager = state_manager or StateManager()
        self.settings = settings or autonomous_settings
        self.validator = SEOValidator(self.settings)

    def validate_master_ready_gate(self, episode_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Verify that episode is in MASTER_READY state and all required master/script
        artifacts exist with verified cryptographic integrity.
        """
        episode = self.state_manager.get_episode(episode_id)
        if not episode:
            return False, f"Episode '{episode_id}' not found in database.", {}

        # 1. State check
        if episode.status == EpisodeState.REVIEW_REQUIRED.value or episode.current_stage == EpisodeState.REVIEW_REQUIRED.value:
            return False, f"Episode '{episode_id}' is in REVIEW_REQUIRED state. Rejected by Phase 11 gate.", {}

        if episode.status == EpisodeState.FAILED.value or episode.current_stage == EpisodeState.FAILED.value:
            return False, f"Episode '{episode_id}' is in FAILED state. Rejected by Phase 11 gate.", {}

        if episode.status != EpisodeState.MASTER_READY.value and episode.current_stage != EpisodeState.MASTER_READY.value:
            return False, f"Episode '{episode_id}' state is {episode.status}. Phase 11 requires MASTER_READY.", {}

        ep_dir = Path(episode.output_directory)
        if not ep_dir.is_absolute():
            ep_dir = BASE_DIR / ep_dir

        master_dir = ep_dir / "master"
        script_dir = ep_dir / "script"

        master_mp4 = master_dir / "master.mp4"
        master_manifest_path = master_dir / "master_manifest.json"
        master_report_path = master_dir / "master_generation_report.json"
        validated_script_path = script_dir / "validated_script.json"
        script_path = script_dir / "script.json"

        # Check required files
        if not master_mp4.exists() or master_mp4.stat().st_size == 0:
            return False, f"master.mp4 missing or empty for '{episode_id}'.", {}

        if not master_manifest_path.exists():
            return False, f"master_manifest.json missing for '{episode_id}'.", {}

        if not master_report_path.exists():
            return False, f"master_generation_report.json missing for '{episode_id}'.", {}

        if not validated_script_path.exists() and not script_path.exists():
            return False, f"No script or validated_script.json found for '{episode_id}'.", {}

        # 2. Manifest integrity & Master Hash verification
        try:
            with open(master_manifest_path, "r", encoding="utf-8") as f:
                master_manifest = json.load(f)
        except Exception as e:
            return False, f"Corrupt master_manifest.json: {e}", {}

        expected_master_hash = master_manifest.get("master_video_sha256")
        if not expected_master_hash:
            return False, "master_manifest.json does not contain 'master_video_sha256'.", {}

        actual_master_hash = _sha256_file(master_mp4)
        if actual_master_hash != expected_master_hash:
            return False, f"Master video hash mismatch (expected {expected_master_hash}, got {actual_master_hash}).", {}

        # 3. Read validated script
        active_script_path = validated_script_path if validated_script_path.exists() else script_path
        try:
            with open(active_script_path, "r", encoding="utf-8") as f:
                script_data = json.load(f)
        except Exception as e:
            return False, f"Corrupt script JSON: {e}", {}

        return True, "MASTER_READY gate passed.", {
            "ep_dir": ep_dir,
            "master_mp4": master_mp4,
            "master_hash": actual_master_hash,
            "master_manifest": master_manifest,
            "master_manifest_hash": _sha256_file(master_manifest_path),
            "script_data": script_data,
            "script_path": active_script_path,
            "script_hash": _sha256_file(active_script_path),
            "validated_script_hash": _sha256_file(validated_script_path) if validated_script_path.exists() else _sha256_file(script_path),
            "episode": episode,
        }

    def generate_seo(self, episode_id: str, offline: bool = False) -> Tuple[bool, Optional[SEOResult], Optional[SEOValidationReport], str]:
        """
        Execute Phase 11 SEO generation:
        MASTER_READY -> METADATA_GENERATING -> METADATA_READY
        """
        # 1. Gate check
        valid, msg, gate_data = self.validate_master_ready_gate(episode_id)
        if not valid:
            logger.error(f"Phase 11 input gate failed for '{episode_id}': {msg}")
            return False, None, None, msg

        ep_dir = gate_data["ep_dir"]
        script_data = gate_data["script_data"]
        master_hash = gate_data["master_hash"]
        script_hash = gate_data["script_hash"]
        validated_script_hash = gate_data["validated_script_hash"]
        episode = gate_data["episode"]

        # Transition to METADATA_GENERATING
        try:
            self.state_manager.transition(episode_id, EpisodeState.METADATA_GENERATING)
        except Exception as e:
            logger.warning(f"Could not transition to METADATA_GENERATING: {e}")

        # Load claims/evidence if available
        research_claims = []
        evidence_file = ep_dir / "research" / "research_evidence.json"
        if evidence_file.exists():
            try:
                with open(evidence_file, "r", encoding="utf-8") as f:
                    ev_data = json.load(f)
                    research_claims = ev_data.get("claims", [])
            except Exception:
                pass

        topic_meta = {
            "topic": episode.topic,
            "category": episode.category,
        }

        # 2. LLM or Deterministic Candidate Generation
        llm_used = False
        model_name = getattr(self.settings, "ollama_model", "gemma3:4b")
        fallback_used = True
        fallback_reason = "Offline mode requested."

        llm_output = None
        if not offline:
            llm_output, model_name, err = self._query_ollama_seo(script_data, topic_meta, research_claims)
            if llm_output:
                llm_used = True
                fallback_used = False
                fallback_reason = None
            else:
                fallback_reason = f"Ollama unavailable or unparseable: {err}"

        # 3. Build Title Candidates
        candidates = self._generate_title_candidates(script_data, topic_meta, research_claims, llm_output)

        # 4. Score Candidates Deterministically
        scored_candidates = self._score_title_candidates(candidates, script_data, topic_meta, research_claims)
        # Select best candidate
        best_candidate = scored_candidates[0] if scored_candidates else TitleCandidate(
            title=str(script_data.get("title", episode.topic)),
            language="ta",
            score=100.0,
            reasons=["Default fallback title"]
        )

        # 5. Build Description
        description_res = self._generate_description(script_data, topic_meta, research_claims, llm_output)

        # 6. Build Tags
        tag_set = self._generate_tags(script_data, topic_meta, research_claims, llm_output)

        # 7. Build Hashtags
        hashtag_set = self._generate_hashtags(script_data, topic_meta, llm_output)

        # 8. Validate SEO Package
        validation_report = self.validator.validate(
            title=best_candidate.title,
            description=description_res.full_description,
            tags=tag_set.accepted_tags,
            hashtags=hashtag_set.hashtags,
            validated_script=script_data,
            research_claims=research_claims,
            topic_metadata=topic_meta,
        )

        if not validation_report.passed:
            err_summary = "; ".join(validation_report.errors)
            logger.error(f"SEO Validation failed for '{episode_id}': {err_summary}")
            try:
                self.state_manager.transition(
                    episode_id,
                    EpisodeState.REVIEW_REQUIRED,
                    error_message=f"SEO_VALIDATION_FAILED: {err_summary}"
                )
            except Exception:
                pass
            return False, None, validation_report, f"SEO validation failed: {err_summary}"

        # 9. Assemble SEOResult
        now_iso = datetime.now(timezone.utc).isoformat()
        seo_result = SEOResult(
            episode_id=episode_id,
            selected_title=best_candidate.title,
            title_candidates=scored_candidates,
            description=description_res.full_description,
            description_structured=description_res.to_dict(),
            tags=tag_set.accepted_tags,
            rejected_tags=tag_set.rejected_tags,
            total_tag_chars=tag_set.total_characters,
            hashtags=hashtag_set.hashtags,
            language="ta",
            secondary_language="en",
            category="Education",
            category_id=27,
            llm_used=llm_used,
            model_name=model_name,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            source_master_sha256=master_hash,
            source_script_sha256=script_hash,
            source_validated_script_sha256=validated_script_hash,
            generator_version="1.0.0",
            schema_version="1.0.0",
            created_at=now_iso,
        )

        # 10. Atomic Write to publish directory
        publish_dir = ep_dir / "publish"
        publish_dir.mkdir(parents=True, exist_ok=True)

        seo_json_path = publish_dir / "seo.json"
        tmp_seo_json = publish_dir / "seo.tmp.json"
        with open(tmp_seo_json, "w", encoding="utf-8") as f:
            json.dump(seo_result.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp_seo_json, seo_json_path)

        seo_report_path = publish_dir / "seo_validation_report.json"
        tmp_report = publish_dir / "seo_validation_report.tmp.json"
        with open(tmp_report, "w", encoding="utf-8") as f:
            json.dump(validation_report.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp_report, seo_report_path)

        # Transition to METADATA_READY
        try:
            self.state_manager.transition(episode_id, EpisodeState.METADATA_READY)
            self.state_manager.update_episode(
                episode_id,
                metadata_path=str(seo_json_path)
            )
        except Exception as e:
            logger.error(f"Failed to transition to METADATA_READY: {e}")
            return False, seo_result, validation_report, str(e)

        logger.info(f"Phase 11: SEO Metadata successfully generated and validated for '{episode_id}'.")
        return True, seo_result, validation_report, "SEO metadata generated successfully."

    def _query_ollama_seo(
        self,
        script_data: Dict[str, Any],
        topic_meta: Dict[str, Any],
        research_claims: List[Dict[str, Any]]
    ) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
        """Call Ollama to propose title, description, and tag candidates."""
        url = getattr(self.settings, "ollama_url", "http://localhost:11434").rstrip("/") + "/api/generate"
        model_name = getattr(self.settings, "ollama_model", "gemma3:4b")

        topic = topic_meta.get("topic", "")
        script_title = script_data.get("title", topic)
        narrations = []
        for s in script_data.get("scenes", []):
            narrations.append(s.get("narration_tamil", ""))
            narrations.append(s.get("narration_english", ""))
        context_text = " ".join(narrations)[:1000]

        prompt = f"""You are the metadata assistant for the Tamil documentary channel 'Kaalapadhivugal' (@kaalapadhivugal).
Topic: "{topic}"
Script Title: "{script_title}"
Episode Context: "{context_text}"

Generate JSON with:
- "titles": list of 5-8 documentary title candidates in Tamil (or Tamil with English keywords), under 90 characters, factual, intriguing, no clickbait.
- "hook": 1-2 sentence compelling factual opening hook.
- "summary": 2-3 sentence documentary summary grounded in the context.
- "tags": list of 15-20 relevant search tags (mix of Tamil, English transliteration, historical period, archaeology terms).
- "hashtags": list of 4-6 hashtags (must include #Kaalapadhivugal).

Format: JSON only."""

        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.3,
                "top_p": 0.8
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "").strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            data = json.loads(raw_text.strip())
            return data, model_name, None
        except Exception as e:
            return None, model_name, str(e)

    def _generate_title_candidates(
        self,
        script_data: Dict[str, Any],
        topic_meta: Dict[str, Any],
        research_claims: List[Dict[str, Any]],
        llm_output: Optional[Dict[str, Any]]
    ) -> List[TitleCandidate]:
        """Generate 5-10 title candidates using LLM proposals plus deterministic templates."""
        candidates: List[TitleCandidate] = []
        seen = set()

        # 1. Add LLM proposals if available
        if llm_output and isinstance(llm_output.get("titles"), list):
            for t in llm_output["titles"]:
                clean = str(t).strip()
                if clean and clean not in seen:
                    candidates.append(TitleCandidate(
                        title=clean,
                        language="ta",
                        score=0.0,
                        reasons=["LLM candidate"]
                    ))
                    seen.add(clean)

        # 2. Add Deterministic Factual Candidates
        base_title = script_data.get("title", topic_meta.get("topic", "வரலாற்றுப் பதிவு"))
        topic = topic_meta.get("topic", base_title)

        deterministic_templates = [
            f"{base_title}",
            f"{base_title} — வரலாற்றுச் சான்றுகள்",
            f"{topic}: மறைந்த வரலாறு மற்றும் தொல்லியல் சான்றுகள்",
            f"{base_title} | Kaalapadhivugal Records of Time",
            f"{topic} — ஓர் ஆவணப் பார்வை",
            f"{base_title} | தமிழ் வரலாற்றுப் பதிவுகள்",
            f"{topic}: பண்டைய துறைமுகமும் கடல்சார் வணிகமும்",
        ]

        for dt in deterministic_templates:
            clean = dt.strip()
            if clean and clean not in seen:
                candidates.append(TitleCandidate(
                    title=clean,
                    language="ta",
                    score=0.0,
                    reasons=["Deterministic factual template"]
                ))
                seen.add(clean)

        return candidates[:10]

    def _score_title_candidates(
        self,
        candidates: List[TitleCandidate],
        script_data: Dict[str, Any],
        topic_meta: Dict[str, Any],
        research_claims: List[Dict[str, Any]]
    ) -> List[TitleCandidate]:
        """Score each title candidate deterministically."""
        scored = []
        max_len = getattr(self.settings, "max_title_length", 100)

        for cand in candidates:
            title = cand.title
            score = 50.0  # baseline
            reasons = []

            # Length penalty/reward
            t_len = len(title)
            if t_len > max_len:
                score -= 100.0
                reasons.append(f"Exceeds max length ({t_len} > {max_len})")
            elif 30 <= t_len <= 85:
                score += 15.0
                reasons.append("Optimal title length (30-85 chars)")
            elif t_len < 15:
                score -= 20.0
                reasons.append("Too brief")

            # Sensationalism / Clickbait penalty
            sensational_found = False
            for pattern in [r"\b100%\s*proof\b", r"\bshocking\b", r"அதிர்ச்சி", r"மறைக்கப்பட்ட\s+உண்மை"]:
                if re.search(pattern, title, re.IGNORECASE):
                    sensational_found = True
                    score -= 40.0
                    reasons.append("Contains unverified sensational phrasing")
                    break
            cand.is_clickbait = sensational_found

            # Factual Grounding check
            is_grounded = self.validator._is_title_grounded(title, script_data, research_claims, topic_meta)
            cand.is_grounded = is_grounded
            if is_grounded:
                score += 25.0
                reasons.append("Grounded in episode topic/script")
            else:
                score -= 50.0
                reasons.append("Lacks grounding in episode content")

            # Tamil context reward
            has_tamil = bool(re.search(r"[\u0B80-\u0BFF]", title))
            if has_tamil:
                score += 15.0
                reasons.append("Features native Tamil typography")

            # Documentary quality (clean punctuation, branding)
            if "—" in title or "|" in title or ":" in title:
                score += 10.0
                reasons.append("Documentary formatting")

            cand.score = round(score, 1)
            cand.reasons.extend(reasons)
            scored.append(cand)

        # Sort descending by score
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def _generate_description(
        self,
        script_data: Dict[str, Any],
        topic_meta: Dict[str, Any],
        research_claims: List[Dict[str, Any]],
        llm_output: Optional[Dict[str, Any]]
    ) -> DescriptionResult:
        """Construct structured, factual YouTube description."""
        topic = topic_meta.get("topic", script_data.get("title", ""))
        channel_cfg = self.settings.channel

        # 1. Opening Hook
        if llm_output and llm_output.get("hook"):
            hook = str(llm_output["hook"]).strip()
        else:
            hook = f"பண்டைய தமிழகத்தின் பெருமைமிகு வரலாற்றை, தொல்லியல் மற்றும் வரலாற்றுச் சான்றுகளுடன் ஆராயும் ஆவணப் பதிவு: {topic}."

        # 2. Episode Summary
        summaries = []
        for s in script_data.get("scenes", []):
            ta = s.get("narration_tamil", "")
            if ta and not s.get("scene_type") in ("host_intro", "host_outro"):
                summaries.append(ta)
        summary_text = " ".join(summaries) if summaries else f"{topic} குறித்த விரிவான வரலாற்றுப் பதிவு."

        # 3. Historical Context
        context_text = f"தொல்லியல் சான்றுகள், இலக்கியக் குறிப்புகள் மற்றும் வரலாற்று ஆய்வுகளின் அடிப்படையில் இந்த ஆவணம் உருவாக்கப்பட்டுள்ளது."

        # 4. Uncertainty / Debate Disclosure
        has_debates = any(
            c.get("classification") in ("SUPPORTED_HYPOTHESIS", "UNRESOLVED_DEBATE")
            for c in research_claims
        )
        if has_debates:
            uncertainty_note = "ஆய்வுகள் தெரிவிப்பது மற்றும் வரலாற்றாசிரியர்களின் கருத்துக்களின் அடிப்படையில், மாற்று வாதங்களும் ஆய்வுக்கு உட்படுத்தப்பட்டுள்ளன."
        else:
            uncertainty_note = "சான்றுகளின் அடிப்படையில் வரலாற்று உண்மைகள் பதிவு செய்யப்பட்டுள்ளன."

        # 5. Channel Branding & Host
        branding = (
            f"சேனல்: {channel_cfg.channel_name_ta} ({channel_cfg.channel_name_en})\n"
            f"ஹேண்டில்: {channel_cfg.channel_handle}\n"
            f"வழங்குபவர்: {channel_cfg.host_name_ta} ({channel_cfg.host_name_en})\n"
            f"நோக்கம்: {channel_cfg.tagline_ta}"
        )

        # 6. Call to action
        cta = (
            f"நமது வரலாற்றை அடுத்த தலைமுறைக்குக் கொண்டு சேர்க்க, {channel_cfg.channel_handle} பக்கத்தைப் பின்தொடருங்கள்.\n"
            f"உங்கள் கருத்துக்களை கீழே பதிவிடுங்கள்!"
        )

        # Full Description Assembly
        full_desc = (
            f"{hook}\n\n"
            f"📌 ஆவணக் சுருக்கம்:\n{summary_text}\n\n"
            f"🏛️ வரலாற்றுப் பின்னணி:\n{context_text}\n\n"
            f"🔍 ஆய்வுத் தகவல்:\n{uncertainty_note}\n\n"
            f"-----------------------------------------\n"
            f"{branding}\n"
            f"-----------------------------------------\n\n"
            f"{cta}\n\n"
            f"#Kaalapadhivugal #TamilHistory #Archaeology"
        )

        return DescriptionResult(
            hook=hook,
            summary=summary_text,
            context=context_text,
            uncertainty_note=uncertainty_note,
            channel_branding=branding,
            cta=cta,
            full_description=full_desc,
        )

    def _generate_tags(
        self,
        script_data: Dict[str, Any],
        topic_meta: Dict[str, Any],
        research_claims: List[Dict[str, Any]],
        llm_output: Optional[Dict[str, Any]]
    ) -> TagSet:
        """Generate deduplicated tags within YouTube 500 characters limit."""
        max_budget = getattr(self.settings, "max_tags_length", 500)
        raw_candidates = []

        # LLM proposed tags
        if llm_output and isinstance(llm_output.get("tags"), list):
            raw_candidates.extend(llm_output["tags"])

        # Core topic and identity tags
        topic = topic_meta.get("topic", "")
        category = topic_meta.get("category", "")
        script_title = script_data.get("title", "")

        default_tags = [
            "Kaalapadhivugal",
            "காலப் பதிவுகள்",
            "@kaalapadhivugal",
            "Yaazhini",
            "யாழினி",
            topic,
            category,
            script_title,
            "Tamil History",
            "Ancient Tamil Civilization",
            "Archaeology Tamil Nadu",
            "தமிழ் வரலாறு",
            "தொல்லியல் ஆய்வுகள்",
            "சோழர் வரலாறு",
            "பூம்புகார் துறைமுகம்",
            "பண்டைய துறைமுகம்",
            "Chola Maritime Trade",
            "Poompuhar Port",
        ]
        raw_candidates.extend(default_tags)

        # Deduplicate case-insensitively and filter length
        accepted = []
        rejected = []
        seen = set()
        current_len = 0

        for t in raw_candidates:
            clean = str(t).strip().replace("\n", "").replace(",", "")
            if not clean or len(clean) < 2:
                continue
            lower_clean = clean.lower()
            if lower_clean in seen:
                rejected.append({"tag": clean, "reason": "duplicate"})
                continue

            # Length addition (tag length + comma separator)
            add_len = len(clean) + (1 if accepted else 0)
            if current_len + add_len > max_budget:
                rejected.append({"tag": clean, "reason": "budget_exceeded"})
                continue

            accepted.append(clean)
            seen.add(lower_clean)
            current_len += add_len

        return TagSet(
            accepted_tags=accepted,
            rejected_tags=rejected,
            total_characters=current_len
        )

    def _generate_hashtags(
        self,
        script_data: Dict[str, Any],
        topic_meta: Dict[str, Any],
        llm_output: Optional[Dict[str, Any]]
    ) -> HashtagSet:
        """Generate small relevant set of max 10 hashtags."""
        max_ht = getattr(self.settings, "max_hashtags", 10)
        candidates = ["#Kaalapadhivugal", "#காலப்பதிவுகள்"]

        if llm_output and isinstance(llm_output.get("hashtags"), list):
            for h in llm_output["hashtags"]:
                clean = str(h).strip()
                if not clean.startswith("#"):
                    clean = f"#{clean}"
                candidates.append(clean)

        # Default fallback hashtags
        topic_words = re.findall(r"[\w]+", topic_meta.get("topic", ""))
        for w in topic_words[:2]:
            if len(w) > 3:
                candidates.append(f"#{w}")

        candidates.extend(["#TamilHistory", "#Archaeology", "#AncientTamil"])

        # Deduplicate
        seen = set()
        final_ht = []
        for h in candidates:
            hl = h.lower()
            if hl not in seen and len(final_ht) < max_ht:
                final_ht.append(h)
                seen.add(hl)

        return HashtagSet(hashtags=final_ht)
