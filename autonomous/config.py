"""
autonomous/config.py - Core Configuration for the Autonomous YouTube Channel System.
Loads configuration from environment variables (.env) with safe production defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
EPISODES_DIR = OUTPUTS_DIR / "episodes"
REVIEW_DIR = OUTPUTS_DIR / "review_required"
FAILED_DIR = OUTPUTS_DIR / "failed"
ARCHIVE_DIR = OUTPUTS_DIR / "archive"

# Ensure runtime directories exist
for p in [EPISODES_DIR, REVIEW_DIR, FAILED_DIR, ARCHIVE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class ChannelIdentity:
    """Canonical Channel & Presenter Identity Configuration for Kaalapadhivugal."""
    channel_name_ta: str = "காலப் பதிவுகள்"
    channel_name_en: str = "Kaalapadhivugal"
    channel_description_en: str = "Records of Time"
    channel_handle: str = "@kaalapadhivugal"
    tagline_ta: str = "கடந்த காலத்தை, சான்றுகளுடன் மீண்டும் பார்ப்போம்."
    tagline_en: str = "Let’s revisit the past, through the evidence."
    host_enabled: bool = True
    host_name_ta: str = "யாழினி"
    host_name_en: str = "Yaazhini"
    host_character_id: str = "host_yaazhini"
    host_description: str = (
        "South Indian female documentary presenter with warm skin tone, expressive natural face, "
        "dark wavy hair, traditional jhumka earrings, wearing an earthy olive-green kurti with patterned dupatta, "
        "seated in an archaeological study with ancient artifacts and historical maps in background"
    )
    host_consistency_token: str = "yaazhini_host"
    host_reference_image: str = "assets/yaazhini_presenter.jpg"

    def get_character_config(self) -> Optional[dict]:
        """Return character specification dict for VisualPlanner if host_enabled is True."""
        if not self.host_enabled:
            return None
        return {
            "character_id": self.host_character_id,
            "name": self.host_name_en,
            "name_ta": self.host_name_ta,
            "description": self.host_description,
            "consistency_token": self.host_consistency_token,
            "reference_image": self.host_reference_image,
            "rules": [
                f"Maintain consistent presenter appearance for {self.host_name_en} across all host scenes.",
                "Professional, respectful documentary presenter framing.",
                "Natural lighting consistent with historical scene environment."
            ]
        }


@dataclass
class AutonomousSettings:
    """Settings controlling autonomous channel behavior, safety gates, and hardware budgets."""

    # Canonical Channel & Host Identity
    channel: ChannelIdentity = field(default_factory=ChannelIdentity)

    # Operational Modes
    autonomous_mode: bool = os.getenv("AUTONOMOUS_MODE", "true").lower() == "true"
    auto_publish: bool = os.getenv("AUTO_PUBLISH", "false").lower() == "true"
    default_youtube_privacy: str = os.getenv("DEFAULT_YOUTUBE_PRIVACY", "private").lower()  # private, unlisted, public
    max_episodes_per_day: int = int(os.getenv("MAX_EPISODES_PER_DAY", "1"))
    quality_mode: str = os.getenv("QUALITY_MODE", "balanced").lower()  # safe, balanced, cinematic, experimental

    # Safety & Quality Thresholds
    min_research_confidence: float = float(os.getenv("MIN_RESEARCH_CONFIDENCE", "0.85"))
    min_motion_safety_score: float = float(os.getenv("MIN_MOTION_SAFETY_SCORE", "0.70"))
    max_motion_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    max_script_validation_retries: int = int(os.getenv("MAX_SCRIPT_RETRIES", "2"))

    # Feature Toggles (RTX 2050 4GB Safety)
    enable_true_i2v: bool = os.getenv("ENABLE_TRUE_I2V", "false").lower() == "true"  # Disabled on 4GB VRAM
    enable_interpolation: bool = os.getenv("ENABLE_INTERPOLATION", "false").lower() == "true"

    # Phase 9: Audio & SadTalker Settings (RTX 2050 4GB Safety)
    tts_provider: str = os.getenv("TTS_PROVIDER", "edge_tts")  # edge_tts, deterministic_fallback, mock
    tts_voice_tamil: str = os.getenv("TTS_VOICE_TAMIL", "ta-IN-PallaviNeural")
    tts_voice_english: str = os.getenv("TTS_VOICE_ENGLISH", "en-IN-NeerjaNeural")
    tts_allow_network: bool = os.getenv("TTS_ALLOW_NETWORK", "true").lower() == "true"
    enable_english_narration: bool = os.getenv("ENABLE_ENGLISH_NARRATION", "false").lower() == "true"
    enable_sadtalker: bool = os.getenv("ENABLE_SADTALKER", "true").lower() == "true"
    max_audio_retries: int = int(os.getenv("MAX_AUDIO_RETRIES", "2"))
    sadtalker_face_size: int = int(os.getenv("SADTALKER_FACE_SIZE", "256"))

    # Phase 10: Video Mastering & QC Settings (RTX 2050 4GB Safety - CPU/FFmpeg first)
    master_width: int = int(os.getenv("MASTER_WIDTH", "1280"))
    master_height: int = int(os.getenv("MASTER_HEIGHT", "720"))
    master_fps: int = int(os.getenv("MASTER_FPS", "25"))
    master_crf: int = int(os.getenv("MASTER_CRF", "18"))
    master_preset: str = os.getenv("MASTER_PRESET", "fast")
    master_video_codec: str = os.getenv("MASTER_VIDEO_CODEC", "libx264")
    master_audio_codec: str = os.getenv("MASTER_AUDIO_CODEC", "aac")
    master_audio_bitrate: str = os.getenv("MASTER_AUDIO_BITRATE", "192k")
    target_lufs: float = float(os.getenv("TARGET_LUFS", "-14.0"))
    enable_subtitles: bool = os.getenv("ENABLE_SUBTITLES", "true").lower() == "true"
    enable_bgm: bool = os.getenv("ENABLE_BGM", "false").lower() == "true"
    qc_min_score: float = float(os.getenv("QC_MIN_SCORE", "0.85"))
    qc_review_score: float = float(os.getenv("QC_REVIEW_SCORE", "0.70"))
    max_black_frame_ratio: float = float(os.getenv("MAX_BLACK_FRAME_RATIO", "0.15"))
    max_frozen_frame_duration_s: float = float(os.getenv("MAX_FROZEN_FRAME_DURATION_S", "6.0"))

    # Phase 11: SEO Metadata & Thumbnail Settings (RTX 2050 4GB Safety - CPU/PIL/OpenCV first)
    thumbnail_width: int = int(os.getenv("THUMBNAIL_WIDTH", "1280"))
    thumbnail_height: int = int(os.getenv("THUMBNAIL_HEIGHT", "720"))
    thumbnail_format: str = os.getenv("THUMBNAIL_FORMAT", "jpg")
    max_title_length: int = int(os.getenv("MAX_TITLE_LENGTH", "100"))
    max_tags_length: int = int(os.getenv("MAX_TAGS_LENGTH", "500"))
    max_hashtags: int = int(os.getenv("MAX_HASHTAGS", "10"))
    min_thumbnail_contrast: float = float(os.getenv("MIN_THUMBNAIL_CONTRAST", "20.0"))

    # Phase 12: YouTube Publication Architecture (Mock / Offline-first, Dormant Real API)
    youtube_integration_enabled: bool = os.getenv("YOUTUBE_INTEGRATION_ENABLED", "false").lower() == "true"
    auto_publish_public: bool = os.getenv("AUTO_PUBLISH_PUBLIC", "false").lower() == "true"
    youtube_default_mode: str = os.getenv("YOUTUBE_DEFAULT_MODE", "DRY_RUN")  # DRY_RUN, MOCK_PRIVATE_TEST
    youtube_made_for_kids: bool = os.getenv("YOUTUBE_MADE_FOR_KIDS", "false").lower() == "true"
    youtube_channel_id: Optional[str] = os.getenv("YOUTUBE_CHANNEL_ID", None)
    youtube_client_secret_file: Optional[str] = os.getenv("GOOGLE_CLIENT_SECRET_FILE", None)
    youtube_token_file: Optional[str] = os.getenv("YOUTUBE_TOKEN_FILE", None)

    # LLM Configuration
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")

    # Database
    db_path: str = os.getenv("AUTONOMOUS_DB_PATH", str(BASE_DIR / "universe_platform.db"))

    # Topic Scoring & Discovery Settings (Phase 3)
    weight_educational_value: float = float(os.getenv("WEIGHT_EDUCATIONAL_VALUE", "0.20"))
    weight_storytelling_potential: float = float(os.getenv("WEIGHT_STORYTELLING_POTENTIAL", "0.20"))
    weight_visual_potential: float = float(os.getenv("WEIGHT_VISUAL_POTENTIAL", "0.15"))
    weight_researchability: float = float(os.getenv("WEIGHT_RESEARCHABILITY", "0.15"))
    weight_novelty: float = float(os.getenv("WEIGHT_NOVELTY", "0.10"))
    weight_audience_potential: float = float(os.getenv("WEIGHT_AUDIENCE_POTENTIAL", "0.10"))
    weight_episode_suitability: float = float(os.getenv("WEIGHT_EPISODE_SUITABILITY", "0.05"))
    weight_category_balance: float = float(os.getenv("WEIGHT_CATEGORY_BALANCE", "0.05"))

    # Recency & Penalty Rules
    topic_recent_days: int = int(os.getenv("TOPIC_RECENT_DAYS", "30"))
    topic_recent_penalty: float = float(os.getenv("TOPIC_RECENT_PENALTY", "0.40"))
    category_repetition_penalty: float = float(os.getenv("CATEGORY_REPETITION_PENALTY", "0.15"))
    failure_penalty_base: float = float(os.getenv("FAILURE_PENALTY_BASE", "0.20"))
    topic_duplicate_similarity_threshold: float = float(os.getenv("TOPIC_DUPLICATE_SIMILARITY_THRESHOLD", "0.90"))

    # Historical Categories for Autonomous Topic Discovery
    categories: List[str] = field(default_factory=lambda: [
        "Ancient Tamil history",
        "Chola history",
        "Pandya history",
        "Chera history",
        "Indian history",
        "Ancient engineering",
        "Archaeology",
        "Lost cities",
        "Maritime history",
        "Ancient technology",
        "Temples and architecture",
        "Historical mysteries",
        "Important historical personalities",
        "Cultural history",
        "Science and technology history",
        "Forgotten civilizations",
        "Ancient trade",
        "Battles",
        "Ancient manuscripts",
        "Archaeological discoveries"
    ])


autonomous_settings = AutonomousSettings()
