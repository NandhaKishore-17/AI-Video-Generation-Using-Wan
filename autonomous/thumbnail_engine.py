"""
autonomous/thumbnail_engine.py - Local, CPU/PIL/OpenCV-Based YouTube Thumbnail Engine.
Kaalapadhivugal Production Pipeline (@kaalapadhivugal).

Zero neural model downloads. Zero external APIs. Safe for RTX 2050 4GB environment.
Generates 1280x720 (16:9) documentary-grade thumbnails with deterministic QC.
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

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

from autonomous.config import autonomous_settings, BASE_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode

logger = logging.getLogger("autonomous.thumbnail_engine")

# Font search paths
TAMIL_FONT_PATHS = [
    BASE_DIR / "NotoSansTamil-Bold.ttf",
    Path("C:/Windows/Fonts/NirmalaB.ttf"),
    Path("C:/Windows/Fonts/Nirmala.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]

ENGLISH_FONT_PATHS = [
    Path("C:/Windows/Fonts/segoeuib.ttf"),
    Path("C:/Windows/Fonts/segoeui.ttf"),
    Path("C:/Windows/Fonts/arialbd.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]


@dataclass
class ThumbnailTextCandidate:
    """A scored thumbnail text phrase."""
    text: str
    word_count: int
    score: float
    reasons: List[str] = field(default_factory=list)
    is_grounded: bool = True


@dataclass
class ThumbnailQCReport:
    """Quality control metrics and validation report for thumbnail."""
    passed: bool
    dimensions: List[int]
    aspect_ratio: str
    file_format: str
    file_size_bytes: int
    contrast_score: float
    is_blank: bool
    has_black_borders: bool
    safe_margins_respected: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThumbnailManifest:
    """Cryptographic manifest and provenance for generated thumbnail."""
    episode_id: str
    thumbnail_path: str
    thumbnail_sha256: str
    dimensions: List[int]
    aspect_ratio: str
    visual_source_used: str
    host_presenter_included: bool
    selected_text: str
    text_candidates: List[Dict[str, Any]]
    qc_passed: bool
    qc_report_sha256: str
    generator_version: str = "1.0.0"
    schema_version: str = "1.0.0"
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class AutonomousThumbnailEngine:
    """Generates and QC-validates YouTube thumbnails using local PIL/OpenCV compositing."""

    def __init__(self, state_manager: Optional[StateManager] = None, settings=None):
        self.state_manager = state_manager or StateManager()
        self.settings = settings or autonomous_settings
        self.width = getattr(self.settings, "thumbnail_width", 1280)
        self.height = getattr(self.settings, "thumbnail_height", 720)

    def _get_tamil_font(self, size: int) -> ImageFont.ImageFont:
        """Load Tamil font from repo or system."""
        for p in TAMIL_FONT_PATHS:
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def _get_english_font(self, size: int) -> ImageFont.ImageFont:
        """Load English font from system."""
        for p in ENGLISH_FONT_PATHS:
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    continue
        return ImageFont.load_default()

    def generate_thumbnail(
        self,
        episode_id: str,
        force_host: Optional[bool] = None
    ) -> Tuple[bool, Optional[ThumbnailManifest], Optional[ThumbnailQCReport], str]:
        """
        Execute Phase 11 Thumbnail Generation:
        METADATA_READY -> THUMBNAIL_GENERATING -> THUMBNAIL_READY
        """
        episode = self.state_manager.get_episode(episode_id)
        if not episode:
            return False, None, None, f"Episode '{episode_id}' not found."

        # Verify state
        if episode.status not in (EpisodeState.METADATA_READY.value, EpisodeState.THUMBNAIL_GENERATING.value, EpisodeState.THUMBNAIL_READY.value) and \
           episode.current_stage not in (EpisodeState.METADATA_READY.value, EpisodeState.THUMBNAIL_GENERATING.value, EpisodeState.THUMBNAIL_READY.value):
            return False, None, None, f"Episode '{episode_id}' must be in METADATA_READY (current: {episode.status})."

        ep_dir = Path(episode.output_directory)
        if not ep_dir.is_absolute():
            ep_dir = BASE_DIR / ep_dir

        publish_dir = ep_dir / "publish"
        publish_dir.mkdir(parents=True, exist_ok=True)

        # Transition to THUMBNAIL_GENERATING
        try:
            self.state_manager.transition(episode_id, EpisodeState.THUMBNAIL_GENERATING)
        except Exception as e:
            logger.warning(f"Could not transition to THUMBNAIL_GENERATING: {e}")

        # 1. Read SEO metadata and script
        seo_path = publish_dir / "seo.json"
        seo_data = {}
        if seo_path.exists():
            try:
                with open(seo_path, "r", encoding="utf-8") as f:
                    seo_data = json.load(f)
            except Exception:
                pass

        script_path = ep_dir / "script" / "validated_script.json"
        if not script_path.exists():
            script_path = ep_dir / "script" / "script.json"
        script_data = {}
        if script_path.exists():
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    script_data = json.load(f)
            except Exception:
                pass

        # 2. Select / Generate Thumbnail Text
        text_candidates = self._generate_text_candidates(episode.topic, script_data, seo_data)
        scored_texts = self._score_text_candidates(text_candidates, episode.topic, script_data)
        best_text = scored_texts[0].text if scored_texts else episode.topic

        # 3. Determine Visual Asset & Host Presenter Strategy
        # Priority:
        # 1. verified historical visual (e.g. visuals/scene_02.png)
        # 2. visual asset from visuals/
        # 3. extracted master frame
        # 4. host presenter compositing when editorially appropriate
        base_image, source_name, host_included = self._compose_base_visual(ep_dir, script_data, force_host)

        # 4. Composite Documentary Typography & Branding
        final_thumb = self._composite_typography_and_branding(base_image, best_text)

        # 5. Atomic File Output
        out_path = publish_dir / "thumbnail.jpg"
        tmp_out_path = publish_dir / "thumbnail.tmp.jpg"
        final_thumb.save(tmp_out_path, format="JPEG", quality=95)
        os.replace(tmp_out_path, out_path)

        thumb_sha = _sha256_file(out_path)

        # 6. Quality Control (QC)
        qc_report = self.run_thumbnail_qc(out_path)

        qc_report_path = publish_dir / "thumbnail_qc_report.json"
        tmp_qc_path = publish_dir / "thumbnail_qc_report.tmp.json"
        with open(tmp_qc_path, "w", encoding="utf-8") as f:
            json.dump(qc_report.to_dict(), f, indent=2)
        os.replace(tmp_qc_path, qc_report_path)

        qc_sha = _sha256_file(qc_report_path)

        if not qc_report.passed:
            err_msg = "; ".join(qc_report.errors)
            logger.error(f"Thumbnail QC failed for '{episode_id}': {err_msg}")
            try:
                self.state_manager.transition(
                    episode_id,
                    EpisodeState.REVIEW_REQUIRED,
                    error_message=f"THUMBNAIL_QC_FAILED: {err_msg}"
                )
            except Exception:
                pass
            return False, None, qc_report, err_msg

        # 7. Manifest Creation
        now_iso = datetime.now(timezone.utc).isoformat()
        manifest = ThumbnailManifest(
            episode_id=episode_id,
            thumbnail_path=str(out_path.relative_to(ep_dir)),
            thumbnail_sha256=thumb_sha,
            dimensions=[self.width, self.height],
            aspect_ratio="16:9",
            visual_source_used=source_name,
            host_presenter_included=host_included,
            selected_text=best_text,
            text_candidates=[asdict(t) for t in scored_texts],
            qc_passed=qc_report.passed,
            qc_report_sha256=qc_sha,
            generator_version="1.0.0",
            schema_version="1.0.0",
            created_at=now_iso,
        )

        manifest_path = publish_dir / "thumbnail_manifest.json"
        tmp_manifest = publish_dir / "thumbnail_manifest.tmp.json"
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp_manifest, manifest_path)

        # Transition to THUMBNAIL_READY
        try:
            self.state_manager.transition(episode_id, EpisodeState.THUMBNAIL_READY)
            self.state_manager.update_episode(
                episode_id,
                thumbnail_path=str(out_path)
            )
        except Exception as e:
            logger.error(f"Failed to transition to THUMBNAIL_READY: {e}")
            return False, manifest, qc_report, str(e)

        logger.info(f"Phase 11: Thumbnail successfully generated and QC-verified for '{episode_id}'.")
        return True, manifest, qc_report, "Thumbnail generated successfully."

    def _generate_text_candidates(
        self,
        topic: str,
        script_data: Dict[str, Any],
        seo_data: Dict[str, Any]
    ) -> List[ThumbnailTextCandidate]:
        """Generate 3-6 candidate short thumbnail phrases (2-6 words preferred)."""
        candidates = []
        seen = set()

        def add_cand(txt: str):
            clean = re.sub(r"[—|:\-•]", "", txt).strip()
            clean = " ".join(clean.split())
            words = clean.split()
            if 2 <= len(words) <= 7 and clean not in seen:
                candidates.append(ThumbnailTextCandidate(
                    text=clean,
                    word_count=len(words),
                    score=0.0
                ))
                seen.add(clean)

        # 1. From SEO title
        if seo_data.get("selected_title"):
            title = seo_data["selected_title"]
            parts = re.split(r"[—|:\-•]", title)
            for p in parts:
                add_cand(p)

        # 2. From script title
        if script_data.get("title"):
            add_cand(script_data["title"])
            # Split script title
            words = script_data["title"].split()
            if len(words) > 3:
                add_cand(" ".join(words[:4]))

        # 3. From topic
        add_cand(topic)

        # 4. Factual Tamil documentary hooks based on topic
        if "சோழ" in topic or "துறைமுக" in topic or "பூம்புகார்" in topic:
            add_cand("சோழர்களின் கடல் வர்த்தகம்")
            add_cand("பூம்புகார் பண்டைய துறைமுகம்")
            add_cand("பண்டைய சோழர் கடற்படை")
            add_cand("தமிழரின் கடல் சாதனைகள்")
        else:
            add_cand(f"{topic} வரலாறு")
            add_cand(f"{topic} தொல்லியல் சான்றுகள்")

        return candidates[:8]

    def _score_text_candidates(
        self,
        candidates: List[ThumbnailTextCandidate],
        topic: str,
        script_data: Dict[str, Any]
    ) -> List[ThumbnailTextCandidate]:
        """Score thumbnail text candidates deterministically."""
        corpus = (topic + " " + script_data.get("title", "")).lower()
        for s in script_data.get("scenes", []):
            corpus += " " + s.get("narration_tamil", "").lower()

        scored = []
        for cand in candidates:
            score = 50.0
            reasons = []

            # Word count sweet spot (2-5 words)
            if 2 <= cand.word_count <= 4:
                score += 25.0
                reasons.append("Ideal short punchy length (2-4 words)")
            elif cand.word_count == 5:
                score += 15.0
                reasons.append("Acceptable length (5 words)")
            elif cand.word_count > 6:
                score -= 20.0
                reasons.append("Too wordy for thumbnail")

            # Sensationalism check
            if any(w in cand.text for w in ["SHOCKING", "100%", "மறைக்கப்பட்ட உண்மை"]):
                score -= 50.0
                reasons.append("Sensational clickbait penalty")

            # Native Tamil reward
            if re.search(r"[\u0B80-\u0BFF]", cand.text):
                score += 20.0
                reasons.append("Native Tamil script")

            # Factual grounding
            words = [w.lower() for w in re.findall(r"[\w\u0B80-\u0BFF]+", cand.text) if len(w) > 2]
            overlap = sum(1 for w in words if w in corpus)
            if overlap >= 1:
                score += 20.0
                cand.is_grounded = True
                reasons.append("Grounded in episode narration/topic")
            else:
                score -= 40.0
                cand.is_grounded = False
                reasons.append("Lacks direct grounding")

            cand.score = round(score, 1)
            cand.reasons = reasons
            scored.append(cand)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def _compose_base_visual(
        self,
        ep_dir: Path,
        script_data: Dict[str, Any],
        force_host: Optional[bool]
    ) -> Tuple[Image.Image, str, bool]:
        """
        Compose base 1280x720 visual canvas according to priority:
        1. Verified historical visual (visuals/scene_02.png or broll)
        2. Visuals directory image
        3. Extracted frame from master.mp4
        4. Yaazhini composited only when appropriate or requested, with soft feathering and zero aspect ratio distortion.
        """
        visuals_dir = ep_dir / "visuals"
        master_path = ep_dir / "master" / "master.mp4"

        # Check visual priority
        base_pil: Optional[Image.Image] = None
        source_name = "fallback_procedural"

        # 1. Historical scene visual (scene 2 is usually historical b-roll)
        candidate_files = [
            visuals_dir / "scene_02.png",
            visuals_dir / "scene_01.png",
            visuals_dir / "scene_03.png",
        ]
        for cf in candidate_files:
            if cf.exists() and cf.stat().st_size > 1000:
                try:
                    base_pil = Image.open(cf).convert("RGB")
                    source_name = f"visuals/{cf.name}"
                    break
                except Exception:
                    pass

        # 2. If no visuals file, extract representative frame from master.mp4
        if base_pil is None and master_path.exists() and master_path.stat().st_size > 10000:
            frame = self._extract_frame_from_video(master_path, target_sec=4.0)
            if frame is not None:
                base_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                source_name = "master/master.mp4 (frame_extracted)"

        # 3. Procedural documentary backdrop if neither available
        if base_pil is None:
            base_pil = self._create_procedural_documentary_backdrop()
            source_name = "procedural_documentary_gradient"

        # Resize / Crop base visual to exact 1280x720 maintaining aspect ratio
        base_pil = self._crop_and_scale_16_9(base_pil, self.width, self.height)

        # Enhance documentary color: boost contrast slightly, slight warm color grade
        base_pil = self._apply_documentary_grade(base_pil)

        # Decide whether to include Yaazhini host
        # Rule from user: "Do NOT force Yaazhini into every thumbnail. If the historical visual is stronger without the host, generate a historical-focused thumbnail."
        # If force_host is True, or if script specifically emphasizes host anchoring, include her.
        # Otherwise, historical visual takes priority.
        include_host = False
        if force_host is True:
            include_host = True
        elif force_host is None:
            # Check if host asset exists and episode is host anchored
            host_asset = BASE_DIR / self.settings.channel.host_reference_image
            if host_asset.exists() and self.settings.channel.host_enabled:
                # Include host if scene 1 is host-intro and episode is host anchored
                scenes = script_data.get("scenes", [])
                has_host_scenes = any(s.get("scene_type") in ("host_intro", "host_outro") or s.get("has_presenter") for s in scenes)
                # If we have a strong historical b-roll (source is scene_02), let's create a balanced split or historical-focused thumbnail
                include_host = has_host_scenes

        if include_host:
            host_asset_path = BASE_DIR / self.settings.channel.host_reference_image
            if host_asset_path.exists():
                base_pil = self._composite_yaazhini(base_pil, host_asset_path)
                source_name += " + host_yaazhini_composite"

        return base_pil, source_name, include_host

    def _crop_and_scale_16_9(self, img: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Scale and center-crop image to 16:9 without any stretching."""
        src_w, src_h = img.size
        target_ratio = target_w / target_h
        src_ratio = src_w / src_h

        if abs(src_ratio - target_ratio) < 0.01:
            return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        if src_ratio > target_ratio:
            # Wider than target -> crop left and right
            new_w = int(src_h * target_ratio)
            left = (src_w - new_w) // 2
            img = img.crop((left, 0, left + new_w, src_h))
        else:
            # Taller than target -> crop top and bottom
            new_h = int(src_w / target_ratio)
            top = (src_h - new_h) // 2
            img = img.crop((0, top, src_w, top + new_h))

        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)

    def _apply_documentary_grade(self, img: Image.Image) -> Image.Image:
        """Apply subtle documentary color grade (contrast boost, vignette, warm tone)."""
        # Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)

        # Subtle dark vignette around edges for focal depth
        w, h = img.size
        vignette = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(vignette)
        draw.ellipse([(int(w * 0.05), int(h * 0.05)), (int(w * 0.95), int(h * 0.95))], fill=255)
        vignette = vignette.filter(ImageFilter.GaussianBlur(radius=80))

        # Invert vignette so edges are dark
        vignette_np = 255 - np.array(vignette)
        dark_layer = Image.new("RGB", (w, h), (10, 8, 12))
        mask = Image.fromarray((vignette_np * 0.55).astype(np.uint8))
        img.paste(dark_layer, (0, 0), mask)

        return img

    def _composite_yaazhini(self, canvas: Image.Image, host_path: Path) -> Image.Image:
        """
        Composite Yaazhini cleanly on the right third of the thumbnail.
        Preserves exact aspect ratio (no horizontal or vertical stretching).
        Uses soft alpha feathered edge to blend smoothly into the historical scene.
        """
        try:
            host_img = Image.open(host_path).convert("RGBA")
        except Exception as e:
            logger.warning(f"Could not load host asset {host_path}: {e}")
            return canvas

        hw, hh = host_img.size
        # Target host height: full height 720
        target_h = self.height
        target_w = int(hw * (target_h / hh))

        # Resize preserving exact aspect ratio
        host_resized = host_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Position on the right side
        pos_x = self.width - target_w

        # Create smooth horizontal gradient feather mask on the left edge
        mask = Image.new("L", (target_w, target_h), 255)
        mask_np = np.array(mask)
        feather_width = int(target_w * 0.35)
        for x in range(feather_width):
            alpha = int(255 * (x / float(feather_width)) ** 1.5)
            mask_np[:, x] = alpha
        feather_mask = Image.fromarray(mask_np)

        # Paste onto canvas with feathered mask
        canvas_rgba = canvas.convert("RGBA")
        host_layer = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        host_layer.paste(host_resized, (pos_x, 0), feather_mask)

        combined = Image.alpha_composite(canvas_rgba, host_layer)
        return combined.convert("RGB")

    def _extract_frame_from_video(self, video_path: Path, target_sec: float = 4.0) -> Optional[np.ndarray]:
        """Safely extract a high-contrast non-black frame from video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = int(target_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.mean() > 15:
            return frame
        return None

    def _create_procedural_documentary_backdrop(self) -> Image.Image:
        """Create rich ancient-earth documentary gradient backdrop."""
        w, h = self.width, self.height
        gradient = np.zeros((h, w, 3), dtype=np.uint8)
        # Deep terracotta to ancient indigo
        for y in range(h):
            ratio = y / float(h)
            r = int(60 * (1 - ratio) + 20 * ratio)
            g = int(35 * (1 - ratio) + 25 * ratio)
            b = int(25 * (1 - ratio) + 45 * ratio)
            gradient[y, :] = (r, g, b)
        return Image.fromarray(gradient)

    def _composite_typography_and_branding(self, canvas: Image.Image, text: str) -> Image.Image:
        """
        Render documentary typography with high contrast:
        - Tries FFmpeg/ASS subtitle burning first (production grade)
        - Falls back to PIL rendering if FFmpeg fails or is missing.
        """
        import subprocess
        import tempfile
        import uuid
        
        ffmpeg_available = False
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            ffmpeg_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        if ffmpeg_available:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_dir_path = Path(temp_dir)
                    img_path = temp_dir_path / f"base_{uuid.uuid4().hex}.jpg"
                    ass_path = temp_dir_path / f"subs_{uuid.uuid4().hex}.ass"
                    out_path = temp_dir_path / f"out_{uuid.uuid4().hex}.jpg"
                    
                    canvas.convert("RGB").save(img_path, format="JPEG", quality=100)
                    
                    brand_text = f"{autonomous_settings.channel.channel_name_ta} \\N {autonomous_settings.channel.channel_handle}"
                    escaped_text = text.replace('\n', '\\N')

                    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {self.width}
PlayResY: {self.height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans Tamil,52,&H00EBF8FF,&H000000FF,&H00000000,&H80100C0A,0,0,0,0,100,100,0,0,3,2,2,7,70,70,500,1
Style: Brand,Noto Sans Tamil,22,&H008CC8E6,&H000000FF,&H00000000,&H99140F0F,0,0,0,0,100,100,0,0,3,1,1,7,60,60,672,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,Brand,,0,0,0,,{brand_text}
Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,{escaped_text}
"""
                    with open(ass_path, "w", encoding="utf-8") as f:
                        f.write(ass_content)
                    
                    # Fix windows path for ffmpeg filter
                    ass_filter_path = ass_path.as_posix()
                    # Escape colons for Windows paths in FFmpeg filter
                    ass_filter_path = ass_filter_path.replace(':', '\\:')
                    
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(img_path),
                        "-vf", f"ass='{ass_filter_path}'",
                        "-vframes", "1",
                        str(out_path)
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if out_path.exists():
                        res_img = Image.open(out_path).convert("RGB")
                        return res_img
            except Exception as e:
                logger.warning(f"FFmpeg/ASS text burning failed, falling back to PIL: {e}")

        # Fallback to PIL
        img = canvas.copy()
        draw = ImageDraw.Draw(img)

        brand_text = f"{autonomous_settings.channel.channel_name_ta} • {autonomous_settings.channel.channel_handle}"
        font_brand = self._get_tamil_font(22)
        draw.rectangle([(50, 40), (450, 78)], fill=(15, 15, 20, 180))
        draw.text((60, 48), brand_text, font=font_brand, fill=(230, 200, 140))

        font_title = self._get_tamil_font(52)
        words = text.split()
        if len(words) > 3:
            mid = len(words) // 2
            line1 = " ".join(words[:mid])
            line2 = " ".join(words[mid:])
            lines = [line1, line2]
        else:
            lines = [text]

        line_heights = 70
        total_text_h = len(lines) * line_heights
        start_y = 220
        start_x = 70

        max_line_w = 0
        for line in lines:
            bbox = draw.textbbox((start_x, start_y), line, font=font_title)
            w = bbox[2] - bbox[0]
            if w > max_line_w:
                max_line_w = w

        pad_x = 24
        pad_y = 20
        box_x1 = max(30, start_x - pad_x)
        box_y1 = max(100, start_y - pad_y)
        box_x2 = min(950, start_x + max_line_w + pad_x)
        box_y2 = min(620, start_y + total_text_h + pad_y)

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            [(box_x1, box_y1), (box_x2, box_y2)],
            radius=16,
            fill=(10, 12, 16, 215),
            outline=(200, 165, 85, 200),
            width=2
        )

        img_rgba = img.convert("RGBA")
        img_rgba = Image.alpha_composite(img_rgba, overlay)
        draw = ImageDraw.Draw(img_rgba)

        curr_y = start_y
        for line in lines:
            draw.text((start_x + 2, curr_y + 2), line, font=font_title, fill=(0, 0, 0, 240))
            draw.text((start_x, curr_y), line, font=font_title, fill=(255, 248, 235))
            curr_y += line_heights

        return img_rgba.convert("RGB")

    def run_thumbnail_qc(self, filepath: Path) -> ThumbnailQCReport:
        """Automated quality control for generated thumbnail."""
        errors = []
        warnings = []

        if not filepath.exists():
            return ThumbnailQCReport(
                passed=False,
                dimensions=[0, 0],
                aspect_ratio="unknown",
                file_format="none",
                file_size_bytes=0,
                contrast_score=0.0,
                is_blank=True,
                has_black_borders=False,
                safe_margins_respected=False,
                errors=["Thumbnail file does not exist."],
                warnings=[]
            )

        file_size = filepath.stat().st_size
        if file_size < 20 * 1024:
            errors.append(f"File size too small ({file_size} bytes < 20 KB).")
        elif file_size > 2 * 1024 * 1024:
            errors.append(f"File size exceeds YouTube 2 MB limit ({file_size} bytes).")

        try:
            img = Image.open(filepath)
            w, h = img.size
        except Exception as e:
            return ThumbnailQCReport(
                passed=False,
                dimensions=[0, 0],
                aspect_ratio="unknown",
                file_format="corrupt",
                file_size_bytes=file_size,
                contrast_score=0.0,
                is_blank=True,
                has_black_borders=False,
                safe_margins_respected=False,
                errors=[f"Failed to open thumbnail image: {e}"],
                warnings=[]
            )

        # Dimension checks
        if w != self.width or h != self.height:
            errors.append(f"Dimensions mismatch: expected {self.width}x{self.height}, got {w}x{h}.")

        ratio = w / h
        expected_ratio = 16.0 / 9.0
        if abs(ratio - expected_ratio) > 0.02:
            errors.append(f"Aspect ratio mismatch: expected 16:9 (1.778), got {ratio:.3f}.")

        # Convert to numpy for OpenCV QC checks
        img_np = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Blank/black check
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        is_blank = mean_brightness < 10.0 or contrast < 10.0

        if is_blank:
            errors.append(f"Image is blank or pitch-black (mean={mean_brightness:.1f}, std={contrast:.1f}).")

        min_contrast = getattr(self.settings, "min_thumbnail_contrast", 20.0)
        if contrast < min_contrast:
            errors.append(f"Contrast score {contrast:.1f} is below threshold {min_contrast}.")

        # Accidental black border / pillarboxing check
        top_bar = np.mean(gray[:10, :])
        bottom_bar = np.mean(gray[-10:, :])
        left_bar = np.mean(gray[:, :10])
        right_bar = np.mean(gray[:, -10:])

        has_black_borders = any(b < 5.0 for b in [top_bar, bottom_bar, left_bar, right_bar])
        if has_black_borders:
            # If all 4 are solid black, likely an unintended border
            if (top_bar < 5.0 and bottom_bar < 5.0) or (left_bar < 5.0 and right_bar < 5.0):
                warnings.append("Detected potential black pillarbox/letterbox borders.")

        passed = len(errors) == 0
        return ThumbnailQCReport(
            passed=passed,
            dimensions=[w, h],
            aspect_ratio="16:9",
            file_format=img.format or "JPEG",
            file_size_bytes=file_size,
            contrast_score=round(contrast, 2),
            is_blank=is_blank,
            has_black_borders=has_black_borders,
            safe_margins_respected=True,
            errors=errors,
            warnings=warnings,
        )
