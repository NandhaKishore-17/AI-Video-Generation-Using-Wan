"""
autonomous/visual_planner.py - Phase 6 Autonomous Visual Planning Engine.

Transforms a validated, evidence-grounded 4-scene script, verified claims, and research dossier
into a deterministic, production-ready Visual Plan (visual_plan.json) for Phase 7+ consumption.

Strict boundaries:
- Phase 6 generates ONLY visual plan specifications (prompts, composition, motion plans, asset contracts).
- Does NOT generate static images, invoke Wan, SadTalker, Depth Anything, TTS, MoviePy, or YouTube APIs.
- Operates strictly on CPU; 0 VRAM allocated.
- Never invents historical facts or unsupported visual details.
- Character specifications are configuration-driven (never hardcoded or invented if unconfigured).
- Strict state gate: Accepts ONLY EpisodeState.SCRIPT_VALIDATED episodes.
"""

import os
import re
import json
import logging
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Set

from autonomous.config import autonomous_settings, EPISODES_DIR
from autonomous.state_manager import StateManager, EpisodeState, AutonomousEpisode
from autonomous.claim_models import VerifiedClaim, ClaimClassification, ClaimRelevanceClass

logger = logging.getLogger("autonomous.visual_planner")


class SceneType(str, Enum):
    HOST = "host"
    BROLL = "broll"
    UNKNOWN = "unknown"

def normalize_scene_type(raw_type: str) -> SceneType:
    if not raw_type:
        return SceneType.UNKNOWN
    
    t = str(raw_type).strip().lower()
    
    if t in ("host", "host_vlog", "host_intro", "host_outro"):
        return SceneType.HOST
    elif t in ("broll", "b_roll", "historical_broll", "historical_reconstruction", "documentary_broll", "documentary_b_roll"):
        return SceneType.BROLL
    else:
        return SceneType.UNKNOWN

class GroundingType(str, Enum):
    """Factual epistemic basis for visual composition."""
    HISTORICAL_CLAIM_GROUNDED = "HISTORICAL_CLAIM_GROUNDED"  # Directly depicts verified historical claim
    RECONSTRUCTION = "RECONSTRUCTION"                        # Scholarly hypothetical architectural/settlement 3D model
    ILLUSTRATIVE = "ILLUSTRATIVE"                            # Artistic depiction of ancient daily crafts/activities
    ATMOSPHERIC = "ATMOSPHERIC"                              # Landscape, river, sky, lighting, or environmental mood
    ABSTRACT = "ABSTRACT"                                    # Conceptual diagrams, maps, or structural graphics
    HOST_ANCHORED = "HOST_ANCHORED"                          # Explicitly configured canonical host character
    VERIFIED_FACT = "VERIFIED_FACT"                          # Direct factual evidence


@dataclass
class VisualMotionPlan:
    """Safe 2.5D parallax motion strategy for downstream Phase 8 MotionEngine."""
    camera_motion: str  # subtle_drift, pan_left, pan_right, slow_dolly_forward, tilt_up, static_safe
    motion_budget: str  # LOW, MEDIUM, HIGH
    depth_estimation_required: bool = True
    subject_protection_mask_required: bool = False
    primary_motion: str = "camera_dolly_forward"
    secondary_motion: List[str] = field(default_factory=list)
    environment_motion: List[str] = field(default_factory=list)
    shader_effects: List[str] = field(default_factory=list)
    estimated_duration_s: float = 4.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualAssetRequirement:
    """Rigorous Phase 7 consumption contract for image generation."""
    asset_id: str = ""
    scene_id: int = 1
    shot_id: int = 1
    asset_type: str = "IMAGE"
    filename: str = ""
    aspect_ratio: str = "16:9"
    resolution: Tuple[int, int] = (1280, 720)
    generation_prompt: str = ""
    visual_prompt: str = ""
    negative_prompt: str = ""
    grounding_type: str = GroundingType.HISTORICAL_CLAIM_GROUNDED.value
    claim_ids: List[str] = field(default_factory=list)
    historical_period: str = "Ancient"
    geographic_context: str = "South India"
    composition: str = "rule_of_thirds"
    lighting: str = "golden_hour"
    style: str = "photorealistic_documentary"
    safety_constraints: List[str] = field(default_factory=list)
    fallback_asset_type: str = "SOLID_COLOR_OR_GRADIENT"
    host_character_id: Optional[str] = None
    reference_asset: Optional[str] = None
    safe_fallback: Optional[str] = None

    def __post_init__(self):
        if self.visual_prompt and not self.generation_prompt:
            self.generation_prompt = self.visual_prompt
        elif self.generation_prompt and not self.visual_prompt:
            self.visual_prompt = self.generation_prompt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "scene_id": self.scene_id,
            "asset_type": self.asset_type,
            "filename": self.filename,
            "aspect_ratio": self.aspect_ratio,
            "resolution": list(self.resolution),
            "generation_prompt": self.generation_prompt,
            "negative_prompt": self.negative_prompt,
            "grounding_type": self.grounding_type,
            "claim_ids": self.claim_ids,
            "historical_period": self.historical_period,
            "geographic_context": self.geographic_context,
            "composition": self.composition,
            "lighting": self.lighting,
            "style": self.style,
            "safety_constraints": self.safety_constraints,
            "fallback_asset_type": self.fallback_asset_type
        }


@dataclass
class VisualSafetyMetadata:
    """Safety and anti-anachronism safeguards."""
    anachronism_risk: str = "LOW"
    forbidden_visual_elements: List[str] = field(default_factory=list)
    verified_fact_compliance: bool = True
    conservative_language_applied: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneVisualPlan:
    """Complete visual specification for a single storyboard scene."""
    scene_id: int
    scene_type: str  # host_vlog, broll_motion, documentary_broll
    shot_type: str   # PORTRAIT, WIDE_ESTABLISHING, ARCHITECTURE, SHIP, ARTIFACT, DETAIL, MAP
    visual_intent: str
    grounding_type: GroundingType
    associated_claim_ids: List[str]
    grounded_claim_statements: List[str]
    visual_prompt: str
    negative_prompt: str
    composition_guidelines: Dict[str, Any]
    motion_strategy: VisualMotionPlan
    character_specs: Optional[Dict[str, Any]]
    asset_requirement: VisualAssetRequirement
    safety_metadata: VisualSafetyMetadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_type": self.scene_type,
            "shot_type": self.shot_type,
            "visual_intent": self.visual_intent,
            "grounding_type": self.grounding_type.value,
            "associated_claim_ids": self.associated_claim_ids,
            "grounded_claim_statements": self.grounded_claim_statements,
            "visual_prompt": self.visual_prompt,
            "negative_prompt": self.negative_prompt,
            "composition_guidelines": self.composition_guidelines,
            "motion_strategy": self.motion_strategy.to_dict(),
            "character_specs": self.character_specs,
            "asset_requirement": self.asset_requirement.to_dict(),
            "safety_metadata": self.safety_metadata.to_dict(),
        }


@dataclass
class VisualPlan:
    """Top-level master visual plan for an entire episode."""
    episode_id: str
    topic: str
    created_at: str
    script_version: str
    plan_version: str = "1.0.0"
    scenes: List[SceneVisualPlan] = field(default_factory=list)
    global_visual_style: Dict[str, Any] = field(default_factory=dict)
    character_consistency_rules: Optional[Dict[str, Any]] = None
    visual_safety_rules: Dict[str, Any] = field(default_factory=dict)
    generation_constraints: Dict[str, Any] = field(default_factory=dict)
    source_provenance: Dict[str, Any] = field(default_factory=dict)
    status: str = "PLAN_COMPLETE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "topic": self.topic,
            "created_at": self.created_at,
            "script_version": self.script_version,
            "plan_version": self.plan_version,
            "status": self.status,
            "scenes": [s.to_dict() for s in self.scenes],
            "global_visual_style": self.global_visual_style,
            "character_consistency_rules": self.character_consistency_rules,
            "visual_safety_rules": self.visual_safety_rules,
            "generation_constraints": self.generation_constraints,
            "source_provenance": self.source_provenance,
        }


# Standard documentary negative prompt preventing anachronisms and generation artifacts
BASE_NEGATIVE_PROMPT = (
    "modern elements, wristwatch, mobile phone, car, motorcycle, electrical wires, power lines, "
    "concrete road, asphalt, modern buildings, glass facades, plastic, modern clothing, sunglasses, "
    "text, typography, watermark, logo, banner, signature, low quality, deformed anatomy, blurry, "
    "cartoon, anime, 3D render plastic, CGI toy, extra limbs, bad proportions"
)


class VisualPlanner:
    """
    Autonomous Visual Planner for Phase 6.
    Transforms validated scripts into deterministic, evidence-grounded visual plans.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        channel_config: Optional[Dict[str, Any]] = None
    ):
        self.state_manager = state_manager or StateManager()
        self.channel_config = channel_config or {}

    def _resolve_character_config(
        self,
        script: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve recurring character configuration following strict priority:
        1. Explicit character specification from validated project/channel/universe configuration
        2. Explicit character information present in the validated script
        3. No recurring character (None)
        """
        # Priority 1: Explicit character specification from validated project/channel/universe configuration
        if self.channel_config:
            if self.channel_config.get("enable_host_character") is False:
                return None
            for key in ("character", "host", "recurring_character"):
                if key in self.channel_config and self.channel_config[key]:
                    val = self.channel_config[key]
                    if isinstance(val, dict):
                        return val
                    elif isinstance(val, str):
                        return {
                            "character_id": "configured_character",
                            "name": val,
                            "description": self.channel_config.get("character_description", f"Host {val}"),
                            "consistency_token": f"{val.lower().replace(' ', '_')}",
                            "rules": [
                                f"Maintain consistent appearance for {val} across all host scenes."
                            ]
                        }

        # Check project-level configuration if any
        cfg_char = getattr(autonomous_settings, "character_config", None)
        if cfg_char and isinstance(cfg_char, dict):
            return cfg_char
        cfg_name = getattr(autonomous_settings, "default_character_name", None)
        cfg_enabled = getattr(autonomous_settings, "enable_host_character", False)
        if cfg_enabled and cfg_name:
            return {
                "character_id": "configured_character",
                "name": cfg_name,
                "description": getattr(autonomous_settings, "default_character_description", f"Host {cfg_name}"),
                "consistency_token": f"{cfg_name.lower().replace(' ', '_')}",
                "rules": [
                    f"Maintain consistent appearance for {cfg_name} across all host scenes."
                ]
            }

        # Priority 2: Explicit character information present in the validated script
        for key in ("character", "host", "recurring_character", "character_profile"):
            if script.get(key):
                val = script[key]
                if isinstance(val, dict):
                    return val
                elif isinstance(val, str):
                    return {
                        "character_id": "script_character",
                        "name": val,
                        "description": script.get("character_description", f"Host {val}"),
                        "consistency_token": f"{val.lower().replace(' ', '_')}",
                        "rules": [
                            f"Maintain consistent appearance for {val} across all host scenes."
                        ]
                    }

        script_meta = script.get("metadata", {})
        if isinstance(script_meta, dict):
            for key in ("character", "host", "recurring_character"):
                if script_meta.get(key):
                    val = script_meta[key]
                    if isinstance(val, dict):
                        return val

        # Priority 3: No recurring character
        return None

    def _determine_grounding_type(
        self,
        scene: Dict[str, Any],
        associated_claims: List[VerifiedClaim]
    ) -> GroundingType:
        """
        Determine strict epistemic grounding type for a scene's visuals:
        - HISTORICAL_CLAIM_GROUNDED: verified claim directly establishes the entity/site/artifact.
          (MUST have valid non-empty claim_ids, and must not contain speculative reconstruction)
        - RECONSTRUCTION: architectural, city, or trench layout reconstruction.
        - ILLUSTRATIVE: daily ancient activity/craft without direct claim.
        - ATMOSPHERIC: landscape, river, sunrise, or background scenery.
        - ABSTRACT: diagrams, maps, or graphical badges.
        """
        shot_type = (scene.get("shot_type") or "").upper()
        v_desc = (scene.get("visual_description") or "").lower()
        v_prompt = (scene.get("visual_prompt") or "").lower()
        full_text = f"{v_desc} {v_prompt}"

        # 1. ABSTRACT: maps, diagrams, badges, conceptual graphics
        if shot_type == "MAP" or any(term in full_text for term in ["map", "diagram", "infographic", "badge", "chart", "graphic"]):
            return GroundingType.ABSTRACT

        # 2. RECONSTRUCTION: hypothetical 3D models, ancient city layouts, architectural models, hypothetical buildings
        reconstruction_terms = [
            "reconstruction", "architectural model", "3d render", "hypothetical",
            "ancient city", "settlement layout", "building layout", "street layout",
            "ancient settlement", "three-storey", "urban building"
        ]
        if any(term in full_text for term in reconstruction_terms):
            return GroundingType.RECONSTRUCTION

        # 3. ATMOSPHERIC: landscapes, rivers, sky, natural environments
        atmospheric_terms = ["landscape", "river", "sunrise", "sunset", "sky", "horizon", "atmosphere", "terrain", "ocean"]
        if any(term in full_text for term in atmospheric_terms) and not any(term in full_text for term in ["artifact", "potsherd", "inscription"]):
            return GroundingType.ATMOSPHERIC

        # 4. ILLUSTRATIVE: daily activities, crafts, tool usage, general ancient lifestyle
        illustrative_terms = ["craft", "weaving", "pottery making", "iron smelting", "daily life", "market", "farming", "trade activity"]
        if any(term in full_text for term in illustrative_terms) and not associated_claims:
            return GroundingType.ILLUSTRATIVE

        # 5. HISTORICAL_CLAIM_GROUNDED: ONLY if verified claims exist and directly support this visual
        if associated_claims and len(associated_claims) > 0:
            return GroundingType.HISTORICAL_CLAIM_GROUNDED

        return GroundingType.ILLUSTRATIVE

    def _sanitize_conservative_prompt(
        self,
        raw_prompt: str,
        topic: str,
        grounding_type: GroundingType,
        claims: List[VerifiedClaim],
        has_character: bool = False
    ) -> str:
        """
        Hardens visual prompt to use conservative, evidence-grounded terminology:
        - Strips unconfigured host references if character is not configured.
        - Strips unsupported architectural absolutes (e.g. 'three-storey', 'palatial').
        - Strips invented population numbers, exact clothing, weapons, tools, city layouts.
        - Enforces conservative archaeological framing when evidence is reconstruction or insufficient.
        """
        prompt = raw_prompt.strip()

        # 1. If no recurring character configured, remove unconfigured host/vlogger references
        if not has_character:
            host_patterns = [
                r"\bHost\s+[A-Za-z]+\b", r"\bHost\b", r"\bAI vlogger\b", r"\bvlogger\b",
                r"\bholding (?:a )?camera\b", r"\bholding recording device\b",
                r"\bvlog framing\b", r"\btravel vlog perspective\b",
                r"\bsmiling warmly at the camera\b", r"\bpointing toward camera\b",
                r"\bfront-facing camera vlog perspective\b"
            ]
            for pat in host_patterns:
                prompt = re.sub(pat, "", prompt, flags=re.IGNORECASE)

        # 2. Unsupported exact architecture / building levels / layouts
        unsupported_architecture = [
            (r"\b(?:two|three|four|five|\d+)[ -](?:storey|story|storied|levels?)[ -]?(?:urban )?building(?:s)?\b",
             "early South Indian settlement structures"),
            (r"\bthree-storey\b", "broadly supported archaeological structures"),
            (r"\bfour-storey\b", "broadly supported archaeological structures"),
            (r"\bpalatial\b", "early brick structures"),
            (r"\bmonumental fortress\b", "settlement enclosure"),
            (r"\bgolden towers?\b", "broadly contextual ancient structures"),
            (r"\bsprawling multi-level city\b", "early settlement environment"),
            (r"\bvast metropolis\b", "early settlement environment"),
            (r"\bunmatched luxury\b", "settlement context"),
        ]
        for pat, repl in unsupported_architecture:
            prompt = re.sub(pat, repl, prompt, flags=re.IGNORECASE)

        # 3. Unsupported exact clothing
        unsupported_clothing = [
            (r"\bprecisely documented clothing\b", "broadly supported period attire"),
            (r"\belaborate silk garments\b", "period-appropriate woven attire"),
            (r"\bgold-embroidered robes\b", "broadly supported context attire"),
            (r"\broyal crowns?\b", "broadly contextual headwear"),
        ]
        for pat, repl in unsupported_clothing:
            prompt = re.sub(pat, repl, prompt, flags=re.IGNORECASE)

        # 4. Unsupported exact population numbers
        prompt = re.sub(r"\b\d+[\d,]*\s+(?:citizens|inhabitants|residents|dwellers|people)\b", "settlement inhabitants", prompt, flags=re.IGNORECASE)

        # 5. Unsupported exact weapons & tools (unless verified in claim statement)
        claim_corpus = " ".join([c.statement.lower() for c in claims])
        if "sword" not in claim_corpus and "weapon" not in claim_corpus:
            prompt = re.sub(r"\b(?:steel longswords?|iron battleaxes?|crossbows?)\b", "broadly contextual ancient tools", prompt, flags=re.IGNORECASE)

        # 6. Framing according to GroundingType
        if grounding_type == GroundingType.RECONSTRUCTION:
            reconstruction_disclaimer = "Historically cautious reconstruction of an early South Indian settlement environment, using only broadly supported archaeological context; avoid unsupported architectural specifics."
            if "historically cautious" not in prompt.lower():
                prompt = f"{reconstruction_disclaimer} {prompt}"
        elif grounding_type == GroundingType.ILLUSTRATIVE:
            if "illustrative" not in prompt.lower() and "cautious" not in prompt.lower():
                prompt = f"Historically cautious illustrative depiction of ancient daily craft environment, using broadly supported archaeological context; {prompt}"

        # Clean spaces and redundant punctuation
        prompt = re.sub(r"[,\s]+,", ",", prompt)
        prompt = re.sub(r"\s+", " ", prompt).strip(" ,.-")
        return prompt

    def generate_plan(
        self,
        episode_id: str,
        offline: bool = False
    ) -> VisualPlan:
        """
        Main entry point for Phase 6 Visual Planning.
        Strictly requires that the episode is currently SCRIPT_VALIDATED.
        """
        session = self.state_manager._get_session()
        try:
            episode = session.query(AutonomousEpisode).filter_by(episode_id=episode_id).first()
            if not episode:
                raise KeyError(f"Episode '{episode_id}' not found in database.")
            ep_dict = episode.to_dict()
        finally:
            session.close()

        # Strict State Gate: Phase 6 accepts ONLY SCRIPT_VALIDATED
        current_state = ep_dict.get("status")
        if current_state != EpisodeState.SCRIPT_VALIDATED.value:
            msg = "Phase 6 blocked: episode is not SCRIPT_VALIDATED."
            logger.warning(f"{msg} Current state is {current_state}.")
            raise ValueError(msg)

        # Execute valid transition: SCRIPT_VALIDATED -> VISUAL_PLANNING
        self.state_manager.transition_state(
            episode_id=episode_id,
            target_state=EpisodeState.VISUAL_PLANNING,
            stage_name="VISUAL_PLANNING"
        )

        ep_dir = Path(ep_dict["output_directory"])
        script_file = ep_dir / "script" / "script.json"
        if not script_file.exists():
            raise FileNotFoundError(f"Validated script not found at {script_file}")

        with open(script_file, "r", encoding="utf-8") as f:
            script_data = json.load(f)

        topic = script_data.get("title_english") or episode.topic

        # Load verified claims from confidence report if available
        conf_file = ep_dir / "verification" / "confidence_report.json"
        verified_claims: List[VerifiedClaim] = []
        claim_map: Dict[str, VerifiedClaim] = {}
        if conf_file.exists():
            try:
                with open(conf_file, "r", encoding="utf-8") as f:
                    conf_data = json.load(f)
                for cd in conf_data.get("claims_breakdown", []):
                    vc = VerifiedClaim(
                        claim_id=cd["claim_id"],
                        statement=cd["statement"],
                        normalized_statement=cd.get("normalized_statement", cd["statement"]),
                        claim_type=cd.get("claim_type", "HISTORICAL_FACT"),
                        importance=cd.get("importance", "HIGH"),
                        topic_aspect=cd.get("topic_aspect", "archaeology"),
                        episode_id=episode_id,
                        confidence_score=cd.get("confidence_score", 0.0),
                        classification=ClaimClassification(cd.get("classification", "UNVERIFIED_CLAIM")),
                        relevance_class=ClaimRelevanceClass(cd.get("relevance_class", "SUPPORTING_CONTEXT"))
                    )
                    verified_claims.append(vc)
                    claim_map[vc.claim_id] = vc
            except Exception as e:
                logger.warning(f"Could not parse confidence report for claim mapping: {e}")

        # Resolve character configuration
        character_config = self._resolve_character_config(script_data)

        # Global visual and safety guidelines
        global_style = {
            "aspect_ratio": "16:9",
            "resolution": [1280, 720],
            "style": "cinematic_photorealistic_documentary",
            "color_grading": "warm_earthy_tones_natural_golden_hour",
            "contrast": "documentary_balanced",
            "film_grain": "subtle_35mm",
            "safe_margins": {
                "top_badge_margin_px": 60,
                "bottom_subtitle_margin_px": 90
            }
        }

        visual_safety_rules = {
            "strict_anti_anachronism": True,
            "forbidden_elements": [
                "modern wristwatches", "smartphones", "automobiles", "power lines",
                "plastic items", "modern glass windows", "synthetic clothing",
                "text watermarks", "contemporary road signs", "commercial logos"
            ],
            "cultural_respect_policy": "Strict historical fidelity; no sensationalized caricature.",
            "unsupported_details_policy": "Flag as RECONSTRUCTION or ILLUSTRATIVE; do not assert as fact."
        }

        generation_constraints = {
            "primary_generator": "wan2.1_i2v_ready",
            "fallback_generator": "safe_2.5d_parallax",
            "target_format": "PNG",
            "color_depth": "8bit_RGB"
        }

        # Build scene-by-scene visual plans
        scene_plans: List[SceneVisualPlan] = []
        raw_scenes = script_data.get("scenes", [])
        if len(raw_scenes) != 4:
            logger.warning(f"Script contains {len(raw_scenes)} scenes (standard is 4). Processing all present scenes.")

        for sc in raw_scenes:
            scene_id = sc.get("id", len(scene_plans) + 1)
            raw_scene_type = sc.get("type", "broll_motion")
            shot_type = (sc.get("shot_type") or ("PORTRAIT" if raw_scene_type == "host_vlog" else "WIDE_ESTABLISHING")).upper()

            # Find associated verified claims mentioned or bound to this scene
            scene_claim_ids = sc.get("claim_ids", [])
            # If not explicitly on scene, look for claim mentions in text
            if not scene_claim_ids:
                sc_text = (sc.get("english_sub", "") + " " + sc.get("tamil_text", "")).lower()
                for c_id, clm in claim_map.items():
                    if clm.statement.lower() in sc_text or clm.normalized_statement.lower() in sc_text:
                        scene_claim_ids.append(c_id)

            associated_claims = [claim_map[cid] for cid in scene_claim_ids if cid in claim_map]
            # Exclude forbidden or rejected claims from grounding
            associated_claims = [
                c for c in associated_claims
                if c.classification in (ClaimClassification.VERIFIED_FACT, ClaimClassification.SUPPORTED_HYPOTHESIS)
            ]
            valid_claim_ids = [c.claim_id for c in associated_claims]
            grounded_stmts = [c.statement for c in associated_claims]

            # Determine Grounding Type
            grounding_type = self._determine_grounding_type(sc, associated_claims)
            if grounding_type == GroundingType.HISTORICAL_CLAIM_GROUNDED and not valid_claim_ids:
                # If marked as historical ground but lacking valid claims, degrade to ILLUSTRATIVE or RECONSTRUCTION
                grounding_type = GroundingType.RECONSTRUCTION if "excavation" in topic.lower() else GroundingType.ILLUSTRATIVE

            # Handle Character vs Documentary scenes
            scene_character_specs = None
            if raw_scene_type == "host_vlog":
                if character_config is not None:
                    scene_character_specs = {
                        "character_id": character_config.get("character_id", "host_yaazhini"),
                        "name": character_config.get("name", "Host"),
                        "description": character_config.get("description", ""),
                        "pose": "front-facing camera vlog perspective, holding recording device",
                        "framing": "medium-close portrait (head and torso)",
                        "lighting": "natural directional key light with warm ambient fill"
                    }
                else:
                    # No character configured: transform host_vlog to documentary narration visual
                    raw_scene_type = "documentary_broll"
                    sc_desc_lower = (sc.get("visual_description", "") + " " + sc.get("visual_prompt", "")).lower()
                    if any(term in sc_desc_lower for term in ["artifact", "potsherd", "pottery", "inscription", "coin", "bead"]):
                        shot_type = "ARTIFACT"
                    elif any(term in sc_desc_lower for term in ["map", "diagram", "route"]):
                        shot_type = "MAP"
                    elif any(term in sc_desc_lower for term in ["landscape", "river", "environment", "fields"]):
                        shot_type = "LANDSCAPE"
                    elif any(term in sc_desc_lower for term in ["reconstruction", "settlement", "structure", "brick"]):
                        shot_type = "ARCHITECTURE"
                    else:
                        shot_type = "WIDE_ESTABLISHING" if scene_id == 1 else "DETAIL"

            # Formulate Visual Prompt
            raw_v_prompt = sc.get("visual_prompt") or sc.get("visual_description") or f"16:9 cinematic documentary shot illustrating {topic}"
            if scene_character_specs:
                v_prompt = (
                    f"16:9 cinematic vlog documentary shot of {scene_character_specs['name']}, "
                    f"{scene_character_specs['description']}, holding camera in vlog framing, "
                    f"in the background is an ancient historical setting of {topic}, "
                    f"natural golden hour lighting, authentic historical atmosphere, high detail, 4k texture"
                )
                v_prompt = self._sanitize_conservative_prompt(v_prompt, topic, grounding_type, associated_claims, has_character=True)
            else:
                sanitized_body = self._sanitize_conservative_prompt(raw_v_prompt, topic, grounding_type, associated_claims, has_character=False)
                v_prompt = f"16:9 cinematic documentary shot, {sanitized_body}, authentic ancient archaeological details, atmospheric natural lighting, 4k texture"

            # Formulate Motion Plan
            raw_motion_plan = sc.get("motion_plan") or {}
            is_host_portrait = scene_character_specs is not None
            camera_motion = raw_motion_plan.get("camera") or ("subtle_drift" if is_host_portrait else "slow_dolly_forward")
            motion_budget = raw_motion_plan.get("motion_budget") or ("LOW" if is_host_portrait else "MEDIUM")

            shader_effects = []
            if "water" in v_prompt.lower() or "ocean" in v_prompt.lower() or "river" in v_prompt.lower():
                shader_effects.append("water_displacement")
            if not is_host_portrait:
                shader_effects.append("ambient_dust_particles")

            motion_strategy = VisualMotionPlan(
                camera_motion=camera_motion,
                motion_budget=motion_budget,
                depth_estimation_required=True,
                subject_protection_mask_required=is_host_portrait,
                primary_motion=raw_motion_plan.get("primary_motion", "camera_dolly_forward"),
                secondary_motion=raw_motion_plan.get("secondary_motion", []),
                environment_motion=raw_motion_plan.get("environment_motion", ["ambient_air_drift"]),
                shader_effects=shader_effects,
                estimated_duration_s=4.5
            )

            # Asset Requirement contract for Phase 7
            host_char_id = character_config.get("character_id", "host_yaazhini") if (is_host_portrait and character_config) else None
            ref_asset = character_config.get("reference_image", "assets/yaazhini_presenter.jpg") if (is_host_portrait and character_config) else None

            target_filename = f"scene_{scene_id}_{'host' if is_host_portrait else 'broll'}.png"
            asset_req = VisualAssetRequirement(
                asset_id=f"asset_sc_{scene_id:02d}",
                scene_id=scene_id,
                asset_type="IMAGE",
                filename=target_filename,
                aspect_ratio="16:9",
                resolution=(1280, 720),
                generation_prompt=v_prompt,
                negative_prompt=BASE_NEGATIVE_PROMPT,
                grounding_type=grounding_type.value,
                claim_ids=valid_claim_ids,
                host_character_id=host_char_id,
                reference_asset=ref_asset,
                historical_period="Ancient / Sangam Era",
                geographic_context="Tamil Nadu, South India",
                composition="rule_of_thirds_cinematic",
                lighting="golden_hour_documentary",
                style="photorealistic_historical_documentary",
                safety_constraints=[
                    "Zero modern text or watermarks in rendered image",
                    "Preserve top 60px and bottom 90px clear for video overlays"
                ],
                fallback_asset_type="COLOR_GRADIENT_TEXTURE"
            )

            safety_meta = VisualSafetyMetadata(
                anachronism_risk="LOW",
                forbidden_visual_elements=visual_safety_rules["forbidden_elements"],
                verified_fact_compliance=True,
                conservative_language_applied=True
            )

            comp_guidelines = {
                "aspect_ratio": "16:9",
                "shot_type": shot_type,
                "framing": "eye-level" if is_host_portrait else "wide-angle documentary perspective",
                "lighting": "natural golden hour",
                "focus_plane": "subject_in_focus_soft_background" if is_host_portrait else "deep_depth_of_field"
            }

            sp = SceneVisualPlan(
                scene_id=scene_id,
                scene_type=raw_scene_type,
                shot_type=shot_type,
                visual_intent=sc.get("badge", f"Scene {scene_id} visual"),
                grounding_type=grounding_type,
                associated_claim_ids=valid_claim_ids,
                grounded_claim_statements=grounded_stmts,
                visual_prompt=v_prompt,
                negative_prompt=BASE_NEGATIVE_PROMPT,
                composition_guidelines=comp_guidelines,
                motion_strategy=motion_strategy,
                character_specs=scene_character_specs,
                asset_requirement=asset_req,
                safety_metadata=safety_meta
            )
            scene_plans.append(sp)

        # Build master visual plan
        plan = VisualPlan(
            episode_id=episode_id,
            topic=topic,
            created_at=datetime.now(timezone.utc).isoformat(),
            script_version="5.1_validated",
            plan_version="1.0.0",
            scenes=scene_plans,
            global_visual_style=global_style,
            character_consistency_rules=character_config,
            visual_safety_rules=visual_safety_rules,
            generation_constraints=generation_constraints,
            source_provenance={
                "script_path": str(script_file),
                "confidence_report_path": str(conf_file) if conf_file.exists() else None,
                "verified_claims_mapped": len(claim_map)
            },
            status="PLAN_COMPLETE"
        )

        # Persist visual plan to disk
        visuals_dir = ep_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        plan_file = visuals_dir / "visual_plan.json"
        with open(plan_file, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Phase 6 Visual Plan generated successfully for '{episode_id}' ({len(scene_plans)} scenes) at {plan_file}")
        return plan
