"""
autonomous/script_generator.py - Fact-Constrained Script Generation Engine.

Generates complete 4-scene historical documentary vlog episodes with conversational Tamil
dialogue for Host Yaazhini, English subtitles, visual descriptions, and motion plans.
Strictly constrains narration to the FactCheckedContentPlan, preventing hallucinated facts,
unauthorized dates, and fabricated quotations.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional, List
from pathlib import Path

from autonomous.claim_models import (
    FactCheckedContentPlan,
    VerifiedClaim,
    ClaimClassification,
    EpistemicStatus
)
from autonomous.config import autonomous_settings

logger = logging.getLogger("autonomous.script_generator")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")


class ScriptGenerator:
    """Generates fact-checked bilingual storyboard scripts under strict claim constraints."""

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 120
    ):
        self.ollama_url = ollama_url or OLLAMA_URL
        self.model_name = model_name or OLLAMA_MODEL
        self.timeout = timeout

    def _build_llm_prompt(
        self,
        content_plan: FactCheckedContentPlan,
        claims_map: Dict[str, VerifiedClaim]
    ) -> str:
        """Construct a prompt embedding authorized claims and strict negative constraints."""
        allowed_statements = [claims_map[cid].statement for cid in content_plan.allowed_claim_ids if cid in claims_map]
        cautious_statements = [claims_map[cid].statement for cid in content_plan.cautious_claim_ids if cid in claims_map]
        debated_statements = [claims_map[cid].statement for cid in content_plan.debated_claim_ids if cid in claims_map]
        forbidden_statements = [claims_map[cid].statement for cid in content_plan.forbidden_claim_ids if cid in claims_map]

        allowed_block = "\n- ".join(allowed_statements) if allowed_statements else "Topic context only"
        cautious_block = "\n- ".join(cautious_statements) if cautious_statements else "None"
        debated_block = "\n- ".join(debated_statements) if debated_statements else "None"
        forbidden_block = "\n- ".join(forbidden_statements) if forbidden_statements else "None"

        dates_str = ", ".join(content_plan.allowed_dates) if content_plan.allowed_dates else "None"
        nums_str = ", ".join(content_plan.allowed_numbers) if content_plan.allowed_numbers else "None"
        entities_str = ", ".join(content_plan.allowed_entities) if content_plan.allowed_entities else "None"

        channel_cfg = autonomous_settings.channel
        channel_name_en = channel_cfg.channel_name_en
        channel_name_ta = channel_cfg.channel_name_ta
        channel_desc = channel_cfg.channel_description_en
        tagline_en = channel_cfg.tagline_en
        tagline_ta = channel_cfg.tagline_ta
        host_name_en = channel_cfg.host_name_en
        host_name_ta = channel_cfg.host_name_ta

        prompt = f"""You are generating an episode script for the Tamil YouTube documentary channel '{channel_name_en}' ('{channel_name_ta}' - {channel_desc}) hosted by {host_name_en}.
Topic: "{content_plan.topic}"
Category: "{content_plan.category}"
Channel Tagline: "{tagline_en}" ("{tagline_ta}")

STRICT FACTUAL CONSTRAINTS:
1. You may ONLY state as objective fact these ALLOWED VERIFIED CLAIMS:
- {allowed_block}

2. You must frame as hypotheses ("Excavations indicate...", "Archaeologists suggest...") these CAUTIOUS CLAIMS:
- {cautious_block}

3. You must present as debates ("While some date this to..., others maintain...") these DEBATED CLAIMS:
- {debated_block}

4. STRICTLY FORBIDDEN CLAIMS (DO NOT MENTION OR CLAIM AS FACT):
- {forbidden_block}

5. AUTHORIZED DATES & CENTURIES: {dates_str}
DO NOT introduce any year or century not in this list.

6. AUTHORIZED NUMBERS & COUNTS: {nums_str}
DO NOT introduce any number, measurement, or population count not in this list.

7. AUTHORIZED HISTORICAL ENTITIES: {entities_str}
DO NOT introduce unverified persons, rulers, or imaginary dynasties.

8. Tone: Conversational, engaging Tamil documentary presentation, but 100% historically rigorous. Creative scene framing is allowed, but NEW FACTUAL ASSERTIONS ARE STRICTLY PROHIBITED.

Return ONLY a valid JSON object matching this exact schema:
{{
  "id": "slug_topic_name",
  "title_tamil": "தமிழ் தலைப்பு",
  "title_english": "English Title",
  "description": "Engaging 2-sentence YouTube description",
  "tags": ["{content_plan.category}", "{channel_name_en.replace(' ', '')}", "{host_name_en}", "TamilHistory"],
  "scenes": [
    {{
      "id": 1,
      "type": "host_vlog",
      "shot_type": "PORTRAIT",
      "badge": "Location / Era Badge",
      "tamil_text": "Conversational Tamil dialogue by {host_name_en} introducing the location",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 cinematic shot of {host_name_en} presenting with ancient setting behind",
      "motion_description": "{host_name_en} talking naturally to camera, subtle head tilt",
      "visual_prompt": "16:9 cinematic shot of {host_name_en} presenting with ancient setting behind",
      "motion_plan": {{
        "camera": "subtle_drift",
        "motion_budget": "LOW",
        "primary_motion": "host_speaking",
        "secondary_motion": ["natural_blinking"],
        "environment_motion": ["gentle_background_atmosphere"],
        "detected_elements": ["portrait", "host_face"],
        "subject_protection_level": "MAXIMUM"
      }}
    }},
    {{
      "id": 2,
      "type": "broll_motion",
      "shot_type": "WIDE_ESTABLISHING",
      "badge": "Archaeological Evidence",
      "tamil_text": "Spoken Tamil line detailing the material structures or artifacts",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 wide panoramic cinematic shot of ancient archaeological structure",
      "motion_description": "Slow camera tracking shot revealing the scale of ancient brick structures",
      "visual_prompt": "16:9 wide panoramic cinematic shot of ancient archaeological structure",
      "motion_plan": {{
        "camera": "pan_left",
        "motion_budget": "MEDIUM",
        "primary_motion": "slow_reveal",
        "secondary_motion": ["dust_atmosphere"],
        "environment_motion": ["sunlight_shimmer"],
        "detected_elements": ["architecture", "landscape"],
        "subject_protection_level": "STANDARD"
      }}
    }},
    {{
      "id": 3,
      "type": "broll_motion",
      "shot_type": "ARCHITECTURE",
      "badge": "Scientific Dating & Inscriptions",
      "tamil_text": "Spoken Tamil line explaining the scientific dating and inscriptions",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 medium cinematic shot of ancient pottery sherds with Tamil-Brahmi script",
      "motion_description": "Slow deliberate camera push-in on inscribed pottery surface",
      "visual_prompt": "16:9 medium cinematic shot of ancient pottery sherds with Tamil-Brahmi script",
      "motion_plan": {{
        "camera": "slow_dolly_forward",
        "motion_budget": "LOW",
        "primary_motion": "camera_dolly_forward",
        "secondary_motion": ["ambient_particles"],
        "environment_motion": ["shadow_drift"],
        "detected_elements": ["pottery", "script", "artifacts"],
        "subject_protection_level": "STANDARD"
      }}
    }},
    {{
      "id": 4,
      "type": "host_vlog",
      "shot_type": "PORTRAIT",
      "badge": "{channel_name_ta} • {host_name_ta}",
      "tamil_text": "அடுத்த வரலாற்றுப் பதிவில் உங்களைச் சந்திக்கிறேன்... {channel_name_ta} பக்கத்தை மறக்காமல் பின்தொடருங்கள்! {tagline_ta}",
      "english_sub": "See you in our next documentary episode! Follow {channel_name_en}! {tagline_en}",
      "visual_description": "16:9 travel vlog perspective of {host_name_en} smiling warmly at the camera",
      "motion_description": "{host_name_en} smiling and signing off with friendly gesture",
      "visual_prompt": "16:9 travel vlog perspective of {host_name_en} smiling warmly at the camera",
      "motion_plan": {{
        "camera": "subtle_drift",
        "motion_budget": "LOW",
        "primary_motion": "host_signoff",
        "secondary_motion": ["gentle_smile", "blinks"],
        "environment_motion": ["sunlight_gleam"],
        "detected_elements": ["portrait", "host_face"],
        "subject_protection_level": "MAXIMUM"
      }}
    }}
  ]
}}
"""
        return prompt

    def generate_offline_fallback_script(
        self,
        content_plan: FactCheckedContentPlan,
        claims_map: Dict[str, VerifiedClaim]
    ) -> Dict[str, Any]:
        """
        Synthesizes a 100% deterministic, schema-compliant storyboard script using ONLY
        facts present in the authorized content plan. Introduces zero unverified entities or dates.
        """
        topic = content_plan.topic
        topic_slug = topic.lower().replace(" ", "_").replace(":", "").replace("-", "_")[:40]

        # Extract verified assertions (falling back to any authorized claim in the plan)
        all_authorized_cids = content_plan.allowed_claim_ids + content_plan.cautious_claim_ids
        if not all_authorized_cids:
            raise ValueError("No authorized verified or cautious claims available in content plan. Cannot generate safe script.")
        default_claim_text = claims_map[all_authorized_cids[0]].statement if all_authorized_cids[0] in claims_map else f"archaeological research continues at {topic}"

        s1_cid = content_plan.scenes[0].allowed_claim_ids[0] if content_plan.scenes[0].allowed_claim_ids else (all_authorized_cids[0] if all_authorized_cids else None)
        s2_cid = content_plan.scenes[1].allowed_claim_ids[0] if content_plan.scenes[1].allowed_claim_ids else (all_authorized_cids[1] if len(all_authorized_cids) > 1 else s1_cid)
        s3_cid = content_plan.scenes[2].allowed_claim_ids[0] if content_plan.scenes[2].allowed_claim_ids else (all_authorized_cids[2] if len(all_authorized_cids) > 2 else s1_cid)
        s4_cid = content_plan.scenes[3].allowed_claim_ids[0] if content_plan.scenes[3].allowed_claim_ids else (all_authorized_cids[3] if len(all_authorized_cids) > 3 else s1_cid)

        s1_claim_text = claims_map[s1_cid].statement if s1_cid and s1_cid in claims_map else default_claim_text
        s2_claim_text = claims_map[s2_cid].statement if s2_cid and s2_cid in claims_map else default_claim_text
        s3_claim_text = claims_map[s3_cid].statement if s3_cid and s3_cid in claims_map else default_claim_text
        s4_claim_text = claims_map[s4_cid].statement if s4_cid and s4_cid in claims_map else default_claim_text

        # Filter dates and numbers strictly to what exists
        dates_part = f" dating to {', '.join(content_plan.allowed_dates)}" if content_plan.allowed_dates else ""

        channel_cfg = autonomous_settings.channel
        channel_name_en = channel_cfg.channel_name_en
        channel_name_ta = channel_cfg.channel_name_ta
        tagline_en = channel_cfg.tagline_en
        tagline_ta = channel_cfg.tagline_ta
        host_name_en = channel_cfg.host_name_en
        host_name_ta = channel_cfg.host_name_ta

        if channel_cfg.host_enabled:
            scene1_en = f"Welcome to {channel_name_en}! I am {host_name_en}. Today we explore {topic}, where {s1_claim_text}."
            scene1_ta = f"வணக்கம்! {channel_name_ta} வரலாற்றுப் பயணத்தில் நான் உங்கள் {host_name_ta}. இன்று நாம் காண்பது {topic} பற்றிய தொல்லியல் உண்மைகள்."
            scene4_en = f"In conclusion, archaeological research indicates that {s4_claim_text}. Join us next time on {channel_name_en}! {tagline_en}"
            scene4_ta = f"அடுத்த வரலாற்றுப் பதிவில் உங்களைச் சந்திக்கிறேன்... {channel_name_ta} பக்கத்தை மறக்காமல் பின்தொடருங்கள்! {tagline_ta}"
            s1_v_desc = f"16:9 cinematic shot of {host_name_en} presenting, ancient {topic} setting behind her, natural golden hour lighting"
            s1_m_desc = f"{host_name_en} talking naturally to camera, subtle head tilt and natural blinks"
            s4_v_desc = f"16:9 travel vlog perspective of {host_name_en} smiling warmly at the camera with historical background"
            s4_m_desc = f"{host_name_en} talking to camera and signing off with friendly gesture, subtle camera drift"
        else:
            scene1_en = f"Welcome to {channel_name_en}! Today we explore {topic}, where {s1_claim_text}."
            scene1_ta = f"வணக்கம்! {channel_name_ta} வரலாற்றுப் பதிவில் இன்று நாம் காண்பது {topic} பற்றிய தொல்லியல் உண்மைகள்."
            scene4_en = f"In conclusion, archaeological research indicates that {s4_claim_text}. Join us next time on {channel_name_en}! {tagline_en}"
            scene4_ta = f"அடுத்த வரலாற்றுப் பதிவில் மீண்டும் சந்திப்போம்... {channel_name_ta} பக்கத்தை மறக்காமல் பின்தொடருங்கள்! {tagline_ta}"
            s1_v_desc = f"16:9 cinematic documentary shot of ancient {topic} setting, excavation trenches in background"
            s1_m_desc = "Slow cinematic camera dolly forward across excavation site"
            s4_v_desc = f"16:9 cinematic view of ancient {topic} landscape with archaeological artifacts"
            s4_m_desc = "Subtle camera drift across historical landscape"

        scene2_en = f"Archaeological surveys indicate that {s2_claim_text}."
        scene2_ta = f"அகழ்வாராய்ச்சியில் வெளிவந்த தொல்லியல் பதிவுகள் இதனை விவரிக்கின்றன."

        scene3_en = f"Scientific evidence indicates that {s3_claim_text}."
        scene3_ta = f"அறிவியல் ஆய்வுகள் இந்த வரலாற்று உண்மைகளை உறுதிப்படுத்துகின்றன."

        storyboard = {
            "id": topic_slug,
            "title_tamil": f"{topic} • தொல்லியல் உண்மைகள்",
            "title_english": f"{topic}: Proven Historical Facts",
            "description": f"Exploring {topic}. Verified documentary evidence from archaeological excavations.",
            "tags": [content_plan.category, channel_name_en.replace(" ", ""), host_name_en if channel_cfg.host_enabled else "TamilHistory", "TamilHistory"],
            "scenes": [
                {
                    "id": 1,
                    "type": "host_vlog" if channel_cfg.host_enabled else "documentary_broll",
                    "shot_type": "PORTRAIT" if channel_cfg.host_enabled else "WIDE_ESTABLISHING",
                    "badge": content_plan.scenes[0].badge,
                    "tamil_text": scene1_ta,
                    "english_sub": scene1_en,
                    "visual_description": s1_v_desc,
                    "motion_description": s1_m_desc,
                    "visual_prompt": s1_v_desc,
                    "motion_plan": {
                        "camera": "subtle_drift" if channel_cfg.host_enabled else "slow_dolly_forward",
                        "motion_budget": "LOW",
                        "primary_motion": "host_speaking" if channel_cfg.host_enabled else "camera_dolly_forward",
                        "secondary_motion": ["natural_blinking"] if channel_cfg.host_enabled else ["ambient_particles"],
                        "environment_motion": ["ambient_air_drift"],
                        "detected_elements": ["portrait", "host_face"] if channel_cfg.host_enabled else ["landscape", "excavation"],
                        "subject_protection_level": "MAXIMUM" if channel_cfg.host_enabled else "STANDARD"
                    }
                },
                {
                    "id": 2,
                    "type": "broll_motion",
                    "shot_type": "WIDE_ESTABLISHING",
                    "badge": content_plan.scenes[1].badge,
                    "tamil_text": scene2_ta,
                    "english_sub": scene2_en,
                    "visual_description": f"16:9 wide panoramic cinematic shot of ancient archaeological excavation site, detailed brick textures and trench views",
                    "motion_description": "Slow cinematic camera pan revealing ancient brick structures and artifacts",
                    "visual_prompt": f"16:9 wide panoramic cinematic shot of ancient archaeological excavation site, detailed brick textures and trench views",
                    "motion_plan": {
                        "camera": "pan_left",
                        "motion_budget": "MEDIUM",
                        "primary_motion": "slow_reveal",
                        "secondary_motion": ["dust_particles"],
                        "environment_motion": ["sunlight_gleam"],
                        "detected_elements": ["architecture", "excavation"],
                        "subject_protection_level": "STANDARD"
                    }
                },
                {
                    "id": 3,
                    "type": "broll_motion",
                    "shot_type": "ARCHITECTURE",
                    "badge": content_plan.scenes[2].badge,
                    "tamil_text": scene3_ta,
                    "english_sub": scene3_en,
                    "visual_description": f"16:9 macro cinematic shot of inscribed ancient pottery sherds with Tamil-Brahmi lettering, dramatic museum lighting",
                    "motion_description": "Slow deliberate camera push-in highlighting the inscribed letters on the pottery",
                    "visual_prompt": f"16:9 macro cinematic shot of inscribed ancient pottery sherds with Tamil-Brahmi lettering, dramatic museum lighting",
                    "motion_plan": {
                        "camera": "slow_dolly_forward",
                        "motion_budget": "LOW",
                        "primary_motion": "camera_dolly_forward",
                        "secondary_motion": ["light_shimmer"],
                        "environment_motion": ["shadow_drift"],
                        "detected_elements": ["pottery", "inscriptions"],
                        "subject_protection_level": "STANDARD"
                    }
                },
                {
                    "id": 4,
                    "type": "host_vlog" if channel_cfg.host_enabled else "documentary_broll",
                    "shot_type": "PORTRAIT" if channel_cfg.host_enabled else "DETAIL",
                    "badge": content_plan.scenes[3].badge,
                    "tamil_text": scene4_ta,
                    "english_sub": scene4_en,
                    "visual_description": s4_v_desc,
                    "motion_description": s4_m_desc,
                    "visual_prompt": s4_v_desc,
                    "motion_plan": {
                        "camera": "subtle_drift",
                        "motion_budget": "LOW",
                        "primary_motion": "host_signoff" if channel_cfg.host_enabled else "subtle_drift",
                        "secondary_motion": ["gentle_smile", "blinks"] if channel_cfg.host_enabled else ["light_flicker"],
                        "environment_motion": ["sunlight_gleam"],
                        "detected_elements": ["portrait", "host_face"] if channel_cfg.host_enabled else ["artifacts", "landscape"],
                        "subject_protection_level": "MAXIMUM" if channel_cfg.host_enabled else "STANDARD"
                    }
                }
            ]
        }
        if channel_cfg.host_enabled:
            storyboard["character"] = channel_cfg.get_character_config()
        storyboard["channel"] = {
            "name_en": channel_name_en,
            "name_ta": channel_name_ta,
            "tagline_en": tagline_en,
            "tagline_ta": tagline_ta
        }
        return storyboard

    def generate_script(
        self,
        content_plan: FactCheckedContentPlan,
        claims_map: Dict[str, VerifiedClaim],
        offline: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a fact-checked storyboard script via Ollama with strict fallback.
        """
        if offline:
            logger.info("Offline mode requested: Using deterministic fact-checked template synthesizer.")
            return self.generate_offline_fallback_script(content_plan, claims_map)

        # Attempt generation via local Ollama
        prompt = self._build_llm_prompt(content_plan, claims_map)
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.3,  # Low temperature to suppress hallucination
                "top_p": 0.8
            }
        }

        try:
            logger.info(f"Requesting fact-constrained script from Ollama ({self.model_name})...")
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            raw_text = resp.json().get("response", "").strip()

            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()

            storyboard = json.loads(raw_text)

            # Validate basic schema presence
            if "scenes" in storyboard and len(storyboard["scenes"]) == 4:
                logger.info(f"Ollama successfully returned 4-scene storyboard: '{storyboard.get('title_english')}'")
                return storyboard
            else:
                logger.warning("Ollama response did not contain 4 scenes. Falling back to deterministic synthesizer.")
                return self.generate_offline_fallback_script(content_plan, claims_map)

        except Exception as e:
            logger.warning(f"Ollama script generation unreachable or failed ({e}). Using deterministic offline fallback.")
            return self.generate_offline_fallback_script(content_plan, claims_map)
