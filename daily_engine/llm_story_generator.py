"""
llm_story_generator.py - Local LLM Episode & Story Generator for Kaalapadhivugal.
Uses local Ollama (qwen / gemma3) to autonomously generate:
- Engaging historical documentary topics & mystery hooks
- Conversational Tamil dialogue for Host Yaazhini
- English translation subtitles
- Top broadcast badges
- Image / video visual prompts
"""

import os
import sys
import json
import requests
from pathlib import Path

# Ensure UTF-8 output on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")


SYSTEM_PROMPT = """You are an expert scriptwriter for the Tamil documentary channel 'Kaalapadhivugal' (@kaalapadhivugal, காலப் பதிவுகள்) hosted by South Indian female presenter Yaazhini (யாழினி).
Tagline: 'கடந்த காலத்தை, சான்றுகளுடன் மீண்டும் பார்ப்போம்.' (Let’s revisit the past, through the evidence.)
Your task is to generate complete historical documentary episode storyboards in Tamil with English subtitles.
The tone must be informative, curious, engaging, and respectful of Tamil history and heritage.
Always return ONLY a valid, parseable JSON object without markdown fences or extraneous text."""


def generate_story_with_llm(topic_prompt: str, model_name: str = OLLAMA_MODEL) -> dict:
    """
    Generate a full structured episode storyboard JSON using local Ollama.
    Produces decoupled visual and motion descriptions with shot-aware motion planning.
    """
    prompt = f"""Generate an exciting 4-scene historical documentary episode on this topic:
"{topic_prompt}"

Return ONLY a JSON object matching this schema:
{{
  "id": "slug_topic_name",
  "title_tamil": "தமிழ் தலைப்பு",
  "title_english": "English Title",
  "description": "Engaging 2-sentence YouTube description",
  "tags": ["TopicTag", "Yaazhini", "Kaalapadhivugal", "TamilHistory"],
  "scenes": [
    {{
      "id": 1,
      "type": "host_vlog",
      "shot_type": "PORTRAIT",
      "badge": "Time Period or Location Badge",
      "tamil_text": "Spoken Tamil line by Host Yaazhini introducing the mystery or location in conversational Tamil",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 cinematic shot of Host Yaazhini, South Indian female presenter in olive-green kurti at documentary desk, ancient setting behind her, warm natural lighting, high detail",
      "motion_description": "Host Yaazhini talking naturally to camera, subtle head tilt, natural blinks, slight camera breathing drift",
      "visual_prompt": "16:9 cinematic shot of Host Yaazhini, South Indian female presenter in olive-green kurti at documentary desk, ancient setting behind her, warm natural lighting, high detail",
      "motion_plan": {{
        "camera": "subtle_drift",
        "motion_budget": "LOW",
        "primary_motion": "host_speaking",
        "secondary_motion": ["natural_blinking", "subtle_breathing"],
        "environment_motion": ["gentle_background_atmosphere"],
        "detected_elements": ["portrait", "host_face", "architecture"],
        "subject_protection_level": "MAXIMUM"
      }}
    }},
    {{
      "id": 2,
      "type": "broll_motion",
      "shot_type": "WIDE_ESTABLISHING",
      "badge": "Ancient Architecture or Action Badge",
      "tamil_text": "Spoken Tamil line detailing the historical wonder or action",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 wide panoramic cinematic shot of ancient structure or landscape, detailed foreground textures, atmospheric perspective, golden hour lighting, no text",
      "motion_description": "Slow cinematic camera tracking shot revealing the scale of the structure, environmental elements reacting naturally",
      "visual_prompt": "16:9 wide panoramic cinematic shot of ancient structure or landscape, detailed foreground textures, atmospheric perspective, golden hour lighting, no text",
      "motion_plan": {{
        "camera": "pan_left",
        "motion_budget": "MEDIUM",
        "primary_motion": "slow_environmental_reveal",
        "secondary_motion": ["wind_swaying_foliage"],
        "environment_motion": ["atmospheric_mist_drift"],
        "detected_elements": ["architecture", "landscape", "foliage"],
        "subject_protection_level": "STANDARD"
      }}
    }},
    {{
      "id": 3,
      "type": "broll_motion",
      "shot_type": "SHIP",
      "badge": "Artifact or Climax Badge",
      "tamil_text": "Spoken Tamil line describing the discovery or ancient secret",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 medium cinematic shot of ancient action or nautical discovery, rich surface materials, ocean water or temple stones, dramatic rim lighting",
      "motion_description": "Slow deliberate camera push-in, subtle natural water ripples around hulls, atmospheric depth",
      "visual_prompt": "16:9 medium cinematic shot of ancient action or nautical discovery, rich surface materials, ocean water or temple stones, dramatic rim lighting",
      "motion_plan": {{
        "camera": "slow_dolly_forward",
        "motion_budget": "MEDIUM",
        "primary_motion": "camera_dolly_forward",
        "secondary_motion": ["water_ripples", "sails_reacting_to_breeze"],
        "environment_motion": ["ocean_shimmer"],
        "detected_elements": ["water", "ships", "ocean"],
        "subject_protection_level": "STANDARD"
      }}
    }},
    {{
      "id": 4,
      "type": "host_vlog",
      "shot_type": "PORTRAIT",
      "badge": "காலப் பதிவுகள் • யாழினி",
      "tamil_text": "கடந்த காலத்தை சான்றுகளுடன் மீண்டும் பார்ப்போம்... காலப் பதிவுகள் சேனலை மறக்காம சப்ஸ்கிரைப் பண்ணுங்க!",
      "english_sub": "Let’s revisit the past, through the evidence. Don't forget to subscribe to Kaalapadhivugal!",
      "visual_description": "16:9 cinematic documentary desk perspective of Host Yaazhini smiling warmly at the camera with historical background",
      "motion_description": "Host Yaazhini talking to camera and signing off with friendly gesture, subtle camera drift",
      "visual_prompt": "16:9 cinematic documentary desk perspective of Host Yaazhini smiling warmly at the camera with historical background",
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

    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
        }
    }

    print(f"[LLM] Requesting story from Ollama ({model_name}) for '{topic_prompt}'...")
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        data = json.loads(raw_text)

        # Ensure backward compatibility and fallback fields for all scenes
        for scene in data.get("scenes", []):
            if not scene.get("visual_prompt"):
                scene["visual_prompt"] = scene.get("visual_description", "")
            if not scene.get("visual_description"):
                scene["visual_description"] = scene.get("visual_prompt", "")
            if not scene.get("motion_description"):
                scene["motion_description"] = f"Slow cinematic camera motion, natural environmental atmosphere"
            if not scene.get("shot_type"):
                scene["shot_type"] = "PORTRAIT" if scene.get("type") == "host_vlog" else "WIDE_ESTABLISHING"
            if not scene.get("motion_plan"):
                is_portrait = scene.get("shot_type") == "PORTRAIT"
                scene["motion_plan"] = {
                    "camera": "subtle_drift" if is_portrait else "slow_dolly_forward",
                    "motion_budget": "LOW" if is_portrait else "MEDIUM",
                    "primary_motion": "host_speaking" if is_portrait else "camera_dolly_forward",
                    "secondary_motion": ["natural_blinking"] if is_portrait else ["environmental_movement"],
                    "environment_motion": ["ambient_atmosphere"],
                    "detected_elements": ["portrait"] if is_portrait else ["landscape", "architecture"],
                    "subject_protection_level": "MAXIMUM" if is_portrait else "STANDARD"
                }

        print(f"[LLM] Successfully generated story: {data.get('title_english')}")
        return data
    except Exception as e:
        print(f"[LLM] Ollama generation error: {e}. Checking fallback models...")
        if model_name != "gemma3:4b":
            return generate_story_with_llm(topic_prompt, model_name="gemma3:4b")
        raise


def generate_autonomous_story(model_name: str = OLLAMA_MODEL) -> dict:
    """
    Autonomously invent an untold Tamil historical topic and generate a complete episode storyboard
    with decoupled visual/motion descriptions and shot-aware motion planning.
    """
    prompt = """Invent an exciting untold historical mystery from ancient Tamil history (e.g., King Karikala Chola building Kallanai dam, Kumari Kandam lost continent, Pandya pearl divers of Korkai, Keezhadi ancient civilization, or Mamallapuram shore temples).
Generate a complete 4-scene historical time-travel documentary vlog episode on this topic.

Return ONLY a JSON object matching this schema:
{
  "id": "slug_topic_name",
  "title_tamil": "தமிழ் தலைப்பு",
  "title_english": "English Title",
  "description": "Engaging 2-sentence YouTube description",
  "tags": ["TopicTag", "Yaazhini", "Kaalapadhivugal", "TamilHistory"],
  "scenes": [
    {
      "id": 1,
      "type": "host_vlog",
      "shot_type": "PORTRAIT",
      "badge": "Time Period or Location Badge",
      "tamil_text": "Spoken Tamil line by Host Yaazhini introducing the mystery in conversational Tamil",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 cinematic shot of Host Yaazhini, South Indian female presenter in olive-green kurti at documentary desk, ancient setting behind her, warm natural lighting, high detail",
      "motion_description": "Host Yaazhini talking naturally to camera, subtle head tilt, natural blinks, slight camera breathing drift",
      "visual_prompt": "16:9 cinematic shot of Host Yaazhini, South Indian female presenter in olive-green kurti at documentary desk, ancient setting behind her, warm natural lighting, high detail",
      "motion_plan": {
        "camera": "subtle_drift",
        "motion_budget": "LOW",
        "primary_motion": "host_speaking",
        "secondary_motion": ["natural_blinking"],
        "environment_motion": ["ambient_light"],
        "detected_elements": ["portrait", "host_face"],
        "subject_protection_level": "MAXIMUM"
      }
    },
    {
      "id": 2,
      "type": "broll_motion",
      "shot_type": "WIDE_ESTABLISHING",
      "badge": "Ancient Architecture or Action Badge",
      "tamil_text": "Spoken Tamil line detailing the historical wonder or action",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 wide panoramic cinematic shot of ancient wonder, detailed foreground, atmospheric lighting",
      "motion_description": "Slow cinematic camera tracking shot across the ancient structure",
      "visual_prompt": "16:9 wide panoramic cinematic shot of ancient wonder, detailed foreground, atmospheric lighting",
      "motion_plan": {
        "camera": "pan_left",
        "motion_budget": "MEDIUM",
        "primary_motion": "panoramic_reveal",
        "secondary_motion": ["foliage_sway"],
        "environment_motion": ["mist_drift"],
        "detected_elements": ["architecture", "landscape"],
        "subject_protection_level": "STANDARD"
      }
    },
    {
      "id": 3,
      "type": "broll_motion",
      "shot_type": "ARCHITECTURE",
      "badge": "Artifact or Climax Badge",
      "tamil_text": "Spoken Tamil line describing the discovery or ancient secret",
      "english_sub": "English subtitle translation",
      "visual_description": "16:9 medium cinematic shot of ancient mystery or artifact, rich stone textures, dramatic lighting",
      "motion_description": "Slow deliberate camera push-in, subtle light flickering from torches or lamps",
      "visual_prompt": "16:9 medium cinematic shot of ancient mystery or artifact, rich stone textures, dramatic lighting",
      "motion_plan": {
        "camera": "slow_dolly_forward",
        "motion_budget": "LOW",
        "primary_motion": "camera_dolly_forward",
        "secondary_motion": ["lamp_flicker"],
        "environment_motion": ["dust_particles"],
        "detected_elements": ["architecture", "stone", "lamps"],
        "subject_protection_level": "STANDARD"
      }
    },
    {
      "id": 4,
      "type": "host_vlog",
      "shot_type": "PORTRAIT",
      "badge": "காலப் பதிவுகள் • யாழினி",
      "tamil_text": "கடந்த காலத்தை சான்றுகளுடன் மீண்டும் பார்ப்போம்... காலப் பதிவுகள் சேனலை மறக்காம சப்ஸ்கிரைப் பண்ணுங்க!",
      "english_sub": "Let’s revisit the past, through the evidence. Don't forget to subscribe to Kaalapadhivugal!",
      "visual_description": "Host Yaazhini smiling and presenting at documentary desk in 16:9 perspective",
      "motion_description": "Host Yaazhini smiling warmly and signing off, gentle camera breathing drift",
      "visual_prompt": "Host Yaazhini smiling and presenting at documentary desk in 16:9 perspective",
      "motion_plan": {
        "camera": "subtle_drift",
        "motion_budget": "LOW",
        "primary_motion": "host_signoff",
        "secondary_motion": ["smile", "blinks"],
        "environment_motion": ["ambient_glow"],
        "detected_elements": ["portrait", "host_face"],
        "subject_protection_level": "MAXIMUM"
      }
    }
  ]
}
"""
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.8,
            "top_p": 0.9,
        }
    }

    print(f"[LLM] Autonomously brainstorming untold historical topic via Ollama ({model_name})...")
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        data = json.loads(raw_text)

        # Ensure backward compatibility and fallback fields for all scenes
        for scene in data.get("scenes", []):
            if not scene.get("visual_prompt"):
                scene["visual_prompt"] = scene.get("visual_description", "")
            if not scene.get("visual_description"):
                scene["visual_description"] = scene.get("visual_prompt", "")
            if not scene.get("motion_description"):
                scene["motion_description"] = f"Slow cinematic camera motion, natural environmental atmosphere"
            if not scene.get("shot_type"):
                scene["shot_type"] = "PORTRAIT" if scene.get("type") == "host_vlog" else "WIDE_ESTABLISHING"
            if not scene.get("motion_plan"):
                is_portrait = scene.get("shot_type") == "PORTRAIT"
                scene["motion_plan"] = {
                    "camera": "subtle_drift" if is_portrait else "slow_dolly_forward",
                    "motion_budget": "LOW" if is_portrait else "MEDIUM",
                    "primary_motion": "host_speaking" if is_portrait else "camera_dolly_forward",
                    "secondary_motion": ["natural_blinking"] if is_portrait else ["environmental_movement"],
                    "environment_motion": ["ambient_atmosphere"],
                    "detected_elements": ["portrait"] if is_portrait else ["landscape", "architecture"],
                    "subject_protection_level": "MAXIMUM" if is_portrait else "STANDARD"
                }

        print(f"[LLM] Successfully brainstormed & generated story: {data.get('title_english')}")
        return data
    except Exception as e:
        print(f"[LLM] Ollama autonomous generation error: {e}. Trying fallback to gemma3:4b...")
        if model_name != "gemma3:4b":
            return generate_autonomous_story(model_name="gemma3:4b")
        raise


if __name__ == "__main__":
    topic = "The Lost Sunken Port of Poompuhar (Kaveripoompattinam)"
    result = generate_story_with_llm(topic)
    print(json.dumps(result, ensure_ascii=False, indent=2))
