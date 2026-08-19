import os
import json
import logging
import hashlib
from typing import Dict, Any, Optional, List

logger = logging.getLogger("voice_manager")

# Default Voice Library categorized by gender, archetype, and TTS voice model ID.
# Using high-quality Microsoft Azure Neural open-source EdgeTTS models for 24kHz+ natural human voices.
DEFAULT_VOICE_LIBRARY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "en": {
        "male": {
            "young_male": {
                "id": "en-US-AndrewNeural",
                "name": "Andrew",
                "pitch": "+0Hz",
                "rate": "+0%",
                "description": "Young energetic natural male voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "deep_male": {
                "id": "en-US-ChristopherNeural",
                "name": "Christopher",
                "pitch": "-4Hz",
                "rate": "-3%",
                "description": "Resonant deep cinematic male voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "calm_male": {
                "id": "en-US-EricNeural",
                "name": "Eric",
                "pitch": "+0Hz",
                "rate": "-2%",
                "description": "Calm and articulate professional male voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "energetic_male": {
                "id": "en-US-GuyNeural",
                "name": "Guy",
                "pitch": "+2Hz",
                "rate": "+5%",
                "description": "Dynamic energetic male voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "deep_villain_male": {
                "id": "en-AU-WilliamNeural",
                "name": "William",
                "pitch": "-6Hz",
                "rate": "-5%",
                "description": "Deep authoritative villain male voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "multilingual_male": {
                "id": "en-US-BrianMultilingualNeural",
                "name": "Brian (Multilingual)",
                "pitch": "+0Hz",
                "rate": "+0%",
                "description": "Hyper-realistic conversational human male voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "warm_male": {
                "id": "en-US-SteffanNeural",
                "name": "Steffan",
                "pitch": "+0Hz",
                "rate": "-1%",
                "description": "Warm natural conversational male voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "british_male": {
                "id": "en-GB-RyanNeural",
                "name": "Ryan (British)",
                "pitch": "-1Hz",
                "rate": "-2%",
                "description": "Polished British cinematic male voice",
                "provider": "Edge-TTS Azure Neural"
            }
        },
        "female": {
            "young_female": {
                "id": "en-US-AnaNeural",
                "name": "Ana",
                "pitch": "+2Hz",
                "rate": "+0%",
                "description": "Bright young female voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "calm_female": {
                "id": "en-US-AriaNeural",
                "name": "Aria",
                "pitch": "+0Hz",
                "rate": "-2%",
                "description": "Calm poised female voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "strong_female": {
                "id": "en-US-MichelleNeural",
                "name": "Michelle",
                "pitch": "-2Hz",
                "rate": "+0%",
                "description": "Strong determined female voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "soft_female": {
                "id": "en-US-AvaNeural",
                "name": "Ava",
                "pitch": "+1Hz",
                "rate": "-3%",
                "description": "Soft gentle natural female voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "multilingual_female": {
                "id": "en-US-EmmaMultilingualNeural",
                "name": "Emma (Multilingual)",
                "pitch": "+0Hz",
                "rate": "+0%",
                "description": "Hyper-realistic conversational human female voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "friendly_female": {
                "id": "en-US-JennyNeural",
                "name": "Jenny",
                "pitch": "+0Hz",
                "rate": "+0%",
                "description": "Expressive friendly human female voice",
                "provider": "Edge-TTS Azure Neural"
            },
            "british_female": {
                "id": "en-GB-SoniaNeural",
                "name": "Sonia (British)",
                "pitch": "+0Hz",
                "rate": "-1%",
                "description": "British elegant cinematic female voice",
                "provider": "Edge-TTS Azure Neural"
            }
        }
    }
}

# Pre-seeded archetype overrides for prominent story characters
PRESET_CHARACTER_ARCHETYPES: Dict[str, Dict[str, str]] = {
    "KAELEN VANCE": {"gender": "male", "archetype": "deep_male"},
    "KAELEN": {"gender": "male", "archetype": "deep_male"},
    "NOVA THORNE": {"gender": "female", "archetype": "calm_female"},
    "NOVA": {"gender": "female", "archetype": "calm_female"},
    "DIRECTOR VANE": {"gender": "male", "archetype": "deep_villain_male"},
    "VANE": {"gender": "male", "archetype": "deep_villain_male"},
    "ARIA": {"gender": "female", "archetype": "young_female"},
    "ELENA": {"gender": "female", "archetype": "strong_female"},
    "MARK": {"gender": "male", "archetype": "young_male"},
    "REX": {"gender": "male", "archetype": "energetic_male"},
}


class VoiceManager:
    """
    Manages natural human voices, character voice assignment, consistency across episodes,
    and voice library selection.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or os.path.abspath("./voice_mappings.json")
        self.voice_library = DEFAULT_VOICE_LIBRARY
        self.character_mappings: Dict[str, Dict[str, Any]] = {}
        self._load_mappings()

    def _load_mappings(self):
        """Loads persistent character to voice assignments from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.character_mappings = json.load(f)
                logger.info(f"Loaded {len(self.character_mappings)} character voice mappings.")
            except Exception as e:
                logger.warning(f"Failed to load voice mappings file: {e}")

    def _save_mappings(self):
        """Saves character voice assignments to JSON file for cross-episode consistency."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.character_mappings, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save voice mappings file: {e}")

    def infer_gender_from_name(self, character_name: str) -> str:
        """Infers gender from character name heuristics."""
        name_upper = character_name.upper().strip()
        female_indicators = [
            "NOVA", "ARIA", "ELENA", "EVE", "SARAH", "AVA", "ZOE", "MAYA", "CHLOE",
            "LADY", "QUEEN", "MADAM", "MRS", "MS", "MISS", "GIRL", "WOMAN", "FEMALE"
        ]
        if any(indicator in name_upper for indicator in female_indicators):
            return "female"
        return "male"

    def get_or_assign_voice(self, character_name: str, language: str = "en") -> Dict[str, Any]:
        """
        Gets an existing voice assignment or assigns a consistent voice for a character.
        If a new character appears, randomly/deterministically picks an unused voice from the library.
        """
        clean_name = character_name.strip().upper()
        if not clean_name:
            clean_name = "NARRATOR"

        # Check if already assigned
        if clean_name in self.character_mappings:
            return self.character_mappings[clean_name]

        lang_library = self.voice_library.get(language, self.voice_library["en"])

        # Check preset character override
        if clean_name in PRESET_CHARACTER_ARCHETYPES:
            preset = PRESET_CHARACTER_ARCHETYPES[clean_name]
            gender = preset["gender"]
            archetype = preset["archetype"]
            voice_config = lang_library[gender].get(archetype)
            if voice_config:
                assignment = {
                    "character": clean_name,
                    "language": language,
                    "gender": gender,
                    "archetype": archetype,
                    "voice_id": voice_config["id"],
                    "pitch": voice_config["pitch"],
                    "rate": voice_config["rate"],
                    "name": voice_config["name"],
                    "description": voice_config["description"]
                }
                self.character_mappings[clean_name] = assignment
                self._save_mappings()
                return assignment

        # Deduce gender and pick voice from library
        gender = self.infer_gender_from_name(clean_name)
        gender_voices = lang_library.get(gender, lang_library["male"])

        # Find used archetypes for this gender
        used_archetypes = {
            v["archetype"]
            for v in self.character_mappings.values()
            if v.get("gender") == gender and v.get("language") == language
        }

        # Filter available (unused) archetypes
        available_archetypes = [a for a in gender_voices.keys() if a not in used_archetypes]

        if not available_archetypes:
            # All voices used, fall back to all available archetypes for deterministic hash selection
            available_archetypes = list(gender_voices.keys())

        # Select deterministically based on character name hash to keep consistency
        name_hash = int(hashlib.md5(clean_name.encode("utf-8")).hexdigest(), 16)
        selected_archetype = available_archetypes[name_hash % len(available_archetypes)]

        voice_config = gender_voices[selected_archetype]
        assignment = {
            "character": clean_name,
            "language": language,
            "gender": gender,
            "archetype": selected_archetype,
            "voice_id": voice_config["id"],
            "pitch": voice_config["pitch"],
            "rate": voice_config["rate"],
            "name": voice_config["name"],
            "description": voice_config["description"]
        }

        self.character_mappings[clean_name] = assignment
        self._save_mappings()
        logger.info(f"Assigned voice '{voice_config['name']}' ({selected_archetype}) to character '{clean_name}'")
        return assignment

    def assign_voice_from_character_data(self, character_name: str, char_data: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
        """
        Intelligently assigns a voice based on character metadata like gender, role, and personality.
        """
        clean_name = character_name.strip().upper()
        if not clean_name:
            clean_name = "NARRATOR"

        # Check if already assigned
        if clean_name in self.character_mappings:
            return self.character_mappings[clean_name]

        lang_library = self.voice_library.get(language, self.voice_library["en"])

        # Determine gender
        explicit_gender = char_data.get("gender", "").lower()
        if explicit_gender in ["male", "female"]:
            gender = explicit_gender
        else:
            gender = self.infer_gender_from_name(clean_name)

        gender_voices = lang_library.get(gender, lang_library["male"])

        # Intelligently select archetype based on role and personality
        role = str(char_data.get("role", "")).lower()
        personality = str(char_data.get("personality", "")).lower()
        bio = str(char_data.get("bio", "")).lower()
        combined_text = f"{role} {personality} {bio}"

        preferred_archetype = None
        
        # Simple heuristic mapping for archetypes
        if "villain" in combined_text or "antagonist" in combined_text or "evil" in combined_text:
            preferred_archetype = "deep_villain_male" if gender == "male" else "strong_female"
        elif "old" in combined_text or "wise" in combined_text or "mentor" in combined_text:
            preferred_archetype = "deep_male" if gender == "male" else "calm_female"
        elif "young" in combined_text or "child" in combined_text or "energetic" in combined_text:
            preferred_archetype = "young_male" if gender == "male" else "young_female"
        elif "calm" in combined_text or "professional" in combined_text or "stoic" in combined_text:
            preferred_archetype = "calm_male" if gender == "male" else "calm_female"
        elif "strong" in combined_text or "warrior" in combined_text or "brave" in combined_text:
            preferred_archetype = "energetic_male" if gender == "male" else "strong_female"

        # If preferred archetype is valid and exists, try to use it
        available_archetypes = list(gender_voices.keys())
        
        used_archetypes = {
            v["archetype"]
            for v in self.character_mappings.values()
            if v.get("gender") == gender and v.get("language") == language
        }
        
        unused_archetypes = [a for a in available_archetypes if a not in used_archetypes]

        if preferred_archetype and preferred_archetype in available_archetypes:
            # Prefer unused instances of the archetype if possible, but it's more important to match the character traits
            selected_archetype = preferred_archetype
        else:
            if not unused_archetypes:
                unused_archetypes = available_archetypes

            # Deterministic selection
            name_hash = int(hashlib.md5(clean_name.encode("utf-8")).hexdigest(), 16)
            selected_archetype = unused_archetypes[name_hash % len(unused_archetypes)]

        voice_config = gender_voices[selected_archetype]
        assignment = {
            "character": clean_name,
            "language": language,
            "gender": gender,
            "archetype": selected_archetype,
            "voice_id": voice_config["id"],
            "pitch": voice_config["pitch"],
            "rate": voice_config["rate"],
            "name": voice_config["name"],
            "description": voice_config["description"]
        }

        self.character_mappings[clean_name] = assignment
        self._save_mappings()
        logger.info(f"Intelligently assigned voice '{voice_config['name']}' ({selected_archetype}) to character '{clean_name}' based on traits")
        return assignment

    def add_custom_language_voices(self, language: str, gender: str, archetype: str, voice_config: Dict[str, Any]):
        """Modular extension allowing registration of voices for additional languages."""
        if language not in self.voice_library:
            self.voice_library[language] = {"male": {}, "female": {}}
        self.voice_library[language][gender][archetype] = voice_config
        logger.info(f"Added custom voice archetype '{archetype}' for language '{language}'")

    def get_available_voices(self, language: str = "en") -> List[Dict[str, Any]]:
        """Returns flat list of all realistic human voices available in library."""
        lang_lib = self.voice_library.get(language, self.voice_library.get("en", {}))
        result = []
        for gender, archetypes in lang_lib.items():
            for arch_key, cfg in archetypes.items():
                result.append({
                    "id": cfg["id"],
                    "name": cfg["name"],
                    "gender": gender,
                    "archetype": arch_key,
                    "pitch": cfg.get("pitch", "+0Hz"),
                    "rate": cfg.get("rate", "+0%"),
                    "description": cfg.get("description", ""),
                    "provider": cfg.get("provider", "Edge-TTS Azure Neural")
                })
        return result


voice_manager = VoiceManager()

