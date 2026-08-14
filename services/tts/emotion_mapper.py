import re
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("emotion_mapper")

# Supported Emotion Categories
EMOTION_STYLES = [
    "normal",
    "angry",
    "fear",
    "sad",
    "happy",
    "whisper",
    "serious",
    "excited"
]

# Prosody modifications for TTS synthesis per emotion
EMOTION_PROSODY_MAP: Dict[str, Dict[str, Any]] = {
    "normal": {
        "rate": "+0%",
        "pitch": "+0Hz",
        "volume": "+0%",
        "ssml_style": "chat",
        "description": "Standard natural conversational tone"
    },
    "angry": {
        "rate": "+15%",
        "pitch": "-4Hz",
        "volume": "+25%",
        "ssml_style": "angry",
        "description": "Aggressive high-volume intense voice"
    },
    "fear": {
        "rate": "+20%",
        "pitch": "+8Hz",
        "volume": "+10%",
        "ssml_style": "fearful",
        "description": "Rapid, high-pitched panicked tone"
    },
    "sad": {
        "rate": "-20%",
        "pitch": "-8Hz",
        "volume": "-15%",
        "ssml_style": "sad",
        "description": "Slow, low-pitch sombre voice"
    },
    "happy": {
        "rate": "+10%",
        "pitch": "+6Hz",
        "volume": "+15%",
        "ssml_style": "cheerful",
        "description": "Upbeat enthusiastic tone"
    },
    "whisper": {
        "rate": "-10%",
        "pitch": "-2Hz",
        "volume": "-40%",
        "ssml_style": "whisper",
        "description": "Hushed, muted breathy voice"
    },
    "serious": {
        "rate": "-10%",
        "pitch": "-5Hz",
        "volume": "+5%",
        "ssml_style": "serious",
        "description": "Calm, firm, grave authoritative tone"
    },
    "excited": {
        "rate": "+25%",
        "pitch": "+10Hz",
        "volume": "+30%",
        "ssml_style": "excited",
        "description": "High energy urgent tone"
    }
}

# Emotion Keyword Triggers
KEYWORD_EMOTION_PATTERNS = {
    "whisper": [
        r"\bshh\b", r"\bquiet\b", r"\bwhisper\b", r"\bsilent\b", r"\bsoftly\b",
        r"\bdon't make a sound\b", r"\bhush\b", r"\bsecret\b"
    ],
    "fear": [
        r"\brun!\b", r"\bpanic\b", r"\bterrified\b", r"\bmonsters?\b", r"\bkill us\b",
        r"\btrap\b", r"\bdie\b", r"\bdanger\b", r"\bget out\b", r"\bno no no\b"
    ],
    "angry": [
        r"\bdammit\b", r"\bshut up\b", r"\bidiot\b", r"\btraitor\b", r"\bdestroy\b",
        r"\bliar\b", r"\bnever\b", r"\bfight\b", r"\bhate\b", r"\bcurse\b"
    ],
    "sad": [
        r"\bsorry\b", r"\blost\b", r"\bdead\b", r"\bgone\b", r"\bcrying\b",
        r"\bmiss you\b", r"\btear\b", r"\bhopeless\b", r"\bfailed\b", r"\bforgive me\b"
    ],
    "happy": [
        r"\bwe did it\b", r"\bawesome\b", r"\bgreat\b", r"\byes!\b", r"\bhooray\b",
        r"\bvictory\b", r"\blove\b", r"\bamazing\b", r"\bcelebrate\b"
    ],
    "serious": [
        r"\bcompromised\b", r"\bprotocol\b", r"\boverride\b", r"\bbreached\b",
        r"\bemergency\b", r"\bsituation\b", r"\bmission\b", r"\bstatus report\b",
        r"\bstand down\b", r"\bconfirm\b"
    ],
    "excited": [
        r"\bhurry\b", r"\bnot leaving\b", r"\bnow!\b", r"\bfast\b", r"\bcharge\b",
        r"\bgo go go\b", r"\bhold the line\b"
    ]
}


class EmotionMapper:
    """
    Infers emotional state from dialogue context and outputs speech prosody parameters.
    """

    def infer_emotion(self, line: str, scene_context: str = "") -> Tuple[str, Dict[str, Any]]:
        """
        Analyzes dialogue text and optional scene context to determine the best fitting emotion style.
        Returns tuple of (emotion_name, prosody_config).
        """
        line_clean = line.strip()
        line_lower = line_clean.lower()
        context_lower = scene_context.lower()

        # 1. Check for explicit stage direction brackets (e.g. "(whispering) Stay down!")
        parenthetical_match = re.search(r"[\(\[\{](.*?)[\)\]\}]", line_lower)
        if parenthetical_match:
            tag = parenthetical_match.group(1)
            for style in EMOTION_STYLES:
                if style in tag or (style == "fear" and "scared" in tag) or (style == "excited" and "urgent" in tag):
                    return style, EMOTION_PROSODY_MAP[style]

        # 2. Check for explicit keyword triggers
        for emotion, patterns in KEYWORD_EMOTION_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, line_lower) or re.search(pat, context_lower):
                    return emotion, EMOTION_PROSODY_MAP[emotion]

        # 3. Punctuation and formatting cues
        # ALL CAPS shouting detection (e.g., "RUN!", "NO!")
        words = [w for w in line_clean.split() if w.isalpha()]
        uppercase_words = [w for w in words if w.isupper() and len(w) > 1]
        if words and len(uppercase_words) / len(words) > 0.4:
            if "!" in line_clean:
                return "angry", EMOTION_PROSODY_MAP["angry"]
            return "serious", EMOTION_PROSODY_MAP["serious"]

        if "!" in line_clean:
            if any(w in line_lower for w in ["run", "no", "stop", "watch out", "look out", "hurry"]):
                return "fear", EMOTION_PROSODY_MAP["fear"]
            return "excited", EMOTION_PROSODY_MAP["excited"]

        if "..." in line_clean or "…" in line_clean:
            if any(w in line_lower for w in ["cannot", "lost", "sorry", "why"]):
                return "sad", EMOTION_PROSODY_MAP["sad"]
            return "whisper", EMOTION_PROSODY_MAP["whisper"]

        # Default to normal/serious based on text length and tone
        if len(line_clean) > 80:
            return "serious", EMOTION_PROSODY_MAP["serious"]

        return "normal", EMOTION_PROSODY_MAP["normal"]

    def clean_dialogue_text(self, text: str) -> str:
        """Removes stage directions in parentheses before passing text to TTS synthesizer."""
        cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", text)
        return re.sub(r"\s+", " ", cleaned).strip()


emotion_mapper = EmotionMapper()
