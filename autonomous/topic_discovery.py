"""
autonomous/topic_discovery.py - Autonomous Topic Candidate Discovery Engine.
Loads candidate topics from existing project catalogs (daily_engine/topics.json),
built-in historical repertoires, and database history.
Normalizes candidate metadata and prepares structured TopicCandidate objects.
"""

import os
import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from autonomous.config import autonomous_settings, BASE_DIR

logger = logging.getLogger("autonomous.topic_discovery")

DEFAULT_TOPICS_FILE = BASE_DIR / "daily_engine" / "topics.json"


def normalize_title(title: str) -> str:
    """Normalize a title for robust comparison (lowercased, punctuation-stripped, single-spaced)."""
    if not title:
        return ""
    # Remove punctuation and special characters
    clean = re.sub(r"[^\w\s]", " ", title.lower(), flags=re.UNICODE)
    # Collapse whitespace
    return " ".join(clean.split())


@dataclass
class TopicCandidate:
    """Structured representation of a potential historical video topic."""
    topic_id: str
    title: str
    title_tamil: Optional[str] = None
    category: str = "Ancient Tamil history"
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    historical_period: str = "Ancient"
    geography: str = "South India"
    source: str = "catalog"  # catalog, curated, database, feed

    # Measurable dimensions (0.0 to 1.0)
    visual_potential: float = 0.80
    educational_value: float = 0.85
    storytelling_potential: float = 0.80
    researchability: float = 0.80
    novelty: float = 0.75
    audience_potential: float = 0.80
    episode_suitability: float = 0.90

    # Operational metrics
    difficulty: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    estimated_episode_length: float = 180.0
    previously_used: bool = False
    last_used_at: Optional[str] = None
    usage_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_title(self) -> str:
        """Returns the normalized string of the title."""
        return normalize_title(self.title)

    def to_dict(self) -> Dict[str, Any]:
        """Convert candidate to a serializable dictionary."""
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "title_tamil": self.title_tamil,
            "category": self.category,
            "description": self.description,
            "keywords": self.keywords,
            "historical_period": self.historical_period,
            "geography": self.geography,
            "source": self.source,
            "visual_potential": self.visual_potential,
            "educational_value": self.educational_value,
            "storytelling_potential": self.storytelling_potential,
            "researchability": self.researchability,
            "novelty": self.novelty,
            "audience_potential": self.audience_potential,
            "episode_suitability": self.episode_suitability,
            "difficulty": self.difficulty,
            "estimated_episode_length": self.estimated_episode_length,
            "previously_used": self.previously_used,
            "last_used_at": self.last_used_at,
            "usage_count": self.usage_count,
            "failure_count": self.failure_count,
            "metadata": self.metadata,
        }


# Rich curated historical topics covering diverse categories
CURATED_HISTORICAL_CATALOG: List[Dict[str, Any]] = [
    {
        "topic_id": "poompuhar_sunken_harbor",
        "title": "Poompuhar: The Sunken Chola Port Atlantis of South India",
        "title_tamil": "பூம்புகார்: கடலில் மூழ்கிய சோழர் துறைமுக நகரம்",
        "category": "Lost cities",
        "description": "Marine archaeological expeditions off Kaveripoompattinam reveal submerged horseshoe stone docks, pottery, and ruins swallowed by the sea over 2,000 years ago.",
        "keywords": ["Poompuhar", "SubmergedCity", "Kaveripoompattinam", "MarineArchaeology", "CholaPort"],
        "historical_period": "Sangam Era (300 BCE - 300 CE)",
        "geography": "Kaveri Delta, Tamil Nadu",
        "visual_potential": 0.95,
        "educational_value": 0.92,
        "storytelling_potential": 0.94,
        "researchability": 0.88,
        "novelty": 0.85,
        "audience_potential": 0.90,
    },
    {
        "topic_id": "kallanai_grand_anicut_engineering",
        "title": "Kallanai Grand Anicut: The 2,000-Year-Old Water Engineering Miracle",
        "title_tamil": "கல்லணை: கரிகால் சோழனின் 2000 ஆண்டு கால நீர்ப்பாசன அற்புதம்",
        "category": "Ancient engineering",
        "description": "Built across the Kaveri river by Karikala Chola in the 2nd century CE, the Grand Anicut remains one of the world's oldest active water-regulator structures, built without mortar.",
        "keywords": ["Kallanai", "GrandAnicut", "KarikalaChola", "HydraulicEngineering", "AncientDams"],
        "historical_period": "Early Chola (2nd Century CE)",
        "geography": "Trichy / Thanjavur, Tamil Nadu",
        "visual_potential": 0.88,
        "educational_value": 0.95,
        "storytelling_potential": 0.86,
        "researchability": 0.92,
        "novelty": 0.80,
        "audience_potential": 0.85,
    },
    {
        "topic_id": "keezhadi_urban_sangam_civilization",
        "title": "Keezhadi: 2,600-Year-Old Tamil Urban Civilization and Early Literacy",
        "title_tamil": "கீழடி: 2600 ஆண்டுகள் பழமையான தமிழ் நாகரிகம்",
        "category": "Archaeological discoveries",
        "description": "Carbon dating confirms Keezhadi dates back to the 6th century BCE, revealing an advanced brick-built urban river valley society with early Tamil-Brahmi literate commoners.",
        "keywords": ["Keezhadi", "VaigaiCivilization", "TamilBrahmi", "SangamUrbanism", "ASIExcavations"],
        "historical_period": "6th Century BCE - 3rd Century CE",
        "geography": "Sivagangai District, Tamil Nadu",
        "visual_potential": 0.85,
        "educational_value": 0.98,
        "storytelling_potential": 0.90,
        "researchability": 0.95,
        "novelty": 0.90,
        "audience_potential": 0.92,
    },
    {
        "topic_id": "pandya_pearl_fishery_korkai",
        "title": "Korkai & The Ancient Pandya Pearl Fishery Trade with Greco-Roman Empires",
        "title_tamil": "கொற்கை: பாண்டியர்களின் சர்வதேச முத்து வர்த்தகம்",
        "category": "Ancient trade",
        "description": "How the ancient Pandya capital of Korkai monopolized the global trade of Gulf of Mannar pearls, recorded by Megasthenes, Pliny the Elder, and the Periplus of the Erythraean Sea.",
        "keywords": ["Korkai", "PandyaDynasty", "PearlFishery", "RomanTrade", "GulfOfMannar"],
        "historical_period": "Early Historic (400 BCE - 200 CE)",
        "geography": "Thoothukudi Coast, Tamil Nadu",
        "visual_potential": 0.90,
        "educational_value": 0.88,
        "storytelling_potential": 0.88,
        "researchability": 0.85,
        "novelty": 0.88,
        "audience_potential": 0.82,
    },
    {
        "topic_id": "chera_muziris_spice_emporium",
        "title": "Muziris: The Chera Ocean Port where Rome Shipped Gold for Pepper",
        "title_tamil": "முசிறி: ரோமானியர்கள் தங்கம் கொடுத்து மிளகு வாங்கிய சேரர் துறைமுகம்",
        "category": "Maritime history",
        "description": "The Papyrus Vindobonensis maritime contract documents massive Roman merchant ships anchored at Muziris in the Chera kingdom, exchanging gold and wine for Malabar black pepper.",
        "keywords": ["Muziris", "CheraKingdom", "BlackPepperTrade", "RomanEmpire", "IndianOceanMaritime"],
        "historical_period": "1st Century BCE - 2nd Century CE",
        "geography": "Pattanam, Kerala / Ancient Tamilakam",
        "visual_potential": 0.92,
        "educational_value": 0.92,
        "storytelling_potential": 0.90,
        "researchability": 0.90,
        "novelty": 0.84,
        "audience_potential": 0.86,
    },
    {
        "topic_id": "mamallapuram_shore_temple_tsunami_secrets",
        "title": "Mamallapuram Shore Temple: Monolithic Granite and Tsunami Inscriptions",
        "title_tamil": "மாமல்லபுரம் கடற்கரைக் கோவில்: பல்லவர் கல்வெட்டுகளும் ஆழிப்பேரலையும்",
        "category": "Temples and architecture",
        "description": "Built by Narasimhavarman II in the 8th century CE, the 2004 Indian Ocean tsunami briefly receded to uncover submerged relief carvings and foundations of ancient pagodas.",
        "keywords": ["Mamallapuram", "ShoreTemple", "PallavaDynasty", "TsunamiDiscovery", "UNESCOHeritage"],
        "historical_period": "Pallava Era (7th - 8th Century CE)",
        "geography": "Chengalpattu, Tamil Nadu",
        "visual_potential": 0.96,
        "educational_value": 0.90,
        "storytelling_potential": 0.92,
        "researchability": 0.90,
        "novelty": 0.82,
        "audience_potential": 0.88,
    },
    {
        "topic_id": "maravarman_rajasimha_pandya_clash",
        "title": "Maravarman Rajasimha I: The Pandya Emperor who Defeated the Chalukyas",
        "title_tamil": "முதலாம் மாறவர்மன் ராஜசிம்மன்: சாளுக்கியர்களை வென்ற பாண்டிய பேரரசன்",
        "category": "Pandya history",
        "description": "The 8th-century battles of Venbai and Sennilam where the First Pandya Empire expanded its boundaries against Pallava and Western Chalukyan coalitions, documented in Velvikkudi plates.",
        "keywords": ["MaravarmanRajasimha", "PandyaEmpire", "VelvikkudiPlates", "AncientBattles", "TamilKings"],
        "historical_period": "8th Century CE",
        "geography": "Madurai / South Tamil Nadu",
        "visual_potential": 0.84,
        "educational_value": 0.88,
        "storytelling_potential": 0.88,
        "researchability": 0.82,
        "novelty": 0.92,
        "audience_potential": 0.80,
    },
    {
        "topic_id": "gangaikonda_cholapuram_water_harvesting",
        "title": "Solagangam: Rajendra Chola's 16-Mile Liquid Pillar of Victory",
        "title_tamil": "சோழகங்கம்: ராஜேந்திர சோழனின் 16 மைல் பிரம்மாண்ட ஏரி",
        "category": "Ancient engineering",
        "description": "After his northern conquest reaching the Ganges river, Rajendra Chola poured consecrated Ganga waters into a gargantuan 16-mile artificial lake called the 'Liquid Pillar of Victory' (Cholagangam).",
        "keywords": ["GangaikondaCholapuram", "RajendraChola", "Solagangam", "AncientReservoirs", "CholaArchitecture"],
        "historical_period": "Medieval Chola (1025 - 1040 CE)",
        "geography": "Ariyalur District, Tamil Nadu",
        "visual_potential": 0.90,
        "educational_value": 0.94,
        "storytelling_potential": 0.92,
        "researchability": 0.88,
        "novelty": 0.86,
        "audience_potential": 0.85,
    },
    {
        "topic_id": "wootz_steel_kodumanal_swords",
        "title": "Kodumanal Crucible Steel: Ancient Tamil Swords that Forged Damascus Blades",
        "title_tamil": "கொடுமணல் உருக்கு எஃகு: டமாஸ்கஸ் வாள்களை உருவாக்கிய தமிழ் தொழில்நுட்பம்",
        "category": "Ancient technology",
        "description": "Excavations at Kodumanal revealed ancient crucible steel furnaces from the 3rd century BCE producing ultra-high carbon Wootz steel exported across Arabia to become Damascus steel swords.",
        "keywords": ["Kodumanal", "WootzSteel", "CrucibleSteel", "AncientMetallurgy", "DamascusBlades"],
        "historical_period": "Sangam / Iron Age (300 BCE - 100 CE)",
        "geography": "Erode District, Tamil Nadu",
        "visual_potential": 0.88,
        "educational_value": 0.96,
        "storytelling_potential": 0.90,
        "researchability": 0.90,
        "novelty": 0.94,
        "audience_potential": 0.88,
    }
]


class TopicDiscovery:
    """Discovers and normalizes topic candidates from catalogs, feeds, and project records."""

    def __init__(self, topics_file: Optional[Path] = None):
        self.topics_file = Path(topics_file or DEFAULT_TOPICS_FILE)

    def load_from_topics_json(self) -> List[TopicCandidate]:
        """Load and adapt candidates from daily_engine/topics.json."""
        if not self.topics_file.exists():
            logger.warning(f"Topics file missing at {self.topics_file}")
            return []

        candidates: List[TopicCandidate] = []
        try:
            with open(self.topics_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for ep in data.get("episodes", []):
                ep_id = ep.get("id", "untitled_topic")
                title_en = ep.get("title_english") or ep.get("title") or ep_id
                title_ta = ep.get("title_tamil")
                desc = ep.get("description", "")
                tags = ep.get("tags", [])

                # Infer category from tags/title
                category = "Ancient Tamil history"
                text_corpus = f"{title_en} {desc} {' '.join(tags)}".lower()
                if "maritime" in text_corpus or "navy" in text_corpus or "sea" in text_corpus:
                    category = "Maritime history"
                elif "temple" in text_corpus or "architecture" in text_corpus:
                    category = "Temples and architecture"
                elif "chola" in text_corpus:
                    category = "Chola history"
                elif "pandya" in text_corpus:
                    category = "Pandya history"
                elif "engineering" in text_corpus:
                    category = "Ancient engineering"

                candidate = TopicCandidate(
                    topic_id=ep_id,
                    title=title_en,
                    title_tamil=title_ta,
                    category=category,
                    description=desc,
                    keywords=tags,
                    source="topics.json",
                    visual_potential=0.92,
                    educational_value=0.90,
                    storytelling_potential=0.90,
                    researchability=0.88,
                    novelty=0.75,
                    audience_potential=0.85,
                    metadata={"scene_count": len(ep.get("scenes", []))}
                )
                candidates.append(candidate)
        except Exception as e:
            logger.error(f"Error reading topics.json ({e})")

        return candidates

    def load_curated_catalog(self) -> List[TopicCandidate]:
        """Load the curated historical repertoire."""
        candidates: List[TopicCandidate] = []
        for item in CURATED_HISTORICAL_CATALOG:
            cand = TopicCandidate(
                topic_id=item["topic_id"],
                title=item["title"],
                title_tamil=item.get("title_tamil"),
                category=item.get("category", "Ancient Tamil history"),
                description=item.get("description", ""),
                keywords=item.get("keywords", []),
                historical_period=item.get("historical_period", "Ancient"),
                geography=item.get("geography", "South India"),
                source="curated_catalog",
                visual_potential=item.get("visual_potential", 0.80),
                educational_value=item.get("educational_value", 0.85),
                storytelling_potential=item.get("storytelling_potential", 0.80),
                researchability=item.get("researchability", 0.80),
                novelty=item.get("novelty", 0.80),
                audience_potential=item.get("audience_potential", 0.80),
            )
            candidates.append(cand)
        return candidates

    def discover_candidates(self) -> List[TopicCandidate]:
        """
        Discover candidate topics across all sources.
        Deduplicates identical topic IDs and normalized titles.
        """
        all_candidates: List[TopicCandidate] = []
        seen_ids = set()
        seen_titles = set()

        # 1. Load from topics.json first (user's project catalog)
        json_candidates = self.load_from_topics_json()
        for c in json_candidates:
            if c.topic_id not in seen_ids and c.normalized_title not in seen_titles:
                seen_ids.add(c.topic_id)
                seen_titles.add(c.normalized_title)
                all_candidates.append(c)

        # 2. Load curated historical repertoire
        curated = self.load_curated_catalog()
        for c in curated:
            if c.topic_id not in seen_ids and c.normalized_title not in seen_titles:
                seen_ids.add(c.topic_id)
                seen_titles.add(c.normalized_title)
                all_candidates.append(c)

        logger.info(f"Discovered {len(all_candidates)} unique topic candidates.")
        return all_candidates
