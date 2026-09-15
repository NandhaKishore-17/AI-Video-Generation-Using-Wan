"""
autonomous/research_planner.py - Research Planning Engine for Autonomous Episodes.

Produces structured research plans from selected topics, defining targeted research
questions, chronological and geographic boundaries, key entities, and evidence types.
Note: LLM-generated research questions are planning aids, NOT factual evidence.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger("autonomous.research_planner")


@dataclass
class ResearchPlan:
    """Structured plan specifying research questions and search parameters for an episode."""
    topic: str
    research_questions: List[str]
    important_entities: List[str] = field(default_factory=list)
    historical_period: str = "Ancient Tamil / Indian History"
    geographic_scope: str = "Tamil Nadu / South India"
    key_terminology: List[str] = field(default_factory=list)
    likely_evidence_types: List[str] = field(default_factory=lambda: [
        "Archaeological artifacts",
        "Radiocarbon (AMS) dating",
        "Epigraphical inscriptions",
        "Stratigraphical excavation reports",
        "Classical Sangam literature references",
        "Comparative maritime port records"
    ])
    expected_source_types: List[str] = field(default_factory=lambda: [
        "Archaeological Survey of India (ASI) reports",
        "Tamil Nadu State Department of Archaeology publications",
        "Peer-reviewed historical & archaeological journals",
        "Epigraphia Indica",
        "Scholarly university monographs"
    ])
    potential_controversial_areas: List[str] = field(default_factory=list)
    likely_claims_requiring_stronger_evidence: List[str] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchPlan":
        return cls(
            topic=data.get("topic", "Untitled Topic"),
            research_questions=data.get("research_questions", []),
            important_entities=data.get("important_entities", []),
            historical_period=data.get("historical_period", "Ancient History"),
            geographic_scope=data.get("geographic_scope", "South India"),
            key_terminology=data.get("key_terminology", []),
            likely_evidence_types=data.get("likely_evidence_types", []),
            expected_source_types=data.get("expected_source_types", []),
            potential_controversial_areas=data.get("potential_controversial_areas", []),
            likely_claims_requiring_stronger_evidence=data.get("likely_claims_requiring_stronger_evidence", []),
            search_queries=data.get("search_queries", [])
        )


class ResearchPlanner:
    """Generates structured research plans from topic candidates or manual strings."""

    def __init__(self, ollama_client=None, use_llm: bool = False):
        self.ollama_client = ollama_client
        self.use_llm = use_llm

    def create_plan(self, topic: str, category: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ResearchPlan:
        """Create a comprehensive research plan with questions and search queries."""
        meta = metadata or {}
        
        # Try local LLM planner if requested and available
        if self.use_llm and self.ollama_client:
            try:
                plan = self._create_plan_llm(topic, category, meta)
                if plan and len(plan.research_questions) >= 4:
                    return plan
            except Exception as e:
                logger.warning(f"Ollama planning failed, falling back to deterministic planner: {e}")

        # Deterministic, domain-aware planning template
        return self._create_plan_deterministic(topic, category, meta)

    def _create_plan_deterministic(self, topic: str, category: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> ResearchPlan:
        """Deterministic domain-aware research planning."""
        topic_lower = topic.lower()

        # Tailor questions to domain keywords
        questions = [
            f"What primary archaeological or historical evidence documents {topic}?",
            f"What is the scientifically established chronology and dating for {topic}?",
            f"What material infrastructure, architecture, or artifacts have been excavated or preserved?",
            f"What epigraphical inscriptions or primary texts substantiate the historical record of {topic}?",
            f"Which scholarly interpretations of {topic} represent consensus, and which remain debated?",
            f"What claims regarding {topic} are commonly exaggerated or unverified in popular media?",
            f"What international trade, maritime links, or inter-regional connections are documented?"
        ]

        # Extract entities and period heuristics
        entities = []
        period = "Early Historic (c. 6th century BCE – 3rd century CE)"
        geographic = "Tamil Nadu, South India"
        terminology = ["Sangam", "Archaeology", "Stratigraphy", "Chronology", "Epigraphy"]
        controversial = ["Exact chronological boundaries", "Literacy origin dates", "Maritime extent claims"]
        strong_evidence_needed = ["Claims of world-first technologies", "Extreme antiquity assertions (>1000 BCE without AMS)"]

        if "keezhadi" in topic_lower or "keeladi" in topic_lower:
            period = "Sangam Era (c. 6th century BCE – 3rd century CE)"
            geographic = "Vaigai River Valley, Sivaganga District, Tamil Nadu"
            entities = ["Tamil Nadu State Department of Archaeology", "ASI", "Amarnath Ramakrishna", "K. Rajan", "Vaigai Valley"]
            terminology = ["Tamil-Brahmi script", "Black and red ware", "Brick structures", "Ring wells", "Carnelian beads", "AMS Dating"]
            controversial = ["Dating of Tamil-Brahmi script to 6th century BCE vs later", "Urban status compared to Indus/Gangetic sites"]
            strong_evidence_needed = ["Carbon dating laboratory reports (Beta Analytic)", "Epigraphical graffiti readouts"]

        elif "poompuhar" in topic_lower or "kaveripoompattinam" in topic_lower:
            period = "Early Historic to Chola (c. 3rd century BCE – 12th century CE)"
            geographic = "Mouth of Cauvery River, Bay of Bengal, Tamil Nadu"
            entities = ["National Institute of Oceanography (NIO)", "S.R. Rao", "Sangam literature (Pattinappalai, Silappadikaram)"]
            terminology = ["Submerged port", "Wharf excavations", "Intertidal archaeology", "Roman amphorae", "Lead ingots"]
            controversial = ["Extent of submerged city vs natural geological features", "Tsunami inundation vs gradual coastal erosion"]
            strong_evidence_needed = ["Side-scan sonar and underwater archaeological core samples"]

        elif "tanjore" in topic_lower or "thanjavur" in topic_lower or "brihadisvara" in topic_lower:
            period = "Middle Chola Period (1000 – 1010 CE)"
            geographic = "Thanjavur, Cauvery Delta, Tamil Nadu"
            entities = ["Raja Raja Chola I", "Kunjara Mallan Raja Raja Perunthachan (architect)", "Karuvur Devar", "ASI"]
            terminology = ["Vimana", "Kumbam (capstone)", "Granite interlocking masonry", "Chola inscriptions", "Fresco murals"]
            controversial = ["Ramp construction technique for raising the 80-tonne capstone", "Shadow myth (claim that Vimana casts no shadow)"]
            strong_evidence_needed = ["Engineering analysis of incline/scaffolding", "Epigraphical records detailing donors and dancers"]

        elif "chola" in topic_lower and "navy" in topic_lower:
            period = "Imperial Chola Period (c. 1014 – 1044 CE)"
            geographic = "Bay of Bengal, Straits of Malacca, Srivijaya (Sumatra/Malay Peninsula)"
            entities = ["Rajendra Chola I", "Srivijaya Kingdom", "Sangrama Vijayottunggavarman", "Thiruvalangadu plates"]
            terminology = ["Kalam (ships)", "Meikeerthi", "Maritime trade route", "Straits of Malacca", "Kadaram"]
            controversial = ["Whether expedition was annexation/occupation vs punitive trade expedition", "Exact fleet size and ship construction"]
            strong_evidence_needed = ["Primary copper plate inscriptions (Thiruvalangadu, Karandai)", "Song Dynasty Chinese annals"]

        elif "kallanai" in topic_lower or "grand anicut" in topic_lower:
            period = "Early Chola Period (c. 2nd century CE, renovated 19th c.)"
            geographic = "Cauvery River, Tiruchirappalli, Tamil Nadu"
            entities = ["Karikala Chola", "Sir Arthur Cotton", "Cauvery Delta Irrigation"]
            terminology = ["Check dam", "Unhewn stone in mud mortar", "Silt scouring sluices", "Anicut"]
            controversial = ["Original Early Historic masonry remnants vs British colonial reconstruction", "Original hydraulic flow calculations"]
            strong_evidence_needed = ["Hydrological survey reports and archaeological core excavations"]

        # Generate targeted search queries
        search_queries = [
            f"{topic} archaeology excavation",
            f"{topic} chronology dating",
            f"{topic} inscriptions epigraphy",
            f"{topic} scholarly research paper",
            f"{topic} site report ASI"
        ]

        return ResearchPlan(
            topic=topic,
            research_questions=questions,
            important_entities=entities,
            historical_period=period,
            geographic_scope=geographic,
            key_terminology=terminology,
            potential_controversial_areas=controversial,
            likely_claims_requiring_stronger_evidence=strong_evidence_needed,
            search_queries=search_queries
        )

    def _create_plan_llm(self, topic: str, category: Optional[str], metadata: Dict[str, Any]) -> Optional[ResearchPlan]:
        """Use local Ollama to augment research questions if running."""
        prompt = f"""You are an authoritative historical research director for documentary production.
Create a structured research plan for the topic: '{topic}' (Category: {category or 'Ancient History'}).

Respond with ONLY a JSON object formatted as follows:
{{
  "topic": "{topic}",
  "research_questions": [
    "What archaeological evidence exists for...",
    "What is the accepted chronology...",
    "What material culture has been excavated...",
    "What primary inscriptions document...",
    "What scholarly debates exist regarding..."
  ],
  "important_entities": ["Site A", "Scholar B", "Dynasty C"],
  "historical_period": "Historical Era",
  "geographic_scope": "Geographic Region",
  "key_terminology": ["term1", "term2"],
  "potential_controversial_areas": ["debate1", "debate2"],
  "likely_claims_requiring_stronger_evidence": ["claim1"],
  "search_queries": ["query1", "query2", "query3"]
}}
"""
        response_text = self.ollama_client.generate(prompt=prompt, system="You output pure JSON only.")
        clean_json = response_text.strip()
        if "```json" in clean_json:
            clean_json = clean_json.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_json:
            clean_json = clean_json.split("```")[1].split("```")[0].strip()

        data = json.loads(clean_json)
        return ResearchPlan.from_dict(data)
