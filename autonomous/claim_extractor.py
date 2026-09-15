"""
autonomous/claim_extractor.py - Atomic Claim Extraction, Normalization & Entity Tagging.

Extracts candidate atomic historical claims from ResearchDossier findings and evidence passages.
Extracts structured dates, numbers, named entities, and superlative markers to support
rigorous, deterministic claim verification.
"""

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from autonomous.claim_models import (
    ClaimType,
    ClaimImportance,
    VerifiedClaim,
    EntityType
)
from autonomous.research_dossier import ResearchDossier, PreliminaryFinding

logger = logging.getLogger("autonomous.claim_extractor")

# Superlative indicators that strictly elevate a claim to CRITICAL
SUPERLATIVE_TERMS = {
    "first", "oldest", "earliest", "largest", "only", "most ancient",
    "greatest", "biggest", "first-ever", "unmatched", "sole", "pioneer"
}

# Regex patterns for structured extraction
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}(?:st|nd|rd|th)?\s+century\s+(?:bce|ce|bc|ad))\b", re.IGNORECASE),
    re.compile(r"\b(\d{3,4}\s*(?:bce|ce|bc|ad))\b", re.IGNORECASE),
    re.compile(r"\b(\d{3,4})\s*(?:years\s+old|bp)\b", re.IGNORECASE),
    re.compile(r"\b(?:circa|c\.)\s*(\d{3,4})\b", re.IGNORECASE),
]

NUMBER_PATTERNS = [
    re.compile(r"\b(\d{1,3}(?:,\d{3})+(?:\.\d+)?)\b"),  # 10,000 or 1,000.5
    re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:metres|meters|km|kilometers|tonnes|tons|acres|hectares|artifacts|potsherds|wells|structures)\b", re.IGNORECASE),
    re.compile(r"\b(\d{2,6})\b"),  # General multi-digit numbers (years, populations, item counts)
]

ENTITY_KEYWORDS = [
    "keeladi", "keezhadi", "vaigai", "sangam", "tamil-brahmi", "brahmi", "madurai",
    "pandya", "chola", "cheran", "pallava", "tnsda", "asi", "archaeological survey of india",
    "korkai", "poompuhar", "kaveripoompattinam", "thanjavur", "brihadeeswarar", "kallanai",
    "karikala", "rajendra chola", "raja raja chola", "amarnath ramakrishna", "carbon dating",
    "ams", "accelerator mass spectrometry", "brick structures", "drainage", "ring well",
    "beads", "carnelian", "agate", "quartz", "iron", "pottery", "graffiti", "urbanization",
    "ashoka", "alexander", "maurya", "mauryan", "greece", "greek", "rome", "roman",
    "muzhiris", "ptolemy", "pliny", "megasthenes"
]


ENTITY_REGISTRY: Dict[str, EntityType] = {
    # ARCHAEOLOGICAL_SITE
    "keeladi": EntityType.ARCHAEOLOGICAL_SITE,
    "keezhadi": EntityType.ARCHAEOLOGICAL_SITE,
    "korkai": EntityType.ARCHAEOLOGICAL_SITE,
    "poompuhar": EntityType.ARCHAEOLOGICAL_SITE,
    "kaveripoompattinam": EntityType.ARCHAEOLOGICAL_SITE,
    "kodumanal": EntityType.ARCHAEOLOGICAL_SITE,
    "adichanallur": EntityType.ARCHAEOLOGICAL_SITE,
    "arikamedu": EntityType.ARCHAEOLOGICAL_SITE,
    "pattanam": EntityType.ARCHAEOLOGICAL_SITE,
    "muzhiris": EntityType.ARCHAEOLOGICAL_SITE,
    "brihadeeswarar": EntityType.ARCHAEOLOGICAL_SITE,
    "kallanai": EntityType.ARCHAEOLOGICAL_SITE,

    # PERSON
    "amarnath ramakrishna": EntityType.PERSON,
    "karikala": EntityType.PERSON,
    "rajendra chola": EntityType.PERSON,
    "raja raja chola": EntityType.PERSON,
    "ashoka": EntityType.PERSON,
    "alexander": EntityType.PERSON,
    "ptolemy": EntityType.PERSON,
    "pliny": EntityType.PERSON,
    "megasthenes": EntityType.PERSON,

    # ORGANIZATION
    "tnsda": EntityType.ORGANIZATION,
    "asi": EntityType.ORGANIZATION,
    "archaeological survey of india": EntityType.ORGANIZATION,
    "tamil nadu state department of archaeology": EntityType.ORGANIZATION,

    # DYNASTY
    "chola": EntityType.DYNASTY,
    "pandya": EntityType.DYNASTY,
    "cheran": EntityType.DYNASTY,
    "pallava": EntityType.DYNASTY,
    "maurya": EntityType.DYNASTY,
    "mauryan": EntityType.DYNASTY,

    # PLACE
    "madurai": EntityType.PLACE,
    "vaigai": EntityType.PLACE,
    "sivaganga": EntityType.PLACE,
    "thanjavur": EntityType.PLACE,
    "greece": EntityType.PLACE,
    "greek": EntityType.PLACE,
    "rome": EntityType.PLACE,
    "roman": EntityType.PLACE,

    # CULTURAL_OBJECT
    "tamil-brahmi": EntityType.CULTURAL_OBJECT,
    "brahmi": EntityType.CULTURAL_OBJECT,
    "ring well": EntityType.CULTURAL_OBJECT,
    "brick structures": EntityType.CULTURAL_OBJECT,
    "carnelian": EntityType.CULTURAL_OBJECT,
    "agate": EntityType.CULTURAL_OBJECT,
    "quartz": EntityType.CULTURAL_OBJECT,
    "potsherds": EntityType.CULTURAL_OBJECT,
    "pottery": EntityType.CULTURAL_OBJECT,
    "graffiti": EntityType.CULTURAL_OBJECT,

    # HISTORICAL_PERIOD
    "sangam": EntityType.HISTORICAL_PERIOD,
    "sangam age": EntityType.HISTORICAL_PERIOD,
    "iron age": EntityType.HISTORICAL_PERIOD,
}

COMMON_WORD_TOKENS: Set[str] = {
    "tamil", "urban", "year", "old", "nadu", "state", "early", "literacy",
    "civilization", "excavation", "excavations", "site", "history", "ancient",
    "recent", "date", "findings", "evidence", "study", "research", "department",
    "beads", "iron", "drainage", "carbon dating", "ams", "accelerator mass spectrometry",
    "urbanization", "the", "this", "these", "those", "archaeologists", "archaeological"
}


class ClaimExtractor:
    """Extracts atomic, normalized, and classified candidate claims from research dossier."""

    def __init__(self):
        pass

    @staticmethod
    def classify_entity(entity_str: str) -> EntityType:
        """
        Differentiate meaningful named entities (PERSON, PLACE, ORG, SITE, etc.)
        from ordinary token fragments (COMMON_WORD).
        """
        clean = entity_str.strip().lower()
        if clean in COMMON_WORD_TOKENS:
            return EntityType.COMMON_WORD
        if clean in ENTITY_REGISTRY:
            return ENTITY_REGISTRY[clean]
        for k, v in ENTITY_REGISTRY.items():
            if k == clean or (len(k) > 3 and k in clean):
                return v
        return EntityType.COMMON_WORD

    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract all chronological date references from text."""
        dates = []
        for pattern in DATE_PATTERNS:
            for match in pattern.findall(text):
                clean = match.strip().lower()
                if clean not in dates:
                    dates.append(clean)
        return dates

    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """Extract explicit numerical quantities and counts from text."""
        numbers = []
        for pattern in NUMBER_PATTERNS:
            for match in pattern.findall(text):
                clean = match.strip().replace(",", "")
                if clean not in numbers:
                    numbers.append(clean)
        return numbers

    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """Extract known historical and archaeological entities and prominent proper nouns."""
        text_lower = text.lower()
        entities = []
        for kw in ENTITY_KEYWORDS:
            if kw in text_lower and kw not in entities:
                entities.append(kw)
        common_words = {
            "the", "this", "these", "those", "archaeologists", "archaeological", "excavations",
            "excavation", "historical", "evidence", "artifacts", "imagine", "welcome", "here",
            "there", "our", "their", "first", "ancient", "today", "yesterday", "tomorrow",
            "settlement", "tradition", "scientific", "cultural", "material", "recent", "analysis",
            "studies", "surveys", "findings", "scholars", "historians", "researchers", "according",
            "general", "king", "emperor", "monastery", "temple", "palace", "city", "town", "port",
            "one", "two", "many", "some", "such", "during", "under", "over", "after", "before",
            "while", "when", "where", "what", "how", "why", "who", "which", "into", "from", "with",
            "about", "that", "this", "they", "them", "then", "than", "more", "most", "also", "only",
            "been", "have", "were", "well", "will", "would", "could", "should", "shall", "must"
        }
        proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        for pn in proper_nouns:
            pnl = pn.lower()
            if pnl not in common_words and pnl not in entities:
                entities.append(pnl)
        return entities

    @classmethod
    def extract_meaningful_entities(cls, text: str) -> List[str]:
        """Extract ONLY meaningful entities, strictly excluding COMMON_WORD token fragments."""
        all_ents = cls.extract_entities(text)
        return [e for e in all_ents if cls.classify_entity(e) != EntityType.COMMON_WORD]

    @staticmethod
    def has_superlative(text: str) -> Tuple[bool, List[str]]:
        """Detect any superlative assertions requiring CRITICAL burden of proof."""
        tokens = re.findall(r"\b[a-zA-Z\-]+\b", text.lower())
        found = [t for t in tokens if t in SUPERLATIVE_TERMS]
        # Also check multi-word like "most ancient"
        if "most ancient" in text.lower():
            found.append("most ancient")
        return (len(found) > 0, list(set(found)))

    @classmethod
    def determine_claim_type(cls, text: str, dates: List[str], numbers: List[str], superlatives: List[str]) -> ClaimType:
        """Categorize the factual claim based on content analysis."""
        t_lower = text.lower()
        if superlatives:
            return ClaimType.SUPERLATIVE_ASSERTION
        if dates or any(w in t_lower for w in ["dating", "bce", "ce", "century", "chronology", "age", "period"]):
            return ClaimType.DATING_CHRONOLOGY
        if any(w in t_lower for w in ["excavation", "brick", "ring well", "pottery", "inscription", "brahmi", "bead", "artifact", "pottery"]):
            return ClaimType.ARCHAEOLOGICAL_EVIDENCE
        if any(w in t_lower for w in ["river", "district", "village", "site", "tamil nadu", "sivaganga", "bank", "location"]):
            return ClaimType.GEOGRAPHICAL_LOCATION
        if any(w in t_lower for w in ["king", "ruler", "emperor", "dynasty", "chola", "pandya", "archaeologist", "poet"]):
            return ClaimType.HISTORICAL_FIGURE
        if any(w in t_lower for w in ["trade", "literacy", "script", "urbanization", "civilization", "sangam age", "culture"]):
            return ClaimType.CULTURAL_PRACTICE
        if numbers:
            return ClaimType.STATISTICAL_MEASUREMENT
        return ClaimType.HISTORICAL_FACT

    @classmethod
    def determine_importance(cls, claim_type: ClaimType, has_superlative: bool, is_central: bool) -> ClaimImportance:
        """
        Assign claim importance:
        - Superlative assertions are strictly CRITICAL
        - Central dating or core civilizational claims are CRITICAL or HIGH
        - Archaeological findings are HIGH
        - Incidental details are MEDIUM or LOW
        """
        if has_superlative or claim_type == ClaimType.SUPERLATIVE_ASSERTION:
            return ClaimImportance.CRITICAL
        if claim_type == ClaimType.DATING_CHRONOLOGY and is_central:
            return ClaimImportance.CRITICAL
        if claim_type in (ClaimType.ARCHAEOLOGICAL_EVIDENCE, ClaimType.DATING_CHRONOLOGY, ClaimType.HISTORICAL_FIGURE):
            return ClaimImportance.HIGH
        if claim_type in (ClaimType.GEOGRAPHICAL_LOCATION, ClaimType.CULTURAL_PRACTICE, ClaimType.MATERIAL_CULTURE):
            return ClaimImportance.MEDIUM
        return ClaimImportance.LOW

    @staticmethod
    def normalize_statement(text: str) -> str:
        """Strip conversational padding, quotes, and normalize whitespace."""
        cleaned = text.strip()
        # Strip common quotation marks
        cleaned = re.sub(r'^["\'“]+|["\'”]+$', '', cleaned).strip()
        # Remove attribution prefixes like 'Documentation from Wikimedia indicates: '
        cleaned = re.sub(r"^(?:documentation\s+from\s+[^:]+indicates\s*:\s*)+", "", cleaned, flags=re.IGNORECASE)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def split_into_atomic_propositions(cls, statement: str) -> List[str]:
        """
        Split a compound statement into atomic propositions if multiple distinct facts are joined.
        Avoids breaking sentences that describe single integrated actions.
        """
        norm = cls.normalize_statement(statement)
        # Split on clear multi-clause connectors like '; ' or ' Additionally, '
        propositions = []
        semicolon_parts = [p.strip() for p in norm.split(";") if p.strip()]
        for p in semicolon_parts:
            # Further check if joined by strong coordinating clause transitions
            and_also_parts = re.split(r",\s*(?:and\s+furthermore|and\s+additionally|moreover)\s+", p, flags=re.IGNORECASE)
            for part in and_also_parts:
                cleaned = cls.normalize_statement(part)
                if len(cleaned.split()) >= 4:  # Minimum 4 words to be a meaningful proposition
                    propositions.append(cleaned)
        return propositions if propositions else [norm]

    def extract_claims_from_dossier(
        self,
        dossier: ResearchDossier,
        episode_id: str
    ) -> List[VerifiedClaim]:
        """
        Extract structured, atomic candidate claims from a Phase 4 research dossier.
        """
        claims: List[VerifiedClaim] = []
        seen_statements: Set[str] = set()
        claim_counter = 1

        # 1. Ingest preliminary findings
        for idx, finding in enumerate(dossier.preliminary_findings):
            if isinstance(finding, dict):
                raw_statement = finding.get("statement", "")
                topic_aspect = finding.get("topic_aspect") or dossier.topic
                supporting_chunk_ids = finding.get("supporting_chunk_ids", [])
                supporting_source_ids = finding.get("supporting_source_ids", [])
                independence_groups = finding.get("independence_groups", [])
                source_origins = finding.get("source_origins", [])
                evidence_weight_classes = finding.get("evidence_weight_classes", [])
            else:
                raw_statement = getattr(finding, "statement", "")
                topic_aspect = getattr(finding, "topic_aspect", dossier.topic)
                supporting_chunk_ids = getattr(finding, "supporting_chunk_ids", [])
                supporting_source_ids = getattr(finding, "supporting_source_ids", [])
                independence_groups = getattr(finding, "independence_groups", [])
                source_origins = getattr(finding, "source_origins", [])
                evidence_weight_classes = getattr(finding, "evidence_weight_classes", [])

            # Section 4: topics.json must never produce claims
            source_ids_lower = [str(sid).lower() for sid in supporting_source_ids]
            if any("topics.json" in sid or "local_15fef56c" in sid for sid in source_ids_lower):
                if all("topics" in sid or "local_15fef56c" in sid for sid in source_ids_lower):
                    continue

            atomic_props = self.split_into_atomic_propositions(raw_statement)

            for prop in atomic_props:
                # Section 6: Filter out raw JSON, code, or internal metadata syntax
                machine_markers = [
                    '"id":', '"badge":', '"title_tamil":', '"english_sub":', '"episodes":',
                    '{', '}', '[', ']', '\\"', '<', '>', 'def ', 'import ', 'class ',
                    '.json', '.py', '.txt', 'key:', 'value:', 'table:', 'db_', 'topics.json'
                ]
                if any(m in prop.lower() for m in machine_markers) or prop.count('"') >= 2 or '\\' in prop:
                    continue

                prop_key = re.sub(r"[^\w\s]", "", prop.lower())
                if prop_key in seen_statements or len(prop_key) < 10:
                    continue
                seen_statements.add(prop_key)

                dates = self.extract_dates(prop)
                numbers = self.extract_numbers(prop)
                entities = self.extract_entities(prop)
                has_sup, sup_terms = self.has_superlative(prop)

                claim_type = self.determine_claim_type(prop, dates, numbers, sup_terms)
                importance = self.determine_importance(claim_type, has_sup, is_central=(idx < 3))

                claim_id = f"clm_{claim_counter:03d}"
                claim_counter += 1

                vc = VerifiedClaim(
                    claim_id=claim_id,
                    statement=prop,
                    normalized_statement=self.normalize_statement(prop),
                    claim_type=claim_type,
                    importance=importance,
                    topic_aspect=topic_aspect,
                    episode_id=episode_id,
                    supporting_chunk_ids=list(supporting_chunk_ids),
                    supporting_source_ids=list(supporting_source_ids),
                    independence_groups=list(independence_groups),
                    source_origins=list(source_origins),
                    evidence_weight_classes=list(evidence_weight_classes),
                    extracted_entities=entities,
                    extracted_dates=dates,
                    extracted_numbers=numbers
                )
                claims.append(vc)

        # 2. Extract from explicit evidence conflicts (crucial for detecting debated claims)
        for conflict in dossier.conflicting_evidence:
            if isinstance(conflict, dict):
                stmt_a = conflict.get("statement_a") or conflict.get("claim_a", "")
                stmt_b = conflict.get("statement_b") or conflict.get("claim_b", "")
                src_a = conflict.get("sources_a") or [conflict.get("source_a_id", "")]
                src_b = conflict.get("sources_b") or [conflict.get("source_b_id", "")]
                aspect = conflict.get("aspect", dossier.topic)
            else:
                stmt_a = getattr(conflict, "statement_a", None) or getattr(conflict, "claim_a", "")
                stmt_b = getattr(conflict, "statement_b", None) or getattr(conflict, "claim_b", "")
                src_a = getattr(conflict, "sources_a", None) or [getattr(conflict, "source_a_id", "")]
                src_b = getattr(conflict, "sources_b", None) or [getattr(conflict, "source_b_id", "")]
                aspect = getattr(conflict, "aspect", dossier.topic)

            for side_text, src_ids in [(stmt_a, src_a), (stmt_b, src_b)]:
                if not side_text:
                    continue
                prop = self.normalize_statement(side_text)
                prop_key = re.sub(r"[^\w\s]", "", prop.lower())
                if prop_key in seen_statements or len(prop_key) < 10:
                    continue
                seen_statements.add(prop_key)

                dates = self.extract_dates(prop)
                numbers = self.extract_numbers(prop)
                entities = self.extract_entities(prop)
                has_sup, sup_terms = self.has_superlative(prop)
                claim_type = self.determine_claim_type(prop, dates, numbers, sup_terms)

                claim_id = f"clm_{claim_counter:03d}"
                claim_counter += 1

                vc = VerifiedClaim(
                    claim_id=claim_id,
                    statement=prop,
                    normalized_statement=prop,
                    claim_type=claim_type,
                    importance=ClaimImportance.HIGH,
                    topic_aspect=aspect or dossier.topic,
                    episode_id=episode_id,
                    supporting_source_ids=[s for s in src_ids if s],
                    extracted_entities=entities,
                    extracted_dates=dates,
                    extracted_numbers=numbers
                )
                claims.append(vc)

        logger.info(f"Extracted {len(claims)} atomic candidate claims from research dossier for episode '{episode_id}'")
        return claims

    def extract_claims(
        self,
        text: str,
        source_id: str = "",
        chunk_id: str = "",
        topic: str = "general",
        episode_id: str = "default_ep"
    ) -> List[VerifiedClaim]:
        """
        Extract atomic candidate claims directly from a chunk text or proposition string.
        """
        claims: List[VerifiedClaim] = []
        atomic_props = self.split_into_atomic_propositions(text)
        claim_counter = 1

        for prop in atomic_props:
            prop_clean = self.normalize_statement(prop)
            if len(prop_clean) < 10:
                continue

            dates = self.extract_dates(prop_clean)
            numbers = self.extract_numbers(prop_clean)
            entities = self.extract_entities(prop_clean)
            has_sup, sup_terms = self.has_superlative(prop_clean)

            claim_type = self.determine_claim_type(prop_clean, dates, numbers, sup_terms)
            importance = self.determine_importance(claim_type, has_sup, is_central=False)

            claim_id = f"clm_chunk_{claim_counter:03d}"
            claim_counter += 1

            vc = VerifiedClaim(
                claim_id=claim_id,
                statement=prop_clean,
                normalized_statement=prop_clean,
                claim_type=claim_type,
                importance=importance,
                topic_aspect=topic,
                episode_id=episode_id,
                supporting_chunk_ids=[chunk_id] if chunk_id else [],
                supporting_source_ids=[source_id] if source_id else [],
                extracted_entities=entities,
                extracted_dates=dates,
                extracted_numbers=numbers
            )
            claims.append(vc)

        return claims

