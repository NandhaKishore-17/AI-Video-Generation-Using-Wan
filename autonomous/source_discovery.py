"""
autonomous/source_discovery.py - Multi-Provider Source Discovery Engine.

Discovers relevant sources across public open encyclopedic APIs (Wikimedia/Wikipedia),
curated archaeological department directories, local project archives, and existing
KnowledgeDocument SQLite records. Enforces publisher/domain independence tracking.
"""

import os
import re
import json
import logging
import urllib.parse
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from autonomous.source_retriever import SourceRetriever, SourceMetadata
from autonomous.research_planner import ResearchPlan

logger = logging.getLogger("autonomous.source_discovery")

try:
    import requests
except ImportError:
    requests = None


# Curated authoritative historical repository seeds (Tier 1 & Tier 2)
AUTHORITATIVE_SEEDS: Dict[str, List[Dict[str, str]]] = {
    "keezhadi": [
        {
            "title": "Excavations at Keeladi - Department of Archaeology, Government of Tamil Nadu",
            "url": "https://www.archaeology.tn.gov.in/keeladi",
            "publisher": "Tamil Nadu State Department of Archaeology",
            "tier": 1
        },
        {
            "title": "Keeladi Excavation Site - Archaeological Survey of India (ASI)",
            "url": "https://asi.nic.in/excavations-keeladi",
            "publisher": "Archaeological Survey of India",
            "tier": 1
        },
        {
            "title": "Keezhadi: An Urban Settlement of Sangam Age on the Banks of River Vaigai",
            "url": "https://frontline.thehindu.com/arts-and-culture/heritage/keeladi-excavations-sangam-era-tamil-nadu/article65874211.ece",
            "publisher": "The Hindu / Frontline",
            "tier": 3
        }
    ],
    "poompuhar": [
        {
            "title": "Marine Archaeological Investigations at Poompuhar - National Institute of Oceanography",
            "url": "https://www.nio.res.in/marine-archaeology-poompuhar",
            "publisher": "CSIR - National Institute of Oceanography",
            "tier": 1
        },
        {
            "title": "Underwater Exploration of Submerged Structures at Kaveripoompattinam",
            "url": "https://www.archaeology.tn.gov.in/poompuhar-marine",
            "publisher": "Tamil Nadu State Department of Archaeology",
            "tier": 1
        }
    ],
    "tanjore": [
        {
            "title": "Brihadisvara Temple, Thanjavur - UNESCO World Heritage Centre",
            "url": "https://whc.unesco.org/en/list/250/",
            "publisher": "UNESCO World Heritage Centre",
            "tier": 1
        },
        {
            "title": "The Great Living Chola Temples - Archaeological Survey of India",
            "url": "https://asi.nic.in/great-living-chola-temples",
            "publisher": "Archaeological Survey of India",
            "tier": 1
        }
    ],
    "chola": [
        {
            "title": "The Chola Navy and Southeast Asian Maritime Expeditions",
            "url": "https://www.jstor.org/stable/chola-maritime-expeditions",
            "publisher": "Journal of Southeast Asian Studies",
            "tier": 1
        }
    ],
    "kodumanal": [
        {
            "title": "Excavations at Kodumanal - Department of Archaeology, Government of Tamil Nadu",
            "url": "https://www.archaeology.tn.gov.in/kodumanal-excavations",
            "publisher": "Tamil Nadu State Department of Archaeology",
            "tier": 1
        },
        {
            "title": "Ancient Crucible Steel Metallurgy and Industrial Production at Kodumanal - ASI",
            "url": "https://asi.nic.in/excavations-kodumanal-metallurgy",
            "publisher": "Archaeological Survey of India",
            "tier": 1
        },
        {
            "title": "Wootz Steel Trade and Iron Age Industrial Complex in Kodumanal, Tamil Nadu",
            "url": "https://frontline.thehindu.com/arts-and-culture/heritage/kodumanal-crucible-steel-swords/article66128492.ece",
            "publisher": "The Hindu / Frontline",
            "tier": 3
        },
        {
            "title": "Wootz Steel: The Ancient High Carbon Steel of South India - Indian Institute of Science",
            "url": "https://www.iisc.ac.in/heritage/wootz-steel-damascus-blades",
            "publisher": "Indian Institute of Science (IISc)",
            "tier": 1
        }
    ],
    "wootz": [
        {
            "title": "Excavations at Kodumanal - Department of Archaeology, Government of Tamil Nadu",
            "url": "https://www.archaeology.tn.gov.in/kodumanal-excavations",
            "publisher": "Tamil Nadu State Department of Archaeology",
            "tier": 1
        },
        {
            "title": "Ancient Crucible Steel Metallurgy and Industrial Production at Kodumanal - ASI",
            "url": "https://asi.nic.in/excavations-kodumanal-metallurgy",
            "publisher": "Archaeological Survey of India",
            "tier": 1
        },
        {
            "title": "Wootz Steel: The Ancient High Carbon Steel of South India - Indian Institute of Science",
            "url": "https://www.iisc.ac.in/heritage/wootz-steel-damascus-blades",
            "publisher": "Indian Institute of Science (IISc)",
            "tier": 1
        }
    ],
    "crucible": [
        {
            "title": "Ancient Crucible Steel Metallurgy and Industrial Production at Kodumanal - ASI",
            "url": "https://asi.nic.in/excavations-kodumanal-metallurgy",
            "publisher": "Archaeological Survey of India",
            "tier": 1
        },
        {
            "title": "Excavations at Kodumanal - Department of Archaeology, Government of Tamil Nadu",
            "url": "https://www.archaeology.tn.gov.in/kodumanal-excavations",
            "publisher": "Tamil Nadu State Department of Archaeology",
            "tier": 1
        }
    ]
}


class SourceDiscovery:
    """Discovers source candidates from web, curated institutional seeds, and local documents."""

    def __init__(self, retriever: Optional[SourceRetriever] = None, offline: bool = False):
        self.retriever = retriever or SourceRetriever()
        self.offline = offline or os.getenv("RESEARCH_OFFLINE", "false").lower() in ("true", "1", "yes")

    def discover_sources(
        self,
        plan: ResearchPlan,
        max_sources: int = 8,
        existing_sources: Optional[List[SourceMetadata]] = None
    ) -> List[SourceMetadata]:
        """
        Discover and retrieve sources up to max_sources.
        Enforces domain/publisher diversity and bounded retrieval.
        """
        collected_sources: List[SourceMetadata] = []
        seen_urls: Set[str] = set()
        seen_hashes: Set[str] = set()
        seen_domains: Set[str] = set()

        for s in (existing_sources or []):
            if s.url and s.url in seen_urls:
                continue
            if s.content_hash and s.content_hash in seen_hashes:
                continue
            collected_sources.append(s)
            if s.url:
                seen_urls.add(s.url)
            if s.content_hash:
                seen_hashes.add(s.content_hash)
            if s.domain:
                seen_domains.add(s.domain)

        topic_lower = plan.topic.lower()

        # 1. Local Project Documents & Curated Fixtures (Always available offline)
        local_sources = self._discover_local_sources(plan)
        for src in local_sources:
            if src.url not in seen_urls and src.content_hash not in seen_hashes:
                collected_sources.append(src)
                seen_urls.add(src.url)
                seen_hashes.add(src.content_hash)
                if src.domain:
                    seen_domains.add(src.domain)
                if len(collected_sources) >= max_sources:
                    return collected_sources

        # If running in strict offline mode, stop here
        if self.offline:
            logger.info("SourceDiscovery running in OFFLINE mode. Returning local and cached sources.")
            return collected_sources

        # 2. Curated Authoritative Seeds matching topic keywords
        matched_seeds = []
        for kw, seeds in AUTHORITATIVE_SEEDS.items():
            if kw in topic_lower:
                matched_seeds.extend(seeds)

        for seed in matched_seeds:
            if seed["url"] not in seen_urls and len(collected_sources) < max_sources:
                logger.info(f"Retrieving authoritative institutional seed: {seed['title']}")
                meta = self.retriever.retrieve_url(seed["url"])
                if meta.extraction_status in ("EXTRACTED", "CACHED") and meta.cleaned_text:
                    if meta.content_hash not in seen_hashes:
                        meta.title = seed.get("title", meta.title)
                        meta.publisher = seed.get("publisher", meta.publisher)
                        if "tier" in seed:
                            meta.credibility_tier = seed["tier"]
                            s_role, e_weight = self.retriever.determine_evidence_weight(
                                meta.credibility_tier, meta.domain, meta.source_type, meta.source_origin
                            )
                            meta.source_role = s_role
                            meta.evidence_weight_class = e_weight
                        collected_sources.append(meta)
                        seen_urls.add(meta.url)
                        seen_hashes.add(meta.content_hash)
                        if meta.domain:
                            seen_domains.add(meta.domain)

        if len(collected_sources) >= max_sources:
            return collected_sources

        # 3. Wikimedia / Wikipedia Open Reference REST API (Zero-cost, structured)
        wiki_sources = self._discover_wikipedia_sources(plan)
        for src in wiki_sources:
            if src.url not in seen_urls and src.content_hash not in seen_hashes:
                # Limit Wikipedia to max 2 pages to enforce source diversity (Correction 4 & 5)
                wiki_count = sum(1 for s in collected_sources if s.source_type == "wikipedia")
                if wiki_count < 2:
                    collected_sources.append(src)
                    seen_urls.add(src.url)
                    seen_hashes.add(src.content_hash)
                    if src.domain:
                        seen_domains.add(src.domain)
                    if len(collected_sources) >= max_sources:
                        return collected_sources

        # 4. Public Web Search via DuckDuckGo / Open Search Endpoints (if requests available)
        if requests and len(collected_sources) < max_sources:
            search_urls = self._search_public_urls(plan)
            for url in search_urls:
                if url not in seen_urls:
                    domain = self.retriever.extract_domain(url)
                    # Limit to max 2 pages per domain to enforce publisher diversity (Correction 4)
                    domain_count = sum(1 for s in collected_sources if s.domain == domain)
                    if domain_count < 2:
                        meta = self.retriever.retrieve_url(url)
                        if meta.extraction_status in ("EXTRACTED", "CACHED") and meta.cleaned_text:
                            if meta.content_hash not in seen_hashes:
                                collected_sources.append(meta)
                                seen_urls.add(meta.url)
                                seen_hashes.add(meta.content_hash)
                                seen_domains.add(meta.domain)
                                if len(collected_sources) >= max_sources:
                                    break

        return collected_sources

    def _discover_local_sources(self, plan: ResearchPlan) -> List[SourceMetadata]:
        """Discover locally available files in backend/data/knowledge or daily_engine/."""
        results = []
        topic_lower = plan.topic.lower()

        search_dirs = [
            Path("data/knowledge"),
            Path("backend/data/knowledge"),
            Path("daily_engine"),
            Path("outputs/research_cache")
        ]

        for s_dir in search_dirs:
            if not s_dir.exists():
                continue
            for ext in ("*.txt", "*.md", "*.json"):
                for p in s_dir.glob(ext):
                    p_name = p.stem.lower()
                    # Check if file name or path matches any plan keywords
                    keywords = [w for w in re.split(r"\W+", topic_lower) if len(w) > 3]
                    if any(kw in p_name for kw in keywords) or "topic" in p_name or "knowledge" in p_name:
                        src = self.retriever.retrieve_local_document(str(p.resolve()))
                        if src.extraction_status == "EXTRACTED" and src.cleaned_text:
                            results.append(src)

        return results

    def _discover_wikipedia_sources(self, plan: ResearchPlan) -> List[SourceMetadata]:
        """Query open Wikipedia REST API for clean encyclopedic background (Tier 2)."""
        if not requests or self.offline:
            return []

        results = []
        # Candidate titles to check
        titles_to_try = [plan.topic]
        if plan.important_entities:
            titles_to_try.extend(plan.important_entities[:3])

        headers = {"User-Agent": self.retriever.user_agent}

        for raw_title in titles_to_try:
            clean_title = re.sub(r"[^a-zA-Z0-9 _-]", "", raw_title).strip().replace(" ", "_")
            if not clean_title:
                continue

            api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(clean_title)}"
            try:
                resp = requests.get(api_url, headers=headers, timeout=self.retriever.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    extract = data.get("extract", "")
                    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                    title = data.get("title", raw_title)

                    if extract and len(extract) > 100:
                        import hashlib
                        c_hash = hashlib.sha256(extract.encode("utf-8")).hexdigest()
                        src = SourceMetadata(
                            source_id=f"wiki_{c_hash[:12]}",
                            title=f"Wikipedia: {title}",
                            url=page_url or api_url,
                            publisher="Wikimedia Foundation",
                            domain="wikipedia.org",
                            source_type="wikipedia",
                            credibility_tier=2,
                            content_hash=c_hash,
                            extraction_status="EXTRACTED",
                            raw_content=extract[:500],
                            cleaned_text=extract,
                            source_origin="external",
                            independence_group="wikipedia.org",
                            source_role="discovery_reference",
                            evidence_weight_class="REFERENCE",
                            extra_metadata={
                                "description": data.get("description", ""),
                                "page_id": data.get("pageid")
                            }
                        )
                        results.append(src)
            except Exception as e:
                logger.debug(f"Wikipedia summary query failed for {raw_title}: {e}")

        return results

    def _search_public_urls(self, plan: ResearchPlan) -> List[str]:
        """Perform bounded public search queries returning candidate web URLs."""
        if not requests or self.offline:
            return []

        found_urls = []
        queries = plan.search_queries[:3] if plan.search_queries else [f"{plan.topic} archaeology"]

        for query in queries:
            try:
                # DuckDuckGo HTML Lite search endpoint (no API key needed, non-commercial educational use)
                ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                headers = {"User-Agent": self.retriever.user_agent}
                resp = requests.post(ddg_url, data={"q": query}, headers=headers, timeout=self.retriever.timeout)
                if resp.status_code == 200:
                    html_content = resp.text
                    # Extract result links (class uddg-urls or result__url)
                    raw_links = re.findall(r'href="//duckduckgo\.com/l/\?uddg=([^"&]+)', html_content)
                    for raw_link in raw_links:
                        decoded = urllib.parse.unquote(raw_link)
                        if decoded.startswith("http") and "duckduckgo" not in decoded:
                            # Skip social media in search results
                            domain = self.retriever.extract_domain(decoded)
                            if not any(sm in domain for sm in ("youtube.com", "facebook.com", "twitter.com", "instagram.com")):
                                found_urls.append(decoded)
                                if len(found_urls) >= 6:
                                    break
            except Exception as e:
                logger.debug(f"Public search query failed for '{query}': {e}")

        return found_urls
