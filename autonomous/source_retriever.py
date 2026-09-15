"""
autonomous/source_retriever.py - Bounded Source Retrieval & Credibility Classification.

Safely retrieves textual content from public web endpoints or local files,
normalizes URLs, computes SHA-256 content hashes, classifies credibility tiers (1-5),
and caches retrieved material on disk with full provenance.
"""

import os
import re
import ssl
import time
import uuid
import hashlib
import logging
import urllib.parse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

logger = logging.getLogger("autonomous.source_retriever")

try:
    import requests
except ImportError:
    requests = None


@dataclass
class SourceMetadata:
    """Provenance and credibility metadata for a single research source."""
    source_id: str
    title: str
    url: Optional[str] = None
    publisher: Optional[str] = None
    domain: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[str] = None
    retrieval_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_type: str = "web"  # "web", "wikipedia", "local_doc", "academic_archive"
    credibility_tier: int = 3  # Tier 1 (Institutional/Gov/Peer-reviewed) down to Tier 5 (Unreliable)
    content_hash: str = ""
    language: str = "en"
    extraction_status: str = "EXTRACTED"  # "EXTRACTED", "CACHED", "FAILED", "TRUNCATED"
    chunk_count: int = 0
    raw_content: Optional[str] = None
    cleaned_text: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    # Correction 1 & 2: Source origin & conservative independence grouping
    source_origin: str = "external"  # "external", "local", "cached_external"
    independence_group: str = ""
    # Correction 3: Explicit source role & evidence weight class
    source_role: str = "primary_evidence"  # "primary_evidence", "secondary_evidence", "discovery_reference", "background_local", "unverified_media"
    evidence_weight_class: str = "PRIMARY"  # "PRIMARY", "HIGH", "REFERENCE", "MODERATE", "LOW", "VERY_LOW"
    # Correction 8: Disk cache path reference to avoid retaining large raw strings in memory
    raw_cache_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Avoid huge dumps of raw_content in small summaries
        if d.get("raw_content") and len(d["raw_content"]) > 300:
            d["raw_content"] = d["raw_content"][:300] + "... [TRUNCATED - See raw_cache_path]"
        if d.get("cleaned_text") and len(d["cleaned_text"]) > 600:
            d["cleaned_text"] = d["cleaned_text"][:600] + "... [TRUNCATED - Canonical text in evidence.json chunks]"
        return d


class SimpleHTMLTextExtractor(HTMLParser):
    """Standard library HTML to clean text extractor that strips scripts, styles, and boilerplate."""

    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self.title_parts: List[str] = []
        self.in_script = False
        self.in_style = False
        self.in_nav = False
        self.in_title = False
        self.in_header = False
        self.in_footer = False

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t in ("script", "noscript"):
            self.in_script = True
        elif t == "style":
            self.in_style = True
        elif t in ("nav", "aside"):
            self.in_nav = True
        elif t == "header":
            self.in_header = True
        elif t == "footer":
            self.in_footer = True
        elif t == "title":
            self.in_title = True
        elif t in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "article", "section"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ("script", "noscript"):
            self.in_script = False
        elif t == "style":
            self.in_style = False
        elif t in ("nav", "aside"):
            self.in_nav = False
        elif t == "header":
            self.in_header = False
        elif t == "footer":
            self.in_footer = False
        elif t == "title":
            self.in_title = False
        elif t in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "article", "section"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data.strip())
        elif not (self.in_script or self.in_style or self.in_nav or self.in_footer):
            clean_data = data.strip()
            if clean_data:
                self.text_parts.append(clean_data + " ")

    def get_text(self) -> str:
        raw = "".join(self.text_parts)
        # Collapse multiple empty lines and excessive whitespace
        collapsed = re.sub(r"[ \t]+", " ", raw)
        lines = [line.strip() for line in collapsed.split("\n")]
        return "\n".join([line for line in lines if line])

    def get_title(self) -> str:
        return " ".join([p for p in self.title_parts if p]).strip()


class SourceRetriever:
    """Safely retrieves public web and local documents with quality tiers and provenance."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        timeout: int = 10,
        max_bytes: int = 2 * 1024 * 1024,  # 2MB
        max_retries: int = 2,
        user_agent: str = "AutonomousHistoryDocEngine/1.0 (Educational Archaeological Research)"
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("outputs/research_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_retries = max_retries
        self.user_agent = user_agent

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL for consistent deduplication."""
        if not url:
            return ""
        parsed = urllib.parse.urlparse(url.strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        # Remove standard ports
        if netloc.endswith(":80"):
            netloc = netloc[:-3]
        elif netloc.endswith(":443"):
            netloc = netloc[:-4]

        # Strip tracking query parameters
        query_pairs = urllib.parse.parse_qsl(parsed.query)
        clean_pairs = [
            (k, v) for k, v in query_pairs
            if not k.lower().startswith("utm_") and k.lower() not in ("ref", "fbclid", "gclid", "source")
        ]
        clean_query = urllib.parse.urlencode(clean_pairs)

        # Normalize path
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # Drop fragment
        return urllib.parse.urlunparse((scheme, netloc, path, "", clean_query, ""))

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract root hostname/domain from URL."""
        if not url:
            return "local_source"
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown_domain"

    @classmethod
    def classify_credibility_tier(cls, url: Optional[str], publisher: Optional[str] = None, source_type: str = "web") -> Tuple[int, str]:
        """
        Classify source credibility tier (1 to 5).
        NOTE: Credibility tier is metadata, NOT proof of truth.
        """
        if not url:
            return (2 if source_type == "local_doc" else 3, "Unspecified Local Document")

        url_lower = url.lower()
        domain = cls.extract_domain(url_lower)
        pub_lower = (publisher or "").lower()

        # Tier 1: Government, Archaeological Departments, Peer-Reviewed Universities & Journals
        tier_1_indicators = [
            ".gov", ".gov.in", ".nic.in", "asi.nic.in", "archaeology.tn.gov.in",
            ".edu", ".ac.in", ".ac.uk", "jstor.org", "springer.com", "nature.com",
            "sciencedirect.com", "cambridge.org", "oxfordjournals.org", "persee.fr",
            "archaeology.org", "epigraphia"
        ]
        for ind in tier_1_indicators:
            if ind in domain or ind in url_lower:
                return (1, "Tier 1: Government / Archaeological Institute / Peer-reviewed Research")

        # Tier 2: Reputable Reference & Curated Institutional Repositories
        tier_2_indicators = [
            "britannica.com", "unesco.org", "britishmuseum.org", "metmuseum.org",
            "nationalarchives.gov.uk", "loc.gov", "si.edu", "archive.org",
            "wikipedia.org", "wikimedia.org"
        ]
        for ind in tier_2_indicators:
            if ind in domain:
                if "wikipedia.org" in domain or "wikimedia.org" in domain:
                    return (2, "Tier 2: Public Collaborative Reference (Verify with citations)")
                return (2, "Tier 2: Established Reference / Museum / Cultural Archive")

        # Tier 3: Reputable Journalism & Specialist History Magazines
        tier_3_indicators = [
            "thehindu.com", "frontline.thehindu.com", "nationalgeographic.com",
            "bbc.com", "bbc.co.uk", "smithsonianmag.com", "history.com",
            "theconversation.com", "scroll.in", "thewire.in", "indianexpress.com",
            "timesofindia.indiatimes.com"
        ]
        for ind in tier_3_indicators:
            if ind in domain:
                return (3, "Tier 3: Established Journalism / Reputable Media")

        # Tier 5: Social Media / Video / Unverified Forums
        tier_5_indicators = [
            "youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com",
            "instagram.com", "reddit.com", "quora.com", "pinterest.com",
            "tiktok.com", "medium.com"
        ]
        for ind in tier_5_indicators:
            if ind in domain:
                return (5, "Tier 5: Social Media / User Forum (Low reliability)")

        # Tier 4: General web sites, commercial history blogs, travelogues
        return (4, "Tier 4: General Web / Commercial History Publication")

    @classmethod
    def determine_evidence_weight(
        cls,
        tier: int,
        domain: Optional[str] = None,
        source_type: str = "web",
        source_origin: str = "external"
    ) -> Tuple[str, str]:
        """
        Deterministic mapping of source into (source_role, evidence_weight_class).
        Correction 3:
          Tier 1: PRIMARY / "primary_evidence"
          Tier 2 (Institutional/Museum/Academic): HIGH / "secondary_evidence"
          Wikimedia / Wikipedia: REFERENCE / "discovery_reference"
          Tier 3 (Reputable Journalism): MODERATE / "secondary_evidence"
          Tier 4: LOW / "unverified_media"
          Tier 5: VERY_LOW / "unverified_media"
          Local Project Knowledge: REFERENCE / "background_local"
        """
        dom_lower = (domain or "").lower()
        if source_origin == "local":
            return ("background_local", "REFERENCE")
        if "wikipedia.org" in dom_lower or "wikimedia.org" in dom_lower or source_type == "wikipedia":
            return ("discovery_reference", "REFERENCE")
        if tier == 1:
            return ("primary_evidence", "PRIMARY")
        elif tier == 2:
            return ("secondary_evidence", "HIGH")
        elif tier == 3:
            return ("secondary_evidence", "MODERATE")
        elif tier == 4:
            return ("unverified_media", "LOW")
        else:
            return ("unverified_media", "VERY_LOW")

    @classmethod
    def determine_independence_group(
        cls,
        url: Optional[str],
        domain: Optional[str],
        publisher: Optional[str],
        source_origin: str = "external"
    ) -> str:
        """
        Correction 1 & 2: Determine conservative independence grouping.
        Local filesystem documents -> 'local_filesystem'
        Cached external documents -> original external domain/publisher
        Multiple URLs on same domain -> same domain group (e.g. 'asi.nic.in')
        """
        if source_origin == "local":
            return "local_filesystem"
        target = (domain or "").lower().strip() or (cls.extract_domain(url) if url else "")
        if "wikipedia.org" in target:
            return "wikipedia.org"
        if "wikimedia.org" in target:
            return "wikimedia.org"
        if domain:
            dom = domain.lower().strip()
            if dom.startswith("www."):
                dom = dom[4:]
            if dom and dom not in ("local_filesystem", "local_source", "unknown_domain"):
                return dom
        if url:
            d = cls.extract_domain(url)
            if d and d not in ("local_filesystem", "local_source", "unknown_domain"):
                return d
        if publisher:
            pub = publisher.lower().strip()
            if pub and pub != "local project knowledge":
                return pub
        return "unknown_group"

    def retrieve_url(self, url: str, source_id: Optional[str] = None, episode_id: Optional[str] = None) -> SourceMetadata:
        """Safely retrieve a public URL with timeout, bounds, and caching."""
        norm_url = self.normalize_url(url)
        domain = self.extract_domain(norm_url)
        sid = source_id or f"src_{hashlib.md5(norm_url.encode()).hexdigest()[:12]}"
        tier, tier_desc = self.classify_credibility_tier(norm_url)

        # Check local cache first
        url_hash = hashlib.sha256(norm_url.encode("utf-8")).hexdigest()
        cached_file = self.cache_dir / f"{url_hash}.txt"
        if cached_file.exists():
            try:
                content = cached_file.read_text(encoding="utf-8")
                lines = content.split("\n", 2)
                title = lines[0].replace("TITLE: ", "").strip() if len(lines) > 0 else "Cached Source"
                cleaned_text = lines[2] if len(lines) > 2 else content
                c_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
                s_type = "wikipedia" if ("wikipedia.org" in domain or "wikimedia.org" in domain) else "web"
                s_role, e_weight = self.determine_evidence_weight(tier, domain, s_type, source_origin="cached_external")
                ind_group = self.determine_independence_group(norm_url, domain, domain, source_origin="cached_external")
                return SourceMetadata(
                    source_id=sid,
                    title=title,
                    url=norm_url,
                    publisher=domain,
                    domain=domain,
                    retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                    source_type=s_type,
                    credibility_tier=tier,
                    content_hash=c_hash,
                    extraction_status="CACHED",
                    raw_content=cleaned_text[:500],
                    cleaned_text=cleaned_text,
                    source_origin="cached_external",
                    independence_group=ind_group,
                    source_role=s_role,
                    evidence_weight_class=e_weight,
                    raw_cache_path=str(cached_file.resolve()),
                    extra_metadata={"tier_description": tier_desc, "cached": True}
                )
            except Exception as e:
                logger.warning(f"Cache read failed for {norm_url}: {e}")

        # Real HTTP Request
        if not requests:
            return SourceMetadata(
                source_id=sid,
                title="Requests Library Missing",
                url=norm_url,
                extraction_status="FAILED",
                extra_metadata={"error": "requests module not installed"}
            )

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-US,en;q=0.9,ta;q=0.8",
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(norm_url, headers=headers, timeout=self.timeout, stream=True)
                resp.raise_for_status()

                # Content-Type check
                ctype = resp.headers.get("Content-Type", "").lower()
                if not any(t in ctype for t in ("text/html", "text/plain", "application/json", "xml")):
                    return SourceMetadata(
                        source_id=sid,
                        title=f"Non-text content ({ctype})",
                        url=norm_url,
                        extraction_status="FAILED",
                        extra_metadata={"error": f"Unsupported content-type: {ctype}"}
                    )

                # Bounded read
                content_bytes = bytearray()
                for chunk in resp.iter_content(chunk_size=8192):
                    content_bytes.extend(chunk)
                    if len(content_bytes) > self.max_bytes:
                        logger.warning(f"Source exceeded max document size ({self.max_bytes} bytes), truncating: {norm_url}")
                        break

                encoding = resp.encoding or "utf-8"
                raw_html = content_bytes.decode(encoding, errors="ignore")

                # Parse and clean
                parser = SimpleHTMLTextExtractor()
                parser.feed(raw_html)
                cleaned_text = parser.get_text()
                title = parser.get_title() or f"Source: {domain}"

                if not cleaned_text.strip():
                    cleaned_text = raw_html[:2000]

                c_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()

                # Save to cache
                try:
                    cached_file.write_text(f"TITLE: {title}\nURL: {norm_url}\n{cleaned_text}", encoding="utf-8")
                except Exception as ce:
                    logger.debug(f"Failed to cache {norm_url}: {ce}")

                s_type = "wikipedia" if ("wikipedia.org" in domain or "wikimedia.org" in domain) else "web"
                s_role, e_weight = self.determine_evidence_weight(tier, domain, s_type, source_origin="external")
                ind_group = self.determine_independence_group(norm_url, domain, domain, source_origin="external")

                return SourceMetadata(
                    source_id=sid,
                    title=title,
                    url=norm_url,
                    publisher=domain,
                    domain=domain,
                    retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                    source_type=s_type,
                    credibility_tier=tier,
                    content_hash=c_hash,
                    extraction_status="EXTRACTED",
                    raw_content=raw_html[:500],
                    cleaned_text=cleaned_text,
                    source_origin="external",
                    independence_group=ind_group,
                    source_role=s_role,
                    evidence_weight_class=e_weight,
                    raw_cache_path=str(cached_file.resolve()),
                    extra_metadata={"tier_description": tier_desc, "attempts": attempt}
                )

            except Exception as e:
                last_error = e
                time.sleep(1.0 * attempt)

        return SourceMetadata(
            source_id=sid,
            title=f"Failed to retrieve: {domain}",
            url=norm_url,
            domain=domain,
            extraction_status="FAILED",
            source_origin="external",
            independence_group=self.determine_independence_group(norm_url, domain, domain, source_origin="external"),
            extra_metadata={"error": str(last_error)}
        )

    def retrieve_local_document(self, file_path: str, source_id: Optional[str] = None, title: Optional[str] = None) -> SourceMetadata:
        """Ingest a local document (txt, md) with provenance."""
        path = Path(file_path)
        if not path.exists():
            return SourceMetadata(
                source_id=source_id or f"local_{uuid.uuid4().hex[:8]}",
                title="File Not Found",
                extraction_status="FAILED",
                source_origin="local",
                independence_group="local_filesystem",
                source_role="background_local",
                evidence_weight_class="REFERENCE",
                extra_metadata={"error": f"Path does not exist: {file_path}"}
            )

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            doc_title = title or path.stem.replace("_", " ").title()
            c_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            sid = source_id or f"local_{c_hash[:12]}"
            s_role, e_weight = self.determine_evidence_weight(2, "local_filesystem", "local_doc", source_origin="local")
            ind_group = self.determine_independence_group(str(path.resolve()), "local_filesystem", "Local Project Knowledge", source_origin="local")

            return SourceMetadata(
                source_id=sid,
                title=doc_title,
                url=str(path.resolve()),
                publisher="Local Project Knowledge",
                domain="local_filesystem",
                retrieval_timestamp=datetime.now(timezone.utc).isoformat(),
                source_type="local_doc",
                credibility_tier=2,
                content_hash=c_hash,
                extraction_status="EXTRACTED",
                raw_content=content[:500],
                cleaned_text=content,
                source_origin="local",
                independence_group=ind_group,
                source_role=s_role,
                evidence_weight_class=e_weight,
                raw_cache_path=str(path.resolve()),
                extra_metadata={"file_size": path.stat().st_size}
            )
        except Exception as e:
            return SourceMetadata(
                source_id=source_id or f"local_{uuid.uuid4().hex[:8]}",
                title="Read Error",
                extraction_status="FAILED",
                source_origin="local",
                independence_group="local_filesystem",
                source_role="background_local",
                evidence_weight_class="REFERENCE",
                extra_metadata={"error": str(e)}
            )
