"""
autonomous/research_processor.py - Clean Document Processing & Provenance-Preserving Chunking.

Transforms raw textual sources into clean, normalized passages, executing deterministic
sentence- and paragraph-aware chunking while attaching full source provenance to every chunk.
"""

import re
import hashlib
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

from autonomous.source_retriever import SourceMetadata

logger = logging.getLogger("autonomous.research_processor")


@dataclass
class EvidenceChunk:
    """A discrete unit of evidence extracted from a source, preserving complete provenance."""
    chunk_id: str
    episode_id: str
    source_id: str
    url: Optional[str]
    title: str
    publisher: Optional[str]
    retrieval_timestamp: str
    content_hash: str
    chunk_index: int
    text: str
    credibility_tier: int
    char_start: int = 0
    char_end: int = 0
    word_count: int = 0
    # Correction 6 & 7: Complete provenance fields
    domain: Optional[str] = None
    author: Optional[str] = None
    source_origin: str = "external"  # "external", "local", "cached_external"
    independence_group: str = ""
    source_role: str = "primary_evidence"
    evidence_weight_class: str = "PRIMARY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceChunk":
        return cls(
            chunk_id=data["chunk_id"],
            episode_id=data.get("episode_id", "unknown_episode"),
            source_id=data["source_id"],
            url=data.get("url"),
            title=data.get("title", "Untitled Source"),
            publisher=data.get("publisher"),
            retrieval_timestamp=data.get("retrieval_timestamp", ""),
            content_hash=data.get("content_hash", ""),
            chunk_index=data.get("chunk_index", 0),
            text=data.get("text", ""),
            credibility_tier=data.get("credibility_tier", 3),
            char_start=data.get("char_start", 0),
            char_end=data.get("char_end", 0),
            word_count=data.get("word_count", len(data.get("text", "").split())),
            domain=data.get("domain"),
            author=data.get("author"),
            source_origin=data.get("source_origin", "external"),
            independence_group=data.get("independence_group", data.get("domain") or ""),
            source_role=data.get("source_role", "primary_evidence"),
            evidence_weight_class=data.get("evidence_weight_class", "PRIMARY")
        )


class ResearchProcessor:
    """Cleans document text and generates deterministic chunks with provenance."""

    def __init__(self, chunk_size: int = 750, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text: str) -> str:
        """Normalize whitespace, remove non-printable characters, preserve paragraph breaks."""
        if not text:
            return ""
        # Normalize carriage returns
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse horizontal whitespace
        t = re.sub(r"[ \t]+", " ", t)
        # Collapse multiple newlines to double newline
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    def chunk_source(self, source: SourceMetadata, episode_id: str) -> List[EvidenceChunk]:
        """
        Segment source text into deterministic chunks preserving complete provenance.
        Respects sentence and paragraph boundaries wherever possible.
        """
        raw_text = source.cleaned_text or source.raw_content or ""
        cleaned = self.clean_text(raw_text)

        if not cleaned:
            return []

        chunks: List[EvidenceChunk] = []

        # Split text into paragraphs first
        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]

        current_chunk_text = ""
        current_start = 0
        chunk_idx = 0

        for para in paragraphs:
            # If paragraph fits comfortably in current chunk, append it
            if len(current_chunk_text) + len(para) + 2 <= self.chunk_size:
                if current_chunk_text:
                    current_chunk_text += "\n\n" + para
                else:
                    current_chunk_text = para
            else:
                # If current chunk has content, commit it
                if current_chunk_text:
                    chunk = self._create_chunk(
                        source=source,
                        episode_id=episode_id,
                        chunk_index=chunk_idx,
                        text=current_chunk_text,
                        char_start=current_start,
                        char_end=current_start + len(current_chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

                    # Compute overlap start
                    words = current_chunk_text.split()
                    overlap_words = words[-max(1, len(words) // 5):] if words else []
                    overlap_text = " ".join(overlap_words)
                    current_start += len(current_chunk_text) - len(overlap_text)
                    current_chunk_text = overlap_text + "\n\n" + para if overlap_text else para
                else:
                    # Paragraph itself exceeds chunk_size, split by sentences
                    sentence_chunks = self._split_by_sentences(para, self.chunk_size, self.overlap)
                    for s_text in sentence_chunks:
                        chunk = self._create_chunk(
                            source=source,
                            episode_id=episode_id,
                            chunk_index=chunk_idx,
                            text=s_text,
                            char_start=current_start,
                            char_end=current_start + len(s_text)
                        )
                        chunks.append(chunk)
                        chunk_idx += 1
                        current_start += len(s_text)
                    current_chunk_text = ""

        # Commit final chunk
        if current_chunk_text.strip():
            chunk = self._create_chunk(
                source=source,
                episode_id=episode_id,
                chunk_index=chunk_idx,
                text=current_chunk_text,
                char_start=current_start,
                char_end=current_start + len(current_chunk_text)
            )
            chunks.append(chunk)

        # Update source metadata chunk_count
        source.chunk_count = len(chunks)
        return chunks

    def _split_by_sentences(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Split a long paragraph into sentences respecting length boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = []
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 <= max_size:
                buf = (buf + " " + s).strip()
            else:
                if buf:
                    result.append(buf)
                    buf = s
                else:
                    # Single sentence exceeds max_size, break by words
                    words = s.split()
                    sub_buf = ""
                    for w in words:
                        if len(sub_buf) + len(w) + 1 <= max_size:
                            sub_buf = (sub_buf + " " + w).strip()
                        else:
                            if sub_buf:
                                result.append(sub_buf)
                            sub_buf = w
                    if sub_buf:
                        result.append(sub_buf)
        if buf:
            result.append(buf)
        return result

    def _create_chunk(
        self,
        source: SourceMetadata,
        episode_id: str,
        chunk_index: int,
        text: str,
        char_start: int,
        char_end: int
    ) -> EvidenceChunk:
        """Construct an EvidenceChunk with deterministic ID and complete provenance."""
        # Deterministic chunk ID derived from source_id and index
        chunk_id = f"{source.source_id}_chk_{chunk_index:03d}"
        clean_text = text.strip()
        word_count = len(clean_text.split())

        return EvidenceChunk(
            chunk_id=chunk_id,
            episode_id=episode_id,
            source_id=source.source_id,
            url=source.url,
            title=source.title,
            publisher=source.publisher,
            retrieval_timestamp=source.retrieval_timestamp,
            content_hash=source.content_hash,
            chunk_index=chunk_index,
            text=clean_text,
            credibility_tier=source.credibility_tier,
            char_start=char_start,
            char_end=char_end,
            word_count=word_count,
            domain=source.domain,
            author=source.author,
            source_origin=getattr(source, "source_origin", "external"),
            independence_group=getattr(source, "independence_group", "") or source.domain or "",
            source_role=getattr(source, "source_role", "primary_evidence"),
            evidence_weight_class=getattr(source, "evidence_weight_class", "PRIMARY")
        )
