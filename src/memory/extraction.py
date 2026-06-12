"""Explicit long-term memory extraction."""

from __future__ import annotations

import re

from src.memory.models import MemoryExtractionCandidate
from src.memory.retrieval import normalize_content


def extract_memory_candidates(text: str) -> list[MemoryExtractionCandidate]:
    """Extract explicit memories from text without hidden inference."""
    clean_text = text.strip()
    if not clean_text:
        return []

    patterns = [
        (r"\bremember that (?P<content>.+)", ["explicit", "user"]),
        (r"\bplease remember (?P<content>.+)", ["explicit", "user"]),
        (r"\bmy preference is (?P<content>.+)", ["preference", "user"]),
        (r"\bi prefer (?P<content>.+)", ["preference", "user"]),
    ]
    candidates: list[MemoryExtractionCandidate] = []
    seen_content: set[str] = set()
    for pattern, tags in patterns:
        match = re.search(pattern, clean_text, flags=re.IGNORECASE)
        if not match:
            continue
        content = _strip_sentence(match.group("content"))
        if not content:
            continue
        if "preference" in tags and not content.lower().startswith("user prefers"):
            content = f"User prefers {content}"
        normalized = normalize_content(content)
        if normalized in seen_content:
            continue
        seen_content.add(normalized)
        candidates.append(
            MemoryExtractionCandidate(
                content=content,
                tags=tags,
                metadata={"extracted_from": "user_message"},
                importance=7 if "preference" in tags else 5,
                confidence=0.9,
            )
        )
    return candidates


def _strip_sentence(text: str) -> str:
    return text.strip().strip(".! ")
