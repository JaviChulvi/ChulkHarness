"""Explicit long-term memory extraction."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from chulk.memory.models import MemoryExtractionCandidate
from chulk.memory.retrieval import normalize_content

if TYPE_CHECKING:
    from chulk.memory.policy import MemoryPolicy, MemoryPolicyResult


def extract_memory_candidates(text: str) -> list[MemoryExtractionCandidate]:
    """Extract explicit memories from text without hidden inference."""
    clean_text = text.strip()
    if not clean_text:
        return []

    patterns = [
        (r"\bremember that (?P<content>.+)", ["explicit", "user"], None),
        (r"\bplease remember (?P<content>.+)", ["explicit", "user"], None),
        (
            r"\bmy preference is (?P<content>.+)",
            ["preference", "user"],
            "User prefers",
        ),
        (r"\bi prefer (?P<content>.+)", ["preference", "user"], "User prefers"),
        (r"\brecuerda que (?P<content>.+)", ["explicit", "user"], None),
        (
            r"\bpor favor,? recuerda(?: que)? (?P<content>.+)",
            ["explicit", "user"],
            None,
        ),
        (
            r"\bmi preferencia es (?P<content>.+)",
            ["preference", "user"],
            "El usuario prefiere",
        ),
        (
            r"\bprefiero (?P<content>.+)",
            ["preference", "user"],
            "El usuario prefiere",
        ),
    ]
    candidates: list[MemoryExtractionCandidate] = []
    seen_content: set[str] = set()
    for pattern, tags, preference_prefix in patterns:
        match = re.search(pattern, clean_text, flags=re.IGNORECASE)
        if not match:
            continue
        content = _strip_sentence(match.group("content"))
        if not content:
            continue
        if preference_prefix and not content.casefold().startswith(
            preference_prefix.casefold()
        ):
            content = f"{preference_prefix} {content}"
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


def route_memory_candidates(
    text: str,
    policy: "MemoryPolicy",
    *,
    conversation_id: str | None,
    turn_id: str | None,
) -> "MemoryPolicyResult":
    """Extract candidates and apply the configured persistence/review policy."""
    return policy.handle_candidates(
        extract_memory_candidates(text),
        conversation_id=conversation_id,
        turn_id=turn_id,
        evidence=text,
    )


def _strip_sentence(text: str) -> str:
    return text.strip().strip(".! ")
