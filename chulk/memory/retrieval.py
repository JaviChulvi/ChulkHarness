"""Memory normalization, lexical retrieval, and local vector helpers."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from chulk.memory.constants import DEFAULT_EMBEDDING_DIMENSIONS, PROFILE_MEMORY_TAGS, SEARCH_STOPWORDS
from chulk.memory.models import MemoryRecord


def normalize_tags(tags: list[str]) -> list[str]:
    """Normalize and deduplicate memory tags."""
    clean_tags = []
    for tag in tags:
        clean = str(tag).strip().lower()
        if clean and clean not in clean_tags:
            clean_tags.append(clean)
    return clean_tags


def merge_tags(left: list[str], right: list[str]) -> list[str]:
    """Merge two tag lists using normal tag rules."""
    return normalize_tags([*left, *right])


def normalize_importance(importance: int) -> int:
    """Validate memory importance."""
    if not isinstance(importance, int) or isinstance(importance, bool):
        raise ValueError("Memory importance must be an integer")
    if importance < 1 or importance > 10:
        raise ValueError("Memory importance must be between 1 and 10")
    return importance


def normalize_limit(limit: int) -> int:
    """Validate and cap memory result limits."""
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("Memory limit must be an integer")
    if limit < 1:
        raise ValueError("Memory limit must be greater than zero")
    return min(limit, 100)


def normalize_source(source: str) -> str:
    """Normalize memory source labels."""
    clean_source = str(source).strip().lower().replace(" ", "_")
    return clean_source or "manual"


def normalize_confidence(confidence: float) -> float:
    """Validate confidence values."""
    if not isinstance(confidence, int | float) or isinstance(confidence, bool):
        raise ValueError("Memory confidence must be a number")
    if confidence < 0 or confidence > 1:
        raise ValueError("Memory confidence must be between 0 and 1")
    return float(confidence)


def normalize_embedding(embedding: list[float] | None) -> list[float] | None:
    """Validate optional embedding vectors."""
    if embedding is None:
        return None
    if not isinstance(embedding, list):
        raise ValueError("Memory embedding must be a list of numbers")
    clean_embedding: list[float] = []
    for value in embedding:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError("Memory embedding values must be numbers")
        clean_embedding.append(float(value))
    return clean_embedding


def safe_json_list(value: str) -> list[str]:
    """Return a JSON list or an empty list."""
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def safe_json_float_list(value: str | None) -> list[float] | None:
    """Return a JSON float list or None."""
    if not value:
        return None
    parsed = safe_json_list(value)
    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError):
        return None


def safe_json_dict(value: str) -> dict[str, Any]:
    """Return a JSON object or an empty dict."""
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight local retrieval."""
    terms = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [term for term in terms if len(term) > 2 and term not in SEARCH_STOPWORDS]


def score_memory(memory: MemoryRecord, terms: list[str]) -> int:
    """Score a memory for lexical matches."""
    haystack = " ".join(
        [
            memory.content,
            " ".join(memory.tags),
            json.dumps(memory.metadata, sort_keys=True),
            memory.source,
        ]
    ).lower()
    term_score = sum(haystack.count(term) for term in terms)
    return term_score + memory.importance


def text_to_embedding(text: str, dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS) -> list[float]:
    """Create a small deterministic lexical embedding for local vector search."""
    vector = [0.0] * dimensions
    terms = tokenize(text)
    if not terms:
        return vector
    for term in terms:
        index = sum(ord(char) for char in term) % dimensions
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two equal-length vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def merge_ranked_memories(left: list[MemoryRecord], right: list[MemoryRecord]) -> list[MemoryRecord]:
    """Merge ranked memory lists without duplicate ids."""
    merged: dict[str, MemoryRecord] = {}
    for memory in [*left, *right]:
        merged[memory.id] = memory
    return list(merged.values())


def normalize_content(content: str) -> str:
    """Normalize content for duplicate comparison."""
    return " ".join(tokenize(content))


def content_similarity(left: str, right: str) -> float:
    """Return Jaccard similarity between token sets."""
    left_terms = set(tokenize(left))
    right_terms = set(tokenize(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def choose_memory_to_keep(left: MemoryRecord, right: MemoryRecord) -> tuple[MemoryRecord, MemoryRecord]:
    """Choose the stronger memory and the weaker duplicate."""
    left_score = (left.confidence, left.importance, left.updated_at)
    right_score = (right.confidence, right.importance, right.updated_at)
    return (left, right) if left_score >= right_score else (right, left)


def resolve_profile_conflicts(memories: list[MemoryRecord]) -> list[MemoryRecord]:
    """Keep the strongest non-duplicate profile memories."""
    ordered = sorted(memories, key=lambda item: (item.confidence, item.importance, item.updated_at), reverse=True)
    selected: list[MemoryRecord] = []
    for memory in ordered:
        is_near_duplicate = any(
            set(memory.tags) & set(existing.tags) & PROFILE_MEMORY_TAGS
            and content_similarity(memory.content, existing.content) >= 0.86
            for existing in selected
        )
        if not is_near_duplicate:
            selected.append(memory)
    return selected
