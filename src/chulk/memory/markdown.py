"""Markdown import/export helpers for memory."""

from __future__ import annotations

import re

from chulk.memory.retrieval import normalize_tags


def parse_markdown_memory_line(line: str) -> tuple[str, list[str]] | None:
    """Parse one exported Markdown memory bullet."""
    stripped = line.strip()
    if not stripped.startswith("- "):
        return None
    body = stripped[2:].strip()
    tags: list[str] = []
    tag_match = re.match(r"\[(?P<tags>[^\]]+)\]\s+(?P<content>.+)", body)
    if tag_match:
        tags = [tag.strip() for tag in tag_match.group("tags").split(",")]
        body = tag_match.group("content").strip()
    if not body:
        return None
    return body, normalize_tags(tags)
