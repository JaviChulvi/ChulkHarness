"""Long-term memory tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.memory import SQLiteMemoryStore
from src.tools.registry import Tool, ToolResult


def save_memory_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="save_memory",
        description="Save a durable long-term memory about the user, project, preference, or prior work.",
        args_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Memory content to store.", "minLength": 1},
                "tags": {
                    "type": "array",
                    "description": "Optional tags such as persona, preference, project, task.",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 20,
                },
                "metadata": {"type": "object", "description": "Optional structured metadata."},
                "importance": {"type": "integer", "description": "Importance from 1 to 10.", "minimum": 1, "maximum": 10},
                "source": {
                    "type": "string",
                    "description": "Memory source, for example manual or user_explicit.",
                    "minLength": 1,
                },
                "confidence": {"type": "number", "description": "Confidence from 0 to 1.", "minimum": 0, "maximum": 1},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        callable=lambda arguments: save_memory(arguments, memory_store),
    )


def search_memory_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="search_memory",
        description="Search durable long-term memories using simple keyword matching.",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query.", "minLength": 1},
                "limit": {"type": "integer", "description": "Maximum memories to return.", "minimum": 1, "maximum": 100},
                "include_archived": {"type": "boolean", "description": "Whether archived memories should be searched."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        callable=lambda arguments: search_memory(arguments, memory_store),
    )


def list_memories_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="list_memories",
        description="List recent durable long-term memories.",
        args_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum memories to return.", "minimum": 1, "maximum": 100},
                "include_archived": {"type": "boolean", "description": "Whether archived memories should be listed."},
            },
            "required": [],
            "additionalProperties": False,
        },
        callable=lambda arguments: list_memories(arguments, memory_store),
    )


def delete_memory_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="delete_memory",
        description="Delete a durable long-term memory by id.",
        args_schema={
            "type": "object",
            "properties": {"memory_id": {"type": "string", "description": "Memory id to delete.", "minLength": 1}},
            "required": ["memory_id"],
            "additionalProperties": False,
        },
        callable=lambda arguments: delete_memory(arguments, memory_store),
    )


def update_memory_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="update_memory",
        description="Update an existing durable long-term memory by id.",
        args_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory id to update.", "minLength": 1},
                "content": {"type": "string", "description": "Replacement memory content.", "minLength": 1},
                "tags": {
                    "type": "array",
                    "description": "Replacement tags.",
                    "items": {"type": "string", "minLength": 1},
                    "maxItems": 20,
                },
                "metadata": {"type": "object", "description": "Replacement metadata."},
                "importance": {"type": "integer", "description": "Importance from 1 to 10.", "minimum": 1, "maximum": 10},
                "source": {"type": "string", "description": "Replacement source.", "minLength": 1},
                "confidence": {"type": "number", "description": "Confidence from 0 to 1.", "minimum": 0, "maximum": 1},
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        },
        callable=lambda arguments: update_memory(arguments, memory_store),
    )


def summarize_memories_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="summarize_memories",
        description="Return a compact summary of durable long-term memories.",
        args_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional search query before summarizing.", "minLength": 1},
                "limit": {"type": "integer", "description": "Maximum memories to summarize.", "minimum": 1, "maximum": 100},
            },
            "required": [],
            "additionalProperties": False,
        },
        callable=lambda arguments: summarize_memories(arguments, memory_store),
    )


def archive_memory_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="archive_memory",
        description="Archive a durable long-term memory without deleting it.",
        args_schema={
            "type": "object",
            "properties": {"memory_id": {"type": "string", "description": "Memory id to archive.", "minLength": 1}},
            "required": ["memory_id"],
            "additionalProperties": False,
        },
        callable=lambda arguments: archive_memory(arguments, memory_store),
    )


def restore_memory_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="restore_memory",
        description="Restore an archived durable long-term memory.",
        args_schema={
            "type": "object",
            "properties": {"memory_id": {"type": "string", "description": "Memory id to restore.", "minLength": 1}},
            "required": ["memory_id"],
            "additionalProperties": False,
        },
        callable=lambda arguments: restore_memory(arguments, memory_store),
    )


def compact_memories_tool(memory_store: SQLiteMemoryStore) -> Tool:
    return Tool(
        name="compact_memories",
        description="Archive near-duplicate active memories, keeping the strongest record.",
        args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        callable=lambda arguments: compact_memories(arguments, memory_store),
    )


def import_memories_tool(memory_store: SQLiteMemoryStore, project_root: Path) -> Tool:
    return Tool(
        name="import_memories",
        description="Import simple bullet memories from a Markdown file inside the project directory.",
        args_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Markdown path relative to the project root.",
                    "minLength": 1,
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        callable=lambda arguments: import_memories(arguments, memory_store, project_root),
    )


def export_memories_tool(memory_store: SQLiteMemoryStore, project_root: Path) -> Tool:
    return Tool(
        name="export_memories",
        description="Export active memories to a Markdown file inside the project directory.",
        args_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Markdown path relative to the project root.",
                    "minLength": 1,
                },
                "include_archived": {"type": "boolean", "description": "Whether archived memories should be exported."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        callable=lambda arguments: export_memories(arguments, memory_store, project_root),
    )


def save_memory(arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    memory_id = memory_store.save_memory(
        arguments["content"],
        tags=arguments.get("tags") or [],
        metadata=arguments.get("metadata") or {},
        importance=arguments.get("importance", 1),
        source=arguments.get("source", "manual"),
        confidence=arguments.get("confidence", 1.0),
    )
    return ToolResult("save_memory", True, f"Saved memory {memory_id}.", metadata={"memory_id": memory_id})


def search_memory(arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    memories = memory_store.search_memory(
        arguments["query"],
        limit=arguments.get("limit", 5),
        include_archived=arguments.get("include_archived", False),
    )
    return ToolResult("search_memory", True, _format_memories(memories), metadata={"count": len(memories)})


def list_memories(arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    memories = memory_store.list_memories(
        limit=arguments.get("limit", 50),
        include_archived=arguments.get("include_archived", False),
    )
    return ToolResult("list_memories", True, _format_memories(memories), metadata={"count": len(memories)})


def delete_memory(arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    memory_id = arguments["memory_id"]
    deleted = memory_store.delete_memory(memory_id)
    if not deleted:
        return ToolResult("delete_memory", False, f"Memory not found: {memory_id}", error="not_found")
    return ToolResult("delete_memory", True, f"Deleted memory {memory_id}.", metadata={"memory_id": memory_id})


def update_memory(arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    memory_id = arguments["memory_id"]
    updated = memory_store.update_memory(
        memory_id,
        content=arguments.get("content"),
        tags=arguments.get("tags"),
        metadata=arguments.get("metadata"),
        importance=arguments.get("importance"),
        source=arguments.get("source"),
        confidence=arguments.get("confidence"),
    )
    if not updated:
        return ToolResult("update_memory", False, f"Memory not found: {memory_id}", error="not_found")
    return ToolResult("update_memory", True, f"Updated memory {memory_id}.", metadata={"memory_id": memory_id})


def summarize_memories(arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    summary = memory_store.summarize_memories(query=arguments.get("query"), limit=arguments.get("limit", 10))
    return ToolResult("summarize_memories", True, summary)


def archive_memory(arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    memory_id = arguments["memory_id"]
    archived = memory_store.archive_memory(memory_id)
    if not archived:
        return ToolResult("archive_memory", False, f"Memory not found or already archived: {memory_id}", error="not_found")
    return ToolResult("archive_memory", True, f"Archived memory {memory_id}.", metadata={"memory_id": memory_id})


def restore_memory(arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    memory_id = arguments["memory_id"]
    restored = memory_store.restore_memory(memory_id)
    if not restored:
        return ToolResult("restore_memory", False, f"Memory not found or not archived: {memory_id}", error="not_found")
    return ToolResult("restore_memory", True, f"Restored memory {memory_id}.", metadata={"memory_id": memory_id})


def compact_memories(_arguments: dict[str, Any], memory_store: SQLiteMemoryStore) -> ToolResult:
    archived_count = memory_store.compact_memories()
    return ToolResult("compact_memories", True, f"Archived {archived_count} duplicate memories.")


def import_memories(arguments: dict[str, Any], memory_store: SQLiteMemoryStore, project_root: Path) -> ToolResult:
    path = _resolve_inside_root(project_root, arguments["path"])
    memory_ids = memory_store.import_markdown(path)
    return ToolResult(
        "import_memories",
        True,
        f"Imported {len(memory_ids)} memories.",
        metadata={"memory_ids": memory_ids, "path": str(path.relative_to(project_root))},
    )


def export_memories(arguments: dict[str, Any], memory_store: SQLiteMemoryStore, project_root: Path) -> ToolResult:
    path = _resolve_inside_root(project_root, arguments["path"])
    count = memory_store.export_markdown(path, include_archived=arguments.get("include_archived", False))
    return ToolResult(
        "export_memories",
        True,
        f"Exported {count} memories.",
        metadata={"path": str(path.relative_to(project_root)), "count": count},
    )


def _format_memories(memories: list[Any]) -> str:
    if not memories:
        return "No memories found."
    payload = [
        {
            "id": memory.id,
            "content": memory.content,
            "created_at": memory.created_at,
            "updated_at": memory.updated_at,
            "tags": memory.tags,
            "metadata": memory.metadata,
            "importance": memory.importance,
            "source": memory.source,
            "confidence": memory.confidence,
            "archived_at": memory.archived_at,
            "access_count": memory.access_count,
            "last_accessed_at": memory.last_accessed_at,
        }
        for memory in memories
    ]
    return json.dumps(payload, indent=2, sort_keys=True)


def _resolve_inside_root(project_root: Path, raw_path: str) -> Path:
    root = project_root.resolve()
    candidate = (root / raw_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Path is outside the project root")
    return candidate
