"""Built-in tool registration."""

from __future__ import annotations

from pathlib import Path

from chulk.memory import SQLiteMemoryStore
from chulk.tools.calculator import calculator_tool
from chulk.tools.files import apply_patch_tool, list_files_tool, read_file_tool, search_files_tool, write_file_tool
from chulk.tools.memory import (
    archive_memory_tool,
    compact_memories_tool,
    delete_memory_tool,
    export_memories_tool,
    import_memories_tool,
    list_memories_tool,
    restore_memory_tool,
    save_memory_tool,
    search_memory_tool,
    summarize_memories_tool,
    update_memory_tool,
)
from chulk.tools.registry import ToolRegistry
from chulk.tools.shell import shell_tool


def create_default_tool_registry(
    project_root: Path,
    shell_timeout_seconds: int = 10,
    memory_store: SQLiteMemoryStore | None = None,
) -> ToolRegistry:
    """Create the default tool registry for the agent runtime."""
    registry = ToolRegistry()
    registry.register(calculator_tool())
    registry.register(shell_tool(project_root, timeout_seconds=shell_timeout_seconds))
    registry.register(read_file_tool(project_root))
    registry.register(apply_patch_tool(project_root))
    registry.register(write_file_tool(project_root))
    registry.register(list_files_tool(project_root))
    registry.register(search_files_tool(project_root))
    if memory_store is not None:
        registry.register(save_memory_tool(memory_store))
        registry.register(search_memory_tool(memory_store))
        registry.register(list_memories_tool(memory_store))
        registry.register(delete_memory_tool(memory_store))
        registry.register(update_memory_tool(memory_store))
        registry.register(summarize_memories_tool(memory_store))
        registry.register(archive_memory_tool(memory_store))
        registry.register(restore_memory_tool(memory_store))
        registry.register(compact_memories_tool(memory_store))
        registry.register(import_memories_tool(memory_store, project_root))
        registry.register(export_memories_tool(memory_store, project_root))
    return registry
