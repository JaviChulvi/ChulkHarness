"""Built-in tool registration."""

from __future__ import annotations

from pathlib import Path

from chulk.capabilities import Capabilities, FileAccess, MemoryMode
from chulk.memory import MemoryPolicy, SQLiteMemoryStore
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
from chulk.tools.permissions import ToolPermissionLevel
from chulk.tools.registry import Tool, ToolRegistry, ToolResult
from chulk.tools.shell import (
    DEFAULT_SHELL_STDERR_LIMIT_BYTES,
    DEFAULT_SHELL_STDOUT_LIMIT_BYTES,
    ShellExecutionPolicy,
    shell_tool,
)


def create_default_tool_registry(
    project_root: Path,
    shell_timeout_seconds: int = 10,
    memory_store: SQLiteMemoryStore | None = None,
    capabilities: Capabilities | None = None,
    memory_policy: MemoryPolicy | None = None,
    *,
    max_tool_stdout_bytes: int = DEFAULT_SHELL_STDOUT_LIMIT_BYTES,
    max_tool_stderr_bytes: int = DEFAULT_SHELL_STDERR_LIMIT_BYTES,
    shell_execution_policy: ShellExecutionPolicy | None = None,
    require_shell_containment: bool = False,
) -> ToolRegistry:
    """Create the default tool registry for the agent runtime."""
    selected = capabilities or Capabilities.full()
    registry = ToolRegistry()
    if selected.utilities:
        registry.register(calculator_tool())
    if selected.shell:
        registry.register(
            shell_tool(
                project_root,
                timeout_seconds=shell_timeout_seconds,
                stdout_limit_bytes=max_tool_stdout_bytes,
                stderr_limit_bytes=max_tool_stderr_bytes,
                execution_policy=shell_execution_policy,
                require_containment=require_shell_containment,
            )
        )
    if selected.files in {FileAccess.READ, FileAccess.WRITE}:
        registry.register(read_file_tool(project_root))
        registry.register(list_files_tool(project_root))
        registry.register(search_files_tool(project_root))
    if selected.files is FileAccess.WRITE:
        registry.register(apply_patch_tool(project_root))
        registry.register(write_file_tool(project_root))
    if memory_store is not None and selected.memory is not MemoryMode.OFF:
        registry.register(search_memory_tool(memory_store))
        registry.register(list_memories_tool(memory_store))
        registry.register(summarize_memories_tool(memory_store))
        if selected.memory is MemoryMode.MANUAL and memory_policy is not None:
            registry.register(_manual_save_memory_tool(memory_policy))
        if selected.memory is MemoryMode.AUTOMATIC:
            registry.register(save_memory_tool(memory_store))
            registry.register(delete_memory_tool(memory_store))
            registry.register(update_memory_tool(memory_store))
            registry.register(archive_memory_tool(memory_store))
            registry.register(restore_memory_tool(memory_store))
            registry.register(compact_memories_tool(memory_store))
            registry.register(import_memories_tool(memory_store, project_root))
            registry.register(export_memories_tool(memory_store, project_root))
    return registry


def _manual_save_memory_tool(policy: MemoryPolicy) -> Tool:
    def propose(arguments: dict, context=None) -> ToolResult:
        metadata = context.metadata if context is not None else {}
        result = policy.propose_explicit(
            arguments["content"],
            tags=arguments.get("tags") or [],
            metadata=arguments.get("metadata") or {},
            importance=arguments.get("importance", 1),
            source=arguments.get("source", "user_explicit"),
            confidence=arguments.get("confidence", 1.0),
            conversation_id=metadata.get("conversation_id"),
            turn_id=metadata.get("turn_id"),
        )
        proposal_id = result.proposal_ids[0]
        return ToolResult(
            "save_memory",
            True,
            f"Proposed memory {proposal_id} for application review.",
            metadata={"proposal_id": proposal_id, "review_required": True},
        )

    return Tool(
        name="save_memory",
        description="Propose a durable memory for application review before it is saved.",
        args_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "minLength": 1},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
                "importance": {"type": "integer", "minimum": 1, "maximum": 10},
                "source": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["content"],
            "additionalProperties": False,
        },
        callable=propose,
        accepts_context=True,
        run_in_executor=True,
        permission_level=ToolPermissionLevel.MEMORY,
    )
