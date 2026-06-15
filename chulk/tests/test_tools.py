"""Tests for built-in tools and registry behavior."""

from pathlib import Path
import sys

from chulk.memory import SQLiteMemoryStore
from chulk.tools import Tool, ToolRegistry, calculator_tool, create_default_tool_registry
from chulk.tools.files import apply_patch_tool, list_files_tool, read_file_tool, search_files_tool, write_file_tool
from chulk.tools.output import preview_text
from chulk.tools.permissions import (
    PermissionDecision,
    ToolPermissionLevel,
    ToolPermissionPolicy,
    normalize_permission_level,
    permission_policy_for_profile,
)
from chulk.tools.registry import ToolResult
from chulk.tools.shell import run_shell_command


def test_registry_descriptions_include_registered_tool():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    description = registry.tool_descriptions_for_prompt()

    assert "calculator" in description
    assert "expression" in description
    assert "permission_level" in description
    assert "read" in description


def test_tool_permission_policy_requires_confirmation_by_default():
    tool = Tool(
        name="dangerous",
        description="Dangerous test tool.",
        args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        callable=lambda _arguments: ToolResult("dangerous", True, "ran"),
        requires_confirmation=True,
        permission_level=ToolPermissionLevel.SHELL,
    )
    policy = ToolPermissionPolicy()

    request = policy.request_for_tool(tool, {})
    decision = policy.decide(request)

    assert request.permission_level == ToolPermissionLevel.SHELL
    assert decision.decision == PermissionDecision.ASK
    assert decision.requires_confirmation is True


def test_unknown_permission_level_fails_closed():
    try:
        normalize_permission_level("not-a-level")
    except ValueError as exc:
        assert "Unknown tool permission level" in str(exc)
        assert "read" in str(exc)
    else:
        raise AssertionError("Expected unknown permission level to fail")


def test_registry_rejects_unknown_tool_permission_level():
    registry = ToolRegistry()

    try:
        registry.register(
            Tool(
                name="bad_permission",
                description="Bad permission level.",
                args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
                callable=lambda _arguments: ToolResult("bad_permission", True, "ran"),
                permission_level="not-a-level",
            )
        )
    except ValueError as exc:
        assert "Unknown tool permission level" in str(exc)
    else:
        raise AssertionError("Expected invalid tool permission level registration to fail")


def test_built_in_permission_profiles_have_expected_decisions():
    read_tool = _permission_test_tool("reader", ToolPermissionLevel.READ)
    write_tool = _permission_test_tool("writer", ToolPermissionLevel.WRITE)
    shell_tool = _permission_test_tool("sheller", ToolPermissionLevel.SHELL, requires_confirmation=True)
    destructive_tool = _permission_test_tool("destroyer", ToolPermissionLevel.DESTRUCTIVE)

    read_only = permission_policy_for_profile("read-only")
    workspace_write = permission_policy_for_profile("workspace-write")
    trusted_local = permission_policy_for_profile("trusted-local")
    full_access = permission_policy_for_profile("full-access")

    assert read_only.decide(read_only.request_for_tool(read_tool, {})).decision == PermissionDecision.ALLOW
    assert read_only.decide(read_only.request_for_tool(write_tool, {})).decision == PermissionDecision.DENY
    assert workspace_write.decide(workspace_write.request_for_tool(write_tool, {})).decision == PermissionDecision.ALLOW
    assert workspace_write.decide(workspace_write.request_for_tool(shell_tool, {})).decision == PermissionDecision.ASK
    assert trusted_local.decide(trusted_local.request_for_tool(shell_tool, {})).decision == PermissionDecision.ALLOW
    assert trusted_local.decide(trusted_local.request_for_tool(destructive_tool, {})).decision == PermissionDecision.ASK
    assert full_access.decide(full_access.request_for_tool(destructive_tool, {})).decision == PermissionDecision.ALLOW


def test_registry_logs_tool_calls():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    registry.run("calculator", {"expression": "1 + 1"})

    assert registry.call_log == [
        {
            "tool_name": "calculator",
            "arguments": {"expression": "1 + 1"},
            "success": True,
            "error": None,
            "observation": "1 + 1 = 2",
        }
    ]


def _permission_test_tool(
    name: str,
    permission_level: ToolPermissionLevel,
    *,
    requires_confirmation: bool = False,
) -> Tool:
    return Tool(
        name=name,
        description="Permission profile test tool.",
        args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        callable=lambda _arguments: ToolResult(name, True, "ran"),
        requires_confirmation=requires_confirmation,
        permission_level=permission_level,
    )


def test_registry_validates_required_arguments():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    result = registry.run("calculator", {})

    assert not result.success
    assert result.error == "invalid_arguments"
    assert "Missing required argument" in result.observation
    assert result.metadata["validation_errors"] == [
        {
            "path": "expression",
            "message": "Missing required argument",
            "expected": None,
            "actual": None,
        }
    ]


def test_registry_reports_multiple_argument_validation_errors():
    calls = []
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="profile",
            description="Validate profile arguments.",
            args_schema={
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "minimum": 1, "maximum": 120},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "maxItems": 3,
                    },
                },
                "required": ["age"],
                "additionalProperties": False,
            },
            callable=lambda arguments: calls.append(arguments) or ToolResult("profile", True, "ok"),
        )
    )

    result = registry.run("profile", {"age": 0, "tags": ["ok", ""], "extra": True})

    assert not result.success
    assert result.error == "invalid_arguments"
    assert calls == []
    assert "age: number is too small" in result.observation
    assert "tags[1]: string is too short" in result.observation
    assert "extra: Unknown argument" in result.observation
    assert result.metadata["validation_errors"] == [
        {"path": "extra", "message": "Unknown argument", "expected": None, "actual": None},
        {"path": "age", "message": "number is too small", "expected": ">= 1", "actual": "0"},
        {
            "path": "tags[1]",
            "message": "string is too short",
            "expected": "at least 1 characters",
            "actual": "0 characters",
        },
    ]


def test_registry_rejects_invalid_tool_schema_at_registration():
    registry = ToolRegistry()

    try:
        registry.register(
            Tool(
                name="broken_schema",
                description="Broken schema.",
                args_schema={
                    "type": "object",
                    "properties": {},
                    "required": ["missing"],
                    "additionalProperties": "nope",
                },
                callable=lambda _arguments: ToolResult("broken_schema", True, "unused"),
            )
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected invalid schema registration to fail")

    assert "Invalid schema for tool broken_schema" in message
    assert "required field missing is not declared in properties" in message
    assert "additionalProperties must be boolean" in message


def test_registry_catches_tool_exceptions():
    def broken_tool(_arguments):
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="broken",
            description="Broken test tool",
            args_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            callable=broken_tool,
        )
    )

    result = registry.run("broken", {})

    assert not result.success
    assert "Tool execution failed" in result.observation
    assert "boom" in result.observation
    assert result.error == "boom"
    assert result.metadata["exception_type"] == "RuntimeError"


def test_preview_text_preserves_head_and_tail_when_truncated():
    text = "HEAD-" + ("middle-" * 20) + "IMPORTANT_TAIL"

    preview = preview_text(text, max_chars=80)

    assert preview.truncated
    assert preview.returned_length <= 80
    assert preview.original_length == len(text)
    assert "HEAD-" in preview.text
    assert "IMPORTANT_TAIL" in preview.text
    assert "middle omitted" in preview.text
    assert preview.sha256


def test_registry_returns_safe_unknown_tool_error():
    registry = ToolRegistry()

    result = registry.run("missing", {})

    assert not result.success
    assert "Unknown tool" in result.observation
    assert result.error == "unknown_tool"
    assert result.metadata["available_tools"] == []


def test_calculator_evaluates_arithmetic():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    result = registry.run("calculator", {"expression": "(2 + 3) * 4"})

    assert result.success
    assert result.observation == "(2 + 3) * 4 = 20"


def test_calculator_rejects_non_arithmetic_expression():
    registry = ToolRegistry()
    registry.register(calculator_tool())

    result = registry.run("calculator", {"expression": "__import__('os').system('echo nope')"})

    assert not result.success
    assert result.error == "invalid_expression"


def test_shell_command_success(tmp_path):
    result = run_shell_command({"command": "printf hello"}, tmp_path)

    assert result.success
    assert result.stdout == "hello"
    assert result.exit_code == 0
    assert result.metadata["cwd"] == str(tmp_path.resolve())
    assert result.metadata["stdout_length"] == 5


def test_shell_blocks_destructive_command(tmp_path):
    result = run_shell_command({"command": "rm -rf /"}, tmp_path)

    assert not result.success
    assert result.error == "blocked_command"


def test_shell_blocks_output_redirection_outside_root(tmp_path):
    result = run_shell_command({"command": "printf hello > ../outside.txt"}, tmp_path)

    assert not result.success
    assert result.error == "blocked_command"
    assert "outside the project root" in result.observation


def test_shell_command_timeout(tmp_path):
    command = f"{sys.executable} -c 'import time; time.sleep(2)'"

    result = run_shell_command({"command": command, "timeout_seconds": 1}, tmp_path, default_timeout_seconds=1)

    assert not result.success
    assert result.error == "timeout"


def test_file_tools_read_write_list_and_search(tmp_path):
    registry = ToolRegistry()
    registry.register(write_file_tool(tmp_path))
    registry.register(read_file_tool(tmp_path))
    registry.register(list_files_tool(tmp_path))
    registry.register(search_files_tool(tmp_path))

    write_result = registry.run("write_file", {"path": "notes/example.txt", "content": "hello chulk"})
    read_result = registry.run("read_file", {"path": "notes/example.txt"})
    list_result = registry.run("list_files", {"path": ".", "pattern": "*.txt", "recursive": True})
    search_result = registry.run("search_files", {"query": "chulk", "path": ".", "pattern": "*.txt"})

    assert write_result.success
    assert read_result.success
    assert read_result.observation == "hello chulk"
    assert "notes/example.txt" in list_result.observation
    assert "notes/example.txt" in search_result.observation


def test_search_files_ignores_runtime_trace_artifacts(tmp_path):
    (tmp_path / "chulk" / "core").mkdir(parents=True)
    (tmp_path / "traces").mkdir()
    (tmp_path / "chulk" / "core" / "agent.py").write_text("agentic loop lives here\n", encoding="utf-8")
    (tmp_path / "traces" / "session.jsonl").write_text("agentic loop noisy trace\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(search_files_tool(tmp_path))

    result = registry.run("search_files", {"query": "agentic loop", "path": ".", "max_results": 10})

    assert result.success
    assert "chulk/core/agent.py" in result.observation
    assert "traces/session.jsonl" not in result.observation


def test_apply_patch_tool_modifies_existing_file(tmp_path):
    (tmp_path / "notes.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(apply_patch_tool(tmp_path))

    result = registry.run(
        "apply_patch",
        {
            "patch": "\n".join(
                [
                    "--- a/notes.txt",
                    "+++ b/notes.txt",
                    "@@ -1,3 +1,3 @@",
                    " one",
                    "-two",
                    "+TWO",
                    " three",
                ]
            )
        },
    )

    assert result.success
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "one\nTWO\nthree\n"
    assert result.metadata["paths"] == ["notes.txt"]
    assert result.metadata["changed_count"] == 1
    assert result.metadata["modified_count"] == 1
    assert result.metadata["changes"][0]["sha256_before"]
    assert result.metadata["changes"][0]["sha256_after"]


def test_apply_patch_tool_creates_new_file(tmp_path):
    registry = ToolRegistry()
    registry.register(apply_patch_tool(tmp_path))

    result = registry.run(
        "apply_patch",
        {
            "patch": "\n".join(
                [
                    "--- /dev/null",
                    "+++ b/notes/new.txt",
                    "@@ -0,0 +1,2 @@",
                    "+alpha",
                    "+beta",
                ]
            )
        },
    )

    assert result.success
    assert (tmp_path / "notes" / "new.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert result.metadata["created_count"] == 1
    assert result.metadata["changes"][0]["status"] == "created"
    assert result.metadata["changes"][0]["sha256_before"] is None


def test_apply_patch_tool_blocks_unsafe_paths(tmp_path):
    registry = ToolRegistry()
    registry.register(apply_patch_tool(tmp_path))

    for path in [".env", "chulk/store.sqlite", ".git/config", "secrets.txt", "traces/output.txt"]:
        result = registry.run(
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "--- /dev/null",
                        f"+++ b/{path}",
                        "@@ -0,0 +1 @@",
                        "+blocked",
                    ]
                )
            },
        )

        assert not result.success
        assert result.error == "unsafe_path"
        assert not (tmp_path / path).exists()


def test_apply_patch_tool_is_atomic_on_multi_file_failure(tmp_path):
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(apply_patch_tool(tmp_path))

    result = registry.run(
        "apply_patch",
        {
            "patch": "\n".join(
                [
                    "--- a/a.txt",
                    "+++ b/a.txt",
                    "@@ -1 +1 @@",
                    "-a",
                    "+A",
                    "--- a/b.txt",
                    "+++ b/b.txt",
                    "@@ -1 +1 @@",
                    "-not-b",
                    "+B",
                ]
            )
        },
    )

    assert not result.success
    assert result.error == "patch_context_mismatch"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a\n"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "b\n"


def test_write_file_blocks_unsafe_paths_and_guards_overwrites(tmp_path):
    registry = ToolRegistry()
    registry.register(write_file_tool(tmp_path))

    create_result = registry.run("write_file", {"path": "safe.txt", "content": "hello"})
    exists_result = registry.run("write_file", {"path": "safe.txt", "content": "replace"})
    overwrite_result = registry.run("write_file", {"path": "safe.txt", "content": "replace", "overwrite": True})
    unsafe_result = registry.run("write_file", {"path": ".env", "content": "OPENAI_API_KEY=secret"})
    tokenizer_result = registry.run("write_file", {"path": "tokenizer.json", "content": "{}"})

    assert create_result.success
    assert exists_result.error == "exists"
    assert overwrite_result.success
    assert overwrite_result.metadata["status"] == "modified"
    assert unsafe_result.error == "unsafe_path"
    assert tokenizer_result.success


def test_file_tools_block_path_traversal(tmp_path):
    registry = ToolRegistry()
    registry.register(read_file_tool(tmp_path))

    result = registry.run("read_file", {"path": "../outside.txt"})

    assert not result.success
    assert "outside the project root" in (result.error or result.observation)


def test_default_tool_registry_contains_builtins(tmp_path):
    registry = create_default_tool_registry(tmp_path)
    names = {tool.name for tool in registry.list_tools()}

    assert {"calculator", "run_cmd", "read_file", "apply_patch", "write_file", "list_files", "search_files"} <= names


def test_default_tool_registry_contains_memory_tools_when_store_is_provided(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    registry = create_default_tool_registry(tmp_path, memory_store=memory_store)
    names = {tool.name for tool in registry.list_tools()}

    assert {
        "save_memory",
        "search_memory",
        "list_memories",
        "delete_memory",
        "update_memory",
        "summarize_memories",
        "archive_memory",
        "restore_memory",
        "compact_memories",
        "import_memories",
        "export_memories",
    } <= names


def test_memory_tools_save_search_update_and_delete(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    registry = create_default_tool_registry(tmp_path, memory_store=memory_store)

    save_result = registry.run(
        "save_memory",
        {
            "content": "ChulkHarness should keep memory and skills separate.",
            "tags": ["project", "memory"],
            "importance": 7,
        },
    )
    memory_id = save_result.metadata["memory_id"]
    search_result = registry.run("search_memory", {"query": "skills separate"})
    update_result = registry.run("update_memory", {"memory_id": memory_id, "tags": ["project", "preference"]})
    list_result = registry.run("list_memories", {"limit": 5})
    delete_result = registry.run("delete_memory", {"memory_id": memory_id})

    assert save_result.success
    assert "skills separate" in search_result.observation
    assert update_result.success
    assert "preference" in list_result.observation
    assert delete_result.success


def test_memory_tools_import_export_archive_restore_and_compact(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    registry = create_default_tool_registry(tmp_path, memory_store=memory_store)
    markdown = tmp_path / "MEMORY.md"
    markdown.write_text("- [project] ChulkHarness has importable Markdown memory.\n", encoding="utf-8")

    import_result = registry.run("import_memories", {"path": "MEMORY.md"})
    memory_id = import_result.metadata["memory_ids"][0]
    archive_result = registry.run("archive_memory", {"memory_id": memory_id})
    restore_result = registry.run("restore_memory", {"memory_id": memory_id})
    compact_result = registry.run("compact_memories", {})
    export_result = registry.run("export_memories", {"path": "memory-export.md"})

    assert import_result.success
    assert archive_result.success
    assert restore_result.success
    assert compact_result.success
    assert export_result.success
    assert (tmp_path / "memory-export.md").exists()


def test_memory_export_blocks_unsafe_paths(tmp_path):
    memory_store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    registry = create_default_tool_registry(tmp_path, memory_store=memory_store)

    result = registry.run("export_memories", {"path": "chulk/store.sqlite"})

    assert not result.success
    assert result.error == "unsafe_path"
