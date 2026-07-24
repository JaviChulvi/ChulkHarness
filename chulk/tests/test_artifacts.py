from __future__ import annotations

import json
import os

import pytest

from chulk import Agent, AgentConfig, Tools
from chulk.testing import ScriptedLLMClient
from chulk.tracing import ArtifactAccessError, JSONLTraceLogger, TraceArtifactStore


def test_artifact_reads_are_bounded_and_support_slice_head_tail(tmp_path) -> None:
    logger = JSONLTraceLogger(tmp_path / "traces", "owner")
    reference = logger.write_artifact(
        "tool-output",
        "HEAD-" + ("middle-" * 40) + "TAIL",
    )
    artifact_id = reference["artifact_id"]

    head = logger.read_artifact(artifact_id, mode="head", max_bytes=12)
    tail = logger.read_artifact(artifact_id, mode="tail", max_bytes=12)
    sliced = logger.read_artifact(
        artifact_id,
        mode="slice",
        offset=5,
        max_bytes=14,
    )
    head_tail = logger.read_artifact(
        artifact_id,
        mode="head_tail",
        max_bytes=20,
    )

    assert "path" not in reference
    assert head.content == "HEAD-middle-"
    assert tail.content.endswith("TAIL")
    assert sliced.ranges == ((5, 19),)
    assert sliced.content == "middle-middle-"
    assert "[... omitted bytes ...]" in head_tail.content
    assert head_tail.byte_count == 20
    assert head_tail.truncated is True


def test_artifact_reference_rejects_wrong_owner_forgery_and_path_tamper(tmp_path) -> None:
    traces_dir = tmp_path / "traces"
    owner = JSONLTraceLogger(traces_dir, "owner")
    reference = owner.write_artifact("output", "sensitive")
    artifact_id = reference["artifact_id"]

    with pytest.raises(ArtifactAccessError, match="missing"):
        TraceArtifactStore(traces_dir, "other").read(artifact_id)
    for forged in ("../owner", "art_../../escape", artifact_id + "/tail"):
        with pytest.raises(ArtifactAccessError, match="invalid"):
            owner.read_artifact(forged)

    manifest = owner.artifact_store.manifest_path
    record = json.loads(manifest.read_text(encoding="utf-8"))
    record["filename"] = "../outside.txt"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ArtifactAccessError, match="ownership"):
        owner.read_artifact(artifact_id)


@pytest.mark.skipif(os.name != "posix", reason="symlink behavior")
def test_artifact_reader_rejects_symlink_and_missing_target(tmp_path) -> None:
    logger = JSONLTraceLogger(tmp_path / "traces", "owner")
    first = logger.write_artifact("first", "first content")
    first_path = logger.artifacts_dir / f"{first['artifact_id']}.txt"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside content", encoding="utf-8")
    first_path.unlink()
    first_path.symlink_to(outside)

    with pytest.raises(ArtifactAccessError, match="not a regular file"):
        logger.read_artifact(first["artifact_id"])

    second = logger.write_artifact("second", "second content")
    (logger.artifacts_dir / f"{second['artifact_id']}.txt").unlink()
    with pytest.raises(ArtifactAccessError, match="missing"):
        logger.read_artifact(second["artifact_id"])


def test_artifact_reader_rejects_oversized_and_hash_mismatch(tmp_path) -> None:
    logger = JSONLTraceLogger(tmp_path / "traces", "owner")
    oversized = logger.write_artifact("large", "x" * 128)
    with pytest.raises(ArtifactAccessError, match="integrity-read bound"):
        logger.artifact_store.read(
            oversized["artifact_id"],
            max_artifact_bytes=64,
        )

    tampered = logger.write_artifact("tampered", "original")
    tampered_path = logger.artifacts_dir / f"{tampered['artifact_id']}.txt"
    tampered_path.write_text("modified", encoding="utf-8")
    with pytest.raises(ArtifactAccessError, match="hash"):
        logger.read_artifact(tampered["artifact_id"])


def test_artifact_reader_is_opt_in_as_tool_and_available_as_host_api(tmp_path) -> None:
    client = ScriptedLLMClient(
        [{"type": "final_answer", "content": "done"}]
    )
    with Agent(
        config=AgentConfig(project_root=tmp_path),
        llm=client,
        tools=[Tools.read_trace_artifact],
        skills=[],
    ) as agent:
        reference = agent.runtime.trace_logger.write_artifact(
            "host-evidence",
            "bounded evidence",
        )
        artifact_id = reference["artifact_id"]
        host_view = agent.read_artifact(
            artifact_id,
            mode="slice",
            max_bytes=7,
        )
        tool_result = agent.tool_registry.run(
            "read_trace_artifact",
            {
                "artifact_id": artifact_id,
                "mode": "tail",
                "max_bytes": 8,
            },
        )

    assert host_view["content"] == "bounded"
    assert tool_result.success is True
    assert tool_result.value["content"] == "evidence"

    default_agent = Agent(
        config=AgentConfig(project_root=tmp_path / "default"),
        llm=ScriptedLLMClient(
            [{"type": "final_answer", "content": "done"}]
        ),
        tools=[],
        skills=[],
    )
    try:
        assert "read_trace_artifact" not in {
            tool.name for tool in default_agent.tool_registry.list_tools()
        }
    finally:
        default_agent.close()
