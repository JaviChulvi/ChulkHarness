from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def grounded_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    monkeypatch.setenv("CHULK_EXAMPLE_MODE", "scripted")
    monkeypatch.setenv("CHULK_EXAMPLE_RUNTIME_DIR", str(tmp_path))

    import common
    from grounded_assistant import app

    monkeypatch.setattr(common, "EXAMPLE_STATE_ROOT", tmp_path)
    return app


def test_successful_answer_uses_actual_runtime_and_retrieved_source(
    grounded_app: ModuleType,
) -> None:
    result, sources, citations, mode = grounded_app.run_assistant(
        "How much annual leave is offered?"
    )

    assert mode == "scripted"
    assert result.status == "completed"
    assert result.tool_calls[0].tool_name == "get_source"
    assert result.tool_calls[0].success is True
    assert sources.retrieved_ids == {"handbook-leave"}
    assert citations == ("handbook-leave",)


def test_malformed_citation_is_rejected_when_valid_citation_is_also_present(
    grounded_app: ModuleType,
) -> None:
    responses = [
        {
            "type": "tool_call",
            "tool_name": "get_source",
            "arguments": {"source_id": "handbook-leave"},
        },
        {
            "type": "final_answer",
            "content": (
                "Annual leave is 20 days [source:handbook-leave], with a private "
                "exception [source:other/workspace]."
            ),
        },
    ]

    with pytest.raises(ValueError, match="invalid citation marker"):
        grounded_app.run_assistant("How much annual leave is offered?", responses=responses)


def test_unknown_source_and_unretrieved_citation_are_rejected_by_real_run(
    tmp_path: Path,
    grounded_app: ModuleType,
) -> None:
    responses = [
        {
            "type": "tool_call",
            "tool_name": "get_source",
            "arguments": {"source_id": "other-workspace-private"},
        },
        {
            "type": "final_answer",
            "content": "The private plan is approved [source:other-workspace-private].",
        },
    ]

    with pytest.raises(ValueError, match="not retrieved from the scoped source set"):
        grounded_app.run_assistant("What is the private plan?", responses=responses)

    traces = list(tmp_path.rglob("*.jsonl"))
    assert traces, "the real Agent runtime should write a trace"
    records = [
        json.loads(line)
        for path in traces
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    failed_calls = [
        record["payload"]
        for record in records
        if record["type"] == "tool_call_failed"
    ]
    assert any(
        call["tool_name"] == "get_source" and call["success"] is False
        for call in failed_calls
    )
