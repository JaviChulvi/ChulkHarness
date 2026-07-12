"""Deterministic, read-only repository review built only on the public SDK."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

from chulk import Agent, AgentConfig, Capabilities, FileAccess, MemoryMode, Tool, ToolContext
from chulk.testing import ScriptedLLMClient


HERE = Path(__file__).resolve().parent
FIXTURE_ROOT = HERE / "fixture"

FINDING = (
    "High — The setup guide tells users to commit a production API token. "
    "Keep secrets in an ignored environment file or secret manager, rotate any "
    "token that was committed, and document the safe setup path."
)


@Tool
def read_fixture(relative_path: str, context: ToolContext[Path]) -> str:
    """Read one UTF-8 text file from the review fixture."""
    root = context.require_deps().resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError("relative_path must stay inside the review fixture")
    return target.read_text(encoding="utf-8")


def run_review(runtime_dir: Path) -> tuple[str, tuple[str, ...], Path]:
    """Run the deterministic review and return its public output contract."""
    client = ScriptedLLMClient(
        [
            {
                "type": "tool_call",
                "tool_name": "read_fixture",
                "arguments": {"relative_path": "README.md"},
            },
            {"type": "final_answer", "content": FINDING},
        ]
    )
    capabilities = Capabilities(
        files=FileAccess.OFF,
        memory=MemoryMode.OFF,
        utilities=True,
    )
    with Agent(
        config=AgentConfig(
            project_root=FIXTURE_ROOT,
            runtime_dir=runtime_dir,
            permission_profile="read-only",
        ),
        llm=client,
        tools=[read_fixture],
        skills=[],
        capabilities=capabilities,
        deps=FIXTURE_ROOT,
    ) as reviewer:
        events = list(reviewer.run_events("Review README.md for one high-impact risk."))
        terminal = events[-1]
        result = terminal.payload.result  # type: ignore[union-attr]
        if result.trace_path is None:
            raise RuntimeError("review run did not produce a trace")
        return result.content, tuple(event.name for event in events), result.trace_path


def write_normalized_trace(source: Path, target: Path) -> None:
    """Write a scrubbed, deterministic walkthrough derived from a real trace."""
    selected = {
        "turn_started",
        "model_request_started",
        "tool_call_started",
        "tool_call_completed",
        "final_answer",
        "turn_finished",
    }
    raw_events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    events = [event for event in raw_events if event.get("type") in selected]
    normalized: list[dict[str, object]] = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index, event in enumerate(events):
        event_type = str(event["type"])
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        normalized.append(
            {
                "schema_version": 1,
                "conversation_id": "conversation-demo",
                "turn_id": "turn-demo",
                "timestamp": (base + timedelta(seconds=index)).isoformat(),
                "payload": _normalized_payload(event_type, payload),
                "type": event_type,
            }
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n" for event in normalized),
        encoding="utf-8",
    )


def _normalized_payload(event_type: str, payload: dict[str, object]) -> dict[str, object]:
    if event_type == "turn_started":
        return {"conversation_id": "conversation-demo", "turn_id": "turn-demo"}
    if event_type == "model_request_started":
        return {"request_index": payload.get("request_index", 1), "purpose": "action"}
    if event_type == "tool_call_started":
        return {"tool_name": "read_fixture"}
    if event_type == "tool_call_completed":
        return {"success": True, "tool_name": "read_fixture"}
    if event_type == "final_answer":
        return {"content": FINDING}
    return {"status": "completed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-sample", type=Path, help="write a normalized sample trace")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="chulk-review-") as temporary:
        content, events, trace_path = run_review(Path(temporary))
        if args.write_sample is not None:
            write_normalized_trace(trace_path, args.write_sample)

        print("Repository review")
        print(content)
        print("events: " + " -> ".join(events))
        print(f"trace_path: {trace_path}")
        if args.write_sample is not None:
            print(f"sample_trace: {args.write_sample.resolve()}")


if __name__ == "__main__":
    main()
