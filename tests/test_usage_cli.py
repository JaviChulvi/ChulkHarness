from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

from chulk.cli.parser import build_parser
from chulk.cli.usage import run_usage_command
from chulk.config import load_config
from chulk.main import main
from chulk.model_profiles import ModelProfile, ModelProfileStore
from chulk.profiles import CredentialRef
from chulk.testing import ScriptedLLMClient
from chulk.usage import (
    ExactCost,
    ResourceKind,
    SQLiteUsageStore,
    UsageDimensions,
    UsageEntry,
    UsageLedger,
)


NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)


def _seed(db_path: Path, *, profile_id: str = "default") -> None:
    SQLiteUsageStore(db_path, clock=lambda: NOW).ingest(
        UsageEntry(
            id="entry-1",
            resource_kind=ResourceKind.MODEL,
            source_event_id="request-1",
            dimensions=UsageDimensions(
                profile_id=profile_id,
                channel="cli",
                conversation_id="conversation-1",
                turn_id="turn-1",
            ),
            occurred_at=NOW,
            billing_period="2026-07",
            purpose="agent_action",
            units={
                "model_calls": Decimal(1),
                "input_tokens": Decimal(100),
                "output_tokens": Decimal(20),
                "total_tokens": Decimal(120),
            },
            cost=ExactCost(
                Decimal("0.0125"),
                pricing_known=True,
                estimated=True,
            ),
            provider="openai",
            model="gpt-4.1-mini",
            credential_ref="env:OPENAI_API_KEY",
            model_profile_id="fast",
        )
    )


def test_usage_parser_exposes_query_group_and_export_commands() -> None:
    parser = build_parser()

    range_args = parser.parse_args(
        [
            "usage",
            "range",
            "--from",
            "2026-07-01",
            "--to",
            "2026-07-25",
            "--resource-kind",
            "model",
            "--json",
        ]
    )
    group_args = parser.parse_args(
        ["usage", "group", "--by", "model", "--channel", "cli"]
    )
    export_args = parser.parse_args(
        [
            "usage",
            "export",
            "--format",
            "csv",
            "--output",
            "usage.csv",
        ]
    )

    assert range_args.usage_command == "range"
    assert range_args.start == "2026-07-01"
    assert range_args.resource_kind == "model"
    assert range_args.json_output
    assert group_args.by == "model"
    assert group_args.limit == 10_000
    assert export_args.format == "csv"


def test_usage_today_emits_exact_bounded_json(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    _seed(db_path)
    output: list[str] = []

    exit_code = run_usage_command(
        "today",
        ledger=UsageLedger(db_path),
        json_output=True,
        output_func=output.append,
        clock=lambda: NOW,
    )

    payload = json.loads(output[0])
    assert exit_code == 0
    assert payload["summary"] == {
        "currency": "USD",
        "entry_count": 1,
        "known_cost": "0.0125",
        "model_calls": 1,
        "tool_calls": 0,
        "total_tokens": 120,
        "unknown_cost_entries": 0,
    }
    assert payload["entries"][0]["model_profile_id"] == "fast"
    assert "credential_ref" not in payload["entries"][0]


def test_usage_today_applies_resource_and_channel_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    _seed(db_path)
    output: list[str] = []

    exit_code = run_usage_command(
        "today",
        ledger=UsageLedger(db_path),
        resource_kind="model",
        channel="other",
        json_output=True,
        output_func=output.append,
        clock=lambda: NOW,
    )

    payload = json.loads(output[0])
    assert exit_code == 0
    assert payload["summary"]["entry_count"] == 0
    assert payload["entries"] == []


def test_usage_group_and_export_are_profile_owned(tmp_path: Path) -> None:
    db_path = tmp_path / "store.sqlite"
    _seed(db_path, profile_id="work")
    _seed(tmp_path / "other.sqlite", profile_id="personal")
    output: list[str] = []
    ledger = UsageLedger(db_path, profile_id="work")

    assert (
        run_usage_command(
            "group",
            ledger=ledger,
            group_by="model",
            json_output=True,
            output_func=output.append,
        )
        == 0
    )
    grouped = json.loads(output.pop())
    assert grouped["groups"][0]["key"] == "openai:gpt-4.1-mini"
    assert grouped["groups"][0]["cost"]["amount"] == "0.0125"

    destination = tmp_path / "usage.json"
    assert (
        run_usage_command(
            "export",
            ledger=ledger,
            output_path=destination,
            export_format="json",
            output_func=output.append,
        )
        == 0
    )
    exported = destination.read_text()
    assert '"profile_id": "work"' in exported
    assert "OPENAI_API_KEY" not in exported


def test_main_routes_usage_to_the_selected_runtime_database(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    _seed(config.store_path)
    output: list[str] = []
    errors: list[str] = []
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))

    exit_code = main(
        ["usage", "range", "--from", "2026-07-25", "--to", "2026-07-25", "--json"],
        output_func=output.append,
        error_func=errors.append,
    )

    assert exit_code == 0
    assert errors == []
    payload = json.loads(output[0])
    assert payload["summary"]["entry_count"] == 1
    assert payload["entries"][0]["source_event_id"] == "request-1"


def test_usage_command_reports_invalid_ranges_without_writing(
    tmp_path: Path,
) -> None:
    output: list[str] = []
    errors: list[str] = []

    exit_code = run_usage_command(
        "range",
        ledger=UsageLedger(tmp_path / "store.sqlite"),
        start="2026-07-26",
        end="2026-07-25",
        output_func=output.append,
        error_func=errors.append,
    )

    assert exit_code == 1
    assert output == []
    assert errors == ["usage error: usage query start must be earlier than end"]


def test_model_profile_cost_limit_is_enforced_by_cli_exec(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    ModelProfileStore(
        config.runtime_dir / "control.sqlite",
        base_config=config,
    ).create(
        ModelProfile(
            id="budgeted",
            provider="openai",
            model="gpt-4.1-mini",
            credential_ref=CredentialRef("OPENAI_API_KEY"),
            max_cost_per_turn=Decimal("0.000001"),
        )
    )
    client = ScriptedLLMClient(
        [{"type": "final_answer", "content": "must not run"}]
    )
    client.provider = "openai"
    client.model = "gpt-4.1-mini"
    output: list[str] = []
    errors: list[str] = []
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "configured")

    exit_code = main(
        [
            "--model-profile",
            "budgeted",
            "exec",
            "hello",
            "--json",
        ],
        llm_client_factory=lambda _config: client,
        output_func=output.append,
        error_func=errors.append,
    )

    assert exit_code == 1
    assert errors == []
    payload = json.loads(output[0])
    assert payload["status"] == "budget_exhausted"
    assert payload["error_details"]["category"] == "budget_exhausted"
    assert payload["error_details"]["details"]["failure_kind"] == "budget_exhausted"
    assert client.remaining == 1
