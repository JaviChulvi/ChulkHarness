"""CLI and public-contract coverage for child-task orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from chulk import (
    ChildTask,
    ChildTaskChangedPayload,
    ChildTaskLineage,
    ChildTaskSpec,
    ChildTaskStatus,
    ChildTaskStore,
    DelegationService,
    EventName,
    TaskSupervisor,
)
from chulk.cli.children import run_child_command
from chulk.cli.parser import build_parser
from chulk.config import load_config
from chulk.main import main


NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _task(task_id: str = "child-cli") -> ChildTask:
    return ChildTask(
        id=task_id,
        profile_id="default",
        spec=ChildTaskSpec(instruction="Inspect the CLI surface."),
        lineage=ChildTaskLineage(),
        created_at=NOW,
        updated_at=NOW,
    )


def test_public_api_exports_child_contract_and_typed_events() -> None:
    payload = ChildTaskChangedPayload(
        task_id="child-1",
        status="running",
        revision=2,
        action="attempt_started",
        attempt_id="attempt-1",
    )

    assert isinstance(ChildTaskStore, type)
    assert isinstance(DelegationService, type)
    assert isinstance(TaskSupervisor, type)
    assert payload.task_id == "child-1"
    assert EventName.CHILD_TASK_CREATED.value == "child_task.created"
    assert (
        EventName.CHILD_TASK_DELIVERY_CHANGED.value
        == "child_task.delivery.changed"
    )


def test_child_parser_exposes_revisioned_operator_actions() -> None:
    parser = build_parser()

    cancel = parser.parse_args(
        [
            "child",
            "cancel",
            "child-1",
            "--revision",
            "3",
            "--reason",
            "Stop now",
            "--json",
        ]
    )
    listing = parser.parse_args(
        [
            "child",
            "list",
            "--status",
            "unknown",
            "--parent",
            "parent-1",
        ]
    )

    assert cancel.child_command == "cancel"
    assert cancel.revision == 3
    assert cancel.reason == "Stop now"
    assert listing.status == "unknown"
    assert listing.parent == "parent-1"


def test_child_cli_inspects_cancels_and_rejects_stale_revision(tmp_path) -> None:
    service = DelegationService(ChildTaskStore(tmp_path / "store.sqlite"))
    task = service.store.create(_task())
    output: list[str] = []
    errors: list[str] = []

    assert (
        run_child_command(
            "inspect",
            service=service,
            task_id=task.id,
            json_output=True,
            output_func=output.append,
            error_func=errors.append,
        )
        == 0
    )
    inspected = json.loads(output.pop())
    assert inspected["task"]["id"] == task.id
    assert inspected["events"][0]["kind"] == "child.created"

    assert (
        run_child_command(
            "cancel",
            service=service,
            task_id=task.id,
            expected_revision=task.revision,
            actor="owner",
            json_output=True,
            output_func=output.append,
            error_func=errors.append,
        )
        == 0
    )
    cancelled = json.loads(output.pop())
    assert cancelled["task"]["status"] == ChildTaskStatus.CANCELLED.value

    assert (
        run_child_command(
            "cancel",
            service=service,
            task_id=task.id,
            expected_revision=task.revision,
            actor="stale-owner",
            json_output=True,
            output_func=output.append,
            error_func=errors.append,
        )
        == 1
    )
    assert "revision conflict" in json.loads(output.pop())["error"]


def test_main_routes_child_list_to_selected_profile_store(
    tmp_path,
    monkeypatch,
) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    task = ChildTaskStore(config.store_path).create(_task("visible-child"))
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    output: list[str] = []
    errors: list[str] = []

    exit_code = main(
        ["child", "list", "--json"],
        output_func=output.append,
        error_func=errors.append,
    )

    assert exit_code == 0
    assert errors == []
    assert json.loads(output[0])["tasks"][0]["id"] == task.id
