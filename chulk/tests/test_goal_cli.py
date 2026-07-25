"""CLI coverage for durable goal promotion and revision-safe operations."""

from __future__ import annotations

import json

from chulk.cli.goals import run_goal_command
from chulk.cli.parser import build_parser
from chulk.config import load_config
from chulk.core.state import Plan, PlanStep, TurnState
from chulk.goals import GoalService, GoalStep, GoalStore
from chulk.main import main
from chulk.sessions import SQLiteSessionStore
from chulk.usage import RunBudget


def _seed_plan(store: SQLiteSessionStore) -> None:
    store.create_conversation(
        "conversation-1",
        provider="fake",
        model="fake",
        metadata={"profile_id": "default"},
    )
    turn = TurnState(
        user_message="Make this durable",
        turn_id="turn-1",
        active_plan=Plan(
            summary="Ship the durable change",
            steps=[
                PlanStep(
                    id="build",
                    title="Build",
                    description="Build the change",
                    acceptance_criteria=["The build passes"],
                )
            ],
        ),
    )
    store.save_turn_snapshot("conversation-1", turn.to_dict())


def test_goal_parser_exposes_promotion_and_revisioned_mutations() -> None:
    parser = build_parser()

    promote = parser.parse_args(
        [
            "goal",
            "promote",
            "conversation-1",
            "--max-model-calls",
            "4",
            "--max-cost",
            "1.25",
            "--json",
        ]
    )
    steer = parser.parse_args(
        [
            "goal",
            "steer",
            "goal-1",
            "Prefer",
            "the",
            "safe",
            "path",
            "--revision",
            "3",
        ]
    )

    assert promote.goal_command == "promote"
    assert promote.max_model_calls == 4
    assert promote.max_cost == "1.25"
    assert steer.instruction == ["Prefer", "the", "safe", "path"]
    assert steer.revision == 3


def test_goal_cli_promotes_inspects_and_rejects_stale_revision(tmp_path) -> None:
    db_path = tmp_path / "store.sqlite"
    sessions = SQLiteSessionStore(db_path)
    _seed_plan(sessions)
    service = GoalService(GoalStore(db_path))
    output: list[str] = []
    errors: list[str] = []

    assert (
        run_goal_command(
            "promote",
            service=service,
            session_store=sessions,
            conversation_id="conversation-1",
            max_model_calls=3,
            json_output=True,
            output_func=output.append,
            error_func=errors.append,
        )
        == 0
    )
    promoted = json.loads(output.pop())
    goal_id = promoted["goal"]["id"]
    assert promoted["goal"]["source_turn_id"] == "turn-1"
    assert promoted["goal"]["budget"]["scope"] == "goal"

    assert (
        run_goal_command(
            "approve",
            service=service,
            session_store=sessions,
            goal_id=goal_id,
            expected_revision=0,
            actor="owner",
            json_output=True,
            output_func=output.append,
            error_func=errors.append,
        )
        == 0
    )
    assert json.loads(output.pop())["goal"]["revision"] == 1

    assert (
        run_goal_command(
            "approve",
            service=service,
            session_store=sessions,
            goal_id=goal_id,
            expected_revision=0,
            actor="stale-owner",
            json_output=True,
            output_func=output.append,
            error_func=errors.append,
        )
        == 1
    )
    assert "revision conflict" in json.loads(output.pop())["error"]


def test_main_routes_goal_list_to_selected_profile_store(tmp_path, monkeypatch) -> None:
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})
    service = GoalService(GoalStore(config.store_path))
    goal = service.create(
        title="Visible goal",
        acceptance_criteria=("It is visible",),
        steps=(
            GoalStep(
                id="inspect",
                title="Inspect",
                description="Inspect it",
                acceptance_criterion_ids=("criterion-1",),
            ),
        ),
        budget=RunBudget(),
    )
    monkeypatch.setenv("CHULK_PROJECT_ROOT", str(tmp_path))
    output: list[str] = []
    errors: list[str] = []

    exit_code = main(
        ["goal", "list", "--json"],
        output_func=output.append,
        error_func=errors.append,
    )

    assert exit_code == 0
    assert errors == []
    payload = json.loads(output[0])
    assert payload["goals"][0]["id"] == goal.id
