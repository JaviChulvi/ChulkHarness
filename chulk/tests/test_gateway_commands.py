"""Shared command contract tests."""

from chulk.gateway import (
    SHARED_CHANNEL_COMMANDS,
    parse_channel_command,
    shared_command_help,
    shared_command_spec,
)


def test_shared_channel_commands_have_stable_unique_names() -> None:
    names = tuple(item.name for item in SHARED_CHANNEL_COMMANDS)

    assert names == (
        "new",
        "status",
        "stop",
        "model",
        "skills",
        "memory",
        "agents",
        "jobs",
    )
    assert len(names) == len(set(names))


def test_channel_command_parser_normalizes_adapter_mentions_and_arguments() -> None:
    assert parse_channel_command("hello") is None
    assert parse_channel_command("/") is None
    parsed = parse_channel_command("  /MODEL@ChulkBot   careful  ")

    assert parsed is not None
    assert parsed.name == "model"
    assert parsed.arguments == "careful"
    assert shared_command_spec("/model") is not None
    assert shared_command_spec("unknown") is None


def test_shared_help_is_generated_from_the_registry() -> None:
    rendered = shared_command_help()

    for command in SHARED_CHANNEL_COMMANDS:
        assert command.usage in rendered
        assert command.description in rendered
