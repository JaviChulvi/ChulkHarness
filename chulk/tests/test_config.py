"""Tests for configuration loading."""

from chulk.config import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_MAX_OBSERVATION_CHARS,
    DEFAULT_MAX_SKILL_CONTENT_CHARS,
    DEFAULT_MAX_SKILLS_PER_TURN,
    DEFAULT_MAX_TOOL_STDERR_CHARS,
    DEFAULT_MAX_TOOL_STDOUT_CHARS,
    DEFAULT_MODEL,
    DEFAULT_TRACE_MAX_PROMPT_CHARS,
    load_config,
)


def test_load_config_uses_defaults(tmp_path):
    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    assert config.project_root == tmp_path
    assert config.llm_provider == "openai"
    assert config.model == DEFAULT_MODEL
    assert config.llm_fallback_providers == ()
    assert config.openai_api_key is None
    assert config.deepseek_api_key is None
    assert config.history_limit == 20
    assert config.max_skills_per_turn == DEFAULT_MAX_SKILLS_PER_TURN
    assert config.max_skill_content_chars == DEFAULT_MAX_SKILL_CONTENT_CHARS
    assert config.trace_max_prompt_chars == DEFAULT_TRACE_MAX_PROMPT_CHARS
    assert config.max_observation_chars == DEFAULT_MAX_OBSERVATION_CHARS
    assert config.max_tool_stdout_chars == DEFAULT_MAX_TOOL_STDOUT_CHARS
    assert config.max_tool_stderr_chars == DEFAULT_MAX_TOOL_STDERR_CHARS


def test_load_config_reads_dotenv(tmp_path):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=dotenv-key",
                "CHULK_MODEL=dotenv-model",
                "CHULK_HISTORY_LIMIT=7",
                "CHULK_MAX_SKILLS_PER_TURN=2",
                "CHULK_MAX_SKILL_CONTENT_CHARS=800",
                "CHULK_TRACE_MAX_PROMPT_CHARS=1234",
                "CHULK_MAX_OBSERVATION_CHARS=900",
                "CHULK_MAX_TOOL_STDOUT_CHARS=700",
                "CHULK_MAX_TOOL_STDERR_CHARS=300",
                "DEEPSEEK_API_KEY=deepseek-key",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config({"CHULK_PROJECT_ROOT": str(tmp_path)})

    assert config.project_root == tmp_path
    assert config.openai_api_key == "dotenv-key"
    assert config.deepseek_api_key == "deepseek-key"
    assert config.model == "dotenv-model"
    assert config.history_limit == 7
    assert config.max_skills_per_turn == 2
    assert config.max_skill_content_chars == 800
    assert config.trace_max_prompt_chars == 1234
    assert config.max_observation_chars == 900
    assert config.max_tool_stdout_chars == 700
    assert config.max_tool_stderr_chars == 300


def test_environment_overrides_dotenv(tmp_path):
    (tmp_path / ".env").write_text("CHULK_MODEL=dotenv-model\n", encoding="utf-8")

    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_MODEL": "env-model",
        }
    )

    assert config.model == "env-model"


def test_invalid_integer_config_raises():
    try:
        load_config({"CHULK_HISTORY_LIMIT": "zero"})
    except ValueError as exc:
        assert "CHULK_HISTORY_LIMIT" in str(exc)
    else:
        raise AssertionError("Expected invalid integer config to fail")


def test_deepseek_provider_uses_deepseek_default_model(tmp_path):
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "deepseek-key",
        }
    )

    assert config.llm_provider == "deepseek"
    assert config.model == DEFAULT_DEEPSEEK_MODEL
    assert config.deepseek_api_key == "deepseek-key"


def test_load_config_parses_fallback_providers(tmp_path):
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_PROVIDER": "deepseek",
            "CHULK_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_API_KEY": "deepseek-key",
            "CHULK_LLM_FALLBACK_PROVIDERS": "openai:gpt-4.1-mini, deepseek:deepseek-v4-flash",
        }
    )

    assert [(item.provider, item.model) for item in config.llm_fallback_providers] == [
        ("openai", "gpt-4.1-mini"),
        ("deepseek", "deepseek-v4-flash"),
    ]


def test_load_config_uses_provider_default_for_fallback_without_model(tmp_path):
    config = load_config(
        {
            "CHULK_PROJECT_ROOT": str(tmp_path),
            "CHULK_LLM_FALLBACK_PROVIDERS": "deepseek",
        }
    )

    assert [(item.provider, item.model) for item in config.llm_fallback_providers] == [
        ("deepseek", DEFAULT_DEEPSEEK_MODEL),
    ]


def test_load_config_rejects_unknown_fallback_provider(tmp_path):
    try:
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_FALLBACK_PROVIDERS": "unknown",
            }
        )
    except ValueError as exc:
        assert "CHULK_LLM_FALLBACK_PROVIDERS" in str(exc)
    else:
        raise AssertionError("Expected invalid fallback provider config to fail")


def test_invalid_provider_config_raises(tmp_path):
    try:
        load_config(
            {
                "CHULK_PROJECT_ROOT": str(tmp_path),
                "CHULK_LLM_PROVIDER": "unknown",
            }
        )
    except ValueError as exc:
        assert "CHULK_LLM_PROVIDER" in str(exc)
    else:
        raise AssertionError("Expected invalid provider config to fail")
