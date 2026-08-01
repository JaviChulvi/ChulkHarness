"""Shared test isolation from developer-local provider configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def stable_test_provider(monkeypatch):
    """Keep repository .env values from changing deterministic test defaults."""
    monkeypatch.setenv("CHULK_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CHULK_MODEL", "gpt-4.1-mini")
