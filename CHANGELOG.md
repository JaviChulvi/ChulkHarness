# Changelog

All notable changes to ChulkHarness are documented in this file.

This project follows a lightweight, human-maintained changelog while the public
API is still pre-1.0.

## Unreleased

### Added

- Added native Anthropic and Gemini providers plus OpenAI-compatible,
  OpenRouter, and AWS Bedrock provider adapters.
- Added an all-provider `providers` installation extra while retaining the
  individual `openai`, `anthropic`, and `gemini` extras.
- Added explicit provider profiles, shared connection binding, capability
  metadata, and offline provider conformance tests with injected fake clients.
- Added MIT licensing for repository and package consumers.
- Added a security policy for responsible vulnerability reporting.
- Added GitHub Actions CI for Python 3.11, 3.12, and 3.13.
- Added clean-wheel install validation for public imports, packaged presets,
  bundled skills, runtime defaults, and example imports.

### Documented

- Documented all provider names, optional dependencies, exact credential alias
  precedence, explicit-model requirements, and Bedrock endpoint requirements.
- Clarified that provider tests are offline and live account validation remains
  the responsibility of the application owner.
- Documented that `chulkharness` is the install/package name and `chulk` is the
  import package and CLI command.
