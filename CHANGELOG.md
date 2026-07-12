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
- Added native async model and structured-action requests for every built-in
  provider transport, including async fallback chains and cancellation
  propagation.
- Added deterministic offline evaluation helpers, versioned trace envelopes,
  trace replay, and provider-free regression coverage.
- Added plan-step retry accounting and tool-attempt metadata to runtime state
  and traces.
- Added MIT licensing for repository and package consumers.
- Added a security policy for responsible vulnerability reporting.
- Added GitHub Actions CI for Python 3.11, 3.12, and 3.13.
- Added clean-wheel install validation for public imports, packaged presets,
  bundled skills, runtime defaults, and example imports.

### Changed

- Centralized validated model-action parsing inside the LLM boundary and kept
  synchronous custom clients compatible through explicit adapter paths.
- Improved fallback behavior with shared context budgets, duplicate-request
  prevention, and normalized timeout, connection, rate-limit, and server
  failure classification.
- Extended CI type checking from the external SDK fixture to the package
  implementation and retained the original custom-provider factory contract.
- Extended `chulk doctor` to validate requirements for every primary and
  fallback provider.

### Security

- Reject credential-like content before durable memory storage.
- Deny model-facing reads of secret files, Git and trace data, SQLite state,
  and private key material by default.
- Bound shell output while it is produced, terminate timed-out process groups,
  and expose an explicit host-owned execution policy and containment gate.
- Sanitize provider base URLs in `--show-config` so user information, query
  parameters, and fragments are never printed.

### Documented

- Documented all provider names, optional dependencies, exact credential alias
  precedence, explicit-model requirements, and Bedrock endpoint requirements.
- Clarified that provider tests are offline and live account validation remains
  the responsibility of the application owner.
- Documented that `chulkharness` is the install/package name and `chulk` is the
  import package and CLI command.
