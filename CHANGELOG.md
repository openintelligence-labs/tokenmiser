# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-07-29

### Fixed

- Dashboard is now fully offline: htmx 2.0.4 is vendored into the binary
  (integrity-verified against the official SRI hash, byte-identical across
  unpkg and jsdelivr) and served from the local `/assets/htmx.js` route
  instead of being fetched from a CDN. Regression tests assert the dashboard
  HTML and live fragment contain no external http(s) URLs.

## [0.5.0] - 2026-07-29

First feature release. Rewrites the v0.1 single-crate skeleton as a 9-crate
Cargo workspace with a Pingora-based ingress.

### Added

- **OpenAI-compatible proxy** (`tokenmiser-proxy`): `/v1/chat/completions`
  on a Pingora ingress (`:8443`), streaming (SSE) and non-streaming, with
  per-response audit headers (`x-tokenmiser-cache`, `x-tokenmiser-tier`,
  `x-tokenmiser-difficulty`, `x-tokenmiser-routed-to`).
- **Tiered router** (`tokenmiser-router`): Tier 0 heuristic difficulty
  classifier, Tier 1 semantic classifier (embedding exemplars), Tier 2
  speculative cascade, plus a Rhai **policy DSL** with a
  `tokenmiser policy test` replay command for offline policy evaluation.
- **Dual-layer semantic cache** (`tokenmiser-cache`): L1 exact-match LRU and
  L2 semantic cache (bge-small-en-v1.5 via fastembed, cosine threshold 0.87,
  precision-tuned on a paraphrase benchmark), scoped per tenant.
- **Provider adapters** (`tokenmiser-providers`): OpenAI, Anthropic, and
  Ollama clients behind one `Provider` trait, with a registry that resolves
  model ids, static aliases, and provider prefixes (`ollama:...`).
- **Local-first auto-detection**: probes a local Ollama at startup and routes
  Easy-difficulty traffic to an installed local model with zero API keys and
  zero config (reasoning-mode and embedding-only models excluded).
- **Cost ledger** (`tokenmiser-cost` + `pricing/`): real-time USD
  spent/saved/counterfactual accounting from a canonical, source-annotated
  `pricing.json`.
- **Live dashboard** (`tokenmiser-proxy`): htmx single-page UI at `/` and a
  JSON `/stats` endpoint with cache hit rates and cost totals.
- **Quality judge** (`tokenmiser-quality`): shadow-mode A/B sampling with
  LLM-as-judge verdicts and a per-segment win-rate auto-gate (enabled when a
  frontier API key is present; soft-disabled otherwise).
- **MCP budget gateway** (`tokenmiser-mcp`): per-agent/per-tool spend caps
  enforced through `/v1/mcp/tools/call` and managed via `/v1/mcp/budgets`.
- **Config** (`tokenmiser-config`): YAML config via `TOKENMISER_CONFIG` with
  sensible zero-config defaults; workspace-wide release profile (thin LTO,
  stripped binaries).

### Changed

- Restructured from a single crate into a 9-crate workspace:
  `tokenmiser` (bin), `-proxy`, `-router`, `-cache`, `-cost`, `-providers`,
  `-config`, `-quality`, `-mcp`.
- Proxy ingress moved from Axum to Pingora; Axum retained for the admin
  surface.

### Security

- No telemetry of any kind; all routing decisions are logged locally for
  auditability.

## [0.0.1] - 2026-04-26

### Added

- Initial public release: difficulty classifier + routing logic, cost ledger
  with savings tracking, Axum HTTP server skeleton.

[0.5.1]: https://github.com/openintelligence-labs/tokenmiser/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/openintelligence-labs/tokenmiser/compare/v0.0.1...v0.5.0
[0.0.1]: https://github.com/openintelligence-labs/tokenmiser/releases/tag/v0.0.1
