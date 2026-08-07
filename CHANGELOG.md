# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The cost ledger truncated every request's cost to whole micro-dollars, so
  spend was systematically **under-reported**. A real `gpt-4o-mini` call of 14
  prompt + 1 completion tokens costs $0.0000027 and was recorded as
  $0.000002 — a 26% under-count. The error was per-request rather than
  averaging out, so it compounded across a workload, and it was always
  downward: the one direction a budget enforcer must not err in, since it
  lets real spend run past a configured cap. Requests cheaper than
  $0.000001 recorded as exactly $0.00, making arbitrarily many of them look
  free. Spend is now accumulated in pico-dollars (USD × 1e12) and rounded to
  nearest, which also fixes the matching under-count in
  `counterfactual_usd`/`saved_usd`. Found by comparing `/stats` against the
  provider's own reported `usage` on a live OpenAI call.

### Changed

- README documents which provider adapters are verified against a live API
  (OpenAI, Ollama) and which are covered by unit tests only (Anthropic).

## [0.6.0] - 2026-07-31

### Added

- Streamed responses are cached. `StreamAccumulator` reassembles SSE chunks
  (UTF-8 safe across packet splits; LF, CRLF and CR terminators per WHATWG
  HTML 9.2.5) and writes L1/L2 on clean completion only.
- Concurrent identical requests are coalesced: the first miss computes
  upstream, the rest await its result and are served as cache hits
  (`x-tokenmiser-cache: coalesced`). A cold burst of 120 identical requests
  drops from 120 upstream calls to 61.
- Budget limits. Daily and total thresholds surface via `/stats`, an
  `x-tokenmiser-budget` header and a structured log line. Warn-only by
  default; enforce mode returns 402 for paid routes while cache hits and
  local models always pass.
- `GET /v1/models`, so `client.models.list()` works.
- `security.allowed_origins` for browser apps that need cross-origin access.

### Fixed

- Cache hits on `stream: true` returned a JSON body instead of a stream,
  breaking OpenAI SDK clients. They now replay as simulated chunk streams.
- The semantic cache could answer a different question: `Multiply 3 by 11`
  matched a cached `Add 25 and 30` at cosine 0.876, and the hit seeded L1 so
  the wrong answer persisted. Candidates whose number literals differ are now
  skipped before scoring. Raising the threshold cannot fix this — genuine
  paraphrases score 0.92-0.97, above the false positive.
- A `response_format: json_object` request could be served cached prose.
  Answer-shaping parameters are part of the semantic entry fingerprint.
- The exact cache key ignored `max_tokens` and `top_p`, so a truncated
  response could be replayed to a caller asking for more. Transport-only
  fields stay out of the key so stream and non-stream callers share entries.
- The SSE parser only recognised LF terminators; a spec-legal CRLF provider
  produced no cache writes and no usage accounting.
- The keepalive probe could splice a comment into a partially forwarded
  event, corrupting the client's parse.
- In-band `data: {"error": ...}` events no longer cache partial content.
- The proxy defaulted to `0.0.0.0` with no authentication, exposing an
  endpoint that spends API budget to the local network. It binds `127.0.0.1`
  and warns on any non-loopback bind.
- Browser pages could drive completions via CORS simple requests. Requests
  carrying `Sec-Fetch-Site: cross-site` or a disallowed `Origin` are
  rejected; CLI and SDK callers are unaffected.
- Request bodies are capped at 8 MiB. Twelve concurrent uploads previously
  drove RSS from 21 MB to 322 MB.
- The tenant map is bounded at 256 entries with LRU eviction and tenant ids
  are sanitised; the header is client-supplied and reached cache keys and
  logs.
- Ollama models tagged `*-cloud` run remotely and are no longer reported as
  free. They count as remote requests with `unpriced_requests` in `/stats`,
  since per-token rates are not published.
- `HEAD` requests returned 404 on every route (RFC 9110 9.3.2).
- Test policy files collided between parallel threads within one clock tick.

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
