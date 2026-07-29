# Sources

Canonical per-model pricing in [`pricing.json`](./pricing.json) tracks the
following upstream pages. Each weekly auto-PR cites which page changed.

| Provider | Source page |
|---|---|
| Anthropic | https://www.anthropic.com/pricing |
| OpenAI | https://openai.com/api/pricing/ |
| Google (Gemini) | https://ai.google.dev/pricing |
| DeepSeek | https://api-docs.deepseek.com/quick_start/pricing |
| Cerebras | https://inference-docs.cerebras.ai/pricing |
| DeepInfra | https://deepinfra.com/pricing |

## Update policy
- One PR per provider per pricing change, never a bulk PR.
- PR title format: `pricing: {provider} {model} {field} {old}→{new} ({YYYY-MM-DD})`.
- The bumped `version` field in `pricing.json` must match the calendar
  month of the most recent change.
- Reverts go through the same PR mechanism; we never silently downgrade.

## Why this exists
LiteLLM, Bifrost, and Helicone each maintain their own pricing tables and
each drifts independently. TokenMiser publishes one canonical source so
the ecosystem can sync from a single place. See architecture doc §6
and §11.4 for the moat rationale.
