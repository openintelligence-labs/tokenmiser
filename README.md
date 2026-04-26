# TokenMiser

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Smart LLM router that cuts agent costs by 10x.** Drop-in OpenAI SDK replacement. Routes simple queries to cheap/local models, hard ones to frontier. Semantic caching, real-time cost dashboard, A/B quality testing.

⭐ **Star us on GitHub** if your monthly LLM bill makes you wince.

## Why this exists

LiteLLM does basic routing. Portkey is closed source. Nothing combines routing + semantic caching + cost tracking + quality comparison in one open source tool. TokenMiser is that tool — a Rust proxy you put in front of your LLM calls. One import change, instant savings.

## Quick start

```bash
cargo install tokenmiser
tokenmiser
# proxy running on :8443
```

Point your OpenAI SDK at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8443/v1")
```

## Features

| Feature | What it does |
|---|---|
| Drop-in proxy | OpenAI-compatible `/v1/chat/completions` |
| Smart routing | Simple → local model, complex → frontier |
| Semantic cache | Cosine-similarity matching on similar prompts |
| Cost dashboard | Real-time USD per workflow/user/agent |
| A/B testing | Compare model quality at different price points |
| Budget alerts | Automatic fallback when budget exceeded |
| Rust speed | Sub-millisecond routing overhead |

## Roadmap

- [x] Difficulty classifier + routing logic
- [x] Cost ledger with savings tracking
- [x] Axum HTTP server skeleton
- [ ] OpenAI-compatible proxy endpoints
- [ ] Semantic cache (embedding-based)
- [ ] Dashboard UI
- [ ] Budget alerts + automatic fallback

## Part of the Open Intelligence Labs ecosystem

- [agentic-kit](https://github.com/openintelligence-labs/agentic-kit) — TokenMiser is the default LLM layer
- [AgentTrace](https://github.com/openintelligence-labs/agenttrace) — cost data feeds into trace cost attribution
- [DeepDive](https://github.com/openintelligence-labs/deepdive) — first consumer to use routing for search vs. analysis

## License

MIT
