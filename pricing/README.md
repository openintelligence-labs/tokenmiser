# tokenmiser/pricing

Canonical per-model LLM pricing JSON consumed by [TokenMiser](https://github.com/openintelligence-labs/tokenmiser)
and any other tool that wants one source of truth.

## Consume

```bash
curl -fsSL https://raw.githubusercontent.com/openintelligence-labs/tokenmiser-pricing/main/pricing.json
```

Programmatic (Rust):
```rust
let p: serde_json::Value = serde_json::from_str(include_str!("../pricing.json"))?;
let opus_input = p["models"]["claude-opus-4-7"]["input"].as_f64().unwrap();
```

The schema is published as [`schema.json`](./schema.json) (JSON Schema
draft-07). Validate updates locally with `ajv validate -s schema.json -d pricing.json`.

## Contribute

PRs welcome. See [`SOURCES.md`](./SOURCES.md) for upstream pages and the
update policy. One PR per provider per change; cite the upstream URL.

## License

CC0 — fully public domain. Take it, fork it, mirror it.
