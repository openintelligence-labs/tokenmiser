//! Canonical pricing table — the April 2026 snapshot from
//! `docs/ARCHITECTURE.md` §6. The intent is to publish this as
//! `tokenmiser/pricing` so other gateways sync from us (architecture §11.4 moat).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-million-token pricing for a single model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelPricing {
    /// USD per 1M input tokens.
    pub input_per_million: f64,
    /// USD per 1M output tokens.
    pub output_per_million: f64,
}

impl ModelPricing {
    pub const fn new(input: f64, output: f64) -> Self {
        Self {
            input_per_million: input,
            output_per_million: output,
        }
    }

    /// Compute the cost of a single request in USD.
    pub fn cost_usd(&self, input_tokens: u64, output_tokens: u64) -> f64 {
        (input_tokens as f64 / 1_000_000.0) * self.input_per_million
            + (output_tokens as f64 / 1_000_000.0) * self.output_per_million
    }
}

/// In-memory canonical pricing table. Backed by `pricing.json` at deploy time;
/// hard-coded here so the v0.1 build is self-contained.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PricingTable {
    pub models: HashMap<String, ModelPricing>,
}

impl PricingTable {
    /// April 2026 canonical pricing from architecture doc §6.
    pub fn canonical() -> Self {
        let mut models = HashMap::new();

        // Anthropic
        models.insert("claude-opus-4-7".into(), ModelPricing::new(5.00, 25.00));
        models.insert("claude-sonnet-4-6".into(), ModelPricing::new(3.00, 15.00));
        models.insert("claude-haiku-4-5".into(), ModelPricing::new(1.00, 5.00));

        // OpenAI
        models.insert("gpt-5".into(), ModelPricing::new(1.25, 10.00));
        models.insert("gpt-5.4".into(), ModelPricing::new(2.50, 15.00));
        models.insert("gpt-4o-mini".into(), ModelPricing::new(0.15, 0.60));

        // Google
        models.insert("gemini-2.5-pro".into(), ModelPricing::new(1.00, 10.00));
        models.insert("gemini-2.5-flash".into(), ModelPricing::new(0.30, 2.50));

        // DeepSeek
        models.insert("deepseek-r1".into(), ModelPricing::new(0.29, 0.29));

        // Inference-as-a-service
        models.insert("cerebras-llama3.1-8b".into(), ModelPricing::new(0.10, 0.10));
        models.insert(
            "deepinfra-llama3.1-8b".into(),
            ModelPricing::new(0.03, 0.05),
        );

        Self { models }
    }

    pub fn get(&self, model: &str) -> Option<&ModelPricing> {
        self.models.get(model)
    }

    /// True if this model is free at the point of use (i.e. local Ollama).
    pub fn is_free(model: &str) -> bool {
        model.starts_with("ollama:") || model == "local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opus_pricing_matches_arch_doc() {
        let p = PricingTable::canonical();
        let opus = p.get("claude-opus-4-7").expect("opus 4.7 present");
        assert!((opus.input_per_million - 5.00).abs() < 1e-9);
        assert!((opus.output_per_million - 25.00).abs() < 1e-9);
    }

    #[test]
    fn cost_calculation_is_correct() {
        // Opus 4.7: $5/M input, $25/M output.
        // 1k input + 500 output → $0.005 + $0.0125 = $0.0175
        let p = ModelPricing::new(5.00, 25.00);
        let cost = p.cost_usd(1_000, 500);
        assert!((cost - 0.0175).abs() < 1e-9);
    }

    #[test]
    fn ollama_models_are_free() {
        assert!(PricingTable::is_free("ollama:llama3.2"));
        assert!(PricingTable::is_free("local"));
        assert!(!PricingTable::is_free("gpt-5"));
    }
}
