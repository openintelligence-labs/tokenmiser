//! Canonical per-model pricing table.

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

/// Backed by `pricing.json` at deploy time; the canonical table is hard-coded
/// so the build stays self-contained.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PricingTable {
    pub models: HashMap<String, ModelPricing>,
}

impl PricingTable {
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

    /// True when this model executes on the operator's own hardware.
    ///
    /// Ollama Cloud tags are not free: they look local (same daemon, same API,
    /// an `ollama:` prefix) but generate tokens on a paid account. See
    /// [`Self::is_ollama_cloud`].
    pub fn is_free(model: &str) -> bool {
        if Self::is_ollama_cloud(model) {
            return false;
        }
        model.starts_with("ollama:") || model == "local"
    }

    /// True for an Ollama model whose tag ends in `-cloud`, which executes
    /// remotely on a paid account.
    ///
    /// Matching is on the tag (after the last `:`), so a local model merely
    /// named `cloudy-llama:7b` is unaffected. Case-insensitive, as Ollama tags
    /// are.
    pub fn is_ollama_cloud(model: &str) -> bool {
        let stripped = model.strip_prefix("ollama:").unwrap_or(model);
        let tag = stripped.rsplit(':').next().unwrap_or_default();
        tag.len() > "-cloud".len() && tag.to_ascii_lowercase().ends_with("-cloud")
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
        // $5/M input + $25/M output over 1k + 500 tokens.
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

    #[test]
    fn ollama_cloud_tags_are_not_free() {
        for m in [
            "gpt-oss:20b-cloud",
            "ollama:gpt-oss:20b-cloud",
            "ollama:deepseek-v3.1:671b-cloud",
            "ollama:qwen3-coder:480b-cloud",
            "GPT-OSS:120B-CLOUD",
        ] {
            assert!(PricingTable::is_ollama_cloud(m), "{m} must be cloud");
            assert!(!PricingTable::is_free(m), "{m} must not be free");
        }
    }

    #[test]
    fn local_models_named_cloud_are_still_free() {
        for m in [
            "ollama:llama3.2",
            "ollama:qwen2.5:7b",
            "ollama:cloudy-llama:7b",
            "ollama:nimbus-cloud-chat:7b",
            "ollama:cloud",
            "local",
        ] {
            assert!(!PricingTable::is_ollama_cloud(m), "{m} must not be cloud");
        }
        assert!(PricingTable::is_free("ollama:cloudy-llama:7b"));
        assert!(PricingTable::is_free("ollama:qwen2.5:7b"));
    }

    #[test]
    fn bare_cloud_tag_is_not_treated_as_cloud() {
        assert!(!PricingTable::is_ollama_cloud("ollama:model:-cloud"));
    }
}
