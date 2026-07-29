//! Routing policy: maps Difficulty → concrete (provider, model) target.
//!
//! v0.4 keeps the policy purely declarative (struct + YAML). The Rhai DSL
//! (architecture §11.5 / v0.9 milestone) lands later and slots into the
//! same `RoutingPolicy::choose()` seam.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Difficulty;

/// A concrete routing target — provider name and the actual model id to send.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTarget {
    pub provider: String,
    pub model: String,
}

impl RoutingTarget {
    /// Pass through the model the user explicitly requested. Provider field
    /// stays empty — the registry's own resolution (heuristic, alias, prefix)
    /// takes over.
    pub fn passthrough(model: &str) -> Self {
        Self {
            provider: String::new(),
            model: model.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingPolicy {
    pub tiers: HashMap<Difficulty, RoutingTarget>,
    /// Counterfactual frontier model used for "saved $X" accounting on
    /// every routed-cheaper request.
    pub frontier_model: String,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        let mut tiers = HashMap::new();
        tiers.insert(
            Difficulty::Easy,
            RoutingTarget {
                provider: "ollama".into(),
                // qwen2.5:7b is the most commonly-installed Ollama model in
                // late-2026 surveys; users can override via YAML policy file.
                // The auto-detect path in main.rs may update this with a
                // model that's actually loaded at startup (v0.6 work).
                model: "ollama:qwen2.5:7b".into(),
            },
        );
        tiers.insert(
            Difficulty::Medium,
            RoutingTarget {
                provider: "anthropic".into(),
                model: "claude-haiku-4-5".into(),
            },
        );
        tiers.insert(
            Difficulty::Hard,
            RoutingTarget {
                provider: "anthropic".into(),
                model: "claude-opus-4-7".into(),
            },
        );
        Self {
            tiers,
            frontier_model: "claude-opus-4-7".into(),
        }
    }
}

impl RoutingPolicy {
    pub fn choose(&self, d: Difficulty) -> RoutingTarget {
        self.tiers
            .get(&d)
            .cloned()
            .unwrap_or_else(|| RoutingTarget {
                provider: "anthropic".into(),
                model: self.frontier_model.clone(),
            })
    }

    pub fn frontier_for(&self, _d: Difficulty) -> Option<String> {
        Some(self.frontier_model.clone())
    }
}
