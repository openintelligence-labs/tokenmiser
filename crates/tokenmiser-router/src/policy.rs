//! Maps a `Difficulty` to a concrete (provider, model) target.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Difficulty;

/// A provider name and the model id to send it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTarget {
    pub provider: String,
    pub model: String,
}

impl RoutingTarget {
    /// Pass the requested model through with an empty provider, leaving
    /// resolution to the registry.
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
    /// Frontier model used for counterfactual savings accounting.
    pub frontier_model: String,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        let mut tiers = HashMap::new();
        tiers.insert(
            Difficulty::Easy,
            RoutingTarget {
                provider: "ollama".into(),
                // Overridable via the YAML policy; startup auto-detection may
                // replace this with a model that is actually installed.
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
