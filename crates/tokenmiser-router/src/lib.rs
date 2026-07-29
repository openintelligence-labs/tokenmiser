//! Routing: classify prompt difficulty, pick a model.
//!
//! Architecture §3 specifies a 4-tier router. v0.4 ships Tier 0 + Tier 1.
//! - **Tier 0** (`tier0.rs`): pure-Rust heuristic, sub-microsecond. Cheap
//!   length/keyword/role/JSON-mode signals. Confidently classifies ~30% of
//!   traffic.
//! - **Tier 1** (`tier1.rs`): exemplar-based semantic classifier using the
//!   same bge-small-en-v1.5 embedder the L2 cache loaded. We embed a
//!   curated set of difficulty exemplars at startup; at request time we
//!   embed the prompt and pick the nearest cluster. Drop-in slot for
//!   future RouteLLM ONNX weights.
//!
//! v0.5 (speculative cascade) and v0.6+ (unified Cascade Routing per
//! architecture §14.1) hang off the same `RouteDecision` shape.

use serde::{Deserialize, Serialize};
use tokenmiser_providers::ChatRequest;

pub mod dsl;
pub mod policy;
pub mod replay;
pub mod tier0;
pub mod tier1;
pub mod tier2;

pub use dsl::{PolicyEngine, RequestView};
pub use policy::{RoutingPolicy, RoutingTarget};
pub use replay::{replay, ReplayResult};
pub use tier0::tier0_difficulty;
pub use tier1::Tier1Classifier;
pub use tier2::{should_escalate, CascadeConfig, EscalateDecision};

/// Coarse difficulty class used by the policy to pick a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

/// Final routing decision: what model to call, why we picked it, and
/// (if relevant) which model to use as the counterfactual for cost
/// accounting + shadow A/B (v0.8).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    pub target: RoutingTarget,
    pub difficulty: Difficulty,
    pub tier: RouteTier,
    /// Architecture §14.3 reasoning trace — populated in v1.1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Model we'd have called if we routed everything to the frontier;
    /// drives the "saved $X" counterfactual ledger.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counterfactual_model: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouteTier {
    /// Caller explicitly named a model — no classification done.
    Explicit,
    /// Tier 0 heuristic was sufficient.
    Heuristic,
    /// Tier 1 semantic classifier was used.
    Semantic,
}

/// The orchestrator that runs Tier 0 → Tier 1 → policy in order.
pub struct Router {
    policy: RoutingPolicy,
    tier1: Option<Tier1Classifier>,
}

impl Router {
    pub fn new(policy: RoutingPolicy, tier1: Option<Tier1Classifier>) -> Self {
        Self { policy, tier1 }
    }

    /// Look up the policy target for a difficulty band (used by the v0.5
    /// cascade path to pick its cheap/frontier endpoints).
    pub fn policy_target(&self, d: Difficulty) -> RoutingTarget {
        self.policy.choose(d)
    }

    /// Decide where to send `req`. Honors caller intent if the requested
    /// model is unambiguous (anything other than `auto` / `tokenmiser:auto`).
    pub fn decide(&self, req: &ChatRequest) -> RouteDecision {
        let requested = req.model.as_str();
        let auto = requested == "auto" || requested == "tokenmiser:auto";

        if !auto {
            // Caller picked a specific model; respect it but still record
            // difficulty so /stats remains rich.
            let difficulty = tier0_difficulty(req);
            return RouteDecision {
                target: RoutingTarget::passthrough(requested),
                difficulty,
                tier: RouteTier::Explicit,
                reasoning: None,
                counterfactual_model: self.policy.frontier_for(Difficulty::Hard),
            };
        }

        // Auto-routing: Tier 0 first, then Tier 1 if Tier 0 isn't decisive.
        let t0 = tier0_difficulty(req);
        let (difficulty, tier) = match (t0, &self.tier1) {
            (Difficulty::Medium, Some(t1)) => (t1.classify(req), RouteTier::Semantic),
            _ => (t0, RouteTier::Heuristic),
        };

        let target = self.policy.choose(difficulty);
        let counterfactual = self.policy.frontier_for(Difficulty::Hard);

        RouteDecision {
            target,
            difficulty,
            tier,
            reasoning: None,
            counterfactual_model: counterfactual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenmiser_providers::ChatMessage;

    fn user(model: &str, s: &str) -> ChatRequest {
        ChatRequest {
            model: model.into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(s.into()),
                extra: Default::default(),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn explicit_model_request_is_passthrough() {
        let router = Router::new(RoutingPolicy::default(), None);
        let d = router.decide(&user("gpt-5", "anything"));
        assert!(matches!(d.tier, RouteTier::Explicit));
        assert_eq!(d.target.model, "gpt-5");
    }

    #[test]
    fn auto_easy_routes_to_local() {
        let router = Router::new(RoutingPolicy::default(), None);
        let d = router.decide(&user("auto", "what is 2+2?"));
        assert_eq!(d.difficulty, Difficulty::Easy);
        // default policy maps Easy → ollama:llama3.2
        assert!(d.target.model.contains("llama") || d.target.provider == "ollama");
    }

    #[test]
    fn auto_hard_routes_to_frontier() {
        let router = Router::new(RoutingPolicy::default(), None);
        let d = router.decide(&user("auto", "refactor this auth middleware to JWT"));
        assert_eq!(d.difficulty, Difficulty::Hard);
        assert!(d.target.model.contains("opus") || d.target.model.contains("gpt"));
    }
}
