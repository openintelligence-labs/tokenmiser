//! Classify prompt difficulty and pick a model.
//!
//! Tier 0 (`tier0.rs`) is a sub-microsecond heuristic over length, keyword,
//! role and JSON-mode signals. Tier 1 (`tier1.rs`) is an exemplar-based
//! semantic classifier sharing the L2 cache's bge-small embedder, used when
//! Tier 0 is not decisive.

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

/// What model to call, why, and the counterfactual model used for cost
/// accounting and shadow A/B.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    pub target: RoutingTarget,
    pub difficulty: Difficulty,
    pub tier: RouteTier,
    /// Reasoning trace; not yet populated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// The model a frontier-only route would have called; drives the
    /// counterfactual savings ledger.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counterfactual_model: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RouteTier {
    /// Caller named a model; no classification ran.
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

    /// The policy target for a difficulty band.
    pub fn policy_target(&self, d: Difficulty) -> RoutingTarget {
        self.policy.choose(d)
    }

    /// Decide where to send `req`, honoring an explicitly named model.
    pub fn decide(&self, req: &ChatRequest) -> RouteDecision {
        let requested = req.model.as_str();
        let auto = requested == "auto" || requested == "tokenmiser:auto";

        if !auto {
            // Respect the caller's model but still classify, so /stats keeps
            // reporting difficulty.
            let difficulty = tier0_difficulty(req);
            return RouteDecision {
                target: RoutingTarget::passthrough(requested),
                difficulty,
                tier: RouteTier::Explicit,
                reasoning: None,
                counterfactual_model: self.policy.frontier_for(Difficulty::Hard),
            };
        }

        // Tier 0 first, Tier 1 only when Tier 0 is not decisive.
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
