use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRule {
    pub name: String,
    pub max_tokens_for_local: usize,
    pub local_model: String,
    pub frontier_model: String,
    pub frontier_provider: String,
}

impl Default for RoutingRule {
    fn default() -> Self {
        Self {
            name: "default".into(),
            max_tokens_for_local: 200,
            local_model: "llama3.2".into(),
            frontier_model: "gpt-4o-mini".into(),
            frontier_provider: "openai".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteDecision {
    Local,
    Frontier,
}

pub fn classify_difficulty(prompt: &str, rule: &RoutingRule) -> RouteDecision {
    let word_count = prompt.split_whitespace().count();
    if word_count <= rule.max_tokens_for_local {
        RouteDecision::Local
    } else {
        RouteDecision::Frontier
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CostLedger {
    pub total_usd: f64,
    pub local_requests: u64,
    pub frontier_requests: u64,
    pub cache_hits: u64,
}

impl CostLedger {
    pub fn record_local(&mut self) {
        self.local_requests += 1;
    }

    pub fn record_frontier(&mut self, cost: f64) {
        self.frontier_requests += 1;
        self.total_usd += cost;
    }

    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    pub fn savings_pct(&self) -> f64 {
        let total = self.local_requests + self.frontier_requests + self.cache_hits;
        if total == 0 {
            return 0.0;
        }
        let saved = self.local_requests + self.cache_hits;
        (saved as f64 / total as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_prompt_routes_local() {
        let rule = RoutingRule::default();
        let d = classify_difficulty("What is 2+2?", &rule);
        assert_eq!(d, RouteDecision::Local);
    }

    #[test]
    fn long_prompt_routes_frontier() {
        let rule = RoutingRule {
            max_tokens_for_local: 5,
            ..Default::default()
        };
        let long_prompt = "word ".repeat(100);
        let d = classify_difficulty(&long_prompt, &rule);
        assert_eq!(d, RouteDecision::Frontier);
    }

    #[test]
    fn cost_ledger_tracks_savings() {
        let mut ledger = CostLedger::default();
        ledger.record_local();
        ledger.record_local();
        ledger.record_cache_hit();
        ledger.record_frontier(0.01);
        // 3 out of 4 were free
        assert!((ledger.savings_pct() - 75.0).abs() < 0.01);
        assert!((ledger.total_usd - 0.01).abs() < 1e-9);
    }

    #[test]
    fn empty_ledger_zero_savings() {
        assert_eq!(CostLedger::default().savings_pct(), 0.0);
    }
}
