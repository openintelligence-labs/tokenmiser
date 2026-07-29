//! Cost meter + ledger. v0.1 ships an in-memory atomic ledger; v0.2 will add
//! SSE streaming + per-tenant aggregation + persistence.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokenmiser_config::PricingTable;

#[derive(Debug, Default)]
pub struct CostLedger {
    requests_total: AtomicU64,
    requests_local: AtomicU64,
    requests_frontier: AtomicU64,
    cache_hits: AtomicU64,
    /// Stored as USD * 1e6 (micro-dollars) for atomic counting.
    spent_micro_usd: AtomicU64,
    /// USD * 1e6 of what we *would have* spent if everything went to frontier.
    counterfactual_micro_usd: AtomicU64,
    pricing: PricingTable,
}

impl CostLedger {
    pub fn new(pricing: PricingTable) -> Arc<Self> {
        Arc::new(Self {
            pricing,
            ..Default::default()
        })
    }

    /// Record a request that actually hit a paid provider.
    pub fn record_paid(&self, model: &str, prompt_tokens: u64, completion_tokens: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_frontier.fetch_add(1, Ordering::Relaxed);

        if let Some(p) = self.pricing.get(model) {
            let cost = p.cost_usd(prompt_tokens, completion_tokens);
            self.spent_micro_usd
                .fetch_add((cost * 1_000_000.0) as u64, Ordering::Relaxed);
            self.counterfactual_micro_usd
                .fetch_add((cost * 1_000_000.0) as u64, Ordering::Relaxed);
        }
    }

    /// Record a request that went to a free (local) provider, but tag its
    /// counterfactual cost as if it had gone to the named frontier model.
    pub fn record_free(
        &self,
        counterfactual_model: Option<&str>,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_local.fetch_add(1, Ordering::Relaxed);

        if let Some(m) = counterfactual_model {
            if let Some(p) = self.pricing.get(m) {
                let cost = p.cost_usd(prompt_tokens, completion_tokens);
                self.counterfactual_micro_usd
                    .fetch_add((cost * 1_000_000.0) as u64, Ordering::Relaxed);
            }
        }
    }

    pub fn record_cache_hit(
        &self,
        counterfactual_model: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
    ) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
        if let Some(p) = self.pricing.get(counterfactual_model) {
            let cost = p.cost_usd(prompt_tokens, completion_tokens);
            self.counterfactual_micro_usd
                .fetch_add((cost * 1_000_000.0) as u64, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> CostSnapshot {
        let spent = self.spent_micro_usd.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let counterfactual =
            self.counterfactual_micro_usd.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        CostSnapshot {
            requests_total: self.requests_total.load(Ordering::Relaxed),
            requests_local: self.requests_local.load(Ordering::Relaxed),
            requests_frontier: self.requests_frontier.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            spent_usd: spent,
            counterfactual_usd: counterfactual,
            saved_usd: (counterfactual - spent).max(0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSnapshot {
    pub requests_total: u64,
    pub requests_local: u64,
    pub requests_frontier: u64,
    pub cache_hits: u64,
    pub spent_usd: f64,
    pub counterfactual_usd: f64,
    pub saved_usd: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paid_request_records_spend() {
        let l = CostLedger::new(PricingTable::canonical());
        // Opus 4.7: $5/M input, $25/M output. 1k+500 → $0.0175
        l.record_paid("claude-opus-4-7", 1_000, 500);
        let s = l.snapshot();
        assert_eq!(s.requests_frontier, 1);
        assert!((s.spent_usd - 0.0175).abs() < 1e-6);
        assert!((s.saved_usd).abs() < 1e-6);
    }

    #[test]
    fn local_request_attributes_counterfactual_savings() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_free(Some("claude-opus-4-7"), 1_000, 500);
        let s = l.snapshot();
        assert_eq!(s.requests_local, 1);
        assert!((s.spent_usd).abs() < 1e-6);
        assert!((s.saved_usd - 0.0175).abs() < 1e-6);
    }
}
