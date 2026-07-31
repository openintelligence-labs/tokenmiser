//! In-memory cost ledger and budget evaluation.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokenmiser_config::PricingTable;

/// Daily spend bucket. A mutex rather than atomics because the day index and
/// the amount must roll over together.
#[derive(Debug, Default)]
struct DailySpend {
    day: u64,
    micro_usd: u64,
}

#[derive(Debug, Default)]
pub struct CostLedger {
    requests_total: AtomicU64,
    requests_local: AtomicU64,
    requests_frontier: AtomicU64,
    cache_hits: AtomicU64,
    /// Stored as USD * 1e6 (micro-dollars) for atomic counting.
    spent_micro_usd: AtomicU64,
    /// USD * 1e6 that routing everything to frontier would have cost.
    counterfactual_micro_usd: AtomicU64,
    /// Paid requests with no pricing entry, whose real cost is unknown rather
    /// than zero. Counted separately because reporting them as `$0.00` would
    /// under-report spend — the one lie a cost meter must not tell.
    unpriced_requests: AtomicU64,
    daily: Mutex<DailySpend>,
    pricing: PricingTable,
}

/// Days since the Unix epoch, UTC: the daily-budget bucket key.
fn utc_day_index() -> u64 {
    (chrono::Utc::now().timestamp().max(0) as u64) / 86_400
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
        self.record_paid_at(utc_day_index(), model, prompt_tokens, completion_tokens)
    }

    fn record_paid_at(&self, day: u64, model: &str, prompt_tokens: u64, completion_tokens: u64) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_frontier.fetch_add(1, Ordering::Relaxed);

        if let Some(p) = self.pricing.get(model) {
            let cost = p.cost_usd(prompt_tokens, completion_tokens);
            let micro = (cost * 1_000_000.0) as u64;
            self.spent_micro_usd.fetch_add(micro, Ordering::Relaxed);
            self.counterfactual_micro_usd
                .fetch_add(micro, Ordering::Relaxed);
            let mut daily = self.daily.lock();
            if daily.day != day {
                daily.day = day;
                daily.micro_usd = 0;
            }
            daily.micro_usd += micro;
        } else {
            // No published price to encode. Counted so `/stats` can report an
            // uncomputable cost instead of implying the calls were free.
            self.unpriced_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a free local request, attributing counterfactual cost as if it
    /// had gone to `counterfactual_model`.
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

    /// USD spent during the current UTC day; 0 when the last recorded spend
    /// was on an earlier day.
    pub fn spent_today_usd(&self) -> f64 {
        self.spent_today_usd_at(utc_day_index())
    }

    fn spent_today_usd_at(&self, day: u64) -> f64 {
        let daily = self.daily.lock();
        if daily.day == day {
            daily.micro_usd as f64 / 1_000_000.0
        } else {
            0.0
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
            spent_today_usd: self.spent_today_usd(),
            counterfactual_usd: counterfactual,
            saved_usd: (counterfactual - spent).max(0.0),
            unpriced_requests: self.unpriced_requests.load(Ordering::Relaxed),
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
    pub spent_today_usd: f64,
    pub counterfactual_usd: f64,
    pub saved_usd: f64,
    /// Paid requests whose per-token price is unknown, so their cost is not in
    /// `spent_usd`. Non-zero means `spent_usd` is a lower bound.
    #[serde(default)]
    pub unpriced_requests: u64,
}

/// Budget evaluation over a ledger snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetStatus {
    pub daily_limit_usd: Option<f64>,
    pub total_limit_usd: Option<f64>,
    pub spent_today_usd: f64,
    pub spent_total_usd: f64,
    pub daily_exceeded: bool,
    pub total_exceeded: bool,
    pub exceeded: bool,
    pub enforce: bool,
}

impl BudgetStatus {
    pub fn evaluate(cfg: &tokenmiser_config::BudgetConfig, snap: &CostSnapshot) -> Self {
        let daily_exceeded = cfg
            .daily_usd
            .map(|lim| snap.spent_today_usd >= lim)
            .unwrap_or(false);
        let total_exceeded = cfg
            .total_usd
            .map(|lim| snap.spent_usd >= lim)
            .unwrap_or(false);
        Self {
            daily_limit_usd: cfg.daily_usd,
            total_limit_usd: cfg.total_usd,
            spent_today_usd: snap.spent_today_usd,
            spent_total_usd: snap.spent_usd,
            daily_exceeded,
            total_exceeded,
            exceeded: daily_exceeded || total_exceeded,
            enforce: cfg.enforce,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paid_request_records_spend() {
        let l = CostLedger::new(PricingTable::canonical());
        // $5/M input, $25/M output: 1k + 500 tokens is $0.0175.
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

    #[test]
    fn daily_spend_tracks_current_day() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_paid_at(100, "claude-opus-4-7", 1_000, 500); // $0.0175
        l.record_paid_at(100, "claude-opus-4-7", 1_000, 500);
        assert!((l.spent_today_usd_at(100) - 0.035).abs() < 1e-6);
        assert!((l.spent_today_usd_at(101)).abs() < 1e-9);
        assert!((l.snapshot().spent_usd - 0.035).abs() < 1e-6);
    }

    #[test]
    fn daily_spend_rolls_over_on_new_day() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_paid_at(100, "claude-opus-4-7", 1_000, 500);
        l.record_paid_at(101, "claude-opus-4-7", 1_000, 500);
        assert!((l.spent_today_usd_at(101) - 0.0175).abs() < 1e-6);
        assert!((l.snapshot().spent_usd - 0.035).abs() < 1e-6);
    }

    #[test]
    fn free_and_cached_requests_do_not_touch_daily_spend() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_free(Some("claude-opus-4-7"), 1_000, 500);
        l.record_cache_hit("claude-opus-4-7", 1_000, 500);
        assert!((l.spent_today_usd()).abs() < 1e-9);
    }

    #[test]
    fn unpriced_remote_request_is_counted_not_silently_free() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_paid("gpt-oss:20b-cloud", 1_000, 500);
        let s = l.snapshot();
        assert_eq!(s.requests_frontier, 1, "remote request counted as frontier");
        assert_eq!(s.requests_local, 0, "must not be counted as local/free");
        assert_eq!(s.unpriced_requests, 1, "unknown price surfaced explicitly");
        assert!((s.spent_usd).abs() < 1e-9);
        assert!(
            (s.saved_usd).abs() < 1e-9,
            "an unpriced remote call must not manufacture savings"
        );
    }

    #[test]
    fn priced_requests_do_not_increment_unpriced_counter() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_paid("claude-opus-4-7", 1_000, 500);
        l.record_free(Some("claude-opus-4-7"), 1_000, 500);
        l.record_cache_hit("claude-opus-4-7", 1_000, 500);
        assert_eq!(l.snapshot().unpriced_requests, 0);
    }

    fn budget(
        daily: Option<f64>,
        total: Option<f64>,
        enforce: bool,
    ) -> tokenmiser_config::BudgetConfig {
        tokenmiser_config::BudgetConfig {
            daily_usd: daily,
            total_usd: total,
            enforce,
        }
    }

    #[test]
    fn budget_status_under_limit_is_ok() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_paid_at(utc_day_index(), "claude-opus-4-7", 1_000, 500); // $0.0175
        let st = BudgetStatus::evaluate(&budget(Some(1.0), Some(10.0), false), &l.snapshot());
        assert!(!st.exceeded);
        assert!(!st.daily_exceeded && !st.total_exceeded);
    }

    #[test]
    fn budget_status_daily_exceeded() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_paid_at(utc_day_index(), "claude-opus-4-7", 100_000, 50_000); // $1.75
        let st = BudgetStatus::evaluate(&budget(Some(1.0), None, false), &l.snapshot());
        assert!(st.daily_exceeded);
        assert!(!st.total_exceeded);
        assert!(st.exceeded);
        assert!(!st.enforce);
    }

    #[test]
    fn budget_status_total_exceeded() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_paid_at(utc_day_index(), "claude-opus-4-7", 100_000, 50_000); // $1.75
        let st = BudgetStatus::evaluate(&budget(None, Some(1.5), true), &l.snapshot());
        assert!(st.total_exceeded && st.exceeded && st.enforce);
    }

    #[test]
    fn budget_status_no_limits_never_exceeds() {
        let l = CostLedger::new(PricingTable::canonical());
        l.record_paid_at(utc_day_index(), "claude-opus-4-7", 1_000_000, 500_000);
        let st = BudgetStatus::evaluate(&budget(None, None, false), &l.snapshot());
        assert!(!st.exceeded);
    }
}
