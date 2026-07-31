//! Shadow A/B with an LLM judge.
//!
//! A sampled fraction of routed traffic is replayed against the frontier model
//! in the background, a judge scores the pair, and per-segment win rates are
//! tallied so a regressed segment can raise an alert.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokenmiser_providers::{ChatMessage, ChatRequest, ChatResponse};
use tracing::{info, warn};

pub mod judge;
pub mod scheduler;

pub use judge::JudgeVerdict;
pub use scheduler::ShadowScheduler;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfig {
    /// Fraction of traffic to shadow-test, in [0.0, 1.0].
    pub sample_rate: f32,
    /// Which model to use as the frontier baseline in shadow tests.
    pub frontier_model: String,
    /// Model used as the judge.
    pub judge_model: String,
    /// Cheap-model win rate below this over `min_samples_per_segment` marks
    /// the segment regressed.
    pub auto_gate_floor: f32,
    pub min_samples_per_segment: u32,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            sample_rate: 0.01,
            frontier_model: "claude-opus-4-7".into(),
            judge_model: "claude-sonnet-4-6".into(),
            auto_gate_floor: 0.45,
            min_samples_per_segment: 30,
        }
    }
}

/// One completed shadow comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowSample {
    pub segment: String,
    pub cheap_model: String,
    pub frontier_model: String,
    pub verdict: JudgeVerdict,
}

/// Win-rate aggregator: per-segment running tallies, plus regression flag.
pub struct WinRateAggregator {
    by_segment: Mutex<HashMap<String, SegmentStats>>,
    floor: f32,
    min_samples: u32,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SegmentStats {
    pub cheap_wins: u32,
    pub frontier_wins: u32,
    pub ties: u32,
    pub regressed: bool,
}

impl SegmentStats {
    pub fn cheap_win_rate(&self) -> f32 {
        let total = self.cheap_wins + self.frontier_wins;
        if total == 0 {
            return 1.0;
        }
        self.cheap_wins as f32 / total as f32
    }
    pub fn total(&self) -> u32 {
        self.cheap_wins + self.frontier_wins + self.ties
    }
}

impl WinRateAggregator {
    pub fn new(cfg: &ShadowConfig) -> Arc<Self> {
        Arc::new(Self {
            by_segment: Mutex::new(HashMap::new()),
            floor: cfg.auto_gate_floor,
            min_samples: cfg.min_samples_per_segment,
        })
    }

    pub fn record(&self, sample: &ShadowSample) {
        let mut map = self.by_segment.lock();
        let s = map.entry(sample.segment.clone()).or_default();
        match sample.verdict {
            JudgeVerdict::A => s.cheap_wins += 1,
            JudgeVerdict::B => s.frontier_wins += 1,
            JudgeVerdict::Tie => s.ties += 1,
        }
        if s.total() >= self.min_samples && s.cheap_win_rate() < self.floor && !s.regressed {
            s.regressed = true;
            warn!(
                segment = %sample.segment,
                cheap_win_rate = s.cheap_win_rate(),
                floor = self.floor,
                samples = s.total(),
                "AUTO-GATE: segment regressed; rerouting to frontier"
            );
        }
    }

    pub fn snapshot(&self) -> HashMap<String, SegmentStats> {
        self.by_segment.lock().clone()
    }

    pub fn regressed_segments(&self) -> Vec<String> {
        self.by_segment
            .lock()
            .iter()
            .filter(|(_, s)| s.regressed)
            .map(|(k, _)| k.clone())
            .collect()
    }
}

/// Bucket a request into a coarse aggregation segment, keyed on the opening
/// words of its first user message.
pub fn segment_of(req: &ChatRequest) -> String {
    let user_text = req
        .messages
        .iter()
        .find(|m: &&ChatMessage| m.role == "user")
        .and_then(|m| match &m.content {
            serde_json::Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "<empty>".into());
    user_text
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Glue type the proxy uses to enqueue shadow work without blocking.
pub struct ShadowEnqueue {
    pub req: ChatRequest,
    pub cheap_response: ChatResponse,
    pub cheap_model: String,
    pub segment: String,
}

impl ShadowEnqueue {
    pub fn from_request(
        req: &ChatRequest,
        cheap_response: &ChatResponse,
        cheap_model: &str,
    ) -> Self {
        Self {
            req: req.clone(),
            cheap_response: cheap_response.clone(),
            cheap_model: cheap_model.to_string(),
            segment: segment_of(req),
        }
    }
}

// Surfaced so consumers need no second import.
pub use tokenmiser_providers::ProviderRegistry as Registry;

/// Log a completed shadow sample.
pub fn log_sample(sample: &ShadowSample) {
    info!(
        segment = %sample.segment,
        cheap = %sample.cheap_model,
        frontier = %sample.frontier_model,
        verdict = ?sample.verdict,
        "shadow_sample"
    );
}

// Re-exported so callers can build the Arc without an extra import.
pub use std::sync::Arc as ArcReexport;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_buckets_on_first_words() {
        let req = ChatRequest {
            model: "auto".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(
                    "What is the capital of france please answer briefly".into(),
                ),
                extra: Default::default(),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: None,
            extra: Default::default(),
        };
        assert_eq!(segment_of(&req), "what is the capital of france");
    }

    #[test]
    fn aggregator_auto_gates_on_low_win_rate() {
        let agg = WinRateAggregator::new(&ShadowConfig {
            min_samples_per_segment: 4,
            auto_gate_floor: 0.45,
            ..Default::default()
        });
        // Cheap win rate of 0.25 is below the 0.45 floor.
        for v in [
            JudgeVerdict::A,
            JudgeVerdict::B,
            JudgeVerdict::B,
            JudgeVerdict::B,
        ] {
            agg.record(&ShadowSample {
                segment: "what is".into(),
                cheap_model: "cheap".into(),
                frontier_model: "frontier".into(),
                verdict: v,
            });
        }
        assert_eq!(agg.regressed_segments(), vec!["what is".to_string()]);
    }

    #[test]
    fn aggregator_does_not_gate_below_min_samples() {
        let agg = WinRateAggregator::new(&ShadowConfig {
            min_samples_per_segment: 100,
            ..Default::default()
        });
        agg.record(&ShadowSample {
            segment: "x".into(),
            cheap_model: "c".into(),
            frontier_model: "f".into(),
            verdict: JudgeVerdict::B,
        });
        assert!(agg.regressed_segments().is_empty());
    }
}
