//! Shadow scheduler: samples a fraction of traffic and runs the frontier
//! comparison + judge in the background.
//!
//! The proxy calls `ShadowScheduler::maybe_enqueue(...)` synchronously
//! after returning the cheap-model response to the user. If the sample
//! roll succeeds, the scheduler fires a `tokio::spawn` that runs the
//! frontier call + judge + aggregator update. User-facing latency is
//! unaffected.

use std::sync::Arc;

use tokenmiser_providers::{ChatResponse, ProviderRegistry};
use tracing::warn;

use crate::{
    judge::judge, log_sample, ShadowConfig, ShadowEnqueue, ShadowSample, WinRateAggregator,
};

pub struct ShadowScheduler {
    cfg: ShadowConfig,
    registry: Arc<ProviderRegistry>,
    aggregator: Arc<WinRateAggregator>,
}

impl ShadowScheduler {
    pub fn new(
        cfg: ShadowConfig,
        registry: Arc<ProviderRegistry>,
        aggregator: Arc<WinRateAggregator>,
    ) -> Arc<Self> {
        Arc::new(Self {
            cfg,
            registry,
            aggregator,
        })
    }

    pub fn aggregator(&self) -> &Arc<WinRateAggregator> {
        &self.aggregator
    }

    /// Roll the sample-rate dice. If true, spawn a background shadow
    /// comparison.
    pub fn maybe_enqueue(self: &Arc<Self>, e: ShadowEnqueue) {
        if !roll(self.cfg.sample_rate) {
            return;
        }
        let me = Arc::clone(self);
        tokio::spawn(async move {
            if let Err(e) = me.run(e).await {
                warn!(error = %e, "shadow comparison failed");
            }
        });
    }

    async fn run(self: Arc<Self>, e: ShadowEnqueue) -> anyhow::Result<()> {
        let cheap_text = extract_text(&e.cheap_response);

        // Call the frontier model with the same request.
        let mut frontier_req = e.req.clone();
        frontier_req.model = self.cfg.frontier_model.clone();
        let (frontier_provider, frontier_real) = self
            .registry
            .resolve(&self.cfg.frontier_model)
            .map_err(|err| anyhow::anyhow!("frontier resolve: {err}"))?;
        frontier_req.model = frontier_real.clone();
        let frontier_resp = frontier_provider
            .complete(&frontier_req)
            .await
            .map_err(|err| anyhow::anyhow!("frontier call: {err}"))?;
        let frontier_text = extract_text(&frontier_resp);

        // User prompt = first user message.
        let user_prompt = e
            .req
            .messages
            .iter()
            .find(|m| m.role == "user")
            .and_then(|m| match &m.content {
                serde_json::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();

        let verdict = judge(
            &self.registry,
            &self.cfg.judge_model,
            &user_prompt,
            &cheap_text,
            &frontier_text,
        )
        .await?;

        let sample = ShadowSample {
            segment: e.segment,
            cheap_model: e.cheap_model,
            frontier_model: self.cfg.frontier_model.clone(),
            verdict,
        };
        log_sample(&sample);
        self.aggregator.record(&sample);

        Ok(())
    }
}

fn roll(rate: f32) -> bool {
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }
    // Deterministic-ish using nanos; good enough for sampling. v0.9 can
    // swap in `rand::random::<f32>()` if the bias profile matters.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    let r = (nanos as f32) / (u32::MAX as f32);
    r < rate
}

fn extract_text(resp: &ChatResponse) -> String {
    resp.choices
        .first()
        .map(|c| match &c.message.content {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_default()
}

/// Re-export so tokenmiser-proxy can pull just `tokenmiser_quality::Arc` if
/// it wants symmetry with internal helpers.
#[allow(dead_code)]
fn _arc_typecheck(_: Arc<ShadowScheduler>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roll_extremes() {
        assert!(roll(1.0));
        assert!(!roll(0.0));
    }
}
