//! Tier 2: speculative cascade (architecture §3, AutoMix-style).
//!
//! The cascade is a runtime mode, not a classifier — it always runs the
//! cheap model first, inspects confidence, and escalates only on low
//! confidence. This gives us FrugalGPT-class savings: cheap covers ~70%
//! of traffic, frontier only fires on the genuinely-hard residual.
//!
//! Two confidence signals (in order of preference):
//! 1. **logprobs** — provider returns per-token log-probabilities; we
//!    compute the mean and compare to `min_avg_logprob` (default -1.5).
//!    Available on OpenAI, Ollama, DeepSeek, Cerebras. Not Anthropic.
//! 2. **Length heuristic fallback** — if the cheap model produced a very
//!    short or stop=length response, escalate. Crude but cheap.
//!
//! v0.8 shadow A/B will validate the threshold; v1.1 unified routing
//! will jointly optimize cascade vs classify-then-route.

use serde::{Deserialize, Serialize};
use tokenmiser_providers::{response_visible_content_empty, ChatResponse};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeConfig {
    /// Minimum acceptable mean log-probability across generated tokens.
    /// More negative = looser threshold = less escalation. -1.5 ≈
    /// "the cheap model was about 22% confident per token on average."
    pub min_avg_logprob: f32,
    /// Minimum number of generated tokens before the length-fallback
    /// considers a response "complete enough." Anything shorter triggers
    /// escalation.
    pub min_completion_tokens: u32,
}

impl Default for CascadeConfig {
    fn default() -> Self {
        Self {
            min_avg_logprob: -1.5,
            min_completion_tokens: 5,
        }
    }
}

/// Decide whether the cheap model's response is good enough or we need to
/// escalate to the frontier model.
pub fn should_escalate(resp: &ChatResponse, cfg: &CascadeConfig) -> EscalateDecision {
    // Empty visible content always escalates — a high-confidence logprob over
    // reasoning tokens is meaningless if the visible content is blank.
    // Real-world bug: gemma4 in reasoning mode emits high-confidence
    // logprobs for thinking tokens, then returns an empty message.content.
    if response_visible_content_empty(resp) {
        return EscalateDecision::Yes {
            reason: "empty visible content (reasoning-mode model)".into(),
            signal: Signal::Length(0),
        };
    }

    // Pull token logprobs from the OpenAI-shaped `choices[0].logprobs.content`.
    // Each element should have a `logprob` field.
    if let Some(avg) = avg_logprob(resp) {
        if avg < cfg.min_avg_logprob {
            return EscalateDecision::Yes {
                reason: format!(
                    "avg_logprob {:.3} < threshold {:.3}",
                    avg, cfg.min_avg_logprob
                ),
                signal: Signal::Logprob(avg),
            };
        }
        return EscalateDecision::No {
            signal: Signal::Logprob(avg),
        };
    }

    // Fallback: length heuristic.
    let completion_tokens = resp.usage.completion_tokens as u32;
    let finish = resp
        .choices
        .first()
        .and_then(|c| c.finish_reason.as_deref())
        .unwrap_or("");

    if completion_tokens < cfg.min_completion_tokens {
        return EscalateDecision::Yes {
            reason: format!(
                "completion_tokens {} < min {}",
                completion_tokens, cfg.min_completion_tokens
            ),
            signal: Signal::Length(completion_tokens),
        };
    }
    if finish == "length" {
        // Hit the cap before stopping naturally — likely needs more model.
        return EscalateDecision::Yes {
            reason: "finish_reason=length".into(),
            signal: Signal::Length(completion_tokens),
        };
    }

    EscalateDecision::No {
        signal: Signal::Length(completion_tokens),
    }
}

fn avg_logprob(resp: &ChatResponse) -> Option<f32> {
    let choice = resp.choices.first()?;
    let logprobs = choice.logprobs.as_ref()?;
    let content = logprobs.get("content")?.as_array()?;
    if content.is_empty() {
        return None;
    }
    let mut sum = 0.0_f64;
    let mut n = 0usize;
    for tok in content {
        if let Some(lp) = tok.get("logprob").and_then(|v| v.as_f64()) {
            sum += lp;
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some((sum / n as f64) as f32)
}

#[derive(Debug, Clone)]
pub enum EscalateDecision {
    Yes { reason: String, signal: Signal },
    No { signal: Signal },
}

#[derive(Debug, Clone, Copy)]
pub enum Signal {
    Logprob(f32),
    Length(u32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokenmiser_providers::{ChatChoice, ChatMessage, Usage};

    fn resp_with_logprobs(logprobs: Vec<f64>, completion_tokens: u64) -> ChatResponse {
        let content = logprobs
            .into_iter()
            .map(|lp| json!({ "logprob": lp }))
            .collect::<Vec<_>>();
        ChatResponse {
            id: "t".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "m".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String("hi".into()),
                    extra: Default::default(),
                },
                finish_reason: Some("stop".into()),
                logprobs: Some(json!({"content": content})),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens,
                total_tokens: 10 + completion_tokens,
            },
            extra: Default::default(),
        }
    }

    #[test]
    fn high_confidence_does_not_escalate() {
        let r = resp_with_logprobs(vec![-0.1, -0.2, -0.05], 3);
        match should_escalate(&r, &CascadeConfig::default()) {
            EscalateDecision::No { .. } => {}
            d => panic!("expected No, got {:?}", d),
        }
    }

    #[test]
    fn low_confidence_escalates() {
        let r = resp_with_logprobs(vec![-3.0, -4.0, -2.5], 3);
        match should_escalate(&r, &CascadeConfig::default()) {
            EscalateDecision::Yes { .. } => {}
            d => panic!("expected Yes, got {:?}", d),
        }
    }

    #[test]
    fn empty_content_always_escalates_even_with_high_logprobs() {
        // Mimics gemma4 reasoning-mode: high-confidence logprobs over
        // thinking tokens, but message.content is empty. Should escalate.
        let mut r = resp_with_logprobs(vec![-0.1, -0.05, -0.08], 5);
        r.choices[0].message.content = serde_json::Value::String("".into());
        match should_escalate(&r, &CascadeConfig::default()) {
            EscalateDecision::Yes { reason, .. } => {
                assert!(
                    reason.contains("empty"),
                    "expected empty-content reason, got: {reason}"
                );
            }
            d => panic!("expected escalate on empty content, got {:?}", d),
        }
    }

    #[test]
    fn short_response_falls_back_to_length_and_escalates() {
        let r = ChatResponse {
            id: "t".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "m".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String("ok".into()),
                    extra: Default::default(),
                },
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 1,
                total_tokens: 11,
            },
            extra: Default::default(),
        };
        match should_escalate(&r, &CascadeConfig::default()) {
            EscalateDecision::Yes { .. } => {}
            d => panic!("expected Yes via length, got {:?}", d),
        }
    }
}
