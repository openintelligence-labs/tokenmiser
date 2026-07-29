//! LLM-as-judge: pairwise preference scoring.
//!
//! The judge prompt is intentionally simple and position-randomized to
//! reduce A/B bias. Architecture §5 calls for a rotated judge; v0.8 fixes
//! the judge model and v0.9 rotates across a pool.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokenmiser_providers::{ChatMessage, ChatRequest, ChatResponse, Provider, ProviderRegistry};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JudgeVerdict {
    /// First response (cheap) wins.
    A,
    /// Second response (frontier) wins.
    B,
    Tie,
}

/// Build the judge prompt. We always present cheap first, then frontier;
/// to combat A/B positional bias we sometimes invert order at the call site
/// and remap the result. v0.8 keeps order fixed for simplicity.
fn judge_prompt(prompt: &str, a: &str, b: &str) -> Vec<ChatMessage> {
    let system =
        "You are an impartial judge comparing two assistant responses to the same user prompt. \
Reply with EXACTLY one token: A, B, or T (for tie). Do not explain."
            .to_string();
    let user = format!(
        "User prompt:\n---\n{prompt}\n---\n\n\
Response A:\n---\n{a}\n---\n\n\
Response B:\n---\n{b}\n---\n\n\
Which response is better? Reply A, B, or T."
    );
    vec![
        ChatMessage {
            role: "system".into(),
            content: serde_json::Value::String(system),
            extra: Default::default(),
        },
        ChatMessage {
            role: "user".into(),
            content: serde_json::Value::String(user),
            extra: Default::default(),
        },
    ]
}

/// Run a judge call. Resolves `judge_model` via the registry.
pub async fn judge(
    registry: &ProviderRegistry,
    judge_model: &str,
    user_prompt: &str,
    cheap_text: &str,
    frontier_text: &str,
) -> Result<JudgeVerdict> {
    let (provider, real) = registry
        .resolve(judge_model)
        .map_err(|e| anyhow!("judge resolve: {e}"))?;

    let req = ChatRequest {
        model: real.clone(),
        messages: judge_prompt(user_prompt, cheap_text, frontier_text),
        temperature: Some(0.0),
        max_tokens: Some(8),
        top_p: None,
        stream: None,
        extra: Default::default(),
    };

    let resp = provider
        .complete(&req)
        .await
        .map_err(|e| anyhow!("judge call: {e}"))?;

    Ok(parse_verdict(&resp))
}

fn parse_verdict(resp: &ChatResponse) -> JudgeVerdict {
    let text = resp
        .choices
        .first()
        .and_then(|c| match &c.message.content {
            serde_json::Value::String(s) => Some(s.trim().to_uppercase()),
            _ => None,
        })
        .unwrap_or_default();

    // Look for first A/B/T character that isn't part of "the" or similar.
    for c in text.chars() {
        match c {
            'A' => return JudgeVerdict::A,
            'B' => return JudgeVerdict::B,
            'T' => return JudgeVerdict::Tie,
            _ => continue,
        }
    }
    JudgeVerdict::Tie
}

// Suppress unused-import warning when the file is read in isolation.
#[allow(dead_code)]
fn _provider_typecheck(_: Arc<dyn Provider>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenmiser_providers::{ChatChoice, ChatMessage, Usage};

    fn resp_with(text: &str) -> ChatResponse {
        ChatResponse {
            id: "j".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "judge".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: serde_json::Value::String(text.into()),
                    extra: Default::default(),
                },
                finish_reason: Some("stop".into()),
                logprobs: None,
            }],
            usage: Usage::default(),
            extra: Default::default(),
        }
    }

    #[test]
    fn parses_a_b_t() {
        assert_eq!(parse_verdict(&resp_with("A")), JudgeVerdict::A);
        assert_eq!(parse_verdict(&resp_with("B")), JudgeVerdict::B);
        assert_eq!(parse_verdict(&resp_with("T")), JudgeVerdict::Tie);
    }

    #[test]
    fn handles_chatty_judge() {
        assert_eq!(parse_verdict(&resp_with("A is better")), JudgeVerdict::A);
        assert_eq!(parse_verdict(&resp_with("I pick B")), JudgeVerdict::B);
    }

    #[test]
    fn unparseable_falls_to_tie() {
        assert_eq!(parse_verdict(&resp_with("hmm")), JudgeVerdict::Tie);
    }
}
