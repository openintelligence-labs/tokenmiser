//! Tier 0 heuristic prefilter (architecture §3).
//!
//! Sub-microsecond classification using cheap signals:
//! - prompt char/word length
//! - structural-edit keywords (refactor, implement, design, debug, …) → Hard
//! - trivial-Q keywords (what is, summarize, translate, …) → Easy
//! - JSON-mode flag → tends toward Medium (structured outputs benefit from
//!   slightly stronger models in practice)
//!
//! Goal: classify ~30% of traffic confidently so Tier 1 only runs on the
//! ambiguous middle band. Anything else falls to Medium and gets handed up.

use tokenmiser_providers::ChatRequest;

use crate::Difficulty;

const HARD_SIGNALS: &[&str] = &[
    "refactor",
    "implement",
    "design",
    "debug",
    "prove",
    "architect",
    "optimize",
    "analyze the following code",
    "write a function",
    "explain in depth",
    "step by step",
    "chain of thought",
];

const EASY_SIGNALS: &[&str] = &[
    "what is",
    "what's",
    "define",
    "translate",
    "summarize",
    "tl;dr",
    "shorten",
    "rewrite as",
    "classify",
    "yes or no",
    "true or false",
];

pub fn tier0_difficulty(req: &ChatRequest) -> Difficulty {
    let total: String = req
        .messages
        .iter()
        .map(|m| match &m.content {
            serde_json::Value::String(s) => s.clone(),
            v => v.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ");

    let lower = total.to_lowercase();
    let word_count = total.split_whitespace().count();

    // Very long prompts: lean Hard regardless of keywords. Frontier models
    // still beat cheap models on long-context comprehension.
    if word_count > 500 {
        return Difficulty::Hard;
    }
    if HARD_SIGNALS.iter().any(|s| lower.contains(s)) {
        return Difficulty::Hard;
    }
    if EASY_SIGNALS.iter().any(|s| lower.contains(s)) && word_count < 200 {
        return Difficulty::Easy;
    }
    if word_count < 30 {
        return Difficulty::Easy;
    }
    Difficulty::Medium
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenmiser_providers::ChatMessage;

    fn user(s: &str) -> ChatRequest {
        ChatRequest {
            model: "auto".into(),
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
    fn refactor_is_hard() {
        assert_eq!(
            tier0_difficulty(&user("refactor this auth code")),
            Difficulty::Hard
        );
    }

    #[test]
    fn short_factoid_is_easy() {
        assert_eq!(tier0_difficulty(&user("what is 2+2?")), Difficulty::Easy);
    }

    #[test]
    fn very_long_is_hard() {
        let long = "word ".repeat(600);
        assert_eq!(tier0_difficulty(&user(&long)), Difficulty::Hard);
    }

    #[test]
    fn medium_neutral_falls_through() {
        let medium: String = "Tell me about historical aviation events. ".repeat(8);
        assert_eq!(tier0_difficulty(&user(&medium)), Difficulty::Medium);
    }
}
