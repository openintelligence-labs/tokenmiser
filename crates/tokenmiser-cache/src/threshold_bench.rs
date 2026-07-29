//! Threshold tuning utility. Not a unit test (needs the bge model loaded
//! and takes a few seconds). Run with:
//!
//!     cargo test -p tokenmiser-cache --release threshold_sweep -- --ignored --nocapture
//!
//! Output is a precision/recall/F1 table across cosine thresholds. Use the
//! best-F1 row to set the default `semantic_threshold` in `tokenmiser-config`.

#![allow(dead_code)]

use crate::l2::L2Cache;
use std::time::Duration;
use tokenmiser_providers::{ChatChoice, ChatMessage, ChatRequest, ChatResponse, Usage};

fn req(text: &str) -> ChatRequest {
    ChatRequest {
        model: "ollama:qwen2.5:7b".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: serde_json::Value::String(text.into()),
            extra: Default::default(),
        }],
        temperature: Some(0.0),
        max_tokens: Some(20),
        top_p: None,
        stream: None,
        extra: Default::default(),
    }
}

fn dummy_resp() -> ChatResponse {
    ChatResponse {
        id: "x".into(),
        object: "chat.completion".into(),
        created: 0,
        model: "x".into(),
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
        usage: Usage::default(),
        extra: Default::default(),
    }
}

/// `should_hit` paraphrase pairs — semantically equivalent, cache hit is correct.
fn positive_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("what is the capital of france", "france capital city name"),
        (
            "how do i make pasta carbonara",
            "recipe for pasta carbonara",
        ),
        (
            "what's the boiling point of water in celsius",
            "at what celsius temperature does water boil",
        ),
        (
            "explain how DNS resolution works",
            "walk me through DNS lookup",
        ),
        ("translate hello to spanish", "what is hello in spanish"),
        (
            "summarize the plot of macbeth",
            "give me a synopsis of macbeth",
        ),
        ("convert 32 fahrenheit to celsius", "32F in celsius please"),
        ("who wrote romeo and juliet", "author of romeo and juliet"),
        ("what year did world war 2 end", "when did ww2 end"),
        ("define photosynthesis briefly", "what is photosynthesis"),
    ]
}

/// `should_NOT_hit` pairs — superficially similar but distinct, cache miss is correct.
fn negative_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "what is the capital of france",
            "what is the capital of germany",
        ),
        (
            "how do i make pasta carbonara",
            "how do i make pasta puttanesca",
        ),
        (
            "boiling point of water in celsius",
            "freezing point of water in celsius",
        ),
        ("translate hello to spanish", "translate goodbye to spanish"),
        ("who wrote romeo and juliet", "who wrote hamlet"),
        (
            "what year did world war 2 end",
            "what year did world war 1 end",
        ),
        ("define photosynthesis", "define respiration"),
        ("what is python", "what is javascript"),
        ("how does HTTPS work", "how does FTP work"),
        ("recipe for chocolate cake", "recipe for vanilla cake"),
    ]
}

#[test]
#[ignore]
fn threshold_sweep() {
    let thresholds = [
        0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.85, 0.87, 0.90,
    ];
    let positives = positive_pairs();
    let negatives = negative_pairs();

    println!(
        "\n{:>6} {:>4} {:>4} {:>4} {:>4} {:>6} {:>6} {:>6}",
        "thresh", "TP", "FN", "FP", "TN", "prec", "recall", "F1"
    );
    println!("{}", "-".repeat(56));

    let mut best: (f32, f32) = (0.0, 0.0); // (threshold, F1)
    for &t in &thresholds {
        let cache = L2Cache::new(t, Duration::from_secs(3600), 1024).expect("L2 init");

        let mut tp = 0; // hit on positive (correct)
        let mut fp = 0; // hit on negative (wrong — semantic false positive)
        let mut tn = 0; // miss on negative (correct)
        let mut fn_ = 0; // miss on positive (wrong — missed a paraphrase)

        for (a, b) in &positives {
            // Insert A, look up B with fresh tenant to avoid cross-contamination.
            let tenant = format!("pos-{a}");
            cache.insert(&req(a), &tenant, &dummy_resp());
            if cache.lookup(&req(b), &tenant).is_some() {
                tp += 1;
            } else {
                fn_ += 1;
            }
        }
        for (a, b) in &negatives {
            let tenant = format!("neg-{a}");
            cache.insert(&req(a), &tenant, &dummy_resp());
            if cache.lookup(&req(b), &tenant).is_some() {
                fp += 1;
            } else {
                tn += 1;
            }
        }

        let prec = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f32 / (tp + fn_) as f32
        } else {
            0.0
        };
        let f1 = if prec + recall > 0.0 {
            2.0 * prec * recall / (prec + recall)
        } else {
            0.0
        };

        println!(
            "{:>6.2} {:>4} {:>4} {:>4} {:>4} {:>6.2} {:>6.2} {:>6.3}",
            t, tp, fn_, fp, tn, prec, recall, f1
        );
        if f1 > best.1 {
            best = (t, f1);
        }
    }
    println!("\nBest threshold: {:.2} (F1={:.3})", best.0, best.1);
}
