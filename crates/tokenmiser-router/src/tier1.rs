//! Tier 1 semantic classifier: difficulty exemplars are embedded at startup
//! with the same `bge-small-en-v1.5` model the L2 cache loads, and a request
//! takes the difficulty of its nearest exemplar.
//!
//! Less accurate than a fine-tuned classifier, but it adds no dependencies and
//! reuses the existing embedder. Swapping in trained weights is a change to
//! `classify()` alone.

use anyhow::{anyhow, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use parking_lot::Mutex;
use tokenmiser_providers::ChatRequest;
use tracing::info;

use crate::Difficulty;

struct Exemplar {
    embedding: Vec<f32>,
    difficulty: Difficulty,
}

pub struct Tier1Classifier {
    embedder: Mutex<TextEmbedding>,
    exemplars: Vec<Exemplar>,
}

impl Tier1Classifier {
    /// Build the classifier and pre-embed the default exemplar set.
    pub fn new() -> Result<Self> {
        let opts = InitOptions::new(EmbeddingModel::BGESmallENV15);
        let mut embedder = TextEmbedding::try_new(opts)
            .map_err(|e| anyhow!("bge-small init for Tier1 failed: {e}"))?;

        let raw = default_exemplars();
        let texts: Vec<&str> = raw.iter().map(|(t, _)| *t).collect();
        let embeddings = embedder
            .embed(texts, None)
            .map_err(|e| anyhow!("Tier1 exemplar embed failed: {e}"))?;

        let exemplars = embeddings
            .into_iter()
            .zip(raw.iter())
            .map(|(emb, (_, d))| Exemplar {
                embedding: emb,
                difficulty: *d,
            })
            .collect::<Vec<_>>();

        info!(
            exemplars = exemplars.len(),
            "Tier1 semantic classifier ready"
        );
        Ok(Self {
            embedder: Mutex::new(embedder),
            exemplars,
        })
    }

    /// Classify a request by finding the nearest exemplar.
    pub fn classify(&self, req: &ChatRequest) -> Difficulty {
        let text = req
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .filter_map(|m| match &m.content {
                serde_json::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if text.trim().is_empty() {
            return Difficulty::Medium;
        }

        let mut e = self.embedder.lock();
        let emb = match e.embed(vec![text.as_str()], None) {
            Ok(mut v) => match v.pop() {
                Some(x) => x,
                None => return Difficulty::Medium,
            },
            Err(_) => return Difficulty::Medium,
        };
        drop(e);

        let mut best: Option<(f32, Difficulty)> = None;
        for ex in &self.exemplars {
            let sim = cosine(&emb, &ex.embedding);
            if best.map(|(b, _)| sim > b).unwrap_or(true) {
                best = Some((sim, ex.difficulty));
            }
        }
        best.map(|(_, d)| d).unwrap_or(Difficulty::Medium)
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Exemplars for the three difficulty bands, kept small to bound startup
/// embedding time.
fn default_exemplars() -> Vec<(&'static str, Difficulty)> {
    vec![
        // EASY: short factual queries, simple transforms.
        ("what is the capital of france", Difficulty::Easy),
        ("translate 'hello' to spanish", Difficulty::Easy),
        ("define photosynthesis in one sentence", Difficulty::Easy),
        (
            "summarize: the cat sat on the mat. the mat was red.",
            Difficulty::Easy,
        ),
        ("is 17 a prime number?", Difficulty::Easy),
        (
            "classify this sentence as positive or negative: the food was great",
            Difficulty::Easy,
        ),
        ("convert 32 fahrenheit to celsius", Difficulty::Easy),
        // MEDIUM: light reasoning, longer prose, structured outputs.
        ("write a haiku about programming", Difficulty::Medium),
        (
            "compare REST and GraphQL in three bullet points",
            Difficulty::Medium,
        ),
        (
            "explain how DNS resolution works to a junior dev",
            Difficulty::Medium,
        ),
        (
            "draft an email apologizing for a missed deadline",
            Difficulty::Medium,
        ),
        (
            "what are the trade-offs between SQL and NoSQL databases",
            Difficulty::Medium,
        ),
        // HARD: code edits, architectural reasoning, multi-step proofs.
        (
            "refactor this 400-line authentication middleware to use JWT tokens",
            Difficulty::Hard,
        ),
        (
            "design a distributed rate limiter that handles 1M requests per second",
            Difficulty::Hard,
        ),
        (
            "prove that this sorting algorithm terminates in O(n log n) time",
            Difficulty::Hard,
        ),
        (
            "debug this race condition in my concurrent queue implementation",
            Difficulty::Hard,
        ),
        (
            "implement a B-tree with concurrent insertions",
            Difficulty::Hard,
        ),
        (
            "optimize this SQL query that joins seven tables for sub-100ms latency",
            Difficulty::Hard,
        ),
    ]
}
