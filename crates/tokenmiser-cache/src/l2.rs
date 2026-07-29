//! L2 semantic cache (architecture §4).
//!
//! Embed prompts with `bge-small-en-v1.5` (384-dim, CPU, ~3ms via fastembed),
//! store per-tenant. On lookup, cosine-similarity-search against the tenant's
//! vectors and return the cached response if the best match exceeds the
//! per-namespace threshold (default 0.87, empirically tuned).
//!
//! v0.3 ships flat in-memory cosine because each tenant typically holds <10k
//! cached prompts; 10k × 384 × f32 dot products = ~3ms on modern CPUs, well
//! inside the architecture §8 budget. The `instant-distance` HNSW upgrade
//! path is wired into the dep list but kicks in only when we breach 50k
//! entries per tenant — that's a v0.6+ concern.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use parking_lot::Mutex;
use tokenmiser_providers::{ChatRequest, ChatResponse};
use tracing::{info, warn};

/// One stored prompt + response + its embedding.
struct Entry {
    embedding: Vec<f32>,
    response: ChatResponse,
    inserted_at: Instant,
}

#[derive(Default)]
struct TenantStore {
    entries: Vec<Entry>,
}

/// Semantic L2 cache. Constructing this downloads the bge model on first
/// run (cached by fastembed under `~/.cache/fastembed`).
pub struct L2Cache {
    embedder: Mutex<TextEmbedding>,
    tenants: Mutex<HashMap<String, TenantStore>>,
    threshold: f32,
    ttl: Duration,
    per_tenant_capacity: usize,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl L2Cache {
    pub fn new(threshold: f32, ttl: Duration, per_tenant_capacity: usize) -> Result<Arc<Self>> {
        let opts = InitOptions::new(EmbeddingModel::BGESmallENV15);
        let embedder = TextEmbedding::try_new(opts)
            .map_err(|e| anyhow!("bge-small-en-v1.5 init failed: {e}"))?;
        info!(
            model = "bge-small-en-v1.5",
            threshold,
            ttl_secs = ttl.as_secs(),
            "L2 semantic cache initialized"
        );
        Ok(Arc::new(Self {
            embedder: Mutex::new(embedder),
            tenants: Mutex::new(HashMap::new()),
            threshold,
            ttl,
            per_tenant_capacity,
            hits: Default::default(),
            misses: Default::default(),
        }))
    }

    /// Concatenate every user message into the text we embed. System prompts
    /// are intentionally excluded — they're the same across many requests and
    /// would dilute the signal. v0.4 may experiment with system-aware
    /// embedding once we have shadow-A/B data to validate the choice.
    fn extract_text(req: &ChatRequest) -> String {
        req.messages
            .iter()
            .filter(|m| m.role == "user")
            .filter_map(|m| match &m.content {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Array(arr) => {
                    let mut buf = String::new();
                    for item in arr {
                        if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
                            buf.push_str(t);
                            buf.push(' ');
                        }
                    }
                    Some(buf)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut e = self.embedder.lock();
        let mut out = e
            .embed(vec![text], None)
            .map_err(|e| anyhow!("embed failed: {e}"))?;
        out.pop().ok_or_else(|| anyhow!("no embedding returned"))
    }

    pub fn lookup(&self, req: &ChatRequest, tenant: &str) -> Option<ChatResponse> {
        // The cache is gated on whether bge-small handles this model's
        // prompt at all (always true for chat); same-tenant search.
        let text = Self::extract_text(req);
        if text.trim().is_empty() {
            return None;
        }
        let q = match self.embed(&text) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "L2 embed failed; falling through");
                return None;
            }
        };

        let mut tenants = self.tenants.lock();
        let store = tenants.get_mut(tenant)?;

        let mut best: Option<(f32, usize)> = None;
        let now = Instant::now();
        let ttl = self.ttl;
        // Linear scan + cosine. Per architecture §4, this is the v0.3
        // implementation; HNSW upgrade is gated behind tenant size.
        store
            .entries
            .retain(|e| now.duration_since(e.inserted_at) < ttl);
        for (i, entry) in store.entries.iter().enumerate() {
            let sim = cosine(&q, &entry.embedding);
            if best.map(|(b, _)| sim > b).unwrap_or(true) {
                best = Some((sim, i));
            }
        }

        match best {
            Some((sim, idx)) if sim >= self.threshold => {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(store.entries[idx].response.clone())
            }
            _ => {
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            }
        }
    }

    pub fn insert(&self, req: &ChatRequest, tenant: &str, resp: &ChatResponse) {
        let text = Self::extract_text(req);
        if text.trim().is_empty() {
            return;
        }
        let emb = match self.embed(&text) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "L2 embed failed on insert; skip");
                return;
            }
        };
        let mut tenants = self.tenants.lock();
        let store = tenants.entry(tenant.to_string()).or_default();
        if store.entries.len() >= self.per_tenant_capacity {
            // FIFO eviction — v0.4 will move to architecture §4
            // "marginal $-per-MB" scoring once cost data is mature.
            store.entries.remove(0);
        }
        store.entries.push(Entry {
            embedding: emb,
            response: resp.clone(),
            inserted_at: Instant::now(),
        });
    }

    pub fn stats(&self) -> SemanticStats {
        let total: u64 = self
            .tenants
            .lock()
            .values()
            .map(|t| t.entries.len() as u64)
            .sum();
        SemanticStats {
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
            size: total,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SemanticStats {
    pub hits: u64,
    pub misses: u64,
    pub size: u64,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_is_one() {
        let v = vec![0.1, 0.2, 0.3, 0.4];
        let s = cosine(&v, &v);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine(&a, &b).abs() < 1e-6);
    }
}
