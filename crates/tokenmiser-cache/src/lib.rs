//! Cache layer.
//!
//! v0.2: `L1Cache` — exact-match LRU keyed by sha256(model || system ||
//! normalized user msg || tools || temp bucket || tenant). TTL bounded.
//!
//! v0.3: `L2Cache` — semantic, bge-small-en-v1.5 + cosine, per-tenant
//! namespace, threshold 0.87 by default (empirically tuned, see
//! `threshold_bench`).

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lru::LruCache;
use parking_lot::Mutex;
use tokenmiser_providers::{ChatRequest, ChatResponse};

pub mod key;
pub mod l2;
#[cfg(test)]
mod threshold_bench;

pub use key::exact_key;
pub use l2::{L2Cache, SemanticStats};

#[derive(Debug, Clone)]
struct Entry {
    response: ChatResponse,
    inserted_at: Instant,
}

/// L1 exact-match cache (architecture §4).
///
/// Capacity-bounded LRU with per-entry TTL. Thread-safe via `parking_lot::Mutex`
/// — the critical section is microseconds (hash lookup + clone), so contention
/// is acceptable at v0.2 scale. v0.3 will move to a sharded structure if
/// benchmarks show this is a bottleneck.
pub struct L1Cache {
    inner: Mutex<LruCache<String, Entry>>,
    ttl: Duration,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl L1Cache {
    pub fn new(capacity: usize, ttl: Duration) -> Arc<Self> {
        let cap = NonZeroUsize::new(capacity.max(1)).unwrap();
        Arc::new(Self {
            inner: Mutex::new(LruCache::new(cap)),
            ttl,
            hits: Default::default(),
            misses: Default::default(),
        })
    }

    pub fn lookup(&self, req: &ChatRequest, tenant: &str) -> Option<ChatResponse> {
        let k = exact_key(req, tenant);
        let mut guard = self.inner.lock();
        let entry = guard.get(&k)?;
        if entry.inserted_at.elapsed() > self.ttl {
            guard.pop(&k);
            self.misses
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
        let resp = entry.response.clone();
        self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(resp)
    }

    pub fn insert(&self, req: &ChatRequest, tenant: &str, resp: &ChatResponse) {
        let k = exact_key(req, tenant);
        let mut guard = self.inner.lock();
        guard.put(
            k,
            Entry {
                response: resp.clone(),
                inserted_at: Instant::now(),
            },
        );
    }

    pub fn record_miss(&self) {
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(std::sync::atomic::Ordering::Relaxed),
            misses: self.misses.load(std::sync::atomic::Ordering::Relaxed),
            size: self.inner.lock().len() as u64,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub size: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use tokenmiser_providers::{ChatChoice, ChatMessage, Usage};

    fn req(content: &str) -> ChatRequest {
        ChatRequest {
            model: "gpt-5".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Value::String(content.into()),
                extra: Default::default(),
            }],
            temperature: Some(0.0),
            max_tokens: Some(100),
            top_p: None,
            stream: None,
            extra: Default::default(),
        }
    }

    fn resp(text: &str) -> ChatResponse {
        ChatResponse {
            id: "test".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "gpt-5".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Value::String(text.into()),
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
    fn hit_returns_stored_response() {
        let c = L1Cache::new(8, Duration::from_secs(60));
        c.insert(&req("hello"), "tenant-a", &resp("world"));
        let got = c.lookup(&req("hello"), "tenant-a").expect("hit");
        assert_eq!(
            got.choices[0].message.content,
            Value::String("world".into())
        );
        let stats = c.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn tenant_isolation() {
        let c = L1Cache::new(8, Duration::from_secs(60));
        c.insert(&req("hello"), "tenant-a", &resp("a"));
        assert!(c.lookup(&req("hello"), "tenant-b").is_none());
    }

    #[test]
    fn ttl_expires_entries() {
        let c = L1Cache::new(8, Duration::from_millis(50));
        c.insert(&req("hello"), "tenant-a", &resp("world"));
        std::thread::sleep(Duration::from_millis(80));
        assert!(c.lookup(&req("hello"), "tenant-a").is_none());
    }
}
