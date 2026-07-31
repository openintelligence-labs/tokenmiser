//! Two-layer cache: `L1Cache` is an exact-match TTL LRU, `L2Cache` is a
//! per-tenant semantic cache over bge-small embeddings.

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

/// Exact-match cache: a capacity-bounded LRU with per-entry TTL. A single
/// mutex suffices because the critical section is a hash lookup plus a clone.
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

    /// Look up an exact-match entry.
    ///
    /// Every call is counted exactly once, as a hit or a miss, so
    /// `hits + misses == lookups` always holds. Callers must not count misses
    /// themselves.
    pub fn lookup(&self, req: &ChatRequest, tenant: &str) -> Option<ChatResponse> {
        let k = exact_key(req, tenant);
        let mut guard = self.inner.lock();
        let found = match guard.get(&k) {
            Some(entry) if entry.inserted_at.elapsed() <= self.ttl => Some(entry.response.clone()),
            Some(_) => {
                // Evict now rather than waiting for LRU pressure.
                guard.pop(&k);
                None
            }
            None => None,
        };
        drop(guard);
        match found {
            Some(resp) => {
                self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(resp)
            }
            None => {
                self.misses
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            }
        }
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

    #[test]
    fn every_lookup_is_counted_exactly_once() {
        let c = L1Cache::new(8, Duration::from_millis(50));
        c.insert(&req("hello"), "tenant-a", &resp("world"));

        assert!(c.lookup(&req("hello"), "tenant-a").is_some()); // hit
        assert!(c.lookup(&req("nope"), "tenant-a").is_none()); // miss (absent)
        assert!(c.lookup(&req("hello"), "tenant-b").is_none()); // miss (tenant)
        std::thread::sleep(Duration::from_millis(80));
        assert!(c.lookup(&req("hello"), "tenant-a").is_none()); // miss (expired)

        let s = c.stats();
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 3);
        assert_eq!(s.hits + s.misses, 4, "hits+misses must equal lookups");
    }

    #[test]
    fn concurrent_lookups_and_inserts_keep_stats_consistent() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let c = L1Cache::new(1024, Duration::from_secs(60));
        let lookups = Arc::new(AtomicU64::new(0));

        std::thread::scope(|s| {
            for t in 0..8 {
                let c = Arc::clone(&c);
                let lookups = Arc::clone(&lookups);
                s.spawn(move || {
                    for i in 0..500 {
                        let r = req(&format!("prompt-{}", i % 50));
                        if c.lookup(&r, "tenant").is_none() {
                            c.insert(&r, "tenant", &resp("cached"));
                        }
                        lookups.fetch_add(1, Ordering::Relaxed);
                        // Vary interleaving across threads.
                        if i % 100 == t {
                            std::thread::yield_now();
                        }
                    }
                });
            }
        });

        let s = c.stats();
        assert_eq!(
            s.hits + s.misses,
            lookups.load(Ordering::Relaxed),
            "hits+misses must equal total lookups under concurrency"
        );
        assert!(s.size <= 50);
    }
}
