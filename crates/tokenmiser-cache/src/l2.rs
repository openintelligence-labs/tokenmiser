//! L2 semantic cache: prompts embedded with `bge-small-en-v1.5` and matched
//! per-tenant by cosine similarity against a threshold.
//!
//! The index is a flat in-memory scan. At the expected <10k entries per tenant
//! a full 384-dim pass costs ~3ms; HNSW (`instant-distance`) only pays off
//! past ~50k.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use parking_lot::Mutex;
use tokenmiser_providers::{ChatRequest, ChatResponse};
use tracing::{info, warn};

struct Entry {
    embedding: Vec<f32>,
    /// Canonicalized number literals, precomputed so the lexical guard costs
    /// nothing at lookup.
    numbers: Vec<String>,
    shape: u64,
    response: ChatResponse,
    inserted_at: Instant,
}

/// Hash the request fields that change the shape of a valid answer. Semantic
/// matching is deliberately fuzzy about wording and sampling params, but a
/// prose answer must never be served to a JSON-mode or tool-calling caller.
/// Deterministic within a process, which is all an in-memory cache needs.
fn shape_fingerprint(req: &ChatRequest) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for field in ["tools", "tool_choice", "response_format"] {
        field.hash(&mut h);
        match req.extra.get(field) {
            Some(v) => serde_json::to_string(v).unwrap_or_default().hash(&mut h),
            None => "".hash(&mut h),
        }
    }
    h.finish()
}

#[derive(Default)]
struct TenantStore {
    entries: Vec<Entry>,
    last_used: Option<Instant>,
}

/// Cap on distinct tenants held in memory. The tenant id comes straight from
/// the caller-supplied `x-tokenmiser-tenant` header, so an unbounded map would
/// let any client allocate a fresh store per unique header value.
const MAX_TENANTS: usize = 256;

/// Semantic L2 cache. Constructing this downloads the bge model on first run
/// (cached by fastembed under `~/.cache/fastembed`).
pub struct L2Cache {
    embedder: Mutex<TextEmbedding>,
    tenants: Mutex<HashMap<String, TenantStore>>,
    threshold: f32,
    ttl: Duration,
    per_tenant_capacity: usize,
    /// Skip candidates whose prompt carries a different multiset of number
    /// literals. Kills the instruction-template false-positive class
    /// ("Multiply 3 by 11" matching a cached "Add 4 and 9" above threshold)
    /// while leaving same-numbers and number-free paraphrases untouched.
    numeric_guard: bool,
    hits: std::sync::atomic::AtomicU64,
    misses: std::sync::atomic::AtomicU64,
}

impl L2Cache {
    pub fn new(
        threshold: f32,
        ttl: Duration,
        per_tenant_capacity: usize,
        numeric_guard: bool,
    ) -> Result<Arc<Self>> {
        let opts = InitOptions::new(EmbeddingModel::BGESmallENV15);
        let embedder = TextEmbedding::try_new(opts)
            .map_err(|e| anyhow!("bge-small-en-v1.5 init failed: {e}"))?;
        info!(
            model = "bge-small-en-v1.5",
            threshold,
            ttl_secs = ttl.as_secs(),
            numeric_guard,
            "L2 semantic cache initialized"
        );
        Ok(Arc::new(Self {
            embedder: Mutex::new(embedder),
            tenants: Mutex::new(HashMap::new()),
            threshold,
            ttl,
            per_tenant_capacity,
            numeric_guard,
            hits: Default::default(),
            misses: Default::default(),
        }))
    }

    /// Concatenate every user message into the embedded text. System prompts
    /// are excluded: they repeat across requests and dilute the signal.
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

    /// Record a miss. Every `lookup` outcome funnels through this or the hit
    /// path, so `hits + misses == lookups` always holds.
    fn miss(&self) -> Option<ChatResponse> {
        self.misses
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    pub fn lookup(&self, req: &ChatRequest, tenant: &str) -> Option<ChatResponse> {
        let text = Self::extract_text(req);
        if text.trim().is_empty() {
            return self.miss();
        }
        let q = match self.embed(&text) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "L2 embed failed; falling through");
                return self.miss();
            }
        };

        let mut tenants = self.tenants.lock();
        let store = match tenants.get_mut(tenant) {
            Some(s) => s,
            None => {
                drop(tenants);
                return self.miss();
            }
        };

        let q_numbers = self.numeric_guard.then(|| extract_numbers(&text));
        let q_shape = shape_fingerprint(req);

        let mut best: Option<(f32, usize)> = None;
        let now = Instant::now();
        let ttl = self.ttl;
        store
            .entries
            .retain(|e| now.duration_since(e.inserted_at) < ttl);
        for (i, entry) in store.entries.iter().enumerate() {
            // Both guards skip candidates rather than rejecting the final
            // best, so a correct-but-slightly-farther entry can still win over
            // a closer-but-wrong one. An L2 hit seeds L1, so a bad match here
            // would become sticky under the exact key.
            if entry.shape != q_shape {
                continue;
            }
            if let Some(qn) = &q_numbers {
                if entry.numbers != *qn {
                    continue;
                }
            }
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
        evict_coldest_tenant_if_full(&mut tenants, tenant);
        let store = tenants.entry(tenant.to_string()).or_default();
        store.last_used = Some(Instant::now());
        if store.entries.len() >= self.per_tenant_capacity {
            store.entries.remove(0);
        }
        store.entries.push(Entry {
            embedding: emb,
            numbers: extract_numbers(&text),
            shape: shape_fingerprint(req),
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

/// Evict the least-recently-written tenant once `MAX_TENANTS` is reached.
/// Existing tenants are never evicted by their own writes.
fn evict_coldest_tenant_if_full(tenants: &mut HashMap<String, TenantStore>, incoming: &str) {
    if tenants.len() < MAX_TENANTS || tenants.contains_key(incoming) {
        return;
    }
    if let Some(coldest) = tenants
        .iter()
        .min_by_key(|(_, s)| s.last_used)
        .map(|(k, _)| k.clone())
    {
        tenants.remove(&coldest);
    }
}

/// Every number literal in `text` as a canonicalized, sorted multiset:
/// `"3.50 vs 3.5"` → `["3.5", "3.5"]`.
///
/// Embedding models are sloppy about digits — prompts differing only in their
/// numbers embed nearly identically — so similarity alone cannot separate
/// "Add 4 and 9" from "Multiply 3 by 11".
fn extract_numbers(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // A leading '-' is a sign only when not itself preceded by an
        // alphanumeric, '.' or '-', so "-5" and "(-5)" keep their sign while
        // "5-3", "2026-07-30", "555-1234" and "1e-5" treat it as a separator.
        let negative = i > 0
            && bytes[i - 1] == b'-'
            && (i == 1 || {
                let p = bytes[i - 2];
                !(p.is_ascii_alphanumeric() || p == b'.' || p == b'-')
            });
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        let mut num = text[start..i].to_string();
        // "12,345,678" is one literal iff the leading group has 1-3 digits
        // and every comma is followed by exactly three; "25,30" stays two.
        if num.len() <= 3 {
            let mut j = i;
            while j + 3 < bytes.len()
                && bytes[j] == b','
                && bytes[j + 1].is_ascii_digit()
                && bytes[j + 2].is_ascii_digit()
                && bytes[j + 3].is_ascii_digit()
                && !(j + 4 < bytes.len() && bytes[j + 4].is_ascii_digit())
            {
                j += 4;
            }
            if j > i {
                num = text[start..j].replace(',', "");
                i = j;
            }
        }
        // A '.' is a decimal point only when followed by a digit, so
        // sentence-final "Add 3." parses as "3".
        if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
            let frac_start = i;
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            num.push_str(&text[frac_start..i]);
        }
        if negative {
            num.insert(0, '-');
        }
        // Canonicalize through f64 so "007" == "7"; fall back to the raw
        // digits for values f64 cannot round-trip.
        let canon = num
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .map(|v| format!("{v}"))
            .unwrap_or(num);
        out.push(canon);
    }
    out.sort_unstable();
    out
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
    use tokenmiser_providers::{ChatChoice, ChatMessage, ChatRequest, Usage};

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

    fn store_at(t: Instant) -> TenantStore {
        TenantStore {
            entries: Vec::new(),
            last_used: Some(t),
        }
    }

    #[test]
    fn tenant_map_is_bounded_and_evicts_the_coldest() {
        let mut tenants: HashMap<String, TenantStore> = HashMap::new();
        let base = Instant::now();
        // Increasing recency, so tenant-0 is the coldest.
        for i in 0..MAX_TENANTS {
            tenants.insert(
                format!("tenant-{i}"),
                store_at(base + Duration::from_millis(i as u64)),
            );
        }
        assert_eq!(tenants.len(), MAX_TENANTS);

        evict_coldest_tenant_if_full(&mut tenants, "attacker-new");
        assert_eq!(
            tenants.len(),
            MAX_TENANTS - 1,
            "a new tenant at capacity must evict exactly one"
        );
        assert!(
            !tenants.contains_key("tenant-0"),
            "the least-recently-written tenant must be the one evicted"
        );
        assert!(tenants.contains_key(&format!("tenant-{}", MAX_TENANTS - 1)));

        for i in 0..1000 {
            let name = format!("attacker-{i}");
            evict_coldest_tenant_if_full(&mut tenants, &name);
            tenants.insert(name, store_at(Instant::now()));
            assert!(
                tenants.len() <= MAX_TENANTS,
                "tenant map exceeded its cap at iteration {i}"
            );
        }
    }

    #[test]
    fn existing_tenant_write_does_not_evict() {
        let mut tenants: HashMap<String, TenantStore> = HashMap::new();
        let base = Instant::now();
        for i in 0..MAX_TENANTS {
            tenants.insert(
                format!("tenant-{i}"),
                store_at(base + Duration::from_millis(i as u64)),
            );
        }
        evict_coldest_tenant_if_full(&mut tenants, "tenant-0");
        assert_eq!(tenants.len(), MAX_TENANTS);
        assert!(tenants.contains_key("tenant-0"));
    }

    #[test]
    fn extract_numbers_basic() {
        assert_eq!(
            extract_numbers("Multiply 3 by 11. Reply with the number only."),
            vec!["11".to_string(), "3".to_string()]
        );
        assert_eq!(
            extract_numbers("Add 25 and 30. Reply with the number only."),
            vec!["25".to_string(), "30".to_string()]
        );
        assert!(extract_numbers("What is the capital of France? One word only.").is_empty());
    }

    #[test]
    fn extract_numbers_canonicalizes() {
        assert_eq!(extract_numbers("007 and 7"), vec!["7", "7"]);
        assert_eq!(extract_numbers("3.50 vs 3.5"), vec!["3.5", "3.5"]);
        assert_eq!(extract_numbers("Add 3."), vec!["3"]);
        assert_eq!(
            extract_numbers("30 plus 12"),
            extract_numbers("What is 12 plus 30?")
        );
    }

    #[test]
    fn guard_rejects_multiply_vs_add_template_case() {
        let query = extract_numbers("Multiply 3 by 11. Reply with the number only.");
        for i in 0..40u32 {
            let cached = extract_numbers(&format!(
                "Add {} and {}. Reply with the number only.",
                i * 3 + 1,
                i * 7 + 2
            ));
            assert_ne!(
                query, cached,
                "guard must reject every Add-template entry for the Multiply query"
            );
        }
    }

    #[test]
    fn guard_distinguishes_negative_numbers() {
        assert_ne!(
            extract_numbers("What is -5 plus 3? Number only."),
            extract_numbers("What is 5 plus 3? Number only.")
        );
        assert_eq!(extract_numbers("What is -5 plus 3?"), vec!["-5", "3"]);
        assert_eq!(extract_numbers("(-5) times 2"), vec!["-5", "2"]);
        assert_eq!(extract_numbers("-40 degrees"), vec!["-40"]);
        // Subtraction, dates, phone numbers and exponents: '-' is a separator.
        assert_eq!(extract_numbers("compute 10-4"), vec!["10", "4"]);
        assert_eq!(extract_numbers("on 2026-07-30"), vec!["2026", "30", "7"]);
        assert_eq!(extract_numbers("call 555-1234"), vec!["1234", "555"]);
        assert_eq!(extract_numbers("about 1e-5 units"), vec!["1", "5"]);
    }

    #[test]
    fn guard_groups_thousands_separators() {
        assert_eq!(
            extract_numbers("Add 1,000 and 5."),
            extract_numbers("Add 1000 and 5.")
        );
        assert_eq!(extract_numbers("population 12,345,678"), vec!["12345678"]);
        assert_eq!(extract_numbers("costs 1,234.56 dollars"), vec!["1234.56"]);
        // Comma lists are not thousands groups.
        assert_eq!(extract_numbers("pick 25,30"), vec!["25", "30"]);
        assert_eq!(extract_numbers("pick 1,2 or 3"), vec!["1", "2", "3"]);
        assert_eq!(extract_numbers("ids 1,2345"), vec!["1", "2345"]);
        // Splitting "1,000" into ["1","0"] would collide with this prompt.
        assert_ne!(
            extract_numbers("Add 1,000 and 5. Reply with the number only."),
            extract_numbers("Add 1 and 0 and 5. Reply with the number only.")
        );
    }

    #[test]
    fn shape_fingerprint_tracks_answer_shaping_params() {
        let plain = req("hello");
        let mut json_mode = req("hello");
        json_mode.extra.insert(
            "response_format".into(),
            serde_json::json!({"type": "json_object"}),
        );
        let mut with_tools = req("hello");
        with_tools.extra.insert(
            "tools".into(),
            serde_json::json!([{"type": "function", "function": {"name": "f"}}]),
        );
        assert_ne!(shape_fingerprint(&plain), shape_fingerprint(&json_mode));
        assert_ne!(shape_fingerprint(&plain), shape_fingerprint(&with_tools));
        assert_ne!(
            shape_fingerprint(&json_mode),
            shape_fingerprint(&with_tools)
        );
        // Sampling and transport params stay out: L2 is fuzzy there.
        let mut sampled = req("hello");
        sampled.temperature = Some(0.9);
        sampled.max_tokens = Some(5);
        sampled.stream = Some(true);
        sampled.extra.insert("seed".into(), serde_json::json!(1234));
        assert_eq!(shape_fingerprint(&plain), shape_fingerprint(&sampled));
    }

    #[test]
    fn guard_accepts_numeric_paraphrase() {
        assert_eq!(
            extract_numbers("What is 12 plus 30? Number only."),
            extract_numbers("Compute 12 + 30 and reply with just the number.")
        );
        assert_eq!(
            extract_numbers("What is the capital of France? One word only."),
            extract_numbers("Tell me the capital city of France, answer in a single word.")
        );
    }

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

    fn resp(text: &str) -> ChatResponse {
        ChatResponse {
            id: "x".into(),
            object: "chat.completion".into(),
            created: 0,
            model: "x".into(),
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
    #[ignore = "needs the bge-small model on disk (~seconds); run with --ignored"]
    fn live_template_false_positive_is_killed_by_guard() {
        // Guard off: the false positive must actually reproduce at 0.87,
        // otherwise the guarded half of this test is vacuous.
        let unguarded = L2Cache::new(0.87, Duration::from_secs(3600), 1024, false).unwrap();
        for i in 0..40u32 {
            let p = format!(
                "Add {} and {}. Reply with the number only.",
                i * 3 + 1,
                i * 7 + 2
            );
            unguarded.insert(&req(&p), "t", &resp("wrong"));
        }
        let q = req("Multiply 3 by 11. Reply with the number only.");
        assert!(
            unguarded.lookup(&q, "t").is_some(),
            "expected the unguarded cache to reproduce the false positive \
             (if this stops reproducing, the guard test below is vacuous)"
        );

        // Guard on (the default): the same lookup must miss.
        let guarded = L2Cache::new(0.87, Duration::from_secs(3600), 1024, true).unwrap();
        for i in 0..40u32 {
            let p = format!(
                "Add {} and {}. Reply with the number only.",
                i * 3 + 1,
                i * 7 + 2
            );
            guarded.insert(&req(&p), "t", &resp("wrong"));
        }
        assert!(
            guarded.lookup(&q, "t").is_none(),
            "numeric guard must reject the Multiply-vs-Add template hit"
        );
    }

    #[test]
    #[ignore = "diagnostic: prints cosine sims for candidate pairs"]
    fn live_sim_probe() {
        let opts = InitOptions::new(EmbeddingModel::BGESmallENV15);
        let mut e = TextEmbedding::try_new(opts).unwrap();
        let pairs = [
            (
                "What is -5 plus 3? Number only.",
                "What is 5 plus 3? Number only.",
            ),
            (
                "Add 1,000 and 5. Reply with the number only.",
                "Add 1000 and 5. Reply with the number only.",
            ),
            (
                "What is 12 plus 30? Number only.",
                "Compute 12 + 30 and reply with just the number.",
            ),
            (
                "What is 12 plus 30? Number only.",
                "What is 12 plus 30? Reply with the number only.",
            ),
            (
                "What is 12 plus 30? Number only.",
                "12 plus 30 equals what? Number only.",
            ),
            (
                "Add 12 and 30. Reply with the number only.",
                "Compute 12 + 30 and reply with just the number.",
            ),
            (
                "What is the capital of France? One word only.",
                "Tell me the capital city of France, answer in a single word.",
            ),
            (
                "Multiply 3 by 11. Reply with the number only.",
                "Add 25 and 30. Reply with the number only.",
            ),
        ];
        for (a, b) in pairs {
            let v = e.embed(vec![a, b], None).unwrap();
            println!("{:.4}  {a:?} vs {b:?}", cosine(&v[0], &v[1]));
        }
    }

    #[test]
    #[ignore = "needs the bge-small model on disk (~seconds); run with --ignored"]
    fn live_shape_mismatch_never_hits() {
        let cache = L2Cache::new(0.87, Duration::from_secs(3600), 1024, true).unwrap();
        cache.insert(&req("List three colors."), "t", &resp("red, green, blue"));
        assert!(cache.lookup(&req("List three colors."), "t").is_some());
        let mut json_mode = req("List three colors.");
        json_mode.extra.insert(
            "response_format".into(),
            serde_json::json!({"type": "json_object"}),
        );
        assert!(
            cache.lookup(&json_mode, "t").is_none(),
            "L2 must not serve a prose entry to a JSON-mode request"
        );
    }

    #[test]
    #[ignore = "needs the bge-small model on disk (~seconds); run with --ignored"]
    fn live_paraphrases_still_hit_with_guard() {
        let cache = L2Cache::new(0.87, Duration::from_secs(3600), 1024, true).unwrap();

        cache.insert(
            &req("What is the capital of France? One word only."),
            "t",
            &resp("Paris"),
        );
        let hit = cache.lookup(
            &req("Tell me the capital city of France, answer in a single word."),
            "t",
        );
        assert!(hit.is_some(), "number-free paraphrase must still hit");

        // This paraphrase scores *higher* than the Multiply-vs-Add false
        // positive does, so no threshold can separate the two cases — which
        // is why the numeric guard exists.
        cache.insert(
            &req("Add 12 and 30. Reply with the number only."),
            "t2",
            &resp("42"),
        );
        let hit = cache.lookup(
            &req("Compute 12 + 30 and reply with just the number."),
            "t2",
        );
        assert!(hit.is_some(), "same-numbers paraphrase must still hit");
    }
}
