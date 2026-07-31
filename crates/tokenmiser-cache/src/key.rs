//! L1 exact-match cache key derivation.

use sha2::{Digest, Sha256};
use tokenmiser_providers::ChatRequest;

/// Bucket temperature into 0.1 steps so 0.0 and 0.05 cache together but stay
/// distinct from 0.7.
fn temperature_bucket(t: Option<f32>) -> u8 {
    let t = t.unwrap_or(1.0).clamp(0.0, 2.0);
    (t * 10.0).round() as u8
}

pub fn exact_key(req: &ChatRequest, tenant: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(req.model.as_bytes());
    hasher.update(b"\x00");

    for m in &req.messages {
        hasher.update(m.role.as_bytes());
        hasher.update(b"\x00");
        if let Ok(s) = serde_json::to_string(&m.content) {
            hasher.update(s.as_bytes());
        }
        hasher.update(b"\x01");
    }
    hasher.update(b"\x00");

    if let Some(t) = req.extra.get("tools") {
        if let Ok(s) = serde_json::to_string(t) {
            hasher.update(s.as_bytes());
        }
    }
    hasher.update(b"\x00");

    hasher.update([temperature_bucket(req.temperature)]);
    hasher.update(b"\x00");

    // `max_tokens` truncates the visible answer and `top_p` changes sampling.
    // Without them, a response truncated at `max_tokens: 5` would be replayed
    // to a `max_tokens: 4096` caller.
    if let Some(mt) = req.max_tokens {
        hasher.update(mt.to_le_bytes());
    }
    hasher.update(b"\x00");
    if let Some(tp) = req.top_p {
        hasher.update(tp.to_le_bytes());
    }
    hasher.update(b"\x00");

    // Every unmodeled body field participates (`response_format`, `n`,
    // `stop`, `seed`, `tool_choice`, …) except transport noise that cannot
    // change the answer. serde_json's map is ordered, so iteration is
    // deterministic.
    const KEY_IGNORED_EXTRA: &[&str] = &[
        "tools",          // hashed above
        "stream_options", // usage reporting, not answer content
        "user",           // telemetry attribution
        "metadata",       // telemetry attribution
        "store",          // provider-side persistence flag
        "logprobs",       // adds metadata, not answer content
        "top_logprobs",
    ];
    for (k, v) in req.extra.iter() {
        if KEY_IGNORED_EXTRA.contains(&k.as_str()) {
            continue;
        }
        hasher.update(k.as_bytes());
        hasher.update(b"\x02");
        if let Ok(s) = serde_json::to_string(v) {
            hasher.update(s.as_bytes());
        }
        hasher.update(b"\x01");
    }
    hasher.update(b"\x00");

    hasher.update(tenant.as_bytes());

    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenmiser_providers::ChatMessage;

    fn req(model: &str, content: &str, temp: Option<f32>) -> ChatRequest {
        ChatRequest {
            model: model.into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(content.into()),
                extra: Default::default(),
            }],
            temperature: temp,
            max_tokens: None,
            top_p: None,
            stream: None,
            extra: Default::default(),
        }
    }

    #[test]
    fn same_input_same_key() {
        let a = exact_key(&req("gpt-5", "hi", Some(0.0)), "tenant-1");
        let b = exact_key(&req("gpt-5", "hi", Some(0.0)), "tenant-1");
        assert_eq!(a, b);
    }

    #[test]
    fn different_tenant_different_key() {
        let a = exact_key(&req("gpt-5", "hi", Some(0.0)), "tenant-1");
        let b = exact_key(&req("gpt-5", "hi", Some(0.0)), "tenant-2");
        assert_ne!(a, b);
    }

    #[test]
    fn nearby_temperatures_bucket_together() {
        let a = exact_key(&req("gpt-5", "hi", Some(0.00)), "t");
        let b = exact_key(&req("gpt-5", "hi", Some(0.04)), "t");
        assert_eq!(a, b);
    }

    #[test]
    fn distant_temperatures_split() {
        let a = exact_key(&req("gpt-5", "hi", Some(0.0)), "t");
        let b = exact_key(&req("gpt-5", "hi", Some(0.7)), "t");
        assert_ne!(a, b);
    }

    #[test]
    fn max_tokens_and_top_p_split_the_key() {
        let mut a = req("gpt-5", "hi", Some(0.0));
        let mut b = req("gpt-5", "hi", Some(0.0));
        a.max_tokens = Some(5);
        b.max_tokens = Some(4096);
        assert_ne!(exact_key(&a, "t"), exact_key(&b, "t"));

        let mut c = req("gpt-5", "hi", Some(0.0));
        let mut d = req("gpt-5", "hi", Some(0.0));
        c.top_p = Some(0.1);
        d.top_p = Some(1.0);
        assert_ne!(exact_key(&c, "t"), exact_key(&d, "t"));
    }

    #[test]
    fn answer_shaping_extra_params_split_the_key() {
        let mut a = req("gpt-5", "hi", Some(0.0));
        let b = req("gpt-5", "hi", Some(0.0));
        a.extra.insert(
            "response_format".into(),
            serde_json::json!({"type": "json_object"}),
        );
        assert_ne!(exact_key(&a, "t"), exact_key(&b, "t"));

        for (k, v) in [
            ("n", serde_json::json!(2)),
            ("stop", serde_json::json!(["\n"])),
            ("seed", serde_json::json!(7)),
        ] {
            let mut with = req("gpt-5", "hi", Some(0.0));
            with.extra.insert(k.into(), v);
            assert_ne!(
                exact_key(&with, "t"),
                exact_key(&req("gpt-5", "hi", Some(0.0)), "t"),
                "`{k}` must participate in the cache key"
            );
        }
    }

    #[test]
    fn stream_and_transport_noise_do_not_split_the_key() {
        let plain = req("gpt-5", "hi", Some(0.0));
        let mut streaming = req("gpt-5", "hi", Some(0.0));
        streaming.stream = Some(true);
        streaming.extra.insert(
            "stream_options".into(),
            serde_json::json!({"include_usage": true}),
        );
        streaming
            .extra
            .insert("user".into(), serde_json::json!("abc"));
        assert_eq!(exact_key(&plain, "t"), exact_key(&streaming, "t"));
    }
}
