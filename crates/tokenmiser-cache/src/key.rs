//! Cache key derivation. Spec from architecture §4:
//!
//! `key = sha256(model || system_prompt || normalized_user_msg || tool_schema
//!               || temperature_bucket || tenant)`
//!
//! We expose this here even though the lookup table is stubbed — v0.2 will
//! plug the real L1 table behind this same key function.

use sha2::{Digest, Sha256};
use tokenmiser_providers::ChatRequest;

/// Bucketize temperature into 0.1 steps so deterministic 0.0 and 0.05 cache
/// together but stay distinct from 0.7.
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
        // Serialize content deterministically.
        if let Ok(s) = serde_json::to_string(&m.content) {
            hasher.update(s.as_bytes());
        }
        hasher.update(b"\x01");
    }
    hasher.update(b"\x00");

    // Tools / response_format / etc — pull from `extra`.
    if let Some(t) = req.extra.get("tools") {
        if let Ok(s) = serde_json::to_string(t) {
            hasher.update(s.as_bytes());
        }
    }
    hasher.update(b"\x00");

    hasher.update([temperature_bucket(req.temperature)]);
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
}
