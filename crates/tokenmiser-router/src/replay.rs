//! `tokenmiser policy test` — replay logged requests against a candidate
//! Rhai policy file and print the projected routing delta.
//!
//! The daemon (when configured with `request_log_path`) writes one JSON
//! line per request: `{ "ts": ..., "model": "...", "tenant": "...",
//! "messages": [...] }`. Replay reads that file, runs each request through
//! the candidate policy, and tallies how many would have routed differently
//! plus the projected cost delta if you have a `pricing` table available.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokenmiser_providers::ChatRequest;

use crate::dsl::PolicyEngine;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayResult {
    pub total: u64,
    pub by_target: HashMap<String, u64>,
    pub failed: u64,
    pub by_tenant: HashMap<String, u64>,
    /// Unix-seconds range covered by the replayed entries, if any carried a `ts`.
    pub time_range: Option<(i64, i64)>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ReplayFilter {
    /// Inclusive lower-bound (unix seconds). Entries without `ts` always pass.
    pub since: Option<i64>,
    /// Inclusive upper-bound (unix seconds).
    pub until: Option<i64>,
}

pub fn replay<P: AsRef<Path>>(log_path: P, policy: &PolicyEngine) -> Result<ReplayResult> {
    replay_filtered(log_path, policy, ReplayFilter::default())
}

pub fn replay_filtered<P: AsRef<Path>>(
    log_path: P,
    policy: &PolicyEngine,
    filter: ReplayFilter,
) -> Result<ReplayResult> {
    let raw = std::fs::read_to_string(&log_path)
        .with_context(|| format!("read log {}", log_path.as_ref().display()))?;

    let mut out = ReplayResult::default();
    let mut min_ts: Option<i64> = None;
    let mut max_ts: Option<i64> = None;

    for (line_no, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let entry: LogEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(line = line_no, error = %e, "replay: skip malformed log line");
                out.failed += 1;
                continue;
            }
        };
        // Apply time filter.
        if let Some(ts) = entry.ts {
            if let Some(since) = filter.since {
                if ts < since {
                    continue;
                }
            }
            if let Some(until) = filter.until {
                if ts > until {
                    continue;
                }
            }
            min_ts = Some(min_ts.map_or(ts, |m| m.min(ts)));
            max_ts = Some(max_ts.map_or(ts, |m| m.max(ts)));
        }

        out.total += 1;
        let tenant = entry.tenant.clone().unwrap_or_else(|| "default".into());
        *out.by_tenant.entry(tenant.clone()).or_default() += 1;

        let req = entry.into_request();
        match policy.route(&req, &tenant) {
            Ok(target) => {
                let key = format!("{}::{}", target.provider, target.model);
                *out.by_target.entry(key).or_default() += 1;
            }
            Err(e) => {
                tracing::warn!(line = line_no, error = %e, "replay: policy failed");
                out.failed += 1;
            }
        }
    }
    if let (Some(lo), Some(hi)) = (min_ts, max_ts) {
        out.time_range = Some((lo, hi));
    }
    Ok(out)
}

#[derive(Debug, Deserialize)]
struct LogEntry {
    /// Unix-seconds when this request was logged. Optional so we accept
    /// hand-written test fixtures without a timestamp.
    #[serde(default)]
    ts: Option<i64>,
    model: String,
    /// Tenant the request was scoped to. Honored by the policy when set so
    /// `req.tenant` in Rhai scripts reflects production reality.
    #[serde(default)]
    tenant: Option<String>,
    messages: Vec<serde_json::Value>,
}

impl LogEntry {
    fn into_request(self) -> ChatRequest {
        let messages = self
            .messages
            .into_iter()
            .filter_map(|v| serde_json::from_value(v).ok())
            .collect();
        ChatRequest {
            model: self.model,
            messages,
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: None,
            extra: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn replays_and_tallies() {
        let mut tmp = tempfile_like();
        writeln!(
            tmp.file,
            r#"{{"ts":1700000000,"tenant":"t-a","model":"auto","messages":[{{"role":"user","content":"hello"}}]}}"#
        )
        .unwrap();
        writeln!(
            tmp.file,
            r#"{{"ts":1700000100,"tenant":"t-b","model":"auto","messages":[{{"role":"user","content":"refactor this"}}]}}"#
        )
        .unwrap();
        tmp.file.flush().unwrap();

        let policy_path = std::env::temp_dir().join(format!(
            "replay-policy-{}.rhai",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(
            &policy_path,
            r#"
            fn route(req) {
                if req.has_keyword("refactor") {
                    return #{ provider: "anthropic", model: "claude-opus-4-7" };
                }
                #{ provider: "ollama", model: "ollama:qwen2.5:7b" }
            }
        "#,
        )
        .unwrap();

        let p = PolicyEngine::load(policy_path.clone()).unwrap();
        let r = replay(&tmp.path, &p).unwrap();
        assert_eq!(r.total, 2);
        assert_eq!(r.failed, 0);
        assert_eq!(
            r.by_target.get("ollama::ollama:qwen2.5:7b").copied(),
            Some(1)
        );
        assert_eq!(
            r.by_target.get("anthropic::claude-opus-4-7").copied(),
            Some(1)
        );
        // ts + tenant now actually drive replay output:
        assert_eq!(r.by_tenant.get("t-a").copied(), Some(1));
        assert_eq!(r.by_tenant.get("t-b").copied(), Some(1));
        assert_eq!(r.time_range, Some((1700000000, 1700000100)));

        let _ = std::fs::remove_file(&tmp.path);
        let _ = std::fs::remove_file(&policy_path);
    }

    #[test]
    fn replay_filter_excludes_by_time() {
        let mut tmp = tempfile_like();
        writeln!(
            tmp.file,
            r#"{{"ts":1000,"model":"auto","messages":[{{"role":"user","content":"a"}}]}}"#
        )
        .unwrap();
        writeln!(
            tmp.file,
            r#"{{"ts":2000,"model":"auto","messages":[{{"role":"user","content":"b"}}]}}"#
        )
        .unwrap();
        writeln!(
            tmp.file,
            r#"{{"ts":3000,"model":"auto","messages":[{{"role":"user","content":"c"}}]}}"#
        )
        .unwrap();
        tmp.file.flush().unwrap();

        let policy_path = std::env::temp_dir().join(format!(
            "replay-policy-filter-{}.rhai",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(
            &policy_path,
            r#"fn route(req) { #{ provider: "x", model: "y" } }"#,
        )
        .unwrap();
        let p = PolicyEngine::load(policy_path.clone()).unwrap();

        let r = replay_filtered(
            &tmp.path,
            &p,
            ReplayFilter {
                since: Some(1500),
                until: Some(2500),
            },
        )
        .unwrap();
        assert_eq!(r.total, 1);
        assert_eq!(r.time_range, Some((2000, 2000)));

        let _ = std::fs::remove_file(&tmp.path);
        let _ = std::fs::remove_file(&policy_path);
    }

    struct TmpFile {
        file: std::fs::File,
        path: std::path::PathBuf,
    }
    fn tempfile_like() -> TmpFile {
        let path = std::env::temp_dir().join(format!(
            "replay-{}.jsonl",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let file = std::fs::File::create(&path).unwrap();
        TmpFile { file, path }
    }
}
