//! Built-in htmx dashboard, embedded in the binary and served by Pingora.
//! No build step, no asset pipeline, no external fetches.

/// Vendored htmx 2.0.4 (`dist/htmx.min.js`), embedded so the dashboard works
/// offline.
///
/// Provenance: fetched from unpkg and jsdelivr (byte-identical); the SHA-384
/// matches the official SRI hash htmx publishes for 2.0.4:
///   sha384-HGfztofotfshcF7+8n44JQL2oJmowVChPTg48S+jvZoztPfvwD79OC/LTtG6dMp+
/// SHA-256: e209dda5c8235479f3166defc7750e1dbcd5a5c1808b7792fc2e6733768fb447
pub const HTMX_JS: &str = include_str!("../assets/htmx.min.js");

pub const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>TokenMiser — You saved $X today</title>
<meta name="viewport" content="width=device-width,initial-scale=1" />
<script src="/assets/htmx.js"></script>
<style>
  :root {
    --bg: #0b0d10;
    --fg: #e6edf3;
    --muted: #7d8590;
    --accent: #58e6a0;
    --warn: #f0883e;
    --card: #161b22;
    --border: #30363d;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--fg); padding: 32px; }
  h1 { font-weight: 600; font-size: 22px; margin: 0 0 4px 0; }
  .sub { color: var(--muted); font-size: 14px; margin-bottom: 32px; }
  .hero {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 32px; text-align: center; margin-bottom: 24px;
  }
  .hero-label { color: var(--muted); font-size: 13px; letter-spacing: 0.05em;
                text-transform: uppercase; }
  .hero-value { font-size: 64px; font-weight: 700; color: var(--accent);
                font-variant-numeric: tabular-nums; margin: 8px 0; }
  .hero-counterfactual { color: var(--muted); font-size: 13px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
          gap: 16px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
          padding: 18px; }
  .card-label { color: var(--muted); font-size: 11px; letter-spacing: 0.06em;
                text-transform: uppercase; }
  .card-value { font-size: 32px; font-weight: 600; font-variant-numeric: tabular-nums;
                margin-top: 6px; }
  .card-sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
  .note { margin-top: 16px; padding: 12px 14px; border-radius: 8px; font-size: 13px;
          line-height: 1.5; color: var(--warn); background: rgba(240,136,62,0.08);
          border: 1px solid rgba(240,136,62,0.35); }
  .note code { font-size: 12px; }
  .unpriced { color: var(--warn); }
  .footer { color: var(--muted); font-size: 12px; margin-top: 24px; }
  a { color: var(--accent); text-decoration: none; }
</style>
</head>
<body>
  <h1>TokenMiser</h1>
  <div class="sub">Smart LLM router — drop-in OpenAI proxy with semantic cache + routing.</div>

  <div hx-get="/dashboard/fragment" hx-trigger="load, every 1s" hx-swap="innerHTML">
    Loading…
  </div>

  <div class="footer">
    Set <code>x-tokenmiser-tenant: your-tenant-id</code> on requests to scope cache &amp; spend.
    Use <code>model: "auto"</code> for routing, <code>model: "tokenmiser:cascade"</code> for speculative cascade.
    Docs: <code>github.com/openintelligence-labs/tokenmiser</code>
  </div>
</body>
</html>
"#;

/// Render the live fragment htmx polls, from a `/stats`-shaped blob.
pub fn render_fragment(stats_json: &serde_json::Value) -> String {
    let cost = stats_json.get("cost");
    let saved = cost
        .and_then(|c| c.get("saved_usd"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let spent = cost
        .and_then(|c| c.get("spent_usd"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let counterfactual = cost
        .and_then(|c| c.get("counterfactual_usd"))
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let total = cost
        .and_then(|c| c.get("requests_total"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let local = cost
        .and_then(|c| c.get("requests_local"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let frontier = cost
        .and_then(|c| c.get("requests_frontier"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_hits = cost
        .and_then(|c| c.get("cache_hits"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // When non-zero, `spent_usd` is a lower bound on the real bill and the
    // card must say so, rather than folding unpriced paid calls into a
    // "$0.0000 spent" figure.
    let unpriced = cost
        .and_then(|c| c.get("unpriced_requests"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let l1 = stats_json.get("cache_l1");
    let l1_hits = l1
        .and_then(|c| c.get("hits"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let l1_misses = l1
        .and_then(|c| c.get("misses"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let l2_hits = stats_json
        .get("cache_l2")
        .and_then(|c| c.get("hits"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let hit_rate = if l1_hits + l1_misses > 0 {
        100.0 * (l1_hits as f64) / ((l1_hits + l1_misses) as f64)
    } else {
        0.0
    };

    // Sub-label under "Spent", stating what the number excludes.
    let spent_sub = if unpriced > 0 {
        format!(
            r#"<span class="unpriced">at least — {unpriced} unpriced remote request{s}</span>"#,
            s = if unpriced == 1 { "" } else { "s" },
        )
    } else {
        "to upstream providers".to_string()
    };
    // Only shown when there is something to disclose.
    let unpriced_note = if unpriced > 0 {
        format!(
            r#"<div class="note">⚠ {unpriced} request{s} went to a remote model with no
       published per-token price (e.g. an Ollama Cloud <code>*-cloud</code> tag).
       Those are counted as remote, but their cost is <strong>unknown</strong> and is
       not included above — treat “Spent” and “You saved” as lower/upper bounds.</div>"#,
            s = if unpriced == 1 { "" } else { "s" },
        )
    } else {
        String::new()
    };

    format!(
        r##"
<div class="hero">
  <div class="hero-label">You saved</div>
  <div class="hero-value">${saved:.4}</div>
  <div class="hero-counterfactual">vs. ${counterfactual:.4} if everything had gone to frontier</div>
</div>
<div class="grid">
  <div class="card"><div class="card-label">Requests</div>
    <div class="card-value">{total}</div>
    <div class="card-sub">{local} local · {frontier} frontier · {cache_hits} cached</div></div>
  <div class="card"><div class="card-label">L1 cache</div>
    <div class="card-value">{l1_hits}</div>
    <div class="card-sub">{hit_rate:.1}% hit rate</div></div>
  <div class="card"><div class="card-label">L2 semantic</div>
    <div class="card-value">{l2_hits}</div>
    <div class="card-sub">bge-small-en-v1.5</div></div>
  <div class="card"><div class="card-label">Spent</div>
    <div class="card-value">${spent:.4}</div>
    <div class="card-sub">{spent_sub}</div></div>
</div>
{unpriced_note}
"##,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dashboard must work offline: no external http(s) URLs anywhere in
    /// the served HTML.
    #[test]
    fn dashboard_html_has_no_external_urls() {
        assert!(
            !DASHBOARD_HTML.contains("http://") && !DASHBOARD_HTML.contains("https://"),
            "dashboard HTML must not reference external http(s) URLs"
        );
        assert!(
            !DASHBOARD_HTML.contains("//unpkg.com") && !DASHBOARD_HTML.contains("//cdn."),
            "dashboard HTML must not reference CDNs"
        );
    }

    /// The live fragment htmx polls must also be offline-clean.
    #[test]
    fn dashboard_fragment_has_no_external_urls() {
        let frag = render_fragment(&serde_json::json!({}));
        assert!(
            !frag.contains("http://") && !frag.contains("https://"),
            "dashboard fragment must not reference external http(s) URLs"
        );
    }

    /// An unpriced remote call must be disclosed, not folded into a
    /// `$0.0000 spent` figure.
    #[test]
    fn unpriced_requests_are_disclosed_on_the_dashboard() {
        let frag = render_fragment(&serde_json::json!({
            "cost": { "spent_usd": 0.0, "unpriced_requests": 3, "requests_total": 3 }
        }));
        assert!(frag.contains("unpriced remote request"));
        assert!(
            frag.contains("unknown"),
            "the note must say the cost is unknown, not zero"
        );
        assert!(frag.contains('3'), "the count must be shown");
    }

    /// And stays out of the way entirely when everything is priced.
    #[test]
    fn no_unpriced_note_when_all_requests_are_priced() {
        let frag = render_fragment(&serde_json::json!({
            "cost": { "spent_usd": 1.5, "unpriced_requests": 0, "requests_total": 10 }
        }));
        assert!(!frag.contains("unpriced"));
        assert!(frag.contains("to upstream providers"));
    }

    /// A dashboard that says "1 requests" undermines trust in the numbers
    /// next to it.
    #[test]
    fn unpriced_note_is_grammatical_for_one_request() {
        let frag = render_fragment(&serde_json::json!({
            "cost": { "spent_usd": 0.0, "unpriced_requests": 1 }
        }));
        assert!(frag.contains("1 unpriced remote request<"));
        assert!(!frag.contains("1 unpriced remote requests"));
    }

    /// htmx is referenced from the local route and the vendored asset is
    /// really htmx 2.0.4.
    #[test]
    fn htmx_is_vendored_and_served_locally() {
        assert!(DASHBOARD_HTML.contains(r#"<script src="/assets/htmx.js"></script>"#));
        assert!(
            HTMX_JS.contains(r#"version:"2.0.4""#),
            "vendored htmx must be 2.0.4"
        );
        assert!(HTMX_JS.len() > 10_000, "vendored htmx looks truncated");
    }
}
