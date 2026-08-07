//! Pingora-based OpenAI-compatible proxy ingress.
//!
//! Requests terminate in the gateway rather than reverse-proxying: cache, cost
//! accounting, shadow A/B and the judge all need the full response in hand,
//! and terminating is also what lets Anthropic/OpenAI/Ollama be normalized at
//! the wire boundary.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use pingora::prelude::*;
use pingora_core::Result;
use pingora_http::{RequestHeader, ResponseHeader};
use pingora_proxy::{ProxyHttp, Session};
use serde_json::Value;
use std::time::Duration;
use tokenmiser_cache::{L1Cache, L2Cache};
use tokenmiser_config::{PricingTable, ProviderKind, TokenmiserConfig};
use tokenmiser_cost::{BudgetStatus, CostLedger};
use tokenmiser_mcp::McpBudgetGateway;
use tokenmiser_providers::{
    ollama::OllamaProvider, ChatRequest, ChatResponse, ProviderRegistry, StreamChunk,
};
use tokenmiser_quality::{ShadowEnqueue, ShadowScheduler};
use tokenmiser_router::{should_escalate, CascadeConfig, EscalateDecision, RouteTier, Router};
use tracing::warn;

pub mod backpressure;
pub mod dashboard;
pub mod mcp_route;
pub mod singleflight;
pub mod sse;
pub use backpressure::TokenBucket;
pub use singleflight::{Flight, FlightLease, FlightMap, FlightOutcome};
pub use sse::{cached_response_to_sse, openai_error_body, StreamAccumulator};

/// How long a single-flight follower waits for its leader. Generous on
/// purpose: leaders are already bound by their own upstream timeouts, and a
/// premature fallback just duplicates the call being deduplicated.
const FOLLOWER_WAIT: Duration = Duration::from_secs(300);

/// Hard cap on a buffered request body. The proxy buffers each body in memory
/// before parsing, so without a cap any local process can drive RSS linearly
/// with connection count. 8 MiB is far above any real completion payload.
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

/// Cap on the `x-tokenmiser-tenant` header. The tenant string becomes a
/// cache-key component and an L2 tenant-map key.
const MAX_TENANT_LEN: usize = 128;

/// Read a request body, returning `Err(limit)` once it exceeds
/// `MAX_BODY_BYTES` so the caller can answer 413.
async fn read_body_capped(session: &mut Session) -> Result<std::result::Result<Vec<u8>, usize>> {
    let mut buf = Vec::new();
    loop {
        match session.read_request_body().await? {
            Some(chunk) if !chunk.is_empty() => {
                if buf.len() + chunk.len() > MAX_BODY_BYTES {
                    return Ok(Err(MAX_BODY_BYTES));
                }
                buf.extend_from_slice(&chunk);
            }
            Some(_) => {}
            None => break,
        }
    }
    Ok(Ok(buf))
}

/// Sanitize the caller-supplied tenant id, falling back to `default`. Keeps
/// control characters and newlines out of structured log output and bounds the
/// memory a tenant string can claim.
fn sanitize_tenant(raw: &str) -> String {
    if raw.is_empty() || raw.len() > MAX_TENANT_LEN {
        return "default".to_string();
    }
    if raw
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
    {
        raw.to_string()
    } else {
        "default".to_string()
    }
}

async fn send_payload_too_large(session: &mut Session, limit: usize) -> Result<()> {
    send_error(
        session,
        413,
        &format!("request body exceeds the {limit}-byte limit"),
    )
    .await
}

pub mod admin;
pub use admin::AdminState;

/// Shared application state passed into the proxy on every request.
pub struct AppState {
    pub config: TokenmiserConfig,
    pub registry: ProviderRegistry,
    pub router: Router,
    pub l1: Arc<L1Cache>,
    pub l2: Option<Arc<L2Cache>>,
    pub ledger: Arc<CostLedger>,
    pub shadow: Option<Arc<ShadowScheduler>>,
    pub mcp: Arc<McpBudgetGateway>,
    /// One upstream call per exact cache key; concurrent identical misses
    /// coalesce onto it.
    pub flights: Arc<FlightMap>,
    /// Edge-detector so the budget alert logs once per transition into
    /// "exceeded", not once per request.
    budget_alerted: AtomicBool,
}

impl AppState {
    /// Evaluate budget thresholds against the current ledger, logging on the
    /// not-exceeded to exceeded transition.
    pub fn budget_status(&self) -> BudgetStatus {
        let status = BudgetStatus::evaluate(&self.config.budget, &self.ledger.snapshot());
        if status.exceeded {
            if !self.budget_alerted.swap(true, Ordering::Relaxed) {
                warn!(
                    target: "tokenmiser::budget",
                    spent_today_usd = status.spent_today_usd,
                    spent_total_usd = status.spent_total_usd,
                    daily_limit_usd = status.daily_limit_usd,
                    total_limit_usd = status.total_limit_usd,
                    enforce = status.enforce,
                    "budget exceeded"
                );
            }
        } else {
            // Re-arm the alert, e.g. after the UTC-day rollover.
            self.budget_alerted.store(false, Ordering::Relaxed);
        }
        status
    }
}

impl AppState {
    pub fn new(
        config: TokenmiserConfig,
        registry: ProviderRegistry,
        router: Router,
        ledger: Arc<CostLedger>,
        shadow: Option<Arc<ShadowScheduler>>,
        mcp: Arc<McpBudgetGateway>,
    ) -> Arc<Self> {
        // 16k entries at ~10KB average response bounds L1 at roughly 160MB.
        let l1 = if config.cache.l1_enabled {
            L1Cache::new(16_384, Duration::from_secs(3600))
        } else {
            L1Cache::disabled(Duration::from_secs(3600))
        };

        let l2 = if config.cache.l2_enabled {
            match L2Cache::new(
                config.cache.semantic_threshold,
                Duration::from_secs(3600),
                4_096,
                config.cache.numeric_guard,
            ) {
                Ok(c) => Some(c),
                Err(e) => {
                    warn!(error = %e, "L2 cache init failed; continuing without semantic cache");
                    None
                }
            }
        } else {
            None
        };

        Arc::new(Self {
            config,
            registry,
            router,
            l1,
            l2,
            ledger,
            shadow,
            mcp,
            flights: FlightMap::new(),
            budget_alerted: AtomicBool::new(false),
        })
    }
}

/// Per-request scratch state owned by Pingora's session.
#[derive(Default)]
pub struct ReqCtx {
    pub path: String,
    pub tenant: String,
}

pub struct TokenmiserProxy {
    state: Arc<AppState>,
}

impl TokenmiserProxy {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl ProxyHttp for TokenmiserProxy {
    type CTX = ReqCtx;

    fn new_ctx(&self) -> Self::CTX {
        ReqCtx::default()
    }

    /// Pingora's trait requires an upstream peer even for gateway-terminated
    /// requests. `request_filter` short-circuits before this sentinel is ever
    /// dialed.
    async fn upstream_peer(
        &self,
        _session: &mut Session,
        _ctx: &mut Self::CTX,
    ) -> Result<Box<HttpPeer>> {
        Ok(Box::new(HttpPeer::new(
            ("127.0.0.1", 1),
            false,
            "tokenmiser-noop".into(),
        )))
    }

    /// All request handling happens here; returning `Ok(true)` stops Pingora
    /// from dialing an upstream.
    async fn request_filter(&self, session: &mut Session, ctx: &mut Self::CTX) -> Result<bool> {
        let (method, path, tenant, origin_verdict, origin_dbg) = {
            let req: &RequestHeader = session.req_header();
            let header = |name: &str| {
                req.headers
                    .get(name)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned)
            };
            let sec_fetch_site = header("sec-fetch-site");
            let sec_fetch_mode = header("sec-fetch-mode");
            let origin = header("origin");
            let verdict = evaluate_origin(
                &self.state.config.security,
                sec_fetch_site.as_deref(),
                sec_fetch_mode.as_deref(),
                origin.as_deref(),
            );
            (
                req.method.as_str().to_string(),
                req.uri.path().to_string(),
                req.headers
                    .get("x-tokenmiser-tenant")
                    .and_then(|v| v.to_str().ok())
                    .map(sanitize_tenant)
                    .unwrap_or_else(|| "default".to_string()),
                verdict,
                (sec_fetch_site, origin),
            )
        };
        ctx.path = path.clone();
        ctx.tenant = tenant;

        // Runs before routing so the CSRF guard covers every endpoint. See
        // `evaluate_origin` for why loopback alone is not a boundary against
        // the browser.
        if origin_verdict == OriginVerdict::Deny {
            let (sec_fetch_site, origin) = origin_dbg;
            warn!(
                target: "tokenmiser::security",
                method = %method,
                path = %path,
                sec_fetch_site = sec_fetch_site.as_deref().unwrap_or("-"),
                origin = origin.as_deref().unwrap_or("-"),
                "rejected cross-origin browser request (CSRF guard)"
            );
            send_csrf_blocked(session).await?;
            return Ok(true);
        }

        // HEAD is served like GET minus the body (RFC 9110 §9.3.2); the
        // send_* helpers detect HEAD and skip the body write.
        let route_method = if method == "HEAD" {
            "GET"
        } else {
            method.as_str()
        };

        match (route_method, path.as_str()) {
            ("GET", "/healthz") => {
                send_text(session, 200, "ok").await?;
                Ok(true)
            }
            ("GET", "/stats") => {
                let snap = self.state.ledger.snapshot();
                let l1 = self.state.l1.stats();
                let l2 = self.state.l2.as_ref().map(|c| c.stats());
                let budget = self
                    .state
                    .config
                    .budget
                    .is_active()
                    .then(|| self.state.budget_status());
                let body = serde_json::json!({
                    "cost": snap,
                    "cache_l1": l1,
                    "cache_l2": l2,
                    "singleflight": self.state.flights.stats(),
                    "budget": budget,
                });
                let body = serde_json::to_vec_pretty(&body).unwrap_or_default();
                send_json(session, 200, Bytes::from(body)).await?;
                Ok(true)
            }
            ("GET", "/v1/models") => {
                handle_models(&self.state, session).await?;
                Ok(true)
            }
            ("POST", "/v1/chat/completions") => {
                handle_chat_completions(&self.state, session, ctx).await?;
                Ok(true)
            }
            ("GET", "/" | "/dashboard") => {
                send_html(session, 200, dashboard::DASHBOARD_HTML).await?;
                Ok(true)
            }
            ("GET", "/assets/htmx.js") => {
                send_js(session, 200, dashboard::HTMX_JS).await?;
                Ok(true)
            }
            ("GET", "/dashboard/fragment") => {
                let snap = self.state.ledger.snapshot();
                let l1 = self.state.l1.stats();
                let l2 = self.state.l2.as_ref().map(|c| c.stats());
                let json = serde_json::json!({
                    "cost": snap,
                    "cache_l1": l1,
                    "cache_l2": l2,
                });
                let frag = dashboard::render_fragment(&json);
                send_html(session, 200, &frag).await?;
                Ok(true)
            }
            ("POST", "/v1/mcp/tools/call") => {
                handle_mcp_tools_call(&self.state, session).await?;
                Ok(true)
            }
            ("POST", "/v1/mcp/budgets") => {
                handle_mcp_set_budget(&self.state, session).await?;
                Ok(true)
            }
            ("GET", "/v1/mcp/budgets") => {
                let snap = self.state.mcp.snapshot();
                let body = serde_json::to_vec_pretty(&snap).unwrap_or_default();
                send_json(session, 200, Bytes::from(body)).await?;
                Ok(true)
            }
            _ => {
                // OpenAI error shape under /v1/ so SDK clients get a parseable
                // body; plain text elsewhere.
                if path.starts_with("/v1/") {
                    send_error(session, 404, &format!("unknown route: {method} {path}")).await?;
                } else {
                    send_text(session, 404, "not found").await?;
                }
                Ok(true)
            }
        }
    }
}

/// `GET /v1/models`: router pseudo-models, configured aliases, and installed
/// Ollama models. The Ollama list is probed live so newly pulled models appear
/// without a daemon restart.
async fn handle_models(state: &AppState, session: &mut Session) -> Result<()> {
    let mut seen = std::collections::HashSet::new();
    let mut data: Vec<Value> = Vec::new();
    let push = |id: &str,
                owned_by: &str,
                seen: &mut std::collections::HashSet<String>,
                data: &mut Vec<Value>| {
        if seen.insert(id.to_string()) {
            data.push(serde_json::json!({
                "id": id,
                "object": "model",
                "created": 0,
                "owned_by": owned_by,
            }));
        }
    };

    for id in [
        "auto",
        "tokenmiser:auto",
        "tokenmiser:cascade",
        "auto:cascade",
    ] {
        push(id, "tokenmiser", &mut seen, &mut data);
    }
    let mut aliases: Vec<_> = state.config.routing.aliases.iter().collect();
    aliases.sort_by(|a, b| a.0.cmp(b.0));
    for (alias, target) in aliases {
        push(alias, &target.provider, &mut seen, &mut data);
    }
    // Soft-fail: a missing local Ollama must not break the listing.
    if let Some(p) = state
        .config
        .providers
        .iter()
        .find(|p| p.kind == ProviderKind::Ollama)
    {
        match OllamaProvider::detect(&p.base_url).await {
            Ok(mut models) => {
                models.sort();
                for m in models {
                    push(&m, "ollama", &mut seen, &mut data);
                }
            }
            Err(e) => {
                warn!(error = %e, "ollama probe failed during /v1/models; listing without local models");
            }
        }
    }

    let body = serde_json::json!({"object": "list", "data": data});
    let body = serde_json::to_vec(&body).unwrap_or_default();
    send_json(session, 200, Bytes::from(body)).await
}

async fn handle_chat_completions(
    state: &AppState,
    session: &mut Session,
    _ctx: &mut ReqCtx,
) -> Result<()> {
    let buf = match read_body_capped(session).await? {
        Ok(b) => b,
        Err(limit) => return send_payload_too_large(session, limit).await,
    };

    let parsed: ChatRequest = match serde_json::from_slice(&buf) {
        Ok(p) => p,
        Err(e) => {
            return send_error(session, 400, &format!("bad request body: {e}")).await;
        }
    };

    // Validate before any routing/provider/cache work, so a missing model
    // field returns 400 rather than a confusing 401 from the default
    // provider's absent API key.
    if let Err(msg) = validate_chat_request(&parsed) {
        return send_error(session, 400, &msg).await;
    }

    let want_stream = parsed.stream == Some(true);

    // A hit is served in whichever wire format the client asked for: JSON, or
    // a simulated OpenAI chunk stream for `stream: true` callers.
    let cache_hit: Option<(ChatResponse, &'static str)> =
        match state.l1.lookup(&parsed, &_ctx.tenant) {
            Some(cached) => Some((cached, "l1-hit")),
            None => state.l2.as_ref().and_then(|l2| {
                l2.lookup(&parsed, &_ctx.tenant).map(|cached| {
                    // Seed L1 so the next identical hit takes the fast path.
                    state.l1.insert(&parsed, &_ctx.tenant, &cached);
                    (cached, "l2-hit")
                })
            }),
        };

    if let Some((cached, hit_kind)) = cache_hit {
        state.ledger.record_cache_hit(
            &parsed.model,
            cached.usage.prompt_tokens,
            cached.usage.completion_tokens,
        );
        let mut out = cached;
        out.model = parsed.model.clone();
        if want_stream {
            return send_cached_stream(state, session, &out, &parsed.model, hit_kind, None).await;
        }
        let body = serde_json::to_vec(&out).unwrap_or_default();
        return send_json_with_header(
            session,
            200,
            Bytes::from(body),
            "x-tokenmiser-cache",
            hit_kind,
        )
        .await;
    }

    let is_cascade = matches!(parsed.model.as_str(), "tokenmiser:cascade" | "auto:cascade");

    if is_cascade {
        if want_stream {
            // Scoring confidence needs the full cheap response in hand, which
            // streaming pass-through cannot provide. Fail loudly rather than
            // routing the pseudo-model name upstream.
            return send_error(
                session,
                400,
                "cascade routing (`tokenmiser:cascade`) does not support `stream: true` yet; \
                 use `auto` or a concrete model for streaming",
            )
            .await;
        }
        // Cascade bypasses single-flight: the escalate decision is
        // per-response, so its two-step call graph does not fit the
        // one-key-one-outcome model.
        return handle_cascade(state, session, _ctx, &parsed).await;
    }

    // Both cache layers missed, so this request is about to pay for an
    // upstream call; coalesce onto an identical one already in flight.
    let flight_key = tokenmiser_cache::exact_key(&parsed, &_ctx.tenant);
    let lease: Option<FlightLease> = match state.flights.begin(&flight_key) {
        Flight::Leader(lease) => Some(lease),
        Flight::Follower(rx) => {
            match singleflight::await_outcome(&state.flights, rx, FOLLOWER_WAIT).await {
                Some(FlightOutcome::Response {
                    response,
                    routed_to,
                }) => {
                    return serve_coalesced(
                        state,
                        session,
                        &parsed,
                        &response,
                        &routed_to,
                        want_stream,
                    )
                    .await;
                }
                Some(FlightOutcome::Error { status, message }) => {
                    return send_error(session, status, &message).await;
                }
                // Leader abandoned: fall back to an own upstream call.
                None => None,
            }
        }
    };

    if want_stream {
        return handle_stream(state, session, _ctx, &parsed, lease).await;
    }

    complete_upstream(state, session, _ctx, &parsed, lease).await
}

/// Non-streaming completion: route, resolve, budget gate, call, record, cache,
/// respond. `lease` is `Some` when this request leads a single-flight; every
/// terminal outcome publishes to followers, or its drop releases them.
async fn complete_upstream(
    state: &AppState,
    session: &mut Session,
    _ctx: &mut ReqCtx,
    parsed: &ChatRequest,
    lease: Option<FlightLease>,
) -> Result<()> {
    let decision = state.router.decide(parsed);
    let target_model = decision.target.model.clone();

    let (provider, real_model) = match state.registry.resolve(&target_model) {
        Ok(t) => t,
        Err(e) => {
            let msg = format!("model resolution: {e}");
            publish_error(lease, 400, &msg);
            return send_error(session, 400, &msg).await;
        }
    };

    // Warn-only by default; with `enforce: true` a paid route is rejected
    // with 402. Free local traffic always passes.
    let budget = state
        .config
        .budget
        .is_active()
        .then(|| state.budget_status());
    let route_is_local = route_is_local(provider.name(), &real_model, &target_model);
    if budget_blocks(&budget, route_is_local) {
        publish_error(lease, 402, BUDGET_BLOCKED_MSG);
        return send_budget_blocked(session).await;
    }

    let mut upstream_req = parsed.clone();
    upstream_req.model = real_model.clone();

    match provider.complete(&upstream_req).await {
        Ok(resp) => {
            let prompt_tokens = resp.usage.prompt_tokens;
            let completion_tokens = resp.usage.completion_tokens;
            let is_local = route_is_local;

            record_upstream(
                state,
                is_local,
                &real_model,
                decision.counterfactual_model.as_deref(),
                prompt_tokens,
                completion_tokens,
            );

            // Some models (gemma reasoning, deepseek-r1) return no visible
            // content on simple prompts. Caching that would amplify the
            // failure across the tenant's whole prompt cluster.
            let visible_empty = response_visible_content_empty(&resp);
            if visible_empty {
                warn!(
                    model = %real_model,
                    "upstream returned empty message.content; skipping cache + flagging response"
                );
            } else {
                state.l1.insert(parsed, &_ctx.tenant, &resp);
                if let Some(l2) = &state.l2 {
                    l2.insert(parsed, &_ctx.tenant, &resp);
                }
            }

            // Shadow A/B samples non-frontier responses for background
            // comparison; the client already has its answer.
            if let Some(shadow) = &state.shadow {
                if !is_frontier_model(&real_model) {
                    shadow.maybe_enqueue(ShadowEnqueue::from_request(parsed, &resp, &real_model));
                }
            }

            // Published after the cache insert above, so a late arrival that
            // misses the flight map still finds the L1 entry.
            publish_response(lease, &resp, &real_model);

            // Echo the requested model (including `auto`) rather than the
            // routed-to one.
            let mut out = resp;
            out.model = parsed.model.clone();
            let body = match serde_json::to_vec(&out) {
                Ok(b) => b,
                Err(e) => {
                    return send_error(session, 500, &format!("serialize: {e}")).await;
                }
            };
            send_routed_response(
                session,
                200,
                Bytes::from(body),
                &decision,
                &real_model,
                visible_empty,
                budget.as_ref(),
            )
            .await
        }
        Err(e) => {
            let status = provider_error_status(&e);
            let msg = e.to_string();
            publish_error(lease, status, &msg);
            send_routed_error(session, status, &msg, &decision, &real_model).await
        }
    }
}

/// Serve a single-flight follower from its leader's published response.
///
/// Ledgered as a cache hit — the response came from memory, not an upstream
/// call — so the leader's call stays the only recorded spend. The header says
/// `coalesced` so operators can tell dedup from a real cache hit.
async fn serve_coalesced(
    state: &AppState,
    session: &mut Session,
    parsed: &ChatRequest,
    cached: &ChatResponse,
    routed_to: &str,
    want_stream: bool,
) -> Result<()> {
    state.ledger.record_cache_hit(
        &parsed.model,
        cached.usage.prompt_tokens,
        cached.usage.completion_tokens,
    );
    let mut out = cached.clone();
    out.model = parsed.model.clone();
    if want_stream {
        return send_cached_stream(
            state,
            session,
            &out,
            &parsed.model,
            "coalesced",
            Some(routed_to),
        )
        .await;
    }
    let bytes = Bytes::from(serde_json::to_vec(&out).unwrap_or_default());
    let mut resp = ResponseHeader::build(200, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    resp.insert_header("x-tokenmiser-cache", "coalesced")?;
    resp.insert_header("x-tokenmiser-routed-to", routed_to.to_string())?;
    finish_response(session, resp, bytes).await
}

/// No-op without a lease, i.e. on the follower-fallback path.
fn publish_response(lease: Option<FlightLease>, resp: &ChatResponse, routed_to: &str) {
    if let Some(l) = lease {
        l.publish(FlightOutcome::Response {
            response: Arc::new(resp.clone()),
            routed_to: routed_to.to_string(),
        });
    }
}

fn publish_error(lease: Option<FlightLease>, status: u16, message: &str) {
    if let Some(l) = lease {
        l.publish(FlightOutcome::Error {
            status,
            message: message.to_string(),
        });
    }
}

/// Whether a request carries a browser-origin signal we trust.
///
/// Loopback binding is not a boundary against the browser: a hostile page can
/// POST `content-type: text/plain` to 127.0.0.1, which is a CORS *simple*
/// request, so no preflight blocks it and the operator's budget is spent. CORS
/// only prevents the page from reading the reply.
///
/// The defense is to detect that a browser sent the request at all, via
/// headers a page cannot forge (both are forbidden header names): the
/// always-present `Sec-Fetch-Site`, and `Origin` on every cross-origin request
/// and POST. Neither header means a non-browser client (curl, the SDKs) and is
/// allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OriginVerdict {
    /// No browser-origin signal, or a signal that resolves to this origin.
    Allow,
    /// A browser is driving this request from an origin we don't trust.
    Deny,
}

/// Evaluate the raw browser-origin headers (`None` when absent). Same-origin
/// traffic, which is what the dashboard's htmx polling is, always passes.
fn evaluate_origin(
    security: &tokenmiser_config::SecurityConfig,
    sec_fetch_site: Option<&str>,
    sec_fetch_mode: Option<&str>,
    origin: Option<&str>,
) -> OriginVerdict {
    if security.allows_any_origin() {
        return OriginVerdict::Allow;
    }

    // An allow-listed Origin wins over Sec-Fetch-Site: this is the opt-in
    // path for a legitimate browser app.
    if let Some(o) = origin {
        if security.origin_allowed(o) {
            return OriginVerdict::Allow;
        }
    }

    // Top-level navigations must pass: Chrome labels them `cross-site`
    // whenever the previous page was another site, including `chrome://newtab`
    // — which is how every fresh dashboard visit looks.
    //
    // Safe because a navigation is not the attack vector. `Sec-Fetch-Mode` is
    // a forbidden header name, so a page cannot set it from `fetch`, and a
    // real navigation replaces the attacker's page with ours. The CSRF vector
    // needs a background request (`mode: cors`/`no-cors`), still rejected.
    if sec_fetch_mode == Some("navigate") {
        // GET/HEAD only: a cross-site form POST is also `navigate` and can
        // carry an attacker-chosen body.
        let is_form_post = origin.is_some();
        if !is_form_post {
            return OriginVerdict::Allow;
        }
    }

    // `same-site` denies too: a sibling subdomain is not our own page.
    match sec_fetch_site {
        Some("cross-site") | Some("same-site") => return OriginVerdict::Deny,
        Some("same-origin") | Some("none") => return OriginVerdict::Allow,
        Some(_) => return OriginVerdict::Deny, // unknown value: fail closed
        None => {}
    }

    // Without Sec-Fetch-Site, an Origin means an older browser that is not
    // allow-listed; no Origin at all means a plain CLI/SDK client.
    match origin {
        Some(_) => OriginVerdict::Deny,
        None => OriginVerdict::Allow,
    }
}

const CSRF_BLOCKED_MSG: &str = concat!(
    "cross-origin browser request rejected: TokenMiser is a local, unauthenticated ",
    "proxy that spends your API budget, so it refuses requests driven by a web page. ",
    "Add the origin to `security.allowed_origins` in your config to opt in."
);

async fn send_csrf_blocked(session: &mut Session) -> Result<()> {
    send_error(session, 403, CSRF_BLOCKED_MSG).await
}

/// True when this route executes on the operator's own hardware and is
/// therefore free at the point of use.
///
/// An `ollama:` prefix does not mean local: Ollama Cloud models are served by
/// the same local daemon under the same provider name, but generate tokens on
/// paid infrastructure. The cloud check therefore wins over every
/// looks-like-ollama signal, or paid traffic would be reported as free and
/// waved through an enforce-mode budget.
fn route_is_local(provider: &str, real_model: &str, target_model: &str) -> bool {
    if PricingTable::is_ollama_cloud(real_model) || PricingTable::is_ollama_cloud(target_model) {
        return false;
    }
    real_model.starts_with("ollama") || provider == "ollama" || target_model.starts_with("ollama")
}

/// Record one completed upstream call on the correct side of the free/paid
/// line. Centralized so the completion, streaming and cascade paths cannot
/// drift apart on what counts as spend.
fn record_upstream(
    state: &AppState,
    is_local: bool,
    real_model: &str,
    counterfactual: Option<&str>,
    prompt_tokens: u64,
    completion_tokens: u64,
) {
    if is_local {
        state
            .ledger
            .record_free(counterfactual, prompt_tokens, completion_tokens);
    } else {
        state
            .ledger
            .record_paid(real_model, prompt_tokens, completion_tokens);
    }
}

/// True when an enforce-mode budget must block this request. Paid routes only:
/// cache hits never reach this and free local traffic passes.
fn budget_blocks(budget: &Option<BudgetStatus>, route_is_local: bool) -> bool {
    match budget {
        Some(b) => b.exceeded && b.enforce && !route_is_local,
        None => false,
    }
}

fn budget_header_value(budget: Option<&BudgetStatus>) -> Option<&'static str> {
    budget.map(|b| if b.exceeded { "exceeded" } else { "ok" })
}

const BUDGET_BLOCKED_MSG: &str =
    "tokenmiser budget exceeded (budget.enforce = true); request to a paid provider rejected";

async fn send_budget_blocked(session: &mut Session) -> Result<()> {
    let body = openai_error_body(402, BUDGET_BLOCKED_MSG);
    let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
    let mut resp = ResponseHeader::build(402, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    resp.insert_header("x-tokenmiser-budget", "exceeded")?;
    finish_response(session, resp, bytes).await
}

/// Map a `ProviderError` to an HTTP status. Single source of truth so the
/// streaming and non-streaming paths agree.
fn provider_error_status(e: &tokenmiser_providers::ProviderError) -> u16 {
    use tokenmiser_providers::ProviderError::*;
    match e {
        Upstream { status, .. } => *status,
        MissingApiKey(_) => 401,
        UnknownModel { .. } | NotFound { .. } => 400,
        _ => 502,
    }
}

/// SSE streaming path: pump upstream chunks downstream with per-connection
/// backpressure, reassembling them so a cleanly finished stream lands in
/// L1/L2 exactly like a non-streaming response.
async fn handle_stream(
    state: &AppState,
    session: &mut Session,
    ctx: &mut ReqCtx,
    parsed: &ChatRequest,
    lease: Option<FlightLease>,
) -> Result<()> {
    let decision = state.router.decide(parsed);
    let target_model = decision.target.model.clone();

    let (provider, real_model) = match state.registry.resolve(&target_model) {
        Ok(t) => t,
        Err(e) => {
            let status = provider_error_status(&e);
            let msg = e.to_string();
            publish_error(lease, status, &msg);
            return send_error(session, status, &msg).await;
        }
    };

    // Same budget policy as the non-streaming path.
    let budget = state
        .config
        .budget
        .is_active()
        .then(|| state.budget_status());
    let route_is_local = route_is_local(provider.name(), &real_model, &target_model);
    if budget_blocks(&budget, route_is_local) {
        publish_error(lease, 402, BUDGET_BLOCKED_MSG);
        return send_budget_blocked(session).await;
    }

    let mut upstream_req = parsed.clone();
    upstream_req.model = real_model.clone();
    ensure_include_usage(&mut upstream_req);

    let mut stream = match provider.stream(&upstream_req).await {
        Ok(s) => s,
        Err(e) => {
            let status = provider_error_status(&e);
            let msg = e.to_string();
            publish_error(lease, status, &msg);
            return send_routed_error(session, status, &msg, &decision, &real_model).await;
        }
    };

    let mut resp = ResponseHeader::build(200, None)?;
    resp.insert_header("content-type", "text/event-stream")?;
    resp.insert_header("cache-control", "no-cache")?;
    resp.insert_header("connection", "keep-alive")?;
    resp.insert_header("x-tokenmiser-cache", "miss")?;
    resp.insert_header("x-tokenmiser-routed-to", real_model.clone())?;
    resp.insert_header(
        "x-tokenmiser-difficulty",
        match decision.difficulty {
            tokenmiser_router::Difficulty::Easy => "easy",
            tokenmiser_router::Difficulty::Medium => "medium",
            tokenmiser_router::Difficulty::Hard => "hard",
        },
    )?;
    resp.insert_header(
        "x-tokenmiser-tier",
        match decision.tier {
            RouteTier::Explicit => "explicit",
            RouteTier::Heuristic => "heuristic",
            RouteTier::Semantic => "semantic",
        },
    )?;
    if let Some(v) = budget_header_value(budget.as_ref()) {
        resp.insert_header("x-tokenmiser-budget", v)?;
    }
    session.write_response_header(Box::new(resp), false).await?;

    let mut bucket = TokenBucket::default_streaming();
    let mut total_bytes_out: usize = 0;
    let mut acc = StreamAccumulator::new();
    let mut client_disconnected = false;
    let mut upstream_error: Option<String> = None;

    // Disconnect detection is bounded by upstream pace: cancellation lands on
    // the next upstream chunk, or on the next probe during a long silence.
    const LIVENESS_PROBE: std::time::Duration = std::time::Duration::from_millis(100);

    loop {
        // Race the next upstream chunk against the probe timer, so a silent
        // upstream still wakes this loop to notice a closed client.
        let next = tokio::time::timeout(LIVENESS_PROBE, stream.next()).await;
        match next {
            Err(_elapsed) => {
                // An SSE comment, which clients ignore per the spec, but whose
                // bytes hit the wire so a closed downstream surfaces as a write
                // error. Pingora coalesces empty-byte writes, so those do not
                // reliably probe the socket.
                //
                // Never splice the comment into a half-written event: that
                // corrupts the client's `data:` line. Skipping is safe because
                // the next upstream chunk closes the event.
                if acc.is_mid_event() {
                    continue;
                }
                let probe = bytes::Bytes::from_static(b": keepalive\n\n");
                if session
                    .write_response_body(Some(probe), false)
                    .await
                    .is_err()
                {
                    warn!(
                        bytes_out = total_bytes_out,
                        "client disconnected (detected via SSE keepalive probe); aborting"
                    );
                    client_disconnected = true;
                    break;
                }
                continue;
            }
            Ok(None) => break,
            Ok(Some(Ok(StreamChunk::Sse(bytes)))) => {
                bucket.acquire(bytes.len()).await;
                total_bytes_out += bytes.len();

                // Fed before forwarding, so token counts survive a client
                // disconnect mid-stream.
                acc.push(&bytes);

                if session
                    .write_response_body(Some(bytes), false)
                    .await
                    .is_err()
                {
                    warn!(
                        bytes_out = total_bytes_out,
                        "client disconnected mid-stream; aborting"
                    );
                    client_disconnected = true;
                    break;
                }
            }
            Ok(Some(Ok(StreamChunk::Done))) => break,
            Ok(Some(Err(e))) => {
                warn!(error = %e, "upstream stream error; closing");
                upstream_error = Some(e.to_string());
                break;
            }
        }
    }

    if !client_disconnected {
        if let Some(msg) = &upstream_error {
            // Surface the failure in-band so SDK clients raise a typed error
            // instead of reading the silent truncation as a complete answer:
            // the stream ends with no finish_reason and no [DONE], which most
            // clients treat as a normal end of iteration. Only at an event
            // boundary, or the client's parse is corrupted.
            if !acc.is_mid_event() {
                let body = openai_error_body(502, &format!("upstream stream error: {msg}"));
                let ev = format!("data: {body}\n\n");
                let _ = session
                    .write_response_body(Some(Bytes::from(ev)), false)
                    .await;
            }
        }
        let _ = session
            .write_response_body(Some(bytes::Bytes::new()), true)
            .await;
    }

    // Zero when the upstream emitted no usage, so the request is still
    // counted without inventing a cost.
    let (prompt_tokens, completion_tokens) = acc
        .usage
        .as_ref()
        .map(|u| (u.prompt_tokens, u.completion_tokens))
        .unwrap_or((0, 0));

    record_upstream(
        state,
        route_is_local,
        &real_model,
        decision.counterfactual_model.as_deref(),
        prompt_tokens,
        completion_tokens,
    );

    // Cached only when the stream finished cleanly and reassembled into a
    // non-empty single-choice response. On disconnect or an uncacheable shape
    // the lease drop releases followers to make their own upstream calls.
    if !client_disconnected && upstream_error.is_none() {
        if let Some(full) = acc.into_chat_response(&real_model) {
            state.l1.insert(parsed, &ctx.tenant, &full);
            if let Some(l2) = &state.l2 {
                l2.insert(parsed, &ctx.tenant, &full);
            }
            publish_response(lease, &full, &real_model);
        }
    } else if let Some(msg) = &upstream_error {
        publish_error(lease, 502, &format!("upstream stream error: {msg}"));
    }

    Ok(())
}

/// Serve a cache hit to a `stream: true` client by replaying the cached
/// response as a simulated OpenAI chunk stream.
async fn send_cached_stream(
    state: &AppState,
    session: &mut Session,
    cached: &ChatResponse,
    requested_model: &str,
    hit_kind: &'static str,
    routed_to: Option<&str>,
) -> Result<()> {
    let budget = state
        .config
        .budget
        .is_active()
        .then(|| state.budget_status());

    let mut resp = ResponseHeader::build(200, None)?;
    resp.insert_header("content-type", "text/event-stream")?;
    resp.insert_header("cache-control", "no-cache")?;
    resp.insert_header("connection", "keep-alive")?;
    resp.insert_header("x-tokenmiser-cache", hit_kind)?;
    if let Some(r) = routed_to {
        resp.insert_header("x-tokenmiser-routed-to", r.to_string())?;
    }
    if let Some(v) = budget_header_value(budget.as_ref()) {
        resp.insert_header("x-tokenmiser-budget", v)?;
    }
    session.write_response_header(Box::new(resp), false).await?;

    let chunks = cached_response_to_sse(cached, requested_model);
    let n = chunks.len();
    for (i, chunk) in chunks.into_iter().enumerate() {
        let end = i + 1 == n;
        if session.write_response_body(Some(chunk), end).await.is_err() {
            break;
        }
    }
    Ok(())
}

/// Force `stream_options.include_usage` on outbound streaming requests so
/// providers emit a usage block in the final SSE event.
fn ensure_include_usage(req: &mut ChatRequest) {
    let entry = req
        .extra
        .entry("stream_options")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if let Value::Object(map) = entry {
        map.entry("include_usage").or_insert(Value::Bool(true));
    }
}

/// Speculative cascade: cheap model first, escalate on low confidence.
async fn handle_cascade(
    state: &AppState,
    session: &mut Session,
    ctx: &mut ReqCtx,
    parsed: &ChatRequest,
) -> Result<()> {
    let cfg = CascadeConfig::default();

    let easy_target = state
        .router
        .policy_target(tokenmiser_router::Difficulty::Easy);
    let hard_target = state
        .router
        .policy_target(tokenmiser_router::Difficulty::Hard);

    // logprobs are what the confidence score is derived from.
    let mut cheap_req = parsed.clone();
    cheap_req.model = easy_target.model.clone();
    cheap_req
        .extra
        .insert("logprobs".into(), serde_json::Value::Bool(true));
    cheap_req
        .extra
        .insert("top_logprobs".into(), serde_json::Value::from(0));

    let (cheap_provider, cheap_real) = match state.registry.resolve(&easy_target.model) {
        Ok(t) => t,
        Err(e) => return send_error(session, 400, &format!("cascade resolve cheap: {e}")).await,
    };
    cheap_req.model = cheap_real.clone();

    // The cheap leg needs its own gate: the Easy tier defaults to Ollama but
    // is user-overridable to a paid model.
    let cheap_budget = state
        .config
        .budget
        .is_active()
        .then(|| state.budget_status());
    let cheap_is_local = route_is_local(cheap_provider.name(), &cheap_real, &easy_target.model);
    if budget_blocks(&cheap_budget, cheap_is_local) {
        return send_budget_blocked(session).await;
    }

    let cheap_resp = match cheap_provider.complete(&cheap_req).await {
        Ok(r) => r,
        Err(e) => {
            return send_error(session, 502, &format!("cascade cheap call failed: {e}")).await
        }
    };

    let decision = should_escalate(&cheap_resp, &cfg);
    match decision {
        EscalateDecision::No { signal } => {
            // Free only when the cheap leg really ran locally; an Ollama Cloud
            // cheap model is a paid remote call like any other.
            record_upstream(
                state,
                cheap_is_local,
                &cheap_real,
                Some(hard_target.model.as_str()),
                cheap_resp.usage.prompt_tokens,
                cheap_resp.usage.completion_tokens,
            );
            state.l1.insert(parsed, &ctx.tenant, &cheap_resp);
            if let Some(l2) = &state.l2 {
                l2.insert(parsed, &ctx.tenant, &cheap_resp);
            }
            let mut out = cheap_resp;
            out.model = parsed.model.clone();
            let body = serde_json::to_vec(&out).unwrap_or_default();
            send_cascade_response(
                session,
                200,
                Bytes::from(body),
                "no-escalate",
                &cheap_real,
                signal,
            )
            .await
        }
        EscalateDecision::Yes { reason: _, signal } => {
            // Unless an enforce-mode budget puts the paid call off the table,
            // in which case the cheap response is the best legal answer.
            let budget = state
                .config
                .budget
                .is_active()
                .then(|| state.budget_status());
            let frontier_is_local = PricingTable::is_free(&hard_target.model);
            if budget_blocks(&budget, frontier_is_local) {
                warn!(
                    target: "tokenmiser::budget",
                    "cascade wanted to escalate but budget.enforce blocks paid calls; serving cheap response"
                );
                record_upstream(
                    state,
                    cheap_is_local,
                    &cheap_real,
                    Some(hard_target.model.as_str()),
                    cheap_resp.usage.prompt_tokens,
                    cheap_resp.usage.completion_tokens,
                );
                let mut out = cheap_resp;
                out.model = parsed.model.clone();
                let body = serde_json::to_vec(&out).unwrap_or_default();
                return send_cascade_response(
                    session,
                    200,
                    Bytes::from(body),
                    "escalate-blocked-by-budget",
                    &cheap_real,
                    signal,
                )
                .await;
            }

            let (frontier_provider, frontier_real) =
                match state.registry.resolve(&hard_target.model) {
                    Ok(t) => t,
                    Err(e) => {
                        return send_error(session, 400, &format!("cascade resolve frontier: {e}"))
                            .await
                    }
                };
            let mut frontier_req = parsed.clone();
            frontier_req.model = frontier_real.clone();
            let frontier_resp = match frontier_provider.complete(&frontier_req).await {
                Ok(r) => r,
                Err(e) => {
                    return send_error(session, 502, &format!("cascade frontier call failed: {e}"))
                        .await
                }
            };
            // One client request is one ledger request: only the frontier
            // call is recorded, or the cheap probe double-counts
            // `requests_total`.
            state.ledger.record_paid(
                &frontier_real,
                frontier_resp.usage.prompt_tokens,
                frontier_resp.usage.completion_tokens,
            );
            state.l1.insert(parsed, &ctx.tenant, &frontier_resp);
            if let Some(l2) = &state.l2 {
                l2.insert(parsed, &ctx.tenant, &frontier_resp);
            }
            let mut out = frontier_resp;
            out.model = parsed.model.clone();
            let body = serde_json::to_vec(&out).unwrap_or_default();
            send_cascade_response(
                session,
                200,
                Bytes::from(body),
                "escalate",
                &frontier_real,
                signal,
            )
            .await
        }
    }
}

async fn send_cascade_response(
    session: &mut Session,
    status: u16,
    body: Bytes,
    action: &'static str,
    routed_to: &str,
    signal: tokenmiser_router::tier2::Signal,
) -> Result<()> {
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", body.len().to_string())?;
    resp.insert_header("x-tokenmiser-cache", "miss")?;
    resp.insert_header("x-tokenmiser-tier", "cascade")?;
    resp.insert_header("x-tokenmiser-cascade", action)?;
    resp.insert_header("x-tokenmiser-routed-to", routed_to.to_string())?;
    let sig_str = match signal {
        tokenmiser_router::tier2::Signal::Logprob(v) => format!("logprob={:.3}", v),
        tokenmiser_router::tier2::Signal::Length(n) => format!("length={}", n),
    };
    resp.insert_header("x-tokenmiser-cascade-signal", sig_str)?;
    finish_response(session, resp, body).await
}

async fn handle_mcp_tools_call(state: &AppState, session: &mut Session) -> Result<()> {
    use mcp_route::*;
    let buf = match read_body_capped(session).await? {
        Ok(b) => b,
        Err(limit) => return send_payload_too_large(session, limit).await,
    };
    let req: McpToolsCallRequest = match serde_json::from_slice(&buf) {
        Ok(r) => r,
        Err(e) => return send_error(session, 400, &format!("invalid JSON-RPC body: {e}")).await,
    };

    let agent = req
        .params
        .agent
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let tool = req.params.name.clone();

    match state.mcp.check(&agent, &tool) {
        Ok(spend) => {
            if let Some(actual) = req.params.actual_cost_usd {
                state.mcp.record(&agent, &tool, actual);
            } else if let Some(est) = req.params.estimated_cost_usd {
                // Pre-record the estimate so concurrent calls see the budget
                // effect before the actual cost is reported.
                state.mcp.record(&agent, &tool, est);
            }
            let resp = McpToolsCallResponse {
                jsonrpc: "2.0",
                id: req.id,
                result: Some(McpResult {
                    allowed: true,
                    agent,
                    tool,
                    spent_usd: spend.spent_usd,
                    calls: spend.calls,
                }),
                error: None,
            };
            let body = serde_json::to_vec(&resp).unwrap_or_default();
            send_json(session, 200, Bytes::from(body)).await
        }
        Err(e) => {
            let resp = McpToolsCallResponse {
                jsonrpc: "2.0",
                id: req.id,
                result: None,
                error: Some(McpError {
                    code: ERR_BUDGET_EXCEEDED,
                    message: e.to_string(),
                }),
            };
            let body = serde_json::to_vec(&resp).unwrap_or_default();
            send_json(session, 402, Bytes::from(body)).await
        }
    }
}

async fn handle_mcp_set_budget(state: &AppState, session: &mut Session) -> Result<()> {
    use mcp_route::*;
    let buf = match read_body_capped(session).await? {
        Ok(b) => b,
        Err(limit) => return send_payload_too_large(session, limit).await,
    };
    let req: SetBudgetRequest = match serde_json::from_slice(&buf) {
        Ok(r) => r,
        Err(e) => return send_error(session, 400, &format!("invalid body: {e}")).await,
    };
    let (agent, tool, budget) = req.into_budget();
    state.mcp.set_budget(&agent, &tool, budget);
    let body = serde_json::json!({"status":"ok"});
    let body = serde_json::to_vec(&body).unwrap_or_default();
    send_json(session, 200, Bytes::from(body)).await
}

/// Sanity-check a parsed request, returning a user-facing message when it is
/// structurally invalid.
fn validate_chat_request(req: &ChatRequest) -> std::result::Result<(), String> {
    if req.model.trim().is_empty() {
        return Err("`model` is required and must not be empty".into());
    }
    if req.messages.is_empty() {
        return Err("`messages` must contain at least one message".into());
    }
    for (i, m) in req.messages.iter().enumerate() {
        if m.role.trim().is_empty() {
            return Err(format!("messages[{i}].role must not be empty"));
        }
        // Tool/function messages legitimately carry no content.
        let role_needs_content = matches!(m.role.as_str(), "user" | "system" | "assistant");
        if role_needs_content {
            match &m.content {
                Value::Null => {
                    return Err(format!(
                        "messages[{i}].content must not be null for role {}",
                        m.role
                    ));
                }
                Value::String(s) if s.is_empty() => {
                    return Err(format!(
                        "messages[{i}].content must not be empty for role {}",
                        m.role
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

/// Frontier models are not shadowed: there is no upgrade path to compare against.
fn is_frontier_model(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("opus") || m == "gpt-5.4" || m.contains("o3") || m.contains("o1-pro")
}

/// True when `message.content` is empty or whitespace. Reasoning-mode models
/// sometimes emit every token into a `reasoning` field, leaving the visible
/// content blank — a broken response from the caller's point of view.
fn response_visible_content_empty(resp: &tokenmiser_providers::ChatResponse) -> bool {
    let Some(choice) = resp.choices.first() else {
        return true;
    };
    match &choice.message.content {
        Value::String(s) => s.trim().is_empty(),
        Value::Null => true,
        Value::Array(arr) => {
            // Multi-part content: empty when it holds no text blocks.
            !arr.iter().any(|item| {
                item.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| !s.trim().is_empty())
                    .unwrap_or(false)
            })
        }
        _ => false,
    }
}

/// Write a response header and body, suppressing the body for HEAD while
/// keeping its headers per RFC 9110 §9.3.2. Every terminal non-SSE response
/// funnels through here so HEAD works uniformly on every route.
async fn finish_response(session: &mut Session, resp: ResponseHeader, body: Bytes) -> Result<()> {
    let is_head = session.req_header().method == http::Method::HEAD;
    session
        .write_response_header(Box::new(resp), is_head)
        .await?;
    if !is_head {
        session.write_response_body(Some(body), true).await?;
    }
    Ok(())
}

async fn send_text(session: &mut Session, status: u16, body: &str) -> Result<()> {
    let bytes = Bytes::from(body.to_owned());
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "text/plain; charset=utf-8")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    finish_response(session, resp, bytes).await
}

async fn send_html(session: &mut Session, status: u16, body: &str) -> Result<()> {
    let bytes = Bytes::from(body.to_owned());
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "text/html; charset=utf-8")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    finish_response(session, resp, bytes).await
}

/// Serve an embedded JavaScript asset; htmx is vendored so the dashboard
/// works offline.
async fn send_js(session: &mut Session, status: u16, body: &'static str) -> Result<()> {
    let bytes = Bytes::from_static(body.as_bytes());
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "text/javascript; charset=utf-8")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    resp.insert_header("cache-control", "public, max-age=31536000, immutable")?;
    finish_response(session, resp, bytes).await
}

async fn send_json(session: &mut Session, status: u16, body: Bytes) -> Result<()> {
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", body.len().to_string())?;
    finish_response(session, resp, body).await
}

async fn send_json_with_header(
    session: &mut Session,
    status: u16,
    body: Bytes,
    hdr_name: &'static str,
    hdr_val: &'static str,
) -> Result<()> {
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", body.len().to_string())?;
    resp.insert_header(hdr_name, hdr_val)?;
    finish_response(session, resp, body).await
}

#[allow(clippy::too_many_arguments)]
async fn send_routed_response(
    session: &mut Session,
    status: u16,
    body: Bytes,
    decision: &tokenmiser_router::RouteDecision,
    real_model: &str,
    visible_empty: bool,
    budget: Option<&BudgetStatus>,
) -> Result<()> {
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", body.len().to_string())?;
    resp.insert_header("x-tokenmiser-cache", "miss")?;
    if visible_empty {
        resp.insert_header(
            "x-tokenmiser-warning",
            "upstream returned empty message.content (likely reasoning-mode model)",
        )?;
    }
    if let Some(v) = budget_header_value(budget) {
        resp.insert_header("x-tokenmiser-budget", v)?;
    }
    insert_route_headers(&mut resp, decision, real_model)?;
    finish_response(session, resp, body).await
}

async fn send_routed_error(
    session: &mut Session,
    status: u16,
    msg: &str,
    decision: &tokenmiser_router::RouteDecision,
    real_model: &str,
) -> Result<()> {
    let body = openai_error_body(status, msg);
    let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    insert_route_headers(&mut resp, decision, real_model)?;
    finish_response(session, resp, bytes).await
}

fn insert_route_headers(
    resp: &mut ResponseHeader,
    decision: &tokenmiser_router::RouteDecision,
    real_model: &str,
) -> Result<()> {
    resp.insert_header(
        "x-tokenmiser-difficulty",
        match decision.difficulty {
            tokenmiser_router::Difficulty::Easy => "easy",
            tokenmiser_router::Difficulty::Medium => "medium",
            tokenmiser_router::Difficulty::Hard => "hard",
        },
    )?;
    resp.insert_header(
        "x-tokenmiser-tier",
        match decision.tier {
            RouteTier::Explicit => "explicit",
            RouteTier::Heuristic => "heuristic",
            RouteTier::Semantic => "semantic",
        },
    )?;
    resp.insert_header("x-tokenmiser-routed-to", real_model.to_string())?;
    Ok(())
}

/// OpenAI-shaped error response, so SDK clients raise typed exceptions
/// instead of parse failures.
async fn send_error(session: &mut Session, status: u16, msg: &str) -> Result<()> {
    let body = openai_error_body(status, msg);
    let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
    send_json(session, status, bytes).await
}

/// Spin up a Pingora proxy server bound to `state.config.listen.proxy_addr`.
pub fn build_server(state: Arc<AppState>) -> pingora::server::Server {
    let mut server = pingora::server::Server::new(None).expect("server build");
    server.bootstrap();

    let mut svc = pingora_proxy::http_proxy_service(
        &server.configuration,
        TokenmiserProxy::new(state.clone()),
    );
    svc.add_tcp(&state.config.listen.proxy_addr);

    server.add_service(svc);
    server
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_accumulator_extracts_usage_from_last_chunk() {
        let mut acc = StreamAccumulator::new();
        // Split mid-event across two packets.
        let part1 = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
                     data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],";
        let part2 =
            "\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n\
                     data: [DONE]\n\n";
        acc.push(part1.as_bytes());
        assert!(acc.usage.is_none(), "no complete usage event yet");
        acc.push(part2.as_bytes());
        let u = acc.usage.as_ref().expect("usage parsed");
        assert_eq!(u.prompt_tokens, 7);
        assert_eq!(u.total_tokens, 10);
        assert_eq!(acc.content, "hi world");
        assert!(acc.saw_done);
        let full = acc.into_chat_response("m").expect("cacheable");
        assert_eq!(
            full.choices[0].message.content,
            Value::String("hi world".into())
        );
    }

    /// Pins the locality decision the cheap leg feeds into `budget_blocks`.
    /// The Easy tier is Ollama today, so this is defense-in-depth against the
    /// tier becoming configurable.
    #[test]
    fn cascade_cheap_leg_treats_paid_easy_tier_as_blockable() {
        let exceeded_enforced = BudgetStatus {
            daily_limit_usd: Some(1.0),
            total_limit_usd: None,
            spent_today_usd: 5.0,
            spent_total_usd: 5.0,
            daily_exceeded: true,
            total_exceeded: false,
            exceeded: true,
            enforce: true,
        };

        let cheap_is_local =
            |real: &str, provider: &str, target: &str| route_is_local(provider, real, target);

        // Today's default Easy tier: free, must never be blocked.
        assert!(cheap_is_local(
            "ollama:qwen2.5:7b",
            "ollama",
            "ollama:qwen2.5:7b"
        ));
        assert!(!budget_blocks(
            &Some(exceeded_enforced.clone()),
            cheap_is_local("ollama:qwen2.5:7b", "ollama", "ollama:qwen2.5:7b")
        ));

        let paid_local = cheap_is_local("gpt-5", "openai", "gpt-5");
        assert!(!paid_local, "a paid model must not be classified as local");
        assert!(
            budget_blocks(&Some(exceeded_enforced), paid_local),
            "cascade's cheap leg must be blockable when it resolves to a paid provider"
        );
    }

    fn security(origins: &[&str]) -> tokenmiser_config::SecurityConfig {
        tokenmiser_config::SecurityConfig {
            allowed_origins: origins.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// The attack this guard exists for: a page on evil.example blind-POSTs
    /// to the local proxy as a CORS simple request, so no preflight rejects it.
    #[test]
    fn cross_site_browser_request_is_rejected() {
        assert_eq!(
            evaluate_origin(
                &security(&[]),
                Some("cross-site"),
                None,
                Some("https://evil.example")
            ),
            OriginVerdict::Deny
        );
        // Sec-Fetch-Site alone is enough; some contexts omit Origin.
        assert_eq!(
            evaluate_origin(&security(&[]), Some("cross-site"), None, None),
            OriginVerdict::Deny
        );
        // An Origin from a browser too old to send Sec-Fetch-Site.
        assert_eq!(
            evaluate_origin(&security(&[]), None, None, Some("https://evil.example")),
            OriginVerdict::Deny
        );
        assert_eq!(
            evaluate_origin(
                &security(&[]),
                Some("same-site"),
                None,
                Some("https://sub.example")
            ),
            OriginVerdict::Deny
        );
        // Unknown Sec-Fetch-Site values fail closed.
        assert_eq!(
            evaluate_origin(&security(&[]), Some("bogus"), None, None),
            OriginVerdict::Deny
        );
    }

    /// The dashboard is served by this same proxy and polls it with htmx, so
    /// same-origin traffic must work with zero configuration.
    #[test]
    fn same_origin_dashboard_requests_are_allowed() {
        assert_eq!(
            evaluate_origin(
                &security(&[]),
                Some("same-origin"),
                None,
                Some("http://127.0.0.1:8443")
            ),
            OriginVerdict::Allow
        );
        // htmx GETs often carry no Origin at all.
        assert_eq!(
            evaluate_origin(&security(&[]), Some("same-origin"), None, None),
            OriginVerdict::Allow
        );
        // A typed URL or bookmark reports `none`.
        assert_eq!(
            evaluate_origin(&security(&[]), Some("none"), None, None),
            OriginVerdict::Allow
        );
    }

    /// Chrome labels a top-level navigation `cross-site` whenever the previous
    /// page was another site, `chrome://newtab` included — which is how every
    /// fresh dashboard visit looks.
    #[test]
    fn top_level_navigation_to_the_dashboard_is_allowed() {
        assert_eq!(
            evaluate_origin(&security(&[]), Some("cross-site"), Some("navigate"), None),
            OriginVerdict::Allow
        );
        assert_eq!(
            evaluate_origin(&security(&[]), Some("none"), Some("navigate"), None),
            OriginVerdict::Allow
        );
    }

    /// A cross-site form POST is also `navigate` but carries an
    /// attacker-chosen body; the `Origin` browsers attach to form POSTs is
    /// what separates it from a real navigation.
    #[test]
    fn cross_site_form_post_navigation_is_still_rejected() {
        assert_eq!(
            evaluate_origin(
                &security(&[]),
                Some("cross-site"),
                Some("navigate"),
                Some("https://evil.example")
            ),
            OriginVerdict::Deny
        );
    }

    /// A background `fetch`, the actual attack, is `cors` or `no-cors`.
    #[test]
    fn cross_site_fetch_modes_are_still_rejected() {
        for mode in ["cors", "no-cors", "same-origin", "websocket"] {
            assert_eq!(
                evaluate_origin(
                    &security(&[]),
                    Some("cross-site"),
                    Some(mode),
                    Some("https://evil.example")
                ),
                OriginVerdict::Deny,
                "cross-site {mode} must be rejected"
            );
            assert_eq!(
                evaluate_origin(&security(&[]), Some("cross-site"), Some(mode), None),
                OriginVerdict::Deny,
                "cross-site {mode} without Origin must be rejected"
            );
        }
    }

    /// curl, the OpenAI SDKs and agent frameworks send neither header.
    #[test]
    fn non_browser_clients_are_unaffected() {
        assert_eq!(
            evaluate_origin(&security(&[]), None, None, None),
            OriginVerdict::Allow
        );
        assert_eq!(
            evaluate_origin(&security(&["http://localhost:3000"]), None, None, None),
            OriginVerdict::Allow
        );
    }

    #[test]
    fn allow_listed_origins_may_drive_the_proxy() {
        let cfg = security(&["http://localhost:3000"]);
        assert_eq!(
            evaluate_origin(
                &cfg,
                Some("cross-site"),
                None,
                Some("http://localhost:3000")
            ),
            OriginVerdict::Allow
        );
        assert_eq!(
            evaluate_origin(&cfg, Some("cross-site"), None, Some("https://evil.example")),
            OriginVerdict::Deny
        );
        // Near-miss origins must not slip through.
        assert_eq!(
            evaluate_origin(
                &cfg,
                Some("cross-site"),
                None,
                Some("http://localhost:3001")
            ),
            OriginVerdict::Deny
        );
        assert_eq!(
            evaluate_origin(
                &cfg,
                Some("cross-site"),
                None,
                Some("http://localhost:3000.evil.test")
            ),
            OriginVerdict::Deny
        );
    }

    #[test]
    fn wildcard_origin_disables_the_guard() {
        let cfg = security(&["*"]);
        assert_eq!(
            evaluate_origin(&cfg, Some("cross-site"), None, Some("https://evil.example")),
            OriginVerdict::Allow
        );
    }

    #[test]
    fn ollama_cloud_routes_are_not_local() {
        // Resolved through the local ollama provider, still remote and paid.
        assert!(!route_is_local(
            "ollama",
            "gpt-oss:20b-cloud",
            "ollama:gpt-oss:20b-cloud"
        ));
        assert!(!route_is_local(
            "ollama",
            "deepseek-v3.1:671b-cloud",
            "ollama:deepseek-v3.1:671b-cloud"
        ));
        assert!(route_is_local("ollama", "qwen2.5:7b", "ollama:qwen2.5:7b"));
        assert!(route_is_local(
            "ollama",
            "cloudy-llama:7b",
            "ollama:cloudy-llama:7b"
        ));
    }

    #[test]
    fn enforced_budget_blocks_ollama_cloud_routes() {
        let exceeded = BudgetStatus {
            daily_limit_usd: Some(1.0),
            total_limit_usd: None,
            spent_today_usd: 5.0,
            spent_total_usd: 5.0,
            daily_exceeded: true,
            total_exceeded: false,
            exceeded: true,
            enforce: true,
        };
        let cloud = route_is_local("ollama", "gpt-oss:20b-cloud", "ollama:gpt-oss:20b-cloud");
        assert!(
            budget_blocks(&Some(exceeded.clone()), cloud),
            "an Ollama Cloud call must be blocked by an enforced budget"
        );
        let local = route_is_local("ollama", "qwen2.5:7b", "ollama:qwen2.5:7b");
        assert!(
            !budget_blocks(&Some(exceeded), local),
            "genuinely local traffic must still pass an enforced budget"
        );
    }

    #[test]
    fn budget_blocks_only_paid_routes_in_enforce_mode() {
        let mk = |exceeded, enforce| BudgetStatus {
            daily_limit_usd: Some(1.0),
            total_limit_usd: None,
            spent_today_usd: if exceeded { 2.0 } else { 0.0 },
            spent_total_usd: 0.0,
            daily_exceeded: exceeded,
            total_exceeded: false,
            exceeded,
            enforce,
        };
        assert!(!budget_blocks(&None, false));
        assert!(!budget_blocks(&Some(mk(true, false)), false));
        assert!(budget_blocks(&Some(mk(true, true)), false));
        assert!(!budget_blocks(&Some(mk(true, true)), true));
        assert!(!budget_blocks(&Some(mk(false, true)), false));
    }

    #[test]
    fn budget_header_reflects_state() {
        assert_eq!(budget_header_value(None), None);
        let mut b = BudgetStatus {
            daily_limit_usd: Some(1.0),
            total_limit_usd: None,
            spent_today_usd: 0.0,
            spent_total_usd: 0.0,
            daily_exceeded: false,
            total_exceeded: false,
            exceeded: false,
            enforce: false,
        };
        assert_eq!(budget_header_value(Some(&b)), Some("ok"));
        b.exceeded = true;
        assert_eq!(budget_header_value(Some(&b)), Some("exceeded"));
    }

    /// The tenant id is a plain request header that flows into cache keys, the
    /// L2 tenant map, and structured logs.
    #[test]
    fn tenant_header_is_sanitized() {
        assert_eq!(sanitize_tenant("acme-prod"), "acme-prod");
        assert_eq!(sanitize_tenant("team_1.eu:west"), "team_1.eu:west");

        assert_eq!(sanitize_tenant(""), "default");

        // Control characters must never reach a log line.
        assert_eq!(sanitize_tenant("a\nb"), "default");
        assert_eq!(sanitize_tenant("x\r\ninjected=1"), "default");
        assert_eq!(sanitize_tenant("a\tb"), "default");

        assert_eq!(sanitize_tenant("<script>alert(1)</script>"), "default");

        let long = "t".repeat(MAX_TENANT_LEN + 1);
        assert_eq!(sanitize_tenant(&long), "default");
        let at_cap = "t".repeat(MAX_TENANT_LEN);
        assert_eq!(sanitize_tenant(&at_cap), at_cap);
    }

    #[test]
    fn ensure_include_usage_sets_flag() {
        let mut req = ChatRequest {
            model: "x".into(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: Some(true),
            extra: Default::default(),
        };
        ensure_include_usage(&mut req);
        let so = req.extra.get("stream_options").unwrap();
        assert_eq!(so.get("include_usage"), Some(&Value::Bool(true)));
    }

    #[test]
    fn ensure_include_usage_preserves_existing_options() {
        let mut existing = serde_json::Map::new();
        existing.insert("foo".into(), Value::String("bar".into()));
        let mut req = ChatRequest {
            model: "x".into(),
            messages: vec![],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: Some(true),
            extra: {
                let mut e = serde_json::Map::new();
                e.insert("stream_options".into(), Value::Object(existing));
                e
            },
        };
        ensure_include_usage(&mut req);
        let so = req.extra.get("stream_options").unwrap();
        assert_eq!(so.get("foo"), Some(&Value::String("bar".into())));
        assert_eq!(so.get("include_usage"), Some(&Value::Bool(true)));
    }
}
