//! Pingora-based OpenAI-compatible proxy ingress.
//!
//! v0.1: non-streaming `/v1/chat/completions`. We terminate the request in
//! Pingora (rather than reverse-proxying), call the upstream provider via
//! `tokenmiser-providers`, and write the response back. This lets us
//! normalize Anthropic ↔ OpenAI ↔ Ollama at the wire boundary and inject
//! cache + cost + routing decisions cleanly.
//!
//! v0.7 will reintroduce true streaming through `ProxyHttp::upstream_peer`
//! for SSE traffic, but the terminate-in-gateway path is the right default
//! for non-streaming requests because every other architectural feature
//! (cache, cost, shadow A/B, judge) needs the full response in hand.

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
use tokenmiser_config::TokenmiserConfig;
use tokenmiser_cost::CostLedger;
use tokenmiser_mcp::McpBudgetGateway;
use tokenmiser_providers::{ChatRequest, ProviderRegistry, StreamChunk};
use tokenmiser_quality::{ShadowEnqueue, ShadowScheduler};
use tokenmiser_router::{should_escalate, CascadeConfig, EscalateDecision, RouteTier, Router};
use tracing::warn;

pub mod backpressure;
pub mod dashboard;
pub mod mcp_route;
pub use backpressure::TokenBucket;

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
        // 16k entries × ~10KB avg response = ~160MB upper-bound. Plenty for v0.2;
        // v0.3 will move to disk-backed sled if memory pressure shows up in
        // production benchmarks.
        let l1 = L1Cache::new(16_384, Duration::from_secs(3600));

        let l2 = if config.cache.l2_enabled {
            match L2Cache::new(
                config.cache.semantic_threshold,
                Duration::from_secs(3600),
                4_096,
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

    /// Pingora demands an upstream peer even when we terminate inside the
    /// gateway. We point it at a sentinel `127.0.0.1:1` — `request_filter`
    /// short-circuits the request before this is ever dialed, but Pingora's
    /// trait shape requires it.
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

    /// All request handling happens here. We read the body inline via
    /// `session.read_request_body()` and write the response, then return
    /// `Ok(true)` so Pingora never tries to dial an upstream. This is the
    /// idiomatic terminate-in-gateway pattern for `ProxyHttp`.
    async fn request_filter(&self, session: &mut Session, ctx: &mut Self::CTX) -> Result<bool> {
        let (method, path, tenant) = {
            let req: &RequestHeader = session.req_header();
            (
                req.method.as_str().to_string(),
                req.uri.path().to_string(),
                req.headers
                    .get("x-tokenmiser-tenant")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("default")
                    .to_string(),
            )
        };
        ctx.path = path.clone();
        ctx.tenant = tenant;

        match (method.as_str(), path.as_str()) {
            ("GET", "/healthz") => {
                send_text(session, 200, "ok").await?;
                Ok(true)
            }
            ("GET", "/stats") => {
                let snap = self.state.ledger.snapshot();
                let l1 = self.state.l1.stats();
                let l2 = self.state.l2.as_ref().map(|c| c.stats());
                let body = serde_json::json!({
                    "cost": snap,
                    "cache_l1": l1,
                    "cache_l2": l2,
                });
                let body = serde_json::to_vec_pretty(&body).unwrap_or_default();
                send_json(session, 200, Bytes::from(body)).await?;
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
                send_text(session, 404, "not found").await?;
                Ok(true)
            }
        }
    }
}

async fn handle_chat_completions(
    state: &AppState,
    session: &mut Session,
    _ctx: &mut ReqCtx,
) -> Result<()> {
    // Drain the full request body. v0.1 = non-streaming, so buffering is fine;
    // v0.7 streaming will fork off here for stream=true requests.
    let mut buf = Vec::new();
    loop {
        match session.read_request_body().await? {
            Some(chunk) if !chunk.is_empty() => buf.extend_from_slice(&chunk),
            Some(_) => {}
            None => break,
        }
    }

    let parsed: ChatRequest = match serde_json::from_slice(&buf) {
        Ok(p) => p,
        Err(e) => {
            return send_error(session, 400, &format!("bad request body: {e}")).await;
        }
    };

    // Validate request shape before any routing / provider / cache work.
    // Returning a clean 400 here keeps caller error messages honest — a
    // missing/empty model field previously fell through to the default
    // provider and 401'd on its missing API key, which was confusing.
    if let Err(msg) = validate_chat_request(&parsed) {
        return send_error(session, 400, &msg).await;
    }

    // L1 cache lookup before any provider work (architecture §4: 10µs).
    if let Some(cached) = state.l1.lookup(&parsed, &_ctx.tenant) {
        state.ledger.record_cache_hit(
            &parsed.model,
            cached.usage.prompt_tokens,
            cached.usage.completion_tokens,
        );
        let mut out = cached;
        out.model = parsed.model.clone();
        let body = serde_json::to_vec(&out).unwrap_or_default();
        return send_json_with_header(
            session,
            200,
            Bytes::from(body),
            "x-tokenmiser-cache",
            "l1-hit",
        )
        .await;
    }
    state.l1.record_miss();

    // L2 semantic lookup (only when enabled).
    if let Some(l2) = &state.l2 {
        if let Some(cached) = l2.lookup(&parsed, &_ctx.tenant) {
            state.ledger.record_cache_hit(
                &parsed.model,
                cached.usage.prompt_tokens,
                cached.usage.completion_tokens,
            );
            // Also seed L1 so the next identical hit is the fast path.
            state.l1.insert(&parsed, &_ctx.tenant, &cached);
            let mut out = cached;
            out.model = parsed.model.clone();
            let body = serde_json::to_vec(&out).unwrap_or_default();
            return send_json_with_header(
                session,
                200,
                Bytes::from(body),
                "x-tokenmiser-cache",
                "l2-hit",
            )
            .await;
        }
    }

    // v0.7: streaming path. When the caller sets `stream: true` we cannot
    // buffer-then-cache; we pump SSE chunks straight through with backpressure
    // and skip cache writes (a v0.7.1 follow-up will reconstruct full
    // responses from streamed chunks for cache insertion).
    if parsed.stream == Some(true) {
        return handle_stream(state, session, _ctx, &parsed).await;
    }

    // v0.5: cascade is a routing mode, not a difficulty class. When the
    // caller asks for `tokenmiser:cascade` (or `auto:cascade`) we try the
    // Easy-tier model first, inspect confidence, escalate on miss.
    if matches!(parsed.model.as_str(), "tokenmiser:cascade" | "auto:cascade") {
        return handle_cascade(state, session, _ctx, &parsed).await;
    }

    // v0.4: route → resolve → call.
    let decision = state.router.decide(&parsed);
    let target_model = decision.target.model.clone();

    let (provider, real_model) = match state.registry.resolve(&target_model) {
        Ok(t) => t,
        Err(e) => {
            return send_error(session, 400, &format!("model resolution: {e}")).await;
        }
    };

    let mut upstream_req = parsed.clone();
    upstream_req.model = real_model.clone();

    match provider.complete(&upstream_req).await {
        Ok(resp) => {
            let prompt_tokens = resp.usage.prompt_tokens;
            let completion_tokens = resp.usage.completion_tokens;
            let is_local = real_model.starts_with("ollama")
                || provider.name() == "ollama"
                || target_model.starts_with("ollama");

            if is_local {
                state.ledger.record_free(
                    decision.counterfactual_model.as_deref(),
                    prompt_tokens,
                    completion_tokens,
                );
            } else {
                state
                    .ledger
                    .record_paid(&real_model, prompt_tokens, completion_tokens);
                // If a cheaper model was the counterfactual baseline (rare
                // — usually frontier IS the counterfactual), record the
                // delta so saved_usd never goes negative.
                if let Some(cf) = &decision.counterfactual_model {
                    if cf != &real_model {
                        // Counterfactual already attributed via record_paid path
                        // when real_model is itself the frontier. No-op here.
                    }
                }
            }

            // Empty-content guard: some models (gemma reasoning, deepseek-r1)
            // can return responses with no visible content on simple prompts.
            // Don't cache those — caching a broken response would amplify
            // the failure across the tenant's whole prompt cluster.
            let visible_empty = response_visible_content_empty(&resp);
            if visible_empty {
                warn!(
                    model = %real_model,
                    "upstream returned empty message.content; skipping cache + flagging response"
                );
            } else {
                state.l1.insert(&parsed, &_ctx.tenant, &resp);
                if let Some(l2) = &state.l2 {
                    l2.insert(&parsed, &_ctx.tenant, &resp);
                }
            }

            // Shadow A/B (architecture §5): if we routed to a non-frontier
            // model, sample a fraction of responses and compare against the
            // frontier in the background. The user already got their
            // response; this never blocks them.
            if let Some(shadow) = &state.shadow {
                if !is_frontier_model(&real_model) {
                    shadow.maybe_enqueue(ShadowEnqueue::from_request(&parsed, &resp, &real_model));
                }
            }

            // Echo the originally-requested model so clients see what they
            // asked for (incl. `auto`), not the routed-to model.
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
            )
            .await
        }
        Err(e) => {
            let status = provider_error_status(&e);
            send_routed_error(session, status, &e.to_string(), &decision, &real_model).await
        }
    }
}

/// Map a ProviderError to an HTTP status code. Single source of truth so
/// streaming and non-streaming paths agree on the mapping.
fn provider_error_status(e: &tokenmiser_providers::ProviderError) -> u16 {
    use tokenmiser_providers::ProviderError::*;
    match e {
        Upstream { status, .. } => *status,
        MissingApiKey(_) => 401,
        UnknownModel { .. } | NotFound { .. } => 400,
        _ => 502,
    }
}

/// SSE streaming path. Route → upstream stream → pump chunks with
/// per-connection backpressure → end. Cache write is skipped for v0.7
/// (full-response reassembly for cache insertion is a v0.7.1 follow-up).
async fn handle_stream(
    state: &AppState,
    session: &mut Session,
    _ctx: &mut ReqCtx,
    parsed: &ChatRequest,
) -> Result<()> {
    let decision = state.router.decide(parsed);
    let target_model = decision.target.model.clone();

    let (provider, real_model) = match state.registry.resolve(&target_model) {
        Ok(t) => t,
        Err(e) => return send_error(session, provider_error_status(&e), &e.to_string()).await,
    };

    let mut upstream_req = parsed.clone();
    upstream_req.model = real_model.clone();
    // Ensure the upstream emits `usage` in the final stream chunk so we can
    // record real token counts. OpenAI and Ollama both honor this; Anthropic
    // streams emit usage in the `message_delta` event by default.
    ensure_include_usage(&mut upstream_req);

    let mut stream = match provider.stream(&upstream_req).await {
        Ok(s) => s,
        Err(e) => {
            let status = provider_error_status(&e);
            return send_routed_error(session, status, &e.to_string(), &decision, &real_model)
                .await;
        }
    };

    // SSE response headers.
    let mut resp = ResponseHeader::build(200, None)?;
    resp.insert_header("content-type", "text/event-stream")?;
    resp.insert_header("cache-control", "no-cache")?;
    resp.insert_header("connection", "keep-alive")?;
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
    session.write_response_header(Box::new(resp), false).await?;

    let mut bucket = TokenBucket::default_streaming();
    let mut total_bytes_out: usize = 0;
    // Buffer used to parse SSE chunks across reqwest packet boundaries.
    let mut sse_buf = String::new();
    let mut last_usage: Option<tokenmiser_providers::Usage> = None;
    let mut client_disconnected = false;

    // Liveness probe interval. Architecture §14.4 calls for
    // cancel-within-200ms but measured behavior in v0.7 is closer to
    // "cancel on next upstream chunk" (bounded by upstream pace). For
    // typical LLM streaming (10-30 chunks/sec) this is ~30-100ms; for
    // very slow streams it can be up to one upstream-chunk interval. The
    // zero-byte probe below covers the long-silence case.
    //
    // NOTE: Pingora's `write_response_body(empty_bytes, false)` does not
    // reliably error on a closed downstream until real bytes are queued.
    // Tightening this further is a v1.1 task tracked in the roadmap.
    const LIVENESS_PROBE: std::time::Duration = std::time::Duration::from_millis(100);

    loop {
        // Race the next upstream chunk against a liveness-probe timer. The
        // tokio::select! ensures we wake up periodically even when upstream
        // is silent — that's how we notice the client closed.
        let next = tokio::time::timeout(LIVENESS_PROBE, stream.next()).await;
        match next {
            Err(_elapsed) => {
                // Liveness probe: send an SSE comment (`: ping\n\n`).
                // SSE clients ignore comment lines per the spec; the
                // bytes hit the wire so a closed downstream surfaces as
                // a write error. Empty-byte writes are coalesced inside
                // Pingora and don't reliably probe the socket.
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
            Ok(None) => break, // upstream done
            Ok(Some(Ok(StreamChunk::Sse(bytes)))) => {
                bucket.acquire(bytes.len()).await;
                total_bytes_out += bytes.len();

                // Parse SSE for usage *before* forwarding so we still
                // capture token counts even if the client disconnects
                // mid-stream.
                if let Ok(s) = std::str::from_utf8(&bytes) {
                    sse_buf.push_str(s);
                    drain_complete_sse_events(&mut sse_buf, &mut last_usage);
                }

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
                break;
            }
        }
    }

    if !client_disconnected {
        // Final empty chunk to signal end-of-body.
        let _ = session
            .write_response_body(Some(bytes::Bytes::new()), true)
            .await;
    }

    // Real token counts when the upstream emitted usage; zero otherwise so
    // we still count the request but don't lie about cost.
    let (prompt_tokens, completion_tokens) = last_usage
        .as_ref()
        .map(|u| (u.prompt_tokens, u.completion_tokens))
        .unwrap_or((0, 0));

    if real_model.starts_with("ollama") || provider.name() == "ollama" {
        state.ledger.record_free(
            decision.counterfactual_model.as_deref(),
            prompt_tokens,
            completion_tokens,
        );
    } else {
        state
            .ledger
            .record_paid(&real_model, prompt_tokens, completion_tokens);
    }

    Ok(())
}

/// Force `stream_options.include_usage = true` on outbound streaming
/// requests so providers emit a usage block in the final SSE event.
fn ensure_include_usage(req: &mut ChatRequest) {
    let entry = req
        .extra
        .entry("stream_options")
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    if let Value::Object(map) = entry {
        map.entry("include_usage").or_insert(Value::Bool(true));
    }
}

/// Pull complete `data: {...}\n\n` events from `buf`. For each event whose
/// JSON contains a `usage` field, update `last_usage`. Incomplete trailing
/// events stay in `buf` for the next chunk.
fn drain_complete_sse_events(
    buf: &mut String,
    last_usage: &mut Option<tokenmiser_providers::Usage>,
) {
    while let Some(end) = buf.find("\n\n") {
        let event: String = buf.drain(..end + 2).collect();
        // Each event may have multiple `data:` lines. We only inspect the
        // ones we can parse as JSON; ignore comments / `event: ...` lines.
        for line in event.lines() {
            let payload = match line.strip_prefix("data:") {
                Some(p) => p.trim(),
                None => continue,
            };
            if payload == "[DONE]" || payload.is_empty() {
                continue;
            }
            // Try to parse usage out of the chunk. Many chunks won't have
            // it — that's fine, we just keep looking.
            if let Ok(v) = serde_json::from_str::<Value>(payload) {
                if let Some(u) = v.get("usage").and_then(|u| {
                    serde_json::from_value::<tokenmiser_providers::Usage>(u.clone()).ok()
                }) {
                    *last_usage = Some(u);
                }
            }
        }
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

    // Pick cheap = Easy-tier policy target; frontier = Hard-tier.
    let easy_target = state
        .router
        .policy_target(tokenmiser_router::Difficulty::Easy);
    let hard_target = state
        .router
        .policy_target(tokenmiser_router::Difficulty::Hard);

    // Step 1: try cheap. Ask for logprobs so we can score confidence.
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

    let cheap_resp = match cheap_provider.complete(&cheap_req).await {
        Ok(r) => r,
        Err(e) => {
            return send_error(session, 502, &format!("cascade cheap call failed: {e}")).await
        }
    };

    let decision = should_escalate(&cheap_resp, &cfg);
    match decision {
        EscalateDecision::No { signal } => {
            // Accept the cheap response. Counterfactual = the frontier.
            state.ledger.record_free(
                Some(&hard_target.model),
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
            // Escalate to frontier. The cheap call wasn't wasted — we'll
            // ledger-record both (cheap=free; frontier=paid).
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
            // Cheap was free; frontier was paid.
            state
                .ledger
                .record_free(None, cheap_resp.usage.prompt_tokens, 0);
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
    session.write_response_header(Box::new(resp), false).await?;
    session.write_response_body(Some(body), true).await?;
    Ok(())
}

async fn handle_mcp_tools_call(state: &AppState, session: &mut Session) -> Result<()> {
    use mcp_route::*;
    let mut buf = Vec::new();
    loop {
        match session.read_request_body().await? {
            Some(chunk) if !chunk.is_empty() => buf.extend_from_slice(&chunk),
            Some(_) => {}
            None => break,
        }
    }
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

    // 1) Cap check before any work.
    match state.mcp.check(&agent, &tool) {
        Ok(spend) => {
            // 2) Record actual cost if the client supplied it (post-hoc reporting).
            if let Some(actual) = req.params.actual_cost_usd {
                state.mcp.record(&agent, &tool, actual);
            } else if let Some(est) = req.params.estimated_cost_usd {
                // Pre-record the estimate so concurrent calls see the
                // budget effect. Real impl would reconcile after upstream.
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
            // HTTP 402 Payment Required is semantically right for budget exceeded.
            send_json(session, 402, Bytes::from(body)).await
        }
    }
}

async fn handle_mcp_set_budget(state: &AppState, session: &mut Session) -> Result<()> {
    use mcp_route::*;
    let mut buf = Vec::new();
    loop {
        match session.read_request_body().await? {
            Some(chunk) if !chunk.is_empty() => buf.extend_from_slice(&chunk),
            Some(_) => {}
            None => break,
        }
    }
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

/// Sanity-check a parsed ChatRequest. Returns Err with a user-facing
/// message if the request is structurally invalid.
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
        // Allow tool/function messages with no content; require content
        // for user/system/assistant.
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

/// True if `model` is the frontier — we don't shadow these (no upgrade path).
fn is_frontier_model(model: &str) -> bool {
    let m = model.to_lowercase();
    m.contains("opus") || m == "gpt-5.4" || m.contains("o3") || m.contains("o1-pro")
}

/// Detect responses where `message.content` is empty/whitespace. Reasoning-
/// mode models (gemma4, deepseek-r1, gpt-oss thinking variants) sometimes
/// emit all their tokens into a `reasoning` field and leave the visible
/// content blank — that's a broken response from our caller's POV.
fn response_visible_content_empty(resp: &tokenmiser_providers::ChatResponse) -> bool {
    let Some(choice) = resp.choices.first() else {
        return true;
    };
    match &choice.message.content {
        Value::String(s) => s.trim().is_empty(),
        Value::Null => true,
        Value::Array(arr) => {
            // Multi-part content (vision-style): empty if no text blocks.
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

async fn send_text(session: &mut Session, status: u16, body: &str) -> Result<()> {
    let bytes = Bytes::from(body.to_owned());
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "text/plain; charset=utf-8")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    session.write_response_header(Box::new(resp), false).await?;
    session.write_response_body(Some(bytes), true).await?;
    Ok(())
}

async fn send_html(session: &mut Session, status: u16, body: &str) -> Result<()> {
    let bytes = Bytes::from(body.to_owned());
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "text/html; charset=utf-8")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    session.write_response_header(Box::new(resp), false).await?;
    session.write_response_body(Some(bytes), true).await?;
    Ok(())
}

async fn send_json(session: &mut Session, status: u16, body: Bytes) -> Result<()> {
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", body.len().to_string())?;
    session.write_response_header(Box::new(resp), false).await?;
    session.write_response_body(Some(body), true).await?;
    Ok(())
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
    session.write_response_header(Box::new(resp), false).await?;
    session.write_response_body(Some(body), true).await?;
    Ok(())
}

async fn send_routed_response(
    session: &mut Session,
    status: u16,
    body: Bytes,
    decision: &tokenmiser_router::RouteDecision,
    real_model: &str,
    visible_empty: bool,
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
    insert_route_headers(&mut resp, decision, real_model)?;
    session.write_response_header(Box::new(resp), false).await?;
    session.write_response_body(Some(body), true).await?;
    Ok(())
}

async fn send_routed_error(
    session: &mut Session,
    status: u16,
    msg: &str,
    decision: &tokenmiser_router::RouteDecision,
    real_model: &str,
) -> Result<()> {
    let body = serde_json::json!({
        "error": {"message": msg, "type": "tokenmiser_error"},
    });
    let bytes = Bytes::from(serde_json::to_vec(&body).unwrap_or_default());
    let mut resp = ResponseHeader::build(status, None)?;
    resp.insert_header("content-type", "application/json")?;
    resp.insert_header("content-length", bytes.len().to_string())?;
    insert_route_headers(&mut resp, decision, real_model)?;
    session.write_response_header(Box::new(resp), false).await?;
    session.write_response_body(Some(bytes), true).await?;
    Ok(())
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

async fn send_error(session: &mut Session, status: u16, msg: &str) -> Result<()> {
    let body = serde_json::json!({
        "error": {
            "message": msg,
            "type": "tokenmiser_error",
        }
    });
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

// Silence the unused-import warning when `Value` isn't used in this file
// directly — it's exported through `tokenmiser_providers` and may be
// referenced by future v0.2+ caching code.
#[allow(dead_code)]
fn _value_typecheck(_: &Value) {}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenmiser_providers::Usage;

    #[test]
    fn sse_parser_extracts_usage_from_last_chunk() {
        let mut buf = String::new();
        let mut usage: Option<Usage> = None;

        // Two normal delta chunks, then a chunk with usage, then DONE.
        let stream = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                      data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n\
                      data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n\n\
                      data: [DONE]\n\n";
        buf.push_str(stream);
        drain_complete_sse_events(&mut buf, &mut usage);
        let u = usage.expect("usage parsed");
        assert_eq!(u.prompt_tokens, 7);
        assert_eq!(u.completion_tokens, 3);
        assert_eq!(u.total_tokens, 10);
        assert!(buf.is_empty(), "all events drained");
    }

    #[test]
    fn sse_parser_handles_split_events_across_packets() {
        let mut buf = String::new();
        let mut usage: Option<Usage> = None;

        // Simulate the upstream sending the usage chunk in two TCP packets.
        let part1 = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],";
        let part2 =
            "\"usage\":{\"prompt_tokens\":42,\"completion_tokens\":8,\"total_tokens\":50}}\n\n";

        buf.push_str(part1);
        drain_complete_sse_events(&mut buf, &mut usage);
        assert!(usage.is_none(), "no complete event yet");
        assert!(!buf.is_empty(), "partial event held over");

        buf.push_str(part2);
        drain_complete_sse_events(&mut buf, &mut usage);
        let u = usage.expect("usage parsed after second packet");
        assert_eq!(u.prompt_tokens, 42);
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
