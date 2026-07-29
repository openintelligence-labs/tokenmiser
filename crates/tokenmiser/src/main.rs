//! TokenMiser main daemon.
//!
//! Loads config, builds the provider registry + cost ledger, auto-detects a
//! local Ollama if present, and launches the Pingora ingress.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tokenmiser_config::{PricingTable, ProviderConfig, TokenmiserConfig};
use tokenmiser_cost::CostLedger;
use tokenmiser_mcp::{McpBudgetGateway, ToolBudget};
use tokenmiser_providers::{ollama::OllamaProvider, Provider, ProviderRegistry};
use tokenmiser_proxy::{build_server, AppState};
use tokenmiser_quality::{ShadowConfig, ShadowScheduler, WinRateAggregator};
use tokenmiser_router::{replay, PolicyEngine, Router, RoutingPolicy, Tier1Classifier};
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "tokenmiser", version, about = "Smart LLM router & proxy")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run the proxy daemon (default if no subcommand given).
    Serve,
    /// Policy DSL utilities.
    Policy {
        #[command(subcommand)]
        action: PolicyAction,
    },
}

#[derive(Subcommand)]
enum PolicyAction {
    /// Replay a JSONL request log against a candidate `.rhai` policy and
    /// report projected routing distribution.
    Test {
        /// Path to the request log (one JSON entry per line).
        log: PathBuf,
        /// Path to the candidate policy script.
        policy: PathBuf,
    },
}

fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => serve(),
        Command::Policy {
            action: PolicyAction::Test { log, policy },
        } => policy_test(log, policy),
    }
}

fn policy_test(log: PathBuf, policy: PathBuf) -> Result<()> {
    let engine = PolicyEngine::load(policy.clone())
        .with_context(|| format!("load policy {}", policy.display()))?;
    let result = replay(&log, &engine).context("replay")?;
    println!("Replayed {} request(s).", result.total);
    if result.failed > 0 {
        println!("  Failed:    {}", result.failed);
    }
    let mut targets: Vec<_> = result.by_target.into_iter().collect();
    targets.sort_by_key(|t| std::cmp::Reverse(t.1));
    println!("Projected routing distribution:");
    for (target, n) in targets {
        let pct = if result.total > 0 {
            (n as f64 / result.total as f64) * 100.0
        } else {
            0.0
        };
        println!("  {:>6} ({:>5.1}%)  {}", n, pct, target);
    }
    Ok(())
}

fn serve() -> Result<()> {
    // Required because we enable Pingora's rustls feature; rustls 0.23+ won't
    // pick a CryptoProvider by default when more than one is potentially in
    // the dep graph. We default to aws-lc-rs which Pingora pulls in.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let cfg = load_config()?;
    info!(
        proxy = %cfg.listen.proxy_addr,
        admin = %cfg.listen.admin_addr,
        providers = cfg.providers.len(),
        "starting tokenmiser"
    );

    // Async setup happens on a Tokio runtime built explicitly so the Pingora
    // server can take over the main thread afterward.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;

    let mut registry = ProviderRegistry::from_config(&cfg);

    // Auto-detect a running Ollama on localhost — the local-first hook in
    // architecture §7. Soft-fail if absent. Returns the list of loaded
    // models so we can update the default routing policy to point at one
    // that's actually installed.
    let detected_ollama: Vec<String> = rt.block_on(async {
        let probe = ProviderConfig::ollama_local();
        match OllamaProvider::detect(&probe.base_url).await {
            Ok(models) if !models.is_empty() => {
                info!(
                    models = ?models,
                    "auto-detected local Ollama — registered as free provider"
                );
                models
            }
            Ok(_) => {
                warn!("Ollama reachable but no models loaded");
                vec![]
            }
            Err(e) => {
                info!(error = %e, "no local Ollama detected; remote-only routing");
                vec![]
            }
        }
    });

    // Register the configured Ollama client regardless — the registry's
    // resolver will route `ollama:` and `llama*` traffic to it once a model
    // is loaded later, without a daemon restart.
    let ollama = Arc::new(OllamaProvider::new(ProviderConfig::ollama_local()));
    registry.register("ollama".into(), ollama as Arc<dyn Provider>);

    let ledger = CostLedger::new(PricingTable::canonical());

    // Build the Tier1 semantic classifier. Falls back to None on failure
    // (which means auto-routing degrades to pure Tier0 heuristic — still
    // useful, never blocks startup).
    let tier1 = match Tier1Classifier::new() {
        Ok(t) => Some(t),
        Err(e) => {
            warn!(error = %e, "Tier1 classifier init failed; auto-routing will use Tier0 only");
            None
        }
    };

    let mut policy = RoutingPolicy::default();
    // v0.6: if a local Ollama model is detected, prefer it for Easy
    // difficulty (the zero-config local-first hook from architecture §7).
    //
    // Walk the preferred list in priority order — the *first preferred
    // keyword* that any installed model matches wins, not the first
    // installed model that happens to match any keyword. This avoids
    // picking gemma when qwen/llama are available, since gemma's
    // reasoning-mode output leaves `message.content` empty on simple
    // prompts (real bug observed against gemma4:latest).
    //
    // Also exclude reasoning-mode and embedding-only models.
    let exclude = ["embed", "reasoning", "-thinking"];
    let preferred = [
        "qwen2.5", "llama3", "llama2", "mistral", "phi3", "qwen", "phi", "gemma",
    ];
    let chosen_easy = preferred.iter().find_map(|kw| {
        detected_ollama.iter().find(|m| {
            let lower = m.to_lowercase();
            !exclude.iter().any(|e| lower.contains(e)) && lower.contains(kw)
        })
    });
    if let Some(model) = chosen_easy {
        let model_id = format!("ollama:{}", model);
        info!(model = %model_id, "routing Easy traffic to detected local Ollama model");
        policy.tiers.insert(
            tokenmiser_router::Difficulty::Easy,
            tokenmiser_router::RoutingTarget {
                provider: "ollama".into(),
                model: model_id,
            },
        );
    }
    let router = Router::new(policy, tier1);

    // v0.8: shadow A/B is only meaningful when a frontier API key is
    // available — without one, the judge can't be called. Soft-skip if
    // ANTHROPIC_API_KEY is absent.
    let registry_arc = Arc::new(ProviderRegistry::from_config(&cfg));
    let shadow = if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        let cfg = ShadowConfig::default();
        let agg = WinRateAggregator::new(&cfg);
        Some(ShadowScheduler::new(cfg, registry_arc, agg))
    } else {
        info!("ANTHROPIC_API_KEY absent — shadow A/B disabled");
        None
    };

    let mcp = McpBudgetGateway::new(ToolBudget::default());

    let state = AppState::new(cfg, registry, router, ledger, shadow, mcp);

    // Pingora takes the thread from here.
    let server = build_server(state);
    server.run_forever();
}

// Unreachable but the explicit Ok keeps the type signature uniform with the
// `Command::Policy` arm above.
#[allow(dead_code)]
fn _serve_returns_ok() -> Result<()> {
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,pingora=warn"));
    fmt().with_env_filter(filter).compact().init();
}

fn load_config() -> Result<TokenmiserConfig> {
    let path = std::env::var("TOKENMISER_CONFIG")
        .map(PathBuf::from)
        .ok()
        .filter(|p| p.exists());

    if let Some(path) = path {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read config {}", path.display()))?;
        let cfg: TokenmiserConfig = serde_yaml::from_str(&raw).context("parse config yaml")?;
        Ok(cfg)
    } else {
        Ok(TokenmiserConfig::default())
    }
}
