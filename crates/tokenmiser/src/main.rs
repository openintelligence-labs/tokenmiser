//! TokenMiser daemon entry point.

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
    // rustls 0.23+ refuses to pick a CryptoProvider when several could be in
    // the dep graph; aws-lc-rs is the one Pingora pulls in.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let cfg = load_config()?;
    info!(
        proxy = %cfg.listen.proxy_addr,
        admin = %cfg.listen.admin_addr,
        providers = cfg.providers.len(),
        "starting tokenmiser"
    );

    // No authentication, so a LAN-reachable listener lets anyone on the
    // network spend the operator's API budget and read /stats.
    for (what, addr) in [
        ("proxy", &cfg.listen.proxy_addr),
        ("admin", &cfg.listen.admin_addr),
    ] {
        if is_non_loopback_bind(addr) {
            tracing::warn!(
                target: "tokenmiser::security",
                surface = what,
                addr = %addr,
                "listening on a non-loopback address with NO authentication: anyone who can \
                 reach this port can spend your API budget and read /stats. Bind 127.0.0.1 \
                 unless you intend network-wide exposure."
            );
        }
    }

    // Built explicitly so the Pingora server can take over the main thread
    // once async setup finishes.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;

    let mut registry = ProviderRegistry::from_config(&cfg);

    // Soft-fail if absent. The returned model list is what points the default
    // routing policy at a model that is actually installed.
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

    // Registered even when undetected, so a model loaded later starts serving
    // `ollama:` traffic without a daemon restart.
    let ollama = Arc::new(OllamaProvider::new(ProviderConfig::ollama_local()));
    registry.register("ollama".into(), ollama as Arc<dyn Provider>);

    let ledger = CostLedger::new(PricingTable::canonical());

    // On failure auto-routing degrades to the Tier0 heuristic rather than
    // blocking startup.
    let tier1 = match Tier1Classifier::new() {
        Ok(t) => Some(t),
        Err(e) => {
            warn!(error = %e, "Tier1 classifier init failed; auto-routing will use Tier0 only");
            None
        }
    };

    let mut policy = RoutingPolicy::default();
    // Iterate keywords in the outer loop so the first *preferred keyword* with
    // any installed match wins, rather than the first installed model matching
    // any keyword. Otherwise gemma can win over qwen/llama, and gemma's
    // reasoning-mode output leaves `message.content` empty on simple prompts.
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

    // Shadow A/B needs a frontier key to call the judge.
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

/// True when `addr` is reachable from off-host. Unparseable addresses count as
/// non-loopback, so an unrecognized bind warns rather than staying silent.
fn is_non_loopback_bind(addr: &str) -> bool {
    let host = match addr.rsplit_once(':') {
        Some((h, _)) => h.trim_start_matches('[').trim_end_matches(']'),
        None => addr,
    };
    match host.parse::<std::net::IpAddr>() {
        Ok(ip) => !ip.is_loopback(),
        // Of the hostnames, only `localhost` is definitely loopback.
        Err(_) => !host.eq_ignore_ascii_case("localhost"),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_binds_do_not_warn() {
        for a in [
            "127.0.0.1:8443",
            "127.0.0.1:9443",
            "localhost:8443",
            "[::1]:8443",
            "127.5.5.5:1",
        ] {
            assert!(!is_non_loopback_bind(a), "{a} must be treated as loopback");
        }
    }

    #[test]
    fn exposed_binds_warn() {
        for a in [
            "0.0.0.0:8443",
            "192.168.1.10:8443",
            "[::]:8443",
            "0.0.0.0:1",
        ] {
            assert!(is_non_loopback_bind(a), "{a} must be flagged as exposed");
        }
    }

    #[test]
    fn default_config_binds_loopback_only() {
        let cfg = TokenmiserConfig::default();
        assert!(
            !is_non_loopback_bind(&cfg.listen.proxy_addr),
            "default proxy_addr must be loopback, got {}",
            cfg.listen.proxy_addr
        );
        assert!(
            !is_non_loopback_bind(&cfg.listen.admin_addr),
            "default admin_addr must be loopback, got {}",
            cfg.listen.admin_addr
        );
    }
}
