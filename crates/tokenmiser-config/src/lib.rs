//! Configuration types shared across every crate, kept here to avoid circular
//! dependencies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod pricing;
pub use pricing::{ModelPricing, PricingTable};

/// Drives request shape and base-URL selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderKind {
    #[serde(rename = "openai")]
    OpenAI,
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "ollama")]
    Ollama,
    #[serde(rename = "deepseek")]
    DeepSeek,
    #[serde(rename = "gemini")]
    Gemini,
}

impl ProviderKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OpenAI => "openai",
            Self::Anthropic => "anthropic",
            Self::Ollama => "ollama",
            Self::DeepSeek => "deepseek",
            Self::Gemini => "gemini",
        }
    }
}

/// A single configured upstream provider instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub name: String,
    pub kind: ProviderKind,
    /// Base URL including /v1 prefix where applicable.
    pub base_url: String,
    /// Name of the env var holding the API key; read at runtime, never logged.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Default model used when the request specifies a generic name.
    #[serde(default)]
    pub default_model: Option<String>,
}

impl ProviderConfig {
    pub fn openai() -> Self {
        Self {
            name: "openai".into(),
            kind: ProviderKind::OpenAI,
            base_url: "https://api.openai.com/v1".into(),
            api_key_env: Some("OPENAI_API_KEY".into()),
            default_model: Some("gpt-5".into()),
        }
    }

    pub fn anthropic() -> Self {
        Self {
            name: "anthropic".into(),
            kind: ProviderKind::Anthropic,
            base_url: "https://api.anthropic.com/v1".into(),
            api_key_env: Some("ANTHROPIC_API_KEY".into()),
            default_model: Some("claude-sonnet-4-6".into()),
        }
    }

    pub fn ollama_local() -> Self {
        Self {
            name: "ollama".into(),
            kind: ProviderKind::Ollama,
            base_url: "http://localhost:11434".into(),
            api_key_env: None,
            default_model: Some("llama3.2".into()),
        }
    }
}

/// Top-level TokenMiser daemon config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenmiserConfig {
    pub listen: ListenConfig,
    pub providers: Vec<ProviderConfig>,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub budget: BudgetConfig,
    #[serde(default)]
    pub security: SecurityConfig,
}

impl Default for TokenmiserConfig {
    fn default() -> Self {
        Self {
            listen: ListenConfig::default(),
            providers: vec![
                ProviderConfig::openai(),
                ProviderConfig::anthropic(),
                ProviderConfig::ollama_local(),
            ],
            routing: RoutingConfig::default(),
            cache: CacheConfig::default(),
            budget: BudgetConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

/// Browser-facing defenses for the local proxy.
///
/// Loopback is not a boundary against the browser: any page the operator
/// visits can POST to 127.0.0.1 as a CORS simple request, which is sent
/// without a preflight. CORS blocks reading the reply, but the request has
/// already spent the budget and can poison the cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Origins permitted to drive the proxy from a browser, matched exactly
    /// against the `Origin` header. Empty (the default) accepts no
    /// cross-origin browser traffic; non-browser clients send no such header
    /// and are unaffected either way.
    ///
    /// The literal `"*"` disables the check entirely, re-opening the CSRF hole
    /// for every page the operator visits.
    #[serde(default)]
    pub allowed_origins: Vec<String>,
}

impl SecurityConfig {
    /// True when `origin` may drive the proxy from a browser.
    pub fn origin_allowed(&self, origin: &str) -> bool {
        self.allowed_origins.iter().any(|o| o == "*" || o == origin)
    }

    /// True when the operator has opened the proxy to every origin, which is
    /// the only case where browser-origin signals are ignored.
    pub fn allows_any_origin(&self) -> bool {
        self.allowed_origins.iter().any(|o| o == "*")
    }
}

/// Spend-budget thresholds. A crossed limit is surfaced via `/stats`, the
/// `x-tokenmiser-budget` header and a log line, but blocks requests only under
/// `enforce`, and then only paid routes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Max USD spend per UTC day.
    #[serde(default)]
    pub daily_usd: Option<f64>,
    /// Max USD spend over the daemon's lifetime.
    #[serde(default)]
    pub total_usd: Option<f64>,
    /// Reject paid-provider requests with HTTP 402 once a limit is exceeded.
    #[serde(default)]
    pub enforce: bool,
}

impl BudgetConfig {
    pub fn is_active(&self) -> bool {
        self.daily_usd.is_some() || self.total_usd.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenConfig {
    /// Pingora proxy ingress: the LLM-traffic surface.
    pub proxy_addr: String,
    /// Admin ingress: `/stats`, `/healthz`, dashboard.
    pub admin_addr: String,
}

impl Default for ListenConfig {
    fn default() -> Self {
        // The proxy is unauthenticated and can spend the operator's API
        // budget, so LAN exposure must be an explicit opt-in.
        Self {
            proxy_addr: "127.0.0.1:8443".into(),
            admin_addr: "127.0.0.1:9443".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Static model aliases, e.g. `gpt-5` always routing to `openai`.
    #[serde(default)]
    pub aliases: HashMap<String, ModelTarget>,
    /// Default provider when the requested model is unrecognized.
    #[serde(default)]
    pub default_provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTarget {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    #[serde(default = "default_true")]
    pub l1_enabled: bool,
    #[serde(default = "default_true")]
    pub l2_enabled: bool,
    #[serde(default = "default_semantic_threshold")]
    pub semantic_threshold: f32,
    /// Skip L2 candidates whose prompt carries a different set of number
    /// literals than the query. Embeddings are near-blind to digits, so
    /// without this a cached "Add 4 and 9" can answer "Multiply 3 by 11".
    /// Disable only for workloads that want digit-insensitive matching.
    #[serde(default = "default_true")]
    pub numeric_guard: bool,
    /// `tenant` | `user` | `session` | `global`
    #[serde(default = "default_scope")]
    pub scope: String,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_enabled: true,
            l2_enabled: true,
            semantic_threshold: default_semantic_threshold(),
            numeric_guard: true,
            scope: "tenant".into(),
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_semantic_threshold() -> f32 {
    // Tuned against `tokenmiser-cache::threshold_bench`. Biased toward
    // precision: a false-positive cache hit returns a wrong answer silently.
    0.87
}
fn default_scope() -> String {
    "tenant".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_defaults_to_inactive_warn_only() {
        let cfg = TokenmiserConfig::default();
        assert!(!cfg.budget.is_active());
        assert!(!cfg.budget.enforce);
        assert!(cfg.budget.daily_usd.is_none());
        assert!(cfg.budget.total_usd.is_none());
    }

    #[test]
    fn budget_parses_from_yaml() {
        let yaml = r#"
listen:
  proxy_addr: "127.0.0.1:1"
  admin_addr: "127.0.0.1:2"
providers: []
budget:
  daily_usd: 5.0
  enforce: true
"#;
        let cfg: TokenmiserConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.budget.is_active());
        assert_eq!(cfg.budget.daily_usd, Some(5.0));
        assert_eq!(cfg.budget.total_usd, None);
        assert!(cfg.budget.enforce);
    }

    #[test]
    fn numeric_guard_defaults_on_and_is_config_exposed() {
        assert!(CacheConfig::default().numeric_guard);

        let yaml = r#"
listen:
  proxy_addr: "127.0.0.1:1"
  admin_addr: "127.0.0.1:2"
providers: []
cache:
  numeric_guard: false
  semantic_threshold: 0.91
"#;
        let cfg: TokenmiserConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!cfg.cache.numeric_guard);
        assert_eq!(cfg.cache.semantic_threshold, 0.91);

        // Omitting the field keeps the safe default.
        let yaml = r#"
listen:
  proxy_addr: "127.0.0.1:1"
  admin_addr: "127.0.0.1:2"
providers: []
cache:
  semantic_threshold: 0.87
"#;
        let cfg: TokenmiserConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.cache.numeric_guard);
    }

    #[test]
    fn security_defaults_to_no_allowed_origins() {
        let cfg = TokenmiserConfig::default();
        assert!(cfg.security.allowed_origins.is_empty());
        assert!(!cfg.security.allows_any_origin());
        assert!(!cfg.security.origin_allowed("https://evil.example"));
        assert!(!cfg.security.origin_allowed("http://localhost:3000"));
    }

    #[test]
    fn security_allowed_origins_parse_and_match_exactly() {
        let yaml = r#"
listen:
  proxy_addr: "127.0.0.1:1"
  admin_addr: "127.0.0.1:2"
providers: []
security:
  allowed_origins:
    - "http://localhost:3000"
    - "https://app.example.com"
"#;
        let cfg: TokenmiserConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.security.origin_allowed("http://localhost:3000"));
        assert!(cfg.security.origin_allowed("https://app.example.com"));
        assert!(!cfg.security.origin_allowed("http://localhost:3001"));
        assert!(!cfg.security.origin_allowed("https://localhost:3000"));
        assert!(!cfg
            .security
            .origin_allowed("https://app.example.com.evil.test"));
        assert!(!cfg.security.allows_any_origin());
    }

    #[test]
    fn security_wildcard_opens_every_origin() {
        let yaml = r#"
listen:
  proxy_addr: "127.0.0.1:1"
  admin_addr: "127.0.0.1:2"
providers: []
security:
  allowed_origins: ["*"]
"#;
        let cfg: TokenmiserConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(cfg.security.allows_any_origin());
        assert!(cfg.security.origin_allowed("https://evil.example"));
    }

    #[test]
    fn config_without_budget_section_still_parses() {
        let yaml = r#"
listen:
  proxy_addr: "127.0.0.1:1"
  admin_addr: "127.0.0.1:2"
providers: []
"#;
        let cfg: TokenmiserConfig = serde_yaml::from_str(yaml).unwrap();
        assert!(!cfg.budget.is_active());
    }
}
