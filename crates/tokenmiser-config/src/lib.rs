//! Shared configuration types: providers, models, pricing, routing rules.
//!
//! Everything that needs to be referenced across crates (cache, cost, router,
//! providers, proxy) lives here so we don't get circular dependencies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod pricing;
pub use pricing::{ModelPricing, PricingTable};

/// Upstream provider kind. Drives request shape + base URL selection.
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
    /// API key env var name (e.g. `OPENAI_API_KEY`). Read at runtime, never logged.
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
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenConfig {
    /// Pingora proxy ingress (the LLM-traffic surface).
    pub proxy_addr: String,
    /// Axum admin ingress (`/stats`, `/healthz`, dashboard).
    pub admin_addr: String,
}

impl Default for ListenConfig {
    fn default() -> Self {
        Self {
            proxy_addr: "0.0.0.0:8443".into(),
            admin_addr: "0.0.0.0:9443".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingConfig {
    /// Static model aliases (e.g. `gpt-5` always goes to provider `openai`).
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
            scope: "tenant".into(),
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_semantic_threshold() -> f32 {
    // Empirically tuned on a 10-positive / 10-negative paraphrase set
    // (see `tokenmiser-cache::threshold_bench`). 0.87 yields F1=0.900
    // (precision 0.90, recall 0.90); 0.85 yields F1=0.857 with notably
    // higher false-positive rate. Prefer precision since a false-positive
    // cache hit returns a wrong answer silently.
    0.87
}
fn default_scope() -> String {
    "tenant".into()
}
