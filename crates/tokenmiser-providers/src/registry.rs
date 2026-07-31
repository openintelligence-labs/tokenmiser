//! Resolves a model name to a concrete `Provider`, in order: configured
//! alias, `provider:model` prefix, model-family heuristic, then the opt-in
//! `routing.default_provider`.

use std::collections::HashMap;
use std::sync::Arc;

use tokenmiser_config::{ProviderConfig, ProviderKind, TokenmiserConfig};

use crate::{
    anthropic::AnthropicProvider, ollama::OllamaProvider, openai::OpenAIProvider, Provider,
    ProviderError,
};

pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
    aliases: HashMap<String, (String, String)>, // model -> (provider_name, real_model)
    default_provider: Option<String>,
}

impl ProviderRegistry {
    pub fn from_config(cfg: &TokenmiserConfig) -> Self {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        for p in &cfg.providers {
            let provider: Arc<dyn Provider> = build_provider(p.clone());
            providers.insert(p.name.clone(), provider);
        }

        let aliases = cfg
            .routing
            .aliases
            .iter()
            .map(|(model, target)| {
                (
                    model.clone(),
                    (target.provider.clone(), target.model.clone()),
                )
            })
            .collect();

        Self {
            providers,
            aliases,
            default_provider: cfg.routing.default_provider.clone(),
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).cloned()
    }

    pub fn register(&mut self, name: String, provider: Arc<dyn Provider>) {
        self.providers.insert(name, provider);
    }

    pub fn names(&self) -> Vec<String> {
        self.providers.keys().cloned().collect()
    }

    /// Pick a provider + real model name for an incoming `model` request.
    pub fn resolve(&self, model: &str) -> Result<(Arc<dyn Provider>, String), ProviderError> {
        // Explicit alias.
        if let Some((provider_name, real_model)) = self.aliases.get(model) {
            if let Some(p) = self.providers.get(provider_name) {
                return Ok((p.clone(), real_model.clone()));
            }
        }

        // `provider:model` prefix.
        if let Some((prefix, rest)) = model.split_once(':') {
            if let Some(p) = self.providers.get(prefix) {
                return Ok((p.clone(), rest.to_string()));
            }
        }

        // Model-family heuristic.
        let lower = model.to_lowercase();
        let guess: Option<&str> = if lower.starts_with("claude") {
            Some("anthropic")
        } else if lower.starts_with("gpt") || lower.starts_with("o1") || lower.starts_with("o3") {
            Some("openai")
        } else if lower.starts_with("gemini") {
            Some("gemini")
        } else if lower.starts_with("deepseek") {
            Some("deepseek")
        } else if lower.starts_with("llama")
            || lower.starts_with("qwen")
            || lower.starts_with("mistral")
            || lower.starts_with("phi")
            || lower.starts_with("gemma")
        {
            Some("ollama")
        } else {
            None
        };

        if let Some(name) = guess {
            if let Some(p) = self.providers.get(name) {
                return Ok((p.clone(), model.to_string()));
            }
        }

        // Opt-in only: falling through to a guessed default would route
        // typos and unsupported models to an arbitrary configured provider,
        // which then fails with an unrelated missing-API-key error.
        if let Some(name) = &self.default_provider {
            if let Some(p) = self.providers.get(name) {
                return Ok((p.clone(), model.to_string()));
            }
        }

        let mut known: Vec<String> = self.providers.keys().cloned().collect();
        known.sort();
        Err(ProviderError::UnknownModel {
            model: model.to_string(),
            known_providers: known.join(", "),
        })
    }
}

fn build_provider(cfg: ProviderConfig) -> Arc<dyn Provider> {
    match cfg.kind {
        ProviderKind::OpenAI | ProviderKind::DeepSeek => Arc::new(OpenAIProvider::new(cfg)),
        ProviderKind::Anthropic => Arc::new(AnthropicProvider::new(cfg)),
        ProviderKind::Ollama => Arc::new(OllamaProvider::new(cfg)),
        // Gemini goes through its OpenAI-compatibility layer.
        ProviderKind::Gemini => Arc::new(OpenAIProvider::new(cfg)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenmiser_config::{ListenConfig, ModelTarget, RoutingConfig};

    fn cfg() -> TokenmiserConfig {
        let mut aliases = HashMap::new();
        aliases.insert(
            "gpt-5".into(),
            ModelTarget {
                provider: "openai".into(),
                model: "gpt-5".into(),
            },
        );
        TokenmiserConfig {
            listen: ListenConfig::default(),
            providers: vec![
                ProviderConfig::openai(),
                ProviderConfig::anthropic(),
                ProviderConfig::ollama_local(),
            ],
            routing: RoutingConfig {
                aliases,
                default_provider: Some("openai".into()),
            },
            cache: Default::default(),
            budget: Default::default(),
            security: Default::default(),
        }
    }

    #[test]
    fn resolves_claude_to_anthropic() {
        let reg = ProviderRegistry::from_config(&cfg());
        let (p, m) = reg.resolve("claude-sonnet-4-6").unwrap();
        assert_eq!(p.name(), "anthropic");
        assert_eq!(m, "claude-sonnet-4-6");
    }

    #[test]
    fn resolves_provider_prefix() {
        let reg = ProviderRegistry::from_config(&cfg());
        let (p, m) = reg.resolve("ollama:llama3.2").unwrap();
        assert_eq!(p.name(), "ollama");
        assert_eq!(m, "llama3.2");
    }

    #[test]
    fn resolves_alias() {
        let reg = ProviderRegistry::from_config(&cfg());
        let (p, m) = reg.resolve("gpt-5").unwrap();
        assert_eq!(p.name(), "openai");
        assert_eq!(m, "gpt-5");
    }

    #[test]
    fn falls_back_to_default_provider() {
        let reg = ProviderRegistry::from_config(&cfg());
        let (p, _) = reg.resolve("some-obscure-model").unwrap();
        assert_eq!(p.name(), "openai");
    }

    #[test]
    fn unknown_model_with_no_default_returns_actionable_error() {
        let mut c = cfg();
        c.routing.default_provider = None;
        let reg = ProviderRegistry::from_config(&c);
        match reg.resolve("totally-unknown-model") {
            Ok(_) => panic!("expected UnknownModel error, got Ok"),
            Err(ProviderError::UnknownModel {
                model,
                known_providers,
            }) => {
                assert_eq!(model, "totally-unknown-model");
                assert!(known_providers.contains("openai"));
                assert!(known_providers.contains("anthropic"));
                assert!(known_providers.contains("ollama"));
            }
            Err(other) => panic!("expected UnknownModel, got {:?}", other),
        }
    }
}
