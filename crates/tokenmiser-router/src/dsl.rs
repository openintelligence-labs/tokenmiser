//! Rhai-scripted routing policy (architecture §11.5).
//!
//! Users author `policy.rhai` files that map request shape → routing
//! target. A minimal script looks like:
//!
//! ```rhai
//! // Route extremely long prompts to the frontier; everything else to
//! // the cheapest local model.
//! fn route(req) {
//!     if req.word_count > 500 {
//!         return #{ provider: "anthropic", model: "claude-opus-4-7" };
//!     }
//!     if req.has_keyword("refactor") || req.has_keyword("debug") {
//!         return #{ provider: "anthropic", model: "claude-haiku-4-5" };
//!     }
//!     #{ provider: "ollama", model: "ollama:qwen2.5:7b" }
//! }
//! ```
//!
//! The script gets a single argument: a `RequestView` exposing
//! `word_count`, `model`, `tenant`, `has_keyword(s)`, and the user-message
//! text via `prompt()`. Hot-reload is built in (`PolicyEngine::reload`).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use parking_lot::RwLock;
use rhai::{Dynamic, Engine, Map, Scope, AST};
use serde::{Deserialize, Serialize};
use tokenmiser_providers::ChatRequest;

use crate::policy::RoutingTarget;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestView {
    pub model: String,
    pub tenant: String,
    pub word_count: i64,
    pub prompt: String,
}

impl RequestView {
    pub fn from(req: &ChatRequest, tenant: &str) -> Self {
        let prompt: String = req
            .messages
            .iter()
            .filter(|m| m.role == "user")
            .filter_map(|m| match &m.content {
                serde_json::Value::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        let word_count = prompt.split_whitespace().count() as i64;
        Self {
            model: req.model.clone(),
            tenant: tenant.to_string(),
            word_count,
            prompt,
        }
    }

    fn into_map(self) -> Map {
        let mut m = Map::new();
        m.insert("model".into(), Dynamic::from(self.model));
        m.insert("tenant".into(), Dynamic::from(self.tenant));
        m.insert("word_count".into(), Dynamic::from(self.word_count));
        m.insert("prompt".into(), Dynamic::from(self.prompt));
        m
    }
}

pub struct PolicyEngine {
    engine: Engine,
    ast: RwLock<Arc<AST>>,
    source: RwLock<PathBuf>,
}

impl PolicyEngine {
    pub fn load(path: PathBuf) -> Result<Arc<Self>> {
        let mut engine = Engine::new();
        engine.set_max_expr_depths(64, 64);

        // Register `has_keyword` as a free function so scripts can call
        // `req.has_keyword("refactor")` ergonomically.
        engine.register_fn("has_keyword", |req: Map, kw: &str| -> bool {
            req.get("prompt")
                .and_then(|d| d.clone().into_string().ok())
                .map(|s| s.to_lowercase().contains(&kw.to_lowercase()))
                .unwrap_or(false)
        });

        let src = std::fs::read_to_string(&path)
            .with_context(|| format!("read policy {}", path.display()))?;
        let ast = engine
            .compile(&src)
            .map_err(|e| anyhow!("policy compile: {e}"))?;

        Ok(Arc::new(Self {
            engine,
            ast: RwLock::new(Arc::new(ast)),
            source: RwLock::new(path),
        }))
    }

    pub fn reload(&self) -> Result<()> {
        let path = self.source.read().clone();
        let src = std::fs::read_to_string(&path)
            .with_context(|| format!("reload policy {}", path.display()))?;
        let ast = self
            .engine
            .compile(&src)
            .map_err(|e| anyhow!("policy recompile: {e}"))?;
        *self.ast.write() = Arc::new(ast);
        Ok(())
    }

    pub fn route(&self, req: &ChatRequest, tenant: &str) -> Result<RoutingTarget> {
        let view = RequestView::from(req, tenant);
        let mut scope = Scope::new();
        let ast = self.ast.read().clone();
        let result: Map = self
            .engine
            .call_fn(&mut scope, &ast, "route", (view.into_map(),))
            .map_err(|e| anyhow!("route() call failed: {e}"))?;

        let provider = result
            .get("provider")
            .and_then(|v| v.clone().into_string().ok())
            .ok_or_else(|| anyhow!("route() result missing `provider`"))?;
        let model = result
            .get("model")
            .and_then(|v| v.clone().into_string().ok())
            .ok_or_else(|| anyhow!("route() result missing `model`"))?;

        Ok(RoutingTarget { provider, model })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tokenmiser_providers::ChatMessage;

    fn req(text: &str) -> ChatRequest {
        ChatRequest {
            model: "auto".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: serde_json::Value::String(text.into()),
                extra: Default::default(),
            }],
            temperature: None,
            max_tokens: None,
            top_p: None,
            stream: None,
            extra: Default::default(),
        }
    }

    fn write_policy(src: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("tokenmiser-policy-{}.rhai", rand_suffix()));
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(src.as_bytes()).unwrap();
        path
    }

    fn rand_suffix() -> String {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            .to_string()
    }

    #[test]
    fn keyword_rule_routes_to_frontier() {
        let path = write_policy(
            r#"
            fn route(req) {
                if req.has_keyword("refactor") {
                    return #{ provider: "anthropic", model: "claude-opus-4-7" };
                }
                #{ provider: "ollama", model: "ollama:qwen2.5:7b" }
            }
        "#,
        );
        let p = PolicyEngine::load(path.clone()).unwrap();
        let t = p.route(&req("refactor this code"), "t1").unwrap();
        assert_eq!(t.model, "claude-opus-4-7");
        let t2 = p.route(&req("what is 2+2"), "t1").unwrap();
        assert_eq!(t2.model, "ollama:qwen2.5:7b");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn word_count_rule_works() {
        let path = write_policy(
            r#"
            fn route(req) {
                if req.word_count > 50 {
                    return #{ provider: "anthropic", model: "claude-sonnet-4-6" };
                }
                #{ provider: "ollama", model: "ollama:llama2:latest" }
            }
        "#,
        );
        let p = PolicyEngine::load(path.clone()).unwrap();
        let long = "word ".repeat(100);
        assert_eq!(
            p.route(&req(&long), "t").unwrap().model,
            "claude-sonnet-4-6"
        );
        assert_eq!(
            p.route(&req("hi"), "t").unwrap().model,
            "ollama:llama2:latest"
        );
        let _ = std::fs::remove_file(path);
    }
}
