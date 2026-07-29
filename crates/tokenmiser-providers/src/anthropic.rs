//! Anthropic Messages API client. Translates OpenAI-shaped requests/responses
//! at the wire boundary so the rest of the gateway stays OpenAI-canonical.

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde_json::{json, Value};
use tokenmiser_config::ProviderConfig;

use crate::{ChatChoice, ChatMessage, ChatRequest, ChatResponse, Provider, ProviderError, Usage};

const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    cfg: ProviderConfig,
    client: reqwest::Client,
    api_key: Option<String>,
}

impl AnthropicProvider {
    pub fn new(cfg: ProviderConfig) -> Self {
        let api_key = cfg.api_key_env.as_ref().and_then(|k| std::env::var(k).ok());

        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .build()
            .expect("reqwest client construction");

        Self {
            cfg,
            client,
            api_key,
        }
    }

    fn headers(&self) -> Result<HeaderMap, ProviderError> {
        let mut h = HeaderMap::new();
        h.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        h.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        if let Some(k) = &self.api_key {
            let v = HeaderValue::from_str(k)
                .map_err(|e| ProviderError::Malformed(format!("invalid api key: {e}")))?;
            h.insert("x-api-key", v);
        }
        Ok(h)
    }

    /// Translate an OpenAI-shaped ChatRequest into Anthropic's Messages payload.
    fn to_anthropic_body(req: &ChatRequest) -> Value {
        let mut system: Option<String> = None;
        let mut messages: Vec<Value> = Vec::with_capacity(req.messages.len());

        for m in &req.messages {
            match m.role.as_str() {
                "system" => {
                    // Anthropic uses a top-level `system` field, not a message.
                    if let Some(s) = content_to_string(&m.content) {
                        system = Some(match system {
                            Some(prev) => format!("{prev}\n\n{s}"),
                            None => s,
                        });
                    }
                }
                role => {
                    messages.push(json!({
                        "role": role,
                        "content": m.content,
                    }));
                }
            }
        }

        let mut body = json!({
            "model": req.model,
            "messages": messages,
            // Anthropic requires max_tokens; default if absent.
            "max_tokens": req.max_tokens.unwrap_or(4096),
        });

        if let Some(s) = system {
            body["system"] = Value::String(s);
        }
        if let Some(t) = req.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(p) = req.top_p {
            body["top_p"] = json!(p);
        }

        body
    }

    /// Translate Anthropic's response back into the OpenAI shape.
    fn from_anthropic_body(v: Value, requested_model: &str) -> Result<ChatResponse, ProviderError> {
        let id = v
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("anthropic")
            .to_string();
        let model = v
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(requested_model)
            .to_string();

        // Concatenate all `text` content blocks into one assistant message string.
        let mut text = String::new();
        if let Some(blocks) = v.get("content").and_then(Value::as_array) {
            for b in blocks {
                if b.get("type").and_then(Value::as_str) == Some("text") {
                    if let Some(t) = b.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
            }
        }

        let usage = v.get("usage");
        let prompt_tokens = usage
            .and_then(|u| u.get("input_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let completion_tokens = usage
            .and_then(|u| u.get("output_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);

        let finish_reason = v
            .get("stop_reason")
            .and_then(Value::as_str)
            .map(|s| match s {
                "end_turn" => "stop".to_string(),
                "max_tokens" => "length".to_string(),
                other => other.to_string(),
            });

        Ok(ChatResponse {
            id,
            object: "chat.completion".into(),
            created: chrono::Utc::now().timestamp() as u64,
            model,
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Value::String(text),
                    extra: Default::default(),
                },
                finish_reason,
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            extra: Default::default(),
        })
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        &self.cfg.name
    }

    fn config(&self) -> &ProviderConfig {
        &self.cfg
    }

    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        if self.api_key.is_none() {
            return Err(ProviderError::MissingApiKey(
                self.cfg.api_key_env.clone().unwrap_or_default(),
            ));
        }

        let url = format!("{}/messages", self.cfg.base_url.trim_end_matches('/'));
        let body = Self::to_anthropic_body(req);

        let res = self
            .client
            .post(&url)
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await?;

        let status = res.status();
        let text = res.text().await?;

        if !status.is_success() {
            return Err(ProviderError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }

        let v: Value = serde_json::from_str(&text)?;
        Self::from_anthropic_body(v, &req.model)
    }
}

fn content_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(arr) => {
            let mut buf = String::new();
            for item in arr {
                if let Some(t) = item.get("text").and_then(Value::as_str) {
                    buf.push_str(t);
                }
            }
            if buf.is_empty() {
                None
            } else {
                Some(buf)
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_message_moves_to_top_level() {
        let req = ChatRequest {
            model: "claude-sonnet-4-6".into(),
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: Value::String("you are concise".into()),
                    extra: Default::default(),
                },
                ChatMessage {
                    role: "user".into(),
                    content: Value::String("hi".into()),
                    extra: Default::default(),
                },
            ],
            temperature: Some(0.2),
            max_tokens: Some(100),
            top_p: None,
            stream: None,
            extra: Default::default(),
        };
        let body = AnthropicProvider::to_anthropic_body(&req);
        assert_eq!(body["system"], "you are concise");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["max_tokens"], 100);
    }

    #[test]
    fn parses_anthropic_response_to_openai_shape() {
        let body = json!({
            "id": "msg_01",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        });
        let r = AnthropicProvider::from_anthropic_body(body, "claude-sonnet-4-6").unwrap();
        assert_eq!(r.choices[0].message.content, Value::String("hello".into()));
        assert_eq!(r.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(r.usage.prompt_tokens, 5);
        assert_eq!(r.usage.completion_tokens, 2);
        assert_eq!(r.usage.total_tokens, 7);
    }
}
