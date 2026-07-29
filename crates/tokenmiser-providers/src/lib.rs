//! Upstream provider clients.
//!
//! Every provider speaks a different wire shape; we normalize to the
//! OpenAI `chat/completions` shape both inbound and outbound. The trait
//! `Provider` is the single seam — proxy code never branches on provider
//! kind, it just calls `complete()`.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenmiser_config::ProviderConfig;

pub mod anthropic;
pub mod ollama;
pub mod openai;
pub mod registry;

pub use registry::ProviderRegistry;

/// Canonical (OpenAI-shaped) chat completion request that flows through the gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Pass-through for anything we don't model yet (tools, response_format, …).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: serde_json::Value,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Canonical (OpenAI-shaped) chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    #[serde(default = "default_object")]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    /// Anything the upstream returned that we didn't model yet.
    #[serde(flatten, default)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

fn default_object() -> String {
    "chat.completion".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Provider-emitted token logprobs. Used by the cascade Tier 2 router
    /// to score cheap-model confidence. Passed through opaque to clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// True if the response's visible `message.content` is empty/whitespace.
/// Reasoning-mode models (gemma4, deepseek-r1, gpt-oss thinking variants)
/// can emit all tokens into a `reasoning` field and leave content blank —
/// from the caller's perspective that's a broken response, even if the
/// model "succeeded" technically.
pub fn response_visible_content_empty(resp: &ChatResponse) -> bool {
    let Some(choice) = resp.choices.first() else {
        return true;
    };
    match &choice.message.content {
        serde_json::Value::String(s) => s.trim().is_empty(),
        serde_json::Value::Null => true,
        serde_json::Value::Array(arr) => !arr.iter().any(|item| {
            item.get("text")
                .and_then(|t| t.as_str())
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false)
        }),
        _ => false,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("provider {name} not registered")]
    NotFound { name: String },
    #[error("unknown model `{model}`: no alias, no `provider:model` prefix, no family heuristic match. Known providers: {known_providers}. Hint: use a `provider:model` prefix (e.g. `ollama:llama3.2`) or set `routing.default_provider` in config.")]
    UnknownModel {
        model: String,
        known_providers: String,
    },
    #[error("missing api key env var: {0}")]
    MissingApiKey(String),
    #[error("upstream http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("upstream returned status {status}: {body}")]
    Upstream { status: u16, body: String },
    #[error("response parse error: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("provider returned malformed response: {0}")]
    Malformed(String),
}

/// A single SSE event from an upstream provider. We pass these through to
/// the client as-is for v0.7; cross-provider normalization (Anthropic
/// content_block_delta → OpenAI delta.content) lands in a v0.7.1 iteration.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Raw SSE bytes (typically `data: {...}\n\n`).
    Sse(bytes::Bytes),
    /// End-of-stream marker; sent after the upstream closes naturally.
    Done,
}

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    fn config(&self) -> &ProviderConfig;
    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError>;

    /// Streaming variant. Default impl falls back to non-streaming `complete()`
    /// and emits the full response as one SSE chunk — providers that natively
    /// stream override this. Returns a boxed stream of chunks.
    async fn stream(
        &self,
        req: &ChatRequest,
    ) -> Result<
        futures::stream::BoxStream<'static, Result<StreamChunk, ProviderError>>,
        ProviderError,
    > {
        // Fallback: call complete(), wrap the response as one SSE chunk.
        let resp = self.complete(req).await?;
        let json = serde_json::to_vec(&resp)?;
        let mut sse = b"data: ".to_vec();
        sse.extend_from_slice(&json);
        sse.extend_from_slice(b"\n\n");
        let chunk = StreamChunk::Sse(bytes::Bytes::from(sse));
        let done = StreamChunk::Sse(bytes::Bytes::from_static(b"data: [DONE]\n\n"));
        use futures::stream::StreamExt;
        let s = futures::stream::iter(vec![Ok(chunk), Ok(done), Ok(StreamChunk::Done)]);
        Ok(s.boxed())
    }
}
