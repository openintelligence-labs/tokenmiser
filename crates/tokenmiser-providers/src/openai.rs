//! OpenAI client (also covers DeepSeek, Cerebras, DeepInfra — anything that
//! speaks OpenAI's `/v1/chat/completions` wire shape).

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use tokenmiser_config::ProviderConfig;

use crate::{ChatRequest, ChatResponse, Provider, ProviderError, StreamChunk};

pub struct OpenAIProvider {
    cfg: ProviderConfig,
    client: reqwest::Client,
    api_key: Option<String>,
}

impl OpenAIProvider {
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
        if let Some(key) = &self.api_key {
            let val = HeaderValue::from_str(&format!("Bearer {}", key))
                .map_err(|e| ProviderError::Malformed(format!("invalid api key: {e}")))?;
            h.insert(AUTHORIZATION, val);
        }
        Ok(h)
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        &self.cfg.name
    }

    fn config(&self) -> &ProviderConfig {
        &self.cfg
    }

    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        if self.cfg.api_key_env.is_some() && self.api_key.is_none() {
            return Err(ProviderError::MissingApiKey(
                self.cfg.api_key_env.clone().unwrap_or_default(),
            ));
        }

        let url = format!(
            "{}/chat/completions",
            self.cfg.base_url.trim_end_matches('/')
        );
        let mut body = req.clone();
        body.stream = Some(false); // v0.1: non-streaming only

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

        let parsed: ChatResponse = serde_json::from_str(&text)?;
        Ok(parsed)
    }

    async fn stream(
        &self,
        req: &ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamChunk, ProviderError>>, ProviderError> {
        if self.cfg.api_key_env.is_some() && self.api_key.is_none() {
            return Err(ProviderError::MissingApiKey(
                self.cfg.api_key_env.clone().unwrap_or_default(),
            ));
        }

        let url = format!(
            "{}/chat/completions",
            self.cfg.base_url.trim_end_matches('/')
        );
        let mut body = req.clone();
        body.stream = Some(true);

        let res = self
            .client
            .post(&url)
            .headers(self.headers()?)
            .json(&body)
            .send()
            .await?;

        let status = res.status();
        if !status.is_success() {
            let text = res.text().await?;
            return Err(ProviderError::Upstream {
                status: status.as_u16(),
                body: text,
            });
        }

        // OpenAI/Ollama send SSE on this endpoint. Pass through raw bytes;
        // proxy-side normalization (cross-provider tool-call diffs, usage
        // synthesis) is a v0.7.1 follow-up.
        let stream = res
            .bytes_stream()
            .map(|chunk_res| match chunk_res {
                Ok(b) => Ok(StreamChunk::Sse(Bytes::from(b.to_vec()))),
                Err(e) => Err(ProviderError::Http(e)),
            })
            .chain(futures::stream::once(async { Ok(StreamChunk::Done) }));

        Ok(stream.boxed())
    }
}
