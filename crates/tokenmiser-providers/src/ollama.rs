//! Ollama client — uses the OpenAI-compatible endpoint Ollama exposes on
//! `/v1/chat/completions`, so this is essentially the OpenAI client with no
//! auth header and a localhost base URL.

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::Deserialize;
use tokenmiser_config::ProviderConfig;

use crate::{ChatRequest, ChatResponse, Provider, ProviderError, StreamChunk};

pub struct OllamaProvider {
    cfg: ProviderConfig,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
pub struct OllamaTag {
    pub name: String,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaTag>,
}

impl OllamaProvider {
    pub fn new(cfg: ProviderConfig) -> Self {
        let client = reqwest::Client::builder()
            .pool_max_idle_per_host(32)
            .build()
            .expect("reqwest client construction");
        Self { cfg, client }
    }

    /// Probe localhost:11434 for a running Ollama. Returns the loaded model
    /// names if reachable; used by the daemon at startup for zero-config
    /// local routing (architecture §7).
    pub async fn detect(base_url: &str) -> Result<Vec<String>, ProviderError> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()?;
        let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
        let res = client.get(&url).send().await?;
        if !res.status().is_success() {
            return Err(ProviderError::Upstream {
                status: res.status().as_u16(),
                body: res.text().await.unwrap_or_default(),
            });
        }
        let body: OllamaTagsResponse = res.json().await?;
        Ok(body.models.into_iter().map(|m| m.name).collect())
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        &self.cfg.name
    }

    fn config(&self) -> &ProviderConfig {
        &self.cfg
    }

    async fn complete(&self, req: &ChatRequest) -> Result<ChatResponse, ProviderError> {
        let url = format!(
            "{}/v1/chat/completions",
            self.cfg.base_url.trim_end_matches('/')
        );
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let mut body = req.clone();
        body.stream = Some(false);

        // Ollama prefixes-tolerant: strip `ollama:` if present.
        if let Some(rest) = body.model.strip_prefix("ollama:") {
            body.model = rest.to_string();
        }

        let res = self
            .client
            .post(&url)
            .headers(headers)
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
        let url = format!(
            "{}/v1/chat/completions",
            self.cfg.base_url.trim_end_matches('/')
        );
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let mut body = req.clone();
        body.stream = Some(true);
        if let Some(rest) = body.model.strip_prefix("ollama:") {
            body.model = rest.to_string();
        }

        let res = self
            .client
            .post(&url)
            .headers(headers)
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
