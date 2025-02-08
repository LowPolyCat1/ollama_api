#![allow(unused)]

#[cfg(test)]
pub mod tests;

use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use reqwest::{self, IntoUrl, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::pin::Pin;
use thiserror::Error;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

// Our custom error type.
#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// Implement conversion from LinesCodecError to OllamaError.
impl From<tokio_util::codec::LinesCodecError> for OllamaError {
    fn from(err: tokio_util::codec::LinesCodecError) -> Self {
        Self::Io(std::io::Error::new(std::io::ErrorKind::Other, err))
    }
}

#[derive(Clone)]
pub struct Ollama {
    pub url: Url,
    pub model: String,
    pub client: reqwest::Client,
    pub context: Vec<u64>,
    pub system: String,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Format {
    #[default]
    Json,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OllamaRequest {
    pub model: String,
    pub prompt: String,
    pub suffix: String,
    pub format: String,
    pub system: String,
    pub stream: bool,
    pub raw: bool,
    pub context: Vec<u64>,
}

impl OllamaRequest {
    pub fn new(
        model: impl Into<String>,
        prompt: impl Into<String>,
        suffix: impl Into<String>,
        format: impl Into<String>,
        system: impl Into<String>,
        stream: bool,
        raw: bool,
        context: Vec<u64>,
    ) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            suffix: suffix.into(),
            format: format.into(),
            system: system.into(),
            stream,
            raw,
            context,
        }
    }
}

impl Default for OllamaRequest {
    fn default() -> Self {
        Self::new("llama3.2", "", "", "json", "", false, false, vec![])
    }
}

/// Non-streaming response
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OllamaResponse {
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: u8,
    pub prompt_eval_duration: u64,
    pub eval_count: u16,
    pub eval_duration: u64,
    pub context: Vec<u64>,
    pub response: String,
}

impl From<Value> for OllamaResponse {
    fn from(value: Value) -> Self {
        Self {
            total_duration: value["total_duration"].as_u64().unwrap_or_default(),
            load_duration: value["load_duration"].as_u64().unwrap_or_default(),
            prompt_eval_count: value["prompt_eval_count"].as_u64().unwrap_or_default() as u8,
            prompt_eval_duration: value["prompt_eval_duration"].as_u64().unwrap_or_default(),
            eval_count: value["eval_count"].as_u64().unwrap_or_default() as u16,
            eval_duration: value["eval_duration"].as_u64().unwrap_or_default(),
            context: value["context"]
                .as_array()
                .map(|arr| arr.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default(),
            response: {
                let raw = value["response"].as_str().unwrap_or_default();
                match serde_json::from_str::<Value>(raw) {
                    Ok(inner_val) => inner_val["response"].as_str().unwrap_or(raw).to_string(),
                    Err(_) => raw.to_string(),
                }
            },
        }
    }
}

impl TryFrom<&str> for OllamaResponse {
    type Error = serde_json::Error;

    fn try_from(json_str: &str) -> Result<Self, Self::Error> {
        serde_json::from_str::<Value>(json_str).map(OllamaResponse::from)
    }
}

impl TryFrom<String> for OllamaResponse {
    type Error = serde_json::Error;

    fn try_from(json_str: String) -> Result<Self, Self::Error> {
        serde_json::from_str::<Value>(&json_str).map(OllamaResponse::from)
    }
}

/// Streaming response type
#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaStreamResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub context: Option<Vec<u64>>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u8>,
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default)]
    pub eval_count: Option<u16>,
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl Ollama {
    /// Creates a new instance with the given URL and model.
    pub fn new(url: impl IntoUrl, model: impl Into<String>) -> Result<Ollama, OllamaError> {
        let client = reqwest::Client::new();
        let url = url.into_url()?;
        Ok(Ollama {
            url,
            model: model.into(),
            client,
            context: vec![],
            system: "".into(),
        })
    }

    /// Creates a default instance.
    pub fn default() -> Result<Ollama, OllamaError> {
        Ollama::new("http://localhost:11434", "llama3.2")
    }

    /// Sends a non-streaming request and returns the complete response.
    pub async fn generate(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<OllamaResponse, OllamaError> {
        let request = OllamaRequest::new(
            self.model.as_str(),
            prompt,
            "",
            "",
            self.system.as_str(),
            false, // not streaming
            false,
            self.context.clone(),
        );

        let request_json = serde_json::to_string(&request)?;
        let res = self
            .client
            .post(self.url.as_str())
            .body(request_json)
            .send()
            .await?;
        let res_text = res.text().await?;
        let response = OllamaResponse::try_from(res_text)?;
        self.context = response.context.clone();
        Ok(response)
    }

    /// Sends a streaming request and returns a stream of `OllamaStreamResponse`.
    pub async fn stream_generate(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<impl Stream<Item = Result<OllamaStreamResponse, OllamaError>>, OllamaError> {
        let request = OllamaRequest::new(
            self.model.as_str(),
            prompt,
            "",
            "",
            self.system.as_str(),
            true, // streaming enabled
            false,
            self.context.clone(),
        );

        let request_json = serde_json::to_string(&request)?;
        let res = self
            .client
            .post(self.url.as_str())
            .body(request_json)
            .send()
            .await?;

        // Convert the response body into a stream of bytes.
        let byte_stream = res
            .bytes_stream()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
        let stream_reader = StreamReader::new(byte_stream);

        // Use LinesCodec to split the stream into lines.
        let lines = FramedRead::new(stream_reader, LinesCodec::new());

        // Parse each nonempty line as an OllamaStreamResponse.
        let parsed = lines.filter_map(|line_result| async move {
            match line_result {
                Ok(line) if !line.trim().is_empty() => {
                    Some(serde_json::from_str::<OllamaStreamResponse>(&line).map_err(Into::into))
                }
                Ok(_) => None,
                Err(e) => Some(Err(e.into())),
            }
        });

        Ok(parsed)
    }
}
