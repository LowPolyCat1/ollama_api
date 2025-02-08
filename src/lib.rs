#![allow(unused)]

#[cfg(test)]
pub mod tests;

use reqwest::{self, IntoUrl, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),

    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),
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
    // pub images:          // For example, base64 encoded images could be added here in the future.
    pub format: String,
    // pub options:         // Additional options could be added here later.
    pub system: String,
    pub stream: bool,
    pub raw: bool,
    // pub keep_alive: u16, // not sure how this is supposed to be formatted
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

#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OllamaResponse {
    pub total_duration: u64,
    pub load_duration: u64,
    pub prompt_eval_count: u8, // might change this to u16, u32, or u64 if error
    pub prompt_eval_duration: u64,
    pub eval_count: u16, // might change this to u32 or u64 if error
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
            // Here we attempt to parse the inner JSON string if possible.
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

impl Ollama {
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

    pub fn default() -> Result<Ollama, OllamaError> {
        Ollama::new("http://localhost:11434", "llama3.2")
    }

    pub async fn generate(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<OllamaResponse, OllamaError> {
        let request = OllamaRequest::new(
            self.model.as_str(),
            prompt,
            "",
            "json",
            self.system.as_str(),
            false,
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

    pub fn stream() {
        unimplemented!();
    }
}
