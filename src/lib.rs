#![allow(unused)]

#[cfg(test)]
pub mod tests;

use reqwest::{self, Error, IntoUrl, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use tokio::stream;

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
    // pub images: base64 encoded images
    pub format: String,
    // pub options: Modelfile implementation
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
    pub prompt_eval_count: u8, // might make this u16, u32, or u64 later
    pub prompt_eval_duration: u64,
    pub eval_count: u16, // might make this u32 or u64 later
    pub eval_duration: u64,
    pub context: Vec<u64>,
    pub response: String,
}

impl From<Value> for OllamaResponse {
    fn from(v: Value) -> Self {
        Self {
            total_duration: v["total_duration"].as_u64().unwrap_or_default(),
            load_duration: v["load_duration"].as_u64().unwrap_or_default(),
            prompt_eval_count: v["prompt_eval_count"].as_u64().unwrap_or_default() as u8,
            prompt_eval_duration: v["prompt_eval_duration"].as_u64().unwrap_or_default(),
            eval_count: v["eval_count"].as_u64().unwrap_or_default() as u16,
            eval_duration: v["eval_duration"].as_u64().unwrap_or_default(),
            context: v["context"]
                .as_array()
                .map(|arr| arr.iter().filter_map(Value::as_u64).collect())
                .unwrap_or_default(),
            // this is necessary because we want the string literal not in JSON format
            response: {
                let raw = v["response"].as_str().unwrap_or_default();
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
    pub fn new(url: impl IntoUrl, model: impl Into<String>) -> Result<Ollama, Error> {
        let client: reqwest::Client = reqwest::Client::new();
        let url = url.into_url();
        let url = match url {
            Ok(url) => url,
            Err(error) => return Err(error),
        };
        Ok(Ollama {
            url,
            model: model.into(),
            client,
            context: vec![],
            system: "".into(),
        })
    }

    pub fn default() -> Result<Ollama, Error> {
        Ollama::new("http://localhost:11434", "llama3.2")
    }

    pub async fn generate(&mut self, prompt: impl Into<String>) -> OllamaResponse {
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

        let request = serde_json::to_string(&request).unwrap();
        let res = self
            .client
            .post(self.url.as_str())
            .body(request)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        let response = OllamaResponse::try_from(res).unwrap();
        self.context = response.context.clone();

        response
    }
}
