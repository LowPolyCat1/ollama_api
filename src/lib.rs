#![allow(unused)]
#![warn(missing_docs)]

//! A Rust client for the Ollama API
//!
//! This crate provides both synchronous and streaming interfaces for interacting
//! with Ollama's large language models.
//!
//! # Example
//! ```no_run
//! use rusty_ollama::{Ollama, OllamaError};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), OllamaError> {
//!     let mut ollama = Ollama::create_default()?;
//!     let response = ollama.generate("Why is the sky blue?").await?;
//!     println!("{}", response.response);
//!     Ok(())
//! }
//! ```

use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use reqwest::{self, IntoUrl, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use thiserror::Error;
use tokio_util::codec::{FramedRead, LinesCodec};
use tokio_util::io::StreamReader;

/// Error type for Ollama operations
#[derive(Debug, Error)]
pub enum OllamaError {
    /// Wrapper for reqwest HTTP errors
    #[error("Reqwest error: {0}")]
    Reqwest(#[from] reqwest::Error),

    /// JSON serialization/deserialization errors
    #[error("Serde error: {0}")]
    Serde(#[from] serde_json::Error),

    /// I/O related errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<tokio_util::codec::LinesCodecError> for OllamaError {
    fn from(err: tokio_util::codec::LinesCodecError) -> Self {
        Self::Io(std::io::Error::new(std::io::ErrorKind::Other, err))
    }
}

/// Response format options for model output
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Format {
    /// No specific format (default)
    None,
    /// Force JSON output format
    JSON,
}

impl From<Format> for String {
    fn from(val: Format) -> Self {
        match val {
            Format::None => "".into(),
            Format::JSON => "json".into(),
        }
    }
}

/// Main client for interacting with Ollama API
#[derive(Clone)]
pub struct Ollama {
    /// API endpoint URL
    pub url: Url,
    /// Model name to use for generation
    pub model: String,
    /// Internal HTTP client
    pub client: reqwest::Client,
    /// Conversation context for multi-turn dialogs
    pub context: Vec<u64>,
    /// System prompt for model instructions
    pub system: String,
}

/// Configuration options for generation requests
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OllamaRequestOptions {
    /// Suffix to append to the generated response
    pub suffix: String,
    /// Output format specification
    pub format: Format,
    /// System prompt for model instructions
    pub system: String,
    /// Conversation context tokens
    pub context: Vec<u64>,
}

/// Complete request structure for generation
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OllamaRequest {
    /// Model identifier
    pub model: String,
    /// Input prompt text
    pub prompt: String,
    /// Response suffix
    pub suffix: String,
    /// Output format
    pub format: String,
    /// System instructions
    pub system: String,
    /// Stream response flag
    pub stream: bool,
    /// Bypass formatting flag
    pub raw: bool,
    /// Context tokens for conversation history
    pub context: Vec<u64>,
}

impl OllamaRequest {
    /// Create a new OllamaRequest with specified parameters
    ///
    /// # Arguments
    /// * `model` - Model identifier string
    /// * `prompt` - Input prompt text
    /// * `options` - Configuration options
    /// * `stream` - Enable streaming response
    /// * `raw` - Bypass model formatting
    pub fn new(
        model: impl Into<String>,
        prompt: impl Into<String>,
        options: OllamaRequestOptions,
        stream: bool,
        raw: bool,
    ) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            suffix: options.suffix,
            format: options.format.into(),
            system: options.system,
            stream,
            raw,
            context: options.context,
        }
    }
}

impl Default for OllamaRequest {
    /// Create a default request with empty parameters
    fn default() -> Self {
        OllamaRequest::new(
            "llama3.2",
            "",
            OllamaRequestOptions {
                suffix: "".into(),
                format: Format::None,
                system: "".into(),
                context: vec![],
            },
            false,
            false,
        )
    }
}

/// Complete response from generation request
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct OllamaResponse {
    /// Total request duration in nanoseconds
    pub total_duration: u64,
    /// Model loading duration in nanoseconds
    pub load_duration: u64,
    /// Number of tokens in prompt evaluation
    pub prompt_eval_count: u8,
    /// Prompt evaluation duration in nanoseconds
    pub prompt_eval_duration: u64,
    /// Number of tokens generated
    pub eval_count: u16,
    /// Generation duration in nanoseconds
    pub eval_duration: u64,
    /// Context tokens for subsequent requests
    pub context: Vec<u64>,
    /// Generated response text
    pub response: String,
}

impl From<Value> for OllamaResponse {
    /// Convert JSON Value to OllamaResponse
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

    /// Parse response from JSON string slice
    fn try_from(json_str: &str) -> Result<Self, Self::Error> {
        serde_json::from_str::<Value>(json_str).map(OllamaResponse::from)
    }
}

impl TryFrom<String> for OllamaResponse {
    type Error = serde_json::Error;

    /// Parse response from JSON String
    fn try_from(json_str: String) -> Result<Self, Self::Error> {
        serde_json::from_str::<Value>(&json_str).map(OllamaResponse::from)
    }
}

/// Streaming response chunk
#[derive(Serialize, Deserialize, Debug)]
pub struct OllamaStreamResponse {
    /// Model identifier
    pub model: String,
    /// Timestamp of response creation
    pub created_at: String,
    /// Generated text chunk
    pub response: String,
    /// Completion status flag
    pub done: bool,
    /// Completion reason if finished
    #[serde(default)]
    pub done_reason: Option<String>,
    /// Updated context tokens
    #[serde(default)]
    pub context: Option<Vec<u64>>,
    /// Total duration metrics
    #[serde(default)]
    pub total_duration: Option<u64>,
    /// Model loading duration
    #[serde(default)]
    pub load_duration: Option<u64>,
    /// Prompt evaluation metrics
    #[serde(default)]
    pub prompt_eval_count: Option<u8>,
    /// Prompt evaluation duration
    #[serde(default)]
    pub prompt_eval_duration: Option<u64>,
    /// Generation metrics
    #[serde(default)]
    pub eval_count: Option<u16>,
    /// Generation duration
    #[serde(default)]
    pub eval_duration: Option<u64>,
}

impl Ollama {
    /// Create a new Ollama client instance
    ///
    /// # Arguments
    /// * `url` - API endpoint URL
    /// * `model` - Model identifier string
    ///
    /// # Example
    /// ```no_run
    /// use rusty_ollama::{Ollama, OllamaError};
    ///
    /// # fn main() -> Result<(), OllamaError> {
    /// let ollama = Ollama::new("http://localhost:11434", "llama3.2")?;
    /// # Ok(())
    /// # }
    /// ```
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

    /// Create a client with default settings
    ///
    /// Defaults to localhost:11434 and llama3.2 model
    pub fn create_default() -> Result<Ollama, OllamaError> {
        Ollama::new("http://localhost:11434", "llama3.2")
    }

    /// Generate text completion (blocking)
    ///
    /// # Arguments
    /// * `prompt` - Input text prompt
    ///
    /// Updates internal context for subsequent requests
    pub async fn generate(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<OllamaResponse, OllamaError> {
        let request = OllamaRequest::new(
            self.model.as_str(),
            prompt,
            OllamaRequestOptions {
                suffix: "".into(),
                format: Format::None,
                system: self.system.clone(),
                context: self.context.clone(),
            },
            false,
            false,
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

    /// Stream generated text in real-time
    ///
    /// Returns a stream of OllamaStreamResponse chunks
    ///
    /// # Example
    /// ```no_run
    /// use futures_util::pin_mut;
    /// use futures_util::stream::StreamExt;
    /// use rusty_ollama::{Ollama, OllamaError};
    /// #[tokio::main]
    /// async fn main() -> Result<(), OllamaError> {
    ///     let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4")?;
    ///     let stream = ollama.stream_generate("Why is the sky blue?").await?;
    ///     pin_mut!(stream);
    ///     while let Some(item) = stream.next().await {
    ///         match item {
    ///             Ok(response) => {
    ///                 let word = response.response;
    ///                 print!("{}", word); // Print each word with a space
    ///                 std::io::Write::flush(&mut std::io::stdout()).unwrap(); // Flush stdout immediately
    ///             }
    ///             Err(err) => {
    ///                 eprintln!("\nError while streaming: {}", err);
    ///                 break;
    ///             }
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    pub async fn stream_generate(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<impl Stream<Item = Result<OllamaStreamResponse, OllamaError>>, OllamaError> {
        let request = OllamaRequest::new(
            self.model.as_str(),
            prompt,
            OllamaRequestOptions {
                suffix: "".into(),
                format: Format::None,
                system: self.system.clone(),
                context: self.context.clone(),
            },
            true,
            false,
        );

        let request_json = serde_json::to_string(&request)?;
        let res = self
            .client
            .post(self.url.as_str())
            .body(request_json)
            .send()
            .await?;

        let byte_stream = res
            .bytes_stream()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
        let stream_reader = StreamReader::new(byte_stream);

        let lines = FramedRead::new(stream_reader, LinesCodec::new());

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

    pub fn generate_blocking(
        &mut self,
        prompt: impl Into<String>,
    ) -> Result<OllamaResponse, OllamaError> {
        let request = OllamaRequest::new(
            self.model.as_str(),
            prompt,
            OllamaRequestOptions {
                suffix: "".into(),
                format: Format::None,
                system: self.system.clone(),
                context: self.context.clone(),
            },
            false,
            false,
        );

        let request_json = serde_json::to_string(&request)?;

        let response_text = reqwest::blocking::Client::new()
            .post(self.url.as_str())
            .body(request_json)
            .timeout(Duration::from_secs_f64(300.0))
            .send()?
            .text()?;

        let response = OllamaResponse::try_from(response_text)?;
        self.context = response.context.clone();
        Ok(response)
    }
}
