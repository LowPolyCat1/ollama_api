use futures_util::pin_mut;
use futures_util::stream::StreamExt;
use rusty_ollama::{Ollama, OllamaError};

#[tokio::main]
async fn main() -> Result<(), OllamaError> {
    // Create an instance of Ollama using its default settings.
    let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4").unwrap();

    // Start a streaming request with a sample prompt.
    let stream = ollama.stream_generate("Why is the sky blue?").await?;
    // Pin the stream so that it can be used with the `next` method.
    pin_mut!(stream);

    // Iterate over the stream, printing each streamed response.
    while let Some(item) = stream.next().await {
        match item {
            Ok(response) => {
                print!("{}", response.response);
                if response.done {
                    break;
                }
            }
            Err(err) => {
                eprintln!("\nError while streaming: {}", err);
                break;
            }
        }
    }

    Ok(())
}
