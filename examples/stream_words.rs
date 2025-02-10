use futures_util::pin_mut;
use futures_util::stream::StreamExt;
use rusty_ollama::{Ollama, OllamaError};

#[tokio::main]
async fn main() -> Result<(), OllamaError> {
    let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4")?;

    let stream = ollama.stream_generate("Why is the sky blue?").await?;
    pin_mut!(stream);

    while let Some(item) = stream.next().await {
        match item {
            Ok(response) => {
                let word = response.response;
                print!("{}", word); // Print each word
                std::io::Write::flush(&mut std::io::stdout()).unwrap(); // Flush stdout immediately
            }
            Err(err) => {
                eprintln!("\nError while streaming: {}", err);
                break;
            }
        }
    }

    Ok(())
}
