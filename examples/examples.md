
# Rusty Ollama Examples

Examples demonstrating different usage patterns for the Rusty Ollama library.

## Table of Contents

1. [Blocking Generation](#blocking-generation)
2. [Streaming Paragraphs](#streaming-paragraphs)
3. [Streaming Words](#streaming-words)

---

## Blocking Generation

`generate.rs` - Simple blocking generation example

```rust
use rusty_ollama::Ollama;
use rusty_ollama::OllamaError;

#[tokio::main]
async fn main() -> Result<(), OllamaError> {
    let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4")?;
    let res = ollama.generate("Why is the sky blue?").await?;
    println!("{}", res.response);
    Ok(())
}
```

**Usage:**

```bash
cargo run --example generate
```

**Features:**

- Simple blocking call
- Full response handling
- Best for short prompts

---

## Streaming Paragraphs

`stream_paragraphs.rs` - Stream responses in paragraph chunks

```rust
use futures_util::pin_mut;
use futures_util::stream::StreamExt;
use rusty_ollama::{Ollama, OllamaError};

#[tokio::main]
async fn main() -> Result<(), OllamaError> {
    let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4").unwrap();
    let stream = ollama.stream_generate("Why is the sky blue?").await?;
    pin_mut!(stream);

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
```

**Usage:**

```bash
cargo run --example stream_paragraphs
```

**Features:**

- Stream processing
- Paragraph-level updates
- Early termination detection
- Error handling mid-stream

---

## Streaming Words

`stream_words.rs` - Real-time word streaming

```rust
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
                print!("{}", word);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Err(err) => {
                eprintln!("\nError while streaming: {}", err);
                break;
            }
        }
    }
    Ok(())
}
```

**Usage:**

```bash
cargo run --example stream_words
```

**Features:**

- Character/word-level streaming
- Immediate stdout flushing
- Real-time display
- Low-latency output

---

## Prerequisites

1. Ollama server running locally:

   ```bash
   ollama serve
   ```

2. Required model pulled:

   ```bash
   ollama pull phi4
   ```

3. Rust toolchain installed:

   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

## Building Examples

```bash
cargo build --examples
```

## Notes

- Replace "phi4" with your preferred model name
- Adjust endpoint URL if using remote Ollama instance
- Add `#[allow(unused)]` attributes as needed
- Handle errors according to your application's needs

For more details, see the [Rusty Ollama Documentation](https://github.com/lowpolycat1/rusty_ollama).

````

This markdown file includes:
1. Clear section headers
2. Syntax-highlighted code blocks
3. Usage instructions
4. Feature highlights
5. Prerequisites
6. Build instructions
7. Helpful notes

Would you like me to add any additional sections or modify the existing content?
