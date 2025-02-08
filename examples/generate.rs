use rusty_ollama::Ollama;

#[tokio::main]
async fn main() -> Result<(), OllamaError> {
    let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4")?;

    let res = ollama.generate("Why is the sky blue?").await?;

    println!("{}", res.response);
}
