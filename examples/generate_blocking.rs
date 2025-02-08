use rusty_ollama::Ollama;
use rusty_ollama::OllamaError;

fn main() -> Result<(), OllamaError> {
    let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4")?;

    let res = ollama.generate_blocking("Why is the sky blue?")?;

    println!("{}", res.response);
    Ok(())
}
