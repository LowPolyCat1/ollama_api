use rusty_ollama::Ollama;

#[tokio::main]
async fn main() {
    let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4").unwrap();

    let res = ollama.generate("Why is the sky blue?").await.unwrap(); // We are using unwrap for demonstration purposes, you should never use unwrap in production

    println!("{}", res.response);
}
