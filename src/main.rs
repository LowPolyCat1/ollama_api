use rusty_ollama::Ollama;

#[tokio::main]
async fn main() {
    let mut ollama = Ollama::new("http://localhost:11434/api/generate", "phi4").unwrap();

    ollama.system = "I am a pirate and I need to act as such".to_string();

    let res = ollama
        .generate("what is supposed to be 'system' in the ollama api?")
        .await;
    println!("{}", res.response);
}
