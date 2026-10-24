//! The speech catalog entries against the real Hugging Face (spec speech/01
//! AC 22, GitHub #351): the pins in `catalog.yaml` are the repos' files.
//!
//! Needs the network and downloads the smallest entry (about 1 GB), so it is
//! `--ignored`: `cargo test -p ignis-server --test speech_catalog_hf --
//! --ignored --nocapture`. `fetch` refuses a file whose size or SHA-256 is not
//! the pin's, so a pass is the check.

use ignis_server::download::{Catalog, Downloader};

#[tokio::test]
#[ignore = "downloads about 1 GB from huggingface.co"]
async fn the_smallest_speech_entry_downloads_and_verifies_against_its_pins() {
    let entry = Catalog::built_in().entry("qwen3-asr-0.6b").expect("built in");
    let dir = std::env::temp_dir().join(format!("ignis-speech-catalog-hf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let downloader = Downloader::new("https://huggingface.co", None).expect("downloader");
    let decoder = downloader.fetch(entry, &dir).await.expect("every file matches its pin");
    assert_eq!(decoder, dir.join("Qwen3-ASR-0.6B-Q8_0.gguf"));
    for file in &entry.files {
        let len = std::fs::metadata(dir.join(&file.name)).unwrap_or_else(|e| panic!("{}: {e}", file.name)).len();
        assert_eq!(len, file.bytes, "{}", file.name);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
