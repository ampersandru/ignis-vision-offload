//! The user guide's generated reference stays the code's (ADR 0066, GitHub
//! #347): every page `ignis_server::user_docs` renders equals the file
//! committed under `docs/user/reference/`, so a field, a profile value, a
//! catalog pin, a metric or a route changed without `make docs-reference`
//! turns the suite red, naming the page. Under `IGNIS_BLESS_USER_DOCS=1` the
//! test writes the pages instead, which is all `make docs-reference` runs.

use std::path::PathBuf;

fn reference_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/user/reference")
}

#[test]
fn the_generated_reference_is_the_codes() {
    let dir = reference_dir();
    let bless = std::env::var("IGNIS_BLESS_USER_DOCS").is_ok_and(|v| v == "1");
    let mut stale = Vec::new();
    for page in ignis_server::user_docs::pages() {
        let path = dir.join(page.file);
        if bless {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(&path, &page.text).unwrap();
            continue;
        }
        // A checkout that turns LF into CRLF still holds the same page.
        let committed = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
        if committed != page.text {
            stale.push(page.file);
        }
    }
    assert!(
        stale.is_empty(),
        "docs/user/reference/ is not what the code generates: {}. Run `make docs-reference` and commit the pages.",
        stale.join(", ")
    );
}
