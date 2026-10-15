//! Re-tokenizing a repaired reasoning at a rewind's cut (ADR 0052): history
//! repair gives the model back the *text* it saw, the kept reasoning then the
//! nudge. Does the real tokenizer split that text into the tokens the
//! continuation's prompt carried — the kept tokens, then the nudge's own? A
//! measurement, not a gate (CPU only).
//!
//! ```text
//! cargo run --release -p ignis-server --example loop_retokenize -- ARTIFACT FILE...
//! ```
//!
//! Each `FILE` is a chat completion body (its reasoning is
//! `choices[0].message.reasoning_content`) or plain text. The generated
//! tokens are stood in for by the tokenizer's own split of the reasoning up
//! to its first rewind tag (the struck text included, as it was generated).
//! Cuts: the real one where a file has a tag (the start of its struck text),
//! and every end of a line-break run after the first 1,000 characters, as
//! the guard's cut snaps to one — at most `MAX_CUTS` per file, evenly spaced.
//! At each cut, for each sentence a continuation ends its prompt with (the
//! two nudges, the close): `kept tokens ++ encode(sentence)` against
//! `encode(decode(kept tokens) + sentence)`, and where they first differ,
//! relative to the cut (-1: the last kept token). Each sentence twice: as
//! written (the guard before 2026-10-15) and as
//! `loop_guard::sentence_after` joins it to the kept text (the guard now:
//! without its leading line breaks after a kept text that ends in one).

use std::collections::BTreeMap;
use std::path::Path;

use ignis_server::artifact_template::ArtifactTemplateProvider;
use ignis_server::loop_guard::{sentence_after, NUDGES, TAG_PREFIX};
use ignis_server::template::TemplateProvider;
use ignis_server::thinking::THINKING_CLOSE_TEXT;
use serde_json::Value;

const MAX_CUTS: usize = 100;

/// The reasoning in `path`, and the character its first rewind's struck
/// text starts at, when it has one: the text before the tag, less its
/// trailing blank, less the tag's `m` (the format before the hash counts
/// that way; a hashed tag's `m` includes a struck trailing line break,
/// which this stand-in leaves out).
fn reasoning(path: &str) -> (String, Option<usize>) {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let text = match serde_json::from_str::<Value>(&raw) {
        Ok(body) => body["choices"][0]["message"]["reasoning_content"].as_str().unwrap_or_default().to_owned(),
        Err(_) => raw,
    };
    let Some(at) = text.find(TAG_PREFIX) else { return (text, None) };
    let m: usize = text[at + TAG_PREFIX.len()..]
        .split_once(" TOKENS / ")
        .and_then(|(_, rest)| rest.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|m| m.parse().ok())
        .unwrap_or(0);
    let body = text[..at].trim_end();
    let cut = body.chars().count().checked_sub(m);
    (text[..at].to_owned(), cut)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: loop_retokenize ARTIFACT FILE...");
        std::process::exit(2);
    }
    let reader = ignis_artifact::Reader::open(Path::new(&args[0])).expect("open the artifact");
    let template = ArtifactTemplateProvider::new(ignis_artifact::FrontendSet::from_reader(&reader).expect("its frontend"));
    let encode = |text: &str| template.encode_literal(text).expect("encode");
    let sentences = [("nudge 1", NUDGES[0]), ("nudge 2", NUDGES[1]), ("close", THINKING_CLOSE_TEXT)];
    let rules = [("as written", false), ("joined", true)];

    // (rule, sentence, real cut?) -> (cuts, identical, first divergences relative to the cut)
    let mut tally: BTreeMap<(&str, &str, bool), (usize, usize, BTreeMap<i64, usize>)> = BTreeMap::new();
    let mut prefix_unstable = 0;
    let mut cuts_total = 0;
    let mut examples = Vec::new();
    for path in &args[1..] {
        let (text, real) = reasoning(path);
        if text.chars().count() < 2000 {
            continue;
        }
        let tokens = encode(&text);
        // The reasoning's length in characters after each token, as the
        // guard's view counts it (a character still incomplete waits).
        let mut decoder = template.token_decoder();
        let mut chars = 0;
        let chars_after: Vec<usize> = tokens
            .iter()
            .map(|&t| {
                chars += decoder.push(t).chars().count();
                chars
            })
            .collect();
        let chars_of: Vec<char> = text.chars().collect();
        let mut line_ends: Vec<usize> = (1000..chars_of.len().saturating_sub(1))
            .filter(|&i| chars_of[i - 1] == '\n' && chars_of[i] != '\n')
            .collect();
        if line_ends.len() > MAX_CUTS {
            let step = line_ends.len() as f64 / MAX_CUTS as f64;
            line_ends = (0..MAX_CUTS).map(|k| line_ends[(k as f64 * step) as usize]).collect();
        }
        let cuts = real.map(|cut| (cut, true)).into_iter().chain(line_ends.into_iter().map(|cut| (cut, false)));
        for (cut, is_real) in cuts {
            let kept = chars_after.partition_point(|&c| c <= cut);
            let kept_text = template.render_tokens(&tokens[..kept]);
            cuts_total += 1;
            if encode(&kept_text) != tokens[..kept] {
                prefix_unstable += 1;
            }
            for ((rule, joined), (name, sentence)) in rules.iter().flat_map(|rule| sentences.iter().map(move |s| (rule, s))) {
                let sentence = if *joined { sentence_after(&kept_text, sentence) } else { sentence };
                let view: Vec<_> = tokens[..kept].iter().copied().chain(encode(sentence)).collect();
                let repaired = encode(&format!("{kept_text}{sentence}"));
                let entry = tally.entry((rule, name, is_real)).or_default();
                entry.0 += 1;
                match view.iter().zip(&repaired).position(|(a, b)| a != b).or((view.len() != repaired.len()).then(|| view.len().min(repaired.len()))) {
                    None => entry.1 += 1,
                    Some(at) => {
                        *entry.2.entry(at as i64 - kept as i64).or_default() += 1;
                        if examples.iter().filter(|(of, _)| of == rule).count() < 6 {
                            let show = |ids: &[u32]| ids.iter().map(|&t| format!("{:?}", template.render_tokens(&[t]))).collect::<Vec<_>>().join(" ");
                            let from = at.saturating_sub(2);
                            examples.push((*rule, format!(
                                "{path} cut {cut} ({name}, {rule}): view [{}] vs repaired [{}]",
                                show(&view[from..(at + 3).min(view.len())]),
                                show(&repaired[from..(at + 3).min(repaired.len())])
                            )));
                        }
                    }
                }
            }
        }
    }
    println!("cuts {cuts_total}; kept tokens != encode(kept text): {prefix_unstable}");
    for ((rule, name, real), (cuts, same, diverge)) in &tally {
        let at: Vec<String> = diverge.iter().map(|(at, n)| format!("{at:+}: {n}")).collect();
        println!(
            "{rule:10} {name:8} {:9} cuts {cuts:5}  identical {same:5} ({:.1}%)  first difference at {}",
            if *real { "real cut" } else { "line ends" },
            100.0 * *same as f64 / *cuts as f64,
            if at.is_empty() { "-".to_owned() } else { at.join(", ") }
        );
    }
    for (_, example) in examples {
        println!("{example}");
    }
}
