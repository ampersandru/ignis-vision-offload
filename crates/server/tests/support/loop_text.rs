//! Reasoning text for the loop guard's tests (ADR 0052): prose whose windows
//! score clean and a verbatim loop the guard rewinds.
//!
//! Included by `#[path]` from `src/loop_guard.rs`'s unit tests and from
//! `responses.rs`, so the guard and its surfaces are tested on one text.

#![allow(dead_code)]

/// Plausible reasoning that does not repeat: a running account of numbered,
/// varied sentences, a line break after every fourth. Its windows score
/// 0.45-0.49 normalized, clean with room to spare, as real reasoning's
/// mostly do.
pub fn prose(chars: usize) -> String {
    const WORDS: &[&str] = &[
        "Rome", "founded", "kings", "republic", "senate", "consuls", "tribunes", "plebeians", "wars", "Carthage",
        "Hannibal", "Scipio", "Gracchi", "Marius", "Sulla", "Pompey", "Caesar", "Augustus", "empire", "Nero",
        "Flavians", "Trajan", "Hadrian", "crisis", "Diocletian", "Constantine", "Christianity", "Goths", "Vandals",
        "Odoacer", "popes", "Charlemagne", "communes", "Avignon", "Renaissance", "Borgia", "sack", "Baroque",
        "Bernini", "Napoleon", "unification", "capital", "Fascism", "war", "liberation", "republic", "Jubilee",
        "aqueducts", "Forum", "Colosseum", "Pantheon", "Tiber", "Palatine", "legions", "Latin", "law", "Cicero",
        "Vergil", "Ostia", "plague", "Gregory", "Leo", "Vatican", "Sistine", "Michelangelo", "Raphael", "Trastevere",
        "Garibaldi", "Mazzini", "Cavour", "Mussolini", "Lateran", "Olympics", "Fellini", "Tridentine", "Jesuits",
        "Inquisition", "Galileo", "Campidoglio", "Aventine", "Esquiline", "Quirinal", "Viminal", "Caelian", "Janiculum",
    ];
    let mut out = String::new();
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut n = 0;
    while out.chars().count() < chars {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let len = 6 + (state % 9) as usize;
        out.push_str(&format!("{n}. "));
        for k in 0..len {
            let word = WORDS[((state >> ((k * 5) % 60)) as usize + k * 7 + n) % WORDS.len()];
            out.push_str(word);
            out.push(if k + 1 == len { '.' } else { ' ' });
        }
        out.push(if n % 4 == 3 { '\n' } else { ' ' });
        n += 1;
    }
    out.chars().take(chars).collect()
}

/// A verbatim loop, `chars` long.
pub fn looping(chars: usize) -> String {
    "the \"Dark Ages\" (the \"Dark Ages\", the \"Dark Ages\"), ".chars().cycle().take(chars).collect()
}
