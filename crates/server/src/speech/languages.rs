//! The 30 languages Qwen3-ASR recognises (spec speech/02 §`language`, Q6,
//! Q51): the model's English names, which are also its `language <Name>`
//! labels and the official assistant prefill that forces one, with the ISO
//! 639 codes an OpenAI client sends. Read from the model card's table
//! (`Qwen/Qwen3-ASR-1.7B-hf`, "Supported Languages").

/// One language: the model's English name and its code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Language {
    pub name: &'static str,
    pub code: &'static str,
}

/// Every language, in the model card's order.
pub const LANGUAGES: [Language; 30] = [
    Language { name: "Chinese", code: "zh" },
    Language { name: "English", code: "en" },
    Language { name: "Cantonese", code: "yue" },
    Language { name: "Arabic", code: "ar" },
    Language { name: "German", code: "de" },
    Language { name: "French", code: "fr" },
    Language { name: "Spanish", code: "es" },
    Language { name: "Portuguese", code: "pt" },
    Language { name: "Indonesian", code: "id" },
    Language { name: "Italian", code: "it" },
    Language { name: "Korean", code: "ko" },
    Language { name: "Russian", code: "ru" },
    Language { name: "Thai", code: "th" },
    Language { name: "Vietnamese", code: "vi" },
    Language { name: "Japanese", code: "ja" },
    Language { name: "Turkish", code: "tr" },
    Language { name: "Hindi", code: "hi" },
    Language { name: "Malay", code: "ms" },
    Language { name: "Dutch", code: "nl" },
    Language { name: "Swedish", code: "sv" },
    Language { name: "Danish", code: "da" },
    Language { name: "Finnish", code: "fi" },
    Language { name: "Polish", code: "pl" },
    Language { name: "Czech", code: "cs" },
    Language { name: "Filipino", code: "fil" },
    Language { name: "Persian", code: "fa" },
    Language { name: "Greek", code: "el" },
    Language { name: "Hungarian", code: "hu" },
    Language { name: "Macedonian", code: "mk" },
    Language { name: "Romanian", code: "ro" },
];

/// The language `spelling` names: its code (`it`), `tl` for Filipino as
/// ISO 639-1 spells it, or its English name (`Italian`), in any case.
pub fn find(spelling: &str) -> Option<Language> {
    let spelling = spelling.trim();
    if spelling.eq_ignore_ascii_case("tl") {
        return LANGUAGES.iter().find(|language| language.code == "fil").copied();
    }
    LANGUAGES
        .iter()
        .find(|language| language.code.eq_ignore_ascii_case(spelling) || language.name.eq_ignore_ascii_case(spelling))
        .copied()
}

/// The code of the language the model labelled its transcript with
/// (`Italian` is `it`); a label outside the table is kept, lower-cased.
pub fn code_of_label(label: &str) -> String {
    find(label).map_or_else(|| label.trim().to_ascii_lowercase(), |language| language.code.to_owned())
}

/// The codes, for a refusal to list.
pub fn codes() -> String {
    LANGUAGES.iter().map(|language| language.code).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The card's thirty, each once, each reachable by code and by name.
    #[test]
    fn the_thirty_languages_are_each_found_by_code_and_name() {
        assert_eq!(LANGUAGES.len(), 30);
        for language in LANGUAGES {
            assert_eq!(find(language.code), Some(language));
            assert_eq!(find(&language.name.to_ascii_uppercase()), Some(language));
        }
        let mut codes: Vec<_> = LANGUAGES.iter().map(|language| language.code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), 30, "no code twice");
    }

    #[test]
    fn filipino_is_found_by_the_two_letter_code_too_and_other_codes_are_not() {
        assert_eq!(find("tl").map(|language| language.name), Some("Filipino"));
        for outside in ["xx", "", "klingon", "uk"] {
            assert_eq!(find(outside), None, "{outside}");
        }
    }

    #[test]
    fn a_label_maps_to_its_code_and_an_unknown_label_is_kept() {
        assert_eq!(code_of_label("Italian"), "it");
        assert_eq!(code_of_label("Klingon"), "klingon");
    }
}
