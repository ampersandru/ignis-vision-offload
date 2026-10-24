//! A Realtime transcription session's configuration (spec speech/03, §Client
//! events): what `session.update` sets, every field honoured, inert or
//! refused as the spec's table says, and the effective configuration that
//! `session.created` and `session.updated` carry.
//!
//! An update names only what it changes: a field it leaves out keeps its
//! value, `null` puts it back to its default. One refused field refuses the
//! whole update, and the session keeps the configuration it had.

use serde_json::{json, Map, Value};

use super::super::executor::SessionOpen;
use super::super::route::{self, FieldRefusal, ModelRefusal};

/// OpenAI's PCM16 rate, the default.
pub(crate) const OPENAI_RATE: u32 = 24_000;
/// The model's own rate: accepted as a declared extension, and not resampled.
pub(crate) const MODEL_RATE: u32 = 16_000;

/// `transcription.delay`'s values. Inert: the cadence is
/// `speech.partial_interval_ms`.
const DELAYS: [&str; 5] = ["minimal", "low", "medium", "high", "xhigh"];

/// What a session runs with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionConfig {
    /// The rate its PCM16 arrives at: [`OPENAI_RATE`] or [`MODEL_RATE`].
    pub rate: u32,
    /// The speech model's served id, whatever name the client sent.
    pub model: String,
    pub prompt: Option<String>,
    pub keywords: Vec<String>,
    pub language: Option<String>,
    pub languages: Vec<String>,
    /// The language `language` or `languages` force, by the model's English
    /// name; `None` detects it.
    pub forced: Option<String>,
    pub delay: Option<String>,
    /// `session.ignis.provisional_tail`: the provisional tail is sent.
    pub provisional_tail: bool,
}

/// An event or a field refused: what its `error` event carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub code: &'static str,
    pub message: String,
    pub param: Option<String>,
}

impl Refusal {
    /// A value outside what the field takes.
    pub fn invalid(message: impl Into<String>, param: &str) -> Self {
        Self { code: "invalid_request_error", message: message.into(), param: Some(param.to_owned()) }
    }

    /// A value OpenAI serves and this server does not.
    pub fn unsupported(message: impl Into<String>, param: &str) -> Self {
        Self { code: "unsupported_value", message: message.into(), param: Some(param.to_owned()) }
    }
}

impl From<FieldRefusal> for Refusal {
    fn from(refused: FieldRefusal) -> Self {
        Self::invalid(refused.message, &refused.param)
    }
}

const TRANSCRIPTION: &str = "session.audio.input.transcription";

impl SessionConfig {
    /// A session as it opens: 24 kHz, the speech model, no context, the
    /// language detected, no provisional tail.
    pub fn new(model: String) -> Self {
        Self {
            rate: OPENAI_RATE,
            model,
            prompt: None,
            keywords: Vec::new(),
            language: None,
            languages: Vec::new(),
            forced: None,
            delay: None,
            provisional_tail: false,
        }
    }

    /// What the executor's session opens with: the context, prefilled when
    /// it opens (Q50), and the language forced.
    pub fn open(&self, context_max_tokens: u32) -> SessionOpen {
        SessionOpen {
            context: route::context_of(self.prompt.as_deref(), &self.keywords),
            context_max_tokens,
            language: self.forced.clone(),
        }
    }

    /// This configuration with `session.update`'s `session` applied, or the
    /// first field refused. `model` decides what a
    /// `transcription.model` reaches (spec speech/02's rule).
    pub fn updated(&self, session: &Value, model: impl Fn(&str) -> Result<String, ModelRefusal>) -> Result<Self, Refusal> {
        let Some(session) = session.as_object() else {
            return Err(Refusal::invalid("session.update needs a `session` object", "session"));
        };
        let mut next = self.clone();
        match session.get("type") {
            None | Some(Value::Null) => {}
            Some(Value::String(kind)) if kind == "transcription" => {}
            Some(Value::String(kind)) if kind == "realtime" => {
                return Err(Refusal::unsupported(
                    "Realtime conversation sessions are not served: this socket transcribes, with session.type `transcription`",
                    "session.type",
                ));
            }
            Some(_) => return Err(Refusal::invalid("session.type must be `transcription`", "session.type")),
        }
        if let Some(input) = object(session, "audio", "session.audio")?.map(|audio| object(audio, "input", "session.audio.input")).transpose()?.flatten() {
            next.format(input.get("format"))?;
            next.transcription(input.get("transcription"), &model)?;
            turn_detection(input.get("turn_detection"))?;
            noise_reduction(input.get("noise_reduction"))?;
        }
        include(session.get("include"))?;
        if let Some(ignis) = object(session, "ignis", "session.ignis")? {
            match ignis.get("provisional_tail") {
                None => {}
                Some(Value::Null) => next.provisional_tail = false,
                Some(Value::Bool(on)) => next.provisional_tail = *on,
                Some(_) => return Err(Refusal::invalid("session.ignis.provisional_tail must be true or false", "session.ignis.provisional_tail")),
            }
        }
        next.forced = route::forced(next.language.as_deref(), &next.languages, &format!("{TRANSCRIPTION}.language"), &format!("{TRANSCRIPTION}.languages"))?;
        Ok(next)
    }

    fn format(&mut self, format: Option<&Value>) -> Result<(), Refusal> {
        const PARAM: &str = "session.audio.input.format";
        let format = match format {
            None => return Ok(()),
            Some(Value::Null) => {
                self.rate = OPENAI_RATE;
                return Ok(());
            }
            Some(Value::Object(format)) => format,
            Some(_) => return Err(Refusal::invalid(format!("{PARAM} must be an object"), PARAM)),
        };
        match format.get("type").and_then(Value::as_str) {
            None | Some("audio/pcm") => {}
            Some(g711 @ ("audio/pcmu" | "audio/pcma")) => {
                return Err(Refusal::unsupported(
                    format!("{PARAM}.type `{g711}` (G.711) is not served: send audio/pcm, 16-bit mono at 24000 Hz"),
                    &format!("{PARAM}.type"),
                ));
            }
            Some(other) => return Err(Refusal::invalid(format!("{PARAM}.type `{other}` is not audio/pcm"), &format!("{PARAM}.type"))),
        }
        self.rate = match format.get("rate") {
            None | Some(Value::Null) => OPENAI_RATE,
            Some(rate) => match rate.as_u64() {
                Some(rate) if rate == u64::from(OPENAI_RATE) || rate == u64::from(MODEL_RATE) => rate as u32,
                _ => {
                    return Err(Refusal::unsupported(
                        format!("{PARAM}.rate {rate} is not served: send 24000, OpenAI's, or 16000, the model's own"),
                        &format!("{PARAM}.rate"),
                    ))
                }
            },
        };
        Ok(())
    }

    fn transcription(&mut self, transcription: Option<&Value>, model: &impl Fn(&str) -> Result<String, ModelRefusal>) -> Result<(), Refusal> {
        let transcription = match transcription {
            None => return Ok(()),
            Some(Value::Null) => {
                let model = std::mem::take(&mut self.model);
                *self = Self { rate: self.rate, provisional_tail: self.provisional_tail, ..Self::new(model) };
                return Ok(());
            }
            Some(Value::Object(transcription)) => transcription,
            Some(_) => return Err(Refusal::invalid(format!("{TRANSCRIPTION} must be an object"), TRANSCRIPTION)),
        };
        match transcription.get("model") {
            None => {}
            Some(Value::Null) => self.model = model("").map_err(|refusal| model_refusal(refusal))?,
            Some(Value::String(named)) => self.model = model(named).map_err(|refusal| model_refusal(refusal))?,
            Some(_) => return Err(Refusal::invalid(format!("{TRANSCRIPTION}.model must be a string"), &format!("{TRANSCRIPTION}.model"))),
        }
        if let Some(prompt) = optional_text(transcription, "prompt")? {
            self.prompt = prompt;
        }
        if let Some(keywords) = optional_list(transcription, "keywords")? {
            let param = format!("{TRANSCRIPTION}.keywords");
            if let Some(refused) = route::refused_keyword(&keywords, &param) {
                return Err(refused.into());
            }
            self.keywords = keywords;
        }
        if let Some(language) = optional_text(transcription, "language")? {
            self.language = language;
        }
        if let Some(languages) = optional_list(transcription, "languages")? {
            self.languages = languages;
        }
        if let Some(delay) = optional_text(transcription, "delay")? {
            if let Some(delay) = delay.as_deref().filter(|delay| !DELAYS.contains(delay)) {
                return Err(Refusal::invalid(
                    format!("{TRANSCRIPTION}.delay `{delay}` is not one of {}", DELAYS.join(", ")),
                    &format!("{TRANSCRIPTION}.delay"),
                ));
            }
            self.delay = delay;
        }
        Ok(())
    }

    /// The `session` object of `session.created` and `session.updated`: the
    /// configuration in effect, `turn_detection: null` declared (phase 1,
    /// Q54).
    pub fn effective(&self, id: &str) -> Value {
        let mut transcription = Map::new();
        transcription.insert("model".to_owned(), json!(self.model));
        transcription.insert("prompt".to_owned(), json!(self.prompt.clone().unwrap_or_default()));
        transcription.insert("language".to_owned(), json!(self.language));
        if !self.keywords.is_empty() {
            transcription.insert("keywords".to_owned(), json!(self.keywords));
        }
        if !self.languages.is_empty() {
            transcription.insert("languages".to_owned(), json!(self.languages));
        }
        if let Some(delay) = &self.delay {
            transcription.insert("delay".to_owned(), json!(delay));
        }
        json!({
            "type": "transcription",
            "object": "realtime.transcription_session",
            "id": id,
            "audio": { "input": {
                "format": { "type": "audio/pcm", "rate": self.rate },
                "transcription": transcription,
                "noise_reduction": null,
                "turn_detection": null,
            } },
            "include": [],
            "ignis": { "provisional_tail": self.provisional_tail },
        })
    }
}

fn model_refusal(refusal: ModelRefusal) -> Refusal {
    let param = Some(format!("{TRANSCRIPTION}.model"));
    match refusal {
        ModelRefusal::KindMismatch(message) => Refusal { code: "model_kind_mismatch", message, param },
        ModelRefusal::NotFound(message) => Refusal { code: "model_not_found", message, param },
    }
}

/// `parent[key]` as an object; `None` when absent or `null`.
fn object<'a>(parent: &'a Map<String, Value>, key: &str, param: &str) -> Result<Option<&'a Map<String, Value>>, Refusal> {
    match parent.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(object)) => Ok(Some(object)),
        Some(_) => Err(Refusal::invalid(format!("{param} must be an object"), param)),
    }
}

/// A text field of the transcription object: `None` when absent, `Some(None)`
/// when `null`.
fn optional_text(transcription: &Map<String, Value>, key: &str) -> Result<Option<Option<String>>, Refusal> {
    match transcription.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(None)),
        Some(Value::String(text)) => Ok(Some(Some(text.clone()))),
        Some(_) => Err(Refusal::invalid(format!("{TRANSCRIPTION}.{key} must be a string"), &format!("{TRANSCRIPTION}.{key}"))),
    }
}

/// A list of strings of the transcription object: `None` when absent, empty
/// when `null`.
fn optional_list(transcription: &Map<String, Value>, key: &str) -> Result<Option<Vec<String>>, Refusal> {
    let refused = || Refusal::invalid(format!("{TRANSCRIPTION}.{key} must be an array of strings"), &format!("{TRANSCRIPTION}.{key}"));
    match transcription.get(key) {
        None => Ok(None),
        Some(Value::Null) => Ok(Some(Vec::new())),
        Some(Value::Array(items)) => items.iter().map(|item| item.as_str().map(str::to_owned).ok_or_else(refused)).collect::<Result<_, _>>().map(Some),
        Some(_) => Err(refused()),
    }
}

/// `turn_detection`: `null` (manual commit) is served. In phase 1 `server_vad`
/// is refused (spec speech/05 serves it from phase 2), and `semantic_vad`
/// always.
fn turn_detection(value: Option<&Value>) -> Result<(), Refusal> {
    const PARAM: &str = "session.audio.input.turn_detection";
    let kind = match value {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Object(detection)) => detection.get("type").and_then(Value::as_str),
        Some(_) => None,
    };
    match kind {
        Some("server_vad") => Err(Refusal::unsupported(
            format!("{PARAM} `server_vad` is not served yet: send null and commit with input_audio_buffer.commit"),
            PARAM,
        )),
        Some("semantic_vad") => Err(Refusal::unsupported(
            format!("{PARAM} `semantic_vad` is not served: send null and commit with input_audio_buffer.commit"),
            PARAM,
        )),
        _ => Err(Refusal::invalid(format!("{PARAM} must be null, or an object whose type is server_vad or semantic_vad"), PARAM)),
    }
}

/// `noise_reduction`: `null` is served; `near_field` and `far_field` are
/// refused.
fn noise_reduction(value: Option<&Value>) -> Result<(), Refusal> {
    const PARAM: &str = "session.audio.input.noise_reduction";
    match value {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(reduction)) => match reduction.get("type").and_then(Value::as_str) {
            Some(kind @ ("near_field" | "far_field")) => {
                Err(Refusal::unsupported(format!("{PARAM} `{kind}` is not served: this server reduces no noise; send null"), PARAM))
            }
            _ => Err(Refusal::invalid(format!("{PARAM}.type must be near_field or far_field"), PARAM)),
        },
        Some(_) => Err(Refusal::invalid(format!("{PARAM} must be null or an object"), PARAM)),
    }
}

/// `include`: empty is served; the transcription's logprobs are refused.
fn include(value: Option<&Value>) -> Result<(), Refusal> {
    const PARAM: &str = "session.include";
    let items = match value {
        None | Some(Value::Null) => return Ok(()),
        Some(Value::Array(items)) => items,
        Some(_) => return Err(Refusal::invalid(format!("{PARAM} must be an array"), PARAM)),
    };
    if items.iter().any(|item| item.as_str() == Some("item.input_audio_transcription.logprobs")) {
        return Err(Refusal::unsupported(
            format!("{PARAM} `item.input_audio_transcription.logprobs` is not served: this server reports no logprobs"),
            PARAM,
        ));
    }
    match items.is_empty() {
        true => Ok(()),
        false => Err(Refusal::invalid(format!("{PARAM} names nothing this server includes"), PARAM)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SERVED: &str = "qwen3-asr-1.7b";

    /// Spec speech/02's rule over a fixed set of names: absent, the served id
    /// or a compat name reach the speech model, a text model's id is a kind
    /// mismatch, anything else not found.
    fn model(named: &str) -> Result<String, ModelRefusal> {
        match named {
            "" | SERVED | "whisper-1" | "gpt-4o-transcribe" => Ok(SERVED.to_owned()),
            "qwen3.8-27b" => Err(ModelRefusal::KindMismatch("names qwen3.8-27b, a text model".to_owned())),
            other => Err(ModelRefusal::NotFound(format!("unknown model: {other}"))),
        }
    }

    fn update(config: &SessionConfig, session: Value) -> Result<SessionConfig, Refusal> {
        config.updated(&session, model)
    }

    fn fresh() -> SessionConfig {
        SessionConfig::new(SERVED.to_owned())
    }

    fn input(input: Value) -> Value {
        json!({ "type": "transcription", "audio": { "input": input } })
    }

    /// AC 2, 15: a session opens at 24 kHz on the speech model, with
    /// `turn_detection: null` declared and no provisional tail.
    #[test]
    fn a_session_opens_at_24_khz_with_manual_commit_declared() {
        let effective = fresh().effective("sess_1");
        assert_eq!(effective["type"], "transcription");
        assert_eq!(effective["id"], "sess_1");
        assert_eq!(effective["audio"]["input"]["format"], json!({ "type": "audio/pcm", "rate": 24000 }));
        assert_eq!(effective["audio"]["input"]["transcription"]["model"], SERVED);
        assert!(effective["audio"]["input"]["turn_detection"].is_null());
        assert!(effective["audio"]["input"].as_object().unwrap().contains_key("turn_detection"));
        assert!(effective["audio"]["input"]["noise_reduction"].is_null());
        assert_eq!(effective["ignis"]["provisional_tail"], false);
    }

    /// AC 2, 3: an update names what it changes; 16 kHz is the declared
    /// extension and the effective configuration reports it.
    #[test]
    fn an_update_changes_what_it_names_and_reports_the_effective_rate() {
        let config = update(&fresh(), input(json!({ "format": { "type": "audio/pcm", "rate": 16000 } }))).unwrap();
        assert_eq!(config.rate, 16000);
        assert_eq!(config.effective("s")["audio"]["input"]["format"]["rate"], 16000);
        let config = update(&config, input(json!({ "transcription": { "prompt": "A talk about GPUs." } }))).unwrap();
        assert_eq!(config.rate, 16000, "a field left out keeps its value");
        assert_eq!(config.prompt.as_deref(), Some("A talk about GPUs."));
        let config = update(&config, input(json!({ "format": null }))).unwrap();
        assert_eq!(config.rate, 24000, "null is the default");
        assert_eq!(config.prompt.as_deref(), Some("A talk about GPUs."));
    }

    /// AC 3: G.711 and any other rate are refused, naming the field.
    #[test]
    fn g711_and_other_rates_are_refused() {
        for (format, param) in [
            (json!({ "type": "audio/pcmu" }), "session.audio.input.format.type"),
            (json!({ "type": "audio/pcma" }), "session.audio.input.format.type"),
            (json!({ "type": "audio/pcm", "rate": 44100 }), "session.audio.input.format.rate"),
        ] {
            let refused = update(&fresh(), input(json!({ "format": format }))).unwrap_err();
            assert_eq!(refused.code, "unsupported_value", "{format}");
            assert_eq!(refused.param.as_deref(), Some(param));
        }
        assert!(update(&fresh(), input(json!({ "format": { "type": "audio/pcmu" } }))).unwrap_err().message.contains("G.711"));
    }

    /// AC 8, 15: `turn_detection: null` is served; in phase 1 `server_vad` is
    /// refused, and `semantic_vad` always.
    #[test]
    fn manual_commit_is_served_and_vad_is_refused() {
        assert!(update(&fresh(), input(json!({ "turn_detection": null }))).is_ok());
        for kind in ["server_vad", "semantic_vad"] {
            let refused = update(&fresh(), input(json!({ "turn_detection": { "type": kind } }))).unwrap_err();
            assert_eq!((refused.code, refused.param.as_deref()), ("unsupported_value", Some("session.audio.input.turn_detection")));
            assert!(refused.message.contains(kind) && refused.message.contains("input_audio_buffer.commit"), "{}", refused.message);
        }
    }

    /// AC 8: the rest of the table: `delay` inert, `noise_reduction` null
    /// served and the two kinds refused, `include` logprobs refused, a
    /// conversation session refused.
    #[test]
    fn every_other_field_is_honoured_inert_or_refused_as_the_table_says() {
        let config = update(&fresh(), input(json!({ "transcription": { "delay": "low" }, "noise_reduction": null }))).unwrap();
        assert_eq!(config.delay.as_deref(), Some("low"));
        assert_eq!(config.open(512), fresh().open(512), "delay changes nothing the executor sees");
        assert_eq!(update(&fresh(), input(json!({ "transcription": { "delay": "soon" } }))).unwrap_err().code, "invalid_request_error");
        for kind in ["near_field", "far_field"] {
            let refused = update(&fresh(), input(json!({ "noise_reduction": { "type": kind } }))).unwrap_err();
            assert_eq!((refused.code, refused.param.as_deref()), ("unsupported_value", Some("session.audio.input.noise_reduction")));
        }
        let refused = update(&fresh(), json!({ "type": "transcription", "include": ["item.input_audio_transcription.logprobs"] })).unwrap_err();
        assert_eq!((refused.code, refused.param.as_deref()), ("unsupported_value", Some("session.include")));
        let behind = update(&fresh(), json!({ "include": ["x", "item.input_audio_transcription.logprobs"] })).unwrap_err();
        assert_eq!(behind.code, "unsupported_value", "the logprobs are refused wherever they are listed");
        assert!(update(&fresh(), json!({ "type": "transcription", "include": [] })).is_ok());
        let refused = update(&fresh(), json!({ "type": "realtime" })).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.type"));
        assert!(refused.message.contains("conversation"), "{}", refused.message);
    }

    /// AC 6: the provisional tail is opt-in under `session.ignis`.
    #[test]
    fn the_provisional_tail_is_opt_in() {
        assert!(!fresh().provisional_tail);
        let on = update(&fresh(), json!({ "type": "transcription", "ignis": { "provisional_tail": true } })).unwrap();
        assert!(on.provisional_tail);
        assert_eq!(on.effective("s")["ignis"]["provisional_tail"], true);
        let refused = update(&fresh(), json!({ "ignis": { "provisional_tail": "yes" } })).unwrap_err();
        assert_eq!(refused.param.as_deref(), Some("session.ignis.provisional_tail"));
    }

    /// AC 9, 11: the prompt and keywords are the context, as spec 02 joins
    /// them; one language forces it, several detect, both fields refused; a
    /// keyword the system turn cannot carry is refused.
    #[test]
    fn the_context_and_the_language_follow_spec_02() {
        let config = update(&fresh(), input(json!({ "transcription": { "prompt": "A talk.", "keywords": ["ignis", "Qwen3"], "language": "it" } }))).unwrap();
        let open = config.open(512);
        assert_eq!(open.context, "A talk.\nignis, Qwen3");
        assert_eq!(open.context_max_tokens, 512);
        assert_eq!(open.language.as_deref(), Some("Italian"));
        assert_eq!(fresh().open(512).language, None, "nothing is forced unless asked");
        let detect = update(&fresh(), input(json!({ "transcription": { "languages": ["it", "en"] } }))).unwrap();
        assert_eq!(detect.forced, None);
        let one = update(&fresh(), input(json!({ "transcription": { "languages": ["de"] } }))).unwrap();
        assert_eq!(one.forced.as_deref(), Some("German"));
        let both = update(&config, input(json!({ "transcription": { "languages": ["en"] } }))).unwrap_err();
        assert_eq!(both.param.as_deref(), Some("session.audio.input.transcription.languages"));
        let unknown = update(&fresh(), input(json!({ "transcription": { "language": "xx" } }))).unwrap_err();
        assert_eq!(unknown.param.as_deref(), Some("session.audio.input.transcription.language"));
        let keyword = update(&fresh(), input(json!({ "transcription": { "keywords": ["<b>"] } }))).unwrap_err();
        assert_eq!(keyword.param.as_deref(), Some("session.audio.input.transcription.keywords"));
        // A refused update changes nothing: the caller keeps the old one.
        assert_eq!(config.forced.as_deref(), Some("Italian"));
    }

    /// AC 10: `transcription.model` follows spec 02's resolution; the served id
    /// is what the session reports.
    #[test]
    fn the_model_follows_spec_02() {
        let config = update(&fresh(), input(json!({ "transcription": { "model": "gpt-4o-transcribe" } }))).unwrap();
        assert_eq!(config.model, SERVED);
        assert_eq!(config.effective("s")["audio"]["input"]["transcription"]["model"], SERVED);
        let refused = update(&fresh(), input(json!({ "transcription": { "model": "qwen3.8-27b" } }))).unwrap_err();
        assert_eq!((refused.code, refused.param.as_deref()), ("model_kind_mismatch", Some("session.audio.input.transcription.model")));
        assert_eq!(update(&fresh(), input(json!({ "transcription": { "model": "nope" } }))).unwrap_err().code, "model_not_found");
    }
}
