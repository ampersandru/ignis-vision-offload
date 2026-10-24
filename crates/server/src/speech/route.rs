//! `POST /v1/audio/transcriptions` (spec speech/02, OpenAI's shape): a
//! multipart form with an audio `file`, answered as `json`, `text` or
//! `verbose_json`, or with `stream` as Server-Sent Events.
//!
//! Every field is honoured, inert, or refused with a `400` naming it in
//! `param`; none is dropped in silence (spec server/09). The model is the
//! speech model beside the active one: a request names it by its served id,
//! by one of `speech.compat_names`, or by none. The route sits outside the
//! serving gate (it serves on the CPU during a switch, with the text model
//! unloaded) and outside `MODEL_ROUTES` (a transcription is no use of the
//! text model, so it never resets its idle clock).
//!
//! The audio is decoded in memory ([`super::audio`]) and handed to the
//! compute worker as a job that pushes it to the executor a window at a
//! time ([`super::worker`]). The transcript is not logged, nor the audio:
//! one `ignis.speech.transcription` line carries the numbers (Q13, Q45).

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::extract::{Multipart, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_core::Stream;
use ignis_core::compute::ModelKind;
use serde::Serialize;
use serde_json::{json, Map, Value};
use tokio::sync::{mpsc, oneshot};
use utoipa::ToSchema;

use super::audio;
use super::executor::{LoadSettings, SessionOpen, SpeechExecutor, SpeechFiles, Transcript, WINDOW_SAMPLES};
use super::languages;
use super::worker::{Job, Step};
use super::Unavailable;
use crate::api::{bad_request_param, error_response, error_response_naming, refused_field, unknown_model, ApiError};
use crate::config::schema::SpeechGroup;
use crate::deadline::Timeout;
use crate::metrics::{SpeechEnd, SpeechExecutorLabel, SpeechOutcome};
use crate::openai_fields::{self, Surface};
use crate::Server;

/// The form's `response_format`s this server answers.
const FORMATS: [&str; 3] = ["json", "text", "verbose_json"];

/// `POST /v1/audio/transcriptions`'s form (`multipart/form-data`).
#[derive(Debug, Default, ToSchema)]
#[allow(dead_code)]
pub(crate) struct TranscriptionForm {
    /// The audio: WAV, FLAC, MP3 or AAC, or an Ogg or Matroska/WebM
    /// container holding FLAC. Required.
    #[schema(value_type = String, format = Binary)]
    file: Vec<u8>,
    /// Absent, the speech model's served id, or one of `speech.compat_names`
    /// (`whisper-1` by default): the speech model. A text or image model's id
    /// is `409 model_kind_mismatch`.
    model: Option<String>,
    /// Context for the model: names and terms it should spell right. Becomes
    /// the system turn, capped at `speech.context_max_tokens` (the last
    /// tokens are kept).
    prompt: Option<String>,
    /// Words to spell right, joined after the prompt as one line. One holding
    /// `<`, `>`, CR or LF is refused.
    #[serde(rename = "keywords[]")]
    keywords: Option<Vec<String>>,
    /// Forces the language (ISO 639 code or English name, one of the model's
    /// 30). Forcing Italian Italianises English jargon: leave it off unless
    /// the model guesses wrong.
    language: Option<String>,
    /// One entry forces that language; several mean auto-detect. Not with
    /// `language`.
    #[serde(rename = "languages[]")]
    languages: Option<Vec<String>>,
    /// `json` (default), `text` or `verbose_json`. `srt`, `vtt` and
    /// `diarized_json` are refused.
    response_format: Option<String>,
    /// `segment`, with `verbose_json`. `word` is refused.
    #[serde(rename = "timestamp_granularities[]")]
    timestamp_granularities: Option<Vec<String>>,
    /// Server-Sent Events: `transcript.text.delta`, then
    /// `transcript.text.done`, then `[DONE]`.
    stream: Option<bool>,
    /// Absent or `0`: this server decodes greedily.
    temperature: Option<f64>,
    /// Absent or `null`: one block. `auto` and `server_vad` are refused.
    chunking_strategy: Option<String>,
    /// `logprobs` is refused.
    #[serde(rename = "include[]")]
    include: Option<Vec<String>>,
    /// Refused: no diarization.
    #[serde(rename = "known_speaker_names[]")]
    known_speaker_names: Option<Vec<String>>,
    /// Refused: no diarization.
    #[serde(rename = "known_speaker_references[]")]
    known_speaker_references: Option<Vec<String>>,
}

/// `json`'s answer, and the first part of `verbose_json`'s.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TranscriptionJson {
    text: String,
    usage: Usage,
    /// ignis: the language the model heard, as `[{"code": "it"}]`; absent
    /// when the model labelled it `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    languages: Option<Vec<LanguageCode>>,
}

#[derive(Debug, Serialize, ToSchema)]
struct LanguageCode {
    code: String,
}

/// OpenAI's duration usage.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct Usage {
    r#type: &'static str,
    /// Seconds of audio.
    seconds: f64,
}

/// `verbose_json`'s answer.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct TranscriptionVerbose {
    task: &'static str,
    /// The model's own label (`Italian`); absent when it said `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    /// Seconds of audio.
    duration: f64,
    text: String,
    /// One segment spanning the clip: there is no aligner.
    segments: Vec<Segment>,
    usage: Usage,
}

#[derive(Debug, Serialize, ToSchema)]
struct Segment {
    id: u32,
    seek: u32,
    start: f64,
    end: f64,
    text: String,
    tokens: Vec<u32>,
    /// The fields the engine does not compute are 0.
    temperature: f64,
    avg_logprob: f64,
    compression_ratio: f64,
    no_speech_prob: f64,
}

/// One server-sent event of a streamed transcription.
#[derive(Debug, Serialize, ToSchema)]
#[serde(untagged)]
#[allow(dead_code)]
pub(crate) enum TranscriptionStreamEvent {
    Delta { r#type: String, delta: String },
    Done { r#type: String, text: String, usage: Usage },
    Error { r#type: String, error: ApiError },
}

// ── the form ────────────────────────────────────────────────────────────────

#[derive(Default)]
struct Form {
    file: Option<Vec<u8>>,
    model: Option<String>,
    prompt: Option<String>,
    keywords: Vec<String>,
    language: Option<String>,
    languages: Vec<String>,
    response_format: Option<String>,
    granularities: Vec<String>,
    stream: bool,
    /// The fields `openai_fields` classifies, as the JSON values a body
    /// would carry; the array ones under their `[]` names.
    other: Map<String, Value>,
}

/// The array fields OpenAI names with `[]`, which `openai_fields` rows name
/// the same way.
const REFUSED_ARRAYS: [&str; 3] = ["include[]", "known_speaker_names[]", "known_speaker_references[]"];

async fn read_form(mut multipart: Multipart) -> Result<Form, Response> {
    let malformed = |error: axum::extract::multipart::MultipartError| {
        bad_request_param(&format!("the multipart form could not be read: {}", error.body_text()), "file")
    };
    let mut form = Form::default();
    while let Some(field) = multipart.next_field().await.map_err(malformed)? {
        let name = field.name().unwrap_or_default().to_owned();
        if name == "file" {
            form.file = Some(field.bytes().await.map_err(malformed)?.to_vec());
            continue;
        }
        let text = field.text().await.map_err(malformed)?;
        match name.as_str() {
            "model" => form.model = Some(text),
            "prompt" => form.prompt = Some(text),
            "keywords[]" | "keywords" => form.keywords.push(text),
            "language" => form.language = Some(text),
            "languages[]" | "languages" => form.languages.push(text),
            "response_format" => form.response_format = Some(text),
            "timestamp_granularities[]" | "timestamp_granularities" => form.granularities.push(text),
            "stream" => {
                form.stream = text
                    .trim()
                    .parse()
                    .map_err(|_| bad_request_param(&format!("stream must be true or false, got `{text}`"), "stream"))?;
            }
            "temperature" => {
                let number: f64 = text
                    .trim()
                    .parse()
                    .map_err(|_| bad_request_param(&format!("temperature must be a number, got `{text}`"), "temperature"))?;
                form.other.insert("temperature".to_owned(), json!(number));
            }
            "chunking_strategy" => {
                let value = serde_json::from_str(&text).unwrap_or(Value::String(text));
                form.other.insert("chunking_strategy".to_owned(), value);
            }
            _ => {
                let key = format!("{}{}", name.trim_end_matches("[]"), "[]");
                if REFUSED_ARRAYS.contains(&key.as_str()) {
                    let entry = form.other.entry(key).or_insert_with(|| Value::Array(Vec::new()));
                    if let Value::Array(items) = entry {
                        items.push(Value::String(text));
                    }
                }
                // Any other field is outside the table: ignored, as every
                // route ignores what OpenAI documents and no row lists.
            }
        }
    }
    Ok(form)
}

/// What a request asks for, checked.
struct Asked {
    format: &'static str,
    stream: bool,
    context: String,
    /// The language forced, by the model's English name.
    language: Option<String>,
}

/// The one-line list of keywords after the prompt (spec speech/02 §`keywords[]`).
fn context_of(prompt: Option<&str>, keywords: &[String]) -> String {
    let prompt = prompt.map(str::trim).filter(|prompt| !prompt.is_empty());
    let line = keywords.iter().map(|keyword| keyword.trim()).filter(|keyword| !keyword.is_empty()).collect::<Vec<_>>().join(", ");
    match (prompt, line.is_empty()) {
        (Some(prompt), false) => format!("{prompt}\n{line}"),
        (Some(prompt), true) => prompt.to_owned(),
        (None, false) => line,
        (None, true) => String::new(),
    }
}

/// Every field's rule (spec speech/02's table): the ones the handler reads
/// here, the ones the shared table classifies in [`openai_fields`].
fn checked(form: &Form) -> Result<Asked, Response> {
    if let Err(refusal) = openai_fields::check(&form.other, Surface::Transcriptions) {
        return Err(refused_field(refusal));
    }
    let format = match form.response_format.as_deref().map(str::trim).filter(|format| !format.is_empty()) {
        None => "json",
        Some(asked) => match FORMATS.iter().find(|known| **known == asked) {
            Some(known) => known,
            None => {
                let why = match asked {
                    "srt" | "vtt" => "there are no timestamps to cut subtitles from",
                    "diarized_json" => "this server does not diarize",
                    _ => "it is not a format this server answers",
                };
                return Err(error_response_naming(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "unsupported_value",
                    format!("response_format `{asked}` is not served: {why}; send one of {}", FORMATS.join(", ")),
                    Some("response_format"),
                ));
            }
        },
    };
    for granularity in &form.granularities {
        match granularity.trim() {
            "segment" if format == "verbose_json" => {}
            "segment" => {
                return Err(bad_request_param("timestamp_granularities[] applies to response_format verbose_json only", "timestamp_granularities[]"));
            }
            "word" => {
                return Err(error_response_naming(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "unsupported_value",
                    "timestamp_granularities[] `word` is not served: there is no aligner; send `segment` or omit it",
                    Some("timestamp_granularities[]"),
                ));
            }
            other => {
                return Err(bad_request_param(
                    &format!("timestamp_granularities[] `{other}` is not one of segment, word"),
                    "timestamp_granularities[]",
                ));
            }
        }
    }
    for keyword in &form.keywords {
        if keyword.contains(['<', '>', '\r', '\n']) {
            return Err(bad_request_param("keywords[] must not contain `<`, `>` or a line break", "keywords[]"));
        }
    }
    let language = forced_language(form)?;
    Ok(Asked { format, stream: form.stream, context: context_of(form.prompt.as_deref(), &form.keywords), language })
}

/// The language the form forces: `language`, or `languages[]` with one entry;
/// several entries mean auto-detect, and both fields together are refused.
fn forced_language(form: &Form) -> Result<Option<String>, Response> {
    let named = form.language.as_deref().map(str::trim).filter(|language| !language.is_empty());
    let listed: Vec<&str> = form.languages.iter().map(|language| language.trim()).filter(|language| !language.is_empty()).collect();
    if named.is_some() && !listed.is_empty() {
        return Err(bad_request_param("send `language` or `languages[]`, not both", "languages[]"));
    }
    let find = |spelling: &str, param: &str| {
        languages::find(spelling).ok_or_else(|| {
            bad_request_param(
                &format!("{param} `{spelling}` is not one of the speech model's 30 languages ({}), or their English names", languages::codes()),
                param,
            )
        })
    };
    let mut resolved = Vec::with_capacity(listed.len());
    for spelling in &listed {
        resolved.push(find(spelling, "languages[]")?);
    }
    if let Some(spelling) = named {
        return Ok(Some(find(spelling, "language")?.name.to_owned()));
    }
    Ok(match resolved.as_slice() {
        [only] => Some(only.name.to_owned()),
        _ => None,
    })
}

// ── the model ───────────────────────────────────────────────────────────────

/// Which model a request names (spec speech/02 §Which model): the speech
/// model for no `model`, its served id or a compat name; `409
/// model_kind_mismatch` for a text or image model's id, naming both kinds;
/// `404 model_not_found` for anything else.
fn check_model(server: &Server, group: &SpeechGroup, named: Option<&str>) -> Result<(), Response> {
    let Some(named) = named.map(str::trim).filter(|named| !named.is_empty()) else {
        return Ok(());
    };
    if super::Speech::served_id(group).as_deref() == Some(named) || group.compat_names.iter().any(|compat| compat == named) {
        return Ok(());
    }
    let loaded = server.active();
    let kind = if named == loaded.model_id() {
        Some(loaded.kind())
    } else {
        match &server.switcher {
            Some(switcher) => switcher.kind_of(named),
            None => ignis_core::compute::ModelFamily::of_model_id(named).map(|family| family.kind()),
        }
    };
    let Some(kind) = kind.filter(|kind| *kind != ModelKind::Speech) else {
        return Err(unknown_model(server, named));
    };
    Err(error_response_naming(
        StatusCode::CONFLICT,
        "invalid_request_error",
        "model_kind_mismatch",
        format!(
            "`model` names {named}, {}, and this route serves speech models; name the speech model, one of speech.compat_names, or none",
            crate::api::a_kind(kind)
        ),
        Some("model"),
    ))
}

fn unavailable(Unavailable(reason): Unavailable) -> Response {
    // No `Retry-After`: waiting changes nothing about a model that is off,
    // missing or unrunnable here.
    error_response(StatusCode::SERVICE_UNAVAILABLE, "server_error", "speech_unavailable", format!("the speech model is unavailable: {reason}"))
}

// ── the job ─────────────────────────────────────────────────────────────────

/// What a finished job tells its request.
struct Done {
    transcript: Transcript,
    /// What the executor's calls took, from the first window.
    computed: Duration,
}

enum Failure {
    /// The request went away, or its deadline passed: nobody is waiting.
    Cancelled,
    /// The model would not load.
    Load(String),
    /// A call to the executor failed.
    Executor(String),
}

/// One transcription, a window per step.
struct Transcription {
    executor: Arc<dyn SpeechExecutor>,
    files: SpeechFiles,
    load: LoadSettings,
    open: Option<SessionOpen>,
    samples: Vec<f32>,
    cursor: usize,
    session: Option<Box<dyn super::executor::SpeechSession>>,
    audio_seconds: f64,
    cancelled: Arc<AtomicBool>,
    /// The stable text, as it grows, when the request streams.
    deltas: Option<mpsc::UnboundedSender<String>>,
    emitted: String,
    reply: Option<oneshot::Sender<Result<Done, Failure>>>,
    began: Option<Instant>,
}

impl Transcription {
    fn finish(&mut self, result: Result<Done, Failure>) -> Step {
        self.session = None;
        if let Some(reply) = self.reply.take() {
            let _ = reply.send(result);
        }
        Step::Done
    }

    /// Send what `text` adds to what was sent, when it extends it.
    fn emit(&mut self, text: &str) {
        let Some(deltas) = &self.deltas else { return };
        // Only text that extends what was sent is a delta: the executor's
        // stable text is never revised, and a violation sends nothing rather
        // than deltas that do not join to the answer.
        if text.len() > self.emitted.len() && text.starts_with(self.emitted.as_str()) {
            if deltas.send(text[self.emitted.len()..].to_owned()).is_err() {
                self.cancelled.store(true, Ordering::SeqCst);
            }
            self.emitted = text.to_owned();
        }
    }

    fn advance(&mut self) -> Result<Option<Transcript>, Failure> {
        if self.session.is_none() {
            if !self.executor.is_loaded() {
                self.executor.load(&self.files, self.load).map_err(|error| Failure::Load(error.0))?;
            }
            let open = self.open.take().expect("opened once");
            self.session = Some(self.executor.open(open).map_err(|error| Failure::Executor(error.0))?);
        }
        let session = self.session.as_mut().expect("opened above");
        if self.cursor < self.samples.len() {
            let end = (self.cursor + WINDOW_SAMPLES).min(self.samples.len());
            session.push(&self.samples[self.cursor..end]).map_err(|error| Failure::Executor(error.0))?;
            self.cursor = end;
            if self.deltas.is_some() {
                let stable = session.partial().map_err(|error| Failure::Executor(error.0))?.stable;
                self.emit(&stable);
            }
            return Ok(None);
        }
        let transcript = self.session.take().expect("opened above").commit().map_err(|error| Failure::Executor(error.0))?;
        Ok(Some(transcript))
    }
}

impl Job for Transcription {
    fn audio_seconds(&self) -> f64 {
        self.audio_seconds
    }

    fn step(&mut self) -> Step {
        if self.cancelled.load(Ordering::SeqCst) || self.reply.as_ref().is_some_and(oneshot::Sender::is_closed) {
            return self.finish(Err(Failure::Cancelled));
        }
        let began = *self.began.get_or_insert_with(Instant::now);
        match self.advance() {
            Ok(None) => Step::More,
            Ok(Some(transcript)) => {
                self.emit(&transcript.text);
                self.finish(Ok(Done { transcript, computed: began.elapsed() }))
            }
            Err(failure) => {
                if let Failure::Load(why) | Failure::Executor(why) = &failure {
                    // The executor's own words: no audio, no transcript.
                    tracing::warn!(name: "ignis.speech.failed", variant = %self.files.id, reason = %why, "a transcription failed");
                }
                self.finish(Err(failure))
            }
        }
    }
}

// ── the handler ─────────────────────────────────────────────────────────────

#[utoipa::path(
    post,
    path = "/v1/audio/transcriptions",
    tag = "audio",
    operation_id = "create_transcription",
    summary = "A transcript from audio",
    description = "OpenAI's transcription on the speech model (Qwen3-ASR, ADR 0066): the transcript of `file`, a `multipart/form-data` upload, in OpenAI's shapes. The speech model runs on the CPU beside the active model, is loaded by the first request and unloaded after `speech.idle_unload` seconds without one. `model` is absent, the speech model's served id (`qwen3-asr-1.7b` by default) or one of `speech.compat_names` (`whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-transcribe`, `gpt-live-transcribe` by default, so an unmodified client such as Aider works): the id of a text or image model is `409 model_kind_mismatch`, naming both kinds, and any other `404 model_not_found`.

`file` is WAV, FLAC, MP3 or AAC (ADTS or in MP4/M4A), or an Ogg or Matroska/WebM container holding FLAC; any other format is a `400` naming the accepted set. The audio is decoded in memory and resampled to 16 kHz mono; nothing is written to disk. A clip longer than `speech.max_clip_seconds` (1200) is a `400` naming the limit, and is never split. The body limit is the media one (384 MiB).

`prompt` becomes the context (the model's system turn) and `keywords[]` follow it as one line; together they are capped at `speech.context_max_tokens` (512), keeping the last tokens. A keyword holding `<`, `>`, CR or LF is a `400`. `language` forces the language (one of the model's 30, by ISO code or English name); forcing Italian Italianises English jargon, so leave it off unless the model guesses wrong. `languages[]` with one entry forces it, with several auto-detects; both fields together are a `400`.

`response_format`: `json` (default) is `{text, usage: {type: \"duration\", seconds}}` with `languages: [{code}]` carrying the detected language, absent when the model labelled the clip `None`; `text` is the transcript as the body; `verbose_json` is `{task, language, duration, text, segments, usage}` with `language` the model's own label (absent when `None`) and one segment spanning the clip (there is no aligner: the fields the engine does not compute are 0). `srt`, `vtt` and `diarized_json` are refused. `timestamp_granularities[]` takes `segment` with `verbose_json`; `word` is refused.

`stream: true` answers `text/event-stream`: `transcript.text.delta` events as the text becomes stable, then one `transcript.text.done` with the whole `text` (the deltas joined) and `usage` in audio seconds, then `data: [DONE]`. A transcription that fails or passes its deadline ends with one `error` event.

One compute worker serves the speech model: Realtime sessions first, window by window, and batch requests first in, first out in the gaps. Up to `speech.max_queue` requests wait behind the one running; past that the answer is `503 engine_full` with `Retry-After` (at most 15 s). A request is held to `--server-request-timeout`, or `--server-stream-timeout` when it streams, from its admission (ADR 0058): past it, `504 request_timeout`, or the `error` event, whatever `--server-timeout-reply` says; a transcription has no answer that says it is incomplete. The route serves during a model switch and with the text model unloaded, and a transcription does not reset the text model's idle clock.",
    request_body(content = TranscriptionForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "The transcript. `application/json` for `json` and `verbose_json`, `text/plain` for `text`, `text/event-stream` when `stream` is true.", content(
            (TranscriptionJson = "application/json"),
            (TranscriptionVerbose = "application/json"),
            (String = "text/plain"),
            (TranscriptionStreamEvent = "text/event-stream"),
        )),
        (status = 400, description = "A field outside its rule, naming it in `param`: no `file`, a format this server does not decode (naming the accepted set), a clip over `speech.max_clip_seconds`, a `response_format`, `timestamp_granularities[]`, `language`, `languages[]` or keyword outside what is served, a `temperature` other than 0, a `chunking_strategy` other than null, `include[]` or the diarization fields.", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "`model` names no model this server has (`model_not_found`).", body = ApiError),
        (status = 409, description = "`model` names a text or image model (`model_kind_mismatch`, naming both kinds).", body = ApiError),
        (status = 500, description = "The speech model failed a call (`engine_error`).", body = ApiError),
        (status = 503, description = "The speech model is unavailable (`speech_unavailable`, with the reason, and no `Retry-After`): turned off, its files not on disk (the reason names the download command), or the CPU lacks AVX2. Or the queue is full (`engine_full`, with `Retry-After`).", body = ApiError),
        (status = 504, description = "The request ran past its deadline (`request_timeout`).", body = ApiError),
    ),
)]
pub(crate) async fn create_transcription(State(server): State<Arc<Server>>, multipart: Multipart) -> Response {
    let form = match read_form(multipart).await {
        Ok(form) => form,
        Err(refused) => return refused,
    };
    // The last byte is in: the wait a request reports counts from here.
    let received = Instant::now();
    let live = server.live();
    let group = &live.speech;
    if let Err(refused) = check_model(&server, group, form.model.as_deref()) {
        return refused;
    }
    let files = match server.speech.resolve(group) {
        Ok(files) => files,
        Err(why) => return unavailable(why),
    };
    let asked = match checked(&form) {
        Ok(asked) => asked,
        Err(refused) => return refused,
    };
    let Some(bytes) = form.file else {
        return bad_request_param("`file` is required: the audio to transcribe", "file");
    };
    // A full queue is answered before the clip is decoded: the decode is the
    // route's own CPU work, and a refused request should not cost it. The
    // submit below checks again, the room being gone by then at times.
    if let Err(full) = server.speech.admit(group) {
        return engine_full(group.max_queue, full);
    }
    let max_seconds = group.max_clip_seconds;
    // A decoder that panics on a malformed stream answers this request, not
    // the connection.
    let decoded = match tokio::task::spawn_blocking(move || audio::decode(bytes, max_seconds)).await {
        Ok(decoded) => decoded,
        Err(_) => return bad_request_param("`file` could not be decoded as audio", "file"),
    };
    let clip = match decoded {
        Ok(clip) => clip,
        Err(error) => return bad_request_param(&error.message(), "file"),
    };
    let audio_seconds = clip.seconds();

    let executor = Arc::clone(server.speech.executor());
    let executor_label = executor.name();
    let variant = files.id.clone();
    let (reply, replied) = oneshot::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let (deltas, streamed) = if asked.stream {
        let (tx, rx) = mpsc::unbounded_channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let job = Transcription {
        executor,
        files,
        load: LoadSettings { threads: group.threads, pinned_weights: group.pinned_weights },
        open: Some(SessionOpen {
            context: asked.context,
            context_max_tokens: group.context_max_tokens,
            language: asked.language,
        }),
        samples: clip.samples,
        cursor: 0,
        session: None,
        audio_seconds,
        cancelled: Arc::clone(&cancelled),
        deltas,
        emitted: String::new(),
        reply: Some(reply),
        began: None,
    };
    // The deadline runs from admission (ADR 0058), the queue's wait included.
    let timeout = Timeout::of(&live, asked.stream);
    let deadline = timeout.from_now();
    if let Err(full) = server.speech.submit(Box::new(job), group) {
        return engine_full(group.max_queue, full);
    }
    let ending = Ending {
        server: Arc::clone(&server),
        cancelled,
        executor: SpeechExecutorLabel::of(executor_label),
        executor_name: executor_label,
        variant,
        audio_seconds,
        received,
        ended: false,
    };

    if let Some(deltas) = streamed {
        let (events, stream) = mpsc::unbounded_channel();
        tokio::spawn(stream_answer(ending, replied, deltas, events, deadline, timeout));
        return Sse::new(Events { events: stream }).into_response();
    }
    answer(ending, replied, deadline, timeout, asked.format).await
}

/// `503 engine_full` with the `Retry-After` the queue estimated.
fn engine_full(max_queue: u32, full: super::worker::Full) -> Response {
    let mut response = error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        "engine_full",
        format!("the speech model has {max_queue} transcriptions waiting (speech.max_queue); retry in {} s", full.retry_after_secs),
    );
    response.headers_mut().insert(axum::http::header::RETRY_AFTER, HeaderValue::from(full.retry_after_secs));
    response
}

/// A transcription's end: what is logged and counted once, and the job's
/// cancel when the request goes away without one.
struct Ending {
    server: Arc<Server>,
    cancelled: Arc<AtomicBool>,
    executor: SpeechExecutorLabel,
    executor_name: &'static str,
    variant: String,
    audio_seconds: f64,
    received: Instant,
    ended: bool,
}

impl Ending {
    fn end(&mut self, outcome: SpeechOutcome, transcript: Option<&Transcript>, computed: Option<Duration>) {
        if std::mem::replace(&mut self.ended, true) {
            return;
        }
        let wait = self.received.elapsed();
        if let Some(metrics) = &self.server.metrics {
            metrics.record_speech(&SpeechEnd { executor: self.executor, outcome, audio_seconds: self.audio_seconds, wait });
        }
        // Neither the audio nor the transcript: the numbers (Q13, Q45).
        let x_real_time = computed.filter(|computed| !computed.is_zero()).map(|computed| self.audio_seconds / computed.as_secs_f64());
        tracing::info!(
            name: "ignis.speech.transcription",
            audio_seconds = self.audio_seconds,
            executor = self.executor_name,
            variant = %self.variant,
            x_real_time = x_real_time.unwrap_or(0.0),
            wait_ms = wait.as_millis() as u64,
            language = transcript.and_then(|transcript| transcript.language.as_deref()).unwrap_or("none"),
            outcome = outcome.label(),
            "a transcription ended"
        );
    }
}

impl Drop for Ending {
    /// The request went away before its end: the job stops at its next
    /// window and the transcription counts as cancelled.
    fn drop(&mut self) {
        if !self.ended {
            self.cancelled.store(true, Ordering::SeqCst);
            self.end(SpeechOutcome::Cancelled, None, None);
        }
    }
}

/// How a failed job is answered.
fn failure_response(failure: &Failure) -> Response {
    match failure {
        Failure::Load(why) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "speech_unavailable",
            format!("the speech model could not be loaded: {why}"),
        ),
        Failure::Executor(_) | Failure::Cancelled => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "engine_error",
            "the speech model failed this transcription; the request can be sent again",
        ),
    }
}

/// The answer to a request that does not stream.
async fn answer(
    mut ending: Ending,
    replied: oneshot::Receiver<Result<Done, Failure>>,
    deadline: tokio::time::Instant,
    timeout: Timeout,
    format: &'static str,
) -> Response {
    match tokio::time::timeout_at(deadline, replied).await {
        Ok(Ok(Ok(done))) => {
            ending.end(SpeechOutcome::Completed, Some(&done.transcript), Some(done.computed));
            body(&done.transcript, ending.audio_seconds, format)
        }
        Ok(Ok(Err(failure))) => {
            ending.end(SpeechOutcome::Failed, None, None);
            failure_response(&failure)
        }
        Ok(Err(_)) => {
            ending.end(SpeechOutcome::Failed, None, None);
            failure_response(&Failure::Executor("the worker ended".to_owned()))
        }
        Err(_) => {
            ending.cancelled.store(true, Ordering::SeqCst);
            ending.end(SpeechOutcome::TimedOut, None, None);
            timeout.response()
        }
    }
}

/// Seconds of audio as the answer spells them: milliseconds are plenty.
fn seconds(audio_seconds: f64) -> f64 {
    (audio_seconds * 1000.0).round() / 1000.0
}

/// A finished transcript in `format`.
fn body(transcript: &Transcript, audio_seconds: f64, format: &str) -> Response {
    let seconds = seconds(audio_seconds);
    match format {
        "text" => (
            [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            transcript.text.clone(),
        )
            .into_response(),
        "verbose_json" => Json(TranscriptionVerbose {
            task: "transcribe",
            language: transcript.language.clone(),
            duration: seconds,
            text: transcript.text.clone(),
            segments: vec![Segment {
                id: 0,
                seek: 0,
                start: 0.0,
                end: seconds,
                text: transcript.text.clone(),
                tokens: transcript.tokens.clone(),
                temperature: 0.0,
                avg_logprob: 0.0,
                compression_ratio: 0.0,
                no_speech_prob: 0.0,
            }],
            usage: Usage { r#type: "duration", seconds },
        })
        .into_response(),
        _ => Json(TranscriptionJson {
            text: transcript.text.clone(),
            usage: Usage { r#type: "duration", seconds },
            languages: transcript.language.as_deref().map(|label| vec![LanguageCode { code: languages::code_of_label(label) }]),
        })
        .into_response(),
    }
}

/// What a streamed transcription's channel carries.
enum Sent {
    Event(String),
    Done,
}

/// A streamed transcription's tail: the deltas the job sends become events as
/// they come, then the done event once the job ends, or an `error` event
/// when it fails or passes its deadline; `[DONE]` last.
async fn stream_answer(
    mut ending: Ending,
    mut replied: oneshot::Receiver<Result<Done, Failure>>,
    mut deltas: mpsc::UnboundedReceiver<String>,
    events: mpsc::UnboundedSender<Sent>,
    deadline: tokio::time::Instant,
    timeout: Timeout,
) {
    let send = |event: Value| {
        let _ = events.send(Sent::Event(event.to_string()));
    };
    let result = loop {
        tokio::select! {
            Some(delta) = deltas.recv() => send(json!({ "type": "transcript.text.delta", "delta": delta })),
            result = &mut replied => break Some(result),
            () = tokio::time::sleep_until(deadline) => break None,
            // The client went away: the answer has nobody to go to.
            () = events.closed() => return,
        }
    };
    // What the job sent before it ended.
    while let Ok(delta) = deltas.try_recv() {
        send(json!({ "type": "transcript.text.delta", "delta": delta }));
    }
    match result {
        Some(Ok(Ok(done))) => {
            ending.end(SpeechOutcome::Completed, Some(&done.transcript), Some(done.computed));
            send(json!({
                "type": "transcript.text.done",
                "text": done.transcript.text,
                "usage": { "type": "duration", "seconds": seconds(ending.audio_seconds) },
            }));
        }
        Some(Ok(Err(failure))) => {
            ending.end(SpeechOutcome::Failed, None, None);
            send(json!({ "type": "error", "error": error_event(&failure) }));
        }
        Some(Err(_)) => {
            ending.end(SpeechOutcome::Failed, None, None);
            send(json!({ "type": "error", "error": error_event(&Failure::Executor("the worker ended".to_owned())) }));
        }
        None => {
            ending.cancelled.store(true, Ordering::SeqCst);
            ending.end(SpeechOutcome::TimedOut, None, None);
            send(json!({ "type": "error", "error": body_of(&timeout.error()) }));
        }
    }
    let _ = events.send(Sent::Done);
}

/// The `error` object of an API error, which is what an `error` event carries.
fn body_of(error: &ApiError) -> Value {
    serde_json::to_value(error).expect("an error body serializes")["error"].clone()
}

/// The body of an `error` event.
fn error_event(failure: &Failure) -> Value {
    let (code, message) = match failure {
        Failure::Load(why) => ("speech_unavailable", format!("the speech model could not be loaded: {why}")),
        Failure::Executor(_) | Failure::Cancelled => {
            ("engine_error", "the speech model failed this transcription; the request can be sent again".to_owned())
        }
    };
    body_of(&ApiError::new("server_error", code, &message))
}

/// The events of a streamed transcription, as `data:` lines.
struct Events {
    events: mpsc::UnboundedReceiver<Sent>,
}

impl Stream for Events {
    type Item = Result<Event, Infallible>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.events.poll_recv(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Sent::Event(data))) => Poll::Ready(Some(Ok(Event::default().data(data)))),
            // The OpenAI SDK stops reading on this line, and LocalAI sends it.
            Poll::Ready(Some(Sent::Done)) => {
                self.events.close();
                Poll::Ready(Some(Ok(Event::default().data("[DONE]"))))
            }
            Poll::Ready(None) => Poll::Ready(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_context_is_the_prompt_then_the_keywords_as_one_line() {
        let keywords = vec!["ignis".to_owned(), " Qwen3 ".to_owned(), String::new()];
        assert_eq!(context_of(Some("A talk about GPUs."), &keywords), "A talk about GPUs.\nignis, Qwen3");
        assert_eq!(context_of(None, &keywords), "ignis, Qwen3");
        assert_eq!(context_of(Some("  just a prompt "), &[]), "just a prompt");
        assert_eq!(context_of(None, &[]), "");
        assert_eq!(context_of(Some(" "), &[]), "");
    }

    #[test]
    fn one_language_forces_several_detect_and_both_fields_are_refused() {
        let form = |language: Option<&str>, listed: &[&str]| Form {
            language: language.map(str::to_owned),
            languages: listed.iter().map(|language| (*language).to_owned()).collect(),
            ..Form::default()
        };
        assert_eq!(forced_language(&form(Some("it"), &[])).ok().flatten().as_deref(), Some("Italian"));
        assert_eq!(forced_language(&form(Some("Italian"), &[])).ok().flatten().as_deref(), Some("Italian"));
        assert_eq!(forced_language(&form(None, &["de"])).ok().flatten().as_deref(), Some("German"));
        assert_eq!(forced_language(&form(None, &["de", "en"])).ok().flatten(), None);
        assert_eq!(forced_language(&form(None, &[])).ok().flatten(), None);
        assert!(forced_language(&form(Some("it"), &["en"])).is_err());
        assert!(forced_language(&form(Some("xx"), &[])).is_err());
        assert!(forced_language(&form(None, &["en", "xx"])).is_err());
    }
}
