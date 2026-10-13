//! The **loop guard** (exploration, branch `experiment/loop-rewind`): a task
//! between a chat request's engine stream and its consumer that watches the
//! reasoning for a loop and, in `rewind` mode, takes the generation back to
//! before it.
//!
//! Why: `docs/findings/2026-10-12-reasoning-loops-are-the-models.md` — about a
//! third of long-enumeration turns loop in their reasoning, on ignis and on
//! ninfer alike, and only the thinking budget's close ends them.
//!
//! **Detection.** Every [`CHECK_EVERY_CHARS`] characters of new reasoning, the
//! zlib ratio of its last `window` characters; under `ratio` it is a loop.
//! Measured offline on 2026-10-13 (`.scratch/loops/NOTES.md`): no real agent
//! text of 838 (code included) under 0.12 at a 3,000-character window, and
//! every verbatim and single-token loop of the 2026-10-12 runs over it.
//! Counting loops (~0.21, the band of test code) are not seen.
//!
//! **Rewind.** The engine request is cancelled and a **continuation**
//! submitted: the same input, its prompt extended by the generated tokens
//! kept from before the loop ([`cut_point`]) and a nudge sentence, with what
//! is left of `max_tokens` and of the thinking budget. The prompt's own
//! checkpoint absorbs the prompt, so only the kept reasoning prefills. The
//! client keeps what it has already received and is told what was taken back
//! by a tag in the reasoning ([`tag`]), which the Playground strikes through.
//! After `retries` nudges, the next loop is closed with the thinking budget's
//! own close ([`crate::thinking::THINKING_CLOSE_TEXT`]).
//!
//! Not covered: `/v1/responses`; a request with images (logged, never
//! rewound); a loop in the answer. A continuation's penalty counts start from
//! zero, and a forced tool call (`tool_choice`) is not forced after a close
//! the guard made.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use ignis_core::{RequestClass, RequestId, RequestInput, SchedEvent, SubmitError, TokenId};
use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

use crate::decoder::{Channel, OutputDecoder};
use crate::engine::{Engine, EventStream, RequestNotes};
use crate::template::TemplateProvider;

/// What the guard does with a request (`--model-loop-guard`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopGuardMode {
    /// No guard: the engine's stream goes to the consumer untouched.
    #[default]
    Off,
    /// Detect and log, never act.
    Log,
    /// Detect, take the loop back and nudge; close after the retries.
    Rewind,
}

impl LoopGuardMode {
    /// The modes' written names, in order.
    pub const NAMES: &'static [&'static str] = &["off", "log", "rewind"];

    /// The mode a written name names.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "off" => Some(Self::Off),
            "log" => Some(Self::Log),
            "rewind" => Some(Self::Rewind),
            _ => None,
        }
    }

    /// The mode's written name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Log => "log",
            Self::Rewind => "rewind",
        }
    }
}

/// `--model-loop-guard-ratio`'s default, in thousandths: 0.12.
pub const DEFAULT_RATIO_PERMILLE: u32 = 120;
/// `--model-loop-guard-window`'s default, in characters of reasoning.
pub const DEFAULT_WINDOW_CHARS: u32 = 3000;
/// `--model-loop-guard-margin`'s default, in characters before the window.
pub const DEFAULT_MARGIN_CHARS: u32 = 1000;
/// `--model-loop-guard-retries`'s default: nudges before the close.
pub const DEFAULT_RETRIES: u32 = 2;

/// The guard's knobs, as the live config holds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopGuardParams {
    pub mode: LoopGuardMode,
    /// A window whose zlib ratio is under this many thousandths is a loop.
    pub ratio_permille: u32,
    /// The window, in characters of reasoning.
    pub window_chars: u32,
    /// How far before the flagged window a rewind starts looking for the
    /// loop's start, in characters.
    pub margin_chars: u32,
    /// The nudges a looping reasoning gets before it is closed.
    pub retries: u32,
}

impl Default for LoopGuardParams {
    fn default() -> Self {
        Self {
            mode: LoopGuardMode::Off,
            ratio_permille: DEFAULT_RATIO_PERMILLE,
            window_chars: DEFAULT_WINDOW_CHARS,
            margin_chars: DEFAULT_MARGIN_CHARS,
            retries: DEFAULT_RETRIES,
        }
    }
}

/// How often the window is scored, in characters of new reasoning.
pub const CHECK_EVERY_CHARS: usize = 250;

/// The cut's walk back over the loop's leading edge: a window of this many
/// characters, moved back this many at a time while its ratio stays under
/// [`ONSET_RATIO`]. Calibrated by eye on the 16 flagged reasoning loops of
/// 2026-10-12 (`.scratch/loop-rewind/cut_rule.py`); real reasoning's
/// 400-character windows bottom out at 0.255 (0.295 at the 1st percentile).
const ONSET_WINDOW_CHARS: usize = 400;
const ONSET_STRIDE_CHARS: usize = 50;
const ONSET_RATIO: f64 = 0.35;
/// The cut moves back to the line break before it, when one is this close.
const SNAP_CHARS: usize = 300;

/// The least of `max_tokens` a continuation must have left to be worth one.
const MIN_CONTINUATION_TOKENS: u32 = 512;

/// The nudges, in order: the model's own words when it caught itself
/// (2026-10-12 seed 109: "I'm repeating. Let me stop the repetition and just
/// write the content directly."). The last one repeats when `retries` asks
/// for more.
pub const NUDGES: [&str; 2] = [
    "\n\nWait, I'm starting to repeat myself. Let me stop and move on with what I already have.\n\n",
    "\n\nI keep repeating myself. I already have enough to write the answer, so I'll wrap up my reasoning now.\n\n",
];

/// The tag's fixed opening, for a client that looks for it.
pub const TAG_PREFIX: &str = "[LOOP DETECTED, REWINDING ";

/// The tag the client reads in the reasoning where the guard took `tokens`
/// tokens back: the `chars` characters (Unicode scalar values) of reasoning
/// before it are no longer the model's context. It opens with a line break so
/// that it never joins a character the last token left incomplete.
pub fn tag(tokens: usize, chars: usize) -> String {
    format!("\n\n{TAG_PREFIX}{tokens} TOKENS / {chars} CHARS]\n\n")
}

/// The zlib ratio of `text`: compressed bytes over UTF-8 bytes. Lower is more
/// repetitive; normal reasoning sits at 0.30-0.54 over 3,000 characters.
pub fn compress_ratio(text: &str) -> f64 {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return 1.0;
    }
    miniz_oxide::deflate::compress_to_vec_zlib(bytes, 9).len() as f64 / bytes.len() as f64
}

fn ratio_of(chars: &[char]) -> f64 {
    compress_ratio(&chars.iter().collect::<String>())
}

/// Where to take the reasoning back to, for a loop found in the window that
/// ends at `at`: `window + margin` characters before `at`, walked back over
/// the loop's leading edge while it stays repetitive, then to the line break
/// before it. Never before `floor` (the end of the last nudge).
pub fn cut_point(text: &[char], at: usize, floor: usize, params: &LoopGuardParams) -> usize {
    let back = params.window_chars as usize + params.margin_chars as usize;
    let mut cut = at.saturating_sub(back).max(floor);
    while cut >= floor + ONSET_STRIDE_CHARS
        && cut - ONSET_STRIDE_CHARS + ONSET_WINDOW_CHARS <= text.len()
        && ratio_of(&text[cut - ONSET_STRIDE_CHARS..cut - ONSET_STRIDE_CHARS + ONSET_WINDOW_CHARS]) < ONSET_RATIO
    {
        cut -= ONSET_STRIDE_CHARS;
    }
    let low = cut.saturating_sub(SNAP_CHARS).max(floor);
    match text[low..cut].iter().rposition(|&c| c == '\n') {
        Some(i) => low + i + 1,
        None => cut,
    }
}

/// When to score the window, and from where a loop may be found.
#[derive(Debug, Clone, Copy)]
struct Watch {
    /// No window that starts before this character is scored: the end of
    /// the last nudge, or of the last logged loop.
    floor: usize,
    /// The reasoning length the next score waits for.
    next_check: usize,
}

impl Watch {
    /// A watch whose first score is the first full window after `floor`.
    fn from(floor: usize, params: &LoopGuardParams) -> Self {
        Self { floor, next_check: floor + params.window_chars as usize }
    }

    /// The window's ratio when `text` has grown enough to score it again and
    /// it is a loop.
    fn check(&mut self, text: &[char], params: &LoopGuardParams) -> Option<f64> {
        if text.len() < self.next_check {
            return None;
        }
        self.next_check = text.len() + CHECK_EVERY_CHARS;
        let window = params.window_chars as usize;
        if text.len() < self.floor + window {
            return None;
        }
        let ratio = ratio_of(&text[text.len() - window..]);
        (ratio * 1000.0 < f64::from(params.ratio_permille)).then_some(ratio)
    }
}

/// The generation as the model sees it: its tokens, the reasoning they
/// decoded to, and the reasoning's length after each token.
struct View {
    decoder: OutputDecoder,
    tokens: Vec<TokenId>,
    text: Vec<char>,
    chars_after: Vec<usize>,
}

impl View {
    fn new(decoder: OutputDecoder) -> Self {
        Self { decoder, tokens: Vec::new(), text: Vec::new(), chars_after: Vec::new() }
    }

    fn push(&mut self, token: TokenId) {
        for delta in self.decoder.push(&[token]) {
            if delta.channel == Channel::Reasoning {
                self.text.extend(delta.text.chars());
            }
        }
        self.tokens.push(token);
        self.chars_after.push(self.text.len());
    }

    fn in_reasoning(&self) -> bool {
        self.decoder.channel() == Channel::Reasoning
    }

    /// The tokens whose text ends at or before character `cut`.
    fn tokens_before(&self, cut: usize) -> usize {
        self.chars_after.partition_point(|&chars| chars <= cut)
    }
}

/// What the guard needs from the engine and the template: cancel and submit
/// a request, decode and encode text. A trait so a test can stand in for the
/// model thread.
pub trait GuardOps: Send + Sync + 'static {
    /// Abort a request (fire-and-forget).
    fn cancel(&self, request: RequestId);
    /// Submit a continuation; `None` when the engine would not take it.
    fn submit(&self, input: RequestInput) -> Pin<Box<dyn Future<Output = Option<(RequestId, EventStream)>> + Send + '_>>;
    /// A fresh channel-splitting decoder, starting in the reasoning.
    fn decoder(&self) -> OutputDecoder;
    /// `text` as tokens of the loaded tokenizer.
    fn encode(&self, text: &str) -> Option<Vec<TokenId>>;
}

/// [`GuardOps`] on the live engine and template.
pub struct EngineOps {
    engine: Engine,
    class: RequestClass,
    template: Arc<dyn TemplateProvider>,
}

impl EngineOps {
    pub fn new(engine: Engine, class: RequestClass, template: Arc<dyn TemplateProvider>) -> Self {
        Self { engine, class, template }
    }
}

/// How many times a continuation the engine refused as `Full` is tried
/// again, each after a request leaves (or five seconds).
const SUBMIT_TRIES: u32 = 12;

impl GuardOps for EngineOps {
    fn cancel(&self, request: RequestId) {
        self.engine.cancel(request);
    }

    fn submit(&self, input: RequestInput) -> Pin<Box<dyn Future<Output = Option<(RequestId, EventStream)>> + Send + '_>> {
        Box::pin(async move {
            for _ in 0..SUBMIT_TRIES {
                let mut freed = self.engine.freed();
                freed.borrow_and_update();
                match self.engine.submit_with_notes(input.clone(), self.class, RequestNotes::default()).await {
                    Ok(pair) => return Some(pair),
                    Err(SubmitError::Full) => {
                        let _ = tokio::time::timeout(Duration::from_secs(5), freed.changed()).await;
                    }
                    Err(error) => {
                        tracing::warn!(name: "ignis.loop_guard.submit_refused", error = ?error, "the continuation was refused");
                        return None;
                    }
                }
            }
            None
        })
    }

    fn decoder(&self) -> OutputDecoder {
        OutputDecoder::new(self.template.token_decoder(), true)
    }

    fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
        self.template.encode_literal(text)
    }
}

/// Put the guard between `stream` (the engine's, for `request`, submitted
/// from `input`) and its consumer: the stream returned. `Off` returns
/// `stream` itself.
pub fn guard(params: LoopGuardParams, ops: impl GuardOps, input: RequestInput, request: RequestId, stream: EventStream) -> EventStream {
    if params.mode == LoopGuardMode::Off {
        return stream;
    }
    let (tx, rx) = unbounded_channel();
    tokio::spawn(run(params, ops, input, request, stream, tx));
    rx
}

/// One guarded request's state.
struct Guard<O: GuardOps> {
    params: LoopGuardParams,
    ops: O,
    /// The request as submitted: every continuation is built from it.
    input: RequestInput,
    /// The client's request id: every event it receives carries it.
    original: RequestId,
    /// The engine request generating now.
    current: RequestId,
    view: View,
    watch: Watch,
    /// The nudges given so far.
    rewinds: u32,
    /// The guard closed the reasoning: nothing more to watch.
    closed: bool,
    /// Token events sent to the client.
    forwarded: u32,
    /// The client's token count when the current continuation's first token
    /// arrives: what its `Done` indices are offset by.
    offset: u32,
}

/// A continuation, ready to go: the tokens the client receives before it.
struct Continuation {
    request: RequestId,
    stream: EventStream,
    announce: Vec<TokenId>,
}

impl<O: GuardOps> Guard<O> {
    /// After each token: score the window, and act on a loop.
    async fn after_token(&mut self) -> Option<Continuation> {
        if self.closed || !self.view.in_reasoning() {
            return None;
        }
        let ratio = self.watch.check(&self.view.text, &self.params)?;
        let at = self.view.text.len();
        let rewindable = self.params.mode == LoopGuardMode::Rewind && self.input.multimodal.is_none();
        if !rewindable {
            tracing::info!(
                name: "ignis.loop_guard.detected",
                request = self.original, engine_request = self.current, at_chars = at, ratio = ratio,
                "loop in the reasoning (log only)"
            );
            self.watch = Watch::from(at, &self.params);
            return None;
        }
        self.rewind(at, ratio).await
    }

    async fn rewind(&mut self, at: usize, ratio: f64) -> Option<Continuation> {
        let cut = cut_point(&self.view.text, at, self.watch.floor, &self.params);
        let kept = self.view.tokens_before(cut);
        // The kept tokens' whole text: a decoder holds back a trailing line
        // break until the text after it arrives, so their own count can be
        // one short of what the client has.
        let kept_chars = {
            let mut decoder = self.ops.decoder();
            let mut deltas = decoder.push(&self.view.tokens[..kept]);
            deltas.extend(decoder.finish());
            deltas.iter().filter(|d| d.channel == Channel::Reasoning).map(|d| d.text.chars().count()).sum::<usize>()
        };
        let close = self.rewinds >= self.params.retries;
        let sentence = if close {
            crate::thinking::THINKING_CLOSE_TEXT
        } else {
            NUDGES[(self.rewinds as usize).min(NUDGES.len() - 1)]
        };
        let (Some(nudge), Some(announce)) =
            (self.ops.encode(sentence), self.ops.encode(&tag(self.view.tokens.len() - kept, at - kept_chars)))
        else {
            tracing::warn!(name: "ignis.loop_guard.no_tokenizer", request = self.original, "cannot encode the nudge: log only");
            self.watch = Watch::from(at, &self.params);
            return None;
        };
        let used = (kept + nudge.len()) as u32;
        let mut next = self.input.clone();
        if let Some(max_tokens) = next.params.max_tokens {
            let left = max_tokens.saturating_sub(used);
            if left < MIN_CONTINUATION_TOKENS {
                tracing::info!(name: "ignis.loop_guard.no_room", request = self.original, at_chars = at, left, "loop, no room left to rewind");
                self.watch = Watch::from(at, &self.params);
                return None;
            }
            next.params.max_tokens = Some(left);
        }
        let budget = match self.input.params.max_tokens {
            Some(generation) => ignis_core::thinking_budget::effective(self.input.params.thinking_budget, generation),
            None => self.input.params.thinking_budget,
        };
        next.params.thinking_budget = if close { None } else { budget.map(|b| b.saturating_sub(used)) };
        next.params.starts_in_reasoning = !close;
        next.params.seed = self.input.params.seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(u64::from(self.rewinds) + 1));
        next.tokens.extend_from_slice(&self.view.tokens[..kept]);
        next.tokens.extend_from_slice(&nudge);

        self.ops.cancel(self.current);
        let (request, stream) = self.ops.submit(next).await?;
        tracing::info!(
            name: "ignis.loop_guard.rewound",
            request = self.original, cancelled = self.current, continuation = request,
            at_chars = at, ratio = ratio, cut_chars = kept_chars,
            rewound_tokens = self.view.tokens.len() - kept, rewound_chars = at - kept_chars,
            attempt = self.rewinds + 1, closed = close,
            "loop in the reasoning: rewound"
        );

        // The model's view: the kept tokens, then the nudge.
        let mut view = View::new(self.ops.decoder());
        for &token in self.view.tokens[..kept].iter().chain(&nudge) {
            view.push(token);
        }
        self.view = view;
        self.watch = Watch::from(self.view.text.len(), &self.params);
        self.rewinds += 1;
        self.closed = close;
        self.current = request;
        let mut announce = announce;
        announce.extend_from_slice(&nudge);
        Some(Continuation { request, stream, announce })
    }

    fn send_token(&mut self, tx: &UnboundedSender<SchedEvent>, token: TokenId) -> bool {
        self.forwarded += 1;
        tx.send(SchedEvent::Token { request: self.original, token }).is_ok()
    }
}

async fn run<O: GuardOps>(
    params: LoopGuardParams,
    ops: O,
    input: RequestInput,
    request: RequestId,
    mut stream: EventStream,
    tx: UnboundedSender<SchedEvent>,
) {
    let view = View::new(ops.decoder());
    let mut guard = Guard {
        params,
        ops,
        input,
        original: request,
        current: request,
        view,
        watch: Watch::from(0, &params),
        rewinds: 0,
        closed: false,
        forwarded: 0,
        offset: 0,
    };
    // Once a continuation runs, only its tokens and its end reach the
    // client: its admission and prompt-reuse events describe a request the
    // client never made.
    let mut continued = false;
    while let Some(mut event) = stream.recv().await {
        match event {
            SchedEvent::Token { token, .. } => {
                if !guard.send_token(&tx, token) {
                    guard.ops.cancel(guard.current);
                    return;
                }
                guard.view.push(token);
                if let Some(continuation) = guard.after_token().await {
                    for token in continuation.announce {
                        if !guard.send_token(&tx, token) {
                            guard.ops.cancel(continuation.request);
                            return;
                        }
                    }
                    guard.offset = guard.forwarded;
                    stream = continuation.stream;
                    continued = true;
                }
            }
            SchedEvent::Done { ref mut request, ref mut tokens, ref mut thinking, ref mut reasoning_redirected_at, .. } => {
                *request = guard.original;
                if continued {
                    let offset = guard.offset;
                    *tokens = guard.forwarded;
                    if let Some(outcome) = thinking {
                        outcome.forced_at = outcome.forced_at.map(|at| at + offset);
                        outcome.closed_at = outcome.closed_at.map(|at| at + offset);
                    }
                    *reasoning_redirected_at = reasoning_redirected_at.map(|at| at + offset);
                }
                let _ = tx.send(event);
                return;
            }
            other if !continued => {
                if tx.send(other).is_err() {
                    guard.ops.cancel(guard.current);
                    return;
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::TokenDecoder;
    use ignis_core::{DecodeParams, FinishReason};
    use std::sync::Mutex;

    /// One token per character: its code point.
    struct CharDecoder;
    impl TokenDecoder for CharDecoder {
        fn push(&mut self, token: TokenId) -> String {
            char::from_u32(token).map(String::from).unwrap_or_default()
        }
        fn finish(&mut self) -> String {
            String::new()
        }
    }

    fn tokens_of(text: &str) -> Vec<TokenId> {
        text.chars().map(u32::from).collect()
    }

    fn text_of(tokens: &[TokenId]) -> String {
        tokens.iter().filter_map(|&t| char::from_u32(t)).collect()
    }

    /// Plausible reasoning that does not repeat: a running account of
    /// numbered, varied sentences.
    fn prose(chars: usize) -> String {
        const WORDS: &[&str] = &[
            "Rome", "founded", "kings", "republic", "senate", "consuls", "tribunes", "plebeians", "wars", "Carthage",
            "Hannibal", "Scipio", "Gracchi", "Marius", "Sulla", "Pompey", "Caesar", "Augustus", "empire", "Nero",
            "Flavians", "Trajan", "Hadrian", "crisis", "Diocletian", "Constantine", "Christianity", "Goths", "Vandals",
            "Odoacer", "popes", "Charlemagne", "communes", "Avignon", "Renaissance", "Borgia", "sack", "Baroque",
            "Bernini", "Napoleon", "unification", "capital", "Fascism", "war", "liberation", "republic", "Jubilee",
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

    fn looping(chars: usize) -> String {
        "the \"Dark Ages\" (the \"Dark Ages\", the \"Dark Ages\"), ".chars().cycle().take(chars).collect()
    }

    fn params(mode: LoopGuardMode) -> LoopGuardParams {
        LoopGuardParams { mode, ..LoopGuardParams::default() }
    }

    #[test]
    fn prose_is_not_a_loop_and_a_loop_is() {
        let text = prose(3000);
        assert!(compress_ratio(&text) > 0.2, "prose scored {}", compress_ratio(&text));
        assert!(compress_ratio(&looping(3000)) < 0.05, "loop scored {}", compress_ratio(&looping(3000)));
    }

    #[test]
    fn the_window_is_scored_only_once_it_is_full_and_every_step_after() {
        let p = params(LoopGuardMode::Log);
        let text: Vec<char> = looping(4000).chars().collect();
        let mut watch = Watch::from(0, &p);
        assert_eq!(watch.check(&text[..2999], &p), None, "under one window");
        assert!(watch.check(&text[..3000], &p).is_some());
        assert_eq!(watch.check(&text[..3100], &p), None, "within the step");
        assert!(watch.check(&text[..3250], &p).is_some());
    }

    #[test]
    fn nothing_before_the_floor_is_scored() {
        let p = params(LoopGuardMode::Log);
        let text: Vec<char> = (prose(1000) + &looping(3000)).chars().collect();
        let mut watch = Watch::from(1000, &p);
        assert_eq!(watch.check(&text[..3900], &p), None, "the window would start before the floor");
        assert!(watch.check(&text[..4000], &p).is_some());
    }

    #[test]
    fn the_cut_lands_before_the_loop_on_a_line_break() {
        let p = params(LoopGuardMode::Rewind);
        let head = prose(6000);
        let text: Vec<char> = (head.clone() + &looping(5000)).chars().collect();
        let mut watch = Watch::from(0, &p);
        let at = (3000..=text.len()).find(|&n| watch.check(&text[..n], &p).is_some()).expect("found");
        let cut = cut_point(&text, at, 0, &p);
        assert!(cut <= 6000, "cut {cut} is inside the loop (it starts at 6000)");
        assert!(cut >= 6000 - 4500, "cut {cut} took back far more than the loop");
        assert_eq!(text[cut - 1], '\n', "not on a line break");
    }

    #[test]
    fn the_cut_never_crosses_the_floor() {
        let p = params(LoopGuardMode::Rewind);
        let text: Vec<char> = looping(9000).chars().collect();
        assert_eq!(cut_point(&text, 9000, 5000, &p), 5000);
    }

    #[test]
    fn the_tag_says_what_was_taken_back() {
        assert_eq!(tag(812, 3104), "\n\n[LOOP DETECTED, REWINDING 812 TOKENS / 3104 CHARS]\n\n");
        assert!(tag(1, 2).contains(TAG_PREFIX));
    }

    /// A stand-in engine: records every cancel and submission, and answers a
    /// submission with the next scripted stream.
    struct FakeOps {
        cancelled: Mutex<Vec<RequestId>>,
        submitted: Mutex<Vec<RequestInput>>,
        streams: Mutex<Vec<(RequestId, EventStream)>>,
    }

    impl GuardOps for Arc<FakeOps> {
        fn cancel(&self, request: RequestId) {
            self.cancelled.lock().unwrap().push(request);
        }
        fn submit(&self, input: RequestInput) -> Pin<Box<dyn Future<Output = Option<(RequestId, EventStream)>> + Send + '_>> {
            self.submitted.lock().unwrap().push(input);
            let next = self.streams.lock().unwrap().pop();
            Box::pin(async move { next })
        }
        fn decoder(&self) -> OutputDecoder {
            OutputDecoder::new(Box::new(CharDecoder), true)
        }
        fn encode(&self, text: &str) -> Option<Vec<TokenId>> {
            Some(tokens_of(text))
        }
    }

    fn scripted(request: RequestId, text: &str, done: bool) -> EventStream {
        let (tx, rx) = unbounded_channel();
        for token in tokens_of(text) {
            tx.send(SchedEvent::Token { request, token }).unwrap();
        }
        if done {
            tx.send(done_event(request, text.chars().count() as u32)).unwrap();
        }
        rx
    }

    fn done_event(request: RequestId, tokens: u32) -> SchedEvent {
        SchedEvent::Done {
            request,
            tokens,
            reason: FinishReason::Stop,
            spec: None,
            readout: None,
            attention: None,
            drawn: None,
            thinking: None,
            reasoning_redirected_at: None,
        }
    }

    fn input(prompt: &str, max_tokens: u32) -> RequestInput {
        RequestInput {
            decision: None,
            constrained: None,
            forced_literal: None,
            warm_up: false,
            multimodal: None,
            opener_tokens: None,
            user_turn_tokens: None,
            system_block_tokens: None,
            reuse_boundaries: Vec::new(),
            model: "m".into(),
            tokens: tokens_of(prompt),
            params: DecodeParams {
                max_tokens: Some(max_tokens),
                thinking_budget: Some(max_tokens - 2048),
                starts_in_reasoning: true,
                seed: 7,
                ..DecodeParams::default()
            },
        }
    }

    async fn drain(mut rx: EventStream) -> (String, Option<SchedEvent>) {
        let mut tokens = Vec::new();
        let mut done = None;
        while let Some(event) = rx.recv().await {
            match event {
                SchedEvent::Token { token, request } => {
                    assert_eq!(request, 1, "a token under another id than the client's");
                    tokens.push(token);
                }
                event @ SchedEvent::Done { .. } => done = Some(event),
                _ => {}
            }
        }
        (text_of(&tokens), done)
    }

    #[tokio::test]
    async fn off_hands_the_stream_back_untouched() {
        let ops = Arc::new(FakeOps { cancelled: Mutex::default(), submitted: Mutex::default(), streams: Mutex::default() });
        let (text, done) = drain(guard(params(LoopGuardMode::Off), ops.clone(), input("p", 40_000), 1, scripted(1, "abc", true))).await;
        assert_eq!(text, "abc");
        assert!(done.is_some());
    }

    #[tokio::test]
    async fn log_mode_passes_a_loop_through_and_acts_on_nothing() {
        let ops = Arc::new(FakeOps { cancelled: Mutex::default(), submitted: Mutex::default(), streams: Mutex::default() });
        let reasoning = prose(2000) + &looping(6000);
        let (text, done) =
            drain(guard(params(LoopGuardMode::Log), ops.clone(), input("p", 40_000), 1, scripted(1, &reasoning, true))).await;
        assert_eq!(text, reasoning);
        assert!(done.is_some());
        assert!(ops.cancelled.lock().unwrap().is_empty());
        assert!(ops.submitted.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_loop_is_rewound_nudged_and_continued() {
        let after = "\nRome's story, written.</think>\n\nRoma.";
        let ops = Arc::new(FakeOps {
            cancelled: Mutex::default(),
            submitted: Mutex::default(),
            streams: Mutex::new(vec![(2, scripted(2, after, true))]),
        });
        let head = prose(5000);
        let reasoning = head.clone() + &looping(6000);
        let (text, done) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("PROMPT", 40_000), 1, scripted(1, &reasoning, false)))
                .await;

        assert_eq!(*ops.cancelled.lock().unwrap(), vec![1], "the looping request is cancelled");
        let submitted = ops.submitted.lock().unwrap();
        let next = &submitted[0];
        let continued = text_of(&next.tokens);
        let kept = continued.strip_prefix("PROMPT").expect("the prompt first").strip_suffix(NUDGES[0]).expect("the nudge last");
        assert!(head.starts_with(kept), "the continuation keeps a prefix of the reasoning before the loop");
        assert!(kept.len() > 1000, "kept only {} chars", kept.len());
        let used = (kept.chars().count() + NUDGES[0].chars().count()) as u32;
        assert_eq!(next.params.max_tokens, Some(40_000 - used));
        assert_eq!(next.params.thinking_budget, Some(40_000 - 2048 - used));
        assert!(next.params.starts_in_reasoning);
        assert_ne!(next.params.seed, 7, "a fresh draw");

        // The client: everything it was sent, the tag, the nudge, then the continuation.
        let (sent, rest) = text.split_once("\n\n[LOOP DETECTED, REWINDING ").expect("the tag");
        let (counts, rest) = rest.split_once("]\n\n").expect("the tag's end");
        assert!(reasoning.starts_with(sent));
        let struck = sent.chars().count() - kept.chars().count();
        assert_eq!(counts, format!("{} TOKENS / {struck} CHARS", struck), "one token per char here");
        assert_eq!(rest, format!("{}{after}", NUDGES[0]));
        let Some(SchedEvent::Done { request, tokens, .. }) = done else { panic!("no done") };
        assert_eq!(request, 1);
        assert_eq!(tokens as usize, text.chars().count(), "every token the client received");
    }

    #[tokio::test]
    async fn after_the_retries_the_loop_is_closed() {
        let ops = Arc::new(FakeOps {
            cancelled: Mutex::default(),
            submitted: Mutex::default(),
            streams: Mutex::new(vec![(2, scripted(2, "Roma.", true))]),
        });
        let reasoning = prose(5000) + &looping(6000);
        let p = LoopGuardParams { retries: 0, ..params(LoopGuardMode::Rewind) };
        let (text, _) = drain(guard(p, ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        let submitted = ops.submitted.lock().unwrap();
        let next = &submitted[0];
        assert!(text_of(&next.tokens).ends_with(crate::thinking::THINKING_CLOSE_TEXT));
        assert_eq!(next.params.thinking_budget, None);
        assert!(!next.params.starts_in_reasoning, "the prompt closed the reasoning");
        assert!(text.ends_with(&format!("{}Roma.", crate::thinking::THINKING_CLOSE_TEXT)));
    }

    #[tokio::test]
    async fn a_client_that_leaves_cancels_the_continuation() {
        let (cont_tx, cont_rx) = unbounded_channel();
        let ops = Arc::new(FakeOps {
            cancelled: Mutex::default(),
            submitted: Mutex::default(),
            streams: Mutex::new(vec![(2, cont_rx)]),
        });
        let reasoning = prose(5000) + &looping(6000);
        let mut rx = guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false));
        // Read until the tag, then leave.
        let mut seen = String::new();
        while !seen.contains(TAG_PREFIX) {
            if let Some(SchedEvent::Token { token, .. }) = rx.recv().await {
                seen.push(char::from_u32(token).unwrap());
            }
        }
        drop(rx);
        cont_tx.send(SchedEvent::Token { request: 2, token: u32::from('x') }).unwrap();
        for _ in 0..100 {
            if ops.cancelled.lock().unwrap().contains(&2) {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("the continuation was never cancelled: {:?}", ops.cancelled.lock().unwrap());
    }
}
