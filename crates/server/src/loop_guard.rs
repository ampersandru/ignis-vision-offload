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
//! zlib ratio of the last `window` characters of the reasoning as
//! [`Normalized`] scores it — code taken out, every run of digits read as one
//! `0` — under `ratio` is a loop, when the last `window` characters as
//! written are under `raw_gate` too. Measured offline on 2026-10-13 with this
//! detector (`.scratch/loop-rewind/NOTES.md`): no real reasoning part of
//! 1,420 under 0.27 at a 1,000-character window, and 12 of the 17 template
//! and counting loops the raw text at 3,000 and 0.12 missed under 0.20. The
//! gate is for tables of numbers (Fibonacci, hex, primes), a template once
//! their digits fold, whose digits carry the information as written: on 47M
//! characters of qwen-code reasoning it takes 5 of 7 such fires away and no
//! harmful loop (`.scratch/loop-rewind/research/REPORT.md`).
//!
//! **Rewind.** The engine request is cancelled and a **continuation**
//! submitted: the same input, its prompt extended by the generated tokens
//! kept from before the loop ([`Watch::cut_point`]: back to the start of the
//! last window that scored clean) and a nudge sentence, with what
//! is left of `max_tokens` and of the thinking budget. The whole
//! continuation prompt prefills (it is submitted without a `reuse_source`;
//! reusing the original's state is a follow-up, ADR 0052 Consequences). The
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
use std::sync::atomic::{AtomicU32, Ordering};
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

/// `--model-loop-guard-ratio`'s default, in thousandths: 0.20.
pub const DEFAULT_RATIO_PERMILLE: u32 = 200;
/// `--model-loop-guard-window`'s default, in characters of scored reasoning.
pub const DEFAULT_WINDOW_CHARS: u32 = 1000;
/// `--model-loop-guard-normalize`'s default: the reasoning is scored
/// [`Normalized`].
pub const DEFAULT_NORMALIZE: bool = true;
/// `--model-loop-guard-raw-gate`'s default, in thousandths: 0.30.
pub const DEFAULT_RAW_GATE_PERMILLE: u32 = 300;
/// `--model-loop-guard-clean`'s default, in thousandths: 0.40. The longest
/// run of 1,000-character windows under it in 47 clean Rome runs spans
/// 2,750 characters; going back to the last window over it, a rewind cuts
/// before the onset of each of the 33 labelled loops the detector catches.
pub const DEFAULT_CLEAN_PERMILLE: u32 = 400;
/// `--model-loop-guard-margin`'s default, in characters before the clean
/// window: none.
pub const DEFAULT_MARGIN_CHARS: u32 = 0;
/// `--model-loop-guard-retries`'s default: nudges before the close.
pub const DEFAULT_RETRIES: u32 = 2;

/// The guard's knobs, as the live config holds them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopGuardParams {
    pub mode: LoopGuardMode,
    /// A window whose zlib ratio is under this many thousandths is a loop.
    pub ratio_permille: u32,
    /// The window, in characters of scored reasoning.
    pub window_chars: u32,
    /// Score the reasoning [`Normalized`]; off, as written.
    pub normalize: bool,
    /// A loop's last `window` raw characters must also score under this
    /// many thousandths; 0, no gate.
    pub raw_gate_permille: u32,
    /// A window scoring at least this many thousandths is clean: a rewind
    /// goes back to the start of the last one before the loop.
    pub clean_permille: u32,
    /// How many characters before that window's start a rewind cuts.
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
            normalize: DEFAULT_NORMALIZE,
            raw_gate_permille: DEFAULT_RAW_GATE_PERMILLE,
            clean_permille: DEFAULT_CLEAN_PERMILLE,
            margin_chars: DEFAULT_MARGIN_CHARS,
            retries: DEFAULT_RETRIES,
        }
    }
}

/// How often the window is scored, in characters of new reasoning.
pub const CHECK_EVERY_CHARS: usize = 250;

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
/// that it never joins a character the last token left incomplete; what
/// follows it (a nudge, the close) opens with its own.
pub fn tag(tokens: usize, chars: usize) -> String {
    format!("\n\n{TAG_PREFIX}{tokens} TOKENS / {chars} CHARS]")
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

/// The reasoning as the detector scores it, built one character at a time
/// as it streams. With `normalize` on, fenced code blocks (one still open
/// too: code being written never counts) and inline code are taken out and
/// every run of ASCII digits reads as one `0`: a template loop whose slots
/// are numbers ("the 321 AD, the 322 AD") repeats verbatim here, while code,
/// which repeats by nature, is not scored. Off, it is the reasoning as
/// written.
///
/// A character is kept only once it is decided — a backtick or two that may
/// still open a block, an inline span that may still close on its line,
/// wait — and keeps the offset of the raw character it stands for (a run of
/// digits, its first), so a window found here is cut on the raw reasoning.
/// The steps are those of `.scratch/loop-rewind/check.py` in its order:
/// blocks, then inline code on what is left, then digits. The raw reasoning
/// is kept beside it, for the raw gate and the cut.
pub struct Normalized {
    steps: Vec<Box<dyn Step + Send>>,
    text: Vec<char>,
    raw_at: Vec<usize>,
    raw: Vec<char>,
    /// What one step hands the next.
    pass: Vec<(char, usize)>,
    spare: Vec<(char, usize)>,
}

impl Normalized {
    /// Off, every character is kept as written.
    pub fn new(normalize: bool) -> Self {
        let steps: Vec<Box<dyn Step + Send>> = if normalize {
            vec![Box::new(Fences::default()), Box::new(InlineCode::default()), Box::new(Digits::default())]
        } else {
            Vec::new()
        };
        Self { steps, text: Vec::new(), raw_at: Vec::new(), raw: Vec::new(), pass: Vec::new(), spare: Vec::new() }
    }

    /// The next character of the raw reasoning.
    pub fn push(&mut self, c: char) {
        self.pass.clear();
        self.pass.push((c, self.raw.len()));
        self.raw.push(c);
        for step in &mut self.steps {
            self.spare.clear();
            for &(c, raw) in &self.pass {
                step.push(c, raw, &mut self.spare);
            }
            std::mem::swap(&mut self.pass, &mut self.spare);
        }
        for &(c, raw) in &self.pass {
            self.text.push(c);
            self.raw_at.push(raw);
        }
    }

    /// The scored text so far.
    pub fn text(&self) -> &[char] {
        &self.text
    }

    /// The raw character the scored character `i` stands for.
    pub fn raw_at(&self, i: usize) -> usize {
        self.raw_at[i]
    }

    /// The raw reasoning so far.
    pub fn raw(&self) -> &[char] {
        &self.raw
    }

    /// The raw characters pushed so far.
    pub fn raw_len(&self) -> usize {
        self.raw.len()
    }
}

/// One step of [`Normalized`]: characters in, each with the raw offset it
/// stands for; out, what the next step sees. Another fold is one more step.
trait Step {
    fn push(&mut self, c: char, raw: usize, out: &mut Vec<(char, usize)>);
}

/// Fenced code blocks out, the fences with them: three backticks in a row
/// open a block, the next three close it (```` ```.*?``` ````). One or two
/// backticks outside a block are held until the next character says whether
/// they open one.
#[derive(Default)]
struct Fences {
    open: bool,
    /// The raw offset of the open block's first backtick.
    opened_at: usize,
    /// The raw offsets of the backticks in a row just seen.
    ticks: Vec<usize>,
}

impl Step for Fences {
    fn push(&mut self, c: char, raw: usize, out: &mut Vec<(char, usize)>) {
        if c == '`' {
            self.ticks.push(raw);
            if self.ticks.len() == 3 {
                self.open = !self.open;
                self.opened_at = self.ticks[0];
                self.ticks.clear();
            }
            return;
        }
        if !self.open {
            out.extend(self.ticks.iter().map(|&at| ('`', at)));
            out.push((c, raw));
        }
        self.ticks.clear();
    }
}

/// The first backtick of the code block `text` ends inside, if it does: by
/// [`Fences`]' own rule.
fn open_block(text: &[char]) -> Option<usize> {
    let mut fences = Fences::default();
    let mut out = Vec::new();
    for (at, &c) in text.iter().enumerate() {
        fences.push(c, at, &mut out);
        out.clear();
    }
    fences.open.then_some(fences.opened_at)
}

/// Inline code out: a backtick and what follows it on its line, up to and
/// with the next backtick (`` `[^`\n]*` ``). A line break before that one
/// makes the span text.
#[derive(Default)]
struct InlineCode {
    open: bool,
    span: Vec<(char, usize)>,
}

impl Step for InlineCode {
    fn push(&mut self, c: char, raw: usize, out: &mut Vec<(char, usize)>) {
        if !self.open {
            if c == '`' {
                self.open = true;
                self.span.push((c, raw));
            } else {
                out.push((c, raw));
            }
            return;
        }
        match c {
            '`' => {
                self.open = false;
                self.span.clear();
            }
            '\n' => {
                self.open = false;
                out.append(&mut self.span);
                out.push((c, raw));
            }
            _ => self.span.push((c, raw)),
        }
    }
}

/// Every run of ASCII digits as one `0`.
#[derive(Default)]
struct Digits {
    run: bool,
}

impl Step for Digits {
    fn push(&mut self, c: char, raw: usize, out: &mut Vec<(char, usize)>) {
        if c.is_ascii_digit() {
            if !self.run {
                out.push(('0', raw));
            }
            self.run = true;
        } else {
            self.run = false;
            out.push((c, raw));
        }
    }
}

/// A scored window of reasoning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// Its zlib ratio ([`compress_ratio`]).
    pub ratio: f64,
    /// The zlib ratio of the last `window` raw characters when it was taken.
    pub raw_ratio: f64,
    /// The raw reasoning character it starts at.
    pub start: usize,
    /// The raw reasoning length it was taken at: where it ends.
    pub end: usize,
}

impl Score {
    /// Under `ratio`, and as written under `raw_gate` (when there is one):
    /// a loop.
    pub fn is_loop(&self, params: &LoopGuardParams) -> bool {
        self.ratio * 1000.0 < f64::from(params.ratio_permille)
            && (params.raw_gate_permille == 0 || self.raw_ratio * 1000.0 < f64::from(params.raw_gate_permille))
    }

    /// At `clean` or over: no loop in it.
    pub fn is_clean(&self, params: &LoopGuardParams) -> bool {
        self.ratio * 1000.0 >= f64::from(params.clean_permille)
    }
}

/// When to score the window, from where a loop may be found, and what was
/// scored since.
#[derive(Debug, Clone)]
pub struct Watch {
    /// No window that starts before this raw character is scored: the end
    /// of the last nudge, or of the last logged loop.
    floor: usize,
    /// The raw reasoning length the next score waits for.
    next_check: usize,
    /// Every score taken since the floor, in order.
    history: Vec<Score>,
}

impl Watch {
    /// A watch whose first score is the first full window after `floor`.
    pub fn from(floor: usize, params: &LoopGuardParams) -> Self {
        Self { floor, next_check: floor + params.window_chars as usize, history: Vec::new() }
    }

    /// The window's score when the reasoning has grown enough to score it
    /// again: the first time a whole window of scored text lies after the
    /// floor, then every [`CHECK_EVERY_CHARS`] raw characters. Its raw
    /// window, the last `window` raw characters, lies after the floor too:
    /// the scored one spans at least as many.
    pub fn score(&mut self, scored: &Normalized, params: &LoopGuardParams) -> Option<Score> {
        if scored.raw_len() < self.next_check {
            return None;
        }
        let window = params.window_chars as usize;
        let start = scored.text().len().checked_sub(window)?;
        if scored.raw_at(start) < self.floor {
            return None;
        }
        self.next_check = scored.raw_len() + CHECK_EVERY_CHARS;
        let raw = scored.raw();
        let score = Score {
            ratio: ratio_of(&scored.text()[start..]),
            raw_ratio: ratio_of(&raw[raw.len() - window..]),
            start: scored.raw_at(start),
            end: raw.len(),
        };
        self.history.push(score);
        Some(score)
    }

    /// [`Watch::score`], when it is a loop.
    fn check(&mut self, scored: &Normalized, params: &LoopGuardParams) -> Option<Score> {
        self.score(scored, params).filter(|score| score.is_loop(params))
    }

    /// Where to take the raw reasoning `text` back to, for the loop the last
    /// score found: the start of the last window before it that scored
    /// clean, `margin` characters earlier, then the line break before that
    /// when one is within [`SNAP_CHARS`]; with no clean window since the
    /// floor, the floor. Never before the floor.
    ///
    /// A cut inside a code block moves to just before the block's opening
    /// fence: an open block is never scored, so the continuation would go
    /// unwatched until the model closed it. (Unless the block opened before
    /// the floor, which only a loop logged inside one leaves.)
    pub fn cut_point(&self, text: &[char], params: &LoopGuardParams) -> usize {
        let before = &self.history[..self.history.len().saturating_sub(1)];
        let cut = match before.iter().rev().find(|score| score.is_clean(params)) {
            Some(clean) => clean.start.saturating_sub(params.margin_chars as usize).max(self.floor),
            None => self.floor,
        };
        let low = cut.saturating_sub(SNAP_CHARS).max(self.floor);
        let cut = match text[low..cut].iter().rposition(|&c| c == '\n') {
            Some(i) => low + i + 1,
            None => cut,
        };
        match open_block(&text[..cut]) {
            Some(fence) => fence.max(self.floor),
            None => cut,
        }
    }
}

/// The generation as the model sees it: its tokens, the reasoning they
/// decoded to (as written and as scored), and the reasoning's length after
/// each token.
struct View {
    decoder: OutputDecoder,
    tokens: Vec<TokenId>,
    chars_after: Vec<usize>,
    scored: Normalized,
}

impl View {
    fn new(decoder: OutputDecoder, normalize: bool) -> Self {
        Self { decoder, tokens: Vec::new(), chars_after: Vec::new(), scored: Normalized::new(normalize) }
    }

    fn push(&mut self, token: TokenId) {
        for delta in self.decoder.push(&[token]) {
            if delta.channel == Channel::Reasoning {
                for c in delta.text.chars() {
                    self.scored.push(c);
                }
            }
        }
        self.tokens.push(token);
        self.chars_after.push(self.scored.raw_len());
    }

    /// The reasoning as written.
    fn text(&self) -> &[char] {
        self.scored.raw()
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

/// The generated tokens the guard's rewinds took back, shared with the
/// handler that builds `usage.completion_tokens_details.rewound_tokens`.
/// It counts the struck tokens only (those the client's tag reports as
/// `N TOKENS`): not the tag's own tokens, nor the nudges. The guard adds to
/// it before it sends the `Done` that ends the stream.
pub type Rewound = Arc<AtomicU32>;

/// Put the guard between `stream` (the engine's, for `request`, submitted
/// from `input`) and its consumer: the stream returned. `Off` returns
/// `stream` itself.
pub fn guard(params: LoopGuardParams, ops: impl GuardOps, input: RequestInput, request: RequestId, stream: EventStream) -> EventStream {
    guard_counted(params, ops, input, request, stream, Rewound::default())
}

/// [`guard`], adding the tokens each rewind takes back to `rewound`.
pub fn guard_counted(
    params: LoopGuardParams,
    ops: impl GuardOps,
    input: RequestInput,
    request: RequestId,
    stream: EventStream,
    rewound: Rewound,
) -> EventStream {
    if params.mode == LoopGuardMode::Off {
        return stream;
    }
    let (tx, rx) = unbounded_channel();
    tokio::spawn(run(params, ops, input, request, stream, tx, rewound));
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
    rewound: Rewound,
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
        let score = self.watch.check(&self.view.scored, &self.params)?;
        let at = self.view.text().len();
        let rewindable = self.params.mode == LoopGuardMode::Rewind && self.input.multimodal.is_none();
        if !rewindable {
            tracing::info!(
                name: "ignis.loop_guard.detected",
                request = self.original, engine_request = self.current, at_chars = at, ratio = score.ratio,
                raw_ratio = score.raw_ratio, window_start_chars = score.start,
                cut_chars = self.watch.cut_point(self.view.text(), &self.params),
                "loop in the reasoning (log only)"
            );
            self.watch = Watch::from(at, &self.params);
            return None;
        }
        self.rewind(at, score).await
    }

    async fn rewind(&mut self, at: usize, score: Score) -> Option<Continuation> {
        let ratio = score.ratio;
        let cut = self.watch.cut_point(self.view.text(), &self.params);
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
            at_chars = at, ratio = ratio, raw_ratio = score.raw_ratio, window_start_chars = score.start, cut_chars = kept_chars,
            rewound_tokens = self.view.tokens.len() - kept, rewound_chars = at - kept_chars,
            attempt = self.rewinds + 1, closed = close,
            "loop in the reasoning: rewound"
        );

        self.rewound.fetch_add((self.view.tokens.len() - kept) as u32, Ordering::Relaxed);

        // The model's view: the kept tokens, then the nudge.
        let mut view = View::new(self.ops.decoder(), self.params.normalize);
        for &token in self.view.tokens[..kept].iter().chain(&nudge) {
            view.push(token);
        }
        self.view = view;
        self.watch = Watch::from(self.view.text().len(), &self.params);
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
    rewound: Rewound,
) {
    let view = View::new(ops.decoder(), params.normalize);
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
        rewound,
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
    /// numbered, varied sentences. Its windows score 0.45-0.49 normalized,
    /// clean with room to spare, as real reasoning's mostly do.
    fn prose(chars: usize) -> String {
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

    fn looping(chars: usize) -> String {
        "the \"Dark Ages\" (the \"Dark Ages\", the \"Dark Ages\"), ".chars().cycle().take(chars).collect()
    }

    /// A template loop whose slots are numbers, the shape of the loops the
    /// raw score missed ("the 1527 sack, the 1348 plague").
    fn counting(chars: usize) -> String {
        let mut out = String::new();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        while out.chars().count() < chars {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push_str(&format!("the {} AD (the {}th year, page {}), ", 300 + state % 1700, state / 7 % 90 + 10, state / 11 % 9000));
        }
        out.chars().take(chars).collect()
    }

    /// A table of numbers, the layout of every line the same and every
    /// number new: a template once its digits fold, information as written.
    fn numeric_table(chars: usize) -> String {
        let mut out = String::new();
        let mut n: u64 = 1000;
        while out.chars().count() < chars {
            out.push_str(&format!("{n}^2 = {}, {n}^3 = {}\n", n * n, n * n * n));
            n += 7;
        }
        out.chars().take(chars).collect()
    }

    /// A code block's worth of lines that repeat, as test code does.
    fn code(chars: usize) -> String {
        "    assert_eq!(watch.check(&scored, &p), None);\n".chars().cycle().take(chars).collect()
    }

    fn params(mode: LoopGuardMode) -> LoopGuardParams {
        LoopGuardParams { mode, ..LoopGuardParams::default() }
    }

    fn raw(mode: LoopGuardMode) -> LoopGuardParams {
        LoopGuardParams { normalize: false, ..params(mode) }
    }

    /// `text` as the detector scores it.
    fn normalized(text: &str, normalize: bool) -> Normalized {
        let mut scored = Normalized::new(normalize);
        text.chars().for_each(|c| scored.push(c));
        scored
    }

    fn scored_text(text: &str) -> String {
        normalized(text, true).text().iter().collect()
    }

    /// Push `text` into `scored` up to raw character `n`.
    fn grow(scored: &mut Normalized, text: &[char], n: usize) {
        while scored.raw_len() < n {
            scored.push(text[scored.raw_len()]);
        }
    }

    /// The first loop the guard finds in `text`, looking after every
    /// character: the raw length it was found at, and its score.
    fn first_loop(text: &str, floor: usize, p: &LoopGuardParams) -> Option<(usize, Score)> {
        let mut scored = Normalized::new(p.normalize);
        let mut watch = Watch::from(floor, p);
        for c in text.chars() {
            scored.push(c);
            if let Some(score) = watch.check(&scored, p) {
                return Some((scored.raw_len(), score));
            }
        }
        None
    }

    /// The watch that found the first loop in `text`.
    fn watch_to_loop(text: &str, p: &LoopGuardParams) -> Watch {
        let mut scored = Normalized::new(p.normalize);
        let mut watch = Watch::from(0, p);
        for c in text.chars() {
            scored.push(c);
            if watch.check(&scored, p).is_some() {
                return watch;
            }
        }
        panic!("no loop in the text");
    }

    /// A watch from `floor` that took `checks`, each (window start, ratio),
    /// the last one the loop.
    fn watch_with(floor: usize, checks: &[(usize, f64)]) -> Watch {
        let history = checks.iter().map(|&(start, ratio)| Score { ratio, raw_ratio: ratio, start, end: start + 1000 }).collect();
        Watch { floor, next_check: 0, history }
    }

    /// Text for the cut to read: no line break, no backtick.
    fn flat(chars: usize) -> Vec<char> {
        vec!['x'; chars]
    }

    #[test]
    fn prose_is_not_a_loop_and_a_loop_is() {
        let text = prose(3000);
        assert!(compress_ratio(&text) > 0.2, "prose scored {}", compress_ratio(&text));
        assert!(compress_ratio(&looping(3000)) < 0.05, "loop scored {}", compress_ratio(&looping(3000)));
        for p in [params(LoopGuardMode::Log), raw(LoopGuardMode::Log)] {
            assert_eq!(first_loop(&prose(8000), 0, &p), None, "prose flagged with {p:?}");
        }
    }

    #[test]
    fn the_window_is_scored_only_once_it_is_full_and_every_step_after() {
        let p = params(LoopGuardMode::Log);
        let text: Vec<char> = looping(2000).chars().collect();
        let mut scored = Normalized::new(p.normalize);
        let mut watch = Watch::from(0, &p);
        grow(&mut scored, &text, 999);
        assert_eq!(watch.check(&scored, &p), None, "under one window");
        grow(&mut scored, &text, 1000);
        assert!(watch.check(&scored, &p).is_some());
        grow(&mut scored, &text, 1100);
        assert_eq!(watch.check(&scored, &p), None, "within the step");
        grow(&mut scored, &text, 1250);
        assert!(watch.check(&scored, &p).is_some());
    }

    /// Code is not scored text: the first score waits for a whole window
    /// of what is, however many raw characters that takes.
    #[test]
    fn the_first_score_waits_for_a_whole_window_of_scored_text() {
        let p = params(LoopGuardMode::Log);
        let block = format!("```rust\n{}```", code(2000));
        let text = block.clone() + &looping(2000);
        let (at, score) = first_loop(&text, 0, &p).expect("the loop after the block");
        let block_chars = block.chars().count();
        assert_eq!(at, block_chars + 1000, "scored as soon as 1,000 characters after the block are");
        assert_eq!(score.start, block_chars, "the window starts where the block ends");
    }

    #[test]
    fn nothing_before_the_floor_is_scored() {
        let p = params(LoopGuardMode::Log);
        let text: Vec<char> = (prose(1000) + &looping(3000)).chars().collect();
        let mut scored = Normalized::new(p.normalize);
        let mut watch = Watch::from(1000, &p);
        grow(&mut scored, &text, 1999);
        assert_eq!(watch.check(&scored, &p), None, "the window would start before the floor");
        grow(&mut scored, &text, 2000);
        assert_eq!(watch.check(&scored, &p).map(|score| score.start), Some(1000));
    }

    /// The floor is raw: a window of scored text after it is a window
    /// whose first character comes after it, not `window` raw characters.
    #[test]
    fn a_window_reaching_back_past_the_floor_through_code_is_not_scored() {
        let p = params(LoopGuardMode::Log);
        let after = format!("```\n{}```", code(500)) + &looping(3000);
        let text = prose(1000) + &after;
        let (at, score) = first_loop(&text, 1000, &p).expect("found after the floor");
        let block_chars = after.chars().count() - 3000;
        assert_eq!(score.start, 1000 + block_chars);
        assert_eq!(at, 1000 + block_chars + 1000);
    }

    /// On a stream: the last window before the loop that scored clean is
    /// the reasoning just before it, and the cut is at its start or on the
    /// line break just before.
    #[test]
    fn the_cut_lands_before_the_loop_at_the_last_clean_windows_start() {
        let p = params(LoopGuardMode::Rewind);
        let text = prose(6000) + &looping(5000);
        let watch = watch_to_loop(&text, &p);
        let fire = *watch.history.last().expect("the loop's score");
        let clean = *watch.history.iter().rev().skip(1).find(|score| score.is_clean(&p)).expect("a clean window");
        assert!(clean.start < 6000 && clean.end >= 6000 && fire.end > clean.end, "clean {clean:?}, loop {fire:?}");
        let cut = watch.cut_point(&text.chars().collect::<Vec<_>>(), &p);
        assert!(cut <= clean.start && clean.start - cut <= SNAP_CHARS, "cut {cut}, clean window from {}", clean.start);
    }

    #[test]
    fn the_cut_is_the_last_clean_windows_start_less_the_margin() {
        let p = params(LoopGuardMode::Rewind);
        let text = flat(4000);
        let watch = watch_with(0, &[(1000, 0.45), (1250, 0.41), (1500, 0.38), (1750, 0.30), (2000, 0.15)]);
        assert_eq!(watch.cut_point(&text, &p), 1250);
        assert_eq!(watch.cut_point(&text, &LoopGuardParams { margin_chars: 100, ..p }), 1150);
        assert_eq!(watch.cut_point(&text, &LoopGuardParams { clean_permille: 420, ..p }), 1000);
        let lenient = LoopGuardParams { ratio_permille: 500, clean_permille: 100, ..p };
        assert_eq!(watch_with(0, &[(1000, 0.45), (1250, 0.42)]).cut_point(&text, &lenient), 1000, "the loop's own score is never the clean one");
    }

    #[test]
    fn the_cut_moves_back_to_a_line_break_within_300_characters() {
        let p = params(LoopGuardMode::Rewind);
        let watch = watch_with(0, &[(1000, 0.45), (1250, 0.15)]);
        let mut text = flat(4000);
        text[650] = '\n';
        assert_eq!(watch.cut_point(&text, &p), 1000, "the line break is 350 characters back");
        text[750] = '\n';
        text[800] = '\n';
        assert_eq!(watch.cut_point(&text, &p), 801, "after the last line break before it");
    }

    #[test]
    fn the_cut_never_crosses_the_floor() {
        let p = params(LoopGuardMode::Rewind);
        let mut text = flat(4000);
        text[1100] = '\n';
        let watch = watch_with(1200, &[(1300, 0.45), (1550, 0.15)]);
        assert_eq!(watch.cut_point(&text, &p), 1300, "the line break is before the floor");
        assert_eq!(watch.cut_point(&text, &LoopGuardParams { margin_chars: 500, ..p }), 1200, "the margin stops at the floor");
    }

    #[test]
    fn with_no_clean_window_since_the_floor_the_cut_is_the_floor() {
        let p = params(LoopGuardMode::Rewind);
        let mut text = flat(4000);
        text[600] = '\n';
        let watch = watch_with(700, &[(700, 0.39), (950, 0.25), (1200, 0.12)]);
        assert_eq!(watch.cut_point(&text, &p), 700);
    }

    /// A cut that would keep a code block open moves to just before its
    /// opening fence: the continuation is scored at once, not once the
    /// model closes the block.
    #[test]
    fn a_cut_inside_a_code_block_moves_before_its_fence() {
        let p = params(LoopGuardMode::Rewind);
        // A block from 900 (its fence) to 1411, a line break there.
        let text: Vec<char> =
            format!("{}```rust\n{}```\n{}", "x".repeat(900), code(500), "y".repeat(2000)).chars().collect();
        assert_eq!((&text[900..903], text[1411]), (&['`'; 3][..], '\n'));
        let after_the_block = watch_with(0, &[(1500, 0.45), (1750, 0.15)]);
        assert_eq!(after_the_block.cut_point(&text, &p), 1412, "a block closed before the cut stays");
        let on_its_line_break = watch_with(0, &[(1411, 0.45), (1661, 0.15)]);
        assert_eq!(on_its_line_break.cut_point(&text, &p), 900, "snapped into the block");
        let margin = LoopGuardParams { margin_chars: 300, ..p };
        assert_eq!(after_the_block.cut_point(&text, &margin), 900, "taken back into the block");
        let floor_in_the_block = watch_with(1000, &[(1500, 0.45), (1750, 0.15)]);
        assert_eq!(floor_in_the_block.cut_point(&text, &margin), 1000, "never before the floor");
    }

    /// The raw gate: a table of numbers is a loop to the normalized score
    /// alone, not as written; a counting loop is one both ways.
    #[test]
    fn a_numeric_table_is_not_a_loop_through_the_raw_gate_and_a_counting_loop_is() {
        let p = params(LoopGuardMode::Log);
        let ungated = LoopGuardParams { raw_gate_permille: 0, ..p };
        let table = prose(3000) + &numeric_table(4000);
        let (at, score) = first_loop(&table, 0, &ungated).expect("ungated, the folded table is a loop");
        assert!(at > 3000 && score.raw_ratio >= 0.30, "flagged at {at}: {score:?}");
        assert_eq!(first_loop(&table, 0, &p), None, "the gate lets the table through");
        let counted = prose(3000) + &counting(4000);
        let (at, score) = first_loop(&counted, 0, &p).expect("a counting loop is one through the gate");
        assert!(at > 3000 && score.raw_ratio < 0.30, "flagged at {at}: {score:?}");
    }

    /// A template loop over numbers compresses like prose as written, and
    /// repeats verbatim once its digits fold.
    #[test]
    fn a_counting_loop_is_a_loop_only_normalized() {
        let loop_text = counting(1000);
        assert!(compress_ratio(&loop_text) > 0.215, "as written it must look like prose: {}", compress_ratio(&loop_text));
        let text = prose(3000) + &counting(4000);
        let (at, _) = first_loop(&text, 0, &params(LoopGuardMode::Log)).expect("normalized, a loop");
        assert!(at > 3000, "flagged at {at}, before the loop");
        assert_eq!(first_loop(&text, 0, &raw(LoopGuardMode::Log)), None, "as written, not one");
    }

    #[test]
    fn a_verbatim_loop_is_a_loop_either_way() {
        let text = prose(3000) + &looping(3000);
        for p in [params(LoopGuardMode::Log), raw(LoopGuardMode::Log)] {
            let (at, _) = first_loop(&text, 0, &p).expect("a loop");
            assert!(at > 3000, "flagged at {at}, before the loop, with {p:?}");
        }
    }

    #[test]
    fn code_in_a_block_is_never_scored_closed_or_still_open() {
        let words = prose(3000);
        let (before, after) = words.split_at(1500);
        let closed = format!("{before}\n```rust\n{}```\n{after}", code(6000));
        let open = format!("{before}\n```python\n{}", code(6000));
        for text in [&closed, &open] {
            assert!(first_loop(text, 0, &raw(LoopGuardMode::Log)).is_some(), "the code alone would flag");
            assert_eq!(first_loop(text, 0, &params(LoopGuardMode::Log)), None, "code was scored");
        }
        assert_eq!(scored_text(&closed), scored_text(&format!("{before}\n\n{after}")));
        assert_eq!(scored_text(&open), scored_text(&format!("{before}\n")), "an open block is not scored as it is written");
    }

    #[test]
    fn fences_follow_the_first_three_backticks_in_a_row() {
        assert_eq!(scored_text("a```code```b"), "ab");
        assert_eq!(scored_text("a````x```b"), "ab", "the fourth backtick is inside the block");
        assert_eq!(scored_text("a```x`````b c"), "ab c", "the two backticks after the close are an empty inline span");
        assert_eq!(scored_text("a```b```c```d"), "ac", "the third fence opens a block still open");
        assert_eq!(scored_text("x ``"), "x ", "two backticks wait for a third");
    }

    #[test]
    fn inline_code_is_taken_out_on_its_line() {
        assert_eq!(scored_text("a `x` b `` c `d\ne` f"), "a  b  c `d\ne", "the span still open waits");
        assert_eq!(scored_text("a `x` b `` c `d\ne` f\n"), "a  b  c `d\ne` f\n", "a line break makes it text");
        assert_eq!(scored_text("a `x ```code``` y` b"), "a  b", "a block is out first, then the span around it");
    }

    #[test]
    fn every_run_of_digits_reads_as_one_zero() {
        assert_eq!(scored_text("the 321 AD, the 322 AD; 1.5e10"), "the 0 AD, the 0 AD; 0.0e0");
        assert_eq!(scored_text("12`x`34 5"), "0 0", "a span taken out joins the run");
        assert_eq!(scored_text("١٢ x"), "١٢ x", "ASCII digits only");
    }

    #[test]
    fn off_scores_the_reasoning_as_written() {
        let text = "a `x` 12 ```c```\n";
        let scored = normalized(text, false);
        assert_eq!(scored.text().iter().collect::<String>(), text);
        assert!((0..scored.text().len()).all(|i| scored.raw_at(i) == i));
    }

    /// Each scored character maps to the raw character it stands for, in
    /// Unicode scalars (the decoder's unit): itself, or a run's first digit.
    #[test]
    fn each_scored_character_maps_to_its_raw_character() {
        let text = "è1é23 —`x` ```c``` 🦀4\n";
        let raw_chars: Vec<char> = text.chars().collect();
        let scored = normalized(text, true);
        assert_eq!(scored.text().iter().collect::<String>(), "è0é0 —  🦀0\n");
        assert_eq!(scored.raw_len(), raw_chars.len());
        let raw_at: Vec<usize> = (0..scored.text().len()).map(|i| scored.raw_at(i)).collect();
        assert_eq!(raw_at, [0, 1, 2, 3, 5, 6, 10, 18, 19, 20, 21]);
        for (i, &c) in scored.text().iter().enumerate() {
            let at = scored.raw_at(i);
            if c == '0' {
                assert!(raw_chars[at].is_ascii_digit() && (at == 0 || !raw_chars[at - 1].is_ascii_digit()), "{i}");
            } else {
                assert_eq!(raw_chars[at], c, "{i}");
            }
        }
    }

    /// What the scored text of `raw` must be, worked out on the whole text
    /// at once the way `.scratch/loop-rewind/check.py`'s expressions do it,
    /// less what its end leaves undecided: blocks out (one never closed, to
    /// the end; one or two backticks at the very end wait), then on each
    /// line backticks paired in order with what is between them (an odd one
    /// on the last line waits with the rest of it), then digit runs.
    fn reference(raw: &str) -> String {
        let chars: Vec<char> = raw.chars().collect();
        let fence = |i: usize| chars.len() >= i + 3 && chars[i..i + 3] == ['`'; 3];
        let mut blocks_out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if fence(i) {
                i = (i + 3..chars.len()).find(|&j| fence(j)).map_or(chars.len(), |j| j + 3);
            } else {
                blocks_out.push(chars[i]);
                i += 1;
            }
        }
        if raw.ends_with('`') {
            while blocks_out.last() == Some(&'`') {
                blocks_out.pop();
            }
        }
        let text: String = blocks_out.into_iter().collect();
        let lines: Vec<&str> = text.split('\n').collect();
        let mut spans_out = Vec::new();
        for (n, line) in lines.iter().enumerate() {
            let ticks: Vec<usize> = line.char_indices().filter(|&(_, c)| c == '`').map(|(i, _)| i).collect();
            let mut kept = String::new();
            let mut from = 0;
            for pair in ticks.chunks(2) {
                match *pair {
                    [open, close] => {
                        kept.push_str(&line[from..open]);
                        from = close + 1;
                    }
                    [odd] if n + 1 == lines.len() => {
                        kept.push_str(&line[from..odd]);
                        from = line.len();
                    }
                    _ => {}
                }
            }
            kept.push_str(&line[from..]);
            spans_out.push(kept);
        }
        let mut out = String::new();
        let mut run = false;
        for c in spans_out.join("\n").chars() {
            if c.is_ascii_digit() {
                if !run {
                    out.push('0');
                }
                run = true;
            } else {
                run = false;
                out.push(c);
            }
        }
        out
    }

    /// Every prefix of every text, so every way a token can end inside a
    /// fence or a span: the scored text as it streams is the reference's.
    #[test]
    fn the_streamed_text_is_the_whole_texts_at_every_prefix() {
        let mut texts = vec![
            "a```rust\nfn f() {}\n```b `x` c 12 `d\ne`` ```".to_owned(),
            "x``y```z````w`````v``````u\n`1`2`3`\n``\n".to_owned(),
        ];
        let alphabet = ['`', '`', '`', 'a', '1', '2', '\n', 'é', ' ', '🦀'];
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        for _ in 0..300 {
            let mut text = String::new();
            for _ in 0..40 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                text.push(alphabet[(state % alphabet.len() as u64) as usize]);
            }
            texts.push(text);
        }
        for text in &texts {
            let mut scored = Normalized::new(true);
            let mut prefix = String::new();
            for c in text.chars() {
                scored.push(c);
                prefix.push(c);
                assert_eq!(scored.text().iter().collect::<String>(), reference(&prefix), "prefix {prefix:?}");
            }
        }
    }

    #[test]
    fn the_tag_says_what_was_taken_back() {
        assert_eq!(tag(812, 3104), "\n\n[LOOP DETECTED, REWINDING 812 TOKENS / 3104 CHARS]");
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
        let (counts, rest) = rest.split_once(']').expect("the tag's end");
        assert!(reasoning.starts_with(sent));
        let struck = sent.chars().count() - kept.chars().count();
        assert_eq!(counts, format!("{} TOKENS / {struck} CHARS", struck), "one token per char here");
        assert_eq!(rest, format!("{}{after}", NUDGES[0]));
        let Some(SchedEvent::Done { request, tokens, .. }) = done else { panic!("no done") };
        assert_eq!(request, 1);
        assert_eq!(tokens as usize, text.chars().count(), "every token the client received");
    }

    /// The count the handler reports: the struck tokens of every rewind,
    /// not the tags or the nudges.
    #[tokio::test]
    async fn the_rewound_count_is_the_struck_tokens_of_every_rewind() {
        let ops = Arc::new(FakeOps {
            cancelled: Mutex::default(),
            submitted: Mutex::default(),
            streams: Mutex::new(vec![(3, scripted(3, "Roma.", true)), (2, scripted(2, &looping(4000), false))]),
        });
        let reasoning = prose(5000) + &looping(6000);
        let rewound = Rewound::default();
        let (text, _) = drain(guard_counted(
            params(LoopGuardMode::Rewind),
            ops.clone(),
            input("P", 40_000),
            1,
            scripted(1, &reasoning, false),
            rewound.clone(),
        ))
        .await;
        // One token per char here: each tag reports its struck tokens.
        let tags: u32 = text
            .split("[LOOP DETECTED, REWINDING ")
            .skip(1)
            .map(|rest| rest.split_once(" TOKENS").unwrap().0.parse::<u32>().unwrap())
            .sum();
        assert_eq!(ops.submitted.lock().unwrap().len(), 2);
        assert!(tags > 0);
        assert_eq!(rewound.load(Ordering::Relaxed), tags);
    }

    /// After a rewind the watch starts again at the nudge's end: a loop
    /// straight after it has no clean window since, and is cut there, the
    /// first continuation kept whole.
    #[tokio::test]
    async fn the_watch_starts_again_after_a_rewind() {
        let ops = Arc::new(FakeOps {
            cancelled: Mutex::default(),
            submitted: Mutex::default(),
            streams: Mutex::new(vec![(3, scripted(3, "Roma.", true)), (2, scripted(2, &looping(4000), false))]),
        });
        let reasoning = prose(5000) + &looping(6000);
        let (text, _) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        assert_eq!(*ops.cancelled.lock().unwrap(), vec![1, 2]);
        let submitted = ops.submitted.lock().unwrap();
        assert_eq!(submitted.len(), 2, "two loops, two rewinds");
        let first = text_of(&submitted[0].tokens);
        assert!(first.ends_with(NUDGES[0]));
        assert_eq!(text_of(&submitted[1].tokens), format!("{first}{}", NUDGES[1]), "cut at the first nudge's end");
        assert!(text.ends_with(&format!("{}Roma.", NUDGES[1])));
    }

    /// On the normalized score: a counting loop after a code block is found,
    /// and the cut is made on the raw reasoning, after the block.
    #[tokio::test]
    async fn a_counting_loop_after_code_is_rewound_on_the_raw_text() {
        let ops = Arc::new(FakeOps {
            cancelled: Mutex::default(),
            submitted: Mutex::default(),
            streams: Mutex::new(vec![(2, scripted(2, "Roma.", true))]),
        });
        let words = prose(5500);
        let (before, after) = words.split_at(3000);
        let head = format!("{before}\n```rust\n{}```\n{after}", code(3000));
        let reasoning = head.clone() + &counting(4000);
        let (text, _) =
            drain(guard(params(LoopGuardMode::Rewind), ops.clone(), input("P", 40_000), 1, scripted(1, &reasoning, false))).await;
        assert!(text.contains(TAG_PREFIX), "the counting loop was not rewound");
        let submitted = ops.submitted.lock().unwrap();
        let continued = text_of(&submitted[0].tokens);
        let kept = continued.strip_prefix("P").expect("the prompt first").strip_suffix(NUDGES[0]).expect("the nudge last");
        assert!(head.starts_with(kept), "the cut is inside the loop");
        assert!(kept.len() > head.len() - after.len(), "the cut took the code block back too: kept {}", kept.len());
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
