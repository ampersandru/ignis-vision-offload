//! The `small-card` preset (spec 27b-16gb/05 Part B, solution 4, AC 7): one
//! command against a running server that writes ONE JSON a user can paste
//! into a GitHub issue, so phase 2 can collect real 16 GB numbers.
//!
//! It records the card (name, total VRAM, PCIe link), the load's VRAM plan
//! lines and the shape it ran with, decode tok/s on the coding prompts and on
//! agent-like turns, and TTFT at 8K and 64K. Everything it reads comes from
//! what a server already exposes:
//!
//! - the card: `nvidia-smi` beside the server (no server route reports it);
//! - the plan: the metrics listener's `ignis_vram_reserved_bytes` lines and
//!   `ignis_vram_budget_bytes`, the KV pool gauges, the build version, and
//!   the speculative counters read before and after the decode phase;
//! - the shape: `GET /v1/models` (the context the load runs), `GET /v1/config`
//!   (a whitelist of its values) and `GET /v1/config/origins` (where each came
//!   from);
//! - the prompts' length: `POST /v1/tokenize`, so a user needs no artifact.
//!
//! A piece that cannot be read is `null` in the JSON with a line in `notes`,
//! never a crash: a user without `nvidia-smi` on the path, or with the
//! metrics listener off, still gets a record of the rest.
//!
//! The record never carries the endpoint's credentials: the bench sends no
//! authorization header (a server started with an API key is out of reach of
//! this preset), the endpoint is written without any userinfo, and the config
//! is a whitelist of field names, never the whole document (it holds paths).
//!
//! Pure assembly (parsers, views, the decode statistics) is unit-tested here;
//! [`run`] drives the whole thing and is tested against a mock server in
//! `tests/small_card.rs`.

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};

use crate::canary::CANARIES;
use crate::client::{Endpoint, HttpEndpoint, Request};
use crate::time::{unix_now, utc_timestamp};
use crate::trace::RequestClass;
use crate::ttft::{self, CellSpec, PromptTemplate};

/// The record's schema tag: bumped when a field changes meaning.
pub const SCHEMA: &str = "ignis-small-card/1";

/// The TTFT cells by default: 8K and 64K prompt tokens.
pub const DEFAULT_CELLS: [u32; 2] = [8192, 65536];
/// Samples per TTFT cell after the warmup: three keeps the run in minutes on
/// a 16 GB card, where a 64K prefill is tens of seconds.
pub const DEFAULT_SAMPLES: usize = 3;
/// The output budget of a decode request.
pub const DEFAULT_DECODE_TOKENS: u32 = 256;
/// The metrics listener's default port (`DEFAULT_METRICS_BIND`).
pub const DEFAULT_METRICS_PORT: u16 = 9464;

/// The synthetic agent-like turns: system + tool-ish history + a coding ask,
/// as recorded chat bodies. Written for this preset, never real traces.
const AGENT_TURNS: [(&str, &str); 3] = [
    ("agent-fix-test", include_str!("../fixtures/small_card/agent_turn_1.json")),
    ("agent-refactor", include_str!("../fixtures/small_card/agent_turn_2.json")),
    ("agent-add-feature", include_str!("../fixtures/small_card/agent_turn_3.json")),
];

/// What a run measures and where it reads.
#[derive(Debug, Clone)]
pub struct Options {
    /// The server's API base URL.
    pub endpoint: String,
    /// The metrics listener's base URL, or `None` for "do not read it".
    pub metrics: Option<String>,
    /// The TTFT cells, in prompt tokens.
    pub cells: Vec<u32>,
    /// Samples per cell after the warmup.
    pub samples: usize,
    /// The output budget of a decode request.
    pub decode_tokens: u32,
}

// ── card ──────────────────────────────────────────────────────────────────

/// The card, as `nvidia-smi` reports it. A field the tool printed as
/// `[N/A]` (or not at all) is `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Card {
    pub name: Option<String>,
    pub vram_total_mib: Option<u64>,
    pub pcie_gen_current: Option<u32>,
    pub pcie_width_current: Option<u32>,
}

/// The query that [`parse_nvidia_smi`] reads the answer of.
pub const NVIDIA_SMI_ARGS: [&str; 2] = [
    "--query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current",
    "--format=csv,noheader",
];

/// Parse `nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,
/// pcie.link.width.current --format=csv,noheader`: the first card's line
/// (`NVIDIA GeForce RTX 5090, 32607 MiB, 3, 16`), and how many cards there
/// were. `Err` when no line carries four fields.
pub fn parse_nvidia_smi(output: &str) -> Result<(Card, usize), String> {
    let lines: Vec<&str> = output.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let first = lines.first().ok_or_else(|| "nvidia-smi printed nothing".to_string())?;
    let fields: Vec<&str> = first.split(',').map(str::trim).collect();
    if fields.len() != 4 {
        return Err(format!("nvidia-smi line `{first}` has {} fields, expected 4", fields.len()));
    }
    let number = |raw: &str, unit: &str| -> Option<u64> { raw.trim_end_matches(unit).trim().parse().ok() };
    let card = Card {
        name: Some(fields[0].to_string()).filter(|n| !n.is_empty() && !n.starts_with('[')),
        vram_total_mib: number(fields[1], "MiB"),
        pcie_gen_current: number(fields[2], "").map(|v| v as u32),
        pcie_width_current: number(fields[3], "").map(|v| v as u32),
    };
    Ok((card, lines.len()))
}

/// Run `nvidia-smi` for the card's line; `Err` names why it could not.
pub fn run_nvidia_smi() -> Result<String, String> {
    let output = std::process::Command::new("nvidia-smi")
        .args(NVIDIA_SMI_ARGS)
        .output()
        .map_err(|e| format!("could not run nvidia-smi: {e}"))?;
    if !output.status.success() {
        return Err(format!("nvidia-smi exited with {}", output.status));
    }
    String::from_utf8(output.stdout).map_err(|e| format!("nvidia-smi output is not UTF-8: {e}"))
}

// ── metrics ───────────────────────────────────────────────────────────────

/// One sample of a Prometheus text exposition.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub value: f64,
}

/// Parse a Prometheus text exposition: `name{k="v",...} value` lines;
/// comments and blank lines are skipped, a line that does not parse is too.
pub fn parse_prometheus(text: &str) -> Vec<Sample> {
    text.lines().filter_map(parse_sample_line).collect()
}

fn parse_sample_line(line: &str) -> Option<Sample> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let split = line.find(['{', ' ', '\t'])?;
    let name = &line[..split];
    let rest = &line[split..];
    let mut labels = BTreeMap::new();
    let rest = match rest.strip_prefix('{') {
        None => rest,
        Some(body) => {
            let bytes = body.as_bytes();
            let mut i = 0;
            loop {
                match *bytes.get(i)? {
                    b'}' => break,
                    b',' | b' ' => i += 1,
                    _ => {
                        let eq = i + body[i..].find('=')?;
                        let key = body[i..eq].trim().to_string();
                        if *bytes.get(eq + 1)? != b'"' {
                            return None;
                        }
                        // the value, with backslash escapes
                        let mut value = String::new();
                        i = eq + 2;
                        loop {
                            match *bytes.get(i)? {
                                b'"' => {
                                    i += 1;
                                    break;
                                }
                                b'\\' => {
                                    match *bytes.get(i + 1)? {
                                        b'n' => value.push('\n'),
                                        other => value.push(other as char),
                                    }
                                    i += 2;
                                }
                                _ => {
                                    let ch = body[i..].chars().next()?;
                                    value.push(ch);
                                    i += ch.len_utf8();
                                }
                            }
                        }
                        labels.insert(key, value);
                    }
                }
            }
            &body[i + 1..]
        }
    };
    let value: f64 = rest.split_whitespace().next()?.parse().ok()?;
    Some(Sample { name: name.to_string(), labels, value })
}

/// The speculative counters at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Speculative {
    pub rounds: f64,
    pub drafted: f64,
    pub accepted: f64,
}

/// The server's metrics, as the preset reads them.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    samples: Vec<Sample>,
}

impl Metrics {
    pub fn parse(text: &str) -> Self {
        Self { samples: parse_prometheus(text) }
    }

    fn value(&self, name: &str) -> Option<f64> {
        self.samples.iter().find(|s| s.name == name && s.labels.is_empty()).map(|s| s.value)
    }

    /// The build's version, from `ignis_build_info{version="..."}`.
    pub fn build_version(&self) -> Option<String> {
        self.samples
            .iter()
            .find(|s| s.name == "ignis_build_info")
            .and_then(|s| s.labels.get("version").cloned())
    }

    /// The VRAM plan: `ignis_vram_reserved_bytes` by plan line, in bytes.
    pub fn vram_plan(&self) -> BTreeMap<String, u64> {
        self.samples
            .iter()
            .filter(|s| s.name == "ignis_vram_reserved_bytes")
            .filter_map(|s| Some((s.labels.get("line")?.clone(), s.value as u64)))
            .collect()
    }

    /// The device budget the plan was laid out inside.
    pub fn vram_budget_bytes(&self) -> Option<u64> {
        self.value("ignis_vram_budget_bytes").map(|v| v as u64)
    }

    pub fn kv_pool_pages(&self) -> Option<u64> {
        self.value("ignis_kv_pool_pages").map(|v| v as u64)
    }

    pub fn kv_page_bytes(&self) -> Option<u64> {
        self.value("ignis_kv_page_bytes").map(|v| v as u64)
    }

    /// The speculative counters; `None` while the server has run no verify
    /// round (it exports none until one has ended) or has no speculation.
    pub fn speculative(&self) -> Option<Speculative> {
        Some(Speculative {
            rounds: self.value("ignis_speculative_rounds_total")?,
            drafted: self.value("ignis_speculative_drafted_tokens_total")?,
            accepted: self.value("ignis_speculative_accepted_tokens_total")?,
        })
    }
}

/// What the decode phase did to the speculative counters.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpeculativeDelta {
    pub rounds: u64,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    /// accepted / drafted, `None` with nothing drafted.
    pub acceptance: Option<f64>,
    /// accepted / rounds: the drafts a round commits beyond its own token.
    pub accepted_per_round: Option<f64>,
}

/// The change of the speculative counters across a phase. `before` is `None`
/// when the server had exported none yet (counters start at zero then).
pub fn speculative_delta(before: Option<Speculative>, after: Speculative) -> SpeculativeDelta {
    let b = before.unwrap_or(Speculative { rounds: 0.0, drafted: 0.0, accepted: 0.0 });
    let rounds = (after.rounds - b.rounds).max(0.0);
    let drafted = (after.drafted - b.drafted).max(0.0);
    let accepted = (after.accepted - b.accepted).max(0.0);
    SpeculativeDelta {
        rounds: rounds as u64,
        drafted_tokens: drafted as u64,
        accepted_tokens: accepted as u64,
        acceptance: (drafted > 0.0).then(|| accepted / drafted),
        accepted_per_round: (rounds > 0.0).then(|| accepted / rounds),
    }
}

// ── the load's shape ──────────────────────────────────────────────────────

/// The config fields the record keeps, as `group.field`. A whitelist: the
/// config document also holds paths, the bind address and the key's presence.
pub const CONFIG_KEYS: [&str; 22] = [
    "model.max_context",
    "model.prefill_chunk",
    "model.kv_format",
    "model.prefill_attention",
    "model.exception_weights",
    "model.rope_scaling",
    "model.enable_thinking",
    "model.loop_guard",
    "model.adapt_context_step",
    "model.host_margin_bytes",
    "vram.budget_bytes",
    "vram.headroom_bytes",
    "vram.kv_pool_bytes",
    "vram.pinned_expert_bytes",
    "vram.adapt_spare_tokens",
    "vram.allow_oversubscription",
    "reuse.kv_host_pool_bytes",
    "spec.backend",
    "spec.draft_tokens",
    "spec.draft_rows",
    "spec.decode_lanes",
    "spec.draft_head",
];

/// The family section a model id's values live in, beside the group's own.
fn family_scope(model_id: &str) -> &'static str {
    if model_id.contains("flash") {
        "qwen38flashnext"
    } else {
        "qwen38"
    }
}

/// One kept config value and where it came from.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConfigEntry {
    pub value: Value,
    /// `default`, `profile`, `file`, `env`, `flag` or `live`.
    pub source: Option<String>,
    /// The flag, env var or profile name that set it. Never a file's path.
    pub spelling: Option<String>,
}

/// The whitelisted values of `GET /v1/config` for the model `model_id`
/// (the family's own section beats the group's value), each with its origin
/// from `GET /v1/config/origins`. A key the document lacks is absent.
pub fn config_view(config: &Value, origins: Option<&Value>, model_id: &str) -> BTreeMap<String, ConfigEntry> {
    let scope = family_scope(model_id);
    let rows: &[Value] = origins.and_then(|o| o.get("data")).and_then(Value::as_array).map_or(&[], Vec::as_slice);
    let mut out = BTreeMap::new();
    for key in CONFIG_KEYS {
        let Some((group, field)) = key.split_once('.') else { continue };
        let Some(section) = config.get(group) else { continue };
        let scoped = section.get(scope).and_then(|s| s.get(field));
        let Some(value) = scoped.or_else(|| section.get(field)) else { continue };
        let row = |want: Option<&str>| rows.iter().find(|r| r["key"] == key && r["scope"].as_str() == want);
        let origin = if scoped.is_some() { row(Some(scope)).or_else(|| row(None)) } else { row(None) };
        let source = origin.and_then(|r| r["source"].as_str()).map(str::to_string);
        // A file's spelling carries its path: kept out of a record that is
        // pasted into an issue.
        let spelling = match source.as_deref() {
            Some("profile" | "flag" | "env") => origin.and_then(|r| r["spelling"].as_str()).map(str::to_string),
            _ => None,
        };
        out.insert(key.to_string(), ConfigEntry { value: value.clone(), source, spelling });
    }
    out
}

/// The profile the load runs under: the spelling of a `profile`-sourced row.
pub fn profile_name(origins: &Value) -> Option<String> {
    origins
        .get("data")?
        .as_array()?
        .iter()
        .find(|r| r["source"] == "profile")
        .and_then(|r| r["spelling"].as_str())
        .map(str::to_string)
}

fn as_u64(value: &Value) -> Option<u64> {
    value.as_u64().or_else(|| value.as_str().and_then(|s| s.replace(',', "").parse().ok()))
}

/// What the load did to the context the config asked for. The server logs
/// each adaptation step at start but exposes none over HTTP: this is what the
/// two numbers it does expose say. `lowered` is `None` when the config's
/// value is not a number (an `auto`) or the running context is unknown.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContextAdaptation {
    pub configured_max_context: Option<u64>,
    pub running_max_model_len: Option<u64>,
    pub lowered: Option<bool>,
}

pub fn context_adaptation(view: &BTreeMap<String, ConfigEntry>, running: Option<u64>) -> ContextAdaptation {
    let configured = view.get("model.max_context").and_then(|e| as_u64(&e.value));
    ContextAdaptation {
        configured_max_context: configured,
        running_max_model_len: running,
        lowered: configured.zip(running).map(|(c, r)| r < c),
    }
}

// ── decode ────────────────────────────────────────────────────────────────

/// One decode request's reading.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecodeSample {
    /// Which prompt (a canary id or an agent turn's name).
    pub prompt: String,
    pub prompt_tokens: Option<u32>,
    pub completion_tokens: u32,
    pub ttft_ms: f64,
    /// The decode time: the request's end less its first token.
    pub decode_ms: f64,
    /// (completion tokens - 1) / decode time, `None` when the request was
    /// too short to time a rate (under two tokens).
    pub tokens_per_s: Option<f64>,
}

/// A request's rate from its timings: tokens after the first over the time
/// after the first.
pub fn decode_sample(
    prompt: &str,
    prompt_tokens: Option<u32>,
    completion_tokens: u32,
    ttft_ms: f64,
    total_ms: f64,
) -> DecodeSample {
    let decode_ms = (total_ms - ttft_ms).max(0.0);
    let tokens_per_s = (completion_tokens >= 2 && decode_ms > 0.0)
        .then(|| f64::from(completion_tokens - 1) / (decode_ms / 1000.0));
    DecodeSample {
        prompt: prompt.to_string(),
        prompt_tokens,
        completion_tokens,
        ttft_ms,
        decode_ms,
        tokens_per_s,
    }
}

/// One prompt set's decode figures.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DecodeSet {
    pub max_tokens: u32,
    pub samples: Vec<DecodeSample>,
    /// Requests that failed, with the reason.
    pub errors: Vec<String>,
    /// The median of the samples' rates.
    pub median_tokens_per_s: Option<f64>,
    /// Total tokens over total decode time (the pooled rate).
    pub pooled_tokens_per_s: Option<f64>,
}

pub fn summarize_decode(max_tokens: u32, samples: Vec<DecodeSample>, errors: Vec<String>) -> DecodeSet {
    let rates: Vec<f64> = samples.iter().filter_map(|s| s.tokens_per_s).collect();
    let timed: Vec<&DecodeSample> = samples.iter().filter(|s| s.tokens_per_s.is_some()).collect();
    let tokens: f64 = timed.iter().map(|s| f64::from(s.completion_tokens - 1)).sum();
    let seconds: f64 = timed.iter().map(|s| s.decode_ms / 1000.0).sum();
    DecodeSet {
        max_tokens,
        median_tokens_per_s: ttft::median(&rates),
        pooled_tokens_per_s: (seconds > 0.0).then(|| tokens / seconds),
        samples,
        errors,
    }
}

// ── the tokenizer seam over the server ───────────────────────────────────

/// A [`PromptTemplate`] that asks the running server: `POST /v1/tokenize`
/// with one user message and thinking off, the very render a TTFT sample is
/// then served through, so a cell's claimed length is the length the server
/// prefills and a user needs no artifact path.
pub struct ServerTemplate {
    url: String,
    client: reqwest::blocking::Client,
}

impl ServerTemplate {
    pub fn new(base_url: &str) -> Self {
        Self {
            url: format!("{}/v1/tokenize", base_url.trim_end_matches('/')),
            client: http_client(Duration::from_secs(120)),
        }
    }
}

impl PromptTemplate for ServerTemplate {
    fn encode_user_message(&self, content: &str) -> Result<Vec<u32>, String> {
        let body = json!({
            "messages": [{ "role": "user", "content": content }],
            "enable_thinking": false,
            "return_token_ids": true,
        });
        let response = self.client.post(&self.url).json(&body).send().map_err(|e| format!("POST {}: {e}", self.url))?;
        let status = response.status();
        let value: Value = response.json().map_err(|e| format!("POST {}: parse: {e}", self.url))?;
        if !status.is_success() {
            return Err(format!("POST {} -> {status}: {}", self.url, value["error"]["message"].as_str().unwrap_or("")));
        }
        value["token_ids"]
            .as_array()
            .map(|ids| ids.iter().filter_map(|id| id.as_u64().map(|v| v as u32)).collect())
            .ok_or_else(|| format!("POST {}: no token_ids in the answer", self.url))
    }

    fn decode(&self, _ids: &[u32]) -> Result<String, String> {
        Err("the small-card preset builds its prompts by filler growth, never from a corpus".to_string())
    }
}

fn http_client(timeout: Duration) -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder().timeout(timeout).build().expect("the preset's HTTP client builds")
}

fn get_text(client: &reqwest::blocking::Client, url: &str) -> Result<String, String> {
    client
        .get(url)
        .send()
        .map_err(|e| format!("GET {url} failed: {e}"))?
        .error_for_status()
        .map_err(|e| format!("GET {url} -> {e}"))?
        .text()
        .map_err(|e| format!("GET {url}: {e}"))
}

fn get_json(client: &reqwest::blocking::Client, url: &str) -> Result<Value, String> {
    let text = get_text(client, url)?;
    serde_json::from_str(&text).map_err(|e| format!("GET {url}: parse: {e}"))
}

// ── urls ──────────────────────────────────────────────────────────────────

/// `url` without any `user:password@` part, so a credential a user typed into
/// the endpoint never reaches the record.
pub fn redact_url(url: &str) -> String {
    match url.split_once("://") {
        Some((scheme, rest)) => {
            let (authority, path) = rest.split_once('/').map_or((rest, ""), |(a, p)| (a, p));
            let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
            if path.is_empty() {
                format!("{scheme}://{host}")
            } else {
                format!("{scheme}://{host}/{path}")
            }
        }
        None => url.rsplit_once('@').map_or(url, |(_, h)| h).to_string(),
    }
}

/// The metrics listener the server defaults to, on the endpoint's host.
pub fn default_metrics_url(endpoint: &str) -> String {
    let redacted = redact_url(endpoint);
    let (scheme, rest) = redacted.split_once("://").unwrap_or(("http", &redacted));
    let authority = rest.split('/').next().unwrap_or(rest);
    let host = if authority.starts_with('[') {
        authority.split_once(']').map_or(authority, |(h, _)| &authority[..h.len() + 1])
    } else {
        authority.split(':').next().unwrap_or(authority)
    };
    format!("{scheme}://{host}:{DEFAULT_METRICS_PORT}")
}

// ── the record ────────────────────────────────────────────────────────────

/// The server, as the record names it.
#[derive(Debug, Clone, Serialize)]
pub struct ServerFacts {
    pub version: Option<String>,
    pub model_id: Option<String>,
    pub profile: Option<String>,
    pub status: Option<String>,
    pub max_model_len: Option<u64>,
}

/// The load's VRAM plan.
#[derive(Debug, Clone, Serialize)]
pub struct Plan {
    pub budget_bytes: Option<u64>,
    pub reserved_bytes_by_line: Option<BTreeMap<String, u64>>,
    pub kv_pool_pages: Option<u64>,
    pub kv_page_bytes: Option<u64>,
}

/// One TTFT cell as the record keeps it.
#[derive(Debug, Clone, Serialize)]
pub struct TtftCell {
    pub prompt_tokens: u32,
    pub median_ttft_ms: Option<f64>,
    pub samples_ms: Vec<f64>,
    /// Every sample was provably a cold prefix.
    pub all_cold: bool,
    /// Why the cell has no median: not measurable here, or failed.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Decode {
    pub coding: Option<DecodeSet>,
    pub agent: Option<DecodeSet>,
    /// The speculative counters' change across both sets.
    pub speculative: Option<SpeculativeDelta>,
}

/// The JSON a user pastes into an issue.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub schema: &'static str,
    pub bench_version: &'static str,
    pub started_at: String,
    pub finished_at: String,
    pub endpoint: String,
    pub card: Option<Card>,
    pub server: ServerFacts,
    pub plan: Plan,
    pub config: Option<BTreeMap<String, ConfigEntry>>,
    pub context: ContextAdaptation,
    pub decode: Decode,
    pub ttft: Vec<TtftCell>,
    /// Every piece that could not be read or measured, and why.
    pub notes: Vec<String>,
}

fn ttft_cell(cell: &ttft::Cell) -> TtftCell {
    TtftCell {
        prompt_tokens: cell.prompt_tokens,
        median_ttft_ms: cell.median_ttft_ms,
        samples_ms: cell.samples.iter().map(|s| s.ttft_ms).collect(),
        all_cold: cell.all_cold(),
        error: cell.error.clone().or_else(|| {
            cell.void_samples().first().map(|s| s.void_reason.clone().unwrap_or_else(|| "a void sample".into()))
        }),
    }
}

fn decode_request(id: &str, prompt: String, max_tokens: u32) -> Request {
    Request {
        id: id.to_string(),
        class: RequestClass::Main,
        prompt,
        max_tokens,
        stream: true,
        include_usage: true,
        // The server's configured default: an agent client thinks, and the
        // rate counts the reasoning channel as decode work like any other.
        enable_thinking: None,
        images: Vec::new(),
    }
}

fn measure_decode(ep: &dyn Endpoint, prompts: Vec<(String, String)>, max_tokens: u32) -> DecodeSet {
    let mut samples = Vec::new();
    let mut errors = Vec::new();
    for (id, prompt) in prompts {
        eprintln!("decode: {id}");
        match ep.complete(&decode_request(&id, prompt, max_tokens)) {
            Ok(o) => samples.push(decode_sample(&id, o.prompt_tokens, o.n_tokens, o.ttft_ms, o.total_ms)),
            Err(e) => errors.push(format!("{id}: {e}")),
        }
    }
    summarize_decode(max_tokens, samples, errors)
}

/// Run the preset against `opts.endpoint` and assemble the record.
/// `nvidia_smi` is the card probe (the real one is [`run_nvidia_smi`]).
pub fn run(opts: &Options, nvidia_smi: &dyn Fn() -> Result<String, String>) -> Report {
    let started_at = utc_timestamp(unix_now());
    let mut notes = Vec::new();
    let client = http_client(Duration::from_secs(30));
    let base = opts.endpoint.trim_end_matches('/').to_string();

    // The card.
    let card = match nvidia_smi().and_then(|out| parse_nvidia_smi(&out)) {
        Ok((card, count)) => {
            if count > 1 {
                notes.push(format!("nvidia-smi lists {count} cards; the first is recorded"));
            }
            Some(card)
        }
        Err(e) => {
            notes.push(format!("card: {e}"));
            None
        }
    };

    // The shape the server runs with.
    let models = get_json(&client, &format!("{base}/v1/models")).map_err(|e| notes.push(format!("models: {e}"))).ok();
    let active = models.as_ref().and_then(|m| m["data"].as_array()).and_then(|d| d.first()).cloned();
    let model_id = active.as_ref().and_then(|m| m["id"].as_str()).map(str::to_string);
    let max_model_len = active.as_ref().and_then(|m| m["max_model_len"].as_u64());
    if model_id.is_some() && max_model_len.is_none() {
        notes.push("models: the server reports no max_model_len for the loaded model".into());
    }
    let status = models.as_ref().and_then(|m| m["status"].as_str()).map(str::to_string);
    let config = get_json(&client, &format!("{base}/v1/config")).map_err(|e| notes.push(format!("config: {e}"))).ok();
    let origins =
        get_json(&client, &format!("{base}/v1/config/origins")).map_err(|e| notes.push(format!("config origins: {e}"))).ok();
    let view = config.as_ref().map(|c| config_view(c, origins.as_ref(), model_id.as_deref().unwrap_or("")));
    let profile = origins.as_ref().and_then(profile_name);
    if origins.is_some() && profile.is_none() {
        notes.push("profile: no config value comes from a profile (none in use, or all overridden)".into());
    }
    let context = context_adaptation(view.as_ref().unwrap_or(&BTreeMap::new()), max_model_len);
    notes.push(
        "adaptations: the server logs each adaptation step at start and exposes none over HTTP; \
         `context` compares the configured max_context with the context running"
            .into(),
    );

    // The plan (the metrics listener).
    let metrics_text = |notes: &mut Vec<String>| -> Option<Metrics> {
        let url = opts.metrics.as_ref()?;
        match get_text(&client, &format!("{}/metrics", url.trim_end_matches('/'))) {
            Ok(text) => Some(Metrics::parse(&text)),
            Err(e) => {
                notes.push(format!("metrics: {e}"));
                None
            }
        }
    };
    if opts.metrics.is_none() {
        notes.push("metrics: not read (--metrics off): no VRAM plan lines, build version or speculative counters".into());
    }
    let before = metrics_text(&mut notes);

    // Decode.
    let ep = HttpEndpoint::new(&base).as_recorded(true);
    let (mut decode, mut ttft_cells) = (Decode { coding: None, agent: None, speculative: None }, Vec::new());
    if model_id.is_some() {
        // One short request first: a cold server's first request pays one-off costs.
        let _ = ep.complete(&decode_request("warmup", CANARIES[0].prompt.to_string(), 16));
        let coding: Vec<(String, String)> = CANARIES.iter().map(|c| (c.id.to_string(), c.prompt.to_string())).collect();
        decode.coding = Some(measure_decode(&ep, coding, opts.decode_tokens));
        let agent: Vec<(String, String)> =
            AGENT_TURNS.iter().map(|(id, body)| (id.to_string(), body.to_string())).collect();
        decode.agent = Some(measure_decode(&ep, agent, opts.decode_tokens));
        let after = metrics_text(&mut notes);
        match (&before, after.as_ref().and_then(Metrics::speculative)) {
            (Some(b), Some(a)) => decode.speculative = Some(speculative_delta(b.speculative(), a)),
            (Some(_), None) => notes.push(
                "speculative: the server exports no speculative counters (no speculation, or no verify round ran)".into(),
            ),
            _ => {}
        }

        // TTFT.
        let template = ServerTemplate::new(&base);
        let ttft_ep = HttpEndpoint::new(&base);
        let tokenizer_ok = match template.encode_user_message("hello") {
            Ok(_) => true,
            Err(e) => {
                notes.push(format!("ttft: the server's /v1/tokenize is unusable ({e}); no cell can be built"));
                false
            }
        };
        for &prompt_tokens in &opts.cells {
            let reach = u64::from(prompt_tokens) + u64::from(ttft::DEFAULT_MAX_TOKENS);
            if let Some(limit) = max_model_len.filter(|&l| reach > l) {
                ttft_cells.push(TtftCell {
                    prompt_tokens,
                    median_ttft_ms: None,
                    samples_ms: Vec::new(),
                    all_cold: false,
                    error: Some(format!(
                        "does not fit: {prompt_tokens} prompt tokens + {} output exceed the running context of {limit}",
                        ttft::DEFAULT_MAX_TOKENS
                    )),
                });
                continue;
            }
            if !tokenizer_ok {
                ttft_cells.push(TtftCell {
                    prompt_tokens,
                    median_ttft_ms: None,
                    samples_ms: Vec::new(),
                    all_cold: false,
                    error: Some("no tokenizer: /v1/tokenize is unusable".into()),
                });
                continue;
            }
            eprintln!("ttft: {prompt_tokens} tokens x {} samples (+1 warmup)", opts.samples);
            let spec = CellSpec { prompt_tokens, samples: opts.samples };
            let cell = ttft::measure_cell(&ttft_ep, &template, &spec, ttft::DEFAULT_MAX_TOKENS);
            ttft_cells.push(ttft_cell(&cell));
        }
    } else {
        notes.push("no model answered at /v1/models: nothing was measured".into());
    }

    // The plan, read last: a lazily built pool or a switch has landed by now.
    let plan_metrics = metrics_text(&mut notes);
    let plan = Plan {
        budget_bytes: plan_metrics.as_ref().and_then(Metrics::vram_budget_bytes),
        reserved_bytes_by_line: plan_metrics.as_ref().map(Metrics::vram_plan),
        kv_pool_pages: plan_metrics.as_ref().and_then(Metrics::kv_pool_pages),
        kv_page_bytes: plan_metrics.as_ref().and_then(Metrics::kv_page_bytes),
    };
    notes.dedup();
    Report {
        schema: SCHEMA,
        bench_version: env!("CARGO_PKG_VERSION"),
        started_at,
        finished_at: utc_timestamp(unix_now()),
        endpoint: redact_url(&opts.endpoint),
        card,
        server: ServerFacts {
            version: plan_metrics.as_ref().and_then(Metrics::build_version),
            model_id,
            profile,
            status,
            max_model_len,
        },
        plan,
        config: view,
        context,
        decode,
        ttft: ttft_cells,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smi_line_parses_into_the_card() {
        let (card, count) = parse_nvidia_smi("NVIDIA GeForce RTX 5090, 32607 MiB, 3, 16\n").unwrap();
        assert_eq!(count, 1);
        assert_eq!(card.name.as_deref(), Some("NVIDIA GeForce RTX 5090"));
        assert_eq!(card.vram_total_mib, Some(32607));
        assert_eq!((card.pcie_gen_current, card.pcie_width_current), (Some(3), Some(16)));
    }

    #[test]
    fn nvidia_smi_na_fields_are_none_and_extra_cards_are_counted() {
        let (card, count) =
            parse_nvidia_smi("NVIDIA GeForce RTX 5070 Ti, 16303 MiB, [N/A], [N/A]\nNVIDIA T400, 4096 MiB, 3, 16\n").unwrap();
        assert_eq!(count, 2);
        assert_eq!(card.name.as_deref(), Some("NVIDIA GeForce RTX 5070 Ti"));
        assert_eq!(card.vram_total_mib, Some(16303));
        assert_eq!((card.pcie_gen_current, card.pcie_width_current), (None, None));
    }

    #[test]
    fn nvidia_smi_garbage_is_an_error_not_a_panic() {
        assert!(parse_nvidia_smi("").is_err());
        assert!(parse_nvidia_smi("NVIDIA-SMI has failed because it couldn't communicate").is_err());
    }

    const METRICS: &str = "# HELP ignis_build_info Constant build identity with value 1.\n\
# TYPE ignis_build_info gauge\n\
ignis_build_info{version=\"0.7.3\"} 1\n\
# TYPE ignis_vram_reserved_bytes gauge\n\
ignis_vram_reserved_bytes{line=\"weights\"} 12000000000\n\
ignis_vram_reserved_bytes{line=\"trellis_workspace\"} 805306368\n\
ignis_vram_reserved_bytes{line=\"workspace\"} 0\n\
ignis_vram_budget_bytes 17179869184\n\
ignis_kv_pool_pages 4096\n\
ignis_kv_page_bytes 1048576\n\
ignis_speculative_rounds_total 100\n\
ignis_speculative_drafted_tokens_total 800\n\
ignis_speculative_accepted_tokens_total 500\n\
ignis_speculative_position_accepted_total{position=\"1\"} 90\n";

    #[test]
    fn the_plan_lines_budget_pool_and_version_come_out_of_the_metrics() {
        let m = Metrics::parse(METRICS);
        assert_eq!(m.build_version().as_deref(), Some("0.7.3"));
        let plan = m.vram_plan();
        assert_eq!(plan.get("weights"), Some(&12_000_000_000));
        assert_eq!(plan.get("trellis_workspace"), Some(&805_306_368));
        assert_eq!(plan.get("workspace"), Some(&0));
        assert_eq!(m.vram_budget_bytes(), Some(17_179_869_184));
        assert_eq!((m.kv_pool_pages(), m.kv_page_bytes()), (Some(4096), Some(1_048_576)));
    }

    #[test]
    fn labels_with_escapes_commas_and_several_pairs_parse() {
        let s = parse_prometheus("x{a=\"1,2\",b=\"q\\\"uote\\\\\",c=\"\"} 3.5\n");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].labels["a"], "1,2");
        assert_eq!(s[0].labels["b"], "q\"uote\\");
        assert_eq!(s[0].labels["c"], "");
        assert_eq!(s[0].value, 3.5);
        assert!(parse_prometheus("# comment\n\nnot a sample line\nname notanumber\n").is_empty());
    }

    #[test]
    fn a_server_with_no_verify_round_has_no_speculative_counters() {
        assert_eq!(Metrics::parse("ignis_build_info{version=\"1\"} 1\n").speculative(), None);
    }

    #[test]
    fn the_speculative_delta_is_the_change_across_the_decode_phase() {
        let before = Metrics::parse(METRICS).speculative();
        let after = Speculative { rounds: 140.0, drafted: 1120.0, accepted: 740.0 };
        let d = speculative_delta(before, after);
        assert_eq!((d.rounds, d.drafted_tokens, d.accepted_tokens), (40, 320, 240));
        assert_eq!(d.acceptance, Some(0.75));
        assert_eq!(d.accepted_per_round, Some(6.0));
        // no counters before: they started at zero
        let d = speculative_delta(None, after);
        assert_eq!(d.rounds, 140);
        // nothing ran: no division by zero
        let same = Speculative { rounds: 1.0, drafted: 0.0, accepted: 0.0 };
        let d = speculative_delta(Some(same), same);
        assert_eq!((d.acceptance, d.accepted_per_round), (None, None));
    }

    fn config_doc() -> Value {
        json!({
            "server": { "bind": "0.0.0.0:8080", "api_key": "sk-secret" },
            "model": {
                "artifact": "C:/Users/someone/models/x.ninfer",
                "max_context": 262144,
                "prefill_chunk": 1024,
                "qwen38": { "max_context": 262144, "rope_scaling": "none" },
                "rope_scaling": "auto",
            },
            "vram": { "budget_bytes": 17179869184u64 },
            "spec": { "decode_lanes": 2, "draft_head": "auto" },
        })
    }

    fn origins_doc() -> Value {
        json!({ "object": "list", "data": [
            { "key": "model.max_context", "scope": null, "source": "profile", "scoped": false, "spelling": "rtx50-16gb" },
            { "key": "model.max_context", "scope": "qwen38", "source": "profile", "scoped": true, "spelling": "rtx50-16gb" },
            { "key": "model.prefill_chunk", "scope": null, "source": "file", "scoped": false,
              "spelling": "model.prefill_chunk in C:/Users/someone/ignis.yaml" },
            { "key": "vram.budget_bytes", "scope": null, "source": "env", "scoped": false, "spelling": "IGNIS_VRAM_BUDGET_BYTES" },
            { "key": "spec.decode_lanes", "scope": null, "source": "default", "scoped": false },
            { "key": "model.rope_scaling", "scope": "qwen38", "source": "profile", "scoped": true, "spelling": "rtx50-16gb" },
        ]})
    }

    #[test]
    fn the_config_view_keeps_the_whitelist_the_family_value_and_the_origin() {
        let view = config_view(&config_doc(), Some(&origins_doc()), "qwen3.8-27b-16g");
        assert_eq!(view["model.max_context"].value, json!(262144));
        assert_eq!(view["model.max_context"].source.as_deref(), Some("profile"));
        assert_eq!(view["model.max_context"].spelling.as_deref(), Some("rtx50-16gb"));
        // the family's section beats the group's value
        assert_eq!(view["model.rope_scaling"].value, json!("none"));
        assert_eq!(view["vram.budget_bytes"].source.as_deref(), Some("env"));
        assert_eq!(view["vram.budget_bytes"].spelling.as_deref(), Some("IGNIS_VRAM_BUDGET_BYTES"));
        assert_eq!(view["spec.decode_lanes"].source.as_deref(), Some("default"));
        assert_eq!(view["spec.decode_lanes"].spelling, None);
    }

    #[test]
    fn the_config_view_never_carries_a_path_a_key_or_the_bind_address() {
        let view = config_view(&config_doc(), Some(&origins_doc()), "qwen3.8-27b-16g");
        let text = serde_json::to_string(&view).unwrap();
        for leak in ["sk-secret", "api_key", "someone", "artifact", "0.0.0.0", "ignis.yaml"] {
            assert!(!text.contains(leak), "{leak} leaked into {text}");
        }
        assert_eq!(view["model.prefill_chunk"].source.as_deref(), Some("file"));
        assert_eq!(view["model.prefill_chunk"].spelling, None, "a file's spelling is its path");
    }

    #[test]
    fn the_config_view_without_origins_still_has_values() {
        let view = config_view(&config_doc(), None, "qwen3.8-27b");
        assert_eq!(view["model.max_context"].source, None);
        assert_eq!(view["spec.draft_head"].value, json!("auto"));
    }

    #[test]
    fn the_profile_is_the_spelling_of_a_profile_sourced_value() {
        assert_eq!(profile_name(&origins_doc()).as_deref(), Some("rtx50-16gb"));
        assert_eq!(profile_name(&json!({ "data": [{ "key": "a.b", "source": "default" }] })), None);
    }

    #[test]
    fn a_running_context_below_the_configured_one_says_the_load_lowered_it() {
        let view = config_view(&config_doc(), None, "qwen3.8-27b");
        let c = context_adaptation(&view, Some(61_440));
        assert_eq!(
            c,
            ContextAdaptation { configured_max_context: Some(262144), running_max_model_len: Some(61_440), lowered: Some(true) }
        );
        assert_eq!(context_adaptation(&view, Some(262_144)).lowered, Some(false));
        assert_eq!(context_adaptation(&view, None).lowered, None);
        let auto = config_view(&json!({ "model": { "max_context": "auto" } }), None, "qwen3.8-27b");
        assert_eq!(context_adaptation(&auto, Some(1)).lowered, None);
    }

    #[test]
    fn a_decode_rate_counts_the_tokens_after_the_first_over_the_time_after_it() {
        let s = decode_sample("p", Some(10), 101, 500.0, 2500.0);
        assert_eq!(s.decode_ms, 2000.0);
        assert_eq!(s.tokens_per_s, Some(50.0));
        assert_eq!(decode_sample("p", None, 1, 5.0, 9.0).tokens_per_s, None);
        assert_eq!(decode_sample("p", None, 50, 9.0, 9.0).tokens_per_s, None);
    }

    #[test]
    fn a_decode_set_has_a_median_a_pooled_rate_and_keeps_its_errors() {
        let samples = vec![
            decode_sample("a", None, 101, 0.0, 1000.0),  // 100 tok/s
            decode_sample("b", None, 101, 0.0, 2000.0),  // 50 tok/s
            decode_sample("c", None, 101, 0.0, 4000.0),  // 25 tok/s
            decode_sample("short", None, 1, 0.0, 10.0), // untimed
        ];
        let set = summarize_decode(256, samples, vec!["d: timeout".into()]);
        assert_eq!(set.median_tokens_per_s, Some(50.0));
        // 300 tokens over 7 s
        assert!((set.pooled_tokens_per_s.unwrap() - 300.0 / 7.0).abs() < 1e-9);
        assert_eq!(set.errors, vec!["d: timeout".to_string()]);
        assert_eq!(summarize_decode(256, Vec::new(), Vec::new()).median_tokens_per_s, None);
    }

    #[test]
    fn the_endpoint_loses_its_credentials() {
        assert_eq!(redact_url("http://user:pw@127.0.0.1:8080"), "http://127.0.0.1:8080");
        assert_eq!(redact_url("https://tok@host.example/base/"), "https://host.example/base/");
        assert_eq!(redact_url("http://127.0.0.1:8080"), "http://127.0.0.1:8080");
    }

    #[test]
    fn the_default_metrics_listener_is_the_endpoints_host_on_9464() {
        assert_eq!(default_metrics_url("http://127.0.0.1:8080"), "http://127.0.0.1:9464");
        assert_eq!(default_metrics_url("http://u:p@box.lan:8080/"), "http://box.lan:9464");
        assert_eq!(default_metrics_url("http://[::1]:8080"), "http://[::1]:9464");
    }

    #[test]
    fn the_agent_fixtures_are_synthetic_chat_bodies_with_tools_and_history() {
        for (id, body) in AGENT_TURNS {
            let v: Value = serde_json::from_str(body).unwrap_or_else(|e| panic!("{id}: {e}"));
            let messages = v["messages"].as_array().unwrap_or_else(|| panic!("{id}: no messages"));
            assert_eq!(messages[0]["role"], "system", "{id}");
            assert!(messages.len() >= 5, "{id}: system + history + ask");
            assert!(messages.iter().any(|m| m["role"] == "tool"), "{id}: tool-ish history");
            assert_eq!(messages.last().unwrap()["role"], "user", "{id}: ends on the coding ask");
            assert!(crate::client::is_recorded_chat_body(body), "{id}");
            for private in ["F:\\", "F:/", "C:\\", "/home/", "Users", "sk-", ".scratch"] {
                assert!(!body.contains(private), "{id} mentions {private}");
            }
        }
    }
}
