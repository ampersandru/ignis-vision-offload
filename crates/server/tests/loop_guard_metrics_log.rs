//! The loop guard's counters are the tally of its log (GitHub #330, spec
//! server/14): every `ignis.loop_guard.*` line counts once in
//! `ignis_loop_guard_loops_total`, under its own outcome, and nothing else
//! does. Each outcome the HTTP surface can reach, on a server of its own.
//!
//! A binary of its own with one test, which installs the process's
//! subscriber: the guard writes its lines on its own task, which a
//! subscriber scoped to the test's thread never sees.

#[path = "support/responses.rs"]
mod responses;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::Request;
use ignis_core::{MockCompute, SchedulerConfig};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::loop_guard::{LoopGuardMode, LoopGuardParams};
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

use responses::{server, Script, MODEL};

const OUTCOMES: [&str; 7] = ["rewound", "closed", "detected", "too_little_kept", "no_room", "no_tokenizer", "submit_refused"];

/// Every `ignis.loop_guard.*` line `log` holds, per outcome: a `rewound`
/// line that closed the reasoning is `closed`.
fn logged(log: &MemorySink) -> BTreeMap<String, u64> {
    let mut tally = BTreeMap::new();
    for line in log.lines() {
        let event: Value = serde_json::from_str(&line).expect("json");
        let Some(name) = event["event_name"].as_str().and_then(|name| name.strip_prefix("ignis.loop_guard.")) else {
            continue;
        };
        let closed = &event["attributes"]["closed"];
        let outcome = match name {
            "rewound" if closed == &json!(true) || closed == &json!("true") => "closed",
            name => name,
        };
        *tally.entry(outcome.to_owned()).or_insert(0) += 1;
    }
    tally
}

/// One server whose request 0 reasons `reasoning` (its template able to
/// write the nudge when `encoding`), the guard under `guard`, asked once for
/// `max_output_tokens`: what its metrics counted, per outcome.
async fn served(reasoning: &str, guard: LoopGuardParams, max_output_tokens: u32, encoding: bool) -> BTreeMap<String, u64> {
    let mock = Arc::new(MockCompute::new());
    mock.eos_after(1, 20);
    let decode = responses::scripted_text(&mock, 0, reasoning);
    let script = if encoding { Script::encoding(decode) } else { Script::new(decode) };
    let server = server(&script, SchedulerConfig::default(), mock).with_loop_guard(guard).with_metrics();
    let metrics = Arc::clone(server.metrics.as_ref().unwrap());
    let body = json!({ "model": MODEL, "input": "the whole history of Rome", "max_output_tokens": max_output_tokens });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = server.app().oneshot(request).await.unwrap();
    let status = response.status();
    let text = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(status, 200, "{}", String::from_utf8_lossy(&text));
    let exposition = metrics.render();
    OUTCOMES
        .iter()
        .map(|outcome| {
            let series = format!("ignis_loop_guard_loops_total{{outcome=\"{outcome}\"}} ");
            let line = exposition.lines().find_map(|line| line.strip_prefix(&series)).expect("the loop family");
            ((*outcome).to_owned(), line.parse().unwrap())
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_loop_guard_line_counts_once_under_its_outcome() {
    let log = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone())))
        .expect("the only subscriber this binary installs");

    let rewind = LoopGuardParams { mode: LoopGuardMode::Rewind, ..LoopGuardParams::default() };
    let looping = responses::prose(3000) + &responses::looping(4000);
    let runs = [
        (looping.clone(), rewind.clone(), 4000, true),
        (looping.clone(), LoopGuardParams { retries: 0, ..rewind.clone() }, 4000, true),
        (looping.clone(), LoopGuardParams { mode: LoopGuardMode::Log, ..rewind.clone() }, 1000, true),
        (responses::looping(8000), rewind.clone(), 1100, true),
        // The kept reasoning and the nudge leave less than a continuation's
        // 512 tokens of 720.
        (looping.clone(), rewind.clone(), 720, true),
        // A template with no way to write the nudge.
        (looping.clone(), rewind.clone(), 1000, false),
    ];
    let mut counted: BTreeMap<String, u64> = BTreeMap::new();
    for (reasoning, guard, max_output_tokens, encoding) in runs {
        for (outcome, n) in served(&reasoning, guard, max_output_tokens, encoding).await {
            *counted.entry(outcome).or_insert(0) += n;
        }
    }
    for outcome in ["rewound", "closed", "detected", "too_little_kept", "no_room", "no_tokenizer"] {
        assert!(counted[outcome] > 0, "no {outcome} loop was served: {counted:?}");
    }

    // The lines may reach the sink after the response.
    let counted: BTreeMap<String, u64> = counted.into_iter().filter(|&(_, n)| n > 0).collect();
    let deadline = Instant::now() + Duration::from_secs(10);
    while logged(&log) != counted && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(logged(&log), counted, "the log's lines, per outcome, against the counters");
}
