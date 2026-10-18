//! Serve exit with connections still open (the RTX PRO 6000 incident,
//! 2026-10-18: a Knative pod held its GPU for an hour after deletion with a
//! Playground socket open). What `Server::serve_on_until` does once its
//! shutdown resolves while a client still holds:
//! - an idle keep-alive HTTP connection;
//! - an idle Responses WebSocket;
//! - a streamed completion still generating.
//! Each case asserts serve returns within a bound instead of waiting on the
//! client.

#[path = "support/responses.rs"]
mod responses;

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use ignis_core::{MockCompute, SchedulerConfig};
use serde_json::json;

use responses::{server, Script, MODEL};

/// Serve `server` on a fresh listener; the address, the stop, and the serve
/// task to wait on.
async fn serving(
    server: ignis_server::Server,
) -> (std::net::SocketAddr, tokio::sync::oneshot::Sender<()>, tokio::task::JoinHandle<std::io::Result<()>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let task = tokio::spawn(server.serve_on_until(listener, None, async {
        let _ = stopped.await;
    }));
    (addr, stop, task)
}

/// Room for one request that would generate for a long time.
fn long_config() -> SchedulerConfig {
    SchedulerConfig { max_sequence_tokens: 2 << 20, kv_capacity_pages: (2 << 20) / 16 * 2, ..SchedulerConfig::default() }
}

const BOUND: Duration = Duration::from_secs(5);

#[tokio::test]
async fn an_idle_keep_alive_connection_does_not_hold_the_exit() {
    let script = Script::silent();
    let (addr, stop, task) = serving(server(&script, SchedulerConfig::default(), Arc::new(MockCompute::new()))).await;
    let client = reqwest::Client::new();
    let ready = client.get(format!("http://{addr}/health")).send().await.expect("health");
    assert!(ready.status().is_success());
    // The connection stays in the client's pool, idle.
    stop.send(()).expect("serving");
    tokio::time::timeout(BOUND, task).await.expect("serve returned").expect("task").expect("serve");
}

#[tokio::test]
async fn an_idle_responses_socket_does_not_hold_the_exit() {
    let script = Script::silent();
    let (addr, stop, task) = serving(server(&script, SchedulerConfig::default(), Arc::new(MockCompute::new()))).await;
    let (_socket, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/v1/responses")).await.expect("the upgrade");
    stop.send(()).expect("serving");
    tokio::time::timeout(BOUND, task).await.expect("serve returned with a socket open").expect("task").expect("serve");
}

#[tokio::test]
async fn a_stream_still_generating_does_not_hold_the_exit() {
    let script = Script::silent();
    let (addr, stop, task) = serving(server(&script, long_config(), Arc::new(MockCompute::new()))).await;
    let client = reqwest::Client::new();
    let response = client
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({ "model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": true,
                       "max_tokens": 1_000_000, "ignore_eos": true, "enable_thinking": false }))
        .send()
        .await
        .expect("the stream starts");
    assert!(response.status().is_success());
    let mut body = response.bytes_stream();
    body.next().await.expect("a first chunk").expect("chunk");
    stop.send(()).expect("serving");
    tokio::time::timeout(BOUND, task).await.expect("serve returned with a stream generating").expect("task").expect("serve");
    drop(body);
}
