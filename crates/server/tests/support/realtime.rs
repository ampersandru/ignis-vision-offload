//! The Realtime transcription socket's test client (spec speech/03, GitHub
//! #353): a live listener over a server, a `tokio-tungstenite` client at
//! `/v1/realtime`, and PCM16 appends as base64. Included through
//! `support/mod.rs`.

use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use ignis_server::Server;
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

/// A server serving on a listener of its own until dropped.
pub struct Live {
    pub addr: std::net::SocketAddr,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for Live {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}

/// Serve `server` on a fresh listener.
pub async fn live(server: Server) -> Live {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(server.serve_on_until(listener, None, async {
        let _ = stopped.await;
    }));
    Live { addr, stop: Some(stop) }
}

pub type Socket = tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

pub type Handshake = tokio_tungstenite::tungstenite::handshake::client::Response;

/// Open `/v1/realtime{query}` with `headers` on the upgrade: the socket and
/// the handshake, or the refusal.
pub async fn open(live: &Live, query: &str, headers: &[(&str, &str)]) -> Result<(Socket, Handshake), tokio_tungstenite::tungstenite::Error> {
    let mut request = format!("ws://{}/v1/realtime{query}", live.addr).into_client_request().unwrap();
    for (name, value) in headers {
        request
            .headers_mut()
            .append(tokio_tungstenite::tungstenite::http::HeaderName::from_bytes(name.as_bytes()).unwrap(), value.parse().unwrap());
    }
    tokio_tungstenite::connect_async(request).await
}

/// A transcription session, open, and the `session.created` it began with.
pub async fn session(live: &Live) -> (Socket, Value) {
    let (mut socket, _) = open(live, "?intent=transcription", &[]).await.expect("the upgrade succeeds");
    let created = next_event(&mut socket).await;
    assert_eq!(created["type"], "session.created", "{created}");
    (socket, created)
}

/// The status of a refused upgrade, and its body.
pub fn refusal(refused: tokio_tungstenite::tungstenite::Error) -> (u16, axum::http::HeaderMap<axum::http::HeaderValue>, Value) {
    match refused {
        tokio_tungstenite::tungstenite::Error::Http(response) => {
            let status = response.status().as_u16();
            let mut headers = axum::http::HeaderMap::new();
            for (name, value) in response.headers() {
                headers.insert(
                    axum::http::HeaderName::from_bytes(name.as_str().as_bytes()).unwrap(),
                    axum::http::HeaderValue::from_bytes(value.as_bytes()).unwrap(),
                );
            }
            let body = response.body().as_deref().and_then(|body| serde_json::from_slice(body).ok()).unwrap_or(Value::Null);
            (status, headers, body)
        }
        other => panic!("expected an HTTP refusal, got {other:?}"),
    }
}

pub async fn send(socket: &mut Socket, event: Value) {
    socket.send(Message::Text(event.to_string().into())).await.expect("send");
}

/// The next event, within a generous bound (a hang fails loudly).
pub async fn next_event(socket: &mut Socket) -> Value {
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(10), socket.next())
            .await
            .expect("an event within 10 s")
            .expect("the socket is open")
            .expect("a frame");
        match frame {
            Message::Text(text) => return serde_json::from_str(text.as_str()).expect("a JSON event"),
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("unexpected frame {other:?}"),
        }
    }
}

/// The events up to and including the first of type `kind`.
pub async fn until(socket: &mut Socket, kind: &str) -> Vec<Value> {
    let mut events = Vec::new();
    loop {
        let event = next_event(socket).await;
        let done = event["type"] == kind;
        events.push(event);
        if done {
            return events;
        }
    }
}

/// The events of `kind` among `events`.
pub fn of<'a>(events: &'a [Value], kind: &str) -> Vec<&'a Value> {
    events.iter().filter(|event| event["type"] == kind).collect()
}

/// Whether any event arrives within `wait`.
pub async fn quiet_for(socket: &mut Socket, wait: Duration) -> Option<Value> {
    match tokio::time::timeout(wait, socket.next()).await {
        Ok(Some(Ok(Message::Text(text)))) => Some(serde_json::from_str(text.as_str()).unwrap()),
        _ => None,
    }
}

/// Standard base64, padded, of `bytes`.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// An `input_audio_buffer.append` of `samples` 16-bit samples of a quiet
/// tone, at whatever rate the session takes.
pub fn append(samples: usize) -> Value {
    let bytes: Vec<u8> = (0..samples).flat_map(|i| (((i % 64) as i16 - 32) * 64).to_le_bytes()).collect();
    json!({ "type": "input_audio_buffer.append", "audio": base64(&bytes) })
}

pub fn commit() -> Value {
    json!({ "type": "input_audio_buffer.commit" })
}

/// A `session.update` of `session`'s fields, as a transcription session.
pub fn update(mut session: Value) -> Value {
    session["type"] = json!("transcription");
    json!({ "type": "session.update", "session": session })
}

/// `session.update` to 16 kHz, the model's own rate, so a test's samples are
/// the executor's, and to manual commit (`turn_detection: null`), so the
/// test's commits end its items, not a VAD (spec speech/05 makes `server_vad`
/// the default).
pub fn at_16_khz() -> Value {
    update(json!({ "audio": { "input": { "format": { "type": "audio/pcm", "rate": 16000 }, "turn_detection": null } } }))
}

/// `session.update` to manual commit, at whatever rate the session takes.
pub fn manual() -> Value {
    update(json!({ "audio": { "input": { "turn_detection": null } } }))
}

/// Samples in one 8 s window at 16 kHz.
pub const WINDOW: usize = ignis_server::speech::executor::WINDOW_SAMPLES;

/// Wait (bounded) until `check` holds.
pub async fn until_true(what: &str, check: impl Fn() -> bool) {
    for _ in 0..2_000 {
        if check() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("never: {what}");
}
