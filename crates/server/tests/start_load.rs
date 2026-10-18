//! GitHub #329 items 17 and 18: the process during its start-up load.
//!
//! - The API listener answers from before the load (`starting::Starting`):
//!   `GET /health` 200, `/ready` and every other path 503 `server_not_ready`,
//!   then the server takes the listener over on the same address, with no
//!   window where a connection is refused and no start's keep-alive
//!   connection left reading 503.
//! - On unix, the real binary in a load that does not end (its artifact a
//!   FIFO no one writes): the probes answer, and a SIGTERM ends the process
//!   at once with its shutdown lines, where it used to be dropped until the
//!   load was over. The same for a switch's load while serving, which the
//!   exit used to wait for. Windows has no SIGTERM, and a Ctrl+C cannot be
//!   sent to a child that shares no console with the test: the stop's own
//!   logic is the unit tests' in `shutdown.rs`, the serve's
//!   `serve_exit_load.rs`'s.

use std::sync::Arc;
use std::time::Duration;

use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::engine::Engine;
use ignis_server::starting::Starting;
use ignis_server::template::SimpleTemplateProvider;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const BOUND: Duration = Duration::from_secs(5);

fn server() -> Server {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: "test-model".into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
}

/// One HTTP/1.1 `GET path` over a fresh connection, with `header` if any;
/// the whole response.
async fn get(addr: std::net::SocketAddr, path: &str, header: &str) -> String {
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n{header}Connection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.expect("write");
    let mut response = String::new();
    tokio::time::timeout(BOUND, stream.read_to_string(&mut response)).await.expect("answered").expect("read");
    response
}

fn is_starting_refusal(response: &str) -> bool {
    response.starts_with("HTTP/1.1 503")
        && response.to_ascii_lowercase().contains("\r\nretry-after: 5\r\n")
        && response.contains("\"server_not_ready\"")
}

#[tokio::test]
async fn while_starting_health_answers_and_everything_else_is_not_ready() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let starting = Starting::answer(listener).expect("answer");
    let addr = starting.local_addr();

    let health = get(addr, "/health", "").await;
    assert!(health.starts_with("HTTP/1.1 200"), "{health}");
    assert!(health.contains(r#"{"status":"ok"}"#), "{health}");
    for path in ["/ready", "/v1/models", "/v1/chat/completions", "/ui/"] {
        let held = get(addr, path, "").await;
        assert!(is_starting_refusal(&held), "{path}: {held}");
    }
    // Whatever the key: a keyed server's readiness probe sends none, and a
    // 401 would read as "up" to `make start`.
    let keyed = get(addr, "/v1/models", "Authorization: Bearer sk-anything\r\n").await;
    assert!(is_starting_refusal(&keyed), "{keyed}");
}

#[tokio::test]
async fn the_server_takes_the_listener_over_on_the_same_address() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let starting = Starting::answer(listener).expect("answer");
    let addr = starting.local_addr();

    // A client that keeps its connection to the start (HTTP/1.1 keep-alive).
    let mut kept = tokio::net::TcpStream::connect(addr).await.expect("connect");
    kept.write_all(b"GET /ready HTTP/1.1\r\nHost: localhost\r\n\r\n").await.expect("write");
    let mut seen = Vec::new();
    let mut chunk = [0u8; 1024];
    while !String::from_utf8_lossy(&seen).contains("server_not_ready") {
        let n = tokio::time::timeout(BOUND, kept.read(&mut chunk)).await.expect("answered").expect("read");
        assert!(n > 0, "the start closed a keep-alive connection before the hand-over");
        seen.extend_from_slice(&chunk[..n]);
    }

    let listener = tokio::time::timeout(BOUND, starting.finish()).await.expect("the listener came back");
    assert_eq!(listener.local_addr().expect("still bound"), addr);
    // The kept connection is closed, not left answering 503 from the start.
    let n = tokio::time::timeout(BOUND, kept.read(&mut chunk)).await.expect("closed").expect("read");
    assert_eq!(n, 0, "the start's keep-alive connection was closed at the hand-over");

    // A connection made before the server accepts waits in the backlog and
    // is answered by the server: no refused window between the two.
    let mut early = tokio::net::TcpStream::connect(addr).await.expect("connect between start and serve");
    early.write_all(b"GET /ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n").await.expect("write");

    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serving = tokio::spawn(server().serve_on_until(listener, None, async {
        let _ = stopped.await;
    }));
    let mut answer = String::new();
    tokio::time::timeout(BOUND, early.read_to_string(&mut answer)).await.expect("answered").expect("read");
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    let models = get(addr, "/v1/models", "").await;
    assert!(models.starts_with("HTTP/1.1 200") && models.contains("test-model"), "{models}");

    stop.send(()).expect("serving");
    let served = tokio::time::timeout(BOUND, serving).await.expect("serve returned").expect("join").expect("serve");
    assert_eq!(served, ignis_server::Served::Drained);
}

#[cfg(unix)]
mod process {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::os::unix::ffi::OsStrExt;
    use std::path::{Path, PathBuf};
    use std::process::{Child, Command, ExitStatus, Stdio};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use serde_json::Value;

    /// Kills the child (and reaps it) when dropped: a failed assertion must
    /// not leave a process waiting on the FIFO.
    struct KillOnDrop(Child);

    impl Drop for KillOnDrop {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    /// One HTTP/1.1 `method path` with `body`; the whole response.
    fn request(addr: &str, method: &str, path: &str, body: &str) -> String {
        let mut stream = std::net::TcpStream::connect(addr).expect("connect");
        stream.set_read_timeout(Some(Duration::from_secs(5))).expect("timeout");
        write!(
            stream,
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .expect("write");
        let mut response = String::new();
        stream.read_to_string(&mut response).expect("read");
        response
    }

    fn get(addr: &str, path: &str) -> String {
        request(addr, "GET", path, "")
    }

    /// A fresh directory of the test's own.
    fn temp_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ignis-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// An artifact in `dir` whose load never ends on its own: a FIFO no one
    /// writes to, whose open waits for as long as the process lives. The
    /// sidecar is the one file the load looks for before.
    fn stalled_artifact(dir: &Path) -> PathBuf {
        let artifact = dir.join("stalled.ninfer");
        let path = std::ffi::CString::new(artifact.as_os_str().as_bytes()).expect("path");
        // SAFETY: a NUL-terminated path and a mode.
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0, "mkfifo: {}", std::io::Error::last_os_error());
        std::fs::write(dir.join("stalled.ninfer.graft.json"), b"{}").expect("sidecar");
        artifact
    }

    /// The real binary in `dir`, on `artifact` or the placeholder start, its
    /// log lines read off stdout as they come.
    fn spawn(dir: &Path, artifact: Option<&Path>) -> (KillOnDrop, mpsc::Receiver<Value>) {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ignis-server"));
        if let Some(artifact) = artifact {
            command.args(["--model-artifact", artifact.to_str().expect("utf-8")]);
        }
        let child = command
            .args(["--server-bind", "127.0.0.1:0", "--server-metrics", "false"])
            .args(["--download-enabled", "false", "--download-path", dir.to_str().expect("utf-8")])
            .env("IGNIS_LOG_FORMAT", "json")
            .env("IGNIS_LOG_LEVEL", "info")
            // No config file on the machine (spec config-v2/02 §Config-file
            // auto-discovery).
            .current_dir(dir)
            .env("XDG_CONFIG_HOME", dir)
            .env_remove("IGNIS_CONFIG")
            .env_remove("IGNIS_PROFILE")
            .env_remove("IGNIS_MODEL_ARTIFACT")
            .env_remove("IGNIS_SERVER_METRICS")
            .env_remove("IGNIS_SERVER_METRICS_BIND")
            .stdout(Stdio::piped())
            .spawn()
            .expect("spawn ignis-server");
        let mut child = KillOnDrop(child);
        let stdout = child.0.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::channel::<Value>();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                let record = serde_json::from_str(&line)
                    .unwrap_or_else(|error| panic!("not a logging JSON record: {error}: {line:?}"));
                if tx.send(record).is_err() {
                    break;
                }
            }
        });
        (child, rx)
    }

    /// Read `rx` into `records` up to the first `name` line.
    fn read_until(rx: &mpsc::Receiver<Value>, records: &mut Vec<Value>, name: &str) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !records.iter().any(|r| r["event_name"] == name) {
            let left = deadline.saturating_duration_since(Instant::now());
            let record = rx.recv_timeout(left).unwrap_or_else(|_| panic!("no {name}; saw {records:?}"));
            records.push(record);
        }
    }

    /// The address the API listener is bound to, from its line.
    fn address(records: &[Value]) -> String {
        let listening = records
            .iter()
            .find(|r| r["event_name"] == "ignis.server.listening")
            .unwrap_or_else(|| panic!("no ignis.server.listening: {records:?}"));
        listening["attributes"]["address"].as_str().expect("address").to_owned()
    }

    /// SIGTERM `child`, and its exit status once it ends: within 10 s.
    fn sigterm(child: &mut KillOnDrop) -> ExitStatus {
        let sent = Instant::now();
        // SAFETY: a signal to our own child, still running (its handle is held).
        assert_eq!(unsafe { libc::kill(child.0.id() as libc::pid_t, libc::SIGTERM) }, 0);
        loop {
            if let Some(status) = child.0.try_wait().expect("wait") {
                return status;
            }
            assert!(sent.elapsed() < Duration::from_secs(10), "still running 10 s after the SIGTERM");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn names(records: &[Value]) -> Vec<&str> {
        records.iter().filter_map(|r| r["event_name"].as_str()).collect()
    }

    #[test]
    fn a_sigterm_during_the_start_up_load_ends_the_process_at_once() {
        let dir = temp_dir("start-load");
        let artifact = stalled_artifact(&dir);
        let (mut child, rx) = spawn(&dir, Some(&artifact));

        // Up to the load's first line: from there the start is in its load.
        let mut records = Vec::new();
        read_until(&rx, &mut records, "ignis.model.loading");
        let addr = address(&records);

        // Item 18: alive, and not ready.
        let health = get(&addr, "/health");
        assert!(health.starts_with("HTTP/1.1 200"), "{health}");
        let ready = get(&addr, "/ready");
        assert!(ready.starts_with("HTTP/1.1 503") && ready.contains("server_not_ready"), "{ready}");

        // Item 17: a SIGTERM in the load ends the process, now.
        let status = sigterm(&mut child);
        assert_eq!(status.code(), Some(0), "a requested stop, exited by itself: {status:?}");

        records.extend(rx.iter());
        let names = names(&records);
        let stopping = names.iter().position(|&n| n == "ignis.process.stopping").unwrap_or_else(|| panic!("{names:?}"));
        assert_eq!(names.last(), Some(&"ignis.process.stopped"), "the stop's line is the last: {names:?}");
        assert!(stopping < names.len() - 1, "{names:?}");
        assert!(!names.contains(&"ignis.process.started"), "it never served: {names:?}");
        assert_eq!(records.last().expect("records")["attributes"]["phase"], "loading", "{records:?}");

        drop(child);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Item 17 while serving: a switch whose load never ends — the 202
    /// answered, its prepare waiting on the FIFO — does not hold the exit. A
    /// SIGTERM drains HTTP and ends the process under the load, where the
    /// exit used to wait for it.
    #[test]
    fn a_sigterm_during_a_switch_ends_the_process_without_waiting_for_its_load() {
        let dir = temp_dir("switch-load");
        let artifact = stalled_artifact(&dir);
        // The placeholder start: serving at once, a switch away from it allowed.
        let (mut child, rx) = spawn(&dir, None);
        let mut records = Vec::new();
        read_until(&rx, &mut records, "ignis.process.started");
        let addr = address(&records);

        let body = serde_json::json!({ "artifact": artifact.to_str().expect("utf-8"), "model": "stalled" }).to_string();
        let switch = request(&addr, "POST", "/v1/models/switch", &body);
        assert!(switch.starts_with("HTTP/1.1 202"), "{switch}");
        read_until(&rx, &mut records, "ignis.model.switch_started");

        let status = sigterm(&mut child);
        assert_eq!(status.code(), Some(0), "a requested stop, exited by itself: {status:?}");

        records.extend(rx.iter());
        let names = names(&records);
        assert!(names.contains(&"ignis.process.stopping"), "{names:?}");
        assert_eq!(names.last(), Some(&"ignis.process.stopped"), "the stop's line is the last: {names:?}");
        let stopped = &records.last().expect("records")["attributes"];
        assert_eq!((stopped["phase"].as_str(), stopped["to"].as_str()), (Some("switching"), Some("stalled")), "{records:?}");
        assert!(!names.contains(&"ignis.model.switched"), "the load never ended: {names:?}");

        drop(child);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
