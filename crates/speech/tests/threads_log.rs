//! Acceptance 20's log line, in its own test binary (a scoped subscriber sees nothing when a
//! parallel test registers the callsite first): a load logs `ignis.speech.threads` with the
//! count and whether it came from `auto` or the configuration.

mod common;

use std::sync::{Arc, Mutex};

use ignis_speech::{Executor, Features, Settings, Threads};
use tracing_subscriber::layer::SubscriberExt;

struct Capture(Arc<Mutex<Vec<String>>>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Capture {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        struct V(String);
        impl tracing::field::Visit for V {
            fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
                self.0.push_str(&format!(" {}={v:?}", f.name()));
            }
        }
        let mut v = V(event.metadata().name().to_string());
        event.record(&mut v);
        self.0.lock().unwrap().push(v.0);
    }
}

#[test]
fn a_load_logs_its_threads_and_their_source() {
    let files = common::synthetic::pair("log");
    let lines = Arc::new(Mutex::new(Vec::new()));
    let sub = tracing_subscriber::registry().with(Capture(lines.clone()));
    tracing::subscriber::with_default(sub, || {
        for t in [Threads::Auto, Threads::Count(3)] {
            let e = Executor::new(Settings { threads: t, ..Settings::default() }, files.clone(), Features::detect());
            if e.load().is_err() {
                panic!("this CPU lacks AVX2: the executor cannot serve here");
            }
        }
    });
    let lines = lines.lock().unwrap();
    let threads: Vec<&String> = lines.iter().filter(|l| l.starts_with("ignis.speech.threads")).collect();
    let auto = Threads::Auto.resolve().0;
    assert_eq!(threads.len(), 2, "{lines:?}");
    assert!(threads[0].contains(&format!("threads={auto}")) && threads[0].contains("source=\"auto\""), "{}", threads[0]);
    assert!(threads[1].contains("threads=3") && threads[1].contains("source=\"configured\""), "{}", threads[1]);
    assert!(lines.iter().any(|l| l.starts_with("ignis.speech.load")));
}
