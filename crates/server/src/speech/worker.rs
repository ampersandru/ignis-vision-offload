//! The one compute worker of the speech model (spec speech/02 §Admission,
//! Q44): a single thread that does one unit of work at a time.
//!
//! - **Interactive tasks** (Realtime sessions, spec speech/03) run first,
//!   each a window or less, and interleave.
//! - **Batch jobs** (a transcription request) fill the gaps, first in first
//!   out, a window at a time: between two windows of the job at the front the
//!   interactive tasks run, and the front job goes back to the front.
//! - At most `speech.max_queue` batch jobs wait behind the one running; the
//!   next is refused with a `Retry-After` that estimates when room frees.
//! - The same thread is the idle clock: with nothing to do, and the model
//!   unused for `speech.idle_unload` seconds, it unloads the model.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::executor::SpeechExecutor;

/// A batch job, driven a window at a time.
pub trait Job: Send {
    /// The clip's length, for the estimate of when room frees.
    fn audio_seconds(&self) -> f64;
    /// One window of work (or the commit), on the worker thread.
    fn step(&mut self) -> Step;
}

/// What a step left to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    More,
    Done,
}

/// An interactive task: a Realtime window, one call to the executor.
pub type Task = Box<dyn FnOnce() + Send>;

/// The gauges `ignis_speech_sessions_open` and `ignis_speech_queue_length`
/// read, and where they are published.
#[derive(Default)]
pub struct Gauges {
    pub sessions: AtomicUsize,
    pub queue: AtomicUsize,
    sink: Mutex<Option<Arc<crate::metrics::Metrics>>>,
}

impl Gauges {
    /// Publish to `metrics` from now on: the families appear with the first
    /// transcription or session, not before.
    pub fn attach(&self, metrics: Arc<crate::metrics::Metrics>) {
        *self.sink.lock().expect("gauge sink lock") = Some(metrics);
    }

    pub fn publish(&self) {
        if let Some(metrics) = self.sink.lock().expect("gauge sink lock").as_ref() {
            metrics.set_speech_gauges(self.sessions.load(Ordering::SeqCst) as u32, self.queue.load(Ordering::SeqCst) as u32);
        }
    }
}

struct Entry {
    job: Box<dyn Job>,
    /// What its steps have taken so far.
    computed: Duration,
}

#[derive(Default)]
struct State {
    interactive: VecDeque<Task>,
    batch: VecDeque<Entry>,
    /// The audio seconds of the job being stepped now, if one is.
    stepping: Option<f64>,
    stop: bool,
}

impl State {
    /// The jobs in the system: queued and the one being stepped.
    fn jobs(&self) -> usize {
        self.batch.len() + usize::from(self.stepping.is_some())
    }
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    executor: Arc<dyn SpeechExecutor>,
    gauges: Arc<Gauges>,
    /// `speech.idle_unload` in seconds, 0 never.
    idle_unload_secs: AtomicU64,
    last_used: Mutex<Instant>,
    /// The speed the worker has made, in thousandths of audio seconds per
    /// second: the estimate of how long a queued clip takes.
    speed_milli: AtomicU64,
    stopped: AtomicBool,
}

/// The batch queue is full: when room frees, estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Full {
    pub retry_after_secs: u64,
}

/// The longest `Retry-After` any speech refusal gives, like the engine's own
/// (`RETRY_AFTER_OVERDUE_MAX_SECS`).
pub const RETRY_AFTER_MAX_SECS: u64 = crate::engine::RETRY_AFTER_OVERDUE_MAX_SECS;

/// What a clip is assumed to run at before the worker has run one: 3 x real
/// time, the 4-core container's figure (`spec.md`, Requirements).
const ASSUMED_SPEED_MILLI: u64 = 3000;

pub struct Worker {
    shared: Arc<Shared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Worker {
    pub fn start(executor: Arc<dyn SpeechExecutor>, gauges: Arc<Gauges>) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            executor,
            gauges,
            idle_unload_secs: AtomicU64::new(0),
            last_used: Mutex::new(Instant::now()),
            speed_milli: AtomicU64::new(ASSUMED_SPEED_MILLI),
            stopped: AtomicBool::new(false),
        });
        let running = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("ignis-speech".to_owned())
            .spawn(move || running.run())
            .expect("the speech worker thread starts");
        Self { shared, thread: Mutex::new(Some(thread)) }
    }

    /// `speech.idle_unload`, read live.
    pub fn set_idle_unload(&self, secs: u32) {
        let before = self.shared.idle_unload_secs.swap(u64::from(secs), Ordering::SeqCst);
        if before != u64::from(secs) {
            self.shared.wake.notify_all();
        }
    }

    /// Queue a batch job, unless `max_queue` jobs already wait behind the one
    /// running.
    pub fn submit(&self, job: Box<dyn Job>, max_queue: u32) -> Result<(), Full> {
        let mut state = self.shared.lock();
        if state.jobs() > max_queue as usize {
            let front = state.stepping.or_else(|| state.batch.front().map(|entry| entry.job.audio_seconds())).unwrap_or(0.0);
            return Err(Full { retry_after_secs: self.shared.retry_after(front) });
        }
        state.batch.push_back(Entry { job, computed: Duration::ZERO });
        self.shared.queue_changed(&state);
        drop(state);
        self.shared.touch();
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Queue an interactive task, which runs before any batch window.
    pub fn submit_interactive(&self, task: Task) {
        self.shared.lock().interactive.push_back(task);
        self.shared.touch();
        self.shared.wake.notify_all();
    }

    /// The jobs in the system now.
    pub fn jobs(&self) -> usize {
        self.shared.lock().jobs()
    }

    /// The model was used just now: the idle clock restarts.
    pub fn touch(&self) {
        self.shared.touch();
    }

    /// How long since the model was last used.
    pub fn idle_for(&self) -> Duration {
        self.shared.last_used.lock().expect("last used lock").elapsed()
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shared.lock().stop = true;
        self.shared.stopped.store(true, Ordering::SeqCst);
        self.shared.wake.notify_all();
        if let Some(thread) = self.thread.lock().expect("worker thread lock").take() {
            let _ = thread.join();
        }
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("speech worker lock")
    }

    fn touch(&self) {
        *self.last_used.lock().expect("last used lock") = Instant::now();
    }

    fn queue_changed(&self, state: &State) {
        // The gauge counts the batch requests waiting, not the one running.
        self.gauges.queue.store(state.jobs().saturating_sub(1), Ordering::SeqCst);
        self.gauges.publish();
    }

    /// Seconds until room frees: the clip at the front, at the speed made so
    /// far, rounded up, at least one, at most [`RETRY_AFTER_MAX_SECS`].
    fn retry_after(&self, front_audio_seconds: f64) -> u64 {
        let speed = self.speed_milli.load(Ordering::SeqCst).max(1) as f64 / 1000.0;
        ((front_audio_seconds / speed).ceil() as u64).clamp(1, RETRY_AFTER_MAX_SECS)
    }

    fn run(self: &Arc<Self>) {
        loop {
            let mut state = self.lock();
            if state.stop {
                return;
            }
            if let Some(task) = state.interactive.pop_front() {
                drop(state);
                // A panic in a task is that task's, not the worker's.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task));
                self.touch();
                continue;
            }
            if let Some(mut entry) = state.batch.pop_front() {
                state.stepping = Some(entry.job.audio_seconds());
                drop(state);
                let began = Instant::now();
                // A job that panics (an executor fault) is over: its request hears
                // its reply dropped, and the worker serves the next.
                let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| entry.job.step())).unwrap_or(Step::Done);
                entry.computed += began.elapsed();
                self.touch();
                let mut state = self.lock();
                state.stepping = None;
                match step {
                    // Back to the front: first in, first out, with the
                    // interactive tasks in the gap.
                    Step::More => state.batch.push_front(entry),
                    Step::Done => self.learn(entry.job.audio_seconds(), entry.computed),
                }
                self.queue_changed(&state);
                continue;
            }
            // Nothing to do: wait, for work or for the idle clock.
            let idle = Duration::from_secs(self.idle_unload_secs.load(Ordering::SeqCst));
            if idle.is_zero() || !self.executor.is_loaded() {
                drop(self.wake.wait(state).expect("speech worker lock"));
                continue;
            }
            let unused = self.last_used.lock().expect("last used lock").elapsed();
            if unused >= idle {
                drop(state);
                self.executor.unload();
                tracing::info!(
                    name: "ignis.speech.unloaded",
                    idle_seconds = unused.as_secs(),
                    "the speech model was unused for speech.idle_unload and is unloaded"
                );
                continue;
            }
            drop(self.wake.wait_timeout(state, idle - unused).expect("speech worker lock"));
        }
    }

    /// A job ended: fold its speed into the estimate (a moving average, so
    /// one odd clip does not set it).
    fn learn(&self, audio_seconds: f64, computed: Duration) {
        if computed < Duration::from_millis(50) || audio_seconds <= 0.0 {
            return;
        }
        let seen = (audio_seconds / computed.as_secs_f64() * 1000.0) as u64;
        let before = self.speed_milli.load(Ordering::SeqCst);
        self.speed_milli.store((before * 3 + seen) / 4, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::speech::executor::FakeExecutor;
    use std::sync::mpsc;

    /// A job of `windows` steps that logs each into `log` under its name.
    struct Windows {
        name: &'static str,
        windows: u32,
        seen: u32,
        seconds: f64,
        log: Arc<Mutex<Vec<String>>>,
        done: mpsc::Sender<&'static str>,
        hold: Option<Arc<(Mutex<bool>, Condvar)>>,
    }

    impl Job for Windows {
        fn audio_seconds(&self) -> f64 {
            self.seconds
        }

        fn step(&mut self) -> Step {
            if let Some(hold) = self.hold.take() {
                let (gate, opened) = &*hold;
                let mut open = gate.lock().unwrap();
                while !*open {
                    open = opened.wait(open).unwrap();
                }
            }
            self.seen += 1;
            self.log.lock().unwrap().push(format!("{}{}", self.name, self.seen));
            if self.seen == self.windows {
                let _ = self.done.send(self.name);
                Step::Done
            } else {
                Step::More
            }
        }
    }

    fn worker() -> (Worker, Arc<Gauges>) {
        let gauges = Arc::new(Gauges::default());
        (Worker::start(Arc::new(FakeExecutor::saying(&["x"])), Arc::clone(&gauges)), gauges)
    }

    fn job(name: &'static str, windows: u32, seconds: f64, log: &Arc<Mutex<Vec<String>>>, done: &mpsc::Sender<&'static str>) -> Box<Windows> {
        Box::new(Windows { name, windows, seen: 0, seconds, log: Arc::clone(log), done: done.clone(), hold: None })
    }

    fn gate() -> Arc<(Mutex<bool>, Condvar)> {
        Arc::new((Mutex::new(false), Condvar::new()))
    }

    fn open(gate: &Arc<(Mutex<bool>, Condvar)>) {
        *gate.0.lock().unwrap() = true;
        gate.1.notify_all();
    }

    /// Spec speech/02 AC 11: batch jobs run first in, first out, a window at a
    /// time, and an interactive task queued meanwhile runs in the next gap,
    /// ahead of every batch window waiting.
    #[test]
    fn batch_runs_first_in_first_out_and_interactive_takes_the_gaps() {
        let (worker, _) = worker();
        let (done, finished) = mpsc::channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        let held = gate();
        let mut first = job("a", 3, 24.0, &log, &done);
        first.hold = Some(Arc::clone(&held));
        worker.submit(first, 4).unwrap();
        while worker.shared.lock().stepping.is_none() {
            std::thread::sleep(Duration::from_millis(1));
        }
        worker.submit(job("b", 2, 16.0, &log, &done), 4).unwrap();
        let task_log = Arc::clone(&log);
        worker.submit_interactive(Box::new(move || task_log.lock().unwrap().push("rt".to_owned())));
        open(&held);
        assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "a");
        assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "b");
        // The interactive task, queued while a's first window was held, runs
        // in the gap after it, ahead of a's other windows and all of b's.
        assert_eq!(*log.lock().unwrap(), ["a1", "rt", "a2", "a3", "b1", "b2"]);
    }

    /// AC 11: past `max_queue` waiting jobs the next is refused, with a
    /// `Retry-After` of one to fifteen seconds; with room it is taken.
    #[test]
    fn past_max_queue_waiting_jobs_the_next_is_refused_with_a_bounded_retry_after() {
        let (worker, gauges) = worker();
        let (done, finished) = mpsc::channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        let held = gate();
        let mut running = job("run", 1, 600.0, &log, &done);
        running.hold = Some(Arc::clone(&held));
        worker.submit(running, 2).unwrap();
        // Wait until it is the one running, not one waiting.
        while worker.shared.lock().stepping.is_none() {
            std::thread::sleep(Duration::from_millis(1));
        }
        worker.submit(job("w1", 1, 10.0, &log, &done), 2).unwrap();
        worker.submit(job("w2", 1, 10.0, &log, &done), 2).unwrap();
        assert_eq!(gauges.queue.load(Ordering::SeqCst), 2, "two wait behind the one running");
        let full = worker.submit(job("w3", 1, 10.0, &log, &done), 2).unwrap_err();
        assert!((1..=RETRY_AFTER_MAX_SECS).contains(&full.retry_after_secs), "{full:?}");
        // A 600 s clip at 3 x real time is 200 s: capped.
        assert_eq!(full.retry_after_secs, RETRY_AFTER_MAX_SECS);
        open(&held);
        for expected in ["run", "w1", "w2"] {
            assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), expected);
        }
        while worker.jobs() > 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(gauges.queue.load(Ordering::SeqCst), 0);
        worker.submit(job("w4", 1, 10.0, &log, &done), 2).expect("room again");
    }

    /// `max_queue` 0 runs one at a time with none waiting.
    #[test]
    fn a_queue_of_zero_takes_one_job_at_a_time() {
        let (worker, _) = worker();
        let (done, finished) = mpsc::channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        let held = gate();
        let mut first = job("a", 1, 2.0, &log, &done);
        first.hold = Some(Arc::clone(&held));
        worker.submit(first, 0).unwrap();
        let full = worker.submit(job("b", 1, 2.0, &log, &done), 0).unwrap_err();
        assert_eq!(full.retry_after_secs, 1, "a 2 s clip is a second away");
        open(&held);
        assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "a");
    }

    /// A job whose executor panics ends there, and the worker serves the next.
    #[test]
    fn a_job_that_panics_does_not_take_the_worker_with_it() {
        struct Panics;
        impl Job for Panics {
            fn audio_seconds(&self) -> f64 {
                1.0
            }
            fn step(&mut self) -> Step {
                panic!("the executor faulted");
            }
        }
        let (worker, gauges) = worker();
        let (done, finished) = mpsc::channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        worker.submit(Box::new(Panics), 4).unwrap();
        worker.submit(job("after", 1, 1.0, &log, &done), 4).unwrap();
        assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "after");
        while worker.jobs() > 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(gauges.queue.load(Ordering::SeqCst), 0);
    }

    /// The idle clock lives on the worker: a loaded model unused for
    /// `speech.idle_unload` is unloaded, one in use is not.
    #[test]
    fn a_model_unused_for_the_idle_time_is_unloaded() {
        let executor = Arc::new(FakeExecutor::saying(&["x"]));
        let worker = Worker::start(executor.clone(), Arc::new(Gauges::default()));
        executor.loaded.store(true, Ordering::SeqCst);
        worker.set_idle_unload(1);
        worker.touch();
        std::thread::sleep(Duration::from_millis(300));
        assert!(executor.is_loaded(), "not yet idle for a second");
        let until = Instant::now() + Duration::from_secs(10);
        while executor.is_loaded() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!executor.is_loaded());
        assert_eq!(executor.unloads.load(Ordering::SeqCst), 1);
    }
}
