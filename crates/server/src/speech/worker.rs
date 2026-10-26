//! The one compute worker of the speech model (spec speech/02 §Admission,
//! Q44): a single thread that does one unit of work at a time.
//!
//! - **Interactive tasks** (Realtime sessions, spec speech/03) run first,
//!   each a window or less a step. Each session queues its own in order, and
//!   the sessions take turns, one step each: a task with steps left (a
//!   commit) stays first in its session's lane.
//! - **Batch jobs** (a transcription request) fill the gaps, first in first
//!   out, a unit at a time (a window, a refresh, or a step of the commit):
//!   between two steps of the job at the front the interactive tasks run, and
//!   the front job goes back to the front.
//! - At most `speech.max_queue` batch jobs wait behind the one running; the
//!   next is refused with a `Retry-After` that estimates when room frees.
//! - The same thread is the idle clock: with nothing to do, no Realtime
//!   session open, and the model unused for `speech.idle_unload` seconds, it
//!   unloads the model. An open session is a dictation in progress; its end
//!   restarts the clock. `speech.idle_unload` is read live: a change wakes the
//!   clock, which waits for the new value from the model's last use.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::executor::{DiarizerExecutor, NoDiarizer, SpeechExecutor};

/// A batch job, driven a window at a time.
pub trait Job: Send {
    /// The clip's length, for the estimate of when room frees.
    fn audio_seconds(&self) -> f64;
    /// One unit of work (a window, a refresh, a step of the commit), on the
    /// worker thread.
    fn step(&mut self) -> Step;
}

/// What a step left to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    More,
    Done,
}

/// An interactive task: a Realtime unit, one call to the executor, or a
/// commit, one step of it a call. `Step::More` keeps it first in its
/// session's lane, and the session's turn passes.
pub type Task = Box<dyn FnMut() -> Step + Send>;

/// One session's interactive tasks, in the order it queued them.
struct Lane {
    session: u64,
    tasks: VecDeque<Task>,
}

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
    /// The sessions with tasks queued, in turn order.
    interactive: VecDeque<Lane>,
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
    /// The diarizer beside it: unloaded with the speech model (spec
    /// speech/08).
    diarizer: Arc<dyn DiarizerExecutor>,
    gauges: Arc<Gauges>,
    /// `speech.idle_unload` in seconds, 0 never: the live value, which the
    /// speech slot writes.
    idle_unload_secs: Arc<AtomicU64>,
    last_used: Mutex<Instant>,
    /// The speed the worker has made, in thousandths of audio seconds per
    /// second: the estimate of how long a queued clip takes.
    speed_milli: AtomicU64,
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
/// time, the 4-core container's figure (`spec.md`, Requirements). A seed the
/// first finished job replaces, not a threshold an operator tunes.
const ASSUMED_SPEED_MILLI: u64 = 3000;

pub struct Worker {
    shared: Arc<Shared>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Worker {
    /// The worker, its idle clock reading `idle_unload_secs` (0 never).
    pub fn start(executor: Arc<dyn SpeechExecutor>, gauges: Arc<Gauges>, idle_unload_secs: Arc<AtomicU64>) -> Self {
        Self::start_with(executor, Arc::new(NoDiarizer), gauges, idle_unload_secs)
    }

    /// [`Worker::start`] with a diarizer whose weights go when the speech
    /// model's do.
    pub fn start_with(executor: Arc<dyn SpeechExecutor>, diarizer: Arc<dyn DiarizerExecutor>, gauges: Arc<Gauges>, idle_unload_secs: Arc<AtomicU64>) -> Self {
        let shared = Arc::new(Shared {
            state: Mutex::new(State::default()),
            wake: Condvar::new(),
            executor,
            diarizer,
            gauges,
            idle_unload_secs,
            last_used: Mutex::new(Instant::now()),
            speed_milli: AtomicU64::new(ASSUMED_SPEED_MILLI),
        });
        let running = Arc::clone(&shared);
        let thread = std::thread::Builder::new()
            .name("ignis-speech".to_owned())
            .spawn(move || running.run())
            .expect("the speech worker thread starts");
        Self { shared, thread: Mutex::new(Some(thread)) }
    }

    /// Set `speech.idle_unload` (the value the slot shares, tests) and wake the clock.
    #[cfg(test)]
    pub fn set_idle_unload(&self, secs: u32) {
        self.shared.idle_unload_secs.store(u64::from(secs), Ordering::SeqCst);
        self.wake();
    }

    /// Wake the idle clock to read `speech.idle_unload` again. The state lock
    /// is taken first: the clock reads the value and starts waiting under it,
    /// so the wake cannot fall between the two.
    pub fn wake(&self) {
        drop(self.shared.lock());
        self.shared.wake.notify_all();
    }

    /// Whether a batch job would be queued now: `Ok` unless `max_queue` jobs
    /// already wait behind the one running. A look, not a reservation — the
    /// room can go before the caller's [`Worker::submit`], which checks
    /// again — so a route asks it before the work that precedes a submit
    /// (decoding the clip) and still answers the submit's refusal.
    pub fn room(&self, max_queue: u32) -> Result<(), Full> {
        let state = self.shared.lock();
        self.shared.room(&state, max_queue)
    }

    /// Queue a batch job, unless `max_queue` jobs already wait behind the one
    /// running.
    pub fn submit(&self, job: Box<dyn Job>, max_queue: u32) -> Result<(), Full> {
        let mut state = self.shared.lock();
        self.shared.room(&state, max_queue)?;
        state.batch.push_back(Entry { job, computed: Duration::ZERO });
        self.shared.queue_changed(&state);
        drop(state);
        self.shared.touch();
        self.shared.wake.notify_all();
        Ok(())
    }

    /// Queue an interactive task of `session`'s, which runs after the ones
    /// it queued before, before any batch window, and in turn with the other
    /// sessions' tasks.
    pub fn submit_interactive(&self, session: u64, task: Task) {
        let mut state = self.shared.lock();
        match state.interactive.iter_mut().find(|lane| lane.session == session) {
            Some(lane) => lane.tasks.push_back(task),
            None => state.interactive.push_back(Lane { session, tasks: VecDeque::from([task]) }),
        }
        drop(state);
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

    /// `Err` when `max_queue` jobs already wait behind the one running, with
    /// when room frees.
    fn room(&self, state: &State, max_queue: u32) -> Result<(), Full> {
        if state.jobs() > max_queue as usize {
            let front = state.stepping.or_else(|| state.batch.front().map(|entry| entry.job.audio_seconds())).unwrap_or(0.0);
            return Err(Full { retry_after_secs: self.retry_after(front) });
        }
        Ok(())
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
            if let Some(mut lane) = state.interactive.pop_front() {
                let session = lane.session;
                let mut task = lane.tasks.pop_front().expect("a lane holds a task");
                // Its next task waits its turn behind the other sessions'.
                if !lane.tasks.is_empty() {
                    state.interactive.push_back(lane);
                }
                drop(state);
                // A panic in a task is that task's, not the worker's: it is over.
                let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| task())).unwrap_or(Step::Done);
                self.touch();
                if step == Step::More {
                    // First in its session's lane again, ahead of what the
                    // session queued since, with the turn passed on.
                    let mut state = self.lock();
                    match state.interactive.iter_mut().find(|lane| lane.session == session) {
                        Some(lane) => lane.tasks.push_front(task),
                        None => state.interactive.push_back(Lane { session, tasks: VecDeque::from([task]) }),
                    }
                }
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
            if unused >= idle && self.gauges.sessions.load(Ordering::SeqCst) > 0 {
                // A session open holds the model: its end touches the clock.
                drop(self.wake.wait_timeout(state, idle).expect("speech worker lock"));
                continue;
            }
            if unused >= idle {
                drop(state);
                self.executor.unload();
                self.diarizer.unload();
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
    use crate::speech::executor::{FakeDiarizer, FakeExecutor};
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
        (Worker::start(Arc::new(FakeExecutor::saying(&["x"])), Arc::clone(&gauges), Arc::default()), gauges)
    }

    /// A one-step interactive task that logs `unit`.
    fn logs(log: &Arc<Mutex<Vec<String>>>, unit: &str) -> Task {
        let (log, unit) = (Arc::clone(log), unit.to_owned());
        Box::new(move || {
            log.lock().unwrap().push(unit.clone());
            Step::Done
        })
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
        worker.submit_interactive(0, logs(&log, "rt"));
        open(&held);
        assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "a");
        assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "b");
        // The interactive task, queued while a's first window was held, runs
        // in the gap after it, ahead of a's other windows and all of b's.
        assert_eq!(*log.lock().unwrap(), ["a1", "rt", "a2", "a3", "b1", "b2"]);
    }

    /// Spec speech/03 AC 14: sessions interleave unit by unit, whatever each
    /// has queued, and every unit queued runs ahead of the batch job's next
    /// window.
    #[test]
    fn sessions_take_turns_unit_by_unit_ahead_of_batch_windows() {
        let (worker, _) = worker();
        let (done, finished) = mpsc::channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        let held = gate();
        let mut batch = job("b", 2, 16.0, &log, &done);
        batch.hold = Some(Arc::clone(&held));
        worker.submit(batch, 4).unwrap();
        while worker.shared.lock().stepping.is_none() {
            std::thread::sleep(Duration::from_millis(1));
        }
        // Session 1 queues three units, then session 2 two, while the batch
        // job's first window is held.
        for (session, unit) in [(1, "x1"), (1, "x2"), (1, "x3"), (2, "y1"), (2, "y2")] {
            worker.submit_interactive(session, logs(&log, unit));
        }
        open(&held);
        assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "b");
        assert_eq!(*log.lock().unwrap(), ["b1", "x1", "y1", "x2", "y2", "x3", "b2"]);
    }

    /// Spec speech/01 AC 19, 03 AC 14: an interactive task of several steps
    /// (a session's commit) takes one a turn, the other sessions' units
    /// between two, and its session's next task waits for its last.
    #[test]
    fn a_task_of_several_steps_takes_one_a_turn_and_keeps_its_place_in_its_lane() {
        let (worker, _) = worker();
        let (done, finished) = mpsc::channel();
        let log = Arc::new(Mutex::new(Vec::new()));
        let held = gate();
        let mut batch = job("b", 2, 16.0, &log, &done);
        batch.hold = Some(Arc::clone(&held));
        worker.submit(batch, 4).unwrap();
        while worker.shared.lock().stepping.is_none() {
            std::thread::sleep(Duration::from_millis(1));
        }
        let commit_log = Arc::clone(&log);
        let mut steps = 0;
        worker.submit_interactive(
            1,
            Box::new(move || {
                steps += 1;
                commit_log.lock().unwrap().push(format!("c{steps}"));
                if steps == 3 { Step::Done } else { Step::More }
            }),
        );
        worker.submit_interactive(1, logs(&log, "open"));
        worker.submit_interactive(2, logs(&log, "y1"));
        worker.submit_interactive(2, logs(&log, "y2"));
        open(&held);
        assert_eq!(finished.recv_timeout(Duration::from_secs(5)).unwrap(), "b");
        assert_eq!(*log.lock().unwrap(), ["b1", "c1", "y1", "c2", "y2", "c3", "open", "b2"]);
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

    /// Spec speech/03: a Realtime session open keeps the model loaded past
    /// `speech.idle_unload`, however quiet; the clock runs again from its
    /// end.
    #[test]
    fn an_open_session_keeps_the_model_loaded() {
        let executor = Arc::new(FakeExecutor::saying(&["x"]));
        let gauges = Arc::new(Gauges::default());
        let worker = Worker::start(executor.clone(), Arc::clone(&gauges), Arc::default());
        executor.loaded.store(true, Ordering::SeqCst);
        gauges.sessions.store(1, Ordering::SeqCst);
        worker.set_idle_unload(1);
        worker.touch();
        std::thread::sleep(Duration::from_millis(1_600));
        assert!(executor.is_loaded(), "a session is open");
        gauges.sessions.store(0, Ordering::SeqCst);
        worker.touch();
        let until = Instant::now() + Duration::from_secs(10);
        while executor.is_loaded() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!executor.is_loaded(), "unloaded once the session ended");
    }

    /// The idle clock lives on the worker: a loaded model unused for
    /// `speech.idle_unload` is unloaded, one in use is not.
    #[test]
    fn a_model_unused_for_the_idle_time_is_unloaded() {
        let executor = Arc::new(FakeExecutor::saying(&["x"]));
        let worker = Worker::start(executor.clone(), Arc::new(Gauges::default()), Arc::default());
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

    /// Spec speech/08 (AC 15): the diarizer is unloaded with the speech
    /// model at `speech.idle_unload`, and not before.
    #[test]
    fn the_diarizer_is_unloaded_with_the_speech_model() {
        let executor = Arc::new(FakeExecutor::saying(&["x"]));
        let diarizer = Arc::new(FakeDiarizer::hearing(|_| None));
        let worker = Worker::start_with(executor.clone(), diarizer.clone(), Arc::new(Gauges::default()), Arc::default());
        executor.loaded.store(true, Ordering::SeqCst);
        diarizer.loaded.store(true, Ordering::SeqCst);
        worker.set_idle_unload(1);
        worker.touch();
        std::thread::sleep(Duration::from_millis(300));
        assert!(executor.is_loaded() && diarizer.is_loaded(), "not yet idle for a second");
        let until = Instant::now() + Duration::from_secs(10);
        while executor.is_loaded() && Instant::now() < until {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!executor.is_loaded() && !diarizer.is_loaded(), "both go together");
    }

    /// `speech.idle_unload` is read live: a clock waiting on 0 (never) or on
    /// a long time takes a shorter value as soon as it changes, counted from
    /// the model's last use.
    #[test]
    fn a_live_change_of_the_idle_time_reaches_a_waiting_clock() {
        for before in [0, 300] {
            let executor = Arc::new(FakeExecutor::saying(&["x"]));
            let idle = Arc::new(AtomicU64::new(before));
            let worker = Worker::start(executor.clone(), Arc::new(Gauges::default()), Arc::clone(&idle));
            executor.loaded.store(true, Ordering::SeqCst);
            worker.touch();
            // The clock is waiting on `before` by now.
            std::thread::sleep(Duration::from_millis(200));
            assert!(executor.is_loaded());
            idle.store(1, Ordering::SeqCst);
            worker.wake();
            let until = Instant::now() + Duration::from_secs(10);
            while executor.is_loaded() && Instant::now() < until {
                std::thread::sleep(Duration::from_millis(20));
            }
            assert!(!executor.is_loaded(), "from {before} s to 1 s: still loaded");
        }
    }
}
