//! The diarizer's streaming state (spec speech/08): the arrival-order speaker cache (AOSC) and the
//! FIFO of the latest frames, which every step attends to beside its chunk.
//!
//! Both hold rows of the embedder's output, before the encoder: each step encodes them again with
//! its chunk. A step's chunk joins the FIFO; once the FIFO overflows, at least the update period's
//! oldest frames (and as many as it takes to fit again) move to the cache, scored by the
//! probabilities of the step that moves them. Once the cache overflows it is compressed: each frame
//! is scored per speaker from its probabilities, the most recent frames, every speaker's strongest
//! frames and their next strongest are boosted, each speaker keeps its silence slots, and the cache
//! keeps its length's best (speaker, frame) pairs, grouped by speaker in arrival order and in their
//! order within a speaker; a pair scored −∞ is a silence slot. A compressed cache keeps the
//! probabilities it was scored with; an uncompressed one is scored again by each step.
//!
//! The order of equal scores, which torch's top-k leaves undefined, is the lower index first.
//!
//! Provenance: the module's (`diarizer/mod.rs`, ADR 0051).

use super::{Constants, Geometry, SPEAKERS};

/// A frame's speaker probabilities at 80 ms.
pub type Probs = [f32; SPEAKERS];

#[derive(Clone)]
pub struct SpeakerCache {
    width: usize,
    geometry: Geometry,
    constants: Constants,
    min_positive: usize,
    strong: usize,
    weak: usize,
    cache: Vec<f32>,
    cache_probs: Vec<Probs>,
    fifo: Vec<f32>,
    compressed: bool,
}

impl SpeakerCache {
    /// An empty state for rows of `width`.
    pub fn new(width: usize, geometry: Geometry, constants: Constants) -> SpeakerCache {
        // Each speaker's share of the cache, its silence slots excluded, and the frame counts the
        // scores are spent on.
        let budget = (geometry.speaker_cache / SPEAKERS).saturating_sub(constants.speaker_cache_silence_frames_per_speaker) as f64;
        let floor = |rate: f32| (budget * rate as f64).floor() as usize;
        SpeakerCache {
            width,
            geometry,
            constants,
            min_positive: floor(constants.min_positive_scores_rate),
            strong: floor(constants.strong_boost_rate),
            weak: floor(constants.weak_boost_rate),
            cache: Vec::new(),
            cache_probs: Vec::new(),
            fifo: Vec::new(),
            compressed: false,
        }
    }

    /// The cache's frames.
    pub fn cache_len(&self) -> usize {
        self.cache.len() / self.width
    }

    /// The FIFO's frames.
    pub fn fifo_len(&self) -> usize {
        self.fifo.len() / self.width
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }

    /// The rows a step attends to before its chunk: the cache's, then the FIFO's.
    pub fn rows(&self) -> impl Iterator<Item = &[f32]> {
        self.cache.chunks_exact(self.width).chain(self.fifo.chunks_exact(self.width))
    }

    pub fn cache_rows(&self) -> &[f32] {
        &self.cache
    }

    pub fn fifo_rows(&self) -> &[f32] {
        &self.fifo
    }

    /// The probabilities the cache's frames were last stored with.
    pub fn cache_probs(&self) -> &[Probs] {
        &self.cache_probs
    }

    /// Sets the state to these rows and the cache's stored probabilities (the oracle's forcing).
    #[doc(hidden)]
    pub fn set(&mut self, cache: Vec<f32>, cache_probs: Vec<Probs>, fifo: Vec<f32>, compressed: bool) {
        assert_eq!(cache.len(), cache_probs.len() * self.width);
        self.cache_probs = cache_probs;
        self.cache = cache;
        self.fifo = fifo;
        self.compressed = compressed;
    }

    /// After a step over `inputs` (the cache's rows, the FIFO's, the chunk's and its look-ahead,
    /// rows of `width`) with `probs` (the step's probabilities of every input row): the `chunk`
    /// frames after the cache and the FIFO join the FIFO, and the cache takes what overflows it.
    pub fn update(&mut self, inputs: &[f32], probs: &[Probs], silence: &[f32], chunk: usize) {
        let w = self.width;
        let (n_cache, n_fifo) = (self.cache_len(), self.fifo_len());
        assert_eq!(inputs.len() / w, probs.len());
        assert!(n_cache + n_fifo + chunk <= probs.len() && silence.len() == w);
        let start = n_cache + n_fifo;
        let mut fifo = std::mem::take(&mut self.fifo);
        fifo.extend_from_slice(&inputs[start * w..(start + chunk) * w]);
        let fifo_n = fifo.len() / w;
        let popped = if fifo_n <= self.geometry.fifo { 0 } else { self.geometry.update_period.max(fifo_n - self.geometry.fifo).min(fifo_n) };
        if popped > 0 {
            let fifo_probs = &probs[n_cache..n_cache + fifo_n];
            // An uncompressed cache still holds plain frames, which this step scored again; a
            // compressed one is out of order, so its stored probabilities are the only ones.
            let mut cache_probs = if self.compressed { std::mem::take(&mut self.cache_probs) } else { probs[..n_cache].to_vec() };
            let mut cache = std::mem::take(&mut self.cache);
            cache.extend_from_slice(&fifo[..popped * w]);
            cache_probs.extend_from_slice(&fifo_probs[..popped]);
            fifo.drain(..popped * w);
            if cache_probs.len() > self.geometry.speaker_cache {
                (cache, cache_probs) = self.compress(&cache, &cache_probs, silence);
                self.compressed = true;
            }
            self.cache = cache;
            self.cache_probs = cache_probs;
        }
        self.fifo = fifo;
    }

    /// Each frame's score per speaker, −∞ where it does not count.
    fn frame_scores(&self, probs: &[Probs]) -> Vec<Probs> {
        let thr = self.constants.prediction_score_threshold;
        let ln_half = 0.5f32.ln();
        let mut scores: Vec<Probs> = probs
            .iter()
            .map(|p| {
                let lp = p.map(|v| v.max(thr).ln());
                let lc = p.map(|v| (1.0 - v).max(thr).ln());
                let sum: f32 = lc.iter().sum();
                std::array::from_fn(|s| if p[s] > 0.5 { lp[s] - lc[s] + sum - ln_half } else { f32::NEG_INFINITY })
            })
            .collect();
        for s in 0..SPEAKERS {
            let positive = scores.iter().filter(|r| r[s] > 0.0).count();
            if positive >= self.min_positive {
                for (r, p) in scores.iter_mut().zip(probs) {
                    if p[s] > 0.5 && r[s] <= 0.0 {
                        r[s] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        scores
    }

    /// The cache's length's best frames of `rows` (and the silence row), grouped by speaker.
    fn compress(&self, rows: &[f32], probs: &[Probs], silence: &[f32]) -> (Vec<f32>, Vec<Probs>) {
        let w = self.width;
        let n = probs.len();
        let keep = self.geometry.speaker_cache;
        let mut scores = self.frame_scores(probs);
        // The frames past the cache's length are the ones the FIFO just gave it.
        for r in &mut scores[keep..] {
            for v in r.iter_mut() {
                *v += self.constants.latest_frames_score_boost;
            }
        }
        boost(&mut scores, self.strong, -2.0 * 0.5f32.ln());
        boost(&mut scores, self.weak, -(0.5f32.ln()));
        // Speaker-major: (speaker, frame) at speaker·(n + slots) + frame; each speaker's silence
        // slots score +∞.
        let slots = self.constants.speaker_cache_silence_frames_per_speaker;
        let per = n + slots;
        let mut flat: Vec<(f32, usize)> = Vec::with_capacity(SPEAKERS * per);
        for s in 0..SPEAKERS {
            for (f, r) in scores.iter().enumerate() {
                flat.push((r[s], s * per + f));
            }
            for f in n..per {
                flat.push((f32::INFINITY, s * per + f));
            }
        }
        let sentinel = SPEAKERS * per;
        let mut picked: Vec<usize> = top(&flat, keep).into_iter().map(|(v, i)| if v == f32::NEG_INFINITY { sentinel } else { i }).collect();
        picked.sort_unstable();
        let mut out = Vec::with_capacity(keep * w);
        let mut out_probs = Vec::with_capacity(keep);
        for i in picked {
            let f = if i == sentinel { n } else { (i % per).min(n) };
            if f == n {
                out.extend_from_slice(silence);
                out_probs.push([0.0; SPEAKERS]);
            } else {
                out.extend_from_slice(&rows[f * w..(f + 1) * w]);
                out_probs.push(probs[f]);
            }
        }
        (out, out_probs)
    }
}

/// The `k` largest of `(score, index)`, equal scores by lower index first.
fn top(items: &[(f32, usize)], k: usize) -> Vec<(f32, usize)> {
    let mut v = items.to_vec();
    let k = k.min(v.len());
    let cmp = |a: &(f32, usize), b: &(f32, usize)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    if k < v.len() {
        v.select_nth_unstable_by(k, cmp);
        v.truncate(k);
    }
    v
}

/// Adds `by` to each speaker's `k` best frames.
fn boost(scores: &mut [Probs], k: usize, by: f32) {
    for s in 0..SPEAKERS {
        let col: Vec<(f32, usize)> = scores.iter().enumerate().map(|(f, r)| (r[s], f)).collect();
        for (_, f) in top(&col, k) {
            scores[f][s] += by;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constants() -> Constants {
        Constants {
            prediction_score_threshold: 0.25,
            latest_frames_score_boost: 0.05,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_positive_scores_rate: 0.5,
            speaker_cache_silence_frames_per_speaker: 1,
        }
    }

    fn geometry() -> Geometry {
        Geometry { speaker_cache: 32, fifo: 16, chunk: 6, right_context: 2, update_period: 12 }
    }

    /// One step: `rows` index-coded rows of width 1 (a frame's index, −1 the silence row).
    fn step(c: &mut SpeakerCache, first: usize, chunk: usize, speaker: impl Fn(usize) -> Probs) {
        let mut inputs: Vec<f32> = c.rows().map(|r| r[0]).collect();
        inputs.extend((first..first + chunk + 2).map(|f| f as f32));
        let probs: Vec<Probs> = inputs.iter().map(|&f| if f < 0.0 { [0.0; SPEAKERS] } else { speaker(f as usize) }).collect();
        c.update(&inputs, &probs, &[-1.0], chunk);
    }

    #[test]
    fn the_fifo_overflows_into_the_cache_by_the_update_period() {
        let mut c = SpeakerCache::new(1, geometry(), constants());
        let one = |_| { let mut p = [0.02; SPEAKERS]; p[0] = 0.9; p };
        for i in 0..2 {
            step(&mut c, 6 * i, 6, one);
        }
        assert_eq!((c.cache_len(), c.fifo_len()), (0, 12));
        step(&mut c, 12, 6, one);
        // 18 > 16: the update period's 12 oldest move.
        assert_eq!((c.cache_len(), c.fifo_len()), (12, 6));
        assert_eq!(c.cache_rows(), (0..12).map(|f| f as f32).collect::<Vec<_>>());
        assert!(!c.compressed());
    }

    #[test]
    fn a_compressed_cache_keeps_its_length_grouped_by_speaker_with_silence_slots() {
        let mut c = SpeakerCache::new(1, geometry(), constants());
        // Two speakers taking turns every 10 frames.
        let turns = |f: usize| { let mut p = [0.01; SPEAKERS]; p[(f / 10) % 2] = 0.95; p };
        for i in 0..12 {
            step(&mut c, 6 * i, 6, turns);
        }
        assert!(c.compressed());
        assert_eq!(c.cache_len(), 32);
        let rows = c.cache_rows();
        // Speaker 0's frames, its silence slot, speaker 1's, its slot, then the other speakers'
        // slots (and the −∞ picks, silence too).
        let first_silence = rows.iter().position(|&v| v < 0.0).unwrap();
        assert!(rows[..first_silence].iter().all(|&f| (f as usize / 10).is_multiple_of(2)));
        assert!(rows[..first_silence].windows(2).all(|w| w[0] < w[1]), "arrival order inside a speaker");
        let after = &rows[first_silence + 1..];
        let second_silence = after.iter().position(|&v| v < 0.0).unwrap();
        assert!(after[..second_silence].iter().all(|&f| (f as usize / 10) % 2 == 1));
        assert!(after[second_silence..].iter().all(|&v| v < 0.0), "{rows:?}");
        // Each kept frame keeps its probabilities, the silence slots none.
        for (r, p) in rows.iter().zip(c.cache_probs()) {
            if *r < 0.0 { assert_eq!(p, &[0.0; SPEAKERS]) } else { assert_eq!(p, &turns(*r as usize)) }
        }
    }

    #[test]
    fn equal_scores_go_by_the_lower_index() {
        let items = [(1.0, 3), (2.0, 0), (1.0, 1), (1.0, 2)];
        let mut t = top(&items, 2);
        t.sort_by_key(|x| x.1);
        assert_eq!(t, vec![(2.0, 0), (1.0, 1)]);
    }
}
