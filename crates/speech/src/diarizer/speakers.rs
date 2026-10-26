//! Who speaks each 80 ms frame, by spec 08's rules (Q107, Q108): the frames' speakers, an item's
//! speaker, and where an item ends at a change of speaker.
//!
//! - A frame is speech when one of its channels' activity reaches 0.5 (the checkpoint's own line
//!   between speech and not, as its speaker cache draws it). Its most active channel holds it.
//! - **The ghost rule:** a channel is established once it holds `min_speaker_seconds` of speech,
//!   counted on the frames it holds. The session's first channel is established at once, and with
//!   `min_speaker_seconds` 0 every channel is, at its first frame. Until a channel is established,
//!   its frames count for the established channel with the next highest activity.
//! - A frame's speaker is the established channel that holds it, or the one it counts for.
//! - A channel established only now counts from the start of the run that established it: its
//!   frames since the last speech frame another channel held become its own.
//! - An item's speaker is the established channel that holds most of its speech frames.
//! - A change: inside an item, the first speech frame whose speaker is not the item's first
//!   speech frame's. The item ends there.
//!
//! Provenance: the module's (`diarizer/mod.rs`, ADR 0051), for the checkpoint's line between speech
//! and not; the rules are spec 08's.

use super::{Probs, SPEAKERS};

/// A frame is speech once a channel's activity reaches this.
const SPEECH: f32 = 0.5;

/// Frames per second at 80 ms.
const FRAMES_PER_SECOND: f64 = 12.5;

/// A speaker: a channel of the model, 0-7, numbered in the order they first spoke.
pub type Speaker = u8;

#[derive(Clone)]
pub struct Speakers {
    /// The frames a channel must hold to be established (the ghost rule), at least 1.
    min_frames: usize,
    held: [usize; SPEAKERS],
    established: [bool; SPEAKERS],
    /// Per frame from `base`: the channel holding it, `None` when it is not speech.
    holder: Vec<Option<Speaker>>,
    /// Per frame from `base`: its speaker after the ghost rule.
    speaker: Vec<Option<Speaker>>,
    base: usize,
}

impl Speakers {
    /// `min_speaker_seconds`: the ghost rule's length, 0 to turn it off.
    pub fn new(min_speaker_seconds: f64) -> Speakers {
        let min_frames = (min_speaker_seconds.max(0.0) * FRAMES_PER_SECOND).ceil() as usize;
        Speakers { min_frames: min_frames.max(1), held: [0; SPEAKERS], established: [false; SPEAKERS], holder: Vec::new(), speaker: Vec::new(), base: 0 }
    }

    /// The frames known, from the stream's first.
    pub fn len(&self) -> usize {
        self.base + self.speaker.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn established(&self, c: Speaker) -> bool {
        self.established[c as usize]
    }

    /// The next frames, in order.
    pub fn push(&mut self, frames: &[Probs]) {
        for p in frames {
            self.push_one(p);
        }
    }

    fn push_one(&mut self, p: &Probs) {
        let top = (0..SPEAKERS).max_by(|&a, &b| p[a].total_cmp(&p[b]).then(b.cmp(&a))).unwrap();
        if p[top] < SPEECH {
            self.holder.push(None);
            self.speaker.push(None);
            return;
        }
        let c = top as Speaker;
        self.held[top] += 1;
        let first = !self.established.iter().any(|&e| e);
        if !self.established[top] && (first || self.held[top] >= self.min_frames) {
            self.established[top] = true;
            // The run that established it is its own: back to the last speech frame another
            // channel held.
            for i in (0..self.holder.len()).rev() {
                match self.holder[i] {
                    Some(h) if h != c => break,
                    Some(_) => self.speaker[i] = Some(c),
                    None => {}
                }
            }
        }
        let speaker = if self.established[top] {
            c
        } else {
            (0..SPEAKERS).filter(|&s| self.established[s]).max_by(|&a, &b| p[a].total_cmp(&p[b]).then(b.cmp(&a))).unwrap() as Speaker
        };
        self.holder.push(Some(c));
        self.speaker.push(Some(speaker));
    }

    /// Frame `f`'s speaker; `None` when it is not speech, or not known or kept.
    pub fn speaker(&self, f: usize) -> Option<Speaker> {
        f.checked_sub(self.base).and_then(|i| self.speaker.get(i).copied().flatten())
    }

    /// The first change in frames `from..to` (to the frames known): the first speech frame whose
    /// speaker is not the first speech frame's, and that speaker.
    pub fn change(&self, from: usize, to: usize) -> Option<(usize, Speaker)> {
        let mut first = None;
        for f in from.max(self.base)..to.min(self.len()) {
            if let Some(s) = self.speaker(f) {
                match first {
                    None => first = Some(s),
                    Some(a) if a != s => return Some((f, s)),
                    Some(_) => {}
                }
            }
        }
        None
    }

    /// The speaker of frames `from..to`: the one holding most of their speech frames, the earliest
    /// to speak among equals; `None` when none is speech.
    pub fn speaker_of(&self, from: usize, to: usize) -> Option<Speaker> {
        let mut count = [0usize; SPEAKERS];
        let mut order = Vec::new();
        for f in from.max(self.base)..to.min(self.len()) {
            if let Some(s) = self.speaker(f) {
                if count[s as usize] == 0 {
                    order.push(s);
                }
                count[s as usize] += 1;
            }
        }
        order.into_iter().max_by(|&a, &b| count[a as usize].cmp(&count[b as usize]).then(b.cmp(&a)))
    }

    /// Drops what came before frame `f`: nothing before it is asked for again. The run a later
    /// establishment reaches back over is kept: the frames since the last speech frame before
    /// `f`'s run are.
    pub fn forget_before(&mut self, f: usize) {
        let mut cut = f.saturating_sub(self.base).min(self.holder.len());
        // Keep the current run of whichever channel holds the last speech frame before `f`.
        if let Some(last) = self.holder[..cut].iter().rposition(|h| h.is_some()) {
            let c = self.holder[last];
            let mut start = last;
            while start > 0 && (self.holder[start - 1].is_none() || self.holder[start - 1] == c) {
                start -= 1;
            }
            cut = cut.min(start);
        }
        self.holder.drain(..cut);
        self.speaker.drain(..cut);
        self.base += cut;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frame where channel `c` speaks at `a` and channel `d` at `b`, the rest silent.
    fn frame(c: usize, a: f32, d: usize, b: f32) -> Probs {
        let mut p = [0.01; SPEAKERS];
        p[c] = a;
        p[d] = p[d].max(b);
        p
    }

    fn talk(c: usize, n: usize) -> Vec<Probs> {
        vec![frame(c, 0.9, (c + 1) % SPEAKERS, 0.05); n]
    }

    fn silence(n: usize) -> Vec<Probs> {
        vec![[0.01; SPEAKERS]; n]
    }

    #[test]
    fn the_first_channel_is_established_at_once_and_silence_is_no_speaker() {
        let mut s = Speakers::new(2.0);
        s.push(&talk(3, 2));
        s.push(&silence(2));
        assert!(s.established(3));
        assert_eq!((s.speaker(0), s.speaker(1), s.speaker(2)), (Some(3), Some(3), None));
        assert_eq!(s.len(), 4);
    }

    #[test]
    fn a_ghost_counts_for_the_next_most_active_established_channel() {
        let mut s = Speakers::new(2.0); // 25 frames
        s.push(&talk(0, 30));
        // Channel 1, 1 s: its frames count for channel 0, the next most active established one.
        s.push(&vec![frame(1, 0.8, 0, 0.3); 12]);
        assert!(!s.established(1));
        assert_eq!(s.speaker(35), Some(0));
        assert_eq!(s.change(0, s.len()), None);
        assert_eq!(s.speaker_of(30, 42), Some(0));
    }

    #[test]
    fn an_established_channel_takes_back_the_run_that_established_it() {
        let mut s = Speakers::new(2.0); // 25 frames
        s.push(&talk(0, 30));
        s.push(&silence(3));
        s.push(&vec![frame(1, 0.8, 0, 0.3); 24]);
        assert!(!s.established(1));
        assert_eq!(s.speaker(40), Some(0));
        s.push(&[frame(1, 0.8, 0, 0.3); 1]);
        assert!(s.established(1));
        // Every frame of its run, from its first (33), is now its own; the silence before it is
        // no one's.
        assert_eq!(s.speaker(32), None);
        assert!((33..58).all(|f| s.speaker(f) == Some(1)));
        // So the change is at the run's first frame.
        assert_eq!(s.change(0, s.len()), Some((33, 1)));
        assert_eq!(s.speaker_of(0, 33), Some(0));
        assert_eq!(s.speaker_of(33, 58), Some(1));
    }

    #[test]
    fn frames_of_another_channel_before_the_run_stay_where_they_were_counted() {
        let mut s = Speakers::new(1.0); // 13 frames
        s.push(&talk(0, 20));
        s.push(&vec![frame(1, 0.8, 0, 0.3); 8]); // 20..28, counted for 0
        s.push(&talk(0, 2)); // 28..30
        s.push(&[frame(1, 0.8, 0, 0.3); 5]); // 30..35: the 13th of channel 1's frames
        assert!(s.established(1));
        // Only the run since channel 0's last frame (30..) became channel 1's.
        assert!((20..28).all(|f| s.speaker(f) == Some(0)));
        assert!((30..35).all(|f| s.speaker(f) == Some(1)));
        assert_eq!(s.change(0, s.len()), Some((30, 1)));
    }

    #[test]
    fn with_the_rule_off_every_channel_is_established_at_its_first_frame() {
        let mut s = Speakers::new(0.0);
        s.push(&talk(0, 10));
        s.push(&talk(2, 1));
        assert!(s.established(2));
        assert_eq!(s.change(0, s.len()), Some((10, 2)));
        assert_eq!(s.speaker_of(0, 11), Some(0));
    }

    #[test]
    fn a_change_is_looked_for_inside_an_item_only() {
        let mut s = Speakers::new(0.0);
        s.push(&talk(0, 10));
        s.push(&silence(5));
        s.push(&talk(1, 10));
        // An item from frame 15 on starts with channel 1: no change in it.
        assert_eq!(s.change(15, s.len()), None);
        assert_eq!(s.change(0, 12), None);
        assert_eq!(s.change(0, s.len()), Some((15, 1)));
        assert_eq!(s.speaker_of(15, 25), Some(1));
        assert_eq!(s.speaker_of(10, 15), None);
    }

    #[test]
    fn forgetting_keeps_the_run_a_later_establishment_reaches_back_over() {
        let mut s = Speakers::new(2.0);
        s.push(&talk(0, 30));
        s.push(&vec![frame(1, 0.8, 0, 0.3); 20]);
        s.forget_before(45);
        // Channel 1's run started at 30: it is kept.
        assert_eq!(s.speaker(30), Some(0));
        assert_eq!(s.speaker(29), None);
        s.push(&[frame(1, 0.8, 0, 0.3); 5]);
        assert!((30..55).all(|f| s.speaker(f) == Some(1)));
    }
}
