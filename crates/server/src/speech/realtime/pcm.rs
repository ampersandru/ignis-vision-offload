//! The audio a Realtime session appends (spec speech/03, AC 3): base64
//! 16-bit little-endian mono PCM at the session's rate, turned into the
//! 16 kHz f32 samples the executor takes. 24 kHz, OpenAI's rate, goes
//! through the transcription route's streaming resampler
//! ([`super::super::audio::Resampler`]); 16 kHz, the model's own, is passed
//! through.

use super::super::audio::{Resampler, TARGET_RATE};
use crate::media::base64_bytes;

/// One session's audio in: its rate, and what an item carries over from one
/// append to the next.
pub(crate) struct Pcm {
    rate: u32,
    /// `None` at 16 kHz.
    resampler: Option<Resampler>,
    /// The first byte of a sample whose second is in the next append.
    odd: Option<u8>,
    /// The item has given the resampler a sample: its tail is owed.
    fed: bool,
}

impl Pcm {
    pub fn new(rate: u32) -> Self {
        Self { rate, resampler: Self::resampler(rate), odd: None, fed: false }
    }

    fn resampler(rate: u32) -> Option<Resampler> {
        (rate != TARGET_RATE).then(|| Resampler::new(rate, TARGET_RATE))
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// One append's `audio`: its 16 kHz samples onto `out`, and how many
    /// samples it carried at the session's rate; `Err` when it is not
    /// base64.
    pub fn append(&mut self, audio: &str, out: &mut Vec<f32>) -> Result<usize, ()> {
        let bytes = base64_bytes(audio).ok_or(())?;
        let mut samples = Vec::with_capacity(bytes.len() / 2 + 1);
        let mut rest = bytes.as_slice();
        if let (Some(first), Some((&second, tail))) = (self.odd, rest.split_first()) {
            samples.push(sample(first, second));
            self.odd = None;
            rest = tail;
        }
        let pairs = rest.chunks_exact(2);
        if let [last] = pairs.remainder() {
            self.odd = Some(*last);
        }
        samples.extend(pairs.map(|pair| sample(pair[0], pair[1])));
        self.fed |= !samples.is_empty();
        match self.resampler.as_mut() {
            Some(resampler) => resampler.process(&samples, out),
            None => out.extend_from_slice(&samples),
        }
        Ok(samples.len())
    }

    /// An item ends: the resampler's tail onto `out`, and a fresh start for
    /// the next item.
    pub fn finish(&mut self, out: &mut Vec<f32>) {
        self.flush(out);
        self.reset();
    }

    /// The resampler's tail, when the item gave it anything.
    fn flush(&mut self, out: &mut Vec<f32>) {
        if let (Some(resampler), true) = (self.resampler.as_mut(), self.fed) {
            resampler.finish(out);
        }
    }

    /// An item is discarded: nothing of it carries over.
    pub fn reset(&mut self) {
        self.resampler = Self::resampler(self.rate);
        self.odd = None;
        self.fed = false;
    }

    /// The session's rate changes: what the current item holds at the old
    /// rate goes out onto `out` first.
    pub fn set_rate(&mut self, rate: u32, out: &mut Vec<f32>) {
        if rate != self.rate {
            self.flush(out);
            *self = Self::new(rate);
        }
    }
}

fn sample(low: u8, high: u8) -> f32 {
    f32::from(i16::from_le_bytes([low, high])) / 32768.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard base64, padded, of `bytes`.
    fn base64(bytes: &[u8]) -> String {
        const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
            for i in 0..4 {
                out.push(if i <= chunk.len() { ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
            }
        }
        out
    }

    fn pcm16(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
    }

    /// `seconds` of a sine at `frequency`, at `rate`, as PCM16.
    fn tone(frequency: f64, rate: u32, seconds: f64) -> Vec<i16> {
        let n = (f64::from(rate) * seconds) as usize;
        (0..n).map(|i| ((2.0 * std::f64::consts::PI * frequency * i as f64 / f64::from(rate)).sin() * 16_000.0) as i16).collect()
    }

    fn crossings(samples: &[f32]) -> usize {
        samples.windows(2).filter(|pair| (pair[0] < 0.0) != (pair[1] < 0.0)).count()
    }

    /// AC 3: 16 kHz, the declared extension, is passed through sample for
    /// sample.
    #[test]
    fn sixteen_khz_is_passed_through() {
        let mut pcm = Pcm::new(16_000);
        let mut out = Vec::new();
        assert_eq!(pcm.append(&base64(&pcm16(&[0, 16384, -32768, 32767])), &mut out), Ok(4));
        assert_eq!(out, [0.0, 0.5, -1.0, 32767.0 / 32768.0]);
        pcm.finish(&mut out);
        assert_eq!(out.len(), 4, "nothing to flush at the model's rate");
    }

    /// AC 3: a second of a 440 Hz tone at 24 kHz comes out as a second at
    /// 16 kHz, still 440 Hz.
    #[test]
    fn twenty_four_khz_is_resampled_to_sixteen_keeping_pitch_and_length() {
        let mut pcm = Pcm::new(24_000);
        let mut out = Vec::new();
        let bytes = pcm16(&tone(440.0, 24_000, 1.0));
        // In uneven pieces, as a client's appends come.
        let mut carried = 0;
        for piece in bytes.chunks(4_801) {
            carried += pcm.append(&base64(piece), &mut out).unwrap();
        }
        pcm.finish(&mut out);
        assert_eq!(carried, 24_000);
        assert!((out.len() as i64 - 16_000).abs() <= 64, "{} samples", out.len());
        // 440 Hz crosses zero 880 times a second.
        let crossed = crossings(&out[100..out.len() - 100]) as f64 / ((out.len() - 200) as f64 / 16_000.0);
        assert!((crossed - 880.0).abs() < 10.0, "{crossed} crossings a second");
    }

    /// A sample split across two appends decodes as it would whole.
    #[test]
    fn a_sample_split_across_appends_decodes_whole() {
        let bytes = pcm16(&[1000, -2000, 3000]);
        let (mut whole, mut split) = (Vec::new(), Vec::new());
        Pcm::new(16_000).append(&base64(&bytes), &mut whole).unwrap();
        let mut pcm = Pcm::new(16_000);
        assert_eq!(pcm.append(&base64(&bytes[..3]), &mut split), Ok(1));
        assert_eq!(pcm.append(&base64(&bytes[3..]), &mut split), Ok(2));
        assert_eq!(split, whole);
    }

    /// An item that gave the resampler nothing has no tail: a session that
    /// opens at 24 kHz and moves to 16 kHz before its first append starts
    /// from nothing.
    #[test]
    fn nothing_in_is_nothing_out() {
        let mut out = Vec::new();
        let mut pcm = Pcm::new(24_000);
        pcm.set_rate(16_000, &mut out);
        Pcm::new(24_000).finish(&mut out);
        assert!(out.is_empty(), "{} samples", out.len());
    }

    #[test]
    fn audio_that_is_not_base64_is_refused() {
        assert_eq!(Pcm::new(24_000).append("not base64!", &mut Vec::new()), Err(()));
    }
}
