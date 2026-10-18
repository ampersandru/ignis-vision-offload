//! Speculation past 8 verify rows against speculation off, row by row
//! (GitHub #329 item 5, ADR 0056). Included by path from the Flash-Next GPU
//! test that holds a wide budget to the plain decode route, and from the CPU
//! test of these rules.
//!
//! Past 8 rows a verify round's linears and mixes take their tensor-core
//! routes, so its logits are the plain round's up to summation order, not bit
//! for bit: a free-running greedy text can flip at a near-tie and never come
//! back. So the comparison is teacher-forced -- the same tokens through both
//! routes, logits row by row -- and its tolerance is a count of BF16 ulps at
//! the reference row's largest magnitude ([`TOLERANCE_ULPS`]). What stays
//! exact at any budget is the commit: a round commits its own argmax
//! ([`commit_follows_argmax`]).

/// The teacher-forced tolerance, in BF16 ulps at the reference row's largest
/// magnitude: no logit of a verify round's row may sit further than this from
/// the plain round's, and their picks may differ only where the plain row
/// rates the two within it. Measured on the RTX 5090 (2026-10-18, ADR 0056):
/// the default's own 8-row rounds reach 40.2 ulps from spec-off, 16-32 rows
/// 58.3; the next power of two above both. Rows on another context (a lane
/// one moved pick off the text) reach 33-202, 89% of them past it.
pub const TOLERANCE_ULPS: f32 = 64.0;

/// The teacher-forced mean log-probability's tolerance, in nats a token: the
/// wide route's mean over a cell's compared rows against the plain route's.
/// Measured: 0.0077 at 8 rows, 0.0117 at 16-32 (the text is spec-off's own
/// greedy text, so the wide route reads lower); 1.7x the widest.
pub const MEAN_LOG_PROB_NATS: f64 = 0.02;

/// The head's first-position acceptance under which a budget's cell is
/// refused: a wide round's rows or the head's entries gone wrong show as
/// drafts no longer accepted. Measured 0.74-0.94 on these prompts.
pub const HEAD_ACCEPTANCE_FLOOR: f64 = 0.5;

pub fn bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// One BF16 ulp at `x`'s magnitude (`flash_next_forward_gpu.rs`'s rule).
pub fn bf16_ulp(x: f32) -> f32 {
    let exponent = ((x.abs().to_bits() >> 23) & 0xff) as i32 - 127;
    2f32.powi(exponent - 7)
}

/// A row's pick as the engine draws it -- the lowest id among equal maxima
/// (the vendored argmax's rule) -- and its value.
pub fn argmax(row: &[u16]) -> (u32, f32) {
    let (mut best, mut first) = (0u32, f32::NEG_INFINITY);
    for (id, &bits) in row.iter().enumerate() {
        let v = bf16(bits);
        if v > first {
            first = v;
            best = id as u32;
        }
    }
    (best, first)
}

/// `target`'s log-probability under `row` (a log-softmax, accumulated in
/// f64).
pub fn log_prob(row: &[u16], target: u32) -> f64 {
    let (_, top) = argmax(row);
    let sum: f64 = row.iter().map(|&b| (f64::from(bf16(b)) - f64::from(top)).exp()).sum();
    f64::from(bf16(row[target as usize])) - f64::from(top) - sum.ln()
}

/// One teacher-forced row pair: the plain route's (`off`) and the wide
/// verify round's (`on`) after the same tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowDrift {
    /// The largest |on - off| over the vocabulary, in logits.
    pub max_abs: f32,
    /// `max_abs` in BF16 ulps at `off`'s largest magnitude.
    pub ulps: f32,
    /// The id that moved most (the lowest among equals).
    pub worst: u32,
    pub pick_off: u32,
    pub pick_on: u32,
    /// `off[pick_off] - off[pick_on]` in the same ulps: 0 when the picks agree.
    pub gap_ulps: f32,
}

pub fn row_drift(off: &[u16], on: &[u16]) -> RowDrift {
    assert_eq!(off.len(), on.len(), "rows of different vocabularies");
    let scale = off.iter().map(|&b| bf16(b).abs()).fold(0f32, f32::max);
    let ulp = bf16_ulp(scale);
    let (mut worst, mut max_abs) = (0u32, 0f32);
    for (id, (&a, &b)) in off.iter().zip(on).enumerate() {
        let moved = (bf16(a) - bf16(b)).abs();
        if moved > max_abs {
            (worst, max_abs) = (id as u32, moved);
        }
    }
    let (pick_off, top) = argmax(off);
    let (pick_on, _) = argmax(on);
    RowDrift { max_abs, ulps: max_abs / ulp, worst, pick_off, pick_on, gap_ulps: (top - bf16(off[pick_on as usize])) / ulp }
}

/// Within `tolerance_ulps`: every logit, and a different pick only where the
/// plain row rates the two picks within it. A line naming the excess
/// otherwise.
pub fn within(drift: &RowDrift, tolerance_ulps: f32) -> Result<(), String> {
    if drift.ulps > tolerance_ulps {
        return Err(format!("a logit moved {} ({} ulps, past {tolerance_ulps})", drift.max_abs, drift.ulps));
    }
    if drift.pick_off != drift.pick_on && drift.gap_ulps > tolerance_ulps {
        return Err(format!(
            "the pick moved from {} to {}, {} ulps apart in the plain row (past {tolerance_ulps})",
            drift.pick_off, drift.pick_on, drift.gap_ulps
        ));
    }
    Ok(())
}

/// One lane's share of a verify round: its anchor, the drafts it verified
/// (its extent of them), the run it committed, the token that stood pending
/// after it (`None` once the lane's budget is spent), and the picks of its
/// rows, one a column, `picks[j]` after the column holding the anchor (j = 0)
/// or draft j.
pub struct LaneCommit<'a> {
    pub anchor: u32,
    pub drafts: &'a [u32],
    pub committed: &'a [u32],
    pub next: Option<u32>,
    pub picks: &'a [u32],
}

/// The greedy accept, exact at any budget whatever the routes' numerics: the
/// drafts accepted are the longest prefix each of which is the pick after the
/// column before it, the run is the anchor and those drafts, and what stands
/// pending after it is the pick after the run's last column. A line naming
/// the first disagreement otherwise.
pub fn commit_follows_argmax(lane: &LaneCommit<'_>) -> Result<(), String> {
    if lane.picks.len() <= lane.drafts.len() {
        return Err(format!("{} rows for {} drafts", lane.picks.len(), lane.drafts.len()));
    }
    let accepted = lane.drafts.iter().zip(lane.picks).take_while(|(d, p)| d == p).count();
    let mut run = vec![lane.anchor];
    run.extend_from_slice(&lane.drafts[..accepted]);
    if lane.committed != run.as_slice() {
        return Err(format!(
            "committed {:?} where the rows' picks {:?} accept {run:?} of anchor {} and drafts {:?}",
            lane.committed, lane.picks, lane.anchor, lane.drafts
        ));
    }
    match lane.next {
        Some(next) if next != lane.picks[accepted] => Err(format!(
            "{next} stands pending after the run where its last column's pick is {}",
            lane.picks[accepted]
        )),
        _ => Ok(()),
    }
}
