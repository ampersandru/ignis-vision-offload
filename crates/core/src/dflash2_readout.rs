//! The DFlash2 proposal readout (GitHub #267, `kernel/include/ignis_dflash2_readout.h`):
//! what the drafter would propose if a verify round started at a sequence's
//! current state, read without running the round.
//!
//! A measurement seam, not a serving one -- the `out_span_logits` precedent
//! ([`crate::step::prefill_program_span_logits`]). The replay of recorded agent
//! turns (`crates/core/examples/dflash2_replay.rs`) walks a turn's recorded
//! text: it reads the drafter's proposals here, counts how many of them the
//! recorded text accepts, and advances the sequence by prefilling what a
//! round would have committed. A round cannot do that walk, because it
//! commits what the target samples, not what the recording says.
//!
//! Beside the drafts it returns what the selector read to choose them -- each
//! draft column's 16 candidates, their proposal-head logits and the lattice's
//! transition scores -- so a different selection rule can be walked over the
//! same lattice on the host ([`Dflash2Proposal::walk`] is today's rule).

/// The selector's candidates per draft column (`kDflash2SelectorTopK`).
pub const TOP_K: usize = 16;

/// One readout: the drafter's proposals at a sequence's frontier for one
/// anchor, and the lattice the selector walked to choose them. `k` is the
/// load's draft window; every per-column table has `k` rows.
#[derive(Debug, Clone, PartialEq)]
pub struct Dflash2Proposal {
    /// The selector walk's drafts, `[k]`.
    pub drafts: Vec<i32>,
    /// Each draft column's top-16 token ids, `[k][16]`, in the proposal
    /// head's descending order.
    pub candidate_ids: Vec<i32>,
    /// Their proposal-head logits (the selector's unary term), `[k][16]`.
    pub candidate_logits: Vec<f32>,
    /// The transition scores, `[k][16 pred][16 succ]`: successor candidate
    /// `succ` of column `s` after predecessor candidate `pred` of column
    /// `s - 1` (column 0 follows the anchor, so its `pred` rows are equal).
    pub scores: Vec<f32>,
    /// The proposal head's logits over the vocabulary, BF16 `[k][vocab]`,
    /// when asked for.
    pub drafter_logits: Option<Vec<u16>>,
}

impl Dflash2Proposal {
    /// The draft window `k` the readout was taken at.
    pub fn window(&self) -> usize {
        self.drafts.len()
    }

    /// Column `column`'s 16 candidate ids.
    pub fn candidates(&self, column: usize) -> &[i32] {
        &self.candidate_ids[column * TOP_K..(column + 1) * TOP_K]
    }

    /// The transition score of successor `succ` at `column` after predecessor
    /// `pred`.
    pub fn score(&self, column: usize, pred: usize, succ: usize) -> f32 {
        self.scores[(column * TOP_K + pred) * TOP_K + succ]
    }

    /// The selector's walk over these scores, as the vendored
    /// `dflash2_selector_walk` states it: from predecessor 0, each column
    /// takes the successor with the highest score after the previous
    /// column's choice, ties toward the smaller index. Returns the chosen
    /// candidate index per column; the drafts are the candidates at them.
    pub fn walk(&self) -> Vec<usize> {
        let mut pred = 0usize;
        (0..self.window())
            .map(|column| {
                let mut best = 0usize;
                let mut best_score = f32::NEG_INFINITY;
                for succ in 0..TOP_K {
                    let score = self.score(column, pred, succ);
                    if score > best_score {
                        best_score = score;
                        best = succ;
                    }
                }
                pred = best;
                best
            })
            .collect()
    }
}

#[cfg(feature = "cuda")]
mod ffi {
    use std::os::raw::c_char;

    use crate::model_load::ffi::IgnisModel;
    use crate::seq::ffi::{IgnisSeq, IgnisSeqPool};

    /// 1:1 with `struct ignis_dflash2_readout`.
    #[repr(C)]
    pub struct IgnisDflash2Readout {
        pub size: u32,
        pub anchor: i32,
        pub extent: u32,
        pub out_drafts: *mut i32,
        pub out_candidate_ids: *mut i32,
        pub out_candidate_logits: *mut f32,
        pub out_scores: *mut f32,
        pub out_drafter_logits: *mut u16,
    }

    unsafe extern "C" {
        pub fn ignis_dflash2_readout(
            model: *mut IgnisModel,
            pool: *mut IgnisSeqPool,
            seq: *const IgnisSeq,
            readout: *const IgnisDflash2Readout,
        ) -> i32;
        pub fn ignis_verify_readout_logits(model: *mut IgnisModel, lanes: u32, out_logits: *mut u16) -> i32;
        pub fn ignis_dflash2_readout_last_error() -> *const c_char;
    }
}

#[cfg(feature = "cuda")]
fn last_error() -> String {
    let message = unsafe { std::ffi::CStr::from_ptr(ffi::ignis_dflash2_readout_last_error()) };
    message.to_string_lossy().into_owned()
}

/// The drafter's proposals for `sequence` at its frontier, as a verify round
/// with `anchor` pending and `extent` draft columns would compute them.
/// `None` takes the sequence's own pending token: the proposals its next
/// round verifies.
///
/// `window` is the load's draft window ([`crate::Speculation::draft_tokens`]);
/// `extent` is at most that. With `with_logits`, the proposal head's whole
/// logits come back too (4 MB a readout at window 7). Nothing the sequence or
/// any other one reads is written; the leaf refuses a sequence whose window
/// does not stand at its frontier.
#[cfg(feature = "cuda")]
pub fn propose(
    model: &crate::model_load::Model,
    pool: &crate::seq::SeqPool,
    sequence: &crate::seq::Seq<'_>,
    anchor: Option<i32>,
    extent: u32,
    window: u32,
    with_logits: bool,
) -> Result<Dflash2Proposal, String> {
    let k = window as usize;
    let vocab = crate::compute::ModelConfig::qwen38_27b().vocab as usize;
    let mut drafts = vec![-1i32; k];
    let mut candidate_ids = vec![-1i32; k * TOP_K];
    let mut candidate_logits = vec![0f32; k * TOP_K];
    let mut scores = vec![0f32; k * TOP_K * TOP_K];
    let mut drafter_logits = with_logits.then(|| vec![0u16; k * vocab]);
    let readout = ffi::IgnisDflash2Readout {
        size: std::mem::size_of::<ffi::IgnisDflash2Readout>() as u32,
        anchor: anchor.unwrap_or(-1),
        extent,
        out_drafts: drafts.as_mut_ptr(),
        out_candidate_ids: candidate_ids.as_mut_ptr(),
        out_candidate_logits: candidate_logits.as_mut_ptr(),
        out_scores: scores.as_mut_ptr(),
        out_drafter_logits: drafter_logits.as_mut().map_or(std::ptr::null_mut(), |v| v.as_mut_ptr()),
    };
    // The leaf sizes every output by the load's window: a caller's window
    // that is not the load's would over- or under-run these buffers.
    if extent > window {
        return Err(format!("dflash2 readout: extent {extent} is past the window {window}"));
    }
    let rc = unsafe { ffi::ignis_dflash2_readout(model.handle(), pool.handle(), sequence.handle(), &readout) };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(Dflash2Proposal {
        drafts,
        candidate_ids,
        candidate_logits,
        scores,
        drafter_logits,
    })
}

/// The most recent verify round's target logits for its first `lanes` lanes,
/// BF16 `[lanes][window + 1][vocab]`: row `j` of a lane is the target's
/// distribution after its column `j` (its anchor, then its drafts), as the
/// round's accept read it. Valid until the model's next verify round.
#[cfg(feature = "cuda")]
pub fn verify_logits(model: &crate::model_load::Model, lanes: u32, window: u32) -> Result<Vec<u16>, String> {
    let vocab = crate::compute::ModelConfig::qwen38_27b().vocab as usize;
    let mut logits = vec![0u16; lanes as usize * (window as usize + 1) * vocab];
    let rc = unsafe { ffi::ignis_verify_readout_logits(model.handle(), lanes, logits.as_mut_ptr()) };
    if rc != 0 {
        return Err(last_error());
    }
    Ok(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-column lattice worked by hand: column 0's best successor is
    /// candidate 3; after it, column 1's best is candidate 5, though after
    /// predecessor 0 it would have been candidate 1.
    fn two_column_lattice() -> Dflash2Proposal {
        let mut scores = vec![0f32; 2 * TOP_K * TOP_K];
        for pred in 0..TOP_K {
            scores[pred * TOP_K + 3] = 2.0;
        }
        scores[(TOP_K + 0) * TOP_K + 1] = 9.0;
        scores[(TOP_K + 3) * TOP_K + 5] = 4.0;
        scores[(TOP_K + 3) * TOP_K + 1] = 1.0;
        Dflash2Proposal {
            drafts: vec![103, 205],
            candidate_ids: (0..2 * TOP_K as i32).map(|i| (i / TOP_K as i32 + 1) * 100 + i % TOP_K as i32).collect(),
            candidate_logits: vec![0.0; 2 * TOP_K],
            scores,
            drafter_logits: None,
        }
    }

    #[test]
    fn the_walk_follows_the_previous_columns_choice() {
        let proposal = two_column_lattice();
        let path = proposal.walk();
        assert_eq!(path, vec![3, 5]);
        let drafts: Vec<i32> = path.iter().enumerate().map(|(s, &i)| proposal.candidates(s)[i]).collect();
        assert_eq!(drafts, proposal.drafts);
    }

    #[test]
    fn the_walk_breaks_ties_toward_the_smaller_index() {
        let mut proposal = two_column_lattice();
        proposal.scores.iter_mut().for_each(|s| *s = 1.5);
        assert_eq!(proposal.walk(), vec![0, 0]);
    }

    /// uint32 size, int32 anchor, uint32 extent, padding, then five
    /// pointers; the leaf pins the same numbers
    /// (kernel/include/ignis_dflash2_readout.h) and refuses any other `size`.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_readout_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<ffi::IgnisDflash2Readout>(), 56);
        assert_eq!(std::mem::offset_of!(ffi::IgnisDflash2Readout, out_drafts), 16);
        assert_eq!(std::mem::offset_of!(ffi::IgnisDflash2Readout, out_drafter_logits), 48);
    }
}
