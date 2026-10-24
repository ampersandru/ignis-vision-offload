//! `struct ignis_decode_options` (`kernel/include/ignis_step.h`) as a plain
//! `repr(C)` mirror: no kernel is needed to lay it out, so its layout pin
//! runs in the CPU suite (GitHub #340) while [`crate::step`], which fills it,
//! is `cuda`-only.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

/// 1:1 with `struct ignis_decode_options` (ADR 0016; P5-04, GitHub
/// #153). `size` must be `sizeof(struct ignis_decode_options)`; a null
/// options pointer is today's round.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IgnisDecodeOptions {
    pub size: u32,
    pub speculative_window: u32,
    pub drafts: *const i32,
    pub draft_counts: *const u32,
    pub out_committed_counts: *mut i32,
    /// P5-05 (GitHub #155): each lane's extent this round, or null.
    pub out_extents: *mut u32,
    /// P6-06 (GitHub #242): each lane's committed token's probability
    /// within its own permitted set, or 0 for a lane that declared
    /// none. Null asks for nothing. One float per lane rather than a
    /// logits row — the per-digit confidence a number's uncertainty is
    /// summed from has to come from somewhere, and this is the cheapest
    /// thing that is not the host reading logits (ADR 0034).
    pub out_permitted_probs: *mut f32,
    /// GitHub #302: each lane's n-gram table rows on a Flash-Next load,
    /// host `[batch][16][90]` (a verify round: `[batch][window + 1]
    /// [16][90]`); null on the 27B.
    pub ngram_rows: *const u8,
    /// GitHub #307: a Flash-Next MTP load's drafts for each lane's next
    /// round, `[batch][window]`; null asks for nothing.
    pub out_drafts: *mut i32,
    /// GitHub #315 (ADR 0048): per lane, 1 when the round redirected the
    /// lane's successor to its `reasoning_close_id`, else 0; null asks
    /// for nothing.
    pub out_reasoning_redirected: *mut i32,
    /// GitHub #329 item 5 (ADR 0056): a Flash-Next round's BF16 logits,
    /// every row its head produced (`[batch][vocab]`, a verify round
    /// `[batch][window + 1][vocab]`); null asks for nothing.
    pub out_logits: *mut u16,
    /// GitHub #267, test-only: a verify round's teacher forcing, host
    /// `[batch][window + 1]`, or null -- see
    /// `step::decode_program_verify_forced`.
    pub forced_tokens: *const i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// GitHub #267: `ignis_decode_options` with the teacher forcing appended
    /// last, so every field the serving path sets keeps its offset. uint32
    /// size and window, then ten pointers; the leaf pins the same numbers
    /// (kernel/include/ignis_step.h) and refuses any other `size`.
    #[test]
    fn the_decode_options_mirror_is_the_leaf_structs_layout() {
        assert_eq!(std::mem::size_of::<IgnisDecodeOptions>(), 88);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, out_logits), 72);
        assert_eq!(std::mem::offset_of!(IgnisDecodeOptions, forced_tokens), 80);
    }
}
