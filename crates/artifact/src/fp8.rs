//! The FP8 row-scale encoding (`FP8_E4M3FN_ROW_BF16S` / `row-scale-v1`,
//! docs/specs/flash-next/layout.md §6.1) written from a BF16 weight: the 27B's
//! BF16 exception tensors converted at load (GitHub #341).
//!
//! The same contract as the leaf's `ignis_fp8_quantize_rows`
//! (`kernel/include/ignis_fp8_linear.h`), so a load on the CPU device and the
//! f64 layer oracle hold the bytes the GPU writes: per row, `scale` = the
//! least BF16 >= fl32(amax / 448) (1 for an all-zero row) and `code` =
//! E4M3(fl32(w / scale)), rounded to nearest even and saturated to ±448. A
//! payload is the codes `[rows][cols]`, zero padding to a multiple of 256
//! bytes, then the BF16 scales. A weight converted in row blocks (the fused
//! q | k | gate | v) is one payload per block, each starting at a multiple of
//! 256 bytes.

/// Where a payload's BF16 scales start: past the codes, at a multiple of 256.
pub fn scale_offset(rows: u32, cols: u32) -> u64 {
    round_up_256(u64::from(rows) * u64::from(cols))
}

/// The bytes of one payload: codes, zero padding to 256, the BF16 scales.
pub fn payload_bytes(rows: u32, cols: u32) -> u64 {
    scale_offset(rows, cols) + u64::from(rows) * 2
}

/// Each block's payload offset, and (last) the whole weight's bytes.
pub fn block_offsets(blocks: &[u32], cols: u32) -> Vec<u64> {
    let mut at = vec![0u64];
    for &rows in blocks {
        let next = at[at.len() - 1] + round_up_256(payload_bytes(rows, cols));
        at.push(next);
    }
    at
}

fn round_up_256(bytes: u64) -> u64 {
    bytes.div_ceil(256) * 256
}

/// An E4M3FN code's value (no infinities; 0x7F / 0xFF are NaN).
pub fn e4m3_to_f32(code: u8) -> f32 {
    let magnitude = code & 0x7F;
    let value = if magnitude == 0x7F {
        f32::NAN
    } else {
        let exp = i32::from(magnitude >> 3);
        let man = f32::from(magnitude & 7);
        if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) }
    };
    if code & 0x80 != 0 { -value } else { value }
}

/// The E4M3FN code of a finite `v`, rounded to nearest even and saturated to
/// ±448, the sign kept on a zero (what `cvt.rn.satfinite.e4m3x2.f32` does).
pub fn e4m3_from_f32(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs();
    if a >= 448.0 {
        return sign | 0x7E;
    }
    // Below 2^-6 the codes are subnormal, m * 2^-9; scaling by a power of two
    // is exact, so one rounding to an integer is the only one.
    if a < 2f32.powi(-6) {
        let m = (a * 512.0).round_ties_even() as u8; // 0..=8; 8 is 2^-6, code 0x08
        return sign | m;
    }
    let exp = a.log2().floor() as i32; // -6..=8
    let exp = if 2f32.powi(exp) > a { exp - 1 } else if 2f32.powi(exp + 1) <= a { exp + 1 } else { exp };
    let m = ((a / 2f32.powi(exp) - 1.0) * 8.0).round_ties_even() as i32; // 0..=8
    let (exp, m) = if m == 8 { (exp + 1, 0) } else { (exp, m) };
    let code = (((exp + 7) as u8) << 3) | m as u8;
    sign | code.min(0x7E)
}

/// The least BF16 >= `q` (finite, positive).
fn bf16_at_least(q: f32) -> u16 {
    let bits = q.to_bits();
    let upper = (bits >> 16) as u16;
    if bits & 0xFFFF != 0 { upper + 1 } else { upper }
}

/// Write the payload of the BF16 weight `w` (`rows * cols` words, row-major)
/// into `out` (`payload_bytes(rows, cols)` long).
pub fn quantize_rows(w: &[u16], rows: u32, cols: u32, out: &mut [u8]) {
    let (rows, cols) = (rows as usize, cols as usize);
    assert_eq!(w.len(), rows * cols, "the weight is rows x cols BF16 words");
    assert_eq!(out.len() as u64, payload_bytes(rows as u32, cols as u32), "the payload's length");
    let scale_at = scale_offset(rows as u32, cols as u32) as usize;
    out[rows * cols..scale_at].fill(0);
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        let amax = row.iter().map(|&b| f32::from_bits(u32::from(b) << 16).abs()).fold(0.0f32, f32::max);
        let scale_bits = if amax == 0.0 { 0x3F80 } else { bf16_at_least(amax / 448.0) };
        let scale = f32::from_bits(u32::from(scale_bits) << 16);
        out[scale_at + 2 * r..scale_at + 2 * r + 2].copy_from_slice(&scale_bits.to_le_bytes());
        for (c, &b) in row.iter().enumerate() {
            out[r * cols + c] = e4m3_from_f32(f32::from_bits(u32::from(b) << 16) / scale);
        }
    }
}

/// The weight converted in row `blocks` (summing to its rows), from its BF16
/// bytes (little-endian words): one payload per block at [`block_offsets`].
pub fn quantize_blocks(bf16_le: &[u8], blocks: &[u32], cols: u32) -> Vec<u8> {
    let words: Vec<u16> = bf16_le.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect();
    let at = block_offsets(blocks, cols);
    let mut out = vec![0u8; at[blocks.len()] as usize];
    let mut row0 = 0usize;
    for (i, &rows) in blocks.iter().enumerate() {
        let n = rows as usize * cols as usize;
        let len = payload_bytes(rows, cols) as usize;
        quantize_rows(&words[row0..row0 + n], rows, cols, &mut out[at[i] as usize..at[i] as usize + len]);
        row0 += n;
    }
    assert_eq!(row0, words.len(), "the blocks cover the weight");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf16(v: f32) -> u16 {
        let b = v.to_bits();
        ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
    }

    #[test]
    fn every_finite_code_encodes_to_itself_and_saturates_past_448() {
        for code in 0u8..=255 {
            if code & 0x7F == 0x7F {
                continue;
            }
            assert_eq!(e4m3_from_f32(e4m3_to_f32(code)), code, "code {code:#04x}");
        }
        assert_eq!(e4m3_from_f32(448.0), 0x7E);
        assert_eq!(e4m3_from_f32(460.0), 0x7E, "past 448 saturates, never NaN");
        assert_eq!(e4m3_from_f32(-1e9), 0xFE);
        assert_eq!(e4m3_from_f32(-0.0), 0x80, "the sign kept on a zero");
        assert_eq!(e4m3_from_f32(2f32.powi(-11)), 0, "below half the least subnormal rounds to zero");
    }

    #[test]
    fn a_value_between_two_codes_rounds_to_the_nearer_and_a_tie_to_the_even() {
        // 1.0 (0x38) and 1.125 (0x39): 1.0625 is the tie, to 0x38; just above it, 0x39.
        assert_eq!(e4m3_from_f32(1.0625), 0x38);
        assert_eq!(e4m3_from_f32(1.0626), 0x39);
        // 1.125 and 1.25 (0x3A): the tie 1.1875 goes to the even 0x3A.
        assert_eq!(e4m3_from_f32(1.1875), 0x3A);
        // Subnormal: 3 * 2^-9 and 4 * 2^-9; 3.5 * 2^-9 ties to 4 (0x04).
        assert_eq!(e4m3_from_f32(3.5 * 2f32.powi(-9)), 0x04);
        // The top of a binade rounds into the next: 1.9375 * 2^-6 is a tie between 0x0F and 0x10.
        assert_eq!(e4m3_from_f32(1.9375 * 2f32.powi(-6)), 0x10);
        // Subnormal 7.5 * 2^-9 ties between 7 and 8 * 2^-9 = 2^-6 (0x08): the even, 0x08.
        assert_eq!(e4m3_from_f32(7.5 * 2f32.powi(-9)), 0x08);
    }

    #[test]
    fn a_row_scale_is_the_least_bf16_above_amax_over_448_and_no_code_saturates() {
        let cols = 32u32;
        let mut w = vec![0u16; 3 * cols as usize];
        for c in 0..cols as usize {
            w[c] = bf16(0.013 * (c as f32 - 15.0));
            w[2 * cols as usize + c] = bf16(if c == 5 { -3.5 } else { 0.001 });
        }
        let mut out = vec![0xAAu8; payload_bytes(3, cols) as usize];
        quantize_rows(&w, 3, cols, &mut out);
        let scale_at = 256usize;
        let scale = |r: usize| f32::from_bits(u32::from(u16::from_le_bytes([out[scale_at + 2 * r], out[scale_at + 2 * r + 1]])) << 16);
        let amax0 = f32::from_bits(u32::from(w[cols as usize - 1]) << 16).abs();
        assert!(scale(0) >= amax0 / 448.0 && f32::from_bits((scale(0).to_bits() >> 16 << 16) - (1 << 16)) < amax0 / 448.0);
        assert_eq!(scale(1), 1.0, "an all-zero row");
        assert_eq!(&out[cols as usize..2 * cols as usize], &[0u8; 32][..], "its codes are zero");
        assert_eq!(out[2 * cols as usize + 5], e4m3_from_f32(-3.5 / scale(2)));
        assert!(out[..3 * cols as usize].iter().all(|&c| c & 0x7F != 0x7F), "no NaN code");
        assert!(out[3 * cols as usize..scale_at].iter().all(|&b| b == 0), "the padding is zeroed");
    }

    #[test]
    fn blocks_are_payloads_at_multiples_of_256() {
        let at = block_offsets(&[6144, 1024, 6144, 1024], 5120);
        assert_eq!(at[0], 0);
        assert_eq!(at[1], 6144 * 5120 + 6144 * 2);
        assert_eq!(at[2] - at[1], 1024 * 5120 + 2048);
        assert_eq!(at[4], 2 * (6144 * 5120 + 12288) + 2 * (1024 * 5120 + 2048));
        assert!(at.iter().all(|a| a % 256 == 0));
        // A block converts as that block alone would.
        let cols = 64u32;
        let words: Vec<u16> = (0..(16 + 32) * cols).map(|i| bf16(((i * 37 % 101) as f32 - 50.0) * 0.01)).collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let both = quantize_blocks(&bytes, &[16, 32], cols);
        let at = block_offsets(&[16, 32], cols);
        let mut second = vec![0u8; payload_bytes(32, cols) as usize];
        quantize_rows(&words[16 * cols as usize..], 32, cols, &mut second);
        assert_eq!(&both[at[1] as usize..at[1] as usize + second.len()], &second[..]);
    }
}
