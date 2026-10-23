//! BF16 bits to and from floats, for the GPU tests that build activations
//! and read outputs in BF16 (the trellis body's, GitHub #339).

// Each including test takes the conversions it needs.
#![allow(dead_code)]

/// To the nearest BF16, ties to even (finite values).
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn bf16_to_f64(bits: u16) -> f64 {
    f64::from(f32::from_bits(u32::from(bits) << 16))
}
