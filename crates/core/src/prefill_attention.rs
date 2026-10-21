//! The precision of the 27B's prompt attention band pass (GitHub #335, spec
//! 27b-16gb/03 solution 6): the 8-bit band by default, the FP16-accumulate fallback
//! behind a load option. A load argument, fixed for the life of the load:
//! `kernel/src/gqa_layer.cu` hands it to every hq-e8-2b prefill chunk.

use std::fmt;

/// `--model-prefill-attention`: `fp8` (the default) or `fp16`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillAttention {
    /// The band pass in 8 bits: Q·Kᵀ in INT8, P·V in E4M3.
    #[default]
    Fp8,
    /// The FP16-accumulate fallback.
    Fp16,
}

impl PrefillAttention {
    /// The spellings `parse` accepts, canonical first.
    pub const NAMES: [&'static str; 2] = ["fp8", "fp16"];

    /// `struct ignis_model_load_options::prefill_attention`: 0 = 8-bit,
    /// 1 = FP16 accumulate.
    pub const fn abi_code(self) -> u32 {
        match self {
            Self::Fp8 => 0,
            Self::Fp16 => 1,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Fp16 => "fp16",
        }
    }

    /// Case-insensitive.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "fp8" => Ok(Self::Fp8),
            "fp16" => Ok(Self::Fp16),
            other => Err(format!("unknown prefill attention `{other}` (expected `fp8` or `fp16`)")),
        }
    }
}

impl fmt::Display for PrefillAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_is_the_default_and_the_abi_codes_are_the_headers() {
        assert_eq!(PrefillAttention::default(), PrefillAttention::Fp8);
        assert_eq!(PrefillAttention::Fp8.abi_code(), 0);
        assert_eq!(PrefillAttention::Fp16.abi_code(), 1);
    }

    #[test]
    fn parse_is_case_insensitive_and_refuses_the_rest() {
        assert_eq!(PrefillAttention::parse("FP16"), Ok(PrefillAttention::Fp16));
        for name in PrefillAttention::NAMES {
            assert_eq!(PrefillAttention::parse(name).map(PrefillAttention::as_str), Ok(name));
        }
        assert!(PrefillAttention::parse("bf16").unwrap_err().contains("fp16"));
    }
}
