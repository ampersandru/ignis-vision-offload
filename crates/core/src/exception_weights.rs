//! How the 27B holds its nine BF16 exception tensors on the device (GitHub
//! #341, spec prefill-5090/01 lever 1): as stored in the artifact, or
//! converted at load to FP8 row-scale (ADR 0065), which a call past 8 rows --
//! a prefill chunk, a verify round of two lanes or more -- multiplies on the
//! FP8 tensor cores (activations quantized per token, W8A8), and a decode
//! round or a one-lane verify round with BF16 activations (W8A16). A load
//! argument, fixed for the life of the load:
//! the binder places the converted tensors (`ignis_artifact::ModelScope::fp8_exceptions`)
//! and the leaf dispatches on their format.

use std::fmt;

/// `--model-exception-weights`: `bf16` (the default, as stored) or `fp8`.
/// FP8 is ~3% faster in prefill at the same 4K KLD, but moves two of the
/// decision fixtures (finding 2026-10-21-the-bf16-exception-tensors-in-fp8),
/// so it stays opt-in until the owner's call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExceptionWeights {
    /// As the artifact stores them.
    #[default]
    Bf16,
    /// Converted at load to FP8 row-scale.
    Fp8,
}

impl ExceptionWeights {
    /// The spellings `parse` accepts, canonical first.
    pub const NAMES: [&'static str; 2] = ["bf16", "fp8"];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }

    /// Whether the load converts them.
    pub const fn is_fp8(self) -> bool {
        matches!(self, Self::Fp8)
    }

    /// Case-insensitive.
    pub fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::Bf16),
            "fp8" => Ok(Self::Fp8),
            other => Err(format!("unknown exception weights `{other}` (expected `bf16` or `fp8`)")),
        }
    }
}

impl fmt::Display for ExceptionWeights {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_is_the_default() {
        assert_eq!(ExceptionWeights::default(), ExceptionWeights::Bf16);
        assert!(!ExceptionWeights::Bf16.is_fp8() && ExceptionWeights::Fp8.is_fp8());
    }

    #[test]
    fn parse_is_case_insensitive_and_refuses_the_rest() {
        assert_eq!(ExceptionWeights::parse("FP8"), Ok(ExceptionWeights::Fp8));
        for name in ExceptionWeights::NAMES {
            assert_eq!(ExceptionWeights::parse(name).map(ExceptionWeights::as_str), Ok(name));
        }
        assert!(ExceptionWeights::parse("nvfp4").unwrap_err().contains("fp8"));
    }
}
