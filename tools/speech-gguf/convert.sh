#!/usr/bin/env bash
# The speech model's GGUF files (spec speech/01 §6, acceptance 22): both
# sizes of Qwen3-ASR, decoder and audio encoder (mmproj), in bf16, Q8_0 and
# Q4_K_M, converted from Qwen's official weights at the pinned revisions with
# llama.cpp b11840's converter and quantizer. llama.cpp is an out-of-tree tool
# here (ADR 0067): it is never in the tree or the build.
#
# usage: tools/speech-gguf/convert.sh STUDY OUT
#   STUDY  the study directory: models/hf/Qwen3-ASR-{1.7B,0.6B} (Qwen's
#          original repos at 7278e1e7 and 5eb14417: b11840's mmproj converter
#          reads their `thinker_config`, which the -hf twins do not have),
#          src/llama.cpp (b11840, 36e36e63a), bin/llama-b11840-win-cpu-x64,
#          venv/hf (Python with torch, transformers and sentencepiece)
#   OUT    where the files go
set -euo pipefail
study=${1:?study directory}
out=${2:?output directory}
mkdir -p "$out"
py="$study/venv/hf/Scripts/python.exe"
[ -x "$py" ] || py="$study/venv/hf/bin/python"
convert="$study/src/llama.cpp/convert_hf_to_gguf.py"
quantize="$study/bin/llama-b11840-win-cpu-x64/llama-quantize.exe"
[ -x "$quantize" ] || quantize="$study/bin/llama-b11840-ubuntu-x64/llama-quantize"
git -C "$study/src/llama.cpp" rev-parse HEAD | grep -q '^36e36e63a' || { echo "llama.cpp is not b11840" >&2; exit 1; }

for size in 1.7B 0.6B; do
  hf="$study/models/hf/Qwen3-ASR-$size"
  for t in bf16 q8_0; do
    T=$([ "$t" = q8_0 ] && echo Q8_0 || echo bf16)
    "$py" "$convert" "$hf" --outtype "$t" --outfile "$out/Qwen3-ASR-$size-$T.gguf"
    "$py" "$convert" "$hf" --mmproj --outtype "$t" --outfile "$out/mmproj-Qwen3-ASR-$size-$T.gguf"
  done
  "$quantize" "$out/Qwen3-ASR-$size-bf16.gguf" "$out/Qwen3-ASR-$size-Q4_K_M.gguf" Q4_K_M
  # The quantizer knows only text architectures; an mmproj Q4_K_M is attempted, not required.
  "$quantize" "$out/mmproj-Qwen3-ASR-$size-bf16.gguf" "$out/mmproj-Qwen3-ASR-$size-Q4_K_M.gguf" Q4_K_M ||
    echo "llama-quantize b11840 cannot quantize the $size mmproj to Q4_K_M" >&2
done
( cd "$out" && sha256sum *.gguf > SHA256SUMS )
