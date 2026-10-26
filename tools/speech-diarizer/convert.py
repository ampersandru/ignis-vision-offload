"""The diarizer's GGUF (spec speech/08): nvidia/Nemotron-3-Diarization's weights for crates/speech/src/diarizer.

Reads a local snapshot of the checkpoint (config.json, processor_config.json, model.safetensors at f667ed73)
and writes one GGUF v3 file. Our own writer: no llama.cpp, no gguf package.

- The matmul weights are stored in --type: F32 (unchanged), F16, or Q8_0 (blocks of 32, an f16 scale of
  max|x| / 127 and int8 values rounded half away from zero, as ggml's reference quantizer). The norms, the
  biases and the silence embedding stay F32.
- The upsampler's convolution (1536 x 192 x 3) is stored with its taps outermost, 1536 rows of
  [tap][channel], so that frame t's input row is frames t-1, t and t+1 side by side.
- The metadata carries the architecture and the checkpoint's `streaming_config` (the speaker cache's
  constants, its default sizes) and offline sizes under `nemotron3_diarization.*`, the front end's
  parameters, the source revision and the licence (OpenMDW-1.1).

usage: python -I convert.py --model DIR --type F32|F16|Q8_0 --out FILE
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np
from safetensors.numpy import load_file

REVISION = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
WEIGHTS_SHA256 = "c074d86335b3b794f8fa5edc25594558f128bdb3914d27806a3a5a2e44963cb6"
ARCH = "nemotron3_diarization"
ALIGN = 32
# GGUF value and tensor type codes
U32, F32V, BOOL, STR, ARR = 4, 6, 7, 8, 9
T_F32, T_F16, T_Q8_0 = 0, 1, 8


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def enc_value(v):
    if isinstance(v, bool):
        return BOOL, struct.pack("<B", v)
    if isinstance(v, int):
        return U32, struct.pack("<I", v)
    if isinstance(v, float):
        return F32V, struct.pack("<f", v)
    if isinstance(v, str):
        return STR, enc_str(v)
    raise TypeError(v)


def q8_0(x):
    """ggml's quantize_row_q8_0_ref: per 32 values, d = max|x| / 127 as f16, q = round(x / d)."""
    rows, k = x.shape
    b = x.reshape(rows, k // 32, 32).astype(np.float32)
    amax = np.abs(b).max(axis=2, keepdims=True)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0).astype(np.float32)
    v = b * inv
    q = (np.sign(v) * np.floor(np.abs(v) + 0.5)).astype(np.int8)
    out = np.empty((rows, k // 32), dtype=[("d", "<f2"), ("q", "i1", (32,))])
    out["d"] = d[..., 0].astype(np.float16)
    out["q"] = q
    return out.tobytes()


def tensors_of(st, head_cfg):
    """(GGUF name, array, is a matmul weight) in file order; the checkpoint's names in comments."""
    t = []
    tower = "model.audio_tower."
    t.append(("enc.embd.weight", st[tower + "embedder.projection.weight"], True))
    t.append(("enc.input_ln.weight", st[tower + "input_layer_norm.weight"], False))
    t.append(("enc.input_ln.bias", st[tower + "input_layer_norm.bias"], False))
    n_layers = 1 + max(int(k.split(".")[3]) for k in st if k.startswith(tower + "layers."))
    for i in range(n_layers):
        p, q = f"{tower}layers.{i}.", f"enc.blk.{i}."
        t += [(q + "ln1.weight", st[p + "layer_norm1.weight"], False), (q + "ln1.bias", st[p + "layer_norm1.bias"], False),
              (q + "attn_q.weight", st[p + "self_attn.q_proj.weight"], True),
              (q + "attn_k.weight", st[p + "self_attn.k_proj.weight"], True),
              (q + "attn_v.weight", st[p + "self_attn.v_proj.weight"], True),
              (q + "attn_out.weight", st[p + "self_attn.o_proj.weight"], True),
              (q + "attn_out.bias", st[p + "self_attn.o_proj.bias"], False),
              (q + "ln2.weight", st[p + "layer_norm2.weight"], False), (q + "ln2.bias", st[p + "layer_norm2.bias"], False),
              (q + "ffn_up.weight", st[p + "mlp.fc1.weight"], True), (q + "ffn_up.bias", st[p + "mlp.fc1.bias"], False),
              (q + "ffn_down.weight", st[p + "mlp.fc2.weight"], True), (q + "ffn_down.bias", st[p + "mlp.fc2.bias"], False)]
    t += [("enc.post_ln.weight", st[tower + "layer_norm.weight"], False), ("enc.post_ln.bias", st[tower + "layer_norm.bias"], False),
          ("head.proj.weight", st["model.proj.weight"], True), ("head.proj.bias", st["model.proj.bias"], False)]
    conv = st["model.upsampler.conv.weight"]  # (out, in, tap)
    assert conv.shape == (head_cfg["hidden_size"] * head_cfg["subsampling_factor"], head_cfg["hidden_size"], 3), conv.shape
    t += [("head.upsample.weight", np.ascontiguousarray(conv.transpose(0, 2, 1)).reshape(conv.shape[0], -1), True),
          ("head.upsample.bias", st["model.upsampler.conv.bias"], False),
          ("head.dense.weight", st["classifier.dense.weight"], True), ("head.dense.bias", st["classifier.dense.bias"], False),
          ("head.out.weight", st["classifier.out_proj.weight"], True), ("head.out.bias", st["classifier.out_proj.bias"], False),
          ("silence_embd", st["silence_embeds"], False)]
    assert len(t) == len(st), f"{len(t)} tensors written of the checkpoint's {len(st)}"
    return t


def metadata(cfg, proc, ty):
    a, h, s = cfg["audio_config"], cfg["head_config"], cfg["streaming_config"]
    fe = proc["feature_extractor"]
    assert a["rope_parameters"]["rope_type"] == "default" and a["partial_rotary_factor"] == 1.0
    assert a["num_key_value_heads"] == a["num_attention_heads"] and a["hidden_act"] == "gelu"
    kv = {
        "general.architecture": ARCH,
        "general.name": "Nemotron 3 Diarization",
        "general.source.huggingface.repository": "nvidia/Nemotron-3-Diarization",
        "general.source.revision": REVISION,
        "general.license": "OpenMDW-1.1",
        "general.file_type": {T_F32: "F32", T_F16: "F16", T_Q8_0: "Q8_0"}[ty],
        f"{ARCH}.block_count": a["num_hidden_layers"],
        f"{ARCH}.embedding_length": a["hidden_size"],
        f"{ARCH}.feed_forward_length": a["intermediate_size"],
        f"{ARCH}.attention.head_count": a["num_attention_heads"],
        f"{ARCH}.attention.layer_norm_epsilon": 1e-5,
        f"{ARCH}.rope.freq_base": float(a["rope_parameters"]["rope_theta"]),
        f"{ARCH}.num_mel_bins": a["num_mel_bins"],
        f"{ARCH}.subsampling_factor": a["subsampling_factor"],
        f"{ARCH}.head.hidden_size": h["hidden_size"],
        f"{ARCH}.head.num_speakers": h["num_speakers"],
        f"{ARCH}.front_end.n_fft": fe["n_fft"],
        f"{ARCH}.front_end.win_length": fe["win_length"],
        f"{ARCH}.front_end.hop_length": fe["hop_length"],
        f"{ARCH}.front_end.preemphasis": float(fe["preemphasis"]),
        f"{ARCH}.front_end.sampling_rate": fe["sampling_rate"],
        f"{ARCH}.offline.chunk_length": cfg["chunk_length"],
        f"{ARCH}.offline.chunk_right_context": cfg["chunk_right_context"],
        f"{ARCH}.offline.fifo_length": cfg["fifo_length"],
        f"{ARCH}.offline.speaker_cache_update_period": cfg["speaker_cache_update_period"],
    }
    for k in ("speaker_cache_length", "fifo_length", "speaker_cache_update_period", "speaker_cache_silence_frames_per_speaker"):
        kv[f"{ARCH}.streaming.{k}"] = int(s[k])
    for k in ("prediction_score_threshold", "latest_frames_score_boost", "strong_boost_rate", "weak_boost_rate",
              "min_positive_scores_rate"):
        kv[f"{ARCH}.streaming.{k}"] = float(s[k])
    assert s["num_speakers"] == h["num_speakers"] and s["subsampling_factor"] == a["subsampling_factor"]
    return kv


def write(path, kv, tensors, ty):
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for k, v in kv.items():
        code, body = enc_value(v)
        head += enc_str(k) + struct.pack("<I", code) + body
    blobs, off = [], 0
    for name, arr, matmul in tensors:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        t = ty if matmul else T_F32
        if t == T_Q8_0:
            assert arr.ndim == 2 and arr.shape[1] % 32 == 0, (name, arr.shape)
            data = q8_0(arr)
        elif t == T_F16:
            data = arr.astype("<f2").tobytes()
        else:
            data = arr.astype("<f4").tobytes()
        dims = list(reversed(arr.shape))  # ggml order: the row length first
        head += enc_str(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        head += struct.pack("<IQ", t, off)
        blobs.append((off, data))
        off = (off + len(data) + ALIGN - 1) // ALIGN * ALIGN
    head += b"\0" * (-len(head) % ALIGN)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(head)
        base = len(head)
        for o, data in blobs:
            f.write(b"\0" * (base + o - f.tell()))
            f.write(data)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="nvidia/Nemotron-3-Diarization @ f667ed73")
    ap.add_argument("--type", required=True, choices=["F32", "F16", "Q8_0"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    weights = os.path.join(a.model, "model.safetensors")
    if sha256(weights) != WEIGHTS_SHA256:
        sys.exit(f"{weights} is not nvidia/Nemotron-3-Diarization @ {REVISION[:8]}")
    cfg = json.load(open(os.path.join(a.model, "config.json"), encoding="utf-8"))
    proc = json.load(open(os.path.join(a.model, "processor_config.json"), encoding="utf-8"))
    ty = {"F32": T_F32, "F16": T_F16, "Q8_0": T_Q8_0}[a.type]
    st = load_file(weights)
    write(a.out, metadata(cfg, proc, ty), tensors_of(st, cfg["head_config"]), ty)
    print(json.dumps({"out": a.out, "bytes": os.path.getsize(a.out), "sha256": sha256(a.out)}))


if __name__ == "__main__":
    main()
