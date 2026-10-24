"""server_vad's oracle (spec speech/05, acceptance 1): Silero VAD's published model on the CPU.

The reference is the TorchScript model `silero_vad.jit` of the silero-vad 6.2.3 wheel, the one
`load_silero_vad()` returns, driven as its own `utils_vad.py` drives it: `reset_states()`, then
512-sample chunks at 16 kHz with the last one zero-padded, each chunk read with the 64 samples
before it. Two subcommands:

convert    the wheel's `silero_vad.jit`, its 16 kHz model (`_model`): every parameter and the STFT
           basis, unchanged f32, under the names of the wheel's own `tinygrad_model.py`, into one
           safetensors file (crates/speech/vad/). The wheel's `silero_vad_16k.safetensors` is not
           the `.jit`'s weights, so it is not used.
record     per clip, the `.jit`'s per-frame speech probabilities: the fixture clip and the 40 FLEURS
           clips into the repo (--out), the owner's dictations to the study disk (--private-out),
           never in the repo. The same graph written out in torch from the converted file is run
           in fp32 (it must reproduce the `.jit`, which checks the file and the graph) and in fp64:
           the `.jit`'s error against fp64, times --factor and at least --floor, is the bound in
           thresholds.json.

Pinned: torch 2.14.1+cpu, numpy 2.5.4, safetensors 0.8.0 (requirements.txt); the wheel
silero_vad-6.2.3-py3-none-any.whl, sha256 below (snakers4/silero-vad tag v6.2.3, 5cd79456).
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import wave
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

PINS = {"torch": "2.14.1+cpu", "numpy": "2.5.4", "safetensors": "0.8.0"}
WHEEL_SHA256 = "7b7f5436cfcb02fae583a05b512ea96467fd449fe54cb49a5e4f06c51a1e43b8"
JIT = "silero_vad/data/silero_vad.jit"
FIXTURE_CLIP = "it_it-1703"
FRAME = 512
CONTEXT = 64


def check_pins():
    got = {"torch": torch.__version__, "numpy": np.__version__, "safetensors": safetensors.__version__}
    if got != PINS:
        sys.exit(f"pinned {PINS}, running {got}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_jit(wheel):
    if sha256(wheel) != WHEEL_SHA256:
        sys.exit(f"{wheel} is not silero_vad-6.2.3-py3-none-any.whl (sha256 {WHEEL_SHA256})")
    with zipfile.ZipFile(wheel) as z:
        raw = z.read(JIT)
    model = torch.jit.load(io.BytesIO(raw), map_location="cpu")
    model.eval()
    return model, hashlib.sha256(raw).hexdigest()


def weights_of(model):
    """The 16 kHz model's tensors under tinygrad_model.py's names."""
    m = model._model
    conv = lambda i: getattr(m.encoder, str(i)).reparam_conv
    final = getattr(m.decoder.decoder, "2")
    w = {"stft_conv.weight": m.stft.forward_basis_buffer}
    for i in range(4):
        w[f"conv{i + 1}.weight"] = conv(i).weight
        w[f"conv{i + 1}.bias"] = conv(i).bias
    for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        w[f"lstm_cell.{k}"] = getattr(m.decoder.rnn, k)
    w["final_conv.weight"] = final.weight
    w["final_conv.bias"] = final.bias
    # The geometry the port is written for: stride 1, 2, 2, 1, padding 1, kernel 3.
    assert [conv(i).stride[0] for i in range(4)] == [1, 2, 2, 1] and all(conv(i).padding[0] == 1 for i in range(4))
    assert m.stft.hop_length == 128 and m.stft.filter_length == 256
    return {k: v.detach().to(torch.float32).contiguous().clone() for k, v in w.items()}


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == 16000, path
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return a.astype(np.float32) / 32768.0


def fleurs_clips(study):
    rows = list(csv.DictReader(open(os.path.join(study, "data", "clips", "manifest.csv"), encoding="utf-8")))
    return [{"id": r["clip_id"], "path": os.path.join(study, r["wav"].replace("\\", "/"))} for r in rows if r["lang"] in ("it_it", "en_us")]


def dictations(study):
    d = os.path.join(study, "dictations", "16k")
    return [{"id": f"d{i:02d}", "path": os.path.join(d, f"d{i:02d}.wav")} for i in range(1, 12)]


def padded(audio):
    x = torch.from_numpy(audio)
    if x.shape[0] % FRAME:
        x = F.pad(x, (0, FRAME - x.shape[0] % FRAME))
    return x


@torch.no_grad()
def jit_probs(model, audio):
    """As utils_vad.py drives the model: fresh state, 512-sample chunks, the last zero-padded."""
    model.reset_states()
    x = padded(audio).unsqueeze(0)
    return np.array([model(x[:, i:i + FRAME], 16000).item() for i in range(0, x.shape[1], FRAME)], dtype=np.float64)


@torch.no_grad()
def plain_probs(w, audio, dtype):
    """The graph written out: what crates/speech/src/vad.rs computes."""
    W = {k: v.to(dtype) for k, v in w.items()}
    x = padded(audio).to(dtype)
    ctx = torch.zeros(CONTEXT, dtype=dtype)
    h = torch.zeros(1, 128, dtype=dtype)
    c = torch.zeros(1, 128, dtype=dtype)
    out = []
    for i in range(0, x.shape[0], FRAME):
        chunk = torch.cat([ctx, x[i:i + FRAME]])
        ctx = chunk[-CONTEXT:]
        y = F.conv1d(F.pad(chunk[None, None], (0, 64), mode="reflect"), W["stft_conv.weight"], stride=128)
        y = torch.sqrt(y[:, :129] ** 2 + y[:, 129:] ** 2)
        for n, stride in ((1, 1), (2, 2), (3, 2), (4, 1)):
            y = F.relu(F.conv1d(y, W[f"conv{n}.weight"], W[f"conv{n}.bias"], stride=stride, padding=1))
        gates = y.squeeze(-1) @ W["lstm_cell.weight_ih"].T + W["lstm_cell.bias_ih"] + h @ W["lstm_cell.weight_hh"].T + W["lstm_cell.bias_hh"]
        i_, f_, g_, o_ = gates.chunk(4, -1)
        c = torch.sigmoid(f_) * c + torch.sigmoid(i_) * torch.tanh(g_)
        h = torch.sigmoid(o_) * torch.tanh(c)
        p = torch.sigmoid(F.conv1d(F.relu(h.unsqueeze(-1)), W["final_conv.weight"], W["final_conv.bias"]))
        out.append(p.mean().item())
    return np.array(out, dtype=np.float64)


def write_probs(path, clips, meta):
    """`path`.bin (each clip's probabilities, f32 little-endian, concatenated) and `path`.json."""
    index, off = [], 0
    with open(path + ".bin", "wb") as f:
        for cid, p in clips:
            a = np.ascontiguousarray(p, dtype="<f4")
            f.write(a.tobytes())
            index.append({"id": cid, "frames": int(a.shape[0]), "offset": off})
            off += a.nbytes
    write_json(path + ".json", {"clips": index, **meta})


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def cmd_convert(a):
    check_pins()
    torch.set_num_threads(1)
    model, jit_sha = load_jit(a.wheel)
    # One metadata key: safetensors writes a map of several in no fixed order.
    save_file(weights_of(model), a.out, metadata={"source": f"silero-vad 6.2.3 silero_vad.jit (sha256 {jit_sha}), _model (16 kHz)"})
    print(json.dumps({"out": a.out, "sha256": sha256(a.out), "bytes": os.path.getsize(a.out), "jit_sha256": jit_sha}))


def cmd_record(a):
    check_pins()
    torch.set_num_threads(1)
    model, jit_sha = load_jit(a.wheel)
    w = load_file(a.weights)
    converted = weights_of(model)
    assert sorted(w) == sorted(converted) and all(torch.equal(w[k], converted[k]) for k in w), f"{a.weights} is not the .jit's weights"

    worst = {"jit_vs_fp64": 0.0, "plain_fp32_vs_jit": 0.0}

    def probs(cid, path):
        audio = read_wav(path)
        ref = jit_probs(model, audio)
        p32, p64 = plain_probs(w, audio, torch.float32), plain_probs(w, audio, torch.float64)
        worst["jit_vs_fp64"] = max(worst["jit_vs_fp64"], float(np.abs(ref - p64).max()))
        worst["plain_fp32_vs_jit"] = max(worst["plain_fp32_vs_jit"], float(np.abs(ref - p32).max()))
        print(f"{cid}: {len(ref)} frames, {float((ref >= 0.5).mean()):.2f} at or over 0.5", flush=True)
        return cid, ref

    meta = {"reference": "silero-vad 6.2.3 silero_vad.jit, model(chunk, 16000) per 512 samples after reset_states(), the last chunk zero-padded",
            "frame_samples": FRAME, "jit_sha256": jit_sha, "weights_sha256": sha256(a.weights)}
    write_probs(os.path.join(a.out, "fixture"), [probs(FIXTURE_CLIP, a.fixture_clip)], meta)
    write_probs(os.path.join(a.out, "fleurs"), [probs(c["id"], c["path"]) for c in fleurs_clips(a.study)], meta)
    write_probs(os.path.join(a.private_out, "dictations"), [probs(d["id"], d["path"]) for d in dictations(a.study)], meta)
    # The graph written out must be the .jit's, to f32 rounding, or the bound below means nothing.
    assert worst["plain_fp32_vs_jit"] <= a.floor, worst
    write_json(os.path.join(a.out, "thresholds.json"), {
        "schema": 1,
        "metric": "max over frames of |p - the .jit's p|, a probability",
        "jit_fp32_vs_fp64": worst["jit_vs_fp64"],
        "plain_fp32_vs_jit": worst["plain_fp32_vs_jit"],
        "factor": a.factor,
        "floor": a.floor,
        "bound": max(a.factor * worst["jit_vs_fp64"], a.floor),
    })
    print(json.dumps(worst), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    r = sub.add_parser("record")
    for p in (c, r):
        p.add_argument("--wheel", required=True, help="silero_vad-6.2.3-py3-none-any.whl, from PyPI")
    c.add_argument("--out", required=True, help="crates/speech/vad/silero-vad-16k.safetensors")
    r.add_argument("--weights", required=True, help="the converted file")
    r.add_argument("--study", required=True, help="H:/speech-study: the FLEURS clips and the dictations")
    r.add_argument("--fixture-clip", required=True, help="crates/speech/tests/fixtures/oracle/it_it-1703.wav")
    r.add_argument("--out", required=True, help="crates/speech/tests/fixtures/vad")
    r.add_argument("--private-out", required=True, help="where the dictations' records go, outside the repo")
    r.add_argument("--factor", type=float, default=10.0)
    r.add_argument("--floor", type=float, default=1e-5, help="the smallest bound: f32 accumulation-order noise")
    a = ap.parse_args()
    {"convert": cmd_convert, "record": cmd_record}[a.cmd](a)


if __name__ == "__main__":
    main()
