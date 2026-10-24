"""The speech oracle (spec speech/01 §5, ADR 0043): Transformers FP32 on the CPU.

The checkpoint's own modelling code is the reference, never llama.cpp (whose chunking
differs). Two subcommands:

record     per size: the fixture clip's log-mel, every encoder layer, the projector and every
           decoder layer, teacher-forced on the official prompt (empty system turn) with the
           reference's own greedy transcript; and, for every FLEURS clip and dictation, the
           audio-token count, the prompt and the greedy output. FLEURS goes into the repo
           (--out); the owner's dictations and the README context stay on the study disk
           (--private-out), never in the repo.
yardstick  per catalog variant: the same teacher-forced pass with the GGUF pair's weights
           dequantized (gguf-py of llama.cpp b11840) and each matmul's input quantized as the
           executor quantizes it (Q8_K for Q4_K/Q6_K weights, Q8_0 for Q8_0), all in fp32. Its
           error against the record, times --factor and at least --floor (f32 accumulation
           noise, for the stem, which quantizes nothing), is that variant's bound in
           thresholds.json. Also the log-mel's bound: torch's fp32 front end against fp64.

Pinned: torch 2.14.1+cpu, transformers 5.20.2, numpy 2.5.4 (requirements.txt); checkpoints
Qwen/Qwen3-ASR-1.7B-hf @ bcd2b5b7 and Qwen/Qwen3-ASR-0.6B-hf @ 7f1569a4.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
import wave

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

PINS = {"torch": "2.14.1+cpu", "transformers": "5.20.2", "numpy": "2.5.4"}
REVISIONS = {"1.7B": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3", "0.6B": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"}
# Two encoder windows (104 + 14 tokens) and a partial last chunk: 9.06 s of Italian.
FIXTURE_CLIP = "it_it-1703"
ROWS = 6
MAX_NEW = 1024


def check_pins():
    got = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__}
    if got != PINS:
        sys.exit(f"pinned {PINS}, running {got}")


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1 and w.getframerate() == 16000, path
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return a.astype(np.float32) / 32768.0


def fleurs_clips(study):
    rows = list(csv.DictReader(open(os.path.join(study, "data", "clips", "manifest.csv"), encoding="utf-8")))
    out = []
    for r in rows:
        if r["lang"] not in ("it_it", "en_us"):
            continue
        out.append({"id": r["clip_id"], "path": os.path.join(study, r["wav"].replace("\\", "/")), "reference": r["reference"]})
    return out


def dictations(study):
    d = os.path.join(study, "dictations")
    return [{"id": f"d{i:02d}", "path": os.path.join(d, "16k", f"d{i:02d}.wav")} for i in range(1, 12)]


def load(hf_dir, size):
    rev_file = os.path.join(hf_dir, ".cache", "huggingface", "download", "config.json.metadata")
    rev = open(rev_file, encoding="utf-8").read().split()[0]
    if rev != REVISIONS[size]:
        sys.exit(f"{hf_dir} is at {rev}, the oracle pins {REVISIONS[size]}")
    torch.set_grad_enabled(False)
    processor = AutoProcessor.from_pretrained(hf_dir)
    model = AutoModelForMultimodalLM.from_pretrained(hf_dir, dtype=torch.float32, device_map="cpu")
    model.eval()
    return processor, model


def request(processor, audio, prompt=None):
    return processor.apply_transcription_request(audio=audio, prompt=prompt).to("cpu", torch.float32)


def transcribe(processor, model, audio):
    inputs = request(processor, audio)
    gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    prompt = inputs["input_ids"][0].tolist()
    out = gen[0, len(prompt):].tolist()
    parsed = processor.decode(gen[:, len(prompt):], return_format="parsed")[0]
    n_audio = int((inputs["input_ids"][0] == model.config.audio_token_id).sum())
    return inputs, prompt, out, parsed, n_audio


def text_ids(ids, audio_id):
    return [t for t in ids if t != audio_id]


class Taps:
    """Forward hooks on every module the oracle compares: encoder stem (layer 0's input), each
    encoder layer, ln_post, the projector, each decoder layer, the final norm."""

    def __init__(self, model):
        self.out = {}
        inner = model.model
        at, lm = inner.audio_tower, inner.language_model
        self.handles = [at.layers[0].register_forward_pre_hook(self._pre("enc.stem"))]
        for i, layer in enumerate(at.layers):
            self.handles.append(layer.register_forward_hook(self._post(f"enc.layer.{i}")))
        self.handles.append(at.ln_post.register_forward_hook(self._post("enc.post_ln")))
        self.handles.append(inner.multi_modal_projector.register_forward_hook(self._post("proj")))
        for i, layer in enumerate(lm.layers):
            self.handles.append(layer.register_forward_hook(self._post(f"dec.layer.{i}")))
        self.handles.append(lm.norm.register_forward_hook(self._post("dec.norm")))

    @staticmethod
    def _tensor(x):
        while isinstance(x, (tuple, list)):
            x = x[0]
        return x.detach().float().reshape(-1, x.shape[-1]).numpy()

    def _pre(self, key):
        def f(mod, args, kwargs=None):
            self.out[key] = self._tensor(args[0] if args else kwargs["hidden_states"])
        return f

    def _post(self, key):
        def f(mod, args, out):
            self.out[key] = self._tensor(out)
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


def teacher_forced(processor, model, audio, out_ids):
    """One forward over the official prompt plus the reference's greedy output, tapped."""
    inputs = request(processor, audio)
    ids = torch.tensor([inputs["input_ids"][0].tolist() + out_ids])
    taps = Taps(model)
    logits = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        input_features=inputs["input_features"],
        input_features_mask=inputs["input_features_mask"],
    ).logits[0]
    taps.remove()
    mask = inputs["input_features_mask"][0].bool()
    mel = inputs["input_features"][0][:, mask].numpy().astype(np.float32)
    return inputs, ids[0].tolist(), taps.out, logits, mel


def rows_of(n):
    return sorted({int(round(i * (n - 1) / (ROWS - 1))) for i in range(ROWS)})


def sampled(taps, enc_rows, dec_rows):
    out = {}
    for k, v in taps.items():
        out[k] = v[enc_rows if k.startswith(("enc.", "proj")) else dec_rows]
    return out


def write_fixture(path, tensors, meta):
    """`path`.bin (f32 little-endian, concatenated) and `path`.json (the index and `meta`)."""
    index, off = {}, 0
    with open(path + ".bin", "wb") as f:
        for k, v in tensors.items():
            a = np.ascontiguousarray(v, dtype="<f4")
            f.write(a.tobytes())
            index[k] = {"shape": list(a.shape), "offset": off}
            off += a.nbytes
    with open(path + ".json", "w", encoding="utf-8", newline="\n") as f:
        json.dump({"tensors": index, **meta}, f, indent=1, ensure_ascii=False)
        f.write("\n")


def read_fixture(path):
    meta = json.load(open(path + ".json", encoding="utf-8"))
    raw = open(path + ".bin", "rb").read()
    t = {}
    for k, e in meta["tensors"].items():
        n = int(np.prod(e["shape"]))
        t[k] = np.frombuffer(raw, dtype="<f4", count=n, offset=e["offset"]).reshape(e["shape"])
    return t, meta


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def cmd_record(a):
    check_pins()
    processor, model = load(a.hf_dir, a.size)
    audio_id = model.config.audio_token_id
    size_dir = os.path.join(a.out, a.size.lower())
    os.makedirs(size_dir, exist_ok=True)
    t0 = time.time()

    clips = fleurs_clips(a.study)
    fleurs = []
    for c in clips:
        _, prompt, out, parsed, n_audio = transcribe(processor, model, read_wav(c["path"]))
        fleurs.append({"id": c["id"], "n_audio_tokens": n_audio, "text_tokens": len(prompt) - n_audio, "output": out,
                       "language": parsed["language"], "transcript": parsed["transcription"]})
        print(f"{a.size} {c['id']}: {n_audio} audio tokens, {len(out)} out, {time.time() - t0:.0f} s", flush=True)
    no_ctx = text_ids(request(processor, read_wav(clips[0]["path"]))["input_ids"][0].tolist(), audio_id)
    write_json(os.path.join(size_dir, "fleurs.json"), {
        "checkpoint": f"Qwen/Qwen3-ASR-{a.size}-hf@{REVISIONS[a.size]}", "pins": PINS,
        "prompt": "official, empty system turn", "decoding": "greedy",
        "no_context_text_tokens": no_ctx, "clips": fleurs})

    priv = os.path.join(a.private_out, a.size.lower())
    dicts = []
    for d in dictations(a.study):
        _, prompt, out, parsed, n_audio = transcribe(processor, model, read_wav(d["path"]))
        dicts.append({"id": d["id"], "n_audio_tokens": n_audio, "output": out, "language": parsed["language"],
                      "transcript": parsed["transcription"]})
        print(f"{a.size} {d['id']}: {n_audio} audio tokens, {time.time() - t0:.0f} s", flush=True)
    readme = open(os.path.join(a.study, "dictations", "context_readme.txt"), encoding="utf-8").read()
    ctx = text_ids(request(processor, read_wav(clips[0]["path"]), prompt=readme)["input_ids"][0].tolist(), audio_id)
    write_json(os.path.join(priv, "dictations.json"), {
        "checkpoint": f"Qwen/Qwen3-ASR-{a.size}-hf@{REVISIONS[a.size]}", "pins": PINS, "dictations": dicts,
        "readme_context_text_tokens": ctx})

    fx = next(c for c in clips if c["id"] == FIXTURE_CLIP)
    ref = next(f for f in fleurs if f["id"] == FIXTURE_CLIP)
    inputs, ids, taps, logits, mel = teacher_forced(processor, model, read_wav(fx["path"]), ref["output"])
    prompt_len = len(ids) - len(ref["output"])
    n_enc = taps["enc.layer.0"].shape[0]
    enc_rows, dec_rows = rows_of(n_enc), rows_of(len(ids))
    top1 = logits.argmax(-1).tolist()
    tensors = {"mel": mel, **sampled(taps, enc_rows, dec_rows)}
    write_fixture(os.path.join(size_dir, "layers"), tensors, {
        "checkpoint": f"Qwen/Qwen3-ASR-{a.size}-hf@{REVISIONS[a.size]}", "pins": PINS, "clip": FIXTURE_CLIP,
        "input_ids": ids, "prompt_len": prompt_len, "n_audio_tokens": ref["n_audio_tokens"],
        "enc_rows": enc_rows, "dec_rows": dec_rows,
        # top1[p] predicts ids[p + 1]; over the output it is the greedy transcript itself.
        "top1": top1, "wall_s": round(time.time() - t0, 1)})
    print(f"recorded {a.size} in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- yardstick

def fake_q8k(x):
    """f32 → Q8_K → f32, per 256 along the last axis, as the executor quantizes (ties to even)."""
    s = x.shape
    b = x.reshape(-1, 256)
    idx = b.abs().argmax(-1, keepdim=True)
    mx = b.gather(-1, idx)
    iscale = torch.where(mx == 0, torch.zeros_like(mx), -127.0 / mx)
    q = torch.round(b * iscale).clamp(max=127)
    d = torch.where(mx == 0, torch.zeros_like(mx), 1.0 / iscale)
    return (q * d).reshape(s)


def fake_q80(x):
    s = x.shape
    b = x.reshape(-1, 32)
    amax = b.abs().amax(-1, keepdim=True)
    d = (amax / 127.0).to(torch.float16).to(torch.float32)
    idv = torch.where(amax == 0, torch.zeros_like(amax), 127.0 / amax)
    q = torch.round(b * idv)
    return (q * d).reshape(s)


class SimLinear(torch.nn.Module):
    def __init__(self, lin, weight, act):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.from_numpy(weight), requires_grad=False)
        self.bias = lin.bias
        self.act = act

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        return torch.nn.functional.linear(x, self.weight, self.bias)


def gguf_tensors(path, gguf_py):
    sys.path.insert(0, gguf_py)
    import gguf
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        shape = [int(d) for d in reversed(t.shape.tolist())]
        w = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
        out[t.name] = (w, t.tensor_type.name)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def install(model, dec, enc):
    """Replaces every weight the GGUF pair holds; quantized matmuls also quantize their input."""
    inner = model.model
    at, lm, mp = inner.audio_tower, inner.language_model, inner.multi_modal_projector
    act_of = {"Q4_K": fake_q8k, "Q6_K": fake_q8k, "Q8_0": fake_q80}

    def lin(parent, attr, src, name):
        w, ty = src[name + ".weight"]
        old = getattr(parent, attr)
        assert tuple(old.weight.shape) == w.shape, (name, old.weight.shape, w.shape)
        setattr(parent, attr, SimLinear(old, w, act_of.get(ty)))
        if name + ".bias" in src:
            parent_bias = getattr(parent, attr)
            parent_bias.bias = torch.nn.Parameter(torch.from_numpy(src[name + ".bias"][0].reshape(-1)), requires_grad=False)

    def param(p, src, name):
        w, _ = src[name]
        p.data.copy_(torch.from_numpy(w.reshape(p.shape)))

    for i in (1, 2, 3):
        conv = getattr(at, f"conv2d{i}")
        param(conv.weight, enc, f"a.conv2d.{i}.weight")
        param(conv.bias, enc, f"a.conv2d.{i}.bias")
    param(at.conv_out.weight, enc, "a.conv_out.weight")
    for i, layer in enumerate(at.layers):
        p = f"a.blk.{i}."
        for attr, g in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("out_proj", "attn_out")):
            lin(layer.self_attn, attr, enc, p + g)
        lin(layer, "fc1", enc, p + "ffn_up")
        lin(layer, "fc2", enc, p + "ffn_down")
        for ln, g in ((layer.self_attn_layer_norm, "ln1"), (layer.final_layer_norm, "ln2")):
            param(ln.weight, enc, p + g + ".weight")
            param(ln.bias, enc, p + g + ".bias")
    param(at.ln_post.weight, enc, "a.post_ln.weight")
    param(at.ln_post.bias, enc, "a.post_ln.bias")
    lin(mp, "linear_1", enc, "mm.a.mlp.1")
    lin(mp, "linear_2", enc, "mm.a.mlp.2")

    emb = lm.embed_tokens
    emb.weight = torch.nn.Parameter(torch.from_numpy(dec["token_embd.weight"][0].copy()), requires_grad=False)
    head = "output" if "output.weight" in dec else "token_embd"
    lin(model, "lm_head", dec, head)
    for i, layer in enumerate(lm.layers):
        p = f"blk.{i}."
        for attr, g in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("o_proj", "attn_output")):
            lin(layer.self_attn, attr, dec, p + g)
        for attr, g in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")):
            lin(layer.mlp, attr, dec, p + g)
        param(layer.self_attn.q_norm.weight, dec, p + "attn_q_norm.weight")
        param(layer.self_attn.k_norm.weight, dec, p + "attn_k_norm.weight")
        param(layer.input_layernorm.weight, dec, p + "attn_norm.weight")
        param(layer.post_attention_layernorm.weight, dec, p + "ffn_norm.weight")
    param(lm.norm.weight, dec, "output_norm.weight")


def row_rel_l2(y, r):
    """Each row's relative L2 error, then their RMS."""
    e = np.linalg.norm(y.astype(np.float64) - r, axis=-1) / np.maximum(np.linalg.norm(r.astype(np.float64), axis=-1), 1e-30)
    return float(np.sqrt(np.mean(e ** 2)))


def mel_f64(audio, processor):
    """The same front end in fp64 with numpy: reflect pad, Hann, |STFT|², slaney mel, log10."""
    fe = processor.feature_extractor
    x = np.asarray(audio, dtype=np.float64)
    if len(x) < fe.min_length:
        x = np.pad(x, (0, fe.min_length - len(x)))
    n_fft, hop = fe.n_fft, fe.hop_length
    xp = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = 1 + (len(xp) - n_fft) // hop
    spec = np.empty((n_fft // 2 + 1, frames - 1))
    for t in range(frames - 1):
        spec[:, t] = np.abs(np.fft.rfft(xp[t * hop:t * hop + n_fft] * win)) ** 2
    mel = fe.mel_filters.astype(np.float64).T @ spec
    log = np.log10(np.maximum(mel, 1e-10))
    log = np.maximum(log, log.max() - 8.0)
    return (log + 4.0) / 4.0


def cmd_yardstick(a):
    check_pins()
    processor, model = load(a.hf_dir, a.size)
    size_dir = os.path.join(a.out, a.size.lower())
    ref, meta = read_fixture(os.path.join(size_dir, "layers"))
    clip = next(c for c in fleurs_clips(a.study) if c["id"] == meta["clip"])
    audio = read_wav(clip["path"])
    out_ids = meta["input_ids"][meta["prompt_len"]:]

    dec, enc = gguf_tensors(a.decoder, a.gguf_py), gguf_tensors(a.encoder, a.gguf_py)
    install(model, dec, enc)
    _, ids, taps, logits, _ = teacher_forced(processor, model, audio, out_ids)
    assert ids == meta["input_ids"]
    sim = sampled(taps, meta["enc_rows"], meta["dec_rows"])
    errors = {k: row_rel_l2(sim[k], ref[k]) for k in sim}
    span = range(meta["prompt_len"] - 1, len(ids) - 1)
    top1 = logits.argmax(-1).tolist()
    agree = sum(top1[p] == meta["top1"][p] for p in span) / len(span)

    path = os.path.join(a.out, "thresholds.json")
    th = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {"schema": 1, "variants": {}}
    th["metric"] = "row_rel_l2: each sampled row's relative L2 error against the record, then the RMS over rows"
    th["factor"] = a.factor
    # A stage with no quantized matmul (the convolutional stem) has a sim error at f32's own
    # rounding; its bound is the floor, f32 accumulation-order noise, not twice that rounding.
    th["floor"] = a.floor
    th["variants"][a.variant] = {
        "size": a.size,
        "decoder": {"file": os.path.basename(a.decoder), "sha256": sha256(a.decoder)},
        "encoder": {"file": os.path.basename(a.encoder), "sha256": sha256(a.encoder)},
        "sim": {k: round(v, 7) for k, v in errors.items()},
        "bound": {k: round(max(a.factor * v, a.floor), 7) for k, v in errors.items()},
        "sim_top1_agreement": round(agree, 4),
    }
    # The log-mel's bound: torch's fp32 front end against fp64, times the factor.
    mel64 = mel_f64(audio, processor)
    mel_err = float(np.abs(ref["mel"].astype(np.float64) - mel64).max())
    th["mel"] = {"metric": "max |mel − fp64 mel| in normalised log-mel units", "torch_fp32": mel_err,
                 "bound": max(a.factor * mel_err, 1e-5)}
    write_json(path, th)
    print(json.dumps({"variant": a.variant, "top1": agree, "worst": max(errors.items(), key=lambda kv: kv[1]),
                      "mel_fp32_vs_fp64": mel_err}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    y = sub.add_parser("yardstick")
    for p in (r, y):
        p.add_argument("--size", required=True, choices=("1.7B", "0.6B"))
        p.add_argument("--hf-dir", required=True)
        p.add_argument("--study", required=True, help="H:/speech-study: the FLEURS clips and the dictations")
        p.add_argument("--out", required=True, help="crates/speech/tests/fixtures/oracle")
    r.add_argument("--private-out", required=True, help="where the dictations' records go, outside the repo")
    y.add_argument("--variant", required=True)
    y.add_argument("--decoder", required=True)
    y.add_argument("--encoder", required=True)
    y.add_argument("--gguf-py", required=True, help="llama.cpp b11840's gguf-py directory")
    y.add_argument("--factor", type=float, default=2.0)
    y.add_argument("--floor", type=float, default=1e-5, help="the smallest bound: f32 accumulation-order noise")
    a = ap.parse_args()
    {"record": cmd_record, "yardstick": cmd_yardstick}[a.cmd](a)


if __name__ == "__main__":
    main()
