"""Reader for dflash2_replay's per-round records (crates/runtime/examples/dflash2_replay.rs)."""
import json
import os

import numpy as np

K = 7
C = K + 1
TOPK = 16
TT = 20  # the vendored sampler's candidate cap

RECORD = np.dtype([
    ("job", "<u4"), ("turn", "<u4"), ("anchor", "<u4"), ("committed", "<u4"), ("extent", "<u4"),
    ("forced", "<i4", (C,)),
    ("drafts", "<i4", (K,)),
    ("cand_ids", "<i4", (K, TOPK)),
    ("cand_logits", "<f4", (K, TOPK)),
    ("scores", "<f4", (K, TOPK, TOPK)),  # [s][pred][succ]
    ("drafter_lse", "<f4", (K,)),
    ("top_ids", "<i4", (C, TT)),
    ("top_logits", "<f4", (C, TT)),
    ("target_lse", "<f4", (C,)),
    ("forced_logit", "<f4", (C,)),
    ("agreed", "<u4"),
])
assert RECORD.itemsize == 9520, RECORD.itemsize


def load(out_dir, manifest):
    """All rounds of every finished job, and the jobs' categories (per stream position)."""
    parts, cats = [], {}
    for j, job in enumerate(manifest["jobs"]):
        path = os.path.join(out_dir, job["job"] + ".rounds")
        if not os.path.exists(os.path.join(out_dir, job["job"] + ".done")):
            continue
        parts.append(np.fromfile(path, dtype=RECORD))
        cats[j] = np.fromfile(os.path.join(os.path.dirname(manifest["_path"]), job["job"] + ".cat"), dtype="i1")
    return (np.concatenate(parts) if parts else np.zeros(0, RECORD)), cats


def read_manifest(path):
    m = json.load(open(path))
    m["_path"] = path
    return m


def sampling_probs(top_logits, temperature=1.0, top_p=0.95):
    """The vendored sampler's distribution over each row's top-20 (descending logits):
    softmax at `temperature` over the 20, then the smallest prefix whose mass reaches
    `top_p` of theirs (inclusive), renormalized. Rows past the support get 0."""
    x = top_logits.astype(np.float64) / temperature
    e = np.exp(x - x[..., :1])
    total = e.sum(-1, keepdims=True)
    cum = np.cumsum(e, -1)
    # support = first index where cum >= top_p * total, plus one
    reach = cum >= top_p * total
    support = np.argmax(reach, -1) + 1
    keep = np.arange(TT) < support[..., None]
    e = np.where(keep, e, 0.0)
    return e / e.sum(-1, keepdims=True)


def prob_of(top_ids, probs, token):
    """p(token) from (ids, probs) tables whose last axis is the 20 entries; `token`
    broadcasts against the tables' leading dims."""
    out = np.zeros(np.broadcast_shapes(top_ids.shape[:-1], np.shape(token)))
    for k in range(top_ids.shape[-1]):
        out += np.where(top_ids[..., k] == token, probs[..., k], 0.0)
    return out
