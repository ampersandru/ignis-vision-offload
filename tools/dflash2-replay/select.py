"""Pick the ~200 agent turns the DFlash2 acceptance replay walks (GitHub #267).

Source: the copy-drafting dataset (F:/ai/opencode/inference/.scratch/copy-drafting-2026-09-24),
13,102 turns rendered with the artifact's chat template and tokenized with its tokenizer.

Strata: source (A opencode #191 trace, complete bodies; B qwen-code main; C qwen-code
subagents) x kind (think: >= 50% of the turn's output tokens are thinking; content: the
rest). Only turns joined to their own ninfer request-log DFlash2 stats (the per-turn
validation). B/C turns lack qwen-code's system prompt and tool schemas: kept only when the
recorded prompt is longer than the rendered context by a plausible amount (B 15-45K, C
10-30K tokens) and the reconstructed output matches the recorded completion length within
10%. Rendered context <= 150K tokens (GPU cost).

python select.py OUT_DIR

Writes OUT_DIR/manifest.json and OUT_DIR/<job>.tok (int32 LE, the stream up to its last selected
turn's end) and OUT_DIR/<job>.cat (int8: -1 context, 0 think, 1 prose, 2 tool syntax,
3 tool args, 4 edit/write).
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np

DATA = "F:/ai/opencode/inference/.scratch/copy-drafting-2026-09-24/data"
SEED = 267
PER_CELL = {("A", "think"): 25, ("A", "content"): 25,
            ("B", "think"): 38, ("B", "content"): 37,
            ("C", "think"): 38, ("C", "content"): 37}
MAX_CONTEXT = 150_000


def source(t):
    s = t["stream"]
    return "A" if s.startswith("oc") else ("B" if s.startswith("qc") else "C")


def main():
    OUT = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    turns = [json.loads(l) for l in open(os.path.join(DATA, "turns.jsonl"), encoding="utf-8")]
    cats = {}

    def cat(sid):
        if sid not in cats:
            cats[sid] = np.load(os.path.join(DATA, "streams", sid + ".npz"))["cat"]
        return cats[sid]

    pool = defaultdict(list)
    for i, t in enumerate(turns):
        d = t.get("dflash2")
        if not d or t["start"] > MAX_CONTEXT:
            continue
        s = source(t)
        out_len = t["end"] - t["start"]
        if not (0.9 <= out_len / max(d["completion_tokens"], 1) <= 1.1):
            continue
        diff = d["prompt_tokens"] - t["start"]
        if s == "B" and not (15_000 <= diff <= 45_000):
            continue
        if s == "C" and not (10_000 <= diff <= 30_000):
            continue
        c = cat(t["stream"])[t["start"]:t["end"]]
        kind = "think" if (c == 0).mean() >= 0.5 else "content"
        pool[(s, kind)].append((i, t, kind))

    rng = random.Random(SEED)
    chosen = []
    for cell, n in PER_CELL.items():
        cands = sorted(pool[cell], key=lambda x: x[0])
        pick = rng.sample(cands, min(n, len(cands)))
        print(cell, "eligible", len(cands), "picked", len(pick))
        chosen += [(i, t, kind) for i, t, kind in pick]

    by_stream = defaultdict(list)
    for i, t, kind in chosen:
        by_stream[t["stream"]].append((i, t, kind))
    jobs = []
    ctx_tokens = out_tokens = 0
    for sid in sorted(by_stream):
        items = sorted(by_stream[sid], key=lambda x: x[1]["start"])
        tokens = np.load(os.path.join(DATA, "streams", sid + ".npz"))["tokens"].astype("<i4")
        end = max(t["end"] for _, t, _ in items)
        job = "j%03d" % len(jobs)
        tokens[:end].tofile(os.path.join(OUT, job + ".tok"))
        cat(sid)[:end].astype("i1").tofile(os.path.join(OUT, job + ".cat"))
        jobs.append({"job": job, "stream": sid, "tokens": int(end), "turns": [
            {"turn": i, "start": t["start"], "end": t["end"], "source": source(t), "kind": kind,
             "lane_class": t.get("lane_class"), "dflash2": t["dflash2"]} for i, t, kind in items]})
        ctx_tokens += end - sum(t["end"] - t["start"] for _, t, _ in items)
        out_tokens += sum(t["end"] - t["start"] for _, t, _ in items)
    # Shortest first: the first GPU hold reports early.
    jobs.sort(key=lambda j: j["tokens"])
    json.dump({"seed": SEED, "per_cell": {f"{k[0]}-{k[1]}": v for k, v in PER_CELL.items()},
               "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                            "presence_penalty": 0.0, "frequency_penalty": 0.0},
               "jobs": jobs}, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("jobs", len(jobs), "turns", len(chosen), "prefill tokens", ctx_tokens, "output tokens", out_tokens)


if __name__ == "__main__":
    main()
