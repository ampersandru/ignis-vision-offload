"""DFlash2 acceptance on replayed agent turns (GitHub #267): the split and the ceilings.

python analyze.py OUT_DIR MANIFEST

Reads dflash2_replay's records. Conventions: a round's drafts d_1..d_7 are verified by its
columns 0..6 (column j is the target's distribution after the anchor and d_1..d_j); the
recorded tokens after the anchor are `forced[0..7]`. Columns 0..c-1 of a round that
committed c tokens sit on the recorded chain; together the rounds of a turn cover every
position of it once, which gives a per-position table of the target's distribution along
the recorded text (`P`), used for any one-hot proposal's Rao-Blackwellized acceptance:

    E[accepted] = sum_i prod_{j<i} 1[r_j == x_j] * p_rec(x_i | position a+i-1)

(unbiased under the coupling "the recorded token is a sample of the target", exact for the
source-A turns whose contexts are complete).
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import C, K, TOPK, TT, load, prob_of, read_manifest, sampling_probs  # noqa: E402

IM_END, EOT = 248046, 248044  # filled from the tokenizer below if available
CATS = ["think", "prose", "tool-syntax", "tool-args", "edit/write"]


def stop_ids():
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file("F:/ai/models/Qwen3.8-27B-nf4/tokenizer.json")
        return [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")]
    except Exception:
        return [IM_END, EOT]


def leading(match):
    """Leading-True count along the last axis."""
    return np.cumprod(match, -1).sum(-1)


def rb(indicator, p):
    """sum_i prod_{j<i} ind_j * p_i over the last axis (i, j 0-based columns)."""
    before = np.cumprod(np.concatenate([np.ones_like(indicator[..., :1]), indicator[..., :-1]], -1), -1)
    return (before * p).sum(-1)


def walk_greedy(scores):
    """The vendored selector walk: from predecessor 0, argmax successor per column
    (ties toward the smaller index). scores [N, 7, pred, succ] -> [N, 7] indices."""
    n = scores.shape[0]
    pred = np.zeros(n, np.int64)
    out = np.zeros((n, K), np.int64)
    for s in range(K):
        row = scores[np.arange(n), s, pred]  # [N, succ]
        pred = row.argmax(-1)
        out[:, s] = pred
    return out


def walk_expected_length(scores, temperature):
    """The path maximizing the expected accepted length under the drafter's own
    conditional q_s(succ | pred) = softmax(scores[s, pred] / temperature):
    V_s(pred) = max_succ q_s(succ|pred) (1 + V_{s+1}(succ)), V_7 = 0. As temperature -> 0
    q is one-hot on the greedy walk's choice, so the rule is the greedy walk there."""
    n = scores.shape[0]
    x = scores / temperature
    q = np.exp(x - x.max(-1, keepdims=True))
    q /= q.sum(-1, keepdims=True)  # [N, 7, pred, succ]
    value = np.zeros((n, TOPK))
    choice = np.zeros((n, K, TOPK), np.int64)
    for s in range(K - 1, -1, -1):
        gain = q[:, s] * (1 + value[:, None, :])  # [N, pred, succ]
        choice[:, s] = gain.argmax(-1)
        value = gain.max(-1)
    out = np.zeros((n, K), np.int64)
    pred = np.zeros(n, np.int64)
    for s in range(K):
        pred = choice[np.arange(n), s, pred]
        out[:, s] = pred
    return out


def main():
    out_dir = sys.argv[1]
    manifest = read_manifest(sys.argv[2])
    R, cats = load(out_dir, manifest)
    stops = np.array(stop_ids())
    n = len(R)
    print(f"rounds {n}, readout disagreements {int((R['agreed'] == 0).sum())}")
    jobs = manifest["jobs"]
    turn_meta = {}
    for j, job in enumerate(jobs):
        if j not in cats:
            continue  # not replayed (yet)
        for t in job["turns"]:
            turn_meta[(j, t["turn"])] = dict(t, job_index=j)

    probs = sampling_probs(R["top_logits"])  # [N, 8, 20]
    # --- per-position recorded-chain table --------------------------------
    offsets, total = {}, 0
    for key, t in sorted(turn_meta.items()):
        offsets[key] = total
        total += t["end"] - t["start"]
    P_ids = np.full((total, TT), -1, np.int32)
    P_probs = np.zeros((total, TT))
    P_logit0 = np.full(total, np.nan)
    P_lse = np.full(total, np.nan)
    P_mass20 = np.full(total, np.nan)
    P_rec = np.full(total, -1, np.int32)
    P_cat = np.full(total, -1, np.int8)
    covered = np.zeros(total, bool)
    keys = list(zip(R["job"].tolist(), R["turn"].tolist()))
    base = np.array([offsets[k] - turn_meta[k]["start"] for k in keys], np.int64)
    ends = np.array([turn_meta[k]["end"] for k in keys], np.int64)
    for i in range(C):
        on_chain = i < R["committed"]
        idx = (base + R["anchor"] + i)[on_chain]
        P_ids[idx] = R["top_ids"][on_chain, i]
        P_probs[idx] = probs[on_chain, i]
        P_logit0[idx] = R["top_logits"][on_chain, i, 0]
        P_lse[idx] = R["target_lse"][on_chain, i]
        P_mass20[idx] = np.exp(R["top_logits"][on_chain, i].astype(np.float64) - R["target_lse"][on_chain, i, None]).sum(-1)
        P_rec[idx] = R["forced"][on_chain, i]
        covered[idx] = True
    for key, off in offsets.items():
        t = turn_meta[key]
        c = cats[t["job_index"]]
        # the category of the token each position predicts (the one after the
        # turn's last is outside it: -1)
        span = np.full(t["end"] - t["start"], -1, np.int8)
        nxt = c[t["start"] + 1:t["end"]]
        span[:len(nxt)] = nxt
        P_cat[off:off + len(span)] = span
    print(f"positions {total}, covered {covered.mean():.4f}")
    P_pmax = P_probs[:, 0]
    P_argmax = P_ids[:, 0]

    # gather index of a round's columns 0..6 in the per-position table
    col = np.arange(K)
    gidx = base[:, None] + R["anchor"][:, None].astype(np.int64) + col[None, :]
    gvalid = (R["anchor"][:, None].astype(np.int64) + col[None, :]) < ends[:, None]
    gidx = np.where(gvalid, gidx, 0)
    rec = R["forced"][:, :K]  # recorded tokens verified by columns 0..6

    def p_rec(tokens):
        """p_rec(tokens[n, i] | position a+i) along the recorded chain."""
        return np.where(gvalid, prob_of(P_ids[gidx], P_probs[gidx], tokens), 0.0)

    drafts = R["drafts"]
    is_stop = np.isin(drafts, stops)
    no_stop_before = np.cumprod(np.concatenate([np.ones((n, 1), bool), ~is_stop[:, :-1]], 1), 1).astype(bool)

    # --- the actual drafter ------------------------------------------------
    realized = R["committed"].astype(float)                       # coupling walk
    a_exact = prob_of(R["top_ids"][:, :K], probs[:, :K], drafts)  # draft-conditioned, exact
    surv = np.cumprod(a_exact, 1)
    exact = 1 + (surv * no_stop_before).sum(1)
    rb_actual = 1 + rb((rec == drafts).astype(float), p_rec(drafts))
    greedy_match = drafts == R["top_ids"][:, :K, 0]
    greedy = 1 + leading(greedy_match & no_stop_before)

    # --- one-hot oracles and multi-candidate ceilings ----------------------
    g = np.where(gvalid, P_argmax[gidx], -2)
    rb_oracle = 1 + rb((rec == g).astype(float), np.where(gvalid, P_pmax[gidx], 0.0))
    cand = R["cand_ids"]  # [N, 7, 16]
    cand_p = np.where(gvalid[..., None], prob_of(P_ids[gidx][:, :, None, :], P_probs[gidx][:, :, None, :], cand), 0.0)
    best = cand_p.argmax(-1)
    lattice_best_tok = np.take_along_axis(cand, best[..., None], -1)[..., 0]
    rb_lattice = 1 + rb((rec == lattice_best_tok).astype(float), cand_p.max(-1))
    multi = {}
    for m in (2, 4, 16):
        inset = (cand[:, :, :m] == rec[..., None]).any(-1)
        multi[m] = 1 + rb(inset.astype(float), cand_p[:, :, :m].sum(-1))

    # --- selection rules walked over the same lattice (lever 2) ------------
    scores = R["scores"].astype(np.float64)  # [N, 7, pred, succ]
    host_walk = walk_greedy(scores)
    walk_ok = (np.take_along_axis(cand, host_walk[..., None], -1)[..., 0] == drafts).all(1)
    print(f"host greedy walk reproduces the device drafts in {walk_ok.mean():.5f} of rounds")

    def evaluate(path):
        tok = np.take_along_axis(cand, path[..., None], -1)[..., 0]
        return 1 + rb((rec == tok).astype(float), p_rec(tok)), tok

    selection = {}
    for t_d in (0.5, 1.0, 2.0):
        path = walk_expected_length(scores, t_d)
        selection[f"dp_T{t_d}"], tok = evaluate(path)
        print(f"  expected-length walk at drafter temperature {t_d}: differs from greedy in "
              f"{(tok != drafts).any(1).mean():.4f} of rounds, first draft in {(tok[:, 0] != drafts[:, 0]).mean():.4f}")
    unary = R["cand_logits"].astype(np.float64)
    selection["unary_argmax"], _ = evaluate(np.zeros(drafts.shape, np.int64))  # candidate 0 = the head's argmax

    per_round = {"realized": realized, "exact": exact, "rb_actual": rb_actual, "greedy_verdict": greedy,
                 "oracle_argmax": rb_oracle, "lattice_best": rb_lattice,
                 **{f"multi_top{m}": v for m, v in multi.items()}, **selection}
    np.savez_compressed(os.path.join(out_dir, "per_round.npz"), surv=surv, no_stop_before=no_stop_before,
                        committed=R["committed"], turn=R["turn"], job=R["job"], anchor=R["anchor"],
                        cand_p=cand_p.astype(np.float32), gvalid=gvalid,
                        rec_index=np.where((cand == rec[..., None]).any(-1), (cand == rec[..., None]).argmax(-1), -1).astype(np.int8),
                        **{k: v for k, v in per_round.items()})

    # --- per turn, per cell ------------------------------------------------
    round_turn = np.array([turn_meta[k]["turn"] for k in keys])
    cell_of = {t["turn"]: f"{t['source']}-{t['kind']}" for t in turn_meta.values()}
    tokens_of = {t["turn"]: t["end"] - t["start"] for t in turn_meta.values()}
    cells = sorted(set(cell_of.values()))
    # traffic weights: output-token share of each cell in the whole dataset
    weights = {"A-content": 28586, "A-think": 58665, "B-content": 1524816, "B-think": 8088074,
               "C-content": 2085240, "C-think": 6471440}
    report = {"rounds": n, "positions": total, "cells": {}}
    print("\ntok/round (committed per round) by cell")
    head = "cell        turns rounds " + " ".join(f"{k:>14}" for k in per_round)
    print(head)
    pooled = defaultdict(float)
    wsum = 0.0
    for cell in cells + ["all"]:
        sel = np.ones(n, bool) if cell == "all" else np.array([cell_of[t] == cell for t in round_turn])
        turns = {t for t in round_turn[sel]}
        row = {k: float(v[sel].mean()) for k, v in per_round.items()}
        report["cells"][cell] = {"turns": len(turns), "rounds": int(sel.sum()), **row}
        print(f"{cell:11} {len(turns):5} {int(sel.sum()):6} " + " ".join(f"{row[k]:14.3f}" for k in per_round))
        if cell != "all":
            # tok/round of a cell = its tokens / its rounds; traffic pooling by token share:
            # total rounds = sum_c tokens_c / tpr_c
            for k in per_round:
                pooled[k] += weights[cell] / row[k]
            wsum += weights[cell]
    print("traffic-weighted " + " ".join(f"{k}={wsum / pooled[k]:.3f}" for k in per_round))
    report["traffic_weighted"] = {k: wsum / pooled[k] for k in per_round}

    # --- per-position split along the recorded chain -----------------------
    reached = col[None, :] < R["committed"][:, None]  # columns whose context is the recorded chain
    pm = np.where(gvalid, P_pmax[gidx], np.nan)
    pp = a_exact
    tc = np.where(gvalid, P_cat[gidx], -1)
    print("\nper draft column (reached): P(accept) = p(draft), p(argmax), inherent 1-p(argmax), drafter loss")
    split = {}
    for j in range(K):
        s = reached[:, j] & gvalid[:, j]
        split[j] = {"n": int(s.sum()), "p_draft": float(pp[s, j].mean()), "p_argmax": float(pm[s, j].mean()),
                    "draft_is_argmax": float((drafts[s, j] == g[s, j]).mean()),
                    "top16_mass": float(cand_p[s, j].sum(-1).mean()), "top4_mass": float(cand_p[s, j, :4].sum(-1).mean()),
                    "top2_mass": float(cand_p[s, j, :2].sum(-1).mean())}
        v = split[j]
        print(f"  d{j + 1}: n={v['n']:7} p_draft={v['p_draft']:.3f} p_argmax={v['p_argmax']:.3f} "
              f"inherent={1 - v['p_argmax']:.3f} drafter={v['p_argmax'] - v['p_draft']:.3f} "
              f"draft==argmax {v['draft_is_argmax']:.3f} top2 {v['top2_mass']:.3f} top4 {v['top4_mass']:.3f} top16 {v['top16_mass']:.3f}")
    report["per_column"] = split
    print("\nper output kind (reached columns, all positions pooled)")
    kinds = {}
    for ci, name in enumerate(CATS):
        s = reached & gvalid & (tc == ci)
        kinds[name] = {"n": int(s.sum()), "p_draft": float(pp[s].mean()), "p_argmax": float(pm[s].mean()),
                       "draft_is_argmax": float((drafts[s] == g[s]).mean())}
        v = kinds[name]
        print(f"  {name:11} n={v['n']:7} p_draft={v['p_draft']:.3f} p_argmax={v['p_argmax']:.3f} "
              f"inherent={1 - v['p_argmax']:.3f} drafter={v['p_argmax'] - v['p_draft']:.3f} draft==argmax {v['draft_is_argmax']:.3f}")
    report["per_kind"] = kinds
    print(f"\ntarget top-20 mass of the full softmax (T=1), mean {np.nanmean(P_mass20):.4f}; "
          f"recorded token outside the sampler's support: {float(np.mean(prob_of(P_ids, P_probs, P_rec)[P_rec >= 0] == 0)):.4f}")
    json.dump(report, open(os.path.join(out_dir, "analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
