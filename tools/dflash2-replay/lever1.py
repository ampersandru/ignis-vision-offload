"""Lever 1 ceiling (GitHub #267): adaptive draft extent at 8 lanes, from the replay's
per-round acceptance and the measured width-cost curve.

python lever1.py OUT_DIR WIDTH_COST_JSONL CONTEXT MANIFEST

Per round the replay gives the exact survival S(i) = P(first i drafts accepted) under the
turn's sampling (draft-conditioned verify columns). A round at window k commits
T(k) = 1 + sum_{i<=k} S(i) (up to a stop). Approximation: the first k drafts do not depend
on k (the drafter's block is shorter at a smaller window; not measured here).

Policies, 8 lanes of independent rounds drawn with the traffic's cell weights:
  fixed k               every round at window k (today: 7);
  oracle shared k       one window per round for all lanes (what a verify graph per window
                        allows), chosen knowing every lane's S;
  oracle per-lane       each lane its own extent, the round costing its total columns
                        (a ragged verify, which the leaf does not have), knowing S;
  recent-acceptance     the spec's lever: a lane's extent from its own last rounds'
                        accepted counts (censored at its extent), the round's window the
                        widest lane extent.
Oracles maximize the ratio of sums (Dinkelbach: argmax T - lambda * cost per round,
lambda iterated to the achieved throughput).
"""
import json
import sys

import numpy as np

K = 7
LANES = 8
WEIGHTS = {"A-content": 28586, "A-think": 58665, "B-content": 1524816, "B-think": 8088074,
           "C-content": 2085240, "C-think": 6471440}


def load_cost(path, context):
    """cost[lanes][w] in ms for verify width w = 1 (plain decode) .. 8."""
    cost = {}
    for line in open(path):
        r = json.loads(line)
        if r["context"] != context:
            continue
        cost.setdefault(r["lanes"], {})[r["verify_width"]] = r["round"]["median_ms"]
    return cost


def main():
    out_dir, cost_path, context = sys.argv[1], sys.argv[2], int(sys.argv[3])
    d = np.load(f"{out_dir}/per_round.npz")
    manifest = json.load(open(sys.argv[4]))
    cell_of = {t["turn"]: f"{t['source']}-{t['kind']}" for j in manifest["jobs"] for t in j["turns"]}
    surv, ok = d["surv"], d["no_stop_before"]
    gains = np.concatenate([np.ones((len(surv), 1)), surv * ok], 1)  # T(k) - T(k-1) for k = 0..7
    T = np.cumsum(gains, 1)  # [N, 8]: committed at window k = 0..7
    cells = np.array([cell_of[int(t)] for t in d["turn"]])
    # draw rounds with traffic weights: each cell's rounds weighted so the cell's share of
    # rounds matches its share of the traffic's rounds (tokens / tok-per-round)
    tpr = {c: T[cells == c, K].mean() for c in set(cells)}
    share = {c: WEIGHTS[c] / tpr[c] for c in tpr}
    w = np.array([share[c] / (cells == c).sum() for c in cells])
    w /= w.sum()
    cost = load_cost(cost_path, context)
    rng = np.random.default_rng(267)
    draws = rng.choice(len(T), size=(20000, LANES), p=w)
    Tl = T[draws]  # [R, 8 lanes, 8 windows]

    c8 = np.array([cost[LANES][k + 1] for k in range(K + 1)])  # window k -> width k+1
    c1 = np.array([cost[1][k + 1] for k in range(K + 1)])
    out = {"context": context, "cost_ms_8_lanes": c8.tolist(), "cost_ms_1_lane": c1.tolist()}
    fixed8 = Tl.sum(1).mean(0) / c8  # tokens per ms
    fixed1 = T.T @ w / c1
    out["fixed_tok_s_8"] = (fixed8 * 1e3).tolist()
    out["fixed_tok_s_1"] = (fixed1 * 1e3).tolist()
    base8 = fixed8[K]

    # oracle shared window
    tot = Tl.sum(1)  # [R, 8 windows]
    lam = base8
    for _ in range(50):
        pick = (tot - lam * c8).argmax(1)
        new = tot[np.arange(len(tot)), pick].sum() / c8[pick].sum()
        if abs(new - lam) < 1e-9:
            break
        lam = new
    out["oracle_shared_tok_s_8"] = lam * 1e3
    out["oracle_shared_window_share"] = np.bincount(pick, minlength=K + 1).tolist()

    # oracle per lane, ragged cost: interpolate cost(total columns) from the 8-lane curve
    cols = LANES * np.arange(1, K + 2)
    ragged = lambda m: np.interp(m, cols, c8)  # noqa: E731
    lam = base8
    for _ in range(50):
        # greedy marginal allocation of columns per round: add the best lane's next draft
        # while its marginal tokens beat lambda * marginal cost
        ext = np.zeros((len(Tl), LANES), np.int64)
        total_tok = Tl[:, :, 0].sum(1)
        m = np.full(len(Tl), LANES)
        for _step in range(LANES * K):
            nxt = np.minimum(ext + 1, K)
            marg = np.where(ext < K, np.take_along_axis(Tl, nxt[..., None], 2)[..., 0]
                            - np.take_along_axis(Tl, ext[..., None], 2)[..., 0], -1)
            lane = marg.argmax(1)
            best = marg[np.arange(len(Tl)), lane]
            dcost = ragged(m + 1) - ragged(m)
            take = best > lam * dcost
            if not take.any():
                break
            ext[np.arange(len(Tl))[take], lane[take]] += 1
            total_tok = total_tok + np.where(take, best, 0)
            m = m + take
        new = total_tok.sum() / ragged(m).sum()
        if abs(new - lam) < 1e-9:
            break
        lam = new
    out["oracle_ragged_tok_s_8"] = lam * 1e3
    out["gain_oracle_shared"] = out["oracle_shared_tok_s_8"] / (base8 * 1e3) - 1
    out["gain_oracle_ragged"] = out["oracle_ragged_tok_s_8"] / (base8 * 1e3) - 1
    out["gain_best_fixed_8"] = fixed8.max() / base8 - 1

    # The spec's lever: each lane's extent from its own recent accepted counts, which it
    # sees censored at the extent it ran. Lanes walk their turns' rounds in order; a
    # lane's survival is modeled geometric, rho = m / (m + 1) for m its EMA of accepted
    # drafts (a censored run counted as its extent plus that EMA), and it drafts while rho^i
    # beats the column's price, at least one draft so the estimate keeps moving (lambda, the achieved
    # throughput, times the marginal cost). Shared: the round's window is the widest
    # lane's extent (a verify graph per window); ragged: each lane pays its own columns
    # at the 8-lane curve's mean slope. Rounds are scored by their exact T(extent).
    order = np.lexsort((d["anchor"], d["turn"], d["job"]))
    turns = {}
    for n in order:
        turns.setdefault((int(d["job"][n]), int(d["turn"][n])), []).append(n)
    turn_list = list(turns.values())
    turn_w = np.array([w[seq[0]] * len(seq) for seq in turn_list])
    turn_w /= turn_w.sum()
    real_acc = d["committed"].astype(int) - 1
    slope = (c8[K] - c8[1]) / (LANES * (K - 1))  # ms per drafted column, ragged
    shared_step = (c8[1:] - c8[:-1]) / LANES     # a lane's share of widening the window

    def simulate(shared, lam, rounds=6000, beta=0.3):
        tok = cost_sum = 0.0
        lanes = [None] * LANES
        for _ in range(rounds):
            ext = np.zeros(LANES, np.int64)
            idx = np.zeros(LANES, np.int64)
            for l in range(LANES):
                if lanes[l] is None or lanes[l][1] >= len(lanes[l][0]):
                    lanes[l] = [turn_list[rng.choice(len(turn_list), p=turn_w)], 0, 3.0]
                seq, pos, m = lanes[l]
                surv = (m / (m + 1)) ** np.arange(1, K + 1)
                price = lam * (shared_step if shared else np.full(K, slope))
                ext[l] = max(1, int(np.cumprod(surv > price).sum()))  # one draft always probes
                idx[l] = seq[pos]
            if shared:
                window = int(ext.max())
                ext[:] = window
                cost_sum += c8[window]
            else:
                cost_sum += ragged(LANES + ext.sum())
            tok += T[idx, ext].sum()
            for l in range(LANES):
                # a run that accepted its whole extent was censored: count it as the
                # extent plus the lane's current expectation beyond it
                seen = real_acc[idx[l]]
                seen = seen if seen < ext[l] else ext[l] + lanes[l][2]
                lanes[l][2] = (1 - beta) * lanes[l][2] + beta * seen
                lanes[l][1] += 1
        return tok / cost_sum

    for shared in (True, False):
        lam = base8
        for _ in range(4):
            lam = simulate(shared, lam)
        key = "recent_shared" if shared else "recent_ragged"
        out[f"{key}_tok_s_8"] = lam * 1e3
        out[f"gain_{key}"] = lam / base8 - 1
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{out_dir}/lever1_{context}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
