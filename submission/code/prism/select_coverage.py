"""Coverage-optimal highlight selection for kitchen (EPIC) videos.

Action Coverage = share of DISTINCT ground-truth (verb, noun) pairs captured by the highlights.
We treat each predicted action instance i as a pair occurrence with probability p_i, and pick
windows greedily to maximise the expected number of distinct pairs covered per second of budget:

    E[covered pairs] = sum_pair (1 - prod_{i in pair, i covered} (1 - p_i))

Greedy on this submodular objective with a length cost is the standard (1 - 1/e)-style heuristic.
Candidate windows: start grid (step) x lengths; an instance is covered if its midpoint is inside.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class CovCfg:
    lengths: tuple = (4.0, 5.0)  # 2-fold CV best (vs (4,), (4,5,6,8))
    step: float = 0.5
    max_instances: int = 2000
    calib_pow: float = 2.0      # p = (score / max_score) ** calib_pow  (CV best of 0.5/1/2/3)
    p_cap: float = 0.95
    fill_rest: bool = True      # spend leftover budget on uncovered stretches
    widen: bool = True          # then grow windows to use the last (<MIN_SEG) seconds of budget
    dens_pow: float = 0.0       # p_i *= (local action density / mean density) ** dens_pow   (0 = off)
    dens_w: float = 0.2         # + dens_w * (expected #actions in window, from the density model)
    dens_smooth: int = 5        # density smoothing (feature steps) before reading it at instance midpoints


def select(actions: list[dict], dur: float, cfg: CovCfg = CovCfg(), budget: float | None = None,
           density: tuple | None = None):
    """density: optional (rate [T], t_center [T]) from prism.density (expected action midpoints per step)."""
    budget = C.TARGET_FILL * dur if budget is None else budget
    if budget < C.MIN_SEG or not actions:
        if not actions:
            return []
        a = max(actions, key=lambda x: x["score"])
        c = sum(a["segment"]) / 2
        return [(c - C.MIN_SEG / 2, c + C.MIN_SEG / 2)]

    acts = sorted(actions, key=lambda x: -x["score"])[: cfg.max_instances]
    mids = np.array([(a["segment"][0] + a["segment"][1]) / 2 for a in acts])
    sc = np.array([a["score"] for a in acts], float)
    p = (sc / max(sc.max(), 1e-9)) ** cfg.calib_pow
    use_d = density is not None and len(density[0]) > 0 and (cfg.dens_pow or cfg.dens_w)
    if use_d:
        lam, tc = np.asarray(density[0], float), np.asarray(density[1], float)
        k = max(1, cfg.dens_smooth)
        lam_s = np.convolve(np.pad(lam, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, "valid")
        if cfg.dens_pow:
            rel = np.interp(mids, tc, lam_s) / max(lam_s.mean(), 1e-9)
            p = p * rel ** cfg.dens_pow
            p = p / max(p.max(), 1e-9)
    p = np.clip(p, 0, cfg.p_cap)
    pair_ids = {}
    pid = np.array([pair_ids.setdefault((a["verb"], a["noun"]), len(pair_ids)) for a in acts])
    o = np.argsort(mids)
    mids, p, pid = mids[o], p[o], pid[o]
    resid = np.ones(len(pair_ids))           # prod (1 - p) of covered instances per pair
    covered = np.zeros(len(mids), bool)

    # candidate windows
    cands = []
    for L in cfg.lengths:
        if L > dur:
            continue
        st = np.arange(0.0, dur - L + 1e-9, cfg.step)
        cands.append(np.stack([st, st + L], 1))
    cands = np.concatenate(cands) if cands else np.array([[0.0, min(dur, C.MIN_SEG)]])
    lo = np.searchsorted(mids, cands[:, 0], "left")
    hi = np.searchsorted(mids, cands[:, 1], "right")
    length = cands[:, 1] - cands[:, 0]
    alive = np.ones(len(cands), bool)
    dgain = 0.0
    if use_d and cfg.dens_w:
        dcs = np.concatenate([[0.0], np.cumsum(lam)])
        dgain = cfg.dens_w * (dcs[np.searchsorted(tc, cands[:, 1], "right")] - dcs[np.searchsorted(tc, cands[:, 0], "left")])

    chosen, used = [], 0.0
    while True:
        # marginal gain of instance i = p_i * resid[pair_i] (0 if already covered)
        w = np.where(covered, 0.0, p * resid[pid])
        cs = np.concatenate([[0.0], np.cumsum(w)])
        gain = cs[hi] - cs[lo] + dgain
        ratio = np.where(alive & (used + length <= budget + 1e-6), gain / length, -1.0)
        j = int(np.argmax(ratio))
        if ratio[j] <= 1e-9:
            break
        s, e = cands[j]
        chosen.append((float(s), float(e)))
        used += e - s
        inside = np.arange(lo[j], hi[j])
        for i in inside[~covered[inside]]:
            resid[pid[i]] *= 1 - p[i]
        covered[inside] = True
        alive &= (cands[:, 1] <= s) | (cands[:, 0] >= e)   # no overlap (touching allowed)

    if cfg.fill_rest and used + C.MIN_SEG <= budget:
        extra = _fill(chosen, dur, budget - used)
        chosen += extra
        used += sum(e - s for s, e in extra)
    if cfg.widen and chosen and budget - used > 0.05:
        chosen = _widen(sorted(chosen), dur, budget - used)
    return chosen


def _widen(segs, dur, left):
    """Spend leftover budget (< MIN_SEG) by growing every window symmetrically, without overlaps."""
    for _ in range(3):  # a few passes; neighbours/edges may block some growth
        if left <= 0.05:
            break
        per = left / (2 * len(segs))
        out = []
        for i, (s, e) in enumerate(segs):
            lo = out[-1][1] if out else 0.0
            hi = segs[i + 1][0] if i + 1 < len(segs) else dur
            ns, ne = max(lo, s - per), min(hi, e + per)
            left -= (s - ns) + (ne - e)
            out.append((ns, ne))
        segs = out
    return segs


def _fill(chosen, dur, left, L=None):
    """Spread remaining budget as MIN_SEG windows in the largest uncovered gaps."""
    L = L or C.MIN_SEG
    out = []
    taken = sorted(chosen)
    while left >= L:
        gaps, prev = [], 0.0
        for s, e in taken + [(dur, dur)]:
            if s - prev >= L + 1.0:
                gaps.append((s - prev, prev, s))
            prev = max(prev, e)
        if not gaps:
            break
        _, a, b = max(gaps)
        c = (a + b) / 2
        w = (c - L / 2, c + L / 2)
        out.append(w)
        taken = sorted(taken + [w])
        left -= L
    return out
