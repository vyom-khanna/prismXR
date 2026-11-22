"""Highlight-segment utilities shared by both branches.

Every highlight list goes through `finalize`, which guarantees the README rules: sorted, strictly
non-overlapping, inside [0, duration], every kept segment >= MIN_SEG, at least one segment, and a total
no larger than the budget (except for videos too short to fit a single MIN_SEG segment in 15%).
"""
from __future__ import annotations

import numpy as np

from . import config as C


def budget_seconds(dur: float) -> float:
    return C.TARGET_FILL * dur


def _round_down(x):
    return np.floor(x * 100) / 100


def _round_up(x):
    return np.ceil(x * 100) / 100


def finalize(segs: list[tuple[float, float]], dur: float, fallback_center: float | None = None):
    """Clean a list of (start, end) into a valid highlight list of {"segment": [s, e]}."""
    # > 4.00 s: after 2-decimal rounding, e.g. 4.22 - 0.22 == 3.9999999999999996 < 4.0 in float, and a 4.01 s window
    # around an x.xx5 centre rounds inward to exactly 4.00 s.
    min_len = min(C.MIN_SEG + C.MIN_LEN_PAD, dur)
    out = []
    for s, e in sorted(segs):
        s, e = max(0.0, float(s)), min(dur, float(e))
        if e - s < min_len - 1e-6:  # grow short segments to min_len around their centre
            c = (s + e) / 2
            s, e = max(0.0, c - min_len / 2), min(dur, c + min_len / 2)
            if e - s < min_len - 1e-6:  # clipped by a video edge: shift inward
                s, e = (0.0, min_len) if s <= 1e-9 else (dur - min_len, dur)
        if out and s <= out[-1][1] + 0.02:  # overlap / touch / near-touch (rounding gap) -> merge
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    if not out:
        c = dur / 2 if fallback_center is None else fallback_center
        s = min(max(0.0, c - min_len / 2), dur - min_len)
        out = [(s, s + min_len)]

    # enforce budget: drop/trim from the end of the score order is the caller's job; here only trim
    # proportionally if merging grew us over budget, never below MIN_SEG.
    cap = max(C.BUDGET * dur, min_len)
    total = sum(e - s for s, e in out)
    while total > cap + 1e-6 and len(out) > 1:
        # remove the shortest segment (least content) until within budget
        i = min(range(len(out)), key=lambda k: out[k][1] - out[k][0])
        total -= out[i][1] - out[i][0]
        out.pop(i)
    if total > cap + 1e-6:  # single long segment: shrink around its centre
        s, e = out[0]
        c, L = (s + e) / 2, max(cap, min_len)
        s = min(max(0.0, c - L / 2), dur - L)
        out = [(s, s + L)]

    res, last_end = [], -1.0
    dur_r = _round_down(dur)
    for s, e in out:
        s, e = round(float(_round_up(s)), 2), round(float(_round_down(e)), 2)
        if s <= last_end:
            s = round(float(_round_up(last_end + 0.01)), 2)
        if e - s < C.MIN_SEG:  # rounding/neighbour shaved the window: extend end, else pull start back
            e = round(min(s + min_len, dur_r), 2)
            if e - s < C.MIN_SEG:
                s = round(max(e - min_len, last_end + 0.01, 0.0), 2)
        ok = e - s >= C.MIN_SEG  # exactly the validator's float test
        if not ok and res and e > res[-1]["segment"][1]:  # can't stand alone: fold into previous window
            res[-1]["segment"][1] = e
            last_end = e
            continue
        if ok or (not res and e > s):
            res.append({"segment": [s, e]})
            last_end = e
    if not any(h["segment"][1] - h["segment"][0] >= C.MIN_SEG for h in res) and dur >= C.MIN_SEG + 0.01:
        # last resort: one valid window at the start
        res = [{"segment": [0.0, round(min(C.MIN_SEG + 0.01, dur_r), 2)]}]
    return res


def peaks_to_segments(scores: np.ndarray, ts: np.ndarray, dur: float, seg_len: float = 5.0,
                      budget: float | None = None) -> list[tuple[float, float]]:
    """Greedy non-maximum selection of fixed-length windows on a per-sample score curve.

    Picks the highest-scoring window centre, suppresses its neighbourhood, repeats until the budget
    is used. Window score = mean score of samples inside the window.
    """
    budget = budget_seconds(dur) if budget is None else budget
    seg_len = max(C.MIN_SEG, seg_len)
    if budget < C.MIN_SEG or len(scores) == 0:
        c = float(ts[int(np.argmax(scores))]) if len(scores) else dur / 2
        return [(c - C.MIN_SEG / 2, c + C.MIN_SEG / 2)]
    # window score for each candidate centre
    half = seg_len / 2
    cs = np.cumsum(np.concatenate([[0.0], scores]))
    lo = np.searchsorted(ts, ts - half, "left")
    hi = np.searchsorted(ts, ts + half, "right")
    win = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    order = np.argsort(-win)
    chosen, used = [], 0.0
    for i in order:
        if used + seg_len > budget + 1e-6:
            break
        c = float(np.clip(ts[i], half, dur - half))
        s, e = c - half, c + half
        if any(not (e <= a or s >= b) for a, b in chosen):
            continue
        chosen.append((s, e))
        used += seg_len
    return chosen


def uniform_segments(dur: float, seg_len: float = 5.0) -> list[tuple[float, float]]:
    n = max(1, int(budget_seconds(dur) // seg_len))
    gap = dur / n
    return [((i + 0.5) * gap - seg_len / 2, (i + 0.5) * gap + seg_len / 2) for i in range(n)]
