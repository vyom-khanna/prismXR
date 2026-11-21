"""Combine verb detections + noun detections (+ SlowFast clip noun probabilities + train-split prior)
into scored (verb, noun) action segments.

For every verb detection we build a 300-way noun distribution from
  (a) noun detections overlapping it in time (score x tIoU), and
  (b) SlowFast clip-level noun softmax averaged over the segment,
re-weight it with P(noun | verb) from EPIC-100 train, and emit the top-k nouns as actions with
score = verb_score * noun_score. Hyper-parameters are tuned with tools/eval/score.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairCfg:
    top_verbs: int = 1500     # verb detections considered
    nouns_per_verb: int = 8   # nouns emitted per verb detection (2-fold CV: 1..15; >8 within noise)
    w_det: float = 0.3        # weight of noun-detection evidence (2-fold CV with LaViLa: 0.4 -> 0.3)
    w_clip: float = 0.0       # weight of SlowFast clip-softmax noun evidence (used if no segment classifier)
    w_seg: float = 0.4        # weight of segment-classifier noun head (LaViLa noun where matched)
    w_act: float = 0.3        # weight of segment-classifier action head, conditioned on the detected verb
    noun_pow: float = 0.75    # action score = verb_score * noun_evidence ** noun_pow
    prior_pow: float = 0.0    # exponent on P(noun|verb); 0 disables (CV: prior did not help)
    min_tiou: float = 0.1     # noun det must overlap the verb det at least this much
    max_actions: int = 2000


def _tiou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = np.clip(np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def segment_mean(prob: np.ndarray, t_center: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Mean of clip-level probabilities whose centre falls in each segment (nearest clip if none)."""
    cs = np.concatenate([np.zeros((1, prob.shape[1])), np.cumsum(prob, 0)])
    lo = np.searchsorted(t_center, segs[:, 0], "left")
    hi = np.searchsorted(t_center, segs[:, 1], "right")
    empty = hi <= lo
    near = np.clip(np.searchsorted(t_center, segs.mean(1)), 0, len(t_center) - 1)
    lo = np.where(empty, near, lo)
    hi = np.where(empty, near + 1, hi)
    return (cs[hi] - cs[lo]) / (hi - lo)[:, None]


def pair_actions(verb_dets, noun_dets, clip_noun, t_center, prior_n_given_v, verbs, nouns,
                 cfg: PairCfg = PairCfg(), seg_model=None):
    """verb_dets/noun_dets: (segs [N,2], scores [N], labels [N]). Returns list of action dicts.
    seg_model: optional callable segs [N,2] -> (verb probs, noun probs, action probs, pairs [A,2])."""
    vs, vsc, vl = verb_dets
    ns, nsc, nl = noun_dets
    if len(vs) == 0:
        return []
    o = np.argsort(-vsc)[: cfg.top_verbs]
    vs, vsc, vl = vs[o], vsc[o], vl[o]

    # (a) detection evidence: max over overlapping noun dets of score * tIoU, per noun class
    det_ev = np.zeros((len(vs), 300), np.float32)
    if len(ns):
        tm = _tiou_matrix(vs, ns)
        tm[tm < cfg.min_tiou] = 0
        contrib = tm * nsc[None, :]
        for c in np.unique(nl):
            m = nl == c
            det_ev[:, c] = contrib[:, m].max(1)
        det_ev /= np.maximum(det_ev.sum(1, keepdims=True), 1e-9)
    # (b) classifier evidence
    cls_ev = segment_mean(clip_noun, t_center, vs) if clip_noun is not None and len(clip_noun) else np.zeros_like(det_ev)
    cls_ev = cls_ev / np.maximum(cls_ev.sum(1, keepdims=True), 1e-9)

    if seg_model is not None:
        _, seg_n, seg_a, pairs = seg_model(vs)
        # action head -> noun distribution given the detected verb
        amap = np.full((97, 300), -1, np.int64)
        amap[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
        idx = amap[vl]
        act_ev = np.zeros_like(det_ev)
        ok = idx >= 0
        act_ev[ok] = seg_a[np.nonzero(ok)[0], idx[ok]]
        act_ev /= np.maximum(act_ev.sum(1, keepdims=True), 1e-9)
        w = cfg.w_det if len(ns) else 0.0
        rest = cfg.w_clip + cfg.w_seg + cfg.w_act
        ev = w * det_ev + (1 - w) * (cfg.w_clip * cls_ev + cfg.w_seg * seg_n + cfg.w_act * act_ev) / max(rest, 1e-9)
    else:
        w = cfg.w_det if len(ns) else 0.0
        ev = w * det_ev + (1 - w) * cls_ev
    if cfg.prior_pow > 0 and prior_n_given_v is not None:
        ev = ev * prior_n_given_v[vl] ** cfg.prior_pow
    ev = ev / np.maximum(ev.sum(1, keepdims=True), 1e-9)

    k = cfg.nouns_per_verb
    top = np.argsort(-ev, 1)[:, :k]
    acts = []
    for i in range(len(vs)):
        for n in top[i]:
            s = float(vsc[i] * ev[i, n] ** cfg.noun_pow)
            if s <= 0:
                continue
            acts.append((s, float(vs[i, 0]), float(vs[i, 1]), verbs[vl[i]], nouns[n]))
    acts.sort(key=lambda r: -r[0])
    acts = acts[: cfg.max_actions]
    return [{"segment": [s0, s1], "verb": v, "noun": n, "score": sc} for sc, s0, s1, v, n in acts]


def merge_duplicates(acts: list[dict], thr: float = 0.8) -> list[dict]:
    """Collapse predictions of the same (verb, noun) whose segments overlap at tIoU >= thr into one, keeping the
    highest-scoring segment and a noisy-or score 1 - prod(1 - s). Avoids duplicate false positives in pair AP."""
    from collections import defaultdict
    groups = defaultdict(list)
    for a in acts:
        groups[(a["verb"], a["noun"])].append(a)
    out = []
    for g in groups.values():
        keep = []
        for a in sorted(g, key=lambda a: -a["score"]):
            s, e = a["segment"]
            for b in keep:
                x, y = b["segment"]
                i = max(0.0, min(e, y) - max(s, x))
                if i / max((e - s) + (y - x) - i, 1e-6) >= thr:
                    b["score"] = 1 - (1 - b["score"]) * (1 - a["score"])
                    break
            else:
                keep.append(dict(a))
        out += keep
    return out
