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
    w_det: float = 0.5        # weight of noun-detection evidence vs clip-classifier evidence
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
                 cfg: PairCfg = PairCfg()):
    """verb_dets/noun_dets: (segs [N,2], scores [N], labels [N]). Returns list of action dicts."""
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
            s = float(vsc[i] * ev[i, n])
            if s <= 0:
                continue
            acts.append((s, float(vs[i, 0]), float(vs[i, 1]), verbs[vl[i]], nouns[n]))
    acts.sort(key=lambda r: -r[0])
    acts = acts[: cfg.max_actions]
    return [{"segment": [s0, s1], "verb": v, "noun": n, "score": sc} for sc, s0, s1, v, n in acts]
