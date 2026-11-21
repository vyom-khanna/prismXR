"""DEV harness for v3 (LaViLa in): runs prism.pipeline.epic_actions on cached SlowFast + LaViLa outputs for all 138
EPIC val videos, with ActionFormer detections memoised on disk. Config overrides via `run(cfg_overrides)`.

    import dev_v3 as V; V.run("name", C={"LAV_W": 0.7}, pair={"noun_pow": 0.75}, cov={"dens_w": 1.2})
"""
import dataclasses
import os
import pickle

import numpy as np

import epic_dev as E
from prism import config as PC, epic_detector as ED, pairing as PR, pipeline as PL, select_coverage as SC

DETS = r"C:/prism_data/dev_outputs/af_dets_v3.pkl"
_real = ED.ActionFormerEPIC
_det = pickle.load(open(DETS, "rb")) if os.path.exists(DETS) else {}
_cur = [None]


class _MemoAF:
    def __init__(self, kind):
        self.kind = kind

    def detect(self, feats, dur):
        k = (_cur[0], self.kind)
        if k not in _det:
            m = _real(self.kind)
            _det[k] = m.detect(feats, dur)
            del m
        return _det[k]


ED.ActionFormerEPIC = _MemoAF


def lav_of(v):
    p = E.feat_file(v).replace("_sf.npz", "_lav.npz")
    with np.load(p) as z:
        return z["segs"], z["v"].astype(np.float32), z["n"].astype(np.float32)


FEATS, LAV = {}, {}


def actions(C=None, pair=None):
    C, pair = C or {}, pair or {}
    old = {k: getattr(PC, k) for k in C}
    for k, val in C.items():
        setattr(PC, k, val)
    oldp = PR.PairCfg
    if pair:
        PR.PairCfg = lambda: dataclasses.replace(oldp(), **pair)
    out = {}
    try:
        for v in E.EPIC:
            if v not in FEATS:
                with np.load(E.feat_file(v)) as z:
                    FEATS[v] = {k: z[k] for k in z.files}
                LAV[v] = lav_of(v)
            _cur[0] = v
            out[v] = PL.epic_actions(FEATS[v], E.MAN[v][0], {}, lav=LAV[v])
    finally:
        for k, val in old.items():
            setattr(PC, k, val)
        PR.PairCfg = oldp
    if not os.path.exists(DETS):
        pickle.dump(_det, open(DETS, "wb"))
    return out


def score(name, res, cov=None):
    cc = SC.CovCfg(**(cov or {}))
    acts = {v: a for v, (a, d) in res.items()}
    segs = {v: SC.select(a, E.MAN[v][0], cc, density=d) for v, (a, d) in res.items()}
    f = [(E.pair_map(acts, F), E.coverage(segs, F)) for F in E.FOLDS]
    m, c = E.pair_map(acts), E.coverage(segs)
    print(f"{name:40s} mAP {m:6.2f} [{f[0][0]:6.2f} {f[1][0]:6.2f}]  cov {c:6.2f} [{f[0][1]:6.2f} {f[1][1]:6.2f}]"
          f"  w {0.25 * m + 0.4 * c:6.2f}", flush=True)
    return m, c, f


def merge(acts, thr=0.7, mode="or"):
    """Collapse same-(verb, noun) predictions overlapping at tIoU >= thr into one (noisy-or / max / sum score)."""
    from collections import defaultdict
    g = defaultdict(list)
    for a in acts:
        g[(a["verb"], a["noun"])].append(a)
    out = []
    for L in g.values():
        keep = []
        for a in sorted(L, key=lambda a: -a["score"]):
            s, e = a["segment"]
            hit = None
            for b in keep:
                x, y = b["segment"]
                i = max(0.0, min(e, y) - max(s, x))
                if i / max((e - s) + (y - x) - i, 1e-6) >= thr:
                    hit = b
                    break
            if hit is None:
                keep.append(dict(a))
            elif mode == "or":
                hit["score"] = 1 - (1 - hit["score"]) * (1 - a["score"])
            elif mode == "sum":
                hit["score"] += a["score"]
        out += keep
    return sorted(out, key=lambda a: -a["score"])
