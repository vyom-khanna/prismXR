"""DEV harness for EPIC experiments on OUR OWN SlowFast features (the ones run.sh produces).

    C:/prism_env/Scripts/python.exe tools/epic_dev.py build     # run ActionFormer on cached features once
    then `import epic_dev as E` from other tools: E.load(), E.FOLDS, E.pair_map(), E.coverage()

Never part of the submission. Uses EPIC val GT only for scoring.
"""
import csv
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "TIET_Opus5.5high", "code"))
sys.path.insert(0, os.path.join(ROOT, "tools", "eval"))
os.environ.setdefault("PRISM_WEIGHTS", r"C:/prism_data/weights")
import score as S  # noqa: E402
from prism import highlights as H, select_coverage as SC  # noqa: E402
from prism.pairing import PairCfg, pair_actions  # noqa: E402

CACHE = r"C:/prism_data/cache"
DETS = r"C:/prism_data/dev_outputs/own_dets.npz"
MAN = S.load_manifest(S.DEF_MANIFEST)
EPIC = sorted(v for v, (d, ds) in MAN.items() if ds == "epic_kitchens_100")
VERBS, NOUNS = S.load_class_keys(S.DEF_VERBS), S.load_class_keys(S.DEF_NOUNS)
GT = S.load_epic_gt(S.DEF_EPIC_GT, VERBS, NOUNS)
CLS = json.load(open(os.path.join(ROOT, "TIET_Opus5.5high", "code", "prism", "supported_classes.json")))
PRIOR = np.load(r"C:/prism_data/weights/epic_prior.npz")["p_n_given_v"]
_parts = sorted({v.split("_")[0] for v in EPIC})
_A = {v for v in EPIC if _parts.index(v.split("_")[0]) % 2 == 0}
FOLDS = [sorted(_A), sorted(set(EPIC) - _A)]


def feat_file(v):
    return glob.glob(os.path.join(CACHE, f"{v}.MP4_*_sf.npz"))[0]


def build():
    from prism.epic_detector import ActionFormerEPIC
    out = {}
    for kind in ("verb", "noun"):
        m = ActionFormerEPIC(kind)
        for i, v in enumerate(EPIC):
            with np.load(feat_file(v)) as z:
                d = m.detect(z["feats"], MAN[v][0])
            for j, x in enumerate(d):
                out[f"{v}|{kind}|{j}"] = x
            if i % 30 == 0:
                print(kind, i, v, len(d[0]), flush=True)
        del m
    np.savez_compressed(DETS, **out)
    print("saved", DETS)


_D = None


def load():
    """vid -> dict(vd, nd, vprob, nprob, tc, dur)."""
    global _D
    if _D is None:
        z = np.load(DETS)
        _D = {}
        for v in EPIC:
            with np.load(feat_file(v)) as f:
                _D[v] = dict(vd=tuple(z[f"{v}|verb|{j}"] for j in range(3)),
                             nd=tuple(z[f"{v}|noun|{j}"] for j in range(3)),
                             vprob=f["verb"], nprob=f["noun"], tc=f["t_center"], dur=MAN[v][0])
    return _D


def clean(acts, dur):
    out = []
    for a in acts:
        s, e = max(0.0, a["segment"][0]), min(dur, a["segment"][1])
        if e > s:
            out.append({**a, "segment": [s, e]})
    return out


def baseline_actions(v, pc=None):
    d = load()[v]
    return clean(pair_actions(d["vd"], d["nd"], d["nprob"], d["tc"], PRIOR, CLS["verbs"], CLS["nouns"],
                              pc or PairCfg()), d["dur"])


def pair_map(acts_by_vid, vids=None):
    vids = vids or list(acts_by_vid)
    pr = {v: [(a["segment"][0], a["segment"][1], a["verb"], a["noun"], a["score"]) for a in acts_by_vid[v]]
          for v in vids}
    return 100 * S.detection_map({v: GT[v] for v in vids}, pr, lambda a, b: (a, b))["avg"]


def coverage(segs_by_vid, vids=None, variant="mid"):
    vids = vids or list(segs_by_vid)
    r = []
    for v in vids:
        hl = H.finalize(segs_by_vid[v] or H.uniform_segments(MAN[v][0]), MAN[v][0])
        segs = [(h["segment"][0], h["segment"][1]) for h in hl if h["segment"][1] - h["segment"][0] >= 4]
        r.append(S.coverage_video(GT[v], segs, MAN[v][0])[variant])
    return 100 * float(np.mean(r))


def select_all(acts_by_vid, cc=None):
    return {v: SC.select(a, MAN[v][0], cc or SC.CovCfg()) for v, a in acts_by_vid.items()}


def report(name, acts_by_vid, cc=None, segs=None):
    segs = segs if segs is not None else select_all(acts_by_vid, cc)
    m = [pair_map(acts_by_vid, f) for f in FOLDS]
    c = [coverage(segs, f) for f in FOLDS]
    print(f"{name:40s} mAP {np.mean(m):6.2f} [{m[0]:.2f} {m[1]:.2f}]  cov {np.mean(c):6.2f} [{c[0]:.2f} {c[1]:.2f}]",
          flush=True)
    return m, c


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()
        report("baseline (current code)", {v: baseline_actions(v) for v in EPIC})
