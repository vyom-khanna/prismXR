"""Tune PairCfg / CovCfg on cached detections (DEV). 2-fold CV split by participant: pick the best
config on one fold, report it on the other, so reported numbers are not fitted to their own videos.

    py -3.13 tools/tune_epic.py      (needs released_dets.npz from tools/check_detector.py)
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "submission", "code"))
sys.path.insert(0, os.path.join(ROOT, "tools", "eval"))
import score as S  # noqa: E402
from prism import highlights as H, select_coverage as SC  # noqa: E402
from prism.pairing import PairCfg, pair_actions  # noqa: E402

FEAT = r"C:/prism_data/af_features/epic_kitchens/features"
man = S.load_manifest(S.DEF_MANIFEST)
verbs, nouns = S.load_class_keys(S.DEF_VERBS), S.load_class_keys(S.DEF_NOUNS)
GT = S.load_epic_gt(S.DEF_EPIC_GT, verbs, nouns)
cls = json.load(open(os.path.join(ROOT, "submission", "code", "prism", "supported_classes.json")))
prior = np.load(r"C:/prism_data/weights/epic_prior.npz")["p_n_given_v"]
z = np.load(r"C:/prism_data/dev_outputs/released_dets.npz")
vids = sorted({k.split("|")[0] for k in z.files})
D = {}
for v in vids:
    npb = np.load(os.path.join(FEAT, v + ".npz"))["n_prob"]
    D[v] = dict(vd=tuple(z[f"{v}|verb|{j}"] for j in range(3)), nd=tuple(z[f"{v}|noun|{j}"] for j in range(3)),
                npb=npb, tc=(np.arange(len(npb)) * 16 + 16) / 30.0, dur=man[v][0])

parts = sorted({v.split("_")[0] for v in vids})
foldA = {v for v in vids if parts.index(v.split("_")[0]) % 2 == 0}
folds = [sorted(foldA), sorted(set(vids) - foldA)]


def actions(v, pc):
    d = D[v]
    acts = pair_actions(d["vd"], d["nd"], d["npb"], d["tc"], prior, cls["verbs"], cls["nouns"], pc)
    return [x for x in acts if min(d["dur"], x["segment"][1]) > max(0, x["segment"][0])]


def eval_pair(vs, pc):
    pr = {v: [(a["segment"][0], a["segment"][1], a["verb"], a["noun"], a["score"]) for a in actions(v, pc)] for v in vs}
    return S.detection_map({v: GT[v] for v in vs}, pr, lambda a, b: (a, b))["avg"]


def eval_cov(vs, pc, cc):
    out = []
    for v in vs:
        hl = H.finalize(SC.select(actions(v, pc), D[v]["dur"], cc) or H.uniform_segments(D[v]["dur"]), D[v]["dur"])
        segs = [(h["segment"][0], h["segment"][1]) for h in hl if h["segment"][1] - h["segment"][0] >= 4]
        out.append(S.coverage_video(GT[v], segs, D[v]["dur"])["mid"])
    return float(np.mean(out))


pair_grid = [PairCfg(nouns_per_verb=k, w_det=w, prior_pow=p)
             for k, w, p in itertools.product((1, 2, 3, 5), (0.0, 0.3, 0.5, 0.7, 1.0), (0.0, 0.5, 1.0))]
print("== pairing (pair mAP) ==")
res = []
for pc in pair_grid:
    res.append((pc, [eval_pair(f, pc) for f in folds]))
for i in range(2):
    best = max(res, key=lambda r: r[1][i])
    print(f"tuned on fold{i}: k={best[0].nouns_per_verb} w_det={best[0].w_det} prior={best[0].prior_pow} "
          f"-> tune {100 * best[1][i]:.2f}  held-out {100 * best[1][1 - i]:.2f}")
best_pc = max(res, key=lambda r: np.mean(r[1]))[0]
base = [r for r in res if r[0] == PairCfg()][0]
print(f"default cfg: {[round(100 * x, 2) for x in base[1]]}  best overall: {best_pc}")

print("== coverage ==")
cov_grid = [SC.CovCfg(lengths=L, calib_pow=c) for L, c in itertools.product(
    ((4.0,), (4.0, 5.0), (4.0, 5.0, 6.0, 8.0)), (0.5, 1.0, 2.0, 3.0))]
cres = [(cc, [eval_cov(f, best_pc, cc) for f in folds]) for cc in cov_grid]
for i in range(2):
    best = max(cres, key=lambda r: r[1][i])
    print(f"tuned on fold{i}: lengths={best[0].lengths} calib={best[0].calib_pow} "
          f"-> tune {100 * best[1][i]:.2f}  held-out {100 * best[1][1 - i]:.2f}")
for cc, r in cres:
    print(f"  lengths={cc.lengths} calib={cc.calib_pow}: {[round(100 * x, 2) for x in r]}")
