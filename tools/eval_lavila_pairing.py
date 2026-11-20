"""DEV: does LaViLa evidence improve pair mAP / coverage? Uses the gate outputs of tools/check_lavila.py
(LaViLa verb/noun probs for the top NMS'd detection segments of a subset of val videos).

    C:/prism_env/Scripts/python.exe tools/eval_lavila_pairing.py
"""
import glob
import os

import numpy as np

import epic_dev as E
from prism import pairing as PR, select_coverage as SC
from prism.segcls import SegCls

LAV = r"C:/prism_data/dev_outputs/lavila"
D = E.load()
M = SegCls()
PAIRS = np.asarray(M.pairs)
vids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(LAV, "*.npz")))
print(f"{len(vids)} videos with LaViLa outputs")
L = {v: dict(np.load(os.path.join(LAV, v + ".npz"))) for v in vids}
dens = dict(np.load(r"C:/prism_data/dev_outputs/own_density_single.npz"))


def lav_lookup(v, segs, thr):
    """LaViLa probs for each seg via best-tIoU NMS'd segment (>= thr), else None rows (mask False)."""
    ls = L[v]["det_segs"]
    tm = PR._tiou_matrix(segs, ls)
    j = tm.argmax(1)
    ok = tm[np.arange(len(segs)), j] >= thr
    return L[v]["det_v"][j].astype(np.float32), L[v]["det_n"][j].astype(np.float32), ok


def actions(v, wl=0.0, cv=0.0, thr=0.5, alt=0, lam=0.5):
    d = D[v]
    f = np.load(E.feat_file(v))

    def sm(segs):
        pv, pn, pa = M.predict(f["feats"], f["verb"], f["noun"], d["tc"], segs)
        if wl > 0:
            lv, ln, ok = lav_lookup(v, segs, thr)
            pn = np.where(ok[:, None], (1 - wl) * pn + wl * ln, pn)
        return pv, pn, pa, PAIRS

    acts = PR.pair_actions(d["vd"], d["nd"], d["nprob"], d["tc"], E.PRIOR, E.CLS["verbs"], E.CLS["nouns"],
                           PR.PairCfg(), seg_model=sm)
    if cv > 0 or alt:
        segs = np.array([a["segment"] for a in acts])
        lv, ln, ok = lav_lookup(v, segs, thr)
        vid = {k: i for i, k in enumerate(E.CLS["verbs"])}
        if cv > 0:
            for i, a in enumerate(acts):
                if ok[i]:
                    a["score"] *= float(lv[i, vid[a["verb"]]] + 1e-3) ** cv
        if alt:  # LaViLa top verbs x LaViLa top nouns on the NMS'd segments
            for k, (s, e) in enumerate(L[v]["det_segs"]):
                pv, pn = L[v]["det_v"][k].astype(np.float32), L[v]["det_n"][k].astype(np.float32)
                for a_ in np.argsort(-pv)[:alt]:
                    for b_ in np.argsort(-pn)[:alt]:
                        acts.append({"segment": [float(s), float(e)], "verb": E.CLS["verbs"][a_],
                                     "noun": E.CLS["nouns"][b_], "score": lam * float(pv[a_] * pn[b_]) ** 0.75})
        acts = sorted(acts, key=lambda a: -a["score"])[:2000]
    return E.clean(acts, d["dur"])


def run(name, **k):
    a = {v: actions(v, **k) for v in vids}
    segs = {v: SC.select(a[v], D[v]["dur"], SC.CovCfg(), density=(dens[v], D[v]["tc"])) for v in vids}
    print(f"{name:36s} mAP(subset) {E.pair_map(a, vids):6.2f}   cov {E.coverage(segs, vids):6.2f}", flush=True)


run("current (segcls)")
for wl in (0.3, 0.5, 0.7, 1.0):
    run(f"LaViLa noun mix wl={wl}", wl=wl)
for cv in (0.25, 0.5):
    run(f"wl .5 + LaViLa verb rescore c={cv}", wl=0.5, cv=cv)
run("wl .5 + alt LaViLa pairs 3x3", wl=0.5, alt=3)
