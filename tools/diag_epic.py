"""DEV diagnosis: which part of the EPIC pipeline costs the most? (uses val GT -> analysis only)

    C:/prism_env/Scripts/python.exe tools/diag_epic.py
"""
import numpy as np

import epic_dev as E

D = E.load()
base = {v: E.baseline_actions(v) for v in E.EPIC}


def best_gt(v, s, e):
    g = np.array([(a, b) for a, b, _, _ in E.GT[v]])
    inter = np.clip(np.minimum(e, g[:, 1]) - np.maximum(s, g[:, 0]), 0, None)
    iou = inter / ((e - s) + (g[:, 1] - g[:, 0]) - inter)
    j = int(np.argmax(iou))
    return iou[j], E.GT[v][j]


def relabel(acts_by_vid, fix_verb, fix_noun, min_iou=0.1):
    out = {}
    for v, acts in acts_by_vid.items():
        r = []
        for a in acts:
            iou, (_, _, gv, gn) = best_gt(v, *a["segment"])
            if iou < min_iou:
                r.append(a)
                continue
            r.append({**a, "verb": gv if fix_verb else a["verb"], "noun": gn if fix_noun else a["noun"]})
        out[v] = r
    return out


E.report("baseline", base)
# --- localisation vs recognition (mAP + coverage)
E.report("oracle noun on our segments", relabel(base, False, True))
E.report("oracle verb on our segments", relabel(base, True, False))
E.report("oracle verb+noun on our segments", relabel(base, True, True))
# --- coverage ceilings
gt_acts = {v: [{"segment": [s, e], "verb": a, "noun": b, "score": 1.0} for s, e, a, b in E.GT[v]] for v in E.EPIC}
E.report("GT segments + GT labels (ceiling)", gt_acts)
# GT segments, labels predicted from clip probabilities (verb/noun argmax of segment mean)
from prism.pairing import segment_mean  # noqa: E402

gt_pred = {}
for v in E.EPIC:
    d = D[v]
    segs = np.array([[s, e] for s, e, _, _ in E.GT[v]])
    vp, npb = segment_mean(d["vprob"], d["tc"], segs), segment_mean(d["nprob"], d["tc"], segs)
    gt_pred[v] = [{"segment": list(segs[i]), "verb": E.CLS["verbs"][int(vp[i].argmax())],
                   "noun": E.CLS["nouns"][int(npb[i].argmax())], "score": float(vp[i].max() * npb[i].max())}
                  for i in range(len(segs))]
E.report("GT segments + SlowFast clip labels", gt_pred)
# label-free density: every predicted action is its own pair -> windows go where detections are dense
uniq = {v: [{**a, "verb": f"x{i}", "noun": "y"} for i, a in enumerate(acts)] for v, acts in base.items()}
E.report("label-free (each det unique)", uniq)
# proposal recall of the top-2000 verb dets at tIoU 0.5 / 0.3
for t in (0.3, 0.5):
    rec = []
    for v in E.EPIC:
        g = np.array([(a, b) for a, b, _, _ in E.GT[v]])
        ps = D[v]["vd"][0]
        inter = np.clip(np.minimum(ps[:, None, 1], g[None, :, 1]) - np.maximum(ps[:, None, 0], g[None, :, 0]), 0, None)
        iou = inter / ((ps[:, 1] - ps[:, 0])[:, None] + (g[:, 1] - g[:, 0])[None] - inter)
        rec.append((iou.max(0) >= t).mean())
    print(f"verb-det proposal recall @tIoU {t}: {100 * np.mean(rec):.1f}%")
# how many distinct GT pairs appear among our predicted pairs at all (pair recall, label only)
pr = [len({(a["verb"], a["noun"]) for a in base[v]} & {(b, c) for _, _, b, c in E.GT[v]}) /
      len({(b, c) for _, _, b, c in E.GT[v]}) for v in E.EPIC]
print(f"distinct GT pairs present anywhere in our 2000 actions: {100 * np.mean(pr):.1f}%")
