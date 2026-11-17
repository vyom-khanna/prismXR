"""Local scorer for PRISM Track 04 — approximates the four official metrics using public GT.

DEV ONLY. Uses EPIC-100 validation GT and FPVSum annotations; never imported by submission code.
The organizers' exact Coverage rule and over-budget penalty are not published, so several plausible
variants are reported.

    python tools/eval/score.py --pred outputs/                      # all metrics
    python tools/eval/score.py --pred outputs/ --per-video report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEF_MANIFEST = os.path.join(ROOT, "Student package_track04", "Student package_track04", "manifest.csv")
DEF_EPIC_GT = r"C:/prism_ext/epic-kitchens-100-annotations/EPIC_100_validation.csv"
DEF_VERBS = r"C:/prism_ext/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv"
DEF_NOUNS = r"C:/prism_ext/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv"
DEF_FPV_GT = r"C:/prism_ext/fpvsum/annotation/gt_data.mat"
TIOUS = (0.1, 0.2, 0.3, 0.4, 0.5)
MIN_SEG, BUDGET = 4.0, 0.15
WEIGHTS = {"action_map": 0.25, "coverage": 0.40, "highlight_f1": 0.25, "label_acc": 0.10}


# ----------------------------------------------------------------------------- loading
def _ts(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def load_manifest(p):
    with open(p, newline="") as f:
        return {r["video_id"]: (float(r["duration_sec"]), r["dataset"]) for r in csv.DictReader(f)}


def load_class_keys(p):
    with open(p, encoding="utf-8") as f:
        return {int(r["id"]): r["key"] for r in csv.DictReader(f)}


def load_epic_gt(p, verbs, nouns):
    gt = defaultdict(list)  # vid -> [(s, e, verb_key, noun_key)]
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            gt[r["video_id"]].append((_ts(r["start_timestamp"]), _ts(r["stop_timestamp"]),
                                      verbs[int(r["verb_class"])], nouns[int(r["noun_class"])]))
    return gt


def load_fpv_gt(p):
    import scipy.io as sio
    g = sio.loadmat(p, squeeze_me=True, struct_as_record=False)["gt_data"]
    return {e.Name: np.asarray(e.Binary_score).astype(bool) for e in g}


def fpv_key(vid, gt):
    if vid in gt:
        return vid
    alt = re.sub(r"^([A-Za-z]+)__", r"\1_", vid)  # 'Jumping__mU-…' is stored as 'Jumping_mU-…'
    return alt if alt in gt else None


def load_preds(d):
    out = {}
    for f in os.listdir(d):
        if f.endswith(".json"):
            try:
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    out[f[:-5]] = json.load(fh)
            except Exception:  # noqa: BLE001  invalid json scores 0
                pass
    return out


def valid_highlights(pred):
    segs = []
    for h in pred.get("highlights", []) or []:
        try:
            s, e = map(float, h["segment"])
        except Exception:  # noqa: BLE001
            continue
        if e - s >= MIN_SEG:
            segs.append((s, e))
    return segs


# ----------------------------------------------------------------------------- action mAP
def tiou(s, e, gs, ge):
    inter = np.clip(np.minimum(e, ge) - np.maximum(s, gs), 0, None)
    union = (e - s) + (ge - gs) - inter
    return inter / np.maximum(union, 1e-9)


def ap_from_pr(tp, n_gt):
    if n_gt == 0:
        return None
    if len(tp) == 0:
        return 0.0
    tp = np.asarray(tp, float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mprec = np.concatenate([[0], prec, [0]])
    mrec = np.concatenate([[0], rec, [1]])
    for i in range(len(mprec) - 2, -1, -1):  # precision envelope (ActivityNet style)
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx]))


def detection_map(gt, preds, key_fn):
    """gt: vid->[(s,e,v,n)], preds: vid->[(s,e,v,n,score)]. key_fn maps (v,n)->class key."""
    gt_by_c = defaultdict(lambda: defaultdict(list))
    for vid, rows in gt.items():
        for s, e, v, n in rows:
            gt_by_c[key_fn(v, n)][vid].append((s, e))
    pr_by_c = defaultdict(list)
    for vid, rows in preds.items():
        if vid in gt:
            for s, e, v, n, sc in rows:
                pr_by_c[key_fn(v, n)].append((sc, vid, s, e))
    res = {}
    for t in TIOUS:
        aps = []
        for c, by_vid in gt_by_c.items():
            n_gt = sum(len(x) for x in by_vid.values())
            arr = {v: np.array(x) for v, x in by_vid.items()}
            used = {v: np.zeros(len(x), bool) for v, x in by_vid.items()}
            tp = []
            for sc, vid, s, e in sorted(pr_by_c.get(c, []), key=lambda r: -r[0]):
                g = arr.get(vid)
                if g is None:
                    tp.append(0)
                    continue
                ious = tiou(s, e, g[:, 0], g[:, 1])
                ious[used[vid]] = -1
                j = int(np.argmax(ious))
                if ious[j] >= t:
                    used[vid][j] = True
                    tp.append(1)
                else:
                    tp.append(0)
            aps.append(ap_from_pr(tp, n_gt))
        res[t] = float(np.mean(aps)) if aps else 0.0
    res["avg"] = float(np.mean([res[t] for t in TIOUS]))
    return res


# ----------------------------------------------------------------------------- coverage
def coverage_video(gt_rows, segs, dur):
    pairs = {(v, n) for _, _, v, n in gt_rows}
    if not pairs:
        return None
    hit = {"mid": set(), "ov50": set(), "any": set()}
    for s, e, v, n in gt_rows:
        mid, L = (s + e) / 2, max(e - s, 1e-6)
        for hs, he in segs:
            ov = max(0.0, min(e, he) - max(s, hs))
            if hs <= mid <= he:
                hit["mid"].add((v, n))
            if ov / L >= 0.5:
                hit["ov50"].add((v, n))
            if ov > 0:
                hit["any"].add((v, n))
    total = sum(he - hs for hs, he in segs)
    ratio = total / dur if dur > 0 else 0
    penalty = min(1.0, BUDGET / ratio) if ratio > BUDGET else 1.0  # assumed soft penalty
    return {k: penalty * len(h) / len(pairs) for k, h in hit.items()} | {"ratio": ratio, "n_pairs": len(pairs)}


# ----------------------------------------------------------------------------- highlight F1
def f1_video(gt_bin, segs, dur):
    n = len(gt_bin)
    t = (np.arange(n) + 0.5) * dur / n
    pred = np.zeros(n, bool)
    for s, e in segs:
        pred |= (t >= s) & (t < e)
    tp = np.sum(pred & gt_bin)
    if tp == 0:
        return 0.0
    p, r = tp / pred.sum(), tp / gt_bin.sum()
    return float(2 * p * r / (p + r))


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="directory of <video_id>.json")
    ap.add_argument("--manifest", default=DEF_MANIFEST)
    ap.add_argument("--epic-gt", default=DEF_EPIC_GT)
    ap.add_argument("--fpv-gt", default=DEF_FPV_GT)
    ap.add_argument("--only", help="comma list of video ids to restrict to (e.g. downloaded subset)")
    ap.add_argument("--per-video", help="write per-video CSV here")
    ap.add_argument("--json-out", help="write summary JSON here")
    a = ap.parse_args(argv)

    man = load_manifest(a.manifest)
    preds = load_preds(a.pred)
    if a.only:
        keep = set(a.only.split(","))
        man = {k: v for k, v in man.items() if k in keep}
    verbs, nouns = load_class_keys(DEF_VERBS), load_class_keys(DEF_NOUNS)
    egt_all = load_epic_gt(a.epic_gt, verbs, nouns)
    epic_ids = [v for v, (_, ds) in man.items() if ds == "epic_kitchens_100"]
    fpv_ids = [v for v, (_, ds) in man.items() if ds == "fpvsum"]
    egt = {v: egt_all[v] for v in epic_ids}
    fgt = load_fpv_gt(a.fpv_gt)
    rows = []

    # Action mAP (pair, plus verb / noun diagnostics)
    pa = {}
    for v in epic_ids:
        acts = []
        for x in (preds.get(v) or {}).get("actions", []) or []:
            try:
                acts.append((float(x["segment"][0]), float(x["segment"][1]), x["verb"], x["noun"], float(x["score"])))
            except Exception:  # noqa: BLE001
                pass
        pa[v] = acts
    m_pair = detection_map(egt, pa, lambda v, n: (v, n))
    m_verb = detection_map(egt, pa, lambda v, n: v)
    m_noun = detection_map(egt, pa, lambda v, n: n)

    # Coverage
    cov = defaultdict(list)
    for v in epic_ids:
        dur = man[v][0]
        c = coverage_video(egt[v], valid_highlights(preds.get(v, {})), dur)
        if c is None:
            continue
        for k in ("mid", "ov50", "any", "ratio"):
            cov[k].append(c[k])
        rows.append({"video_id": v, "dataset": "epic", "cov_mid": c["mid"], "cov_ov50": c["ov50"],
                     "hl_ratio": c["ratio"], "gt_pairs": c["n_pairs"], "n_actions": len(pa[v])})

    # Highlight F1 + label accuracy
    f1s, acc = [], []
    for v in fpv_ids:
        p = preds.get(v, {})
        k = fpv_key(v, fgt)
        f1 = f1_video(fgt[k], valid_highlights(p), man[v][0]) if k else None
        true = v.split("_")[0].lower()
        name = ((p.get("label") or {}).get("name") or "") if isinstance(p, dict) else ""
        ok = float(name == true)
        if f1 is not None:
            f1s.append(f1)
        acc.append(ok)
        rows.append({"video_id": v, "dataset": "fpvsum", "f1": f1, "label_true": true, "label_pred": name})

    s = {
        "action_map": m_pair["avg"],
        "coverage": float(np.mean(cov["mid"])) if cov["mid"] else 0.0,
        "highlight_f1": float(np.mean(f1s)) if f1s else 0.0,
        "label_acc": float(np.mean(acc)) if acc else 0.0,
    }
    s["weighted_total"] = sum(WEIGHTS[k] * s[k] for k in WEIGHTS)
    detail = {
        "action_map_by_tiou": {str(k): v for k, v in m_pair.items()},
        "verb_map_avg": m_verb["avg"], "noun_map_avg": m_noun["avg"],
        "coverage_variants": {k: float(np.mean(cov[k])) if cov[k] else 0.0 for k in ("mid", "ov50", "any")},
        "epic_mean_highlight_ratio": float(np.mean(cov["ratio"])) if cov["ratio"] else 0.0,
        "n_epic": len(epic_ids), "n_fpv": len(fpv_ids), "n_pred_files": len(preds),
    }
    print("=== PRISM Track04 local score (approximate) ===")
    for k in ("action_map", "coverage", "highlight_f1", "label_acc"):
        print(f"  {k:<14} {100 * s[k]:6.2f}   (weight {WEIGHTS[k]:.2f})")
    print(f"  {'WEIGHTED':<14} {100 * s['weighted_total']:6.2f}")
    print("  --")
    print("  mAP by tIoU:  " + "  ".join(f"{t}:{100 * m_pair[t]:.2f}" for t in TIOUS))
    print(f"  verb mAP {100 * detail['verb_map_avg']:.2f}   noun mAP {100 * detail['noun_map_avg']:.2f}")
    print("  coverage variants: " + "  ".join(f"{k}:{100 * v:.2f}" for k, v in detail["coverage_variants"].items()))
    print(f"  EPIC mean highlight ratio: {100 * detail['epic_mean_highlight_ratio']:.2f}%")
    print(f"  videos: epic {detail['n_epic']}  fpvsum {detail['n_fpv']}  prediction files {detail['n_pred_files']}")

    if a.per_video:
        keys = sorted({k for r in rows for k in r})
        with open(a.per_video, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"summary": s, "detail": detail}, f, indent=1)
    return s


if __name__ == "__main__":
    main()
