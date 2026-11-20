"""DEV gate for LaViLa: (1) top-1 on val GT segments, (2) noun evidence for verb detections -> pair mAP,
on a subset of val videos. Saves per-video LaViLa probabilities for the top unique detection segments.

    C:/prism_env/Scripts/python.exe tools/check_lavila.py --n 20 --topk 200
"""
import argparse
import os
import time

import numpy as np

import epic_dev as E
from prism.lavila_cls import SIZE, LaViLaEPIC, sample_ids
from prism.video_io import iter_frames, probe

OUT = r"C:/prism_data/dev_outputs/lavila"


def video_path(v):
    return rf"C:/prism_data/epic/EPIC-KITCHENS/{v.split('_')[0]}/videos/{v}.MP4"


def nms_segments(segs, scores, thr=0.7, k=200):
    o, keep = np.argsort(-scores), []
    for i in o:
        s, e = segs[i]
        ok = True
        for j in keep:
            a, b = segs[j]
            inter = max(0.0, min(e, b) - max(s, a))
            if inter / ((e - s) + (b - a) - inter + 1e-9) > thr:
                ok = False
                break
        if ok:
            keep.append(i)
        if len(keep) >= k:
            break
    return segs[keep]


def gather_clips(video, segs_list, fps=30.0):
    """Decode once at 30 fps (short side 224, centre crop) keeping only the needed frames."""
    info = probe(video)
    n = int(info["duration"] * fps)
    ids = [sample_ids(s, e, fps, n) for s, e in segs_list]
    need = np.unique(np.concatenate(ids)) if ids else np.array([], int)
    W, H = info["width"], info["height"]
    if info.get("rotation") in (90, -90, 270, -270):
        W, H = H, W
    w = int(round(W * SIZE / H / 2)) * 2
    x0 = (w - SIZE) // 2
    keep, pos = {}, 0
    ni = 0
    for chunk in iter_frames(video, fps, (w, SIZE)):
        for f in chunk:
            while ni < len(need) and need[ni] < pos:
                ni += 1
            if ni < len(need) and need[ni] == pos:
                keep[pos] = f[:, x0:x0 + SIZE]
            pos += 1
    last = max(keep) if keep else 0
    blank = np.zeros((SIZE, SIZE, 3), np.uint8)
    return [np.stack([keep.get(int(i), keep.get(min(int(i), last), blank)) for i in idx]) for idx in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    D = E.load()
    vids = [v for i, v in enumerate(E.EPIC) if i % max(1, len(E.EPIC) // a.n) == 0][: a.n]
    m = LaViLaEPIC()
    vi = {k: i for i, k in enumerate(E.CLS["verbs"])}
    ni_ = {k: i for i, k in enumerate(E.CLS["nouns"])}
    acc = np.zeros(3)
    tot = 0
    for v in vids:
        p = os.path.join(OUT, v + ".npz")
        if os.path.exists(p):
            continue
        t0 = time.perf_counter()
        gt = E.GT[v]
        gsegs = np.array([[s, e] for s, e, _, _ in gt])
        vs, vsc, _ = D[v]["vd"]
        dsegs = nms_segments(vs, vsc, k=a.topk)
        clips = gather_clips(video_path(v), list(gsegs) + list(dsegs))
        t1 = time.perf_counter()
        PV, PN = [], []
        for i in range(0, len(clips), a.bs):
            pv, pn = m.predict(np.stack(clips[i:i + a.bs]))
            PV.append(pv)
            PN.append(pn)
        PV, PN = np.concatenate(PV), np.concatenate(PN)
        g = len(gsegs)
        gv = np.array([vi[x] for _, _, x, _ in gt])
        gn = np.array([ni_[x] for _, _, _, x in gt])
        cv, cn = PV[:g].argmax(1) == gv, PN[:g].argmax(1) == gn
        acc += [cv.sum(), cn.sum(), (cv & cn).sum()]
        tot += g
        np.savez_compressed(p, det_segs=dsegs, det_v=PV[g:].astype(np.float16), det_n=PN[g:].astype(np.float16),
                            gt_v=PV[:g].astype(np.float16), gt_n=PN[:g].astype(np.float16))
        print(f"{v} {D[v]['dur']:.0f}s gt {g} dets {len(dsegs)} decode {t1 - t0:.0f}s model {time.perf_counter() - t1:.0f}s "
              f"running top1 verb {100 * acc[0] / tot:.1f} noun {100 * acc[1] / tot:.1f} action {100 * acc[2] / tot:.1f}",
              flush=True)


if __name__ == "__main__":
    main()
