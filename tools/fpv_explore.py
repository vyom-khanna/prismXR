"""Evaluate candidate importance signals on cached FPVSum features (DEV analysis only).

For each signal: mean per-video Spearman correlation with human importance and the frame-F1 we get
after window selection. Combination weights are chosen with leave-one-category-out CV so the
reported number is not fitted to the category it is tested on.

    C:/prism_env/Scripts/python.exe tools/fpv_explore.py
"""
import glob
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.stats import spearmanr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "submission", "code"))
sys.path.insert(0, os.path.join(ROOT, "tools", "eval"))
os.environ.setdefault("PRISM_WEIGHTS", r"C:/prism_data/weights")
import score as S  # noqa: E402
from prism import highlights as H  # noqa: E402

GT = {e.Name: np.asarray(e.Score, float) for e in sio.loadmat(S.DEF_FPV_GT, squeeze_me=True, struct_as_record=False)["gt_data"]}
GTB = S.load_fpv_gt(S.DEF_FPV_GT)

GENERIC_POS = ["an exciting moment in an action sports video", "a spectacular jump in the air", "a dangerous stunt",
               "a crash or fall", "a breathtaking scenic view", "a close encounter with another person",
               "the most interesting moment of the video"]
GENERIC_NEG = ["a boring moment with nothing happening", "a static view", "a blurry dark frame", "waiting around"]

def z(x):
    return (x - x.mean()) / (x.std() + 1e-6)

def smooth(x, k):
    if k <= 1 or len(x) < k:
        return x
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")

def load():
    from prism.clip_model import Clip
    c = Clip()
    tp, tn = c.encode_text(GENERIC_POS), c.encode_text(GENERIC_NEG)
    data = []
    for f in sorted(glob.glob(r"C:/prism_data/cache/fpv/*.npz")):
        vid = os.path.basename(f)[:-4]
        k = S.fpv_key(vid, GT)
        if not k:
            continue
        d = np.load(f)
        emb, grey, ts, dur = d["emb"].astype(np.float32), d["grey"].astype(np.float32), d["ts"], float(d["dur"])
        n = len(ts)
        mu = emb.mean(0); mu /= np.linalg.norm(mu)
        sig = {}
        sig["novelty"] = 1 - emb @ mu
        sig["change"] = np.concatenate([[0], 1 - np.sum(emb[1:] * emb[:-1], 1)])
        mot = np.concatenate([[0], np.abs(np.diff(grey, axis=0)).mean((1, 2))])
        sig["motion"] = mot
        sig["calm"] = -mot
        sig["prompt"] = (emb @ tp.T).max(1) - (emb @ tn.T).max(1)
        sig["bright"] = grey.mean((1, 2))
        sig["contrast"] = grey.std((1, 2))
        pos = (ts / dur)
        sig["center"] = -np.abs(pos - 0.5)
        # local uniqueness: distance to neighbours within +/-10 s
        W = 20
        sim = emb @ emb.T
        loc = np.array([np.mean(sim[i, max(0, i - W):i + W + 1]) for i in range(n)])
        sig["local_unique"] = -loc
        sig["global_rep"] = sim.mean(1)  # representativeness
        g = GT[k]
        gi = np.interp(ts, (np.arange(len(g)) + 0.5) * dur / len(g), g)
        data.append(dict(vid=vid, cat=vid.split("_")[0].lower(), sig={a: smooth(z(b), 3) for a, b in sig.items()},
                         ts=ts, dur=dur, gt=gi, gtb=GTB[k]))
    return data

def f1_of(d, s, seg_len=4.0):
    segs = H.peaks_to_segments(s, d["ts"], d["dur"], seg_len=seg_len)
    hl = [(x["segment"][0], x["segment"][1]) for x in H.finalize(segs, d["dur"])]
    return S.f1_video(d["gtb"], [(a, b) for a, b in hl if b - a >= 4], d["dur"])

def main():
    data = load()
    names = list(data[0]["sig"])
    print(f"{len(data)} videos")
    print(f"{'signal':<14}{'spearman':>9}{'F1':>8}")
    for nm in names:
        r = np.nanmean([spearmanr(d["sig"][nm], d["gt"])[0] for d in data])
        f = np.mean([f1_of(d, d["sig"][nm]) for d in data])
        print(f"{nm:<14}{r:9.3f}{f:8.3f}")
    rnd = np.mean([f1_of(d, np.random.RandomState(0).rand(len(d["ts"]))) for d in data])
    print(f"{'random':<14}{'':>9}{rnd:8.3f}")

    # leave-one-category-out: linear weights fitted by least squares on other categories
    cats = sorted({d["cat"] for d in data})
    f1s = []
    for c in cats:
        tr = [d for d in data if d["cat"] != c]
        X = np.concatenate([np.stack([d["sig"][n] for n in names], 1) for d in tr])
        y = np.concatenate([z(d["gt"]) for d in tr])
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        for d in data:
            if d["cat"] == c:
                s = np.stack([d["sig"][n] for n in names], 1) @ w
                f1s.append(f1_of(d, s))
    print(f"LOCO linear combo F1: {np.mean(f1s):.3f}")
    X = np.concatenate([np.stack([d["sig"][n] for n in names], 1) for d in data])
    y = np.concatenate([z(d["gt"]) for d in data])
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    print("weights (all data, for inspection):", {n: round(float(v), 3) for n, v in zip(names, w)})

if __name__ == "__main__":
    main()
