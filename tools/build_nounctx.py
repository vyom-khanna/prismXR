"""Fit noun co-occurrence matrices M_sigma and per-verb log-normal duration priors on EPIC-100 TRAIN annotations only."""
import csv, json, sys
from collections import defaultdict
import numpy as np

TRAIN = r"C:/prism_ext/epic-kitchens-100-annotations/EPIC_100_train.csv"
CLS = json.load(open(r"C:/Users/vyomk/OneDrive/Desktop/prismXR/submission/code/prism/supported_classes.json"))


def ts(s):
    h, m, x = s.split(":")
    return 3600 * int(h) + 60 * int(m) + float(x)


vids = defaultdict(list)
with open(TRAIN, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        s, e = ts(r["start_timestamp"]), ts(r["stop_timestamp"])
        vids[r["video_id"]].append(((s + e) / 2, e - s, int(r["verb_class"]), int(r["noun_class"]), r["verb"], r["noun"]))
# sanity: class index -> name matches supported_classes ordering
bad = sum(CLS["nouns"][a[3]] != a[5] for L in vids.values() for a in L[:5])
badv = sum(CLS["verbs"][a[2]] != a[4] for L in vids.values() for a in L[:5])
print("videos", len(vids), "name mismatches noun/verb", bad, badv)

out = {}
for sig in (10, 30, 120, 0):  # 0 = whole video
    M = np.zeros((300, 300))
    for L in vids.values():
        t = np.array([a[0] for a in L]); n = np.array([a[3] for a in L])
        D = np.abs(t[:, None] - t[None, :])
        W = np.ones_like(D) if sig == 0 else np.exp(-D / sig)
        np.fill_diagonal(W, 0)
        oh = np.eye(300)[n]
        M += oh.T @ W @ oh
    M /= np.maximum(M.sum(1, keepdims=True), 1e-9)
    out[f"M{sig}"] = M.astype(np.float32)
    print("sigma", sig, "mean self-persistence", float(np.mean([M[i, i] for i in range(300) if M[i].sum() > 0])))
# per-verb log-normal duration
mu, sd = np.zeros(97), np.zeros(97)
allL = np.log(np.maximum([a[1] for L in vids.values() for a in L], 0.05))
for v in range(97):
    x = np.log(np.maximum([a[1] for L in vids.values() for a in L if a[2] == v], 0.05))
    k = len(x)
    # shrink toward global with 20 pseudo-counts
    mu[v] = (x.sum() + 20 * allL.mean()) / (k + 20) if k else allL.mean()
    sd[v] = np.sqrt(((x - mu[v]) ** 2).sum() + 20 * allL.var()) / np.sqrt(k + 20) if k else allL.std()
out["dur_mu"], out["dur_sd"] = mu.astype(np.float32), sd.astype(np.float32)
np.savez("epic_nounctx.npz", **out)
print("saved epic_nounctx.npz")
