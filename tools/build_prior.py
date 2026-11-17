"""Build the verb-noun co-occurrence prior from the EPIC-100 TRAIN split only (no val/test data).

Output: <weights>/epic_prior.npz with counts[97, 300] and smoothed P(noun | verb).
    python tools/build_prior.py --out C:/prism_data/weights/epic_prior.npz
"""
import argparse
import csv

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--train", default=r"C:/prism_ext/epic-kitchens-100-annotations/EPIC_100_train.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--alpha", type=float, default=0.5, help="additive smoothing")
a = ap.parse_args()

counts = np.zeros((97, 300), np.float64)
with open(a.train, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        counts[int(r["verb_class"]), int(r["noun_class"])] += 1
p_n_given_v = (counts + a.alpha) / (counts + a.alpha).sum(1, keepdims=True)
p_v = counts.sum(1) / counts.sum()
p_n = counts.sum(0) / counts.sum()
np.savez_compressed(a.out, counts=counts.astype(np.float32), p_n_given_v=p_n_given_v.astype(np.float32),
                    p_v=p_v.astype(np.float32), p_n=p_n.astype(np.float32))
print(f"segments {int(counts.sum())}  nonzero pairs {(counts > 0).sum()}  -> {a.out}")
