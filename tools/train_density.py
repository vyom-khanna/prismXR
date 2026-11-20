"""Train the action-density model (prism/density.py) on the EPIC-100 TRAIN split only.

Inputs: ActionFormer's released SlowFast features for train videos (feats, v_prob, n_prob).
Target: number of train-annotation action midpoints per feature step. Loss: Poisson NLL.
10% of train videos (by participant) are held out for early stopping. Val GT is never read here.

    C:/prism_env/Scripts/python.exe tools/train_density.py --out C:/prism_data/weights/epic_density.pt
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "submission", "code"))
from prism.density import DensityNet, make_input  # noqa: E402

FEAT = r"C:/prism_data/af_features/epic_kitchens/features"
TRAIN_CSV = r"C:/prism_ext/epic-kitchens-100-annotations/EPIC_100_train.csv"
STEP = 16 / 30.0


def ts(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def load(singleton=False):
    mids, keys = {}, {}
    for r in csv.DictReader(open(TRAIN_CSV, encoding="utf-8")):
        mids.setdefault(r["video_id"], []).append((ts(r["start_timestamp"]) + ts(r["stop_timestamp"])) / 2)
        keys.setdefault(r["video_id"], []).append((r["verb_class"], r["noun_class"]))
    data = {}
    for v, m in mids.items():
        p = os.path.join(FEAT, v + ".npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            x = make_input(z["feats"], z["v_prob"], z["n_prob"])
        y = np.zeros(len(x), np.float32)
        idx = np.clip(np.floor(np.array(m) / STEP - 0.5).astype(int), 0, len(x) - 1)
        if singleton:  # weight = 1 / (#occurrences of this pair in the video): target = new distinct pairs
            from collections import Counter
            c = Counter(keys[v])
            np.add.at(y, idx, np.array([1.0 / c[k] for k in keys[v]], np.float32))
        else:
            np.add.at(y, idx, 1.0)
        data[v] = (x, y)
    return data


def batches(data, vids, crop, bs, rng):
    order = rng.permutation(len(vids))
    for i in range(0, len(order), bs):
        xs, ys = [], []
        for j in order[i:i + bs]:
            x, y = data[vids[j]]
            if len(x) > crop:
                s = rng.integers(0, len(x) - crop)
                x, y = x[s:s + crop], y[s:s + crop]
            xs.append(x)
            ys.append(y)
        L = max(len(x) for x in xs)
        X = np.zeros((len(xs), L, xs[0].shape[1]), np.float32)
        Y = np.zeros((len(xs), L), np.float32)
        M = np.zeros((len(xs), L), np.float32)
        for k, (x, y) in enumerate(zip(xs, ys)):
            X[k, :len(x)], Y[k, :len(y)], M[k, :len(y)] = x, y, 1
        yield torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:/prism_data/weights/epic_density.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.3, help="feature noise (x feature std) for robustness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--singleton", action="store_true", help="target = 1/pair-count weighted midpoints")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = load(a.singleton)
    vids = sorted(data)
    parts = sorted({v.split("_")[0] for v in vids})
    hold = {p for i, p in enumerate(parts) if i % 10 == 3}
    tr = [v for v in vids if v.split("_")[0] not in hold]
    ho = [v for v in vids if v.split("_")[0] in hold]
    print(f"train videos {len(tr)}  held-out {len(ho)}  (participants held out: {sorted(hold)})", flush=True)
    fstd = float(np.concatenate([data[v][0][:, :2304] for v in tr[:50]]).std())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"d": 256}
    net = DensityNet(**cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_sd = 1e9, None
    for ep in range(a.epochs):
        net.train()
        for X, Y, M in batches(data, tr, 512, 8, rng):
            X, Y, M = X.to(dev), Y.to(dev), M.to(dev)
            if a.noise:
                X[..., :2304] += a.noise * fstd * torch.randn_like(X[..., :2304])
            lr = net(X)
            loss = ((torch.exp(lr) - Y * lr) * M).sum() / M.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        tot, n, cors = 0.0, 0, []
        with torch.no_grad():
            for v in ho:
                x, y = data[v]
                lr = net(torch.from_numpy(x)[None].to(dev))[0].cpu().numpy()
                tot += float((np.exp(lr) - y * lr).sum())
                n += len(y)
                k = 9  # ~4.8 s windows: correlation of predicted vs true counts
                sm = lambda z: np.convolve(z, np.ones(k), "valid")  # noqa: E731
                cors.append(np.corrcoef(sm(np.exp(lr)), sm(y))[0, 1])
        nll = tot / n
        print(f"ep {ep:2d}  held-out NLL {nll:.4f}  window-count corr {np.nanmean(cors):.3f}", flush=True)
        if nll < best:
            best, best_sd = nll, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save({"state_dict": best_sd, "cfg": cfg}, a.out)
    print("saved", a.out, "best held-out NLL", round(best, 4))


if __name__ == "__main__":
    main()
