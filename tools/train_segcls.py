"""Train prism/segcls.py on EPIC-100 TRAIN segments (released SlowFast features + train annotations).

Training samples: GT train segments, plus jittered copies (boundaries moved by up to 30% of the length) so
the classifier also works on imperfect detector segments. Action head = train pairs seen >= MIN_PAIR times.
10% of train participants held out for early stopping. Val GT is never read here.

    C:/prism_env/Scripts/python.exe tools/train_segcls.py --out C:/prism_data/weights/epic_segcls.pt
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "submission", "code"))
from prism.segcls import SegNet, pool  # noqa: E402

FEAT = r"C:/prism_data/af_features/epic_kitchens/features"
TRAIN_CSV = r"C:/prism_ext/epic-kitchens-100-annotations/EPIC_100_train.csv"
MIN_PAIR = 3


def ts(s):
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:/prism_data/weights/epic_segcls.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--jitter", type=int, default=3, help="jittered copies per GT segment")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="fine-tune from this checkpoint (weight-space soups)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--spread", type=float, default=0.0, help="random spread scale s~U(spread,1) of SlowFast dims around the train mean")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    rows = {}
    for r in csv.DictReader(open(TRAIN_CSV, encoding="utf-8")):
        rows.setdefault(r["video_id"], []).append((ts(r["start_timestamp"]), ts(r["stop_timestamp"]),
                                                   int(r["verb_class"]), int(r["noun_class"])))
    cnt = Counter((v, n) for rs in rows.values() for _, _, v, n in rs)
    pairs = sorted(p for p, c in cnt.items() if c >= MIN_PAIR)
    pidx = {p: i for i, p in enumerate(pairs)}
    print(f"{len(pairs)} action classes (pairs seen >= {MIN_PAIR}x), covering "
          f"{100 * sum(cnt[p] for p in pairs) / sum(cnt.values()):.1f}% of train segments", flush=True)

    X, Yv, Yn, Ya, P = [], [], [], [], []
    for vid, rs in rows.items():
        p = os.path.join(FEAT, vid + ".npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            x = np.concatenate([z["feats"], z["v_prob"], z["n_prob"]], 1).astype(np.float32)
        tc = (np.arange(len(x)) * 16 + 16) / 30.0
        segs = [(s, e) for s, e, _, _ in rs]
        labs = [(v, n) for _, _, v, n in rs]
        for _ in range(a.jitter):  # jittered copies
            for (s, e), l in zip(list(segs[:len(rs)]), labs[:len(rs)]):
                L = max(e - s, 0.2)
                segs.append((s + rng.uniform(-0.3, 0.3) * L, e + rng.uniform(-0.3, 0.3) * L))
                labs.append(l)
        segs = np.array(segs)
        segs[:, 1] = np.maximum(segs[:, 1], segs[:, 0] + 0.1)
        X.append(pool(x, tc, segs))
        Yv += [v for v, _ in labs]
        Yn += [n for _, n in labs]
        Ya += [pidx.get(l, -100) for l in labs]
        P += [vid.split("_")[0]] * len(labs)
    X = np.concatenate(X)
    Yv, Yn, Ya, P = map(np.array, (Yv, Yn, Ya, P))
    parts = sorted(set(P))
    hold = {p for i, p in enumerate(parts) if i % 10 == 3}
    tr, ho = ~np.isin(P, list(hold)), np.isin(P, list(hold))
    print(f"samples train {tr.sum()} held-out {ho.sum()}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy(X)
    fstd = float(X[:5000].std())
    net = SegNet(len(pairs)).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu")["state_dict"])
    fmu = Xt[:20000].mean(0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    tri, hoi = np.where(tr)[0], np.where(ho)[0]
    best, best_sd = -1, None
    for ep in range(a.epochs):
        net.train()
        for i in range(0, len(tri), 512):
            b = rng.permutation(tri)[:512] if i == 0 else tri[rng.integers(0, len(tri), 512)]
            xb = Xt[b].to(dev)
            if a.spread:
                s_ = torch.empty(xb.shape[0], 1, device=dev).uniform_(a.spread, 1.0)
                xb = fmu + s_ * (xb - fmu)
            if a.noise:
                xb = xb + a.noise * fstd * torch.randn_like(xb)
            ov, on, oa = net(xb)
            loss = ce(ov, torch.from_numpy(Yv[b]).to(dev)) + ce(on, torch.from_numpy(Yn[b]).to(dev)) + \
                0.5 * ce(oa, torch.from_numpy(Ya[b]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            ov, on, oa = (torch.cat(z) for z in zip(*[net(Xt[hoi[i:i + 4096]].to(dev)) for i in range(0, len(hoi), 4096)]))
        tv = (ov.argmax(1).cpu().numpy() == Yv[hoi]).mean()
        tn = (on.argmax(1).cpu().numpy() == Yn[hoi]).mean()
        pa = np.array(pairs)[oa.argmax(1).cpu().numpy()]
        tact = ((pa[:, 0] == Yv[hoi]) & (pa[:, 1] == Yn[hoi])).mean()
        print(f"ep {ep:2d} held-out top1 verb {100 * tv:.1f} noun {100 * tn:.1f} action {100 * tact:.1f}", flush=True)
        if tv + tn > best:
            best, best_sd = tv + tn, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save({"state_dict": best_sd, "pairs": torch.tensor(pairs), "cfg": {}}, a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
