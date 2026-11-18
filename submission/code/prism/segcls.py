"""Segment classifier on SlowFast-EPIC features (verb, noun and joint action heads).

Pools features over a proposed segment (mean + max inside, mean of a context band around it) and
classifies it. Trained by tools/train_segcls.py on EPIC-KITCHENS-100 **train** segments only. Used by
pairing.py as extra verb / noun / pair evidence for each detection.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from . import config as C

FEAT_DIM = 2304 + 97 + 300


def pool(x: np.ndarray, tc: np.ndarray, segs: np.ndarray, ctx: float = 2.0) -> np.ndarray:
    """x [T, D] per-step inputs, segs [N, 2] seconds -> [N, 3*D + 1] (mean, max, context mean, log length)."""
    T = len(tc)
    cs = np.concatenate([np.zeros((1, x.shape[1]), np.float32), np.cumsum(x, 0, dtype=np.float64).astype(np.float32)])
    lo = np.searchsorted(tc, segs[:, 0], "left")
    hi = np.searchsorted(tc, segs[:, 1], "right")
    near = np.clip(np.searchsorted(tc, segs.mean(1)), 0, T - 1)
    empty = hi <= lo
    lo, hi = np.where(empty, near, lo), np.where(empty, near + 1, hi)
    mean = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
    mx = np.stack([x[a:b].max(0) for a, b in zip(lo, hi)])
    clo = np.searchsorted(tc, segs[:, 0] - ctx, "left")
    chi = np.searchsorted(tc, segs[:, 1] + ctx, "right")
    cmean = (cs[chi] - cs[clo]) / np.maximum(chi - clo, 1)[:, None]
    ln = np.log(np.maximum(segs[:, 1] - segs[:, 0], 0.1))[:, None].astype(np.float32)
    return np.concatenate([mean, mx, cmean, ln], 1).astype(np.float32)


class SegNet(nn.Module):
    def __init__(self, n_act: int, d: int = 1024, p: float = 0.5):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(3 * FEAT_DIM + 1), nn.Dropout(0.2), nn.Linear(3 * FEAT_DIM + 1, d),
                                  nn.GELU(), nn.Dropout(p))
        self.verb, self.noun, self.act = nn.Linear(d, 97), nn.Linear(d, 300), nn.Linear(d, n_act)

    def forward(self, x):
        h = self.body(x)
        return self.verb(h), self.noun(h), self.act(h)


class SegCls:
    def __init__(self, path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(path or os.path.join(C.WEIGHTS_DIR, "epic_segcls.pt"), map_location="cpu")
        self.pairs = np.asarray(sd["pairs"])  # [n_act, 2] (verb id, noun id)
        self.net = SegNet(len(self.pairs), **sd.get("cfg", {}))
        self.net.load_state_dict(sd["state_dict"])
        self.net.eval().to(self.device)

    @torch.no_grad()
    def predict(self, feats, vprob, nprob, tc, segs, bs: int = 512):
        """-> verb probs [N, 97], noun probs [N, 300], action probs [N, n_act]."""
        x = np.concatenate([feats, vprob, nprob], 1).astype(np.float32)
        outs = [[], [], []]
        for i in range(0, len(segs), bs):
            z = torch.from_numpy(pool(x, tc, segs[i:i + bs])).to(self.device)
            for k, o in enumerate(self.net(z)):
                outs[k].append(torch.softmax(o.float(), 1).cpu().numpy())
        return tuple(np.concatenate(o) if o else np.zeros((0, 1), np.float32) for o in outs)
