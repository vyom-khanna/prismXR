"""Action-density model for EPIC highlights.

A small temporal conv net on SlowFast-EPIC features that predicts, for every feature step (16 frames at
30 fps = 0.533 s), the expected number of action *midpoints* in that step. Trained by
tools/train_density.py on the EPIC-KITCHENS-100 **train** split only (released SlowFast features + train
annotations). Coverage selection then puts highlight windows where many distinct actions happen.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from . import config as C

IN_DIM = 2304 + 97 + 300  # SlowFast feature + verb softmax + noun softmax


class _Block(nn.Module):
    def __init__(self, d, dil, p):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dil, dilation=dil)
        self.norm = nn.GroupNorm(8, d)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(torch.nn.functional.gelu(self.norm(self.conv(x))))


class DensityNet(nn.Module):
    def __init__(self, d: int = 256, dils=(1, 2, 4, 8, 16), p: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(IN_DIM), nn.Linear(IN_DIM, d), nn.GELU(), nn.Dropout(p))
        self.blocks = nn.Sequential(*[_Block(d, k, p) for k in dils])
        self.out = nn.Conv1d(d, 1, 1)

    def forward(self, x):  # x [B, T, IN_DIM] -> log-rate [B, T]
        h = self.inp(x).transpose(1, 2)
        return self.out(self.blocks(h))[:, 0]


def make_input(feats: np.ndarray, vprob: np.ndarray, nprob: np.ndarray) -> np.ndarray:
    return np.concatenate([feats, vprob, nprob], 1).astype(np.float32)


class Density:
    def __init__(self, path: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(path or os.path.join(C.WEIGHTS_DIR, "epic_density.pt"), map_location="cpu")
        self.net = DensityNet(**sd.get("cfg", {}))
        self.net.load_state_dict(sd["state_dict"])
        self.net.eval().to(self.device)

    @torch.no_grad()
    def predict(self, feats, vprob, nprob) -> np.ndarray:
        x = torch.from_numpy(make_input(feats, vprob, nprob))[None].to(self.device)
        return torch.exp(self.net(x))[0].float().cpu().numpy()
