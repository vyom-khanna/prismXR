"""Sports (FPVSum-style) frame importance: a ridge-regressed combination of cheap visual signals.

Signals at 2 fps, each z-scored and lightly smoothed:
  motion (grey frame difference, plus 2.5 s smoothed), CLIP scene change (5.5 s smoothed),
  CLIP 'exciting vs boring' prompt score, brightness, contrast, temporal centre prior, local uniqueness.
Weights were fitted by ridge regression (lambda=100) to human importance curves and checked with
leave-one-category-out CV (F1 0.30 vs 0.12 random); see tools/fpv_explore.py.
"""
from __future__ import annotations

import numpy as np

FPS = 2.0
SEG_LEN = 4.0

GENERIC_POS = ["an exciting moment in an action sports video", "a spectacular jump in the air", "a dangerous stunt",
               "a crash or fall", "a breathtaking scenic view", "a close encounter with another person",
               "the most interesting moment of the video"]
GENERIC_NEG = ["a boring moment with nothing happening", "a static view", "a blurry dark frame", "waiting around"]

WEIGHTS = {"change": 0.025, "motion": 0.111, "prompt": 0.069, "bright": 0.015, "contrast": -0.097,
           "center": 0.114, "local_unique": -0.008, "motion_s5": 0.215, "motion_s11": 0.013,
           "motion_s21": 0.033, "change_s11": 0.096}


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-6) if len(x) else x


def _smooth(x, k):
    if k <= 1 or len(x) < k:
        return x
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def importance(frames: np.ndarray, emb: np.ndarray, ts: np.ndarray, dur: float, text_pos: np.ndarray,
               text_neg: np.ndarray) -> np.ndarray:
    """frames uint8 [N,224,224,3] at FPS, emb L2-normalised CLIP [N,D] -> importance [N]."""
    grey = frames[:, ::4, ::4].mean(-1).astype(np.float32)
    n = len(ts)
    raw = {
        "change": np.concatenate([[0.0], 1 - np.sum(emb[1:] * emb[:-1], 1)]),
        "motion": np.concatenate([[0.0], np.abs(np.diff(grey, axis=0)).mean((1, 2))]),
        "prompt": (emb @ text_pos.T).max(1) - (emb @ text_neg.T).max(1),
        "bright": grey.mean((1, 2)),
        "contrast": grey.std((1, 2)),
        "center": -np.abs(ts / dur - 0.5),
    }
    W = 20
    sim = emb @ emb.T
    raw["local_unique"] = -np.array([sim[i, max(0, i - W):i + W + 1].mean() for i in range(n)])
    sig = {k: _smooth(_z(v), 3) for k, v in raw.items()}
    for k in (5, 11, 21):
        sig[f"motion_s{k}"] = _smooth(sig["motion"], k)
    sig["change_s11"] = _smooth(sig["change"], 11)
    return sum(w * sig[k] for k, w in WEIGHTS.items())
