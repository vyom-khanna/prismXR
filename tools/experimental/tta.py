"""Per-video test-time adaptation (unsupervised; uses only the input video's own features).

Our own SlowFast features come out with a smaller spread than the released EPIC features the detectors were trained on
(same videos: std ratio about 0.81, cosine 0.92). `recalibrate` maps one video's features onto the EPIC TRAIN-split
feature statistics (weights/epic_train_featstats.npz, from tools/build_featstats.py) - equivalent to a per-video
re-parameterisation of the detector's input projection.
"""
from __future__ import annotations

import os

import numpy as np

from . import config as C

_ST = None


def _stats():
    global _ST
    if _ST is None:
        with np.load(os.path.join(C.WEIGHTS_DIR, "epic_train_featstats.npz")) as z:
            _ST = z["mean"].astype(np.float32), z["std"].astype(np.float32), float(z["pv_std"])
    return _ST


def recalibrate(feats: np.ndarray, mode: str) -> np.ndarray:
    """mode 'chan': per-channel standardise to train mean/std;  'global': one scale + per-channel train mean;
    'scale': only rescale the spread (keep the video's own channel means)."""
    if not mode or len(feats) < 2:
        return feats
    mu_t, sd_t, pv_t = _stats()
    mu, sd = feats.mean(0), feats.std(0) + 1e-6
    if mode == "chan":
        out = (feats - mu) / sd * sd_t + mu_t
    elif mode == "global":
        out = (feats - mu) * (sd_t.mean() / sd.mean()) + mu_t
    elif mode == "scale":
        out = mu + (feats - mu) * (sd_t.mean() / sd.mean())
    elif mode.startswith("pv"):  # 'pv' / 'pv0.5': spread -> median per-video spread of TRAIN videos (optionally partial)
        a = float(mode[2:] or 1.0)
        out = mu + (feats - mu) * (pv_t / sd.mean()) ** a
    else:
        raise ValueError(mode)
    return out.astype(np.float32)
