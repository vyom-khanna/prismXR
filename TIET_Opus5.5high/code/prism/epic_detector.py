"""ActionFormer temporal action detection on EPIC-100 SlowFast features (separate verb / noun models)."""
from __future__ import annotations

import copy
import os

import numpy as np
import torch

from . import config as C
from .epic_features import CLIP_LEN, FPS, STRIDE

_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "actionformer", "configs")


class ActionFormerEPIC:
    """kind: 'verb' (97 classes) or 'noun' (300 classes)."""

    def __init__(self, kind: str, ckpt: str | None = None, device: str | None = None,
                 max_seg_num: int = 2000, min_score: float = 0.001):
        from .actionformer.core.config import DEFAULTS, load_config
        from .actionformer.modeling import make_meta_arch

        self.kind = kind
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = load_config(os.path.join(_CFG_DIR, f"epic_slowfast_{kind}.yaml"), copy.deepcopy(DEFAULTS))
        cfg["model"]["test_cfg"]["max_seg_num"] = max_seg_num
        cfg["model"]["test_cfg"]["min_score"] = min_score
        self.cfg = cfg
        model = make_meta_arch(cfg["model_name"], **cfg["model"])
        ckpt = ckpt or os.path.join(C.WEIGHTS_DIR, "actionformer", f"epic_slowfast_{kind}_ema.pth.tar")
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict_ema", sd.get("state_dict", sd))
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def detect(self, feats: np.ndarray, duration: float):
        """feats [T, 2304] -> (segs [N, 2] seconds, scores [N], labels [N] class ids)."""
        item = {"video_id": "v", "feats": torch.from_numpy(np.ascontiguousarray(feats.T)).float(),
                "segments": None, "labels": None, "fps": FPS, "duration": float(duration),
                "feat_stride": STRIDE, "feat_num_frames": CLIP_LEN}
        out = self.model([item])[0]
        return out["segments"].numpy(), out["scores"].numpy(), out["labels"].numpy().astype(int)
