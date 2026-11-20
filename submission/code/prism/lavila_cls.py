"""LaViLa TimeSformer-L (CLIP ViT-L/14 init) fine-tuned on EPIC-100 *train* for action recognition.

Only the verb (97) and noun (300) heads are used. The checkpoint's third (action) head indexes a pair list
that LaViLa builds from train + validation annotations, so we do not use it (no validation knowledge).
Input: 16 frames sampled uniformly over a segment, 224x224 RGB centre crop, OpenAI-CLIP mean/std on 0-255.
Code: vendored `lavila/timesformer.py` (MIT, facebookresearch/LaViLa).
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch import nn

from . import config as C

NUM_FRAMES = 16
SIZE = 224
MEAN = torch.tensor([108.3272985, 116.7460125, 104.09373615000001]).view(1, 3, 1, 1, 1)
STD = torch.tensor([68.5005327, 66.6321579, 70.32316305]).view(1, 3, 1, 1, 1)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LaViLaEPIC:
    def __init__(self, path: str | None = None, device: str | None = None):
        from .lavila.timesformer import SpaceTimeTransformer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = path or os.path.join(C.WEIGHTS_DIR, "lavila", "tsf_l_ek100_vn_fp16.pth")
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.visual = SpaceTimeTransformer(img_size=SIZE, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
                                           num_frames=NUM_FRAMES, time_init="zeros", num_classes=0,
                                           attention_style="frozen-in-time", ln_pre=True, act_layer=QuickGELU)
        vis = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
        missing, unexpected = self.visual.load_state_dict(vis, strict=False)
        bad = [k for k in missing if "head" not in k and "pre_logits" not in k]
        if bad:
            raise RuntimeError(f"LaViLa weights missing keys: {bad[:5]}")
        self.fc_verb = nn.Linear(1024, 97)
        self.fc_noun = nn.Linear(1024, 300)
        for fc, i in ((self.fc_verb, 0), (self.fc_noun, 1)):
            fc.weight.data = sd[f"fc_cls.{i}.weight"].float()
            fc.bias.data = sd[f"fc_cls.{i}.bias"].float()
        half = self.device == "cuda"
        dt = torch.float16 if half else torch.float32
        self.dtype = dt
        self.visual = self.visual.eval().to(self.device, dt)
        self.fc_verb = self.fc_verb.eval().to(self.device, dt)
        self.fc_noun = self.fc_noun.eval().to(self.device, dt)

    @torch.no_grad()
    def predict(self, clips: np.ndarray):
        """clips uint8 [B, 16, 224, 224, 3] RGB -> (verb probs [B, 97], noun probs [B, 300])."""
        x = torch.from_numpy(clips).to(self.device).permute(0, 4, 1, 2, 3).float()  # B C T H W
        x = ((x - MEAN.to(x.device)) / STD.to(x.device)).to(self.dtype)
        e = self.visual(x)
        if isinstance(e, list):
            e = e[0]
        return (torch.softmax(self.fc_verb(e).float(), 1).cpu().numpy(),
                torch.softmax(self.fc_noun(e).float(), 1).cpu().numpy())


def sample_ids(t0: float, t1: float, fps: float, n_frames: int) -> np.ndarray:
    """LaViLa's uniform segment sampling (get_frame_ids without jitter) -> frame indices at `fps`."""
    s = int(round(t0 * fps))
    e = s + max(int((t1 - t0) * fps), NUM_FRAMES)
    e = min(e, n_frames)
    seg = float(e - s - 1) / NUM_FRAMES
    ids = [(int(round(seg * i) + s) + min(int(round(seg * (i + 1)) + s), e)) // 2 for i in range(NUM_FRAMES)]
    return np.clip(np.array(ids), 0, n_frames - 1)
