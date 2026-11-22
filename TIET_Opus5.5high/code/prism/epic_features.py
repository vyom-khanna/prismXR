"""SlowFast-R50 (EPIC-100 fine-tuned) clip features, matching ActionFormer's EPIC input format.

ActionFormer: "SlowFast pretrained on EPIC-100 train (action classification), clips of 32 frames at
30 fps with a stride of 16 frames" -> one 2304-d vector per 16/30 s.
Returns per-clip features + per-clip verb/noun softmax (useful for rescoring detections).
Frames are streamed from ffmpeg, so memory stays flat even for 30+ minute videos.
The exact spatial recipe is verified against ActionFormer's released val features (tools/dev).
"""
from __future__ import annotations

import os
from types import SimpleNamespace as NS

import numpy as np
import torch

from . import config as C
from .video_io import iter_frames, probe

FPS = 30
CLIP_LEN = 32
STRIDE = 16


def _cfg():
    return NS(
        DETECTION=NS(ENABLE=False, ALIGNED=True, ROI_XFORM_RESOLUTION=7, SPATIAL_SCALE_FACTOR=16),
        MODEL=NS(ARCH="slowfast", NUM_CLASSES=[97, 300], DROPOUT_RATE=0.5, FC_INIT_STD=0.01),
        RESNET=NS(DEPTH=50, NUM_GROUPS=1, WIDTH_PER_GROUP=64, TRANS_FUNC="bottleneck_transform",
                  STRIDE_1X1=False, INPLACE_RELU=True, ZERO_INIT_FINAL_BN=True,
                  NUM_BLOCK_TEMP_KERNEL=[[3, 3], [4, 4], [6, 6], [3, 3]],
                  SPATIAL_STRIDES=[[1, 1], [2, 2], [2, 2], [2, 2]],
                  SPATIAL_DILATIONS=[[1, 1], [1, 1], [1, 1], [1, 1]]),
        NONLOCAL=NS(LOCATION=[[[], []]] * 4, GROUP=[[1, 1]] * 4, INSTANTIATION="dot_product",
                    POOL=[[[1, 2, 2], [1, 2, 2]]] * 4),
        DATA=NS(NUM_FRAMES=32, CROP_SIZE=224, INPUT_CHANNEL_NUM=[3, 3]),
        SLOWFAST=NS(ALPHA=4, BETA_INV=8, FUSION_CONV_CHANNEL_RATIO=2, FUSION_KERNEL_SZ=7),
    )


def frame_size(video: str, short_side: int) -> tuple[int, int]:
    info = probe(video)
    w, h = info["width"], info["height"]
    if info["rotation"] in (90, -90, 270, -270):
        w, h = h, w
    if w >= h:
        return int(round(w * short_side / h / 2) * 2), short_side
    return short_side, int(round(h * short_side / w / 2) * 2)


class SlowFastEPIC:
    def __init__(self, ckpt: str | None = None, device: str | None = None):
        from .slowfast.video_model_builder import SlowFast

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m = SlowFast(_cfg())
        ckpt = ckpt or C.SLOWFAST_WEIGHTS
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model_state", sd)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"SlowFast checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.m = m.eval().to(self.device)
        self.mean, self.std = 0.45, 0.225

    @torch.no_grad()
    def _forward(self, x: torch.Tensor):
        """x: [B, 3, 32, H, W] normalised -> (feat [B, 2304], verb_p [B, 97], noun_p [B, 300])."""
        m = self.m
        idx = torch.linspace(0, x.shape[2] - 1, x.shape[2] // 4).long().to(x.device)
        y = [torch.index_select(x, 2, idx), x]
        y = m.s1(y); y = m.s1_fuse(y)
        y = m.s2(y); y = m.s2_fuse(y)
        for p in range(2):
            y[p] = getattr(m, f"pathway{p}_pool")(y[p])
        y = m.s3(y); y = m.s3_fuse(y)
        y = m.s4(y); y = m.s4_fuse(y)
        y = m.s5(y)
        feat = torch.cat([t.float().mean(dim=(2, 3, 4)) for t in y], 1)  # global avg pool: 2048 + 256
        vp = torch.softmax(m.head.projection_verb(feat), -1)
        npb = torch.softmax(m.head.projection_noun(feat), -1)
        return feat, vp.float(), npb.float()

    def _run_batch(self, clips, bgr: bool = True):
        x = torch.from_numpy(np.stack(clips)).to(self.device)  # B,T,H,W,3 uint8 (RGB from ffmpeg)
        if bgr:  # EPIC SlowFast was trained on cv2.imread frames, i.e. BGR channel order
            x = x.flip(-1)
        x = x.permute(0, 4, 1, 2, 3).float().div_(255).sub_(self.mean).div_(self.std)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device == "cuda"):
            f, v, n = self._forward(x)
        return f.float().cpu().numpy(), v.cpu().numpy(), n.cpu().numpy()

    def extract(self, video: str, short_side: int = 256, crop: int | None = None, sampling_rate: int = 1,
                batch: int = 8, bgr: bool = True):
        # Defaults reproduce ActionFormer's released features best (tools/check_features.py):
        # BGR order + full short-side-256 frame: cosine 0.93 (RGB was 0.75, 224 centre crop 0.91).
        """Return dict(feats [T, 2304], verb [T, 97], noun [T, 300], t_center [T] s, n_frames)."""
        size = frame_size(video, short_side)
        span = CLIP_LEN * sampling_rate

        def prep(chunk):
            if crop:
                H, W = chunk.shape[1:3]
                y0, x0 = (H - crop) // 2, (W - crop) // 2
                chunk = chunk[:, y0:y0 + crop, x0:x0 + crop]
            return np.ascontiguousarray(chunk)

        buf = None          # rolling frame buffer
        buf_start = 0       # absolute index of buf[0]
        next_start = 0      # next clip start (absolute frame index)
        pending, starts, out = [], [], ([], [], [])

        def flush():
            if pending:
                f, v, n = self._run_batch(pending, bgr)
                out[0].append(f); out[1].append(v); out[2].append(n)
                pending.clear()

        n_total = 0
        for chunk in iter_frames(video, FPS, size, chunk=64):
            chunk = prep(chunk)
            buf = chunk if buf is None else np.concatenate([buf, chunk])
            n_total += len(chunk)
            while next_start + span <= n_total:
                rel = next_start - buf_start
                pending.append(buf[rel:rel + span:sampling_rate])
                starts.append(next_start)
                next_start += STRIDE
                if len(pending) == batch:
                    flush()
            drop = next_start - buf_start  # frames before next_start are no longer needed
            if drop > 0:
                buf, buf_start = buf[drop:], buf_start + drop
        # tail: video shorter than one clip, or leftover frames -> one final padded clip
        if buf is not None and (not starts or n_total - (starts[-1] + span) >= STRIDE // 2):
            s = max(0, n_total - span) if starts else 0
            rel = max(0, s - buf_start)
            clip = buf[rel:]
            if len(clip) < span:
                clip = np.concatenate([clip, np.repeat(clip[-1:], span - len(clip), 0)])
            pending.append(clip[:span:sampling_rate])
            starts.append(s)
        flush()
        t_center = (np.array(starts, float) + span / 2) / FPS
        return {"feats": np.concatenate(out[0]), "verb": np.concatenate(out[1]),
                "noun": np.concatenate(out[2]), "t_center": t_center, "n_frames": n_total}
