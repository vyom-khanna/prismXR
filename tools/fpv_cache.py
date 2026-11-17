"""Cache per-frame signals for FPVSum dev videos (CLIP-L/14 embeddings @2fps + 64x64 grey frames)
so importance-scoring ideas can be evaluated in seconds.

    C:/prism_env/Scripts/python.exe tools/fpv_cache.py
"""
import csv
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "submission", "code"))
os.environ.setdefault("PRISM_WEIGHTS", r"C:/prism_data/weights")

from prism.clip_model import Clip  # noqa: E402
from prism.video_io import probe, read_frames  # noqa: E402

OUT = r"C:/prism_data/cache/fpv"
FPS = 2.0
os.makedirs(OUT, exist_ok=True)
clip = Clip()
for v in sorted(glob.glob(r"C:/prism_data/fpvsum/*.mp4")):
    vid = os.path.splitext(os.path.basename(v))[0]
    dst = os.path.join(OUT, vid + ".npz")
    if os.path.exists(dst):
        continue
    dur = probe(v)["duration"]
    fr, ts = read_frames(v, fps=FPS, size=224)
    emb = clip.encode_frames(fr)
    grey = fr[:, ::4, ::4].mean(-1).astype(np.uint8)  # 56x56
    np.savez_compressed(dst, emb=emb.astype(np.float16), grey=grey, ts=ts, dur=dur)
    print(vid, len(ts), flush=True)
