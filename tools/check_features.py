"""Compare our SlowFast-EPIC features with ActionFormer's released features on locally complete EPIC videos,
for several preprocessing recipes, to find the one that reproduces them.

    C:/prism_env/Scripts/python.exe tools/check_features.py [--videos P01_13,P02_13]
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "TIET_Opus5.5high", "code"))
os.environ.setdefault("PRISM_WEIGHTS", r"C:/prism_data/weights")
from prism.epic_features import SlowFastEPIC  # noqa: E402
from prism.video_io import _tool  # noqa: E402

REL = r"C:/prism_data/af_features/epic_kitchens/features"


def complete_videos():
    out = []
    for f in sorted(glob.glob(r"C:/prism_data/epic/EPIC-KITCHENS/*/videos/*.MP4")):
        r = subprocess.run([_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f],
                           capture_output=True, text=True)
        v = os.path.basename(f)[:-4]
        if r.stdout.strip() and os.path.exists(os.path.join(REL, v + ".npz")):
            out.append((v, f))
    return out


def cos(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return (a * b).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos")
    a = ap.parse_args()
    vids = complete_videos()
    if a.videos:
        keep = set(a.videos.split(","))
        vids = [x for x in vids if x[0] in keep]
    print("videos:", [v for v, _ in vids])
    sf = SlowFastEPIC()
    recipes = {
        "bgr_crop224": dict(short_side=256, crop=224, bgr=True),
        "bgr_full256": dict(short_side=256, crop=None, bgr=True),
        "bgr_crop256": dict(short_side=256, crop=256, bgr=True),
        "rgb_crop256": dict(short_side=256, crop=256, bgr=False),
    }
    for name, kw in recipes.items():
        cs, agree, lens = [], [], []
        for v, f in vids:
            ours = sf.extract(f, **kw)
            rel = np.load(os.path.join(REL, v + ".npz"))
            n = min(len(ours["feats"]), len(rel["feats"]))
            lens.append((len(ours["feats"]), len(rel["feats"])))
            # try small temporal offsets in case of alignment differences
            best = max(((np.mean(cos(ours["feats"][max(0, s):n + min(0, s)], rel["feats"][max(0, -s):n - max(0, s)])), s)
                        for s in (-2, -1, 0, 1, 2)))
            cs.append(best[0])
            agree.append(np.mean(ours["verb"][:n].argmax(1) == rel["v_prob"][:n].argmax(1)))
        print(f"{name:<13} cosine {np.mean(cs):.4f}  verb-argmax agree {np.mean(agree):.3f}  "
              f"lens(ours,rel) {lens[:3]}", flush=True)


if __name__ == "__main__":
    main()
