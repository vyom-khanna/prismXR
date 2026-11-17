"""Dev check: run our vendored ActionFormer (pure-python NMS) + pairing on ActionFormer's RELEASED
val features and score with tools/eval. Verifies the detector port before EPIC videos are local,
and gives the pair-mAP / coverage we can expect once our own SlowFast features match.

    C:/prism_env/Scripts/python.exe tools/check_detector.py --feat-dir C:/prism_data/af_features/epic_kitchens/features
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "submission", "code"))
os.environ.setdefault("PRISM_WEIGHTS", r"C:/prism_data/weights")

from prism import config as C  # noqa: E402
from prism import highlights as H, select_coverage as SC  # noqa: E402
from prism.epic_detector import ActionFormerEPIC  # noqa: E402
from prism.pairing import PairCfg, pair_actions  # noqa: E402

MANIFEST = os.path.join(ROOT, "Student package_track04", "Student package_track04", "manifest.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--feat-dir", required=True)
ap.add_argument("--out", default=r"C:/prism_data/dev_outputs/released_feats")
ap.add_argument("--save-dets", default=r"C:/prism_data/dev_outputs/released_dets.npz")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

with open(MANIFEST) as f:
    epic = [(r["video_id"], float(r["duration_sec"])) for r in csv.DictReader(f) if r["dataset"] == "epic_kitchens_100"]
epic = [(v, d) for v, d in epic if os.path.exists(os.path.join(a.feat_dir, v + ".npz"))]
print(f"{len(epic)} EPIC videos with released features")
clipprob = {}
with open(C.CLASSES_JSON) as f:
    cls = json.load(f)
prior = np.load(os.path.join(C.WEIGHTS_DIR, "epic_prior.npz"))["p_n_given_v"]

dets = {}
for kind in ("verb", "noun"):
    m = ActionFormerEPIC(kind)
    for i, (vid, dur) in enumerate(epic):
        with np.load(os.path.join(a.feat_dir, vid + ".npz")) as z:
            feats = z["feats"].astype(np.float32)
            if kind == "verb":
                clipprob[vid] = z["n_prob"].astype(np.float32)
        dets[(vid, kind)] = m.detect(feats, dur)
        if i % 20 == 0:
            print(kind, i, vid, len(dets[(vid, kind)][0]), flush=True)
    del m
np.savez_compressed(a.save_dets, **{f"{v}|{k}|{j}": x for (v, k), d in dets.items() for j, x in enumerate(d)})

for vid, dur in epic:
    npb = clipprob[vid]
    t_center = (np.arange(len(npb)) * 16 + 16) / 30.0
    acts = pair_actions(dets[(vid, "verb")], dets[(vid, "noun")], npb, t_center, prior,
                        cls["verbs"], cls["nouns"], PairCfg())
    acts = [{"segment": [round(max(0, x["segment"][0]), 2), round(min(dur, x["segment"][1]), 2)],
             "verb": x["verb"], "noun": x["noun"], "score": x["score"]} for x in acts
            if min(dur, x["segment"][1]) > max(0, x["segment"][0])]
    hl = H.finalize(SC.select(acts, dur) or H.uniform_segments(dur), dur)
    with open(os.path.join(a.out, vid + ".json"), "w") as f:
        json.dump({"team_id": "dev", "video_id": vid, "duration_sec": dur,
                   "label": {"name": "cooking", "score": 1.0}, "actions": acts, "highlights": hl}, f)
print("wrote", a.out)
