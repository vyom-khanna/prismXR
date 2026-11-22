"""Write schema-valid placeholder JSONs for manifest videos that could not be obtained
(e.g. YouTube video made private). Content-free on purpose: label 'other' with score 0, no actions,
evenly spaced highlights. Nothing is derived from the video id (no hardcoded labels).

    py -3.13 tools/make_placeholders.py --out TIET_Opus5.5high/outputs --ids Scuba_KD74Ywuy9ng
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "TIET_Opus5.5high", "code"))
from prism import config as C, highlights as H  # noqa: E402
from prism.validate import validate_obj  # noqa: E402

MANIFEST = os.path.join(ROOT, "Student package_track04", "Student package_track04", "manifest.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--ids", required=True)
a = ap.parse_args()
with open(MANIFEST) as f:
    dur = {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}
os.makedirs(a.out, exist_ok=True)
for vid in a.ids.split(","):
    d = dur[vid]
    obj = {"team_id": C.TEAM_ID, "video_id": vid, "duration_sec": d,
           "label": {"name": "other", "score": 0.0}, "actions": [],
           "highlights": H.finalize(H.uniform_segments(d), d)}
    errs, _ = validate_obj(obj)
    assert not errs, errs
    with open(os.path.join(a.out, vid + ".json"), "w") as f:
        json.dump(obj, f, indent=1)
    print("placeholder", vid)
