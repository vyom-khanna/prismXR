"""Integrity self-check for the submission package.

1. Static scan: no manifest video id, GT file name or dev-only path appears in submission/code or run.sh.
2. Rename test (optional, --video): run run.sh on a copy of a video under a random name and compare the
   JSON with the original-name run — everything except video_id must be identical.

    py -3.13 tools/integrity_check.py [--video path/to/video.mp4]
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUB = os.path.join(ROOT, "TIET_Opus5.5high")
MANIFEST = os.path.join(ROOT, "Student package_track04", "Student package_track04", "manifest.csv")
FORBIDDEN = ["EPIC_100_validation", "EPIC_100_train", "gt_data", "raw_data.mat", "prism_ext", "prism_data",
             "manifest.csv", "FPVSum_videolist", "dev_outputs"]


def static_scan():
    with open(MANIFEST) as f:
        ids = [r["video_id"] for r in csv.DictReader(f)]
    hits = []
    for base, _, files in os.walk(SUB):
        if os.sep + "outputs" in base or "__pycache__" in base or os.sep + "weights" in base:
            continue
        for fn in files:
            if not fn.endswith((".py", ".sh", ".yaml", ".json", ".txt", "Dockerfile")):
                continue
            p = os.path.join(base, fn)
            txt = open(p, encoding="utf-8", errors="ignore").read()
            for tok in ids + FORBIDDEN:
                if tok in txt:
                    hits.append((os.path.relpath(p, ROOT), tok))
    return hits


def rename_test(video):
    py = os.environ.get("PRISM_PYTHON", sys.executable)
    env = dict(os.environ, PRISM_PYTHON=py, PRISM_CACHE="")
    tmp = tempfile.mkdtemp()
    ext = os.path.splitext(video)[1]
    a_out, b_out = os.path.join(tmp, "a.json"), os.path.join(tmp, "b.json")
    renamed = os.path.join(tmp, uuid.uuid4().hex + ext)
    shutil.copy(video, renamed)
    for src, dst in ((video, a_out), (renamed, b_out)):
        # full path: on Windows a bare "bash" resolves to WSL's System32\bash.exe before PATH
        subprocess.run([shutil.which("bash") or "bash", os.path.join(SUB, "run.sh"), src, dst], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    a, b = json.load(open(a_out)), json.load(open(b_out))
    a.pop("video_id"), b.pop("video_id")
    shutil.rmtree(tmp, ignore_errors=True)
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    a = ap.parse_args()
    hits = static_scan()
    print("static scan:", "CLEAN" if not hits else f"{len(hits)} hits")
    for h in hits:
        print("   ", h)
    if a.video:
        print("rename test:", "IDENTICAL (pass)" if rename_test(a.video) else "DIFFERENT (FAIL)")


if __name__ == "__main__":
    main()
