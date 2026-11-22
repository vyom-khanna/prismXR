"""Run the submission pipeline over every manifest video that exists locally.

    C:/prism_env/Scripts/python.exe tools/run_all.py --out TIET_Opus5.5high/outputs [--only fpvsum|epic] [--limit N]

Writes <out>/<video_id>.json plus <out>/../timings.csv (per-stage seconds, for the latency report).
Skips videos whose output already exists unless --force.
"""
import argparse
import csv
import os
import sys
import time
import traceback

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "TIET_Opus5.5high", "code"))
os.environ.setdefault("PRISM_WEIGHTS", r"C:/prism_data/weights")
os.environ.setdefault("PRISM_CACHE", r"C:/prism_data/cache")
os.environ.setdefault("PRISM_TEAM_ID", "TIET_Opus5.5high")

from prism.pipeline import run  # noqa: E402

MANIFEST = os.path.join(ROOT, "Student package_track04", "Student package_track04", "manifest.csv")


def video_path(vid, ds):
    if ds == "fpvsum":
        return rf"C:/prism_data/fpvsum/{vid}.mp4"
    return rf"C:/prism_data/epic/EPIC-KITCHENS/{vid.split('_')[0]}/videos/{vid}.MP4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "TIET_Opus5.5high", "outputs"))
    ap.add_argument("--only", choices=["fpvsum", "epic"])
    ap.add_argument("--ids", help="comma-separated video ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))
    if a.only:
        rows = [r for r in rows if (r["dataset"] == "fpvsum") == (a.only == "fpvsum")]
    if a.ids:
        keep = set(a.ids.split(","))
        rows = [r for r in rows if r["video_id"] in keep]
    tpath = os.path.join(os.path.dirname(os.path.abspath(a.out)), "timings.csv")
    new = not os.path.exists(tpath)
    tf = open(tpath, "a", newline="")
    tw = csv.writer(tf)
    if new:
        tw.writerow(["video_id", "dataset", "duration", "label", "total", "stages"])
    done = missing = failed = 0
    for r in rows:
        vid, ds = r["video_id"], r["dataset"]
        src, dst = video_path(vid, ds), os.path.join(a.out, f"{vid}.json")
        if not os.path.exists(src):
            missing += 1
            continue
        if os.path.exists(dst) and not a.force:
            continue
        t, t0 = {}, time.perf_counter()
        try:
            obj = run(src, dst, t)
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAILED {vid}\n{traceback.format_exc()}", file=sys.stderr)
            continue
        tot = time.perf_counter() - t0
        tw.writerow([vid, ds, r["duration_sec"], obj["label"]["name"], f"{tot:.2f}",
                     " ".join(f"{k}={v:.2f}" for k, v in t.items())])
        tf.flush()
        done += 1
        print(f"[{done}] {vid} {obj['label']['name']} ({float(r['duration_sec']):.0f}s video, {tot:.1f}s) "
              + " ".join(f"{k}={v:.1f}" for k, v in t.items()), flush=True)
        if a.limit and done >= a.limit:
            break
    print(f"done={done} missing_video={missing} failed={failed}")


if __name__ == "__main__":
    main()
