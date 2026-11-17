"""Output validator for PRISM 2026 Track 04 (re-implementation of the rules in TEAM_README.md).

The official validate.py was not included in our student package, so this mirrors every rule the
README lists. Standard library only.

    python validate.py --file out/<video_id>.json -v
    python validate.py --dir outputs/ --manifest <manifest csv> --classes supported_classes.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

REQUIRED = ("team_id", "video_id", "duration_sec", "label", "actions", "highlights")
MIN_SEG = 4.0
BUDGET = 0.15
DUR_TOL = 0.5
MAX_ACTIONS = 2000


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _seg(seg, dur, where, errs):
    if not (isinstance(seg, list) and len(seg) == 2 and all(_num(v) for v in seg)):
        errs.append(f"{where}: segment must be [start, end] numbers, got {seg!r}")
        return None
    s, e = seg
    if not (0 <= s < e <= dur + DUR_TOL):
        errs.append(f"{where}: need 0 <= start < end <= duration+{DUR_TOL}, got [{s}, {e}]")
        return None
    return float(s), float(e)


def validate_obj(d, fname=None, manifest=None, classes=None):
    """Return (errors, warnings) for one parsed output object."""
    errs, warns = [], []
    if not isinstance(d, dict):
        return ["top level must be a JSON object"], warns
    for k in REQUIRED:
        if k not in d:
            errs.append(f"missing field '{k}'")
    if errs:
        return errs, warns

    if not (isinstance(d["team_id"], str) and d["team_id"].strip()):
        errs.append("team_id must be a non-empty string")
    vid = d["video_id"]
    if not (isinstance(vid, str) and vid.strip()):
        errs.append("video_id must be a non-empty string")
    elif fname is not None and os.path.splitext(os.path.basename(fname))[0] != vid:
        errs.append(f"video_id '{vid}' does not match filename '{os.path.basename(fname)}'")

    dur = d["duration_sec"]
    if not (_num(dur) and dur > 0):
        errs.append("duration_sec must be a positive number")
        return errs, warns
    if manifest is not None and isinstance(vid, str):
        if vid not in manifest:
            errs.append(f"video_id '{vid}' not in manifest")
        elif abs(manifest[vid] - dur) > DUR_TOL:
            errs.append(f"duration_sec {dur} differs from manifest {manifest[vid]} by > {DUR_TOL}s")

    lab = d["label"]
    if not (isinstance(lab, dict) and isinstance(lab.get("name"), str) and _num(lab.get("score"))):
        errs.append('label must be {"name": str, "score": number}')
    else:
        if not 0 <= lab["score"] <= 1:
            errs.append("label.score must be in [0, 1]")
        if classes and lab["name"] not in classes["labels"]:
            errs.append(f"label.name '{lab['name']}' not in supported labels")

    acts = d["actions"]
    if not isinstance(acts, list):
        errs.append("actions must be an array")
    else:
        if len(acts) > MAX_ACTIONS:
            errs.append(f"actions has {len(acts)} items (max {MAX_ACTIONS})")
        for i, a in enumerate(acts):
            w = f"actions[{i}]"
            if not isinstance(a, dict):
                errs.append(f"{w} must be an object")
                continue
            _seg(a.get("segment"), dur, w, errs)
            for key in ("verb", "noun"):
                if not isinstance(a.get(key), str):
                    errs.append(f"{w}.{key} must be a string")
                elif classes and a[key] not in classes[key + "s"]:
                    errs.append(f"{w}.{key} '{a[key]}' not in supported {key}s")
            if not (_num(a.get("score")) and 0 <= a["score"] <= 1):
                errs.append(f"{w}.score must be a number in [0, 1]")

    hls = d["highlights"]
    if not (isinstance(hls, list) and hls):
        errs.append("highlights must be a non-empty array")
    else:
        segs = []
        for i, h in enumerate(hls):
            if not isinstance(h, dict):
                errs.append(f"highlights[{i}] must be an object")
                continue
            s = _seg(h.get("segment"), dur, f"highlights[{i}]", errs)
            if s:
                segs.append(s)
        for i in range(1, len(segs)):
            if segs[i][0] < segs[i - 1][0]:
                errs.append(f"highlights not chronologically sorted at index {i}")
            if segs[i][0] < segs[i - 1][1]:
                errs.append(f"highlights[{i - 1}] and [{i}] overlap")
        if segs and not any(e - s >= MIN_SEG for s, e in segs):
            errs.append(f"need at least one highlight segment >= {MIN_SEG}s")
        short = sum(1 for s, e in segs if e - s < MIN_SEG)
        if short:
            warns.append(f"{short} highlight segment(s) < {MIN_SEG}s (ignored in scoring)")
        total = sum(e - s for s, e in segs)
        if total > BUDGET * dur + 1e-6:
            warns.append(f"highlights total {total:.2f}s = {100 * total / dur:.2f}% > 15% budget")
    return errs, warns


def load_manifest(path):
    with open(path, newline="") as f:
        return {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}


def load_classes(path):
    with open(path) as f:
        c = json.load(f)
    return {k: set(c[k]) for k in ("labels", "verbs", "nouns")}


def validate_file(path, manifest=None, classes=None):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as ex:  # noqa: BLE001
        return [f"cannot parse JSON: {ex}"], [], None
    e, w = validate_obj(d, path, manifest, classes)
    return e, w, d.get("team_id") if isinstance(d, dict) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file")
    g.add_argument("--dir")
    ap.add_argument("--manifest", help="official manifest CSV: checks ids, durations and completeness")
    ap.add_argument("--classes", help="supported_classes.json: checks vocabulary")
    ap.add_argument("-v", "--verbose", action="store_true", help="print warnings")
    a = ap.parse_args(argv)
    man = load_manifest(a.manifest) if a.manifest else None
    cls = load_classes(a.classes) if a.classes else None

    files = [a.file] if a.file else sorted(
        os.path.join(a.dir, f) for f in os.listdir(a.dir) if f.endswith(".json"))
    n_ok, teams = 0, set()
    for p in files:
        e, w, t = validate_file(p, man, cls)
        if t:
            teams.add(t)
        if e:
            print(f"FAIL {os.path.basename(p)}")
            for m in e:
                print(f"   - {m}")
        else:
            n_ok += 1
            if a.verbose:
                print(f"OK   {os.path.basename(p)}")
        if a.verbose:
            for m in w:
                print(f"   ! {m}")
    problems = n_ok != len(files)
    if len(teams) > 1:
        print(f"FAIL team_id differs across files: {sorted(teams)}")
        problems = True
    if a.dir and man is not None:
        have = {os.path.splitext(os.path.basename(p))[0] for p in files}
        missing = sorted(set(man) - have)
        if missing:
            print(f"MISSING {len(missing)} manifest videos, e.g. {missing[:5]}")
            problems = True
    print(f"\n{n_ok}/{len(files)} files valid")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
