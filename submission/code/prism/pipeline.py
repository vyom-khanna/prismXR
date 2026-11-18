"""Single-video inference: python -m prism.pipeline <video.mp4> <out.json>

  probe -> 1 fps frames -> CLIP embeddings -> label + route (content-based, never the filename)
    cooking -> EPIC branch  : SlowFast(EPIC) features -> ActionFormer verb + noun -> pairing
                              -> coverage-optimal highlights
    sports  -> sports branch: CLIP novelty + motion importance -> peak windows
Models are loaded one at a time (6 GB GPU). The file name is only used to fill `video_id`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np

from . import config as C
from . import highlights as H
from .validate import validate_obj
from .video_io import probe, read_frames


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-6) if len(x) else x


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _cache_path(video: str, tag: str) -> str | None:
    if not C.CACHE_DIR:
        return None
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    st = os.stat(video)
    key = f"{os.path.basename(video)}_{st.st_size}_{tag}.npz"
    return os.path.join(C.CACHE_DIR, key)


def epic_features(video: str) -> dict:
    cp = _cache_path(video, "sf")
    if cp and os.path.exists(cp):
        with np.load(cp) as z:
            return {k: z[k] for k in z.files}
    from .epic_features import SlowFastEPIC
    sf = SlowFastEPIC()
    F = sf.extract(video)
    del sf
    _free()
    if cp:
        np.savez_compressed(cp, **{k: np.asarray(v) for k, v in F.items()})
    return F


def epic_actions(F: dict, dur: float, t: dict) -> tuple[list[dict], tuple]:
    """-> (actions, (density rate, t_center))."""
    from .density import Density
    from .epic_detector import ActionFormerEPIC
    from .pairing import PairCfg, pair_actions
    from .segcls import SegCls

    t0 = time.perf_counter()
    det = {}
    for kind in ("verb", "noun"):
        m = ActionFormerEPIC(kind)
        det[kind] = m.detect(F["feats"], dur)
        del m
        _free()
    t["detect"] = time.perf_counter() - t0

    with open(C.CLASSES_JSON) as f:
        cls = json.load(f)
    prior = np.load(os.path.join(C.WEIGHTS_DIR, "epic_prior.npz"))["p_n_given_v"]
    t0 = time.perf_counter()
    sc = SegCls()
    pairs = sc.pairs.numpy() if hasattr(sc.pairs, "numpy") else np.asarray(sc.pairs)

    def seg_model(segs):
        pv, pn, pa = sc.predict(F["feats"], F["verb"], F["noun"], F["t_center"], segs)
        return pv, pn, pa, pairs

    acts = pair_actions(det["verb"], det["noun"], F["noun"], F["t_center"], prior,
                        cls["verbs"], cls["nouns"], PairCfg(), seg_model=seg_model)
    del sc
    dens = Density().predict(F["feats"], F["verb"], F["noun"])
    _free()
    t["segcls"] = time.perf_counter() - t0
    out = []
    for a in acts:
        s, e = max(0.0, a["segment"][0]), min(dur, a["segment"][1])
        s, e = round(s, 2), round(e, 2)
        if e > s:
            out.append({"segment": [s, e], "verb": a["verb"], "noun": a["noun"],
                        "score": round(min(max(a["score"], 0.0), 1.0), 6)})
    return out, (dens, F["t_center"])


def run(video: str, out: str, timings: dict | None = None) -> dict:
    t = timings if timings is not None else {}
    t0 = time.perf_counter()
    info = probe(video)
    dur = info["duration"]
    from . import sports
    from .video_io import frames_at
    # label/route from 32 evenly spaced frames (fast seeks; no full decode needed for kitchen videos)
    times = (np.arange(C.LABEL_NUM_FRAMES) + 0.5) * dur / C.LABEL_NUM_FRAMES
    lab_frames = frames_at(video, times)
    t["seek32"] = time.perf_counter() - t0

    from .clip_model import Clip
    t0 = time.perf_counter()
    clip = Clip()
    name, score, _ = clip.predict_label(clip.encode_frames(lab_frames))
    frames = ts = emb = text_pos = text_neg = None
    if name != "cooking":  # sports branch needs every frame at 2 fps + prompt embeddings
        t1 = time.perf_counter()
        frames, ts = read_frames(video, fps=sports.FPS, size=224)
        t["decode2fps"] = time.perf_counter() - t1
        emb = clip.encode_frames(frames)
        text_pos, text_neg = clip.encode_text(sports.GENERIC_POS), clip.encode_text(sports.GENERIC_NEG)
    del clip
    _free()
    t["clip"] = time.perf_counter() - t0

    actions = []
    if name == "cooking":
        from . import select_coverage as SC
        t0 = time.perf_counter()
        F = epic_features(video)
        t["slowfast"] = time.perf_counter() - t0
        actions, dens = epic_actions(F, dur, t)
        t0 = time.perf_counter()
        segs = SC.select(actions, dur, SC.CovCfg(), density=dens) or H.uniform_segments(dur)
    else:
        t0 = time.perf_counter()
        imp = sports.importance(frames, emb, ts, dur, text_pos, text_neg)
        segs = H.peaks_to_segments(imp, ts, dur, seg_len=sports.SEG_LEN)
    hls = H.finalize(segs, dur)
    t["select"] = time.perf_counter() - t0

    vid = os.path.splitext(os.path.basename(video))[0]
    obj = {"team_id": C.TEAM_ID, "video_id": vid, "duration_sec": round(dur, 2),
           "label": {"name": name, "score": round(min(max(score, 0.0), 1.0), 4)},
           "actions": actions, "highlights": hls}
    errs, _ = validate_obj(obj)
    if errs:
        raise RuntimeError(f"invalid output for {vid}: {errs[:5]}")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    return obj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    timings = {}
    t0 = time.perf_counter()
    obj = run(a.video, a.out, timings)
    tot = time.perf_counter() - t0
    print(f"[prism] {obj['video_id']} dur={obj['duration_sec']}s label={obj['label']['name']} "
          f"hl={len(obj['highlights'])} actions={len(obj['actions'])} total={tot:.1f}s "
          + " ".join(f"{k}={v:.1f}s" for k, v in timings.items()), file=sys.stderr)


if __name__ == "__main__":
    main()
