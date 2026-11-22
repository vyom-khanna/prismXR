"""Single-video inference: python -m prism.pipeline <video.mp4> <out.json>

  probe -> 1 fps frames -> CLIP embeddings -> label + route (content-based, never the filename)
    cooking -> EPIC branch  : SlowFast(EPIC) features -> ActionFormer verb + noun -> pairing
                              -> coverage-optimal highlights
    sports  -> sports branch: CLIP novelty + motion importance -> peak windows
Models are loaded one at a time (6 GB GPU). The file name is only used to fill `video_id`.
"""
from __future__ import annotations

import argparse
import dataclasses
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


def epic_lavila(video: str, segs: np.ndarray):
    cp = _cache_path(video, "lav")
    have_v = have_n = None
    if cp and os.path.exists(cp):
        with np.load(cp) as z:
            k = min(len(z["segs"]), len(segs))
            if k and np.allclose(z["segs"][:k], segs[:k], atol=0.05):  # shared prefix (segments are in NMS order)
                have_v, have_n = z["v"][:k].astype(np.float32), z["n"][:k].astype(np.float32)
                if k == len(segs):
                    return have_v, have_n
    from .lavila_cls import classify_segments
    k = 0 if have_v is None else len(have_v)
    pv, pn = classify_segments(video, segs[k:])
    if k:
        pv, pn = np.concatenate([have_v, pv]), np.concatenate([have_n, pn])
    _free()
    if cp:
        np.savez_compressed(cp, segs=segs, v=pv.astype(np.float16), n=pn.astype(np.float16))
    return pv, pn


def epic_actions(F: dict, dur: float, t: dict, video: str | None = None, lav=None) -> tuple[list[dict], tuple, list[dict]]:
    """-> (actions, (density rate, t_center), actions for highlight selection). LaViLa runs if `video` is given (or `lav`=(segs, pv, pn))."""
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
    from .pairing import _tiou_matrix
    if lav is None and video is not None:
        from .lavila_cls import nms_segments
        t0 = time.perf_counter()
        vs_, vsc_, _ = det["verb"]
        lsegs = nms_segments(vs_, vsc_, C.LAV_NMS, C.LAV_TOPK)
        lav = (lsegs,) + epic_lavila(video, lsegs)
        t["lavila"] = time.perf_counter() - t0

    def build(lav, pmax):
        sc = SegCls()
        pairs = sc.pairs.numpy() if hasattr(sc.pairs, "numpy") else np.asarray(sc.pairs)

        def seg_model(segs):
            pv, pn, pa = sc.predict(F["feats"], F["verb"], F["noun"], F["t_center"], segs)
            if lav is not None and len(lav[0]) and C.LAV_W > 0:
                tm = _tiou_matrix(segs, lav[0])
                j = tm.argmax(1)
                ok = tm[np.arange(len(segs)), j] >= C.LAV_MATCH
                pn = np.where(ok[:, None], (1 - C.LAV_W) * pn + C.LAV_W * lav[2][j], pn)
            return pv, pn, pa, pairs

        acts = pair_actions(det["verb"], det["noun"], F["noun"], F["t_center"], prior,
                            cls["verbs"], cls["nouns"], dataclasses.replace(PairCfg(), max_actions=pmax), seg_model=seg_model)
        del sc
        if lav is not None and C.LAV_ALT > 0:
            for k, (s, e) in enumerate(lav[0]):
                pv, pn = lav[1][k], lav[2][k]
                for a_ in np.argsort(-pv)[:C.LAV_ALT]:
                    for b_ in np.argsort(-pn)[:C.LAV_ALT]:
                        acts.append({"segment": [float(s), float(e)], "verb": cls["verbs"][a_], "noun": cls["nouns"][b_],
                                     "score": C.LAV_ALT_W * float(pv[a_] * pn[b_]) ** 0.75})
        if C.MERGE_TIOU < 1:
            from .pairing import merge_duplicates
            acts = merge_duplicates(acts, C.MERGE_TIOU)
        acts = sorted(acts, key=lambda a: -a["score"])
        if C.CAP_REPEAT_BETA and pmax > 2000:  # output actions only: repeats of a pair make room for other pairs
            from collections import defaultdict
            rank, key = defaultdict(int), []
            for a in acts:
                rank[(a["verb"], a["noun"])] += 1
                key.append(a["score"] / rank[(a["verb"], a["noun"])] ** C.CAP_REPEAT_BETA)
            keep = sorted(np.argsort(-np.asarray(key), kind="stable")[:2000])
            acts = [acts[i] for i in keep]
        acts = acts[:2000]
        out = []
        for a in acts:
            s, e = max(0.0, a["segment"][0]), min(dur, a["segment"][1])
            s, e = round(s, 2), round(e, 2)
            if e > s:
                out.append({"segment": [s, e], "verb": a["verb"], "noun": a["noun"],
                            "score": round(min(max(a["score"], 0.0), 1.0), 6)})
        return out

    t0 = time.perf_counter()
    out = build(lav, C.PAIR_MAX)
    if C.NOUN_CTX:  # output actions only: coverage selection below keeps its own action set
        from .pairing import noun_context
        sig, gam, eps = C.NOUN_CTX
        M = np.load(os.path.join(C.WEIGHTS_DIR, "epic_nounctx.npz"))["M10"]
        n = len(cls["nouns"])  # pad to the full class list (incl. "other") so any noun maps safely
        M = 0.5 * np.eye(n, dtype=np.float32) + 0.5 * np.pad(M, ((0, n - len(M)), (0, n - len(M))))
        out = [dict(a, score=round(a["score"], 6)) for a in noun_context(out, cls["nouns"], M, sig, gam, eps)]
    # highlight selection: top LAV_SEL_TOPK LaViLa segments and 2000 pairing candidates
    sel = build(None if lav is None else tuple(x[:C.LAV_SEL_TOPK] for x in lav), 2000)
    dens = Density().predict(F["feats"], F["verb"], F["noun"])
    _free()
    t["segcls"] = time.perf_counter() - t0
    return out, (dens, F["t_center"]), sel


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
        actions, dens, sel = epic_actions(F, dur, t, video=video)
        t0 = time.perf_counter()
        segs = SC.select(sel, dur, SC.CovCfg(), density=dens) or H.uniform_segments(dur)
    else:
        t0 = time.perf_counter()
        imp = sports.importance(frames, emb, ts, dur, text_pos, text_neg)
        segs = H.peaks_to_segments(imp, ts, dur, seg_len=sports.SEG_LEN)
        if C.SPORTS_WIDEN and segs:
            from .select_coverage import _widen
            left = H.budget_seconds(dur) - sum(e - s for s, e in segs)
            if left > 0.05:
                segs = _widen(sorted(segs), dur, left)
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
