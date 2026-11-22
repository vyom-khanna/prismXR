"""Video probing and frame decoding through ffmpeg pipes (no OpenCV dependency)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np


def _tool(name: str) -> str:
    env = os.environ.get(f"PRISM_{name.upper()}")
    if env:
        return env
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"{name} not found on PATH (or set PRISM_{name.upper()})")
    return p


def probe(path: str) -> dict:
    """Return {'duration', 'fps', 'width', 'height', 'rotation'} for the first video stream."""
    out = subprocess.run(
        [_tool("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames:stream_side_data=rotation"
         ":format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = (j.get("streams") or [{}])[0]

    def rate(s):
        try:
            a, b = s.split("/")
            return float(a) / float(b) if float(b) else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    fps = rate(st.get("avg_frame_rate", "0/0")) or rate(st.get("r_frame_rate", "0/0")) or 30.0
    dur = float((j.get("format") or {}).get("duration") or st.get("duration") or 0.0)
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    return {"duration": dur, "fps": fps, "width": int(st.get("width", 0)),
            "height": int(st.get("height", 0)), "rotation": rot}


def frames_at(path: str, times, size: int = 224) -> np.ndarray:
    """Grab single frames at the given times (seconds) with fast input seeking; centre-cropped square.
    Used for cheap video-level labelling (32 frames) without decoding the whole video."""
    out = []
    vf = f"scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}"
    for t in times:
        p = subprocess.run([_tool("ffmpeg"), "-v", "error", "-nostdin", "-ss", f"{max(0.0, t):.3f}", "-i", path,
                            "-frames:v", "1", "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True)
        if p.returncode == 0 and len(p.stdout) >= size * size * 3:
            out.append(np.frombuffer(p.stdout[: size * size * 3], np.uint8).reshape(size, size, 3))
    return np.stack(out) if out else np.zeros((0, size, size, 3), np.uint8)


def iter_frames(path: str, fps: float, size: tuple[int, int], chunk: int = 64, hwaccel: bool = True):
    """Stream frames (resampled to fps, resized to exact (w, h)) in uint8 chunks [k, h, w, 3]."""
    w, h = size
    cmd = [_tool("ffmpeg"), "-v", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", "auto"]
    cmd += ["-i", path, "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    fsz = w * h * 3
    # stderr goes to a temp file, not a pipe: files with many decode errors would otherwise fill the
    # undrained stderr pipe and deadlock ffmpeg (blocked writing stderr) against us (blocked reading stdout).
    errf = tempfile.TemporaryFile()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errf, bufsize=fsz * chunk)
    got = 0
    try:
        while True:
            buf = p.stdout.read(fsz * chunk)
            if not buf:
                break
            k = len(buf) // fsz
            if k:
                got += k
                yield np.frombuffer(buf[: k * fsz], np.uint8).reshape(k, h, w, 3)
    finally:
        p.stdout.close()
        p.wait()
        errf.seek(0)
        err = errf.read()
        errf.close()
    if p.returncode != 0 and got == 0:
        if hwaccel:
            yield from iter_frames(path, fps, size, chunk, hwaccel=False)
            return
        raise RuntimeError(err.decode(errors="replace")[-2000:])


def read_frames(path: str, fps: float, size: int | tuple[int, int], crop: bool = True,
                hwaccel: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Decode the whole video resampled to `fps`.

    size: int -> square output (centre crop after short-side resize if crop, else stretch);
          (w, h) -> exact.
    Returns (frames uint8 [N, H, W, 3], timestamps float [N]).
    """
    if isinstance(size, int):
        w = h = size
        vf = (f"fps={fps},scale={size}:{size}:force_original_aspect_ratio=increase,crop={size}:{size}"
              if crop else f"fps={fps},scale={size}:{size}")
    else:
        w, h = size
        vf = f"fps={fps},scale={w}:{h}"
    cmd = [_tool("ffmpeg"), "-v", "error", "-nostdin"]
    if hwaccel:
        cmd += ["-hwaccel", "auto"]
    cmd += ["-i", path, "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 and hwaccel:  # some files/drivers dislike hwaccel; retry in software
        return read_frames(path, fps, size, crop, hwaccel=False)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[-2000:])
    buf = np.frombuffer(p.stdout, np.uint8)
    n = buf.size // (w * h * 3)
    frames = buf[: n * w * h * 3].reshape(n, h, w, 3)
    ts = (np.arange(n) + 0.5) / fps  # centre time of each sampled frame
    return frames, ts
